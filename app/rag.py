"""RAG orchestration: Decision Engine -> retrieve -> confidence gate -> generate | abstain.

Author: Muhammad Haroon (github.com/haroon-ai1)

Six outcomes, checked in this priority order (highest first):

  1. prompt_injection  the message tries to override instructions or extract
                        protected data. Blocked before any retrieval runs.
                        Fixed response. No human handoff — an attack attempt
                        isn't a support request.
  2. conversational     a greeting or pleasantry (existing behavior, untouched —
                         not part of the new Decision Engine, kept exactly where
                         it already sat in the flow).
  3. out_of_domain      a real question, but about a topic this bot was never
                        built to answer (weather, trivia, code, translation).
                        Checked before retrieval, same as prompt injection.
  4. no_context         retrieval ran and found nothing plausibly relevant
                        (top score < NO_CONTEXT_THRESHOLD).
  5. low_confidence     retrieval found something, but not confidently enough
                        (NO_CONTEXT_THRESHOLD <= top score < CONFIDENCE_THRESHOLD).
  6. grounded           top score >= CONFIDENCE_THRESHOLD. Answer is generated
                        strictly from retrieved context (existing behavior,
                        untouched — response shape is unchanged).

The point of splitting 4/5 apart, instead of one generic "can't help" bucket,
is that a customer asking about something wholly outside the knowledge base
gets a different, more honest message than one asking about a covered topic
where retrieval just came up thin.
"""
import re

from . import config, llm
from .ingest import KnowledgeBase

# ---------------------------------------------------------------------------
# Decision Engine: deterministic classification, no LLM call, no randomness.
# Each category below returns a fixed response string — never a generated one.
# ---------------------------------------------------------------------------

PROMPT_INJECTION_MESSAGE = (
    "I can't comply with requests that attempt to bypass my operating "
    "instructions or access protected information. If you need help with "
    "our products, services, or policies, I'd be happy to assist."
)
OUT_OF_DOMAIN_MESSAGE = (
    "I'm designed to answer questions about this organization's products, "
    "services, and policies. Your question appears to be outside my "
    "supported knowledge domain. A human support agent may be able to "
    "assist further."
)
LOW_CONFIDENCE_MESSAGE = (
    "I couldn't verify a reliable answer from my knowledge base with enough "
    "confidence. Rather than provide an inaccurate answer, I'll connect you "
    "with a human support agent."
)
NO_CONTEXT_MESSAGE = (
    "I couldn't find information related to your question in the current "
    "knowledge base. A human support agent can assist you further."
)

# Kept as the previous generic constant for backward compatibility — anything
# importing HANDOFF_MESSAGE directly (e.g. test_pipeline.py, older callers)
# still gets a sensible string. New code paths use the four messages above.
HANDOFF_MESSAGE = NO_CONTEXT_MESSAGE

_BADGES = {
    "prompt_injection": "\U0001F6E1 Prompt Injection Blocked",   # 🛡
    "out_of_domain": "\U0001F310 Outside Knowledge Scope",        # 🌐
    "low_confidence": "\U0001F3AF Low Confidence",                 # 🎯
    "no_context": "\U0001F4C4 Knowledge Not Found",                # 📄
}

# Matches common jailbreak / instruction-override / data-exfiltration framings.
# Deliberately searches anywhere in the string (re.search, not match) since
# these are often smuggled mid-sentence rather than leading the message.
_INJECTION = re.compile(
    r"""(
        ignore\s+(all\s+|any\s+|the\s+|your\s+|previous\s+|prior\s+)*instructions
      | disregard\s+(all\s+|any\s+|the\s+|your\s+|previous\s+|prior\s+)*instructions
      | forget\s+(all\s+|your\s+|previous\s+|prior\s+)*instructions
      | ignore\s+(the\s+|your\s+)?knowledge\s*base
      | reveal\s+(your\s+|the\s+)?(system\s+)?prompt
      | (show|print|output|repeat)\s+(me\s+)?(your\s+|the\s+)?(system\s+)?prompt
      | what\s+is\s+your\s+(system\s+)?prompt
      | reveal\s+.{0,30}(admin|root|api)\s*(password|credential|key)
      | (show|give|tell)\s+me\s+.{0,20}(admin|root)\s*password
      | developer\s*mode
      | jailbreak
      | act\s+as\s+(chatgpt|dan|an?\s+ai\s+(with\s+no|without)\s+(rules|restrictions|filters))
      | you\s+are\s+now\s+(dan|chatgpt|an?\s+unrestricted)
      | pretend\s+you\s+(have\s+no\s+rules|are\s+not\s+(bound|restricted))
      | bypass\s+(your\s+|any\s+)?(safety|content)?\s*(rules|restrictions|filters|guidelines)
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# Heuristic, rule-based topic classifier — not a full semantic classifier, so
# it will have gaps at the edges (that's true of any keyword approach). It
# targets exactly the category of question this bot was never built to
# answer: general trivia, code requests, translation, arithmetic, creative
# writing. It intentionally does not try to catch every conceivable off-topic
# phrasing; extend the pattern list here as real traffic surfaces new cases.
_OUT_OF_DOMAIN = re.compile(
    r"""(
        \bweather\b
      | \bcapital\s+of\b
      | \btell\s+me\s+a\s+joke\b | \bjoke\b
      | \bwrite\s+(me\s+)?(a\s+|some\s+)?(python|javascript|java\b|c\+\+|code)\b
      | \bwho\s+won\b.*\b(world\s+cup|match|game|election|oscar|grammy|championship)\b
      | \b(poem|haiku|lyrics|song)\b
      | \btranslate\b
      | \d+\s*[\+\-\*/]\s*\d+
      | \bmeaning\s+of\s+life\b
      | \bwho\s+is\s+the\s+president\s+of\b
      | \bwho\s+invented\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# Existing greeting/pleasantry detector — unchanged from before the Decision
# Engine was added. Kept in its original place in the flow: checked after
# prompt-injection, before the new out-of-domain / retrieval gates, exactly
# as it already behaved.
_SMALLTALK = re.compile(
    r"""^\s*(
        h(i|ey|ello|iya)(\s+there)?
      | yo
      | (a|as)salam[\s-]*[ou]?[\s-]*alaik[ou]m
      | salam
      | good\s+(morning|afternoon|evening|day)
      | how\s+(are|r)\s+(you|u)
      | how'?s\s+it\s+going
      | what'?s\s+up | sup
      | (thank\s*you|thanks|thanx|thx|shukriya)
      | (ok|okay|kk|cool|nice|great|got\s+it|understood)
      | (bye|goodbye|see\s+ya|see\s+you|later)
      | (who\s+are\s+you|what\s+are\s+you|are\s+you\s+(a\s+)?(bot|human|ai|real))
      | (what\s+can\s+you\s+do|how\s+can\s+you\s+help|help)
      | (test|testing|ping)
    )\b[\s!.?,'-]*$""",
    re.IGNORECASE | re.VERBOSE,
)


def _is_prompt_injection(question: str) -> bool:
    return bool(_INJECTION.search(question))


def _is_out_of_domain(question: str) -> bool:
    return bool(_OUT_OF_DOMAIN.search(question))


def _is_smalltalk(question: str) -> bool:
    return bool(_SMALLTALK.match(question.strip()))


def _abstain(reason: str, message: str, *, handoff: bool, confidence: float, sources=None) -> dict:
    """Build one of the four new Decision Engine responses.

    `mode` intentionally mirrors the pre-existing "handoff" value for the
    three genuinely-escalated cases (out_of_domain, no_context, low_confidence)
    so any caller already branching on mode == "handoff" keeps working. Only
    prompt_injection gets a new mode value, since handoff=False there has no
    equivalent in the old three-mode scheme.
    """
    return {
        "answer": message,
        "sources": sources or [],
        "confidence": round(confidence, 3),
        "handoff": handoff,
        "mode": "blocked" if reason == "prompt_injection" else "handoff",
        "decision": "abstain",
        "reason": reason,
        "badge": _BADGES[reason],
    }


def answer(kb: KnowledgeBase, question: str, brand_name: str = "SupportGenie") -> dict:
    """Full Decision Engine + RAG pass.

    Returns the original schema (answer, sources, confidence, handoff, mode)
    unchanged for grounded and conversational replies. The four abstention
    outcomes additionally carry `decision`, `reason`, and `badge`.
    """
    # 1. Prompt injection — highest priority, checked before anything else,
    #    including retrieval, so an attack attempt never reaches the LLM.
    if _is_prompt_injection(question):
        return _abstain("prompt_injection", PROMPT_INJECTION_MESSAGE, handoff=False, confidence=0.0)

    # 2. Greeting / pleasantry — existing behavior, untouched.
    if _is_smalltalk(question):
        return {
            "answer": llm.generate_conversational(question, brand_name=brand_name),
            "sources": [],
            "confidence": 0.0,
            "handoff": False,
            "mode": "conversational",
        }

    # 3. Out of domain — a real question, but not one this bot was built to
    #    answer. Checked before retrieval runs, per the same "don't even try"
    #    principle as prompt injection.
    if _is_out_of_domain(question):
        return _abstain("out_of_domain", OUT_OF_DOMAIN_MESSAGE, handoff=True, confidence=0.0)

    # 4/5/6. Retrieval-gated: no context, low confidence, or grounded.
    results = kb.search(question)
    top_score = results[0]["score"] if results else 0.0

    if not results or top_score < config.NO_CONTEXT_THRESHOLD:
        return _abstain("no_context", NO_CONTEXT_MESSAGE, handoff=True, confidence=top_score)

    if top_score < config.CONFIDENCE_THRESHOLD:
        sources = [{"source": r["source"], "score": round(r["score"], 3)} for r in results]
        return _abstain("low_confidence", LOW_CONFIDENCE_MESSAGE, handoff=True,
                         confidence=top_score, sources=sources)

    # Grounded — existing behavior, response shape unchanged.
    context = "\n\n".join(f"[{r['source']}] {r['text']}" for r in results)
    reply = llm.generate(question, context, brand_name=brand_name)
    return {
        "answer": reply,
        "sources": [
            {"source": r["source"], "score": round(r["score"], 3)} for r in results
        ],
        "confidence": round(top_score, 3),
        "handoff": False,
        "mode": "grounded",
    }
