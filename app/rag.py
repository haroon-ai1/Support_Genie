"""RAG orchestration: retrieve -> confidence gate -> generate | abstain.

Three outcomes, in priority order:

  grounded       top-1 cosine score >= CONFIDENCE_THRESHOLD. The answer is
                 generated strictly from retrieved context.
  conversational the message is a greeting or pleasantry, so there is nothing
                 to retrieve. Handled by a no-context prompt that is forbidden
                 from stating product facts.
  handoff        a real question we cannot ground. We say so and escalate
                 rather than letting the model improvise a policy.

The third branch is the whole point of the confidence gate: an unanswerable
support question must never be answered from model priors.
"""
import re

from . import config, llm
from .ingest import KnowledgeBase

HANDOFF_MESSAGE = (
    "I don't have that in my knowledge base, so I'd rather not guess. "
    "Let me connect you with a human agent who can help."
)

# Deliberately narrow: this only runs *below* the confidence threshold, and a
# false positive here turns a real question into chit-chat. Anything not on
# this list is treated as a substantive question and escalated.
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


def _is_smalltalk(question: str) -> bool:
    return bool(_SMALLTALK.match(question.strip()))


def answer(kb: KnowledgeBase, question: str, brand_name: str = "SupportGenie") -> dict:
    """Full RAG pass. Returns answer text, sources, confidence, handoff flag, mode."""
    results = kb.search(question)
    top_score = results[0]["score"] if results else 0.0

    if results and top_score >= config.CONFIDENCE_THRESHOLD:
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

    if _is_smalltalk(question):
        return {
            "answer": llm.generate_conversational(question, brand_name=brand_name),
            "sources": [],
            "confidence": round(top_score, 3),
            "handoff": False,
            "mode": "conversational",
        }

    # A real question we can't ground. Escalate instead of improvising.
    return {
        "answer": HANDOFF_MESSAGE,
        "sources": [],
        "confidence": round(top_score, 3),
        "handoff": True,
        "mode": "handoff",
    }
