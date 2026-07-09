"""RAG orchestration: retrieve -> confidence gate -> generate.

The confidence gate is the production-critical piece: if the best retrieved
chunk scores below CONFIDENCE_THRESHOLD, we do NOT let the LLM guess — we
hand off to a human. Support bots must never invent refund policies.
"""
from . import config, llm
from .ingest import KnowledgeBase

HANDOFF_MESSAGE = (
    "I'm not confident I have the right information to answer that. "
    "Let me connect you with a human agent who can help."
)


def answer(kb: KnowledgeBase, question: str) -> dict:
    """Full RAG pass. Returns answer text, sources, confidence, handoff flag."""
    results = kb.search(question)

    top_score = results[0]["score"] if results else 0.0
    if not results or top_score < config.CONFIDENCE_THRESHOLD:
        return {
            "answer": HANDOFF_MESSAGE,
            "sources": [],
            "confidence": round(top_score, 3),
            "handoff": True,
        }

    context = "\n\n".join(
        f"[{r['source']}] {r['text']}" for r in results
    )
    reply = llm.generate(question, context)

    return {
        "answer": reply,
        "sources": [
            {"source": r["source"], "score": round(r["score"], 3)} for r in results
        ],
        "confidence": round(top_score, 3),
        "handoff": False,
    }
