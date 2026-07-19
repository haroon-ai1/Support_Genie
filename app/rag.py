"""RAG orchestration: retrieve -> confidence gate -> generate.

Above the confidence threshold we ground the answer in retrieved context.
Below it, we hand the message to a conversational LLM call that handles
greetings and off-topic requests without inventing product facts.
"""
from . import config, llm
from .ingest import KnowledgeBase

# TODO: reintroduce for explicit escalation
HANDOFF_MESSAGE = (
    "I'm not confident I have the right information to answer that. "
    "Let me connect you with a human agent who can help."
)


def answer(kb: KnowledgeBase, question: str, brand_name: str = "SupportGenie") -> dict:
    """Full RAG pass. Returns answer text, sources, confidence, handoff flag, mode."""
    results = kb.search(question)

    top_score = results[0]["score"] if results else 0.0

    if results and top_score >= config.CONFIDENCE_THRESHOLD:
        context = "\n\n".join(
            f"[{r['source']}] {r['text']}" for r in results
        )
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

    conversational_reply = llm.generate_conversational(question, brand_name=brand_name)
    return {
        "answer": conversational_reply,
        "sources": [],
        "confidence": 0.0,
        "handoff": False,
        "mode": "conversational",
    }
