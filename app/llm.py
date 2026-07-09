"""Provider-agnostic LLM layer.

Any OpenAI-compatible endpoint works (Groq, DeepSeek, OpenRouter, ...).
Swap providers by changing LLM_BASE_URL / LLM_MODEL in .env — zero code changes.
"""
from openai import OpenAI

from . import config

_client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)

SYSTEM_PROMPT = (
    "You are a helpful customer support assistant for a company. "
    "Answer the customer's question using ONLY the provided context from the "
    "company's knowledge base. Be concise and friendly. "
    "If the context does not contain the answer, say you don't have that "
    "information and offer to connect them with a human agent. "
    "Never invent policies, prices, or facts."
)


def generate(question: str, context: str) -> str:
    """Answer a question grounded in retrieved context."""
    user_prompt = (
        f"Context from the knowledge base:\n---\n{context}\n---\n\n"
        f"Customer question: {question}"
    )
    response = _client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()
