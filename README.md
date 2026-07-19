# SupportGenie — RAG Customer Support Chatbot

A white-label customer support chatbot: a business admin uploads their knowledge base
(FAQs, policies, product docs), and customers get answers grounded strictly in those
documents — with source citations, retrieval confidence, and an automatic human-handoff
when the bot isn't confident enough to answer safely.

**Live demo:** [DEMO-LINK] · Try: *"Do you deliver for free?"* then *"What's the capital of France?"* (watch the handoff trigger)

Demo knowledge base: a fictional electronics store ("Volt Electronics") — warranty, returns, shipping, payments, installments.

## Architecture

```
Customer question
      │
      ▼
FastAPI  /api/ask
      │
      ▼
MiniLM embedding ──► FAISS (inner-product, L2-normalized = cosine)
      │                       │
      │              top-k chunks + scores
      ▼                       │
Confidence gate ◄─────────────┘
      │
      ├── score < threshold ──► human-handoff reply (no LLM call)
      │
      └── score ≥ threshold ──► Llama 3.3 70B (Groq) with retrieved context
                                      │
                                      ▼
                        grounded answer + source citations
```

- **Ingestion:** PDF/TXT/MD → recursive chunking (500 chars, 80 overlap, LangChain splitter) → `all-MiniLM-L6-v2` embeddings → FAISS `IndexFlatIP`
- **Confidence gate:** if the best retrieved chunk scores below a calibrated cosine-similarity threshold, the bot escalates to a human instead of letting the LLM guess. Support bots must never invent refund policies.
- **Provider-agnostic LLM layer:** any OpenAI-compatible endpoint (Groq, DeepSeek, OpenRouter) — swap via env config, zero code changes.
- **Admin panel** (`/admin`): upload documents, paste text, view indexed sources, reset — protected by an admin key.

## Retrieval evaluation

Measured on a 20-question held-out eval set (customer-style paraphrases mapped to their answer's source document):

| Metric | Value |
|---|---|
| Hit rate@3 | [HIT-RATE] |
| MRR | [MRR] |

Reproduce with `python eval.py`.

The confidence threshold was calibrated empirically: on-topic questions clustered at 0.29–0.36 cosine similarity,
off-topic questions at −0.09–0.09, so the gate sits in the separation gap.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # add your Groq API key
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 (chat) and http://127.0.0.1:8000/admin (admin panel).
The demo knowledge base seeds automatically on first startup.

## Deploy (Hugging Face Spaces, Docker)

1. Create a new Space → SDK: **Docker**
2. Push this repository to the Space
3. In Space **Settings → Variables and secrets**, add secrets: `LLM_API_KEY`, `ADMIN_KEY`

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `LLM_API_KEY` | — | API key for the LLM provider |
| `LLM_BASE_URL` | Groq endpoint | Any OpenAI-compatible base URL |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Model name at the provider |
| `ADMIN_KEY` | `changeme` | Protects `/api/admin/*` |
| `TOP_K` | `3` | Chunks retrieved per question |
| `CONFIDENCE_THRESHOLD` | `0.35` | Handoff gate (calibrated: `0.20`) |

## Design notes / extensions

- Architecture extends naturally to per-client indexes (multi-tenant) — one FAISS index + chunk store per client key.
- The chat endpoint is a plain JSON API, so any client can consume it: an embeddable website widget, a mobile app, or a WhatsApp bot.
- Chat history is stateless by design; session memory can be added client-side or via a conversation store.

## Stack

Python · FastAPI · LangChain (text splitters) · FAISS · Sentence-Transformers · Llama 3.3 70B via Groq · Docker · Hugging Face Spaces
