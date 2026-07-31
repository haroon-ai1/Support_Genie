# SupportGenie — container image (Render / Hugging Face Spaces / any Docker host)
FROM python:3.11-slim

# Run as a non-root user (required by HF Spaces, good practice everywhere).
RUN useradd -m -u 1000 user
USER user

# Fastembed caches ONNX weights here. Set it explicitly so the model baked in
# at build time is the one found at runtime — the default is a temp dir.
ENV PATH="/home/user/.local/bin:$PATH" \
    FASTEMBED_CACHE_PATH="/home/user/.cache/fastembed" \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Pre-download the embedding model at build time so the first request after a
# deploy doesn't pay a ~90 MB download.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2', cache_dir='/home/user/.cache/fastembed')"

EXPOSE 7860

# sh -c so ${PORT} expands — Render injects it, HF Spaces expects 7860.
# --workers 1 is deliberate: each worker loads its own ONNX session, and the
# 512 MB tier has room for exactly one.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
