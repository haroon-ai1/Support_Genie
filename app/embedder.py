"""Fastembed-backed drop-in replacement for SentenceTransformer.

Fastembed ships ONNX-quantized models that fit within Render's 512 MB tier,
where torch + sentence-transformers OOM'd on import. The public shape mirrors
what ingest.py uses from SentenceTransformer: `encode(...)` returning a numpy
array, plus a dimension accessor.
"""
from __future__ import annotations

import logging
import os
from typing import Iterable

import numpy as np
from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

# sentence-transformers name -> fastembed registry name.
# Fastembed exposes some ST models under their original HF path; a few need
# an explicit alias (e.g. bare "all-MiniLM-L6-v2" without the org prefix).
_MODEL_ALIASES: dict[str, str] = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "paraphrase-multilingual-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "paraphrase-multilingual-mpnet-base-v2": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
    "intfloat/multilingual-e5-large": "intfloat/multilingual-e5-large",
}


class Embedder:
    """Minimal SentenceTransformer-compatible wrapper around fastembed.

    Fastembed L2-normalizes outputs by default, so vectors are unit-length.
    """

    def __init__(self, model_name: str, batch_size: int = 32, threads: int | None = None):
        supported = TextEmbedding.list_supported_models()
        by_name = {m["model"]: m for m in supported}

        resolved = _MODEL_ALIASES.get(model_name, model_name)
        if resolved not in by_name:
            raise ValueError(
                f"Embedding model {model_name!r} (resolved to {resolved!r}) is not "
                f"supported by fastembed. Known models: {sorted(by_name)}"
            )

        self.model_name = resolved
        self.batch_size = batch_size
        self._dim = int(by_name[resolved]["dim"])
        self._model: TextEmbedding | None = None

        # onnxruntime otherwise spawns one thread per core. On a shared/fractional
        # CPU (Render free tier) that costs memory and causes scheduler thrash for
        # no throughput gain, so default to a single thread.
        self._threads = threads if threads is not None else int(os.getenv("EMBED_THREADS", "1"))
        # Must match the cache_dir baked into the image at build time.
        self._cache_dir = os.getenv("FASTEMBED_CACHE_PATH") or None

    def _model_ready(self) -> TextEmbedding:
        if self._model is None:
            logger.info(
                "Loading fastembed model %s (threads=%s, cache_dir=%s)",
                self.model_name, self._threads, self._cache_dir,
            )
            self._model = TextEmbedding(
                model_name=self.model_name,
                cache_dir=self._cache_dir,
                threads=self._threads,
            )
        return self._model

    def encode(self, texts, **kwargs) -> np.ndarray:
        # Accept and ignore ST-specific kwargs that don't apply here.
        kwargs.pop("convert_to_numpy", None)
        kwargs.pop("show_progress_bar", None)

        if isinstance(texts, str):
            items: list[str] = [texts]
        else:
            items = list(texts) if isinstance(texts, Iterable) else [texts]

        if not items:
            return np.zeros((0, self._dim), dtype=np.float32)

        model = self._model_ready()
        vecs = list(model.embed(items, batch_size=self.batch_size))
        return np.asarray(vecs, dtype=np.float32)

    def get_embedding_dimension(self) -> int:
        return self._dim

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim
