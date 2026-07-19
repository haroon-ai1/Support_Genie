"""Knowledge base: document ingestion -> chunking -> embeddings -> FAISS index.

Same pattern as PixSearch: L2-normalized embeddings in an inner-product FAISS
index, so inner product == cosine similarity. Text chunks instead of images.
"""
import json
import logging
import threading
from pathlib import Path

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from . import config

logger = logging.getLogger(__name__)

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def _load_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        logger.warning("Failed to read PDF %s: %s", path, exc)
        return ""


def _load_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _load_pdf(path)
    return path.read_text(encoding="utf-8", errors="ignore")


class KnowledgeBase:
    """FAISS-backed knowledge base with JSON chunk metadata."""

    def __init__(self):
        self.embedder = SentenceTransformer(config.EMBED_MODEL)
        getdim = getattr(self.embedder, "get_embedding_dimension", None) or self.embedder.get_sentence_embedding_dimension
        dim = getdim()
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[dict] = []  # [{"text": ..., "source": ...}]
        # FAISS is not thread-safe; guard read+write access from concurrent workers.
        self._lock = threading.RLock()
        self._load_if_exists()

    # ---------- persistence ----------

    def _load_if_exists(self):
        if not (config.INDEX_PATH.exists() and config.CHUNKS_PATH.exists()):
            return
        try:
            index = faiss.read_index(str(config.INDEX_PATH))
            chunks = json.loads(config.CHUNKS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, RuntimeError) as exc:
            logger.warning("Failed to load persisted index/chunks (%s); starting empty.", exc)
            return
        if not isinstance(chunks, list):
            logger.warning("chunks.json is not a list; starting empty.")
            return
        self.index = index
        self.chunks = chunks

    def save(self):
        config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(config.INDEX_PATH))
        config.CHUNKS_PATH.write_text(
            json.dumps(self.chunks, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ---------- ingestion ----------

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        vecs = vecs.astype("float32")
        faiss.normalize_L2(vecs)  # normalized -> inner product == cosine similarity
        return vecs

    def add_document(self, path: str | Path) -> int:
        """Ingest one file (pdf/txt/md). Returns number of chunks added."""
        path = Path(path)
        text = _load_file(path).strip()
        if not text:
            return 0
        pieces = _splitter.split_text(text)
        if not pieces:
            return 0
        vecs = self._embed(pieces)
        with self._lock:
            self.index.add(vecs)
            self.chunks.extend({"text": p, "source": path.name} for p in pieces)
            self.save()
        return len(pieces)

    def add_text(self, text: str, source: str = "pasted_text") -> int:
        """Ingest raw pasted text (for the future admin panel)."""
        pieces = _splitter.split_text(text.strip())
        if not pieces:
            return 0
        vecs = self._embed(pieces)
        with self._lock:
            self.index.add(vecs)
            self.chunks.extend({"text": p, "source": source} for p in pieces)
            self.save()
        return len(pieces)

    # ---------- retrieval ----------

    def search(self, query: str, k: int | None = None) -> list[dict]:
        """Return top-k chunks: [{"text", "source", "score"}], best first."""
        k = k or config.TOP_K
        with self._lock:
            if self.index.ntotal == 0:
                return []
            qvec = self._embed([query])
            scores, ids = self.index.search(qvec, min(k, self.index.ntotal))
            results = []
            for score, idx in zip(scores[0], ids[0]):
                if idx == -1 or idx >= len(self.chunks):
                    continue
                chunk = self.chunks[idx]
                results.append({**chunk, "score": float(score)})
            return results

    def sources(self) -> dict[str, int]:
        """Chunk count per source document (for the future admin panel)."""
        counts: dict[str, int] = {}
        for c in self.chunks:
            counts[c["source"]] = counts.get(c["source"], 0) + 1
        return counts
