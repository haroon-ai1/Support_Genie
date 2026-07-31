"""SupportGenie API.

Endpoints:
  GET  /                    -> customer chat page
  GET  /admin               -> admin panel page
  POST /api/ask             -> RAG answer {question}
  GET  /api/branding        -> current branding (public)
  POST /api/admin/branding  -> update branding             (X-Admin-Key)
  GET  /api/admin/docs      -> indexed sources             (X-Admin-Key)
  POST /api/admin/upload    -> ingest uploaded file        (X-Admin-Key)
  POST /api/admin/text      -> ingest pasted text          (X-Admin-Key)
  POST /api/admin/delete    -> drop one source from index  (X-Admin-Key)
  POST /api/admin/reset     -> clear index + re-seed       (X-Admin-Key)

On startup, if the index is empty, seeds the knowledge base from data/seed/
so the deployed demo always has content (container storage is ephemeral on
both Render's free tier and HF Spaces).

Concurrency note: the module-level `kb` is briefly None during a reset, and
FastAPI runs these sync handlers in a threadpool, so every handler that
touches the knowledge base goes through `_require_kb()` and returns 503
rather than raising AttributeError.
"""
import gc
import json
import logging
import os
import re
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import config
from .ingest import KnowledgeBase
from .rag import answer

logger = logging.getLogger(__name__)

ADMIN_KEY = os.getenv("ADMIN_KEY", "changeme")
SEED_DIR = config.ROOT_DIR / "data" / "seed"
STATIC_DIR = Path(__file__).resolve().parent / "static"
BRANDING_PATH = config.STORAGE_DIR / "branding.json"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
_ALLOWED_FILENAME = re.compile(r"^[A-Za-z0-9._-]+$")

DEFAULT_BRANDING = {
    "brand_name": "SupportGenie",
    "subtitle": "Customer Support",
    "logo_mode": "initials",
    "logo_initials": "SG",
    "logo_color": "hsl(219, 96%, 56%)",
    "logo_url": "",
}

kb: KnowledgeBase | None = None
# Serialises index-mutating operations (reset / ingest / delete). Non-blocking
# acquire: a second concurrent write returns 409 instead of queueing behind a
# rebuild that may take tens of seconds.
_kb_lock = threading.Lock()


def _require_kb() -> KnowledgeBase:
    """Return the live KB, or 503 if a reset is mid-flight."""
    current = kb
    if current is None:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is rebuilding — try again in a few seconds.",
        )
    return current


def _acquire_or_409() -> None:
    if not _kb_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Another knowledge base operation is already running.",
        )


def _load_branding() -> dict:
    if BRANDING_PATH.exists():
        try:
            saved = json.loads(BRANDING_PATH.read_text(encoding="utf-8"))
            return {**DEFAULT_BRANDING, **saved}
        except (json.JSONDecodeError, OSError):
            pass
    return dict(DEFAULT_BRANDING)


def _save_branding(data: dict) -> None:
    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    BRANDING_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _seed_if_empty(target: KnowledgeBase) -> None:
    """Seed from data/seed/. Takes the KB explicitly so it can never read a
    half-swapped global during a reset."""
    if target.index.ntotal == 0 and SEED_DIR.exists():
        for f in sorted(SEED_DIR.glob("*")):
            if f.suffix.lower() in {".txt", ".md", ".pdf"}:
                target.add_document(f)


def _sanitize_upload_name(raw: str | None) -> str:
    """Strip any directory components and reject shell/traversal characters."""
    if not raw:
        raise HTTPException(status_code=400, detail="Missing filename")
    name = Path(raw.replace("\\", "/")).name
    if not name or name in {".", ".."} or not _ALLOWED_FILENAME.match(name):
        raise HTTPException(status_code=400, detail="Invalid filename")
    return name


@asynccontextmanager
async def lifespan(app: FastAPI):
    global kb
    if not ADMIN_KEY or ADMIN_KEY == "changeme":
        logger.warning(
            "ADMIN_KEY is unset or the default 'changeme'. "
            "Set a strong ADMIN_KEY in the environment before exposing this service."
        )
    kb = KnowledgeBase()
    _seed_if_empty(kb)
    yield


# NOTE: CORS is not configured — same-origin only. If a future deployment needs
# to serve the chat widget from a different origin, add CORSMiddleware and
# restrict allow_origins to that specific origin (never use "*" in production).
app = FastAPI(title="SupportGenie", lifespan=lifespan)


def _check_admin(key: str | None):
    # compare_digest keeps the comparison constant-time so a wrong key can't be
    # recovered byte-by-byte from response timing.
    if not key or not secrets.compare_digest(key, ADMIN_KEY):
        raise HTTPException(status_code=401, detail="Invalid admin key")


class AskRequest(BaseModel):
    question: str


class BrandingRequest(BaseModel):
    brand_name: str
    subtitle: str = ""
    logo_mode: str = "initials"
    logo_initials: str = ""
    logo_color: str = ""
    logo_url: str = ""


@app.get("/")
def chat_page():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/admin")
def admin_page():
    return FileResponse(STATIC_DIR / "admin.html")


@app.get("/api/health")
def health():
    """Must never raise: a 5xx here can make the platform's health check
    restart the container mid-reset, turning a slow rebuild into a crash loop."""
    current = kb
    return {
        "status": "ok" if current is not None else "rebuilding",
        "chunks_indexed": current.index.ntotal if current is not None else 0,
    }


@app.post("/api/ask")
def ask(req: AskRequest):
    current = _require_kb()
    branding = _load_branding()
    brand_name = branding.get("brand_name") or "SupportGenie"
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty")
    if len(question) > 1000:
        raise HTTPException(status_code=400, detail="Question too long")
    try:
        return answer(current, question, brand_name=brand_name)
    except HTTPException:
        raise
    except Exception:
        logger.exception("answer() failed for question of length %d", len(question))
        raise HTTPException(
            status_code=500,
            detail="The assistant is temporarily unavailable. Please try again in a moment.",
        )


@app.get("/api/branding")
def get_branding():
    return _load_branding()


@app.post("/api/admin/branding")
def update_branding(
    req: BrandingRequest,
    x_admin_key: str | None = Header(default=None),
):
    _check_admin(x_admin_key)
    brand_name = req.brand_name.strip()
    if not brand_name:
        raise HTTPException(status_code=400, detail="Brand name is required")
    mode = req.logo_mode.strip().lower()
    if mode not in {"initials", "image"}:
        raise HTTPException(status_code=400, detail="logo_mode must be 'initials' or 'image'")
    data = {
        "brand_name": brand_name[:60],
        "subtitle": req.subtitle.strip()[:80],
        "logo_mode": mode,
        "logo_initials": (req.logo_initials.strip() or brand_name[:2]).upper()[:3],
        "logo_color": req.logo_color.strip() or DEFAULT_BRANDING["logo_color"],
        "logo_url": req.logo_url.strip(),
    }
    _save_branding(data)
    return data


@app.get("/api/admin/docs")
def list_docs(x_admin_key: str | None = Header(default=None)):
    _check_admin(x_admin_key)
    current = _require_kb()
    return {"sources": current.sources(), "total_chunks": current.index.ntotal}


@app.post("/api/admin/upload")
def upload_doc(
    file: UploadFile = File(...),
    x_admin_key: str | None = Header(default=None),
):
    _check_admin(x_admin_key)
    current = _require_kb()
    safe_name = _sanitize_upload_name(file.filename)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail="Only .pdf, .txt, .md files are supported")
    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.UPLOADS_DIR / safe_name
    size = 0
    chunk_size = 64 * 1024
    try:
        with dest.open("wb") as out:
            while True:
                chunk = file.file.read(chunk_size)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="File too large (max 10 MB)")
                out.write(chunk)
    except HTTPException:
        raise
    except OSError as exc:
        logger.warning("Upload write failed for %s: %s", safe_name, exc)
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Could not save uploaded file")

    _acquire_or_409()
    try:
        n = current.add_document(dest)
    finally:
        _kb_lock.release()

    if n == 0:
        raise HTTPException(status_code=400, detail="No text could be extracted from the file")
    return {"ingested_chunks": n, "source": dest.name, "total_chunks": current.index.ntotal}


@app.post("/api/admin/text")
def add_text(
    text: str = Form(...),
    source: str = Form("pasted_text"),
    x_admin_key: str | None = Header(default=None),
):
    _check_admin(x_admin_key)
    current = _require_kb()
    _acquire_or_409()
    try:
        n = current.add_text(text, source=source)
    finally:
        _kb_lock.release()
    if n == 0:
        raise HTTPException(status_code=400, detail="Text is empty")
    return {"ingested_chunks": n, "source": source, "total_chunks": current.index.ntotal}


@app.post("/api/admin/delete")
def delete_doc(
    source: str = Form(...),
    x_admin_key: str | None = Header(default=None),
):
    """Drop every chunk from one source without touching the rest of the index."""
    _check_admin(x_admin_key)
    current = _require_kb()
    source = source.strip()
    if not source:
        raise HTTPException(status_code=400, detail="Source is required")

    _acquire_or_409()
    try:
        removed = current.delete_source(source)
    finally:
        _kb_lock.release()

    if removed == 0:
        raise HTTPException(status_code=404, detail=f"No indexed source named '{source}'")
    return {
        "deleted_chunks": removed,
        "source": source,
        "total_chunks": current.index.ntotal,
    }


@app.post("/api/admin/reset")
def reset(x_admin_key: str | None = Header(default=None)):
    """Clear the index and re-seed from the demo knowledge base."""
    _check_admin(x_admin_key)
    global kb

    _acquire_or_409()
    try:
        # Free the old FAISS index and its Python refs BEFORE building a new one,
        # so we never briefly hold two indices in RAM on a 512 MB tier.
        if kb is not None:
            kb.index = None
            kb.chunks = []
        kb = None
        gc.collect()

        for p in (config.INDEX_PATH, config.CHUNKS_PATH):
            p.unlink(missing_ok=True)

        rebuilt = KnowledgeBase()
        _seed_if_empty(rebuilt)
        kb = rebuilt
        return {"status": "reset", "total_chunks": kb.index.ntotal}
    except Exception:
        # Without this, one failed reset leaves kb=None forever and only a
        # process restart recovers. Fall back to an empty-but-usable KB.
        logger.exception("Reset failed; attempting to recover an empty knowledge base")
        try:
            kb = KnowledgeBase()
        except Exception:
            logger.exception("Could not recover a knowledge base; service needs a restart")
        raise HTTPException(status_code=500, detail="Reset failed — check server logs")
    finally:
        _kb_lock.release()
