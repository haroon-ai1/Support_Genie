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
  POST /api/admin/reset     -> clear index + re-seed       (X-Admin-Key)

On startup, if the index is empty, seeds the knowledge base from data/seed/
so the deployed demo always has content (HF Spaces storage is ephemeral).
"""
import json
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import config
from .ingest import KnowledgeBase
from .rag import answer

ADMIN_KEY = os.getenv("ADMIN_KEY", "changeme")
SEED_DIR = config.ROOT_DIR / "data" / "seed"
STATIC_DIR = Path(__file__).resolve().parent / "static"
BRANDING_PATH = config.STORAGE_DIR / "branding.json"

DEFAULT_BRANDING = {
    "brand_name": "SupportGenie",
    "subtitle": "Customer Support",
    "logo_mode": "initials",
    "logo_initials": "SG",
    "logo_color": "hsl(219, 96%, 56%)",
    "logo_url": "",
}


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

kb: KnowledgeBase | None = None


def _seed_if_empty():
    if kb.index.ntotal == 0 and SEED_DIR.exists():
        for f in sorted(SEED_DIR.glob("*")):
            if f.suffix.lower() in {".txt", ".md", ".pdf"}:
                kb.add_document(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global kb
    kb = KnowledgeBase()
    _seed_if_empty()
    yield


app = FastAPI(title="SupportGenie", lifespan=lifespan)


def _check_admin(key: str | None):
    if key != ADMIN_KEY:
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
    return {"status": "ok", "chunks_indexed": kb.index.ntotal}


@app.post("/api/ask")
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty")
    if len(question) > 1000:
        raise HTTPException(status_code=400, detail="Question too long")
    return answer(kb, question)


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
    return {"sources": kb.sources(), "total_chunks": kb.index.ntotal}


@app.post("/api/admin/upload")
def upload_doc(
    file: UploadFile = File(...),
    x_admin_key: str | None = Header(default=None),
):
    _check_admin(x_admin_key)
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail="Only .pdf, .txt, .md files are supported")
    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.UPLOADS_DIR / Path(file.filename).name
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    n = kb.add_document(dest)
    if n == 0:
        raise HTTPException(status_code=400, detail="No text could be extracted from the file")
    return {"ingested_chunks": n, "source": dest.name, "total_chunks": kb.index.ntotal}


@app.post("/api/admin/text")
def add_text(
    text: str = Form(...),
    source: str = Form("pasted_text"),
    x_admin_key: str | None = Header(default=None),
):
    _check_admin(x_admin_key)
    n = kb.add_text(text, source=source)
    if n == 0:
        raise HTTPException(status_code=400, detail="Text is empty")
    return {"ingested_chunks": n, "source": source, "total_chunks": kb.index.ntotal}


@app.post("/api/admin/reset")
def reset(x_admin_key: str | None = Header(default=None)):
    """Clear the index and re-seed from the demo knowledge base."""
    _check_admin(x_admin_key)
    global kb
    for p in (config.INDEX_PATH, config.CHUNKS_PATH):
        p.unlink(missing_ok=True)
    kb = KnowledgeBase()
    _seed_if_empty()
    return {"status": "reset", "total_chunks": kb.index.ntotal}
