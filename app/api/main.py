"""FastAPI application entry point: ``uvicorn app.api.main:app``."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from sqlalchemy import select

from app.api.deps import verify_api_key
from app.api.routes import batches, documents, exports, review, system
from app.config import get_settings
from app.services.documents import shutdown_executor, submit
from app.storage.database import init_db, session_scope
from app.storage.models import Document, DocumentStatus

settings = get_settings()
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
logger = logging.getLogger("idp")


def _resume_pending() -> None:
    """Re-queue documents left unfinished by a previous shutdown."""
    with session_scope() as db:
        pending = db.scalars(select(Document).where(
            Document.status.in_([DocumentStatus.QUEUED.value, DocumentStatus.PROCESSING.value]))).all()
        ids = [d.id for d in pending]
    for doc_id in ids:
        submit(doc_id)
    if ids:
        logger.info("Resumed %d pending document(s)", len(ids))


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    logger.info("LLM extractor: %s | data dir: %s",
                settings.gemini_model if settings.llm_available else "disabled (heuristic mode)",
                settings.data_dir.resolve())
    _resume_pending()
    yield
    shutdown_executor()


app = FastAPI(
    title="Intelligent Document Processing API",
    version=settings.app_version,
    description=(
        "Extract structured, validated JSON from invoices, receipts, contracts and other business "
        "documents using OCR + Gemini, with confidence scoring, human-in-the-loop review, batch "
        "processing and CSV/Excel export."
    ),
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])

app.include_router(system.router)
secured = [Depends(verify_api_key)]
for module in (documents, review, batches, exports):
    app.include_router(module.router, prefix="/api/v1", dependencies=secured)
app.include_router(system.router, prefix="/api/v1", include_in_schema=False, dependencies=secured)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/docs")
