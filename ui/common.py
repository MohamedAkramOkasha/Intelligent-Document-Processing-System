"""Shared UI helpers."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from api_client import APIError, IDPClient

DOC_TYPES = ["invoice", "receipt", "contract", "generic"]
STATUSES = ["queued", "processing", "needs_review", "auto_approved", "approved", "rejected", "failed"]
STATUS_LABEL = {
    "queued": "⏳ Queued", "processing": "⚙️ Processing", "needs_review": "🟡 Needs review",
    "auto_approved": "🟢 Auto-approved", "approved": "✅ Approved", "rejected": "⛔ Rejected", "failed": "❌ Failed",
}


@st.cache_resource
def client() -> IDPClient:
    return IDPClient()


def call(fn, *args, **kwargs):
    """Call the API and show errors in the UI instead of crashing the page."""
    try:
        return fn(*args, **kwargs)
    except APIError as exc:
        st.error(f"API error: {exc.detail}" if exc.status else str(exc.detail))
        return None


def go_to(page: str) -> None:
    """Switch to another page registered in app.py (``st.session_state['pages']``)."""
    st.switch_page(st.session_state["pages"][page])


def conf_badge(value: float | None) -> str:
    if value is None:
        return "⚪ n/a"
    icon = "🟢" if value >= 0.8 else "🟡" if value >= 0.6 else "🔴"
    return f"{icon} {value:.0%}"


def documents_frame(items: list[dict]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame(items)
    df["status"] = df["status"].map(lambda s: STATUS_LABEL.get(s, s))
    if "created_at" in df:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    cols = [c for c in ["filename", "document_type", "status", "overall_confidence", "page_count", "extractor",
                        "created_at", "reviewed_by", "id"] if c in df]
    return df[cols]


DOC_COLUMN_CONFIG = {
    "filename": st.column_config.TextColumn("File"),
    "document_type": st.column_config.TextColumn("Type"),
    "status": st.column_config.TextColumn("Status"),
    "overall_confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="percent"),
    "page_count": st.column_config.NumberColumn("Pages"),
    "created_at": st.column_config.DatetimeColumn("Uploaded", format="YYYY-MM-DD HH:mm"),
    "reviewed_by": st.column_config.TextColumn("Reviewer"),
    "id": st.column_config.TextColumn("ID"),
}
