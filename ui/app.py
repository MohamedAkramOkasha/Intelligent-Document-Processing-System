"""Streamlit front-end: ``streamlit run ui/app.py``."""

from __future__ import annotations

import streamlit as st
from api_client import APIError
from common import client
from views import batches, dashboard, documents, export, review, upload

st.set_page_config(page_title="Intelligent Document Processing", page_icon="📄", layout="wide")

with st.sidebar:
    st.markdown("### 📄 Intelligent Document Processing")
    try:
        health = client().health()
        llm = health["llm"]
        st.success(f"API online · v{health['version']}")
        st.caption(f"LLM: {'**' + (llm.get('chain') or llm['model']) + '**' if llm['enabled'] else 'disabled (rule-based mode)'}")
        st.caption(f"OCR: {'Tesseract ✓' if health['ocr']['enabled'] else 'not available'}")
    except APIError as exc:
        st.error(f"API offline\n\n{exc.detail}")
    st.session_state.setdefault("reviewer", "reviewer")
    st.text_input("Reviewer name", key="reviewer")

pages = [
    st.Page(dashboard.render, title="Dashboard", icon="📊", url_path="dashboard", default=True),
    st.Page(upload.render, title="Upload & Process", icon="⬆️", url_path="upload"),
    st.Page(review.render, title="Review Queue", icon="📝", url_path="review"),
    st.Page(documents.render, title="Documents", icon="🗂️", url_path="documents"),
    st.Page(batches.render, title="Batches", icon="📦", url_path="batches"),
    st.Page(export.render, title="Export", icon="💾", url_path="export"),
]
st.session_state["pages"] = {p.url_path: p for p in pages}
st.navigation(pages).run()
