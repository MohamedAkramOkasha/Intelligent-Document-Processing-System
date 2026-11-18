from __future__ import annotations

import time

import streamlit as st
from common import DOC_TYPES, STATUS_LABEL, call, client, conf_badge, go_to

ACCEPTED = ["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp", "docx", "txt", "md"]


def _show_result(doc: dict) -> None:
    result = doc.get("result") or {}
    st.markdown(f"#### {doc['filename']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Type", (doc.get("document_type") or "—").title())
    c2.metric("Confidence", conf_badge(doc.get("overall_confidence")))
    c3.metric("Status", STATUS_LABEL.get(doc["status"], doc["status"]))
    c4.metric("Time", f"{result.get('processing_time_ms', 0) / 1000:.1f}s")
    if doc.get("error"):
        st.error(doc["error"])
    for reason in result.get("review_reasons", []):
        st.warning(reason)
    for warning in result.get("warnings", []):
        st.caption(f"⚠️ {warning}")
    tab_data, tab_fields = st.tabs(["Extracted JSON", "Field confidence"])
    tab_data.json(doc.get("final_data") or {})
    if result.get("fields"):
        tab_fields.dataframe(
            [{"field": f["path"], "value": str(f["value"]), "confidence": f["confidence"],
              "issues": "; ".join(f["issues"])} for f in result["fields"]],
            hide_index=True, width="stretch",
            column_config={"confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")})
    if doc["status"] == "needs_review" and st.button("📝 Review this document", key=f"rev_{doc['id']}"):
        st.session_state["review_doc_id"] = doc["id"]
        go_to("review")


def _poll_batch(batch_id: str) -> None:
    bar = st.progress(0.0, text="Processing batch…")
    status_box = st.empty()
    deadline = time.time() + 30 * 60
    batch = None
    while time.time() < deadline:
        batch = call(client().batch, batch_id)
        if not batch:
            return
        bar.progress(batch["progress"], text=f"Processed {batch['completed']}/{batch['total']} documents")
        status_box.caption(" · ".join(f"{STATUS_LABEL.get(k, k)}: {v}" for k, v in batch["counts"].items()))
        if batch["status"] != "processing":
            break
        time.sleep(1.5)
    if batch:
        st.success(f"Batch finished: {batch['status'].replace('_', ' ')}")
        st.dataframe(
            [{"file": d["filename"], "type": d["document_type"], "status": STATUS_LABEL.get(d["status"], d["status"]),
              "confidence": d["overall_confidence"], "error": d["error"]} for d in batch["documents"]],
            hide_index=True, width="stretch",
            column_config={"confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")})


def render() -> None:
    st.title("⬆️ Upload & Process")
    st.caption("Upload invoices, receipts, contracts or any business document. "
               "One file is processed immediately; several files become a batch processed in parallel.")
    files = st.file_uploader("Documents", type=ACCEPTED, accept_multiple_files=True)
    c1, c2 = st.columns(2)
    doc_type = c1.selectbox("Document type", ["Auto-detect", *DOC_TYPES], format_func=str.title)
    batch_name = c2.text_input("Batch name (multi-file uploads)", placeholder="e.g. October AP invoices")
    forced = None if doc_type == "Auto-detect" else doc_type

    if st.button("🚀 Process", type="primary", disabled=not files):
        if len(files) == 1:
            f = files[0]
            with st.spinner(f"Extracting data from {f.name}…"):
                doc = call(client().upload, f.name, f.getvalue(), f.type, forced)
            if doc:
                st.session_state["last_upload"] = doc
        else:
            batch = call(client().create_batch, [(f.name, f.getvalue(), f.type) for f in files],
                         batch_name or None, forced)
            if batch:
                for r in batch.get("rejected", []):
                    st.warning(f"Skipped {r['filename']}: {r['error']}")
                _poll_batch(batch["id"])
            return

    if doc := st.session_state.get("last_upload"):
        _show_result(doc)
