from __future__ import annotations

import streamlit as st
from common import (
    DOC_COLUMN_CONFIG,
    DOC_TYPES,
    STATUS_LABEL,
    STATUSES,
    call,
    client,
    conf_badge,
    documents_frame,
    go_to,
)


def render() -> None:
    st.title("🗂️ Documents")
    c1, c2, c3 = st.columns([1, 1, 2])
    statuses = c1.multiselect("Status", STATUSES, format_func=lambda s: STATUS_LABEL[s])
    doc_type = c2.selectbox("Type", ["All", *DOC_TYPES], format_func=str.title)
    query = c3.text_input("Filename contains")
    listing = call(client().list_documents, status=statuses or None,
                   document_type=None if doc_type == "All" else doc_type, q=query or None, limit=500)
    if not listing or not listing["items"]:
        st.info("No documents match the filters.")
        return
    st.caption(f"{listing['total']} document(s)")
    event = st.dataframe(documents_frame(listing["items"]), hide_index=True, width="stretch",
                         column_config=DOC_COLUMN_CONFIG, on_select="rerun", selection_mode="single-row")
    rows = event.selection.rows if event else []
    if not rows:
        st.caption("Select a row to see details.")
        return
    doc_id = listing["items"][rows[0]]["id"]
    doc = call(client().document, doc_id)
    if not doc:
        return

    st.divider()
    st.subheader(doc["filename"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Status", STATUS_LABEL.get(doc["status"], doc["status"]))
    m2.metric("Type", (doc.get("document_type") or "—").title())
    m3.metric("Confidence", conf_badge(doc.get("overall_confidence")))
    m4.metric("Pages", doc.get("page_count") or 0)
    if doc.get("error"):
        st.error(doc["error"])

    a1, a2, a3, a4 = st.columns(4)
    a1.download_button("⬇️ Original file", data=call(client().original_file, doc_id) or b"", file_name=doc["filename"],
                       width="stretch")
    if a2.button("🔁 Reprocess", width="stretch"):
        with st.spinner("Reprocessing…"):
            if call(client().reprocess, doc_id):
                st.rerun()
    if doc["status"] in {"needs_review", "auto_approved", "approved"} and a3.button("📝 Review / edit",
                                                                                   width="stretch"):
        if doc["status"] != "needs_review":
            call(client().submit_review, doc_id, {"approve": False, "reviewer": st.session_state.get("reviewer"),
                                                  "comment": "Re-opened for review"})
        st.session_state["review_doc_id"] = doc_id
        go_to("review")
    if a4.button("🗑️ Delete", width="stretch"):
        call(client().delete, doc_id)
        st.rerun()

    t1, t2, t3 = st.tabs(["Final data", "Original extraction", "Field confidence"])
    t1.json(doc.get("final_data") or {})
    t2.json(doc.get("extracted_data") or {})
    fields = (doc.get("result") or {}).get("fields", [])
    if fields:
        t3.dataframe([{"field": f["path"], "value": str(f["value"]), "confidence": f["confidence"],
                       "verified": f["human_verified"], "issues": "; ".join(f["issues"])} for f in fields],
                     hide_index=True, width="stretch",
                     column_config={"confidence": st.column_config.ProgressColumn(min_value=0, max_value=1,
                                                                                  format="percent")})
