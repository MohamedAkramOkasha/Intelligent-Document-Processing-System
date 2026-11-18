from __future__ import annotations

from datetime import datetime

import streamlit as st
from common import DOC_TYPES, STATUS_LABEL, STATUSES, call, client

FORMATS = {"Excel (.xlsx)": "xlsx", "CSV (.csv)": "csv", "JSON (.json)": "json"}
MIME = {"xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "csv": "text/csv",
        "json": "application/json"}


def render() -> None:
    st.title("💾 Export")
    st.caption("Excel exports contain a summary sheet, one sheet per document type, line items, "
               "validation issues and low-confidence fields.")
    c1, c2 = st.columns(2)
    fmt = FORMATS[c1.radio("Format", list(FORMATS), horizontal=True)]
    dataset = "documents"
    if fmt == "csv":
        dataset = c2.radio("CSV contents", ["documents", "line_items"], horizontal=True,
                           format_func=lambda d: "One row per document" if d == "documents" else "One row per line item")
    statuses = st.multiselect("Statuses", STATUSES, default=["approved", "auto_approved"],
                              format_func=lambda s: STATUS_LABEL[s],
                              help="By default only approved data is exported")
    c3, c4 = st.columns(2)
    doc_type = c3.selectbox("Document type", ["All", *DOC_TYPES], format_func=str.title)
    batches = call(client().batches) or []
    batch_opts = {"": "All batches", **{b["id"]: b["name"] or b["id"][:8] for b in batches}}
    batch_id = c4.selectbox("Batch", list(batch_opts), format_func=batch_opts.get)

    if st.button("Prepare export", type="primary"):
        result = call(client().export, fmt, statuses or ["all"], None if doc_type == "All" else doc_type,
                      batch_id or None, dataset)
        if result:
            content, count = result
            if not count:
                st.warning("No documents match these filters.")
            st.download_button(f"⬇️ Download {count} document(s)", data=content, mime=MIME[fmt], type="primary",
                               file_name=f"idp_export_{datetime.now():%Y%m%d_%H%M%S}.{fmt}")
