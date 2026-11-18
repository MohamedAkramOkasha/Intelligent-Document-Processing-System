from __future__ import annotations

import time

import pandas as pd
import streamlit as st
from common import DOC_COLUMN_CONFIG, STATUS_LABEL, call, client, documents_frame


def render() -> None:
    st.title("📦 Batches")
    batches = call(client().batches) or []
    if not batches:
        st.info("No batches yet. Upload several files at once on **Upload & Process**.")
        return
    st.dataframe(
        [{"name": b["name"] or b["id"][:8], "created": pd.to_datetime(b["created_at"]), "status": b["status"],
          "progress": b["progress"], "documents": b["total"],
          "breakdown": ", ".join(f"{STATUS_LABEL.get(k, k)} {v}" for k, v in b["counts"].items())} for b in batches],
        hide_index=True, width="stretch",
        column_config={"progress": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
                       "created": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm")})

    options = {b["id"]: f"{b['name'] or b['id'][:8]} — {b['status']} ({b['completed']}/{b['total']})" for b in batches}
    batch_id = st.selectbox("Batch details", list(options), format_func=options.get)
    batch = call(client().batch, batch_id)
    if not batch:
        return
    st.progress(batch["progress"], text=f"{batch['completed']}/{batch['total']} processed")
    st.dataframe(documents_frame(batch["documents"]), hide_index=True, width="stretch",
                 column_config=DOC_COLUMN_CONFIG)
    if batch["status"] == "processing" and st.toggle("Auto-refresh", value=True):
        time.sleep(2)
        st.rerun()
