from __future__ import annotations

import pandas as pd
import streamlit as st
from common import DOC_COLUMN_CONFIG, STATUS_LABEL, call, client, documents_frame


def render() -> None:
    st.title("📊 Dashboard")
    stats = call(client().stats)
    if not stats:
        return
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Documents", stats["total_documents"])
    c2.metric("Awaiting review", stats["review_queue"])
    c3.metric("Auto-approval rate",
              f"{stats['auto_approval_rate']:.0%}" if stats["auto_approval_rate"] is not None else "—")
    c4.metric("Avg. confidence",
              f"{stats['average_confidence']:.0%}" if stats["average_confidence"] is not None else "—")
    c5.metric("Batches", stats["total_batches"])

    if not stats["total_documents"]:
        st.info("No documents yet. Head to **Upload & Process** to get started.")
        return

    left, right = st.columns(2)
    total = stats["total_documents"] or 1
    with left:
        st.subheader("By status")
        df = pd.DataFrame([{"Status": STATUS_LABEL.get(k, k), "Documents": v, "Share": v / total}
                           for k, v in sorted(stats["by_status"].items(), key=lambda kv: -kv[1])])
        st.dataframe(df, hide_index=True, width="stretch", column_config={
            "Share": st.column_config.ProgressColumn("Share", min_value=0, max_value=1, format="percent")})
    with right:
        st.subheader("By document type")
        df = pd.DataFrame([{"Type": k.title(), "Documents": v, "Share": v / total}
                           for k, v in sorted(stats["by_type"].items(), key=lambda kv: -kv[1])])
        st.dataframe(df, hide_index=True, width="stretch", column_config={
            "Share": st.column_config.ProgressColumn("Share", min_value=0, max_value=1, format="percent")})

    st.subheader("Recent documents")
    recent = call(client().list_documents, limit=10)
    if recent:
        st.dataframe(documents_frame(recent["items"]), hide_index=True, width="stretch",
                     column_config=DOC_COLUMN_CONFIG)
