"""Human-in-the-loop review: side-by-side document preview and editable, confidence-annotated fields."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st
from common import DOC_TYPES, call, client, conf_badge

TABLE_COLUMNS = {
    "line_items": ["description", "quantity", "unit_price", "amount"],
    "parties": ["name", "role", "address", "tax_id", "email", "phone"],
    "key_values": ["key", "value"],
}
NUMERIC_COLUMNS = {"quantity", "unit_price", "amount"}


def _label(path: str) -> str:
    return path.replace("_", " ").replace(".", " › ").title()


def _clean(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str):
        return value.strip() or None
    return value


def _field_confidence(fields: dict[str, dict], prefix: str) -> float | None:
    values = [f["confidence"] for p, f in fields.items()
              if p == prefix or p.startswith(prefix + ".") or p.startswith(prefix + "[")]
    return min(values) if values else None


def _scalar_paths(data: dict, prefix: str = "") -> list[str]:
    paths = []
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            paths += _scalar_paths(value, path)
        elif not isinstance(value, list) and key not in TABLE_COLUMNS:
            paths.append(path)
    return paths


def _get(data: dict, path: str) -> Any:
    cur: Any = data
    for part in path.split("."):
        cur = (cur or {}).get(part)
    return cur


def _set(data: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = data
    for part in parts[:-1]:
        cur = cur.setdefault(part, {}) or {}
    cur[parts[-1]] = value


def _preview(doc: dict) -> None:
    n = doc.get("preview_count") or 0
    if n:
        tabs = st.tabs([f"Page {i}" for i in range(1, n + 1)]) if n > 1 else [st.container()]
        for i, tab in enumerate(tabs, start=1):
            img = client().page_image(doc["id"], i)
            if img:
                tab.image(img, width="stretch")
    full = call(client().document, doc["id"], include_text=True)
    with st.expander("Extracted text", expanded=not n):
        st.text((full or {}).get("raw_text") or "(no text)")


def _editor(doc: dict, result: dict, fields: dict[str, dict]) -> dict | None:
    """Render the correction form. Returns the submission payload when a button is pressed."""
    data = doc.get("final_data") or {}
    key = doc["id"]
    with st.form(f"review_{key}"):
        current_type = doc.get("document_type") or "generic"
        new_type = st.selectbox("Document type", DOC_TYPES, index=DOC_TYPES.index(current_type),
                                format_func=str.title, help="Changing the type re-validates against that schema")
        edited: dict = {}

        st.markdown("##### Fields")
        cols = st.columns(2)
        for i, path in enumerate(_scalar_paths(data)):
            value = _get(data, path)
            f = fields.get(path)
            badge = conf_badge(f["confidence"]) if f else "⚪ empty"
            help_text = None
            if f:
                bits = [f"LLM {f['llm_confidence']:.0%}" if f.get("llm_confidence") is not None else None,
                        f"grounding {f['evidence_score']:.0%}" if f.get("evidence_score") is not None else None,
                        f"OCR {f['ocr_confidence']:.0%}" if f.get("ocr_confidence") is not None else None]
                help_text = " · ".join(b for b in bits if b)
                if f.get("evidence"):
                    help_text += f"\n\nEvidence: “{f['evidence']}”"
                if f.get("issues"):
                    help_text += "\n\n⚠️ " + "\n\n⚠️ ".join(f["issues"])
            label = f"{_label(path)}  {badge}"
            with cols[i % 2]:
                if isinstance(value, bool) or path == "confidentiality":
                    options = [None, True, False]
                    choice = st.selectbox(label, options, index=options.index(value), help=help_text,
                                          format_func=lambda v: "—" if v is None else ("Yes" if v else "No"))
                    _set(edited, path, choice)
                else:
                    text = st.text_input(label, "" if value is None else str(value), help=help_text)
                    _set(edited, path, _clean(text))

        for list_key, value in data.items():
            if not isinstance(value, list):
                continue
            badge = conf_badge(_field_confidence(fields, list_key))
            st.markdown(f"##### {_label(list_key)}  {badge}")
            if list_key in TABLE_COLUMNS:
                columns = TABLE_COLUMNS[list_key]
                df = pd.DataFrame(value or [], columns=columns)
                config = {c: st.column_config.NumberColumn(c.replace("_", " ").title(), format="%.2f")
                          for c in columns if c in NUMERIC_COLUMNS}
                out = st.data_editor(df, num_rows="dynamic", width="stretch", key=f"{key}_{list_key}",
                                     column_config=config)
                rows = [{c: _clean(v) for c, v in row.items()} for row in out.to_dict("records")]
                edited[list_key] = [r for r in rows if any(v is not None for v in r.values())]
            else:
                text = st.text_area(f"{_label(list_key)} (one per line)", "\n".join(map(str, value)),
                                    key=f"{key}_{list_key}_ta")
                edited[list_key] = [ln.strip() for ln in text.splitlines() if ln.strip()]

        comment = st.text_input("Review comment", placeholder="What did you change and why?")
        c1, c2, c3 = st.columns(3)
        approve = c1.form_submit_button("✅ Approve", type="primary", width="stretch")
        save = c2.form_submit_button("💾 Save draft", width="stretch")
        reject = c3.form_submit_button("⛔ Reject", width="stretch")

    reviewer = st.session_state.get("reviewer") or "reviewer"
    if reject:
        return {"_reject": True, "reviewer": reviewer, "comment": comment or None}
    if approve or save:
        return {"data": edited, "document_type": new_type, "reviewer": reviewer,
                "comment": comment or None, "approve": bool(approve)}
    return None


def render() -> None:
    st.title("📝 Review Queue")
    type_filter = st.selectbox("Filter by type", ["All", *DOC_TYPES], format_func=str.title)
    queue = call(client().review_queue, None if type_filter == "All" else type_filter) or []
    if not queue:
        st.success("🎉 Nothing to review — all documents are approved or still processing.")
        return

    ids = [d["id"] for d in queue]
    labels = {d["id"]: f"{d['filename']} · {(d['document_type'] or '?').title()} · {conf_badge(d['overall_confidence'])}"
              for d in queue}
    preselect = st.session_state.get("review_doc_id")
    selected = st.selectbox(f"{len(queue)} document(s) awaiting review (least confident first)", ids,
                            index=ids.index(preselect) if preselect in ids else 0, format_func=labels.get)
    doc = call(client().document, selected)
    if not doc:
        return
    result = doc.get("result") or {}
    fields = {f["path"]: f for f in result.get("fields", [])}

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.subheader("Original document")
        _preview(doc)
    with right:
        st.subheader("Extracted data")
        m1, m2, m3 = st.columns(3)
        m1.metric("Overall confidence", conf_badge(doc.get("overall_confidence")))
        m2.metric("Classification", conf_badge(result.get("classification_confidence")))
        m3.metric("Extractor", (result.get("extractor") or "—").split(":")[-1])
        for reason in result.get("review_reasons", []):
            st.warning(reason, icon="⚠️")
        issues = result.get("validation_issues", [])
        if issues:
            with st.expander(f"Validation issues ({len(issues)})", expanded=True):
                for issue in issues:
                    icon = "🔴" if issue["severity"] == "error" else "🟡"
                    st.markdown(f"{icon} **{issue.get('field') or 'document'}** — {issue['message']}")

        payload = _editor(doc, result, fields)
        if payload:
            if payload.pop("_reject", False):
                out = call(client().reject, doc["id"], payload["reviewer"], payload["comment"])
                msg = "Document rejected."
            else:
                out = call(client().submit_review, doc["id"], payload)
                msg = "Approved ✅" if payload["approve"] else "Draft saved — document stays in the queue."
            if out:
                st.session_state.pop("review_doc_id", None)
                st.toast(msg)
                st.rerun()

        history = call(client().history, doc["id"]) or []
        if history:
            with st.expander(f"Audit trail ({len(history)})"):
                st.dataframe(pd.DataFrame(history)[["created_at", "action", "field_path", "old_value", "new_value",
                                                    "reviewer", "comment"]], hide_index=True, width="stretch")
