import io
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _upload(client, text: str, name: str = "invoice.txt", **params):
    return client.post("/api/v1/documents", files={"file": (name, text.encode(), "text/plain")}, params=params)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["llm"]["enabled"] is False


def test_upload_rejects_unsupported_type(client):
    r = client.post("/api/v1/documents", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 422


def test_upload_and_review_flow(client, invoice_text):
    r = _upload(client, invoice_text)
    assert r.status_code == 201, r.text
    doc = r.json()
    assert doc["status"] == "needs_review"
    assert doc["document_type"] == "invoice"
    assert doc["result"]["fields"]
    doc_id = doc["id"]

    queue = client.get("/api/v1/review/queue").json()
    assert doc_id in {d["id"] for d in queue}

    r = client.post(f"/api/v1/review/{doc_id}", json={
        "corrections": {"customer.name": "Lakeside Dental Clinic LLC", "payment_terms": "Net 30"},
        "reviewer": "alice", "comment": "fixed customer", "approve": True,
    })
    assert r.status_code == 200, r.text
    reviewed = r.json()
    assert reviewed["status"] == "approved"
    assert reviewed["final_data"]["customer"]["name"] == "Lakeside Dental Clinic LLC"
    assert reviewed["extracted_data"]["customer"]["name"] == "Lakeside Dental Clinic"
    assert all(f["human_verified"] for f in reviewed["result"]["fields"] if f["value"] is not None)

    history = client.get(f"/api/v1/review/{doc_id}/history").json()
    actions = {(h["action"], h["field_path"]) for h in history}
    assert ("correct", "customer.name") in actions and ("approve", None) in actions


def test_reject(client, invoice_text):
    doc_id = _upload(client, invoice_text).json()["id"]
    r = client.post(f"/api/v1/review/{doc_id}/reject", json={"reviewer": "bob", "comment": "duplicate"})
    assert r.json()["status"] == "rejected"


def test_batch_processing_and_export(client, invoice_text):
    files = [
        ("files", ("a.txt", invoice_text.encode(), "text/plain")),
        ("files", ("b.txt", b"RECEIPT\nCorner Shop\n2026-01-02\nMilk 2.50\nTotal 2.50\nCash", "text/plain")),
        ("files", ("bad.exe", b"MZ", "application/octet-stream")),
    ]
    r = client.post("/api/v1/batches", files=files, data={"name": "test batch"})
    assert r.status_code == 202, r.text
    batch = r.json()
    assert batch["total"] == 2 and len(batch["rejected"]) == 1

    for _ in range(100):
        batch = client.get(f"/api/v1/batches/{batch['id']}").json()
        if batch["status"] != "processing":
            break
        time.sleep(0.1)
    assert batch["completed"] == 2
    types = {d["filename"]: d["document_type"] for d in batch["documents"]}
    assert types == {"a.txt": "invoice", "b.txt": "receipt"}

    # export everything in the batch
    r = client.get("/api/v1/export", params={"format": "xlsx", "status": "all", "batch_id": batch["id"]})
    assert r.status_code == 200 and r.headers["x-document-count"] == "2"
    sheets = pd.read_excel(io.BytesIO(r.content), sheet_name=None)
    assert {"Summary", "Invoices", "Receipts", "Line Items", "Validation Issues"} <= set(sheets)
    assert len(sheets["Summary"]) == 2

    r = client.get("/api/v1/export", params={"format": "csv", "status": "all", "batch_id": batch["id"],
                                             "dataset": "line_items"})
    assert r.status_code == 200
    items = pd.read_csv(io.BytesIO(r.content))
    assert len(items) >= 3

    r = client.get("/api/v1/export", params={"format": "json", "status": "all", "batch_id": batch["id"]})
    assert len(r.json()) == 2


def test_default_export_only_includes_approved(client):
    r = client.get("/api/v1/export", params={"format": "json"})
    assert all(d["status"] in {"approved", "auto_approved"} for d in r.json())


def test_stats_and_404(client):
    stats = client.get("/api/v1/stats").json()
    assert stats["total_documents"] >= 1
    assert client.get("/api/v1/documents/doesnotexist").status_code == 404
    listing = client.get("/api/v1/documents", params={"document_type": "invoice"}).json()
    assert listing["total"] >= 1
