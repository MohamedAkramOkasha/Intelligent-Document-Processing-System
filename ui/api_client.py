"""Small HTTP client used by the Streamlit UI to talk to the FastAPI backend."""

from __future__ import annotations

import os
from typing import Any

import requests


class APIError(RuntimeError):
    def __init__(self, status: int, detail: Any):
        super().__init__(f"{status}: {detail}")
        self.status, self.detail = status, detail


class IDPClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout: int = 300):
        self.base_url = (base_url or os.getenv("API_URL", "http://localhost:8000")).rstrip("/")
        self.session = requests.Session()
        key = api_key if api_key is not None else os.getenv("API_KEY")
        if key:
            self.session.headers["X-API-Key"] = key
        self.timeout = timeout

    # ------------------------------------------------------------------ plumbing
    def _request(self, method: str, path: str, raw: bool = False, **kwargs) -> Any:
        url = f"{self.base_url}{path}"
        try:
            resp = self.session.request(method, url, timeout=kwargs.pop("timeout", self.timeout), **kwargs)
        except requests.RequestException as exc:
            raise APIError(0, f"Cannot reach API at {self.base_url}: {exc}") from exc
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            raise APIError(resp.status_code, detail)
        if raw:
            return resp
        return resp.json() if resp.content else None

    # ------------------------------------------------------------------ system
    def health(self) -> dict:
        return self._request("GET", "/health", timeout=5)

    def stats(self) -> dict:
        return self._request("GET", "/api/v1/stats")

    # ------------------------------------------------------------------ documents
    def upload(self, name: str, data: bytes, mime: str | None, document_type: str | None, wait: bool = True) -> dict:
        form = {"document_type": document_type} if document_type else {}
        return self._request("POST", "/api/v1/documents", params={"wait": wait},
                             files={"file": (name, data, mime or "application/octet-stream")}, data=form)

    def list_documents(self, **params) -> dict:
        return self._request("GET", "/api/v1/documents", params={k: v for k, v in params.items() if v})

    def document(self, doc_id: str, include_text: bool = False) -> dict:
        return self._request("GET", f"/api/v1/documents/{doc_id}", params={"include_text": include_text})

    def page_image(self, doc_id: str, page: int) -> bytes | None:
        try:
            return self._request("GET", f"/api/v1/documents/{doc_id}/pages/{page}", raw=True).content
        except APIError:
            return None

    def original_file(self, doc_id: str) -> bytes:
        return self._request("GET", f"/api/v1/documents/{doc_id}/file", raw=True).content

    def reprocess(self, doc_id: str, document_type: str | None = None) -> dict:
        params = {"document_type": document_type} if document_type else {}
        return self._request("POST", f"/api/v1/documents/{doc_id}/reprocess", params=params)

    def delete(self, doc_id: str) -> None:
        self._request("DELETE", f"/api/v1/documents/{doc_id}")

    # ------------------------------------------------------------------ review
    def review_queue(self, document_type: str | None = None) -> list[dict]:
        return self._request("GET", "/api/v1/review/queue",
                             params={"document_type": document_type} if document_type else {})

    def submit_review(self, doc_id: str, payload: dict) -> dict:
        return self._request("POST", f"/api/v1/review/{doc_id}", json=payload)

    def reject(self, doc_id: str, reviewer: str, comment: str | None) -> dict:
        return self._request("POST", f"/api/v1/review/{doc_id}/reject", json={"reviewer": reviewer, "comment": comment})

    def history(self, doc_id: str) -> list[dict]:
        return self._request("GET", f"/api/v1/review/{doc_id}/history")

    # ------------------------------------------------------------------ batches
    def create_batch(self, files: list[tuple[str, bytes, str | None]], name: str | None,
                     document_type: str | None) -> dict:
        multipart = [("files", (n, d, m or "application/octet-stream")) for n, d, m in files]
        form = {k: v for k, v in {"name": name, "document_type": document_type}.items() if v}
        return self._request("POST", "/api/v1/batches", files=multipart, data=form)

    def batches(self) -> list[dict]:
        return self._request("GET", "/api/v1/batches")

    def batch(self, batch_id: str) -> dict:
        return self._request("GET", f"/api/v1/batches/{batch_id}")

    # ------------------------------------------------------------------ export
    def export(self, fmt: str, statuses: list[str] | None, document_type: str | None,
               batch_id: str | None, dataset: str = "documents") -> tuple[bytes, int]:
        params: dict[str, Any] = {"format": fmt, "dataset": dataset}
        if statuses:
            params["status"] = statuses
        if document_type:
            params["document_type"] = document_type
        if batch_id:
            params["batch_id"] = batch_id
        resp = self._request("GET", "/api/v1/export", params=params, raw=True)
        return resp.content, int(resp.headers.get("X-Document-Count", 0))
