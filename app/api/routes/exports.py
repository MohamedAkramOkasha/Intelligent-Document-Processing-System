from __future__ import annotations

from datetime import datetime
from enum import Enum

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.schemas.documents import DocumentType
from app.services import export as svc
from app.storage.database import get_db
from app.storage.models import APPROVED_STATUSES

router = APIRouter(prefix="/export", tags=["export"])


class ExportFormat(str, Enum):
    csv = "csv"
    xlsx = "xlsx"
    json = "json"


class Dataset(str, Enum):
    documents = "documents"
    line_items = "line_items"


MEDIA = {
    ExportFormat.csv: "text/csv",
    ExportFormat.xlsx: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ExportFormat.json: "application/json",
}


@router.get("", summary="Export extracted data to CSV, Excel or JSON")
def export(
    format: ExportFormat = ExportFormat.xlsx,
    status: list[str] | None = Query(
        None, description="Statuses to include (default: approved + auto_approved; use 'all' for everything)"),
    document_type: DocumentType | None = None,
    batch_id: str | None = None,
    ids: list[str] | None = Query(None, description="Specific document ids"),
    dataset: Dataset = Dataset.documents,
    db: Session = Depends(get_db),
) -> Response:
    if status and "all" in status:
        statuses = None
    else:
        statuses = status or [s.value for s in APPROVED_STATUSES]
    docs = svc.query_documents(db, statuses, document_type.value if document_type else None, batch_id, ids)
    if format == ExportFormat.csv:
        content = svc.to_csv(docs, dataset.value)
    elif format == ExportFormat.json:
        content = svc.to_json(docs)
    else:
        content = svc.to_excel(docs)
    filename = f"idp_export_{datetime.now():%Y%m%d_%H%M%S}.{format.value}"
    return Response(content, media_type=MEDIA[format], headers={
        "Content-Disposition": f'attachment; filename="{filename}"',
        "X-Document-Count": str(len(docs)),
    })
