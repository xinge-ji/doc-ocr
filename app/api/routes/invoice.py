import logging
from typing import Final

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from app.api.deps import LLMClientDep, OCRClientDep
from app.schemas.invoice import ExtractionResponse
from app.services.pipelines.invoice import InvoiceExtractionPipeline

router = APIRouter(prefix="/invoice", tags=["invoice"])

logger = logging.getLogger(__name__)
ALLOWED_CONTENT_TYPES: Final[set[str]] = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "application/pdf",
}


@router.post(
    "/extract",
    summary="Extract invoice fields",
    response_model=ExtractionResponse,
)
async def extract_invoice(
    ocr_client: OCRClientDep,
    llm_client: LLMClientDep,
    llm_node: str | None = Query(default=None, description="Target LLM node name; leave empty for random"),
    file: UploadFile = File(..., description="Invoice image or PDF"),
) -> ExtractionResponse:
    """Run the hybrid invoice extraction pipeline on an uploaded file."""

    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}",
        )

    try:
        payload = await file.read()
    except Exception as exc:  # pragma: no cover - upload IO errors are rare
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file.",
        ) from exc

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    pipeline = InvoiceExtractionPipeline(ocr_client=ocr_client, llm_client=llm_client)
    try:
        invoice_data = await pipeline.run(
            payload,
            filename=file.filename,
            content_type=file.content_type,
            llm_node=llm_node,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - passthrough for service failures
        logger.exception("Invoice extraction failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Invoice extraction failed.",
        ) from exc

    return ExtractionResponse(success=True, data=invoice_data, message="ok")
