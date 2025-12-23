import logging
from typing import Final

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.api.deps import OCRClientDep
from app.schemas.invoice import TemplateExtractionResponse
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
    response_model=TemplateExtractionResponse,
)
async def extract_invoice(
    ocr_client: OCRClientDep,
    file: UploadFile = File(..., description="Invoice image or PDF"),
) -> TemplateExtractionResponse:
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

    pipeline = InvoiceExtractionPipeline(ocr_client=ocr_client)
    try:
        result = await pipeline.run(
            payload,
            filename=file.filename,
            content_type=file.content_type,
        )
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - passthrough for service failures
        logger.exception("Invoice extraction failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Invoice extraction failed.",
        ) from exc

    if not result.complete or result.data is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Rule extraction failed.", "errors": result.errors},
        )

    return TemplateExtractionResponse(
        success=True,
        template_name=result.template_name,
        data=result.data,
        message="ok",
    )
