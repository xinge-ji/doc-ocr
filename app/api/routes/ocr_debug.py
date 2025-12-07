from fastapi import APIRouter, HTTPException, status

router = APIRouter(prefix="/ocr", tags=["ocr"])


@router.post("/debug", summary="Debug OCR output")
async def debug_ocr() -> dict[str, str]:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="OCR debug endpoint is not implemented yet.",
    )
