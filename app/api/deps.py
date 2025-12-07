from typing import Annotated
from fastapi import Depends, HTTPException
from app.services.ocr.base import BaseOcrClient
from app.services.llm.base import BaseLLMClient
from app.state import global_state


async def get_ocr_client() -> BaseOcrClient:
    if not global_state.ocr_client:
        raise HTTPException(status_code=503, detail="OCR Service not initialized")
    return global_state.ocr_client


async def get_llm_client() -> BaseLLMClient:
    if not global_state.llm_client:
        raise HTTPException(status_code=503, detail="LLM Service not initialized")
    return global_state.llm_client


OCRClientDep = Annotated[BaseOcrClient, Depends(get_ocr_client)]
LLMClientDep = Annotated[BaseLLMClient, Depends(get_llm_client)]
