from app.services.ocr.base import BaseOcrClient
from app.services.llm.base import BaseLLMClient


class AppState:
    ocr_client: BaseOcrClient | None = None
    llm_client: BaseLLMClient | None = None


global_state = AppState()
