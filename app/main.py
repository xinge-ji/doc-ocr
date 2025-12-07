from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.invoice import router as invoice_router

from contextlib import asynccontextmanager
from app.core.config import settings
from app.services.ocr.paddle_vl import PaddleVLOcrClient
from app.services.llm.openai_client import OpenAIClient
from app.state import global_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Loading models...")
    # 这里可以加逻辑：如果 settings.use_mock_ocr 为 True，则加载 MockClient
    global_state.ocr_client = PaddleVLOcrClient()
    global_state.llm_client = OpenAIClient()
    yield
    # Shutdown
    print("Shutting down...")


app = FastAPI(title="Doc OCR Service", lifespan=lifespan)

app.include_router(health_router, prefix="/api")
app.include_router(invoice_router, prefix="/api")
