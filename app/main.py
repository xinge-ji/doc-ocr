import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.routes.health import router as health_router
from app.api.routes.invoice import router as invoice_router
from app.services.ocr.paddle_vl import PaddleVLOcrClient
from app.services.llm.openai_client import OpenAIClient
from app.state import global_state

# 1. 导入 setup_logging
from app.core.logging import setup_logging

# 2. 立即初始化日志 (在 app 创建之前)
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Doc OCR Service...")

    # 加载模型
    logger.info("📦 Loading PaddleOCR models...")
    global_state.ocr_client = PaddleVLOcrClient()

    logger.info("🧠 Initializing LLM client...")
    global_state.llm_client = OpenAIClient()

    logger.info("✅ System ready!")
    yield
    logger.info("🛑 Shutting down service...")


app = FastAPI(title="Doc OCR Service", lifespan=lifespan)

app.include_router(health_router, prefix="/api")
app.include_router(invoice_router, prefix="/api")
