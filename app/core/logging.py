import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

# 定义日志目录和文件
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "doc_ocr.log"


def setup_logging():
    """配置全局日志系统：同时输出到控制台和文件"""

    # 1. 确保日志目录存在
    LOG_DIR.mkdir(exist_ok=True)

    # 2. 定义格式
    # 格式：时间 | 级别 | 模块名:行号 | 内容
    log_format = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s")

    # 3. 文件处理器 (File Handler) - 5MB 一个文件，保留 5 个备份
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(log_format)
    file_handler.setLevel(logging.INFO)

    # 4. 控制台处理器 (Console Handler) - 输出到屏幕/nohup
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    console_handler.setLevel(logging.INFO)

    # 5. 配置根记录器 (Root Logger)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 清除旧的 handlers (避免重复打印)
    root_logger.handlers = []

    # 添加新的 handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # 6. 强制接管 Uvicorn 的日志
    # 这样 Uvicorn 的请求日志也会写到我们的文件里
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
        logger = logging.getLogger(logger_name)
        logger.handlers = [file_handler, console_handler]
        logger.propagate = False  # 防止冒泡导致重复

    logging.info(f"✅ Logging initialized. Logs will be written to: {LOG_FILE.absolute()}")
