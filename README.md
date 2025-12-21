# Doc OCR Service

FastAPI service scaffold for OCR + invoice extraction.

## Setup

```bash
uv venv
uv pip install -e ".[dev]"
```

## Run

```bash
uv run uvicorn app.main:app --reload
# nohup uv run uvicorn app.main:app --host 0.0.0.0 --port 8080 --log-level info > app.log 2>&1 &
```

### OCR 前处理

- 默认开启文档方向分类 + 去扭曲（Paddle DocPreprocessor）+ 基础图像增强。
- 通过环境变量控制：`OCR_USE_DOC_ORIENTATION`, `OCR_USE_DOC_UNWARPING`, `OCR_USE_BASIC_ENHANCE`。
- Debug：`OCR_DEBUG_SAVE_BASE64=true` 时，会将送入 OCR 的预处理后图片以 base64 形式落盘到 `OCR_DEBUG_SAVE_DIR`（默认 `debug/ocr/{timestamp}/page_*.b64`），请注意磁盘占用。

## Test

```bash
uv run pytest
# curl -s -X POST "http://127.0.0.1:8080/api/invoice/extract"   -F "file=@/home/pZnuvhq.jpg"   | python3 -c "import sys, json; print(json.dumps(json.load(sys.stdin), indent=2, ensure_ascii=False))"

# Random LLM node
curl -s -X POST "http://127.0.0.1:8080/api/invoice/extract" \
  -F "file=@/path/to/invoice.jpg" \
  | python3 -c "import sys, json; print(json.dumps(json.load(sys.stdin), indent=2, ensure_ascii=False))"

# Specify LLM node
curl -s -X POST "http://127.0.0.1:8080/api/invoice/extract?llm_node=primary" \
  -F "file=@/path/to/invoice.jpg" \
  | python3 -c "import sys, json; print(json.dumps(json.load(sys.stdin), indent=2, ensure_ascii=False))"
```
