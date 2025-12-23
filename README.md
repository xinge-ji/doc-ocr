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

## Preprocess

- 预处理包含表格线抑制与透视可信度判断，避免把表格框当成外框裁剪。

## Rule-based invoice templates

- Templates live under `app/invoice_templates/` as JSON.
- Pipeline flow is OCR -> rules; output fields are defined by the matched template.
- When rule extraction is incomplete, the API returns `422` with rule errors.
- Template `fields` define types, descriptions, and constraints used during validation.

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
