# Paddle OCR Client Plan

## Context
- Add new PaddleOcrClient using PaddleOCR det+rec (PP-OCRv5_server_det/rec)
- Disable doc orientation/unwarping/textline orientation; device=gpu
- Save visualization locally; no JSON persistence in return payload; output OcrResult with left-top/right-bottom bbox

## Steps
1) Add config defaults for Paddle det/rec models, device, visualization dir
2) Implement app/services/ocr/paddle_ocr.py with PaddleOcrClient
3) Export client in app/services/ocr/__init__.py
4) (Optional) Add mock-based unit test for bbox/text parsing
