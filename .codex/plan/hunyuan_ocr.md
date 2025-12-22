任务：接入 Hunyuan OCR 客户端并支持后端切换

方案概述（方案1）：
- 新增独立 `HunyuanOcrClient`，实现 `BaseOcrClient`。
- 复用预处理逐页出图；按本地路径或 data URL 或远程 URL 组装多模态 messages。
- 调用本地 OpenAI 兼容 `/chat/completions`（模型默认 `Tencent-Hunyuan/HunyuanOCR`），解析 `(x,y),(x,y)` 为 `BoundingBox`。
- 保持 Paddle 默认，提供 env `OCR_BACKEND` 切换。

TODO 步骤：
1) 配置：`Settings` 增加 `ocr_backend`/`hunyuan_base_url`/`hunyuan_api_key`/`hunyuan_model`；`.env.example` 补充。
2) 客户端：新增 `app/services/ocr/hunyuan_ocr.py`，实现提取与解析逻辑。
3) 入口：`app/main.py` 按 `OCR_BACKEND` 初始化 Paddle 或 Hunyuan。
4) 测试：新增 `tests/test_hunyuan_ocr_client.py`，mock httpx 返回内容，覆盖 bytes 与路径分支。

备注：
- block_id/line_id 统一填 None，置信度暂不返回。
- 请求头 `Authorization: Bearer <api_key>`，默认 api_key 可为 EMPTY。
