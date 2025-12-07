## 1. 项目目标（Project Goal）

本项目的目标是构建一个基于 **FastAPI** 的 OCR 服务，首个核心功能为：**发票关键信息提取服务（Invoice Information Extraction API）**

采用 **Hybrid Extraction** 策略：

1. **OCR 阶段（结构化感知 OCR）**
    使用 **PaddleOCRVL + 本地 vLLM 服务** 对输入图片或 PDF 页面进行 OCR，
    输出包含 **文本内容 + Bounding Box 坐标 + 布局信息** 的结构化结果。
2. **LLM 阶段（结构化+语义抽取）**
    将 OCR 阶段输出（Text + Coordinates + Layout）交由 **LLM（如 Qwen / GPT，通过 openai SDK 调用兼容 OpenAI 协议的接口）**，
    由大模型根据空间布局和语义，抽取发票关键信息（如：发票号、发票代码、日期、总金额、税额、购买方名称等）。
3. **输出**
    所有接口返回 **标准 JSON**，字段命名清晰，类型稳定，适合后续系统对接。

本项目为 **企业内部使用**，当前不考虑多租户和复杂的鉴权逻辑。

------

## 2. 技术栈与硬约束（Tech Stack & Hard Constraints）

AI 助手在本仓库编写代码时，必须遵守如下约束：

### 2.1 语言 & 运行环境

- 编程语言：**Python 3.12+**
- 仅使用与 Python 3.12 兼容的库与语法特性。

### 2.2 包管理器 & 构建工具

- **必须使用 [`uv`](https://github.com/astral-sh/uv) 进行依赖管理和虚拟环境操作**。
- 不得引入 `pipenv`、`poetry`、`conda` 等其他包管理工具相关配置。
- 所有依赖写入 `pyproject.toml` 的 `[project]` / `[project.optional-dependencies]` 章节中。

约定的常用命令（在文档/脚本中引用时，AI 应遵循）：

```bash
# 创建/激活虚拟环境（示例，具体命令可在 README 中定义）
uv venv
uv pip install -e ".[dev]"

# 运行应用
uv run uvicorn app.main:app --reload

# 运行测试
uv run pytest
```

### 2.3 Web 框架

- Web 框架：**FastAPI**
- 服务启动：使用 **Uvicorn**（建议通过 `uv run uvicorn ...`）

### 2.4 OCR 引擎（PaddleOCRVL + vLLM）

- 已有本地 vLLM 服务，用于视觉+语言 OCR：
- **调用方法固定如下（AI 不得随意改动接口签名和初始化方式）**：

```python
from paddleocr import PaddleOCRVL

doclayout_model_path = "/home/modelscope_cache/models/PaddlePaddle/PP-DocLayoutV2/"

pipeline = PaddleOCRVL(
    vl_rec_backend="vllm-server",
    vl_rec_server_url="http://localhost:8000/v1",
    layout_detection_model_name="PP-DocLayoutV2",
    layout_detection_model_dir=doclayout_model_path,
)

output = pipeline.predict(
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png"
)

for i, res in enumerate(output):
    res.save_to_json(save_path=f"output_{i}.json")
    res.save_to_markdown(save_path=f"output_{i}.md")
```

> 在项目代码中，AI 需要对 `PaddleOCRVL` 做一层 **Service 封装**，输出统一的 OCR 结构（文本 + 坐标），避免在业务层直接散落调用。

未来可能接入其他 OCR / Layout 引擎，AI 设计时需要考虑 **接口抽象**。

### 2.5 LLM 交互

- 使用 **`openai` 官方 SDK**（兼容 OpenAI 协议），但实际可能调用 Zhipu/Qwen 等兼容服务。
- 所有密钥、Base URL、模型名从配置中读取，不得写死到代码。

推荐调用模式（AI 可以用类似伪代码，具体细节由项目维护者调整）：

```python
from openai import OpenAI
from app.core.config import settings  # 通过 pydantic-settings 管理

client = OpenAI(
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
)

resp = client.chat.completions.create(
    model=settings.llm_model_name,
    messages=[
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
    ],
)
```

> AI 在写代码时必须：
>
> - 不硬编码 API Key
> - 不硬编码 Base URL
> - 不硬编码模型名（用配置字段）

### 2.6 代码规范 & Lint

- 代码风格 & Lint：**ruff**
- ruff 要求：
  - 支持 Black 风格的格式化（`ruff format`）
  - 基础 Lint（如 `E`, `F`, `I` 等）
- AI 在修改代码时：
  - 保证新增代码符合 ruff 默认风格（尤其是导入顺序、行宽、引号风格）。
  - 尽可能补全 **类型标注（type hints）**。

示例配置（应由 AI 只在需要时扩展，不随意更改）：

```toml
[tool.ruff]
line-length = 100
target-version = "py312"
fix = true

[tool.ruff.lint]
select = ["E", "F", "I"]
ignore = []
```

### 2.7 测试框架

- 使用 **pytest** 作为测试框架。
- 异步测试：**pytest-asyncio**
- HTTP 接口测试：**httpx.AsyncClient** + FastAPI 提供的 TestClient / ASGI adapter。

AI 编写测试时：

- 对异步路由使用 `pytest.mark.asyncio`。
- 尽量聚焦业务逻辑：
  - 一个测试只验证一个行为
  - 使用清晰的命名：`test_invoice_extract_returns_required_fields` 等。

### 2.8 配置管理

- 使用 **pydantic-settings** 管理配置。
- 所有敏感信息从 `.env` 读取。
- AI 创建一个 `Settings` 类，如：

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_env: str = "dev"
    llm_api_key: str
    llm_base_url: str
    llm_model_name: str = "qwen-plus"
    # 其他：端口、日志级别、模型路径等

settings = Settings()
```

------

## 3. 项目结构约定（Project Layout）

AI 在创建/修改文件时，应遵守以下目录结构（如需新增模块，保持风格一致）：

```text
app/
  main.py                # FastAPI 入口
  core/
    config.py            # pydantic-settings 配置
    logging.py           # 日志配置（如需要）
  api/
    deps.py              # 依赖注入（OCR client、LLM client 等）
    routes/
      health.py          # 健康检查
      invoice.py         # 发票信息提取 API 路由
      ocr_debug.py       # 可选：基础 OCR 调试接口
  services/
    ocr/
      base.py            # OCR 接口抽象
      paddle_vl.py       # PaddleOCRVL 实现
    llm/
      base.py            # 通用 LLM 抽象
      openai_client.py   # 通过 openai SDK 调用 Qwen/GPT 等
    pipelines/
      base.py            # Pipeline 抽象
      invoice.py         # 发票 Hybrid 提取 Pipeline
  schemas/
    common.py            # 通用请求/响应模型
    ocr.py               # OCR 结构模型（文本+坐标）
    invoice.py           # 发票字段模型
  prompts/
    invoice_extraction.md   # 发票抽取相关 prompt
    # 后续其他文档类型的 prompt
tests/
  test_health.py
  test_invoice_api.py
  test_invoice_pipeline.py
pyproject.toml
agents.md
README.md
.env.example
```

------

## 4. Hybrid Extraction 设计约定

AI 在实现发票服务时，应遵守以下逻辑约束：

1. **流程顺序**
   - API 接收图片 / PDF（后续支持多页）。
   - 调用 **OCR Service（PaddleOCRVL 封装）** → 得到统一的 `OcrResult`：
     - `text`
     - `bounding box`（四点坐标）
     - `page`
     - `block/line id`
   - 将 `OcrResult` 以及必要的图片信息输入 **LLM Pipeline**：
     - 构造 prompt：包括原始 OCR 文本、坐标信息、字段说明。
   - LLM 返回结构化 JSON → 使用 Pydantic 模型进行校验。
   - 返回给调用方。
2. **接口抽象**
   - 定义 `BaseOcrClient` 抽象类（或协议），以支持未来接入不同 OCR 引擎。
   - 定义 `BaseLLMClient` 抽象类，以支持不同 LLM 实现。
   - 定义 `BasePipeline`（如 `InvoiceExtractionPipeline` 继承），负责 orchestrate OCR + LLM。
3. **错误处理**
   - OCR 或 LLM 调用失败时：
     - API 返回 4xx/5xx 合理状态码 + 错误信息。
     - 日志中打印详细错误（不要包含敏感配置）。

------

## 5. Prompt 管理规范（Prompts Management）

> 目标：让 Prompt 可版本化、可复用、可方便地由人类和 AI 共同维护。

### 5.1 存储位置

- 所有 Prompt 文件放在 `app/prompts/` 目录。
- 文件使用 **Markdown (`.md`)** 或 `.txt`，命名规则：
  - `{业务域}_{用途}.md`，例如：
    - `invoice_extraction_system.md`
    - `invoice_extraction_user.md`
    - `invoice_extraction_fewshot.md`

### 5.2 加载方式

AI 在编写代码时：

- 不要在代码中硬编码大段 Prompt 文本。
- 应编写一个 `PromptLoader` 工具，例如：

```python
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

def load_prompt(name: str) -> str:
    """
    从 prompts 目录加载指定名称的 prompt 文本。
    例如: name="invoice_extraction_system"
    实际文件: prompts/invoice_extraction_system.md
    """
    path = PROMPT_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8")
```

- 在 LLM 调用处，通过 `load_prompt(...)` 获取文案，并根据需要插入动态部分（如 OCR JSON）。

### 5.3 Prompt 内容规范

AI 生成 Prompt 时：

- 使用简洁、明确的中文或中英混合说明。
- 对输出格式做 **明确且严格的 JSON 结构约定**，例如：

~~~markdown
你是一个发票信息抽取助手。

请根据提供的 OCR 结果（包含文本和坐标）提取以下字段：
- invoice_code: 发票代码
- invoice_number: 发票号码
- issue_date: 开票日期（YYYY-MM-DD）
- buyer_name: 购买方名称
- total_amount: 价税合计金额（数字）

请只返回一个 JSON 对象，不要包含任何多余文字。例如：

```json
{
  "invoice_code": "...",
  "invoice_number": "...",
  "issue_date": "2024-01-01",
  "buyer_name": "...",
  "total_amount": 123.45
}
- 尽量避免“聊天风”，突出“任务说明 + 输出格式定义”。
~~~

## 6. 测试要求（Testing Requirements）

AI 在实现新功能时，应该同步编写/更新测试：

1. **API 层测试**
   - 使用 `httpx.AsyncClient` + `pytest-asyncio` 对 FastAPI 路由进行测试。
   - 测试内容包括：
     - 成功路径（200 OK + 正确字段）
     - 错误输入（不支持的文件类型等）

2. **Pipeline 层测试**
   - 对 `InvoiceExtractionPipeline` 做单元测试。
   - 可以：  
     - 用固定的 OCR 伪造数据，mock LLM 响应  
     - 验证解析逻辑能正确映射字段。

3. **LLM 调用逻辑**
   - 尽量通过 mock（例如 monkeypatch openai client），避免真实调用外部服务。


## 7. 禁止事项（Anti-patterns）

AI 在本项目中**不得**执行/生成以下操作：

1. 使用 `pip`, `pipenv`, `poetry` 等替代 `uv` 进行依赖管理说明。
2. 在代码中硬编码：
   - API Key
   - Base URL
   - 模型名称
3. 将大型 Prompt 文本直接写入 Python 源码中（应放入 `app/prompts/`）。
4. 随意更改项目结构或删除已有模块，而没有明确说明迁移路径。
5. 引入与本项目目标无关的大型依赖（如大型 Web UI 框架）而未说明必要性。

