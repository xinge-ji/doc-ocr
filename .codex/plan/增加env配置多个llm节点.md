# 任务：增加 env 配置多个 LLM 节点，支持指定/随机调用

## 背景
- 现有只支持单 LLM 配置：`llm_base_url`、`llm_api_key`、`llm_model_name`
- 需求：env 中配置多个 LLM 节点（name/base_url/api_key/model），调用时可指定节点或随机选择
- API `/api/invoice/extract` 增加 query `llm_node`，未指定随机，指定不存在报 422
- 更新 README 增加 curl 示例；不用考虑老配置兼容

## 约束
- Python 3.12，FastAPI，openai SDK
- 配置用 pydantic-settings，从 `.env` 读取
- 多节点配置采用 JSON 字符串（环境变量 `LLM_NODES`）
- 沙箱：初始只读，已获写权限

## 执行步骤
1. 调整配置层 `app/core/config.py`
   - 增加 `LlmNode` 模型（name/base_url/api_key/model）
   - `Settings` 新增 `llm_nodes: list[LlmNode]`，要求非空列表
   - 移除单节点字段依赖（保留其他配置）
2. `app/services/llm/openai_client.py`
   - 初始化：为每个节点创建 `AsyncOpenAI` 客户端缓存
   - 增加 `_pick_node(node_name: str | None)`，不传随机，传名不存在抛 `ValueError`
   - `generate_structured` 支持 `node_name` 参数并使用对应客户端
   - base_url 正常化逻辑保留
3. `app/services/pipelines/invoice.py`
   - `run` 增加可选参数 `llm_node`，传给 `generate_structured`
4. `app/api/routes/invoice.py`
   - 增加 query 参数 `llm_node`
   - 节点不存在抛 HTTP 422
5. `.env.example`
   - 增加 `LLM_NODES` JSON 示例（含两个节点）
   - 移除单节点示例变量
6. `README.md`
   - 补充 curl 示例，展示带/不带 `llm_node`（随机/指定）

## 测试与验证
- 手动检查：未传 llm_node 时随机；传错名返回 422；传正确名走对应节点
- 代码自检：ruff 风格、类型提示覆盖新增代码
