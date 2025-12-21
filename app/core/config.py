from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LlmNode(BaseModel):
    name: str
    base_url: str
    api_key: str
    model: str

    @field_validator("name")
    @classmethod
    def _lowercase_name(cls, value: str) -> str:
        """Normalize node name for case-insensitive matching."""

        return value.strip().lower()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_env: str = "dev"
    log_level: str = "info"
    doclayout_model_path: str = "/home/modelscope_cache/models/PaddlePaddle/PP-DocLayoutV2/"
    ocr_vl_rec_backend: str = "vllm-server"
    ocr_vl_rec_server_url: str = "http://localhost:8000/v1"
    ocr_layout_model_name: str = "PP-DocLayoutV2"
    llm_nodes: list[LlmNode]

    @field_validator("llm_nodes")
    @classmethod
    def _ensure_nodes_not_empty(cls, value: list[LlmNode]) -> list[LlmNode]:
        if not value:
            raise ValueError("LLM_NODES must contain at least one node")
        # 去重并保持原有顺序
        seen = set()
        deduped: list[LlmNode] = []
        for node in value:
            if node.name in seen:
                raise ValueError(f"Duplicate LLM node name: {node.name}")
            seen.add(node.name)
            deduped.append(node)
        return deduped


settings = Settings()
