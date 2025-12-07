from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_env: str = "dev"
    llm_api_key: str
    llm_base_url: str
    llm_model_name: str = "qwen-plus"
    doclayout_model_path: str = "/home/modelscope_cache/models/PaddlePaddle/PP-DocLayoutV2/"
    ocr_vl_rec_backend: str = "vllm-server"
    ocr_vl_rec_server_url: str = "http://localhost:8000/v1"
    ocr_layout_model_name: str = "PP-DocLayoutV2"


settings = Settings()
