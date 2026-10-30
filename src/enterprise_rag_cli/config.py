from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field,AnyHttpUrl,SecretStr 
from pydantic_settings import BaseSettings,SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_default=True,
    )

    environment: Literal["development", "test", "production"] = "development"

    data_dir: Path = Path("data/enterprise-rag-bench")
    reports_dir: Path = Path("reports")

    postgres_host: str = "localhost"
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    postgres_user: str = "rag"
    postgres_password: SecretStr = SecretStr("rag-local-only")
    postgres_database: str = "enterprise_rag"

    chunk_size:int = Field(default=1200,ge=200)
    chunk_overlap:int = Field(default=150,ge=0)

    dashscope_api_key: SecretStr | None = None
    dashscope_base_http_api_url: AnyHttpUrl

    embedding_model: str = "qwen3.7-text-embedding-flash"
    embedding_dimensions: int = 1024
    embedding_batch_size: int = Field(default=20, ge=1, le=20)

    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "document_chunks"

    dashscope_compatible_base_url: AnyHttpUrl

    # 全量索引按页读取文档，避免一次加载整个文档库。
    full_index_document_page_size: int = Field(default=500, ge=1)

    # 单个分片主动低于百炼的 50,000 条请求和 500 MB 上限。
    batch_shard_max_requests: int = Field(default=5000, ge=1, le=50_000)
    batch_shard_max_bytes: int = Field(
        default=400_000_000,
        ge=1,
        le=500_000_000,
    )

    # 从结果文件读取向量后，按较小批次写入 Milvus。
    milvus_write_batch_size: int = Field(default=500, ge=1)

    # 生成答案的模型
    generation_model: str = "qwen3.8-flash"
    # 默认Milvus召回最大数量
    default_top_k: int = Field(default=5, ge=1, le=20)


#缓存修饰器
@lru_cache(maxsize=1)
def get_settings()->Settings:
    return Settings()
