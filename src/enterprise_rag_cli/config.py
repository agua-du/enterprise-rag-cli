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
        validate_defaults=True,
    )

    environment: Literal["development", "test", "production"] = "development"

    data_dir: Path = Path("data/enterprise-rag-bench")
    reports_dir: Path = Path("reports")

    postgres_host: str = "localhost"
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    postgres_user: str = "rag"
    postgres_password: SecretStr = SecretStr("rag-local-only")
    postgres_database: str = "enterprise_rag"

    opensearch_url: AnyHttpUrl = AnyHttpUrl("http://localhost:9200")
    request_timeout_seconds: float = Field(default=5.0, ge=0, le=60)

#缓存修饰器
@lru_cache(maxsize=1)
def get_settings()->Settings:
    return Settings()
