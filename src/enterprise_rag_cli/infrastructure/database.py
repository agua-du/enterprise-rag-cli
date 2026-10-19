from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL
from sqlalchemy.orm import Session, sessionmaker

from enterprise_rag_cli.config import Settings

# 生成数据库连接地址
def build_database_url(settings: Settings) -> URL:
    return URL.create(
        drivername="postgresql+psycopg",
        username=settings.postgres_user,
        password=settings.postgres_password.get_secret_value(),
        host=settings.postgres_host,
        port=settings.postgres_port,
        database=settings.postgres_database,
    )

# 创建数据库引擎（Engine）
def create_database_engine(settings: Settings) -> Engine:
    return create_engine(
        build_database_url(settings),
        pool_pre_ping=True,
    )

# 创建 Session 工厂
def create_session_factory(
    settings: Settings,
) -> sessionmaker[Session]:
    return sessionmaker(
        bind=create_database_engine(settings),
        expire_on_commit=False,
    )