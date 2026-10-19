from __future__ import annotations

from logging.config import fileConfig

from alembic import context

from enterprise_rag_cli.config import get_settings
from enterprise_rag_cli.infrastructure.database import (
    build_database_url,
    create_database_engine,
)
from enterprise_rag_cli.infrastructure.tables import Base


# Alembic 当前运行使用的配置对象，对应项目根目录的 alembic.ini。
config = context.config

# 复用 alembic.ini 中的日志配置，让迁移过程能够正常输出日志。
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 把 SQLAlchemy 的表结构交给 Alembic。
# 执行 revision --autogenerate 时，Alembic 会用它和数据库结构进行比较。
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    # 离线模式不直接连接数据库，而是根据数据库地址生成 SQL。
    # 数据库地址仍然从项目配置和 .env 中读取。
    url = build_database_url(get_settings()).render_as_string(
        hide_password=False,
    )
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # 在线模式建立真实数据库连接，并直接执行迁移。
    engine = create_database_engine(get_settings())

    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


# Alembic 根据当前命令选择离线模式或在线模式。
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()