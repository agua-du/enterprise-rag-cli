from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, String, Text, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# DeclarativeBase是 ORM 基类，所有数据库模型都集成它
class Base(DeclarativeBase):
    pass

# 定义表结构
class DocumentRecord(Base):
    # 数据库表名
    __tablename__ = "documents"
    # 表参数
    __table_args__ = (
        # 定义联合唯一约束，source_type和 source_id组合唯一
        UniqueConstraint(
            "source_type",
            "source_id",
            name="uq_documents_source",
        ),
    )
    # document_id为列名，类型字符串，String(36)表示最多 36 个字符，primary_key意思是该列为主键
    # document_id VARCHAR(36) 文档唯一 ID，主键
    document_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
    )
    # index=True表示为该字段创建数据库索引
    source_type: Mapped[str] = mapped_column(
        String(64),
        index=True,
    )
    source_id: Mapped[str] = mapped_column(String(128))
    # source_path TEXT ，TEXT 没有指定长度
    source_path: Mapped[str] = mapped_column(Text)

    title: Mapped[str | None] = mapped_column(Text)
    text: Mapped[str | None] = mapped_column(Text)

    content_sha256: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), index=True)
    error_message: Mapped[str | None] = mapped_column(Text)

    # created_at TIMESTAMP WITH TIME ZONE 创建时间
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        # 插入记录时，如果没有指定该字段，数据库自动生成当前时间
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )