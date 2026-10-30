from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from collections.abc import Iterator

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from enterprise_rag_cli.domain.models import Document, SourceDocument
from enterprise_rag_cli.infrastructure.tables import DocumentRecord


ImportAction = Literal["created", "updated", "unchanged"]

@dataclass(frozen=True, slots=True)
class ImportSummary:
    source_type: str
    parsed: int
    failed: int

# 把解析成功的文档 存储/更新 到数据库中
def save_document(
    session: Session,
    source: SourceDocument,
    document: Document,
    source_path: str,
) -> ImportAction:
    # 查询 document_id 这条数据
    # select * from documents where document_id = "xx"
    record = session.get(DocumentRecord, document.document_id)

    # 这个文档已经录入，并且没有改动
    if (
        record is not None
        and record.content_sha256 == document.content_sha256
        and record.source_path == source_path
        and record.status == "parsed"
    ):
        return "unchanged"

    action: ImportAction = "created" if record is None else "updated"

    # 数据库中没有，添加 ORM 对象
    if record is None:
        record = DocumentRecord(
            document_id=document.document_id,
            source_type=document.source_type,
            source_id=document.source_id,
            source_path=source_path,
            title=document.title,
            text=document.text,
            content_sha256=document.content_sha256,
            status="parsed",
            error_message=None,
        )
        # 这里并不等于立即执行insert，还没提交事务
        session.add(record)
        return action

    # 修改ORM对象
    record.source_type = source.source_type
    record.source_id = source.source_id
    record.source_path = source_path
    record.title = document.title
    record.text = document.text
    record.content_sha256 = document.content_sha256
    record.status = "parsed"
    record.error_message = None
    return action

# 解析失败也要在数据库中创建一个失败记录
def save_failure(
    session: Session,
    source: SourceDocument,
    document_id: str,
    source_path: str,
    error_message: str,
) -> None:
    record = session.get(DocumentRecord, document_id)

    if record is None:
        record = DocumentRecord(
            document_id=document_id,
            source_type=source.source_type,
            source_id=source.source_id,
            source_path=source_path,
            title=None,
            text=None,
            content_sha256=source.content_sha256,
            status="failed",
            error_message=error_message,
        )
        session.add(record)
        return

    record.source_path = source_path
    record.title = None
    record.text = None
    record.content_sha256 = source.content_sha256
    record.status = "failed"
    record.error_message = error_message

# 统计数据库中成功和失败个数，按类型分类
def summarize_imports(session: Session) -> list[ImportSummary]:
    """
    select 表示构建查询语句:select source_type from documents
    func.count() 表示聚合统计：count(*)
    filter 表示条件聚合： count(*) filter (where status = 'parsed')
    group_by 分组统计
    order_by 升序排序

    最终 SQL：
        SELECT
            source_type,
            COUNT(*) FILTER (WHERE status = 'parsed'),
            COUNT(*) FILTER (WHERE status = 'failed')
        FROM documents
        GROUP BY source_type
        ORDER BY source_type;
    """
    statement = (
        select(
            DocumentRecord.source_type,
            func.count().filter(DocumentRecord.status == "parsed"),
            func.count().filter(DocumentRecord.status == "failed"),
        )
        .group_by(DocumentRecord.source_type)
        .order_by(DocumentRecord.source_type)
    )

    # session.execute(statement)执行SQL
    return [
        ImportSummary(
            source_type=source_type,
            parsed=parsed,
            failed=failed,
        )
        for source_type, parsed, failed in session.execute(statement)
    ]

# 从数据库中读取文档
def load_parsed_documents(
    session: Session,
    limit: int | None = None,
) -> list[Document]:
    '''
    SQL语句：
    select * from documents
    where status = 'parsed' and text is not null
    order by document_id ASC
    limit 100
    '''
    statement = (
        select(DocumentRecord)
        .where(
            DocumentRecord.status == "parsed",
            DocumentRecord.text.is_not(None),
        )
        .order_by(DocumentRecord.document_id)
    )

    if limit is not None:
        statement = statement.limit(limit)

    # session.execute() 返回 Result 对象，每行通常一个 ROW
    # session.scalars() 返回 ScalarResult 对象，可以直接获取每行的第一个元素
    records = session.scalars(statement).all()
    return [
        Document(
            document_id=record.document_id,
            source_type=record.source_type,
            source_id=record.source_id,
            title=record.title,
            text=record.text,
            content_sha256=record.content_sha256,
        )
        for record in records
    ]

# 统计 parsed 的文档数量
def count_parsed_documents(session: Session) -> int:
    statement = (
        select(func.count())
        .select_from(DocumentRecord)
        .where(
            DocumentRecord.status == "parsed",
            DocumentRecord.text.is_not(None),
        )
    )
    return session.scalar(statement) or 0

# 分批从数据库中读取文档（大量数据时使用）
def iter_parsed_documents(
    session: Session,
    page_size: int,
) -> Iterator[Document]:
    statement = (
        select(DocumentRecord)
        .where(
            DocumentRecord.status == "parsed",
            DocumentRecord.text.is_not(None),
        )
        .order_by(DocumentRecord.document_id)
        .execution_options(yield_per=page_size)
    )

    for record in session.scalars(statement):
        yield Document(
            document_id=record.document_id,
            source_type=record.source_type,
            source_id=record.source_id,
            title=record.title,
            text=record.text,
            content_sha256=record.content_sha256,
        )