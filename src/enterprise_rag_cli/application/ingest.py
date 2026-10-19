from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from enterprise_rag_cli.domain.ids import make_document_id
from enterprise_rag_cli.domain.models import SourceDocument
from enterprise_rag_cli.infrastructure.document_repository import (
    save_document,
    save_failure,
)
from enterprise_rag_cli.infrastructure.documents import parse_text_document


@dataclass(slots=True)
class IngestStats:
    discovered: int = 0
    created: int = 0
    updated: int = 0
    unchanged: int = 0
    failed: int = 0

# 解析文档，并写入数据库
def ingest_documents(
    session: Session,
    sources: Iterable[SourceDocument],
    documents_dir: Path,
) -> IngestStats:
    stats = IngestStats()

    for source in sources:
        stats.discovered += 1
        source_path = source.path.relative_to(documents_dir).as_posix()
        document_id = make_document_id(
            source.source_type,
            source.source_id,
        )

        try:
            document = parse_text_document(source)
        except (OSError, UnicodeError, ValueError) as error:
            # 解析文档失败，创建失败 ORM 对象
            save_failure(
                session=session,
                source=source,
                document_id=document_id,
                source_path=source_path,
                error_message=str(error),
            )
            stats.failed += 1
            continue

        # 解析文档成功，创建 成功ORM对象
        action = save_document(
            session=session,
            source=source,
            document=document,
            source_path=source_path,
        )

        if action == "created":
            stats.created += 1
        elif action == "updated":
            stats.updated += 1
        else:
            stats.unchanged += 1

    # 执行事务，写入数据库
    session.commit()
    return stats