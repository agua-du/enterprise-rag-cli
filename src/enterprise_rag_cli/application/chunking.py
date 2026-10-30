from __future__ import annotations

from chonkie import TokenChunker

from enterprise_rag_cli.domain.ids import make_chunk_id
from enterprise_rag_cli.domain.models import Chunk, Document

# 创建chunker
def create_chunker(
    chunk_size: int,
    chunk_overlap: int,
) -> TokenChunker:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    return TokenChunker(
        tokenizer="character",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

# 分割文档
def split_document(
    document: Document,
    chunker: TokenChunker,
) -> list[Chunk]:
    # 分割
    source_chunks = chunker.chunk(document.text)

    return [
        Chunk(
            chunk_id=make_chunk_id(
                document.document_id,
                source_chunk.text,
                position,
            ),
            document_id=document.document_id,
            source_type=document.source_type,
            source_id=document.source_id,
            position=position,
            text=source_chunk.text,
            token_count=source_chunk.token_count,
        )
        for position, source_chunk in enumerate(source_chunks)
        if source_chunk.text.strip()
    ]