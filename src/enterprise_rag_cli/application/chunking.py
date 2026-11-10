from __future__ import annotations

from chonkie import (
    BaseChunker,
    RecursiveChunker,
    RecursiveLevel,
    RecursiveRules,
    SentenceChunker,
    TokenChunker,
)

from enterprise_rag_cli.domain.ids import make_chunk_id
from enterprise_rag_cli.domain.models import Chunk, Document

# 定义「递归切分」的分层规则
def paragraph_rules() -> RecursiveRules:
    # 为 RecursiveChunker 定义优先级由高到低的切分层次
    return RecursiveRules(levels=[
        # 段落 / 换行
        RecursiveLevel(delimiters=["\n\n", "\n"]),
        # 句子
        RecursiveLevel(delimiters=[". ", "! ", "? ", "。", "！", "？"]),
        # 空白
        RecursiveLevel(whitespace=True),
        # 单字符兜底
        RecursiveLevel(),
    ])

# 按策略名创建具体 chunker
def create_chunker(
    chunk_size: int,
    chunk_overlap: int,
    strategy: str = "token",
) -> BaseChunker:
    if chunk_size < 1 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("长度必须为正数，重叠长度必须在 0 到切分长度之间")

    if strategy == "token":
        return TokenChunker(
            tokenizer="character", chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    if strategy == "sentence":
        return SentenceChunker(
            tokenizer="character", chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_characters_per_sentence=1,
            delim=[". ", "! ", "? ", "。", "！", "？", "\n"],
        )
    if strategy == "recursive":
        if chunk_overlap != 0:
            raise ValueError("本节的递归切分不使用重叠，请将 overlap 设为 0")
        return RecursiveChunker(
            tokenizer="character", chunk_size=chunk_size,
            rules=paragraph_rules(), min_characters_per_chunk=1,
        )
    raise ValueError(f"不支持的切分方式：{strategy}")


# 执行切分
def split_document(document: Document, chunker: BaseChunker) -> list[Chunk]:
    fallback = TokenChunker(
        tokenizer="character", chunk_size=chunker.chunk_size, chunk_overlap=0,
    )
    chunks = []
    for raw_chunk in chunker.chunk(document.text):
        # 一条很长的句子也可能超过上限，继续按长度拆开。
        pieces = (
            fallback.chunk(raw_chunk.text)
            if len(raw_chunk.text) > chunker.chunk_size else [raw_chunk]
        )
        for piece in pieces:
            text = piece.text
            if not text.strip():
                continue
            if len(text.encode("utf-8")) > 8192:
                raise ValueError("Chunk 超过 Milvus 正文字段的 8192 字节上限，请减小切分长度")
            position = len(chunks)
            chunks.append(Chunk(
                chunk_id=make_chunk_id(document.document_id, text, position),
                document_id=document.document_id,
                source_type=document.source_type,
                source_id=document.source_id,
                position=position,
                text=text,
                token_count=len(text),
            ))
    return chunks