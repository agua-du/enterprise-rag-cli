from __future__ import annotations

from http import HTTPStatus

import dashscope
from dashscope import TextEmbedding

from enterprise_rag_cli.config import Settings

# 向量文本
def embed_texts(
    texts: list[str],
    settings: Settings,
) -> list[list[float]]:
    if not texts:
        return []

    if len(texts) > settings.embedding_batch_size:
        raise ValueError("单批文本数量超过 embedding_batch_size")

    if settings.dashscope_api_key is None:
        raise RuntimeError("请先配置 RAG_DASHSCOPE_API_KEY")

    api_key = settings.dashscope_api_key.get_secret_value().strip()
    if not api_key:
        raise RuntimeError("请先配置 RAG_DASHSCOPE_API_KEY")

    dashscope.base_http_api_url = str(
        settings.dashscope_base_http_api_url
    ).rstrip("/")

    response = TextEmbedding.call(
        api_key=api_key,
        model=settings.embedding_model,
        input=texts,
        dimension=settings.embedding_dimensions,
    )

    if response.status_code != HTTPStatus.OK:
        raise RuntimeError(
            f"Embedding 调用失败：{response.code} {response.message}"
        )

    # 根据index字段来排序
    rows = sorted(
        response.output["embeddings"],
        key=lambda item: item["index"],
    )
    vectors = [row["embedding"] for row in rows]

    if len(vectors) != len(texts):
        raise RuntimeError("Embedding 返回数量与输入数量不一致")

    if any(
        len(vector) != settings.embedding_dimensions
        for vector in vectors
    ):
        raise RuntimeError("Embedding 返回维度与配置不一致")

    return vectors

# embedding问题
def embed_query(
    question: str,
    settings: Settings,
) -> list[float]:
    question = question.strip()
    if not question:
        raise ValueError("问题不能为空")

    return embed_texts([question], settings)[0]