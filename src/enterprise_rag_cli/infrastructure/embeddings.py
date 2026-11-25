from __future__ import annotations

from http import HTTPStatus
from math import isfinite

import dashscope
from dashscope import TextEmbedding
from requests.exceptions import ConnectionError, Timeout
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from enterprise_rag_cli.config import Settings

class TemporaryEmbeddingError(RuntimeError):
    """等待后可以再次尝试的模型调用错误。"""

# Tenacity 的修饰器，根据_call_embedding 的异常类型和 HTTP 状态码来决定是否重试。
@retry(
    # 遇到这三种异常或其子类时才重试
    retry=retry_if_exception_type(
        (TemporaryEmbeddingError, ConnectionError, Timeout),
    ),
    # 总共最多尝试 3 次，包含第一次调用，所以最多重试 2 次
    stop=stop_after_attempt(3),
    # 重试前随机等待，等待时间的上限随尝试次数增长，范围由 min 和 max 限制。随机等待可以避免多个请求同时重试
    wait=wait_random_exponential(multiplier=1, min=1, max=8),
    # 重试耗尽后，直接向调用方抛出最后一次的原始异常，方便定位失败原因
    reraise=True,
)
def _call_embedding(texts: list[str], settings: Settings):
    response = TextEmbedding.call(
        api_key=settings.dashscope_api_key.get_secret_value().strip(),
        model=settings.embedding_model,
        input=texts,
        dimension=settings.embedding_dimensions,
    )
    if response.status_code == HTTPStatus.OK:
        return response

    message = (
        f"Embedding 调用失败：{response.code} {response.message}"
        f"；request_id={response.request_id}"
    )
    # 只对明确的短时限流和服务端故障重试。
    rate_limited = (
        response.status_code == HTTPStatus.TOO_MANY_REQUESTS
        and response.code in {
            "Throttling",
            "Throttling.RateQuota",
            "Throttling.BurstRate",
            "Throttling.AllocationQuota",
            "Throttling.Concurrency",
        }
        and "free allocated quota" not in str(response.message).lower()
    )
    if rate_limited or response.status_code in {500, 502, 503, 504}:
        raise TemporaryEmbeddingError(message)
    raise RuntimeError(message)    

# 向量文本
def embed_texts(
    texts: list[str],
    settings: Settings,
) -> list[list[float]]:
    if not texts:
        return []
    if len(texts) > settings.embedding_batch_size:
        raise ValueError("单批文本数量超过 embedding_batch_size")
    if any(not text.strip() for text in texts):
        raise ValueError("待生成向量的文本不能为空")
    if (
        settings.dashscope_api_key is None
        or not settings.dashscope_api_key.get_secret_value().strip()
    ):
        raise RuntimeError("请先配置 RAG_DASHSCOPE_API_KEY")

    dashscope.base_http_api_url = str(
        settings.dashscope_base_http_api_url,
    ).rstrip("/")
    response = _call_embedding(texts, settings)
    rows = sorted(
        response.output["embeddings"],
        key=lambda item: item["index"],
    )
    # 排序后必须与输入位置一一对应，不能缺失或重复。
    if [row["index"] for row in rows] != list(range(len(texts))):
        raise RuntimeError("Embedding 返回的 index 缺失或重复")

    vectors = [row["embedding"] for row in rows]
    if any(
        len(vector) != settings.embedding_dimensions
        or not all(isfinite(value) for value in vector)
        for vector in vectors
    ):
        raise RuntimeError("Embedding 返回的维度或数值无效")
    return vectors

# 向量化问题
def embed_query(question: str, settings: Settings) -> list[float]:
    question = question.strip()
    if not question:
        raise ValueError("问题不能为空")
    return embed_texts([question], settings)[0]