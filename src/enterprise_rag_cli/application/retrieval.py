from __future__ import annotations

from enterprise_rag_cli.config import Settings
from enterprise_rag_cli.domain.models import RetrievalResult
from enterprise_rag_cli.infrastructure.embeddings import embed_query
from enterprise_rag_cli.infrastructure.vector_store import (
    create_milvus_client,
    search_chunks,
)

# 通过问题召回 top_k的答案
def retrieve_chunks(
    question: str,
    settings: Settings,
    top_k: int = 5,
) -> list[RetrievalResult]:
    if top_k < 1:
        raise ValueError("top_k 必须大于 0")

    query_vector = embed_query(question, settings)
    client = create_milvus_client(settings)

    try:
        return search_chunks(
            client=client,
            collection_name=settings.milvus_collection,
            query_vector=query_vector,
            top_k=top_k,
        )
    finally:
        client.close()