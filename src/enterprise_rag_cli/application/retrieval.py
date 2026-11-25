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
    ef: int | None = None,
) -> list[RetrievalResult]:
    actual_ef = settings.hnsw_ef_search if ef is None else ef
    if not 1 <= top_k <= actual_ef <= 32768:
        raise ValueError("参数需要满足 1 <= top_k <= ef <= 32768")

    query_vector = embed_query(question, settings)
    client = create_milvus_client(settings)
    try:
        client.load_collection(collection_name=settings.milvus_collection)
        return search_chunks(
            client=client,
            collection_name=settings.milvus_collection,
            query_vector=query_vector,
            top_k=top_k,
            ef=actual_ef,
        )
    finally:
        client.close()