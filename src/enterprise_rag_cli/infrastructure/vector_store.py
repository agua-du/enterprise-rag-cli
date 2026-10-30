from __future__ import annotations

from pymilvus import DataType, MilvusClient

from enterprise_rag_cli.config import Settings
from enterprise_rag_cli.domain.models import Chunk,RetrievalResult

# 创建milvus
def create_milvus_client(settings: Settings) -> MilvusClient:
    return MilvusClient(uri=settings.milvus_uri)

# 更新数据库
def upsert_chunks(
    client: MilvusClient,
    collection_name: str,
    chunks: list[Chunk],
    vectors: list[list[float]],
) -> None:
    rows = [
        {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "source_type": chunk.source_type,
            "source_id": chunk.source_id,
            "position": chunk.position,
            "text": chunk.text,
            "vector": vector,
        }
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    client.upsert(collection_name=collection_name, data=rows)
    

# 创建一个新的集合
def create_chunk_collection(
    client: MilvusClient,
    settings: Settings,
    collection_name: str,
) -> None:
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )
    schema.add_field(
        field_name="chunk_id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        max_length=64,
    )
    schema.add_field(
        field_name="document_id",
        datatype=DataType.VARCHAR,
        max_length=64,
    )
    schema.add_field(
        field_name="source_type",
        datatype=DataType.VARCHAR,
        max_length=64,
    )
    schema.add_field(
        field_name="source_id",
        datatype=DataType.VARCHAR,
        max_length=128,
    )
    schema.add_field(
        field_name="position",
        datatype=DataType.INT64,
    )
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=8192,
    )
    # 存储向量，dim 是维度
    schema.add_field(
        field_name="vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=settings.embedding_dimensions,
    )

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        # 使用HNSW索引，这里不过多解释，官方解释已经很清晰( https://milvus.io/docs/hnsw.md )
        index_type="HNSW",
        # metric_type="COSINE" 表示使用余献相似度衡量两个向量相似度
        metric_type="COSINE",
        # M是控制HNSW图构建的邻居连接规模
        # efConstruction是HNSW 建立索引时搜索候选邻居的范围
        params={"M": 16, "efConstruction": 200},
    )
    # 创建集合
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
        # 一致性级别：Bounded 表示有界陈旧一致性，允许一定程度的数据可见性延迟
        consistency_level="Bounded",
    )

# 创建或切换别名
def activate_chunk_collection(
    client: MilvusClient,
    collection_name: str,
    alias: str,
) -> None:
    aliases = client.list_aliases()["aliases"]
    if alias in aliases:
        client.alter_alias(
            collection_name=collection_name,
            alias=alias,
        )
        return

    # 第一次迁移时，上一节创建的是同名实体 Collection。
    # 新版本已经完整写入后，才删除它并建立 Alias。
    if client.has_collection(collection_name=alias):
        client.drop_collection(collection_name=alias)
    client.create_alias(
        collection_name=collection_name,
        alias=alias,
    )

# Milvus寻找相识内容
def search_chunks(
    client: MilvusClient,
    collection_name: str,
    query_vector: list[float],
    top_k: int,
) -> list[RetrievalResult]:
    # 加载集合
    client.load_collection(collection_name=collection_name)

    search_results = client.search(
        collection_name=collection_name,
        # 检索向量字段
        anns_field="vector",
        # data 是数组，因为 Milvus 是支持一次检索多个查询向量
        data=[query_vector],
        # 最多返回条数
        limit=top_k,
        search_params={
            # 使用余弦相似度查找
            "metric_type": "COSINE",
            # HNSW搜索参数，ef 越大，搜索速度慢，召回率高，反之搜索快，召回率低
            "params": {"ef": max(64, top_k)},
        },
        # 查询返回结果的字段
        output_fields=[
            "document_id",
            "source_type",
            "source_id",
            "position",
            "text",
        ],
    )
    # 因为Milvus支持多条检索，而我们只检索一个查询向量，所以只取第 1 条
    hits = search_results[0]
    return [
        RetrievalResult(
            rank=rank,
            score=float(hit["distance"]),
            chunk_id=str(hit["chunk_id"]),
            document_id=str(hit["entity"]["document_id"]),
            source_type=str(hit["entity"]["source_type"]),
            source_id=str(hit["entity"]["source_id"]),
            position=int(hit["entity"]["position"]),
            text=str(hit["entity"]["text"]),
        )
        for rank, hit in enumerate(hits, start=1)
    ]
