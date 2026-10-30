from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy.orm import Session

from enterprise_rag_cli.application.chunking import (
    create_chunker,
    split_document,
)
from enterprise_rag_cli.config import Settings
from enterprise_rag_cli.domain.models import Chunk
from enterprise_rag_cli.infrastructure.batch_embeddings import (
    write_batch_shards,
    create_batch_client,
    get_batch_status,
    submit_embedding_batch,
    download_batch_result,
    iter_batch_vector_groups
)
from enterprise_rag_cli.infrastructure.document_repository import (
    count_parsed_documents,
    iter_parsed_documents,
)

from enterprise_rag_cli.infrastructure.vector_store import (
    activate_chunk_collection,
    create_chunk_collection,
    create_milvus_client,
    upsert_chunks,
)


@dataclass(frozen=True, slots=True)
class FullIndexRun:
    run_id: str
    run_dir: Path
    documents: int
    chunks: int
    shards: int

@dataclass(frozen=True, slots=True)
class AppliedIndex:
    collection_name: str
    chunks: int
    applied_shards: int
    total_shards: int
    activated: bool
    already_active: bool = False

# manifest文件路径
def _manifest_path(settings: Settings, run_id: str) -> Path:
    return settings.reports_dir / "index-jobs" / run_id / "manifest.json"

# 读取manifest文件的内容
def _load_manifest(settings: Settings, run_id: str) -> dict:
    path = _manifest_path(settings, run_id)
    if not path.exists():
        raise ValueError(f"找不到全量索引任务：{run_id}")
    return json.loads(path.read_text(encoding="utf-8"))

# 生成 manifest 文件
def _save_manifest(
    settings: Settings,
    run_id: str,
    manifest: dict,
) -> None:
    path = _manifest_path(settings, run_id)
    # 创建.tmp为后缀的临时文件
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # 替换原文件
    temp_path.replace(path)

# 读取文档，然后转换成 Chunk，再转成分片，并且使用 manifest 记录下任务
def prepare_full_index(
    session: Session,
    settings: Settings,
) -> FullIndexRun:
    document_count = count_parsed_documents(session)
    if document_count == 0:
        raise ValueError("PostgreSQL 中没有可以建立索引的文档")

    chunker = create_chunker(
        settings.chunk_size,
        settings.chunk_overlap,
    )

    def chunks():
        # 读取数据库中文档
        documents = iter_parsed_documents(
            session,
            page_size=settings.full_index_document_page_size,
        )
        # 分成Chunk
        for document in documents:
            yield from split_document(document, chunker)

    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    run_id = f"{timestamp}-{uuid4().hex[:8]}"
    run_dir = settings.reports_dir / "index-jobs" / run_id
    run_dir.mkdir(parents=True)

    # 把 Chunk 拆分成多个分片
    shards = write_batch_shards(
        chunks=chunks(),
        shards_dir=run_dir / "shards",
        settings=settings,
    )

    chunk_count = sum(shard.chunks for shard in shards)
    collection_name = (
        f"{settings.milvus_collection}_{run_id.replace('-', '_')}"
    )
    manifest = {
        "run_id": run_id,
        "collection_name": collection_name,
        "documents": document_count,
        "chunks": chunk_count,
        "shards": [
            {
                "shard_id": shard.shard_id,
                "requests": shard.requests,
                "chunks": shard.chunks,
                "request_bytes": shard.request_bytes,
                "batch_id": None,
                "status": "prepared",
                "applied": False,
            }
            for shard in shards
        ],
    }
    _save_manifest(settings, run_id, manifest)

    return FullIndexRun(
        run_id=run_id,
        run_dir=run_dir,
        documents=document_count,
        chunks=chunk_count,
        shards=len(shards),
    )

# 提交批量 Embedding 处理，并记录在 manifest
def submit_batch_shards(settings: Settings, run_id: str) -> int:
    manifest = _load_manifest(settings, run_id)
    run_dir = _manifest_path(settings, run_id).parent
    client = create_batch_client(settings)
    submitted = 0

    for shard in manifest["shards"]:
        # 已提交且仍在正常处理中，不重复创建任务。
        if shard["batch_id"] is not None and shard["status"] not in {
            "failed",
            "expired",
            "cancelled",
        }:
            continue

        shard_id = shard["shard_id"]
        # 批量Embedding
        batch_id = submit_embedding_batch(
            client=client,
            input_path=(
                run_dir / "shards" / shard_id / "requests.jsonl"
            ),
            task_name=f"enterprise-rag-{run_id}-{shard_id}",
        )
        shard["batch_id"] = batch_id
        shard["status"] = "submitted"
        _save_manifest(settings, run_id, manifest)
        submitted += 1

    return submitted

# 查看批量 Embedding 的状态，并记录在 manifest
def refresh_batch_status(
    settings: Settings,
    run_id: str,
) -> dict[str, int]:
    manifest = _load_manifest(settings, run_id)
    client = create_batch_client(settings)
    counts: dict[str, int] = {}

    for shard in manifest["shards"]:
        batch_id = shard["batch_id"]
        if batch_id is None:
            status = "prepared"
        else:
            status = get_batch_status(client, batch_id)
        shard["status"] = status
        counts[status] = counts.get(status, 0) + 1

    _save_manifest(settings, run_id, manifest)
    return counts


def apply_full_index(
    settings: Settings,
    run_id: str,
) -> AppliedIndex:
    # 读取 Manifest，获取本次索引任务的信息
    manifest = _load_manifest(settings, run_id)
    run_dir = _manifest_path(settings, run_id).parent
    collection_name = manifest["collection_name"]
    shards = manifest["shards"]

    batch_client = create_batch_client(settings)
    milvus = create_milvus_client(settings)

    try:
        # 检查目标 Collection 是否已经激活
        alias_info = None
        aliases = milvus.list_aliases()["aliases"]
        if settings.milvus_collection in aliases:
            alias_info = milvus.describe_alias(
                alias=settings.milvus_collection,
            )

        # Alias：Collection 的别名，可以指向不同的 Collection
        # 这里判断如果当前别名指向的 Collection 已经是 Manifest 的集合了，那说明它已经是当前正式使用的索引了
        if (
            alias_info is not None
            and alias_info["collection_name"] == collection_name
        ):
            return AppliedIndex(
                collection_name=collection_name,
                chunks=manifest["chunks"],
                applied_shards=len(shards),
                total_shards=len(shards),
                activated=True,
                already_active=True,
            )
        # Collection不存在，创建 Collection
        if not milvus.has_collection(collection_name=collection_name):
            create_chunk_collection(
                client=milvus,
                settings=settings,
                collection_name=collection_name,
            )
            # Collection 被重新创建时，所有分片是否应用的状态改成 False。
            for shard in shards:
                shard["applied"] = False
            _save_manifest(settings, run_id, manifest)

        for shard in shards:
            # 跳过已经写过的分片
            if shard["applied"]:
                continue

            batch_id = shard["batch_id"]
            if batch_id is None:
                continue

            # 跳过没有完成的分片
            status = get_batch_status(batch_client, batch_id)
            shard["status"] = status
            if status != "completed":
                continue

            # 查看下载的结果文件，如果没有下载，就下载
            shard_dir = run_dir / "shards" / shard["shard_id"]
            output_path = shard_dir / f"output-{batch_id}.jsonl"
            if not output_path.exists():
                download_batch_result(
                    batch_client,
                    batch_id,
                    output_path,
                )

            # 循环Chunks 和向量结果，依次更新到 Milvus 数据库中
            pending_chunks: list[Chunk] = []
            pending_vectors: list[list[float]] = []

            for current_chunks, current_vectors in (
                iter_batch_vector_groups(
                    output_path=output_path,
                    chunks_path=shard_dir / "chunks.jsonl",
                )
            ):
                if any(
                    len(vector) != settings.embedding_dimensions
                    for vector in current_vectors
                ):
                    raise RuntimeError("Batch 返回的向量维度不正确")

                pending_chunks.extend(current_chunks)
                pending_vectors.extend(current_vectors)

                # 达到写入的批次大小，执行 Upsert
                while (
                    len(pending_chunks)
                    >= settings.milvus_write_batch_size
                ):
                    size = settings.milvus_write_batch_size
                    upsert_chunks(
                        client=milvus,
                        collection_name=collection_name,
                        chunks=pending_chunks[:size],
                        vectors=pending_vectors[:size],
                    )
                    del pending_chunks[:size]
                    del pending_vectors[:size]

            if pending_chunks:
                upsert_chunks(
                    client=milvus,
                    collection_name=collection_name,
                    chunks=pending_chunks,
                    vectors=pending_vectors,
                )
            # 写入Milvus 后，记录 applied 为 True
            shard["applied"] = True
            _save_manifest(settings, run_id, manifest)

        # 统计应用的个数
        applied_shards = sum(
            1 for shard in shards if shard["applied"]
        )
        _save_manifest(settings, run_id, manifest)

        # 没有全部完成
        if applied_shards != len(shards):
            return AppliedIndex(
                collection_name=collection_name,
                chunks=manifest["chunks"],
                applied_shards=applied_shards,
                total_shards=len(shards),
                activated=False,
            )
        # 全部完成后，Milvus 将当前 Collection 的缓冲数据持久化
        milvus.flush(collection_name=collection_name)
        stats = milvus.get_collection_stats(
            collection_name=collection_name,
        )
        # 校验Milvus中的数据量
        if int(stats["row_count"]) != manifest["chunks"]:
            raise RuntimeError("Milvus 写入数量与 Chunk 数量不一致")

        # 使用新Collection
        activate_chunk_collection(
            client=milvus,
            collection_name=collection_name,
            alias=settings.milvus_collection,
        )
        return AppliedIndex(
            collection_name=collection_name,
            chunks=manifest["chunks"],
            applied_shards=applied_shards,
            total_shards=len(shards),
            activated=True,
        )
    finally:
        milvus.close()
