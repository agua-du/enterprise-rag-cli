from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from openai import OpenAI

from enterprise_rag_cli.config import Settings
from enterprise_rag_cli.domain.models import Chunk


@dataclass(frozen=True, slots=True)
class PreparedShard:
    shard_id: str
    requests: int
    chunks: int
    request_bytes: int

# 创建 OpenAI
def create_batch_client(settings: Settings) -> OpenAI:
    if settings.dashscope_api_key is None:
        raise RuntimeError("请先配置 RAG_DASHSCOPE_API_KEY")

    return OpenAI(
        api_key=settings.dashscope_api_key.get_secret_value(),
        base_url=str(settings.dashscope_compatible_base_url).rstrip("/"),
    )

# 创建任务分片，把 Chunk 分成多个分片
def write_batch_shards(
    chunks: Iterable[Chunk],
    shards_dir: Path,
    settings: Settings,
) -> list[PreparedShard]:
    shards_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[PreparedShard] = []
    pending: list[Chunk] = []

    shard_index = 0
    request_file: TextIO | None = None
    chunk_file: TextIO | None = None
    request_count = 0
    chunk_count = 0
    request_bytes = 0

    # 创建一个新的分片目录，打开两个 JSONL 文件，并重置计数器
    def open_shard() -> None:
        # 声明外部变量
        nonlocal request_file, chunk_file
        nonlocal request_count, chunk_count, request_bytes

        # 将整数格式化为至少 4 位，不足的部分在左侧补零
        shard_id = f"{shard_index:04d}"
        shard_dir = shards_dir / shard_id
        shard_dir.mkdir()
        request_file = (shard_dir / "requests.jsonl").open(
            "w",
            encoding="utf-8",
        )
        chunk_file = (shard_dir / "chunks.jsonl").open(
            "w",
            encoding="utf-8",
        )
        request_count = 0
        chunk_count = 0
        request_bytes = 0

    # 关闭当前分片的文件，记录分片统计信息，并清空文件句柄
    def close_shard() -> None:
        nonlocal request_file, chunk_file

        if request_file is None or chunk_file is None:
            return
        request_file.close()
        chunk_file.close()
        prepared.append(
            PreparedShard(
                shard_id=f"{shard_index:04d}",
                requests=request_count,
                chunks=chunk_count,
                request_bytes=request_bytes,
            )
        )
        request_file = None
        chunk_file = None

    # 生成一条 Batch 请求
    def request_line(current: list[Chunk]) -> str:
        request = {
            # 记录这条请求在当前 chunks.jsonl 中的起始位置。
            "custom_id": f"chunk-offset-{chunk_count:012d}",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "model": settings.embedding_model,
                "input": [chunk.text for chunk in current],
                "encoding_format": "float",
                "dimensions": settings.embedding_dimensions,
            },
        }
        return json.dumps(request, ensure_ascii=False) + "\n"

    # 生成分片
    def write_request(current: list[Chunk]) -> None:
        nonlocal shard_index, request_count, chunk_count, request_bytes

        if request_file is None:
            open_shard()

        line = request_line(current)
        line_bytes = len(line.encode("utf-8"))
        if line_bytes > 1_000_000:
            raise ValueError("单条 Batch 请求超过 1 MB")

        # 当前分片已经有请求，并且请求数量达到上限，或者加入新请求后文件大小会超出上限，就切换到下一个分片
        should_rotate = request_count > 0 and (
            request_count >= settings.batch_shard_max_requests
            or request_bytes + line_bytes > settings.batch_shard_max_bytes
        )
        if should_rotate:
            close_shard()
            shard_index += 1
            open_shard()
            line = request_line(current)
            line_bytes = len(line.encode("utf-8"))

        if request_file is None or chunk_file is None:
            raise RuntimeError("分片文件尚未打开，无法写入请求")

        request_file.write(line)
        for chunk in current:
            chunk_file.write(chunk.model_dump_json() + "\n")

        request_count += 1
        chunk_count += len(current)
        request_bytes += line_bytes

    # 批量处理
    for chunk in chunks:
        pending.append(chunk)
        # 达到批量处理的数量，开始批量处理
        if len(pending) == settings.embedding_batch_size:
            write_request(pending)
            pending = []

    # 处理剩余的
    if pending:
        write_request(pending)
    close_shard()

    if not prepared:
        raise ValueError("没有可以生成向量的 Chunk")
    return prepared


# 提交批量处理请求（一个分片一个分片提交）
def submit_embedding_batch(
    client: OpenAI,
    input_path: Path,
    task_name: str,
) -> str:
    with input_path.open("rb") as file:
        uploaded = client.files.create(
            file=file,
            purpose="batch",
        )

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/embeddings",
        completion_window="24h",
        metadata={
            "ds_name": task_name,
        },
    )
    return batch.id

# 获取批量请求的状态，状态依次是（validating、in_progress、finalizing 和 completed）
def get_batch_status(
    client: OpenAI,
    batch_id: str,
) -> str:
    return client.batches.retrieve(batch_id).status


# 下载Embedding 结果
def download_batch_result(
    client: OpenAI,
    batch_id: str,
    output_path: Path,
) -> None:
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        raise RuntimeError(f"Batch 任务尚未完成：{batch.status}")
    if batch.output_file_id is None:
        raise RuntimeError("Batch 任务没有返回结果文件")

    client.files.content(batch.output_file_id).write_to_file(output_path)

# 记录文件中每一行的起始字节位置
def _line_offsets(path: Path) -> list[int]:
    offsets: list[int] = []
    with path.open("rb") as file:
        while True:
            # .tell()是获取文件当前指针的位置,readline()后，指针移动到下一行
            offset = file.tell()
            line = file.readline()
            if not line:
                break
            offsets.append(offset)
    return offsets

# 读取下载的 Batch Embedding 结果文件，通过 custom_id 找到原始 Chunk 的位置，再利用文件偏移量读取对应的 Chunk，最终将 Chunk 和 Embedding 向量一一匹配
# 这里有人可能问，为什么不直接按行读取，这样是可以的，逻辑更加简单，这里用字节偏移量的方式，主要是为了提高大量 Chunk、Batch 结果乱序时的查找效率
def iter_batch_vector_groups(
    output_path: Path,
    chunks_path: Path,
) -> Iterator[tuple[list[Chunk], list[list[float]]]]:
    offsets = _line_offsets(chunks_path)
    seen_starts: set[int] = set()
    processed = 0

    with (
        output_path.open(encoding="utf-8") as output_file,
        chunks_path.open("rb") as chunks_file,
    ):
        for line in output_file:
            # 结果文件每一行就是一个向量化请求结果
            result = json.loads(line)
            if result.get("error") is not None:
                raise RuntimeError(
                    f"Batch 请求失败：{result['custom_id']}"
                )

            body = result["response"]["body"]
            rows = sorted(
                body["data"],
                key=lambda item: item["index"],
            )

            # 根据custom_id定位 Chunk
            prefix = "chunk-offset-"
            custom_id = result["custom_id"]
            if not custom_id.startswith(prefix):
                raise RuntimeError(f"无法识别 custom_id：{custom_id}")
            start = int(custom_id.removeprefix(prefix))
            if start in seen_starts:
                raise RuntimeError(f"Batch 返回重复结果：{custom_id}")
            if start + len(rows) > len(offsets):
                raise RuntimeError(f"Batch 返回越界结果：{custom_id}")

            #seek()指针移动到指定的位置
            chunks_file.seek(offsets[start])
            current = [
                Chunk.model_validate_json(chunks_file.readline())
                for _ in rows
            ]
            vectors = [row["embedding"] for row in rows]

            seen_starts.add(start)
            processed += len(current)
            yield current, vectors

    if processed != len(offsets):
        raise RuntimeError("Batch 返回的向量数量与 Chunk 数量不一致")