from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

#待处理的文件
class SourceDocument(BaseModel):
    #source_type 表示文档来源，例如 fireflies、github 或 slack
    source_type: str = Field(min_length=1)
    #文档在原始数据中的 ID
    source_id: str = Field(min_length=1)
    #文件在本地文档库中的位置
    path: Path
    #文件内容的指纹，用来判断文件内容是否发生变化
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

#解析后的完整文档
class Document(BaseModel):
    #项目内部使用的文档 ID
    document_id: str = Field(min_length=1)
    #原始来源信息
    source_type: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    #从文件中提取出的标题与完整正文
    title: str | None = None
    text: str = Field(min_length=1)
    #让解析结果仍然能够追溯到对应的原始文件内容
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

#可检索的文档片段
class Chunk(BaseModel):
    #分块ID
    chunk_id: str = Field(min_length=1)
    #说明这个分块来自哪一篇文档
    document_id: str = Field(min_length=1)
    #表示它在文档中的顺序，从 0 开始
    position: int = Field(ge=0)
    #分块正文
    text: str = Field(min_length=1)
    #录分块长度，便于后面控制送入模型的上下文大小
    token_count: int = Field(gt=0)