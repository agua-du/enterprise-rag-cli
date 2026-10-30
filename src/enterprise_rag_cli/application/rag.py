from __future__ import annotations

from enterprise_rag_cli.application.retrieval import retrieve_chunks
from enterprise_rag_cli.config import Settings
from enterprise_rag_cli.domain.models import RagAnswer
from enterprise_rag_cli.infrastructure.generation import (
    generate_answer,
    render_answer_prompt,
)

# 回答问题
def answer_question(
    question: str,
    settings: Settings,
    top_k: int,
) -> RagAnswer:
    question = question.strip()
    if not question:
        raise ValueError("问题不能为空")

    evidence = retrieve_chunks(
        question=question,
        settings=settings,
        top_k=top_k,
    )
    if not evidence:
        raise RuntimeError("Milvus 没有返回可用的检索结果")

    prompt = render_answer_prompt(question, evidence)
    answer = generate_answer(prompt, settings)

    return RagAnswer(
        question=question,
        answer=answer,
        evidence=evidence,
    )