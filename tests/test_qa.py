
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from sa_agent.retrieval.qa import answer_question


def test_answer_question_returns_answer_and_sources(monkeypatch):
    documents = [
        Document(
            page_content="企业应关注数据安全和人工智能模型风险。",
            metadata={"source": "risk.txt", "chunk_index": 0},
        )
    ]

    # 模拟检索结果，不调用真实向量模型
    monkeypatch.setattr(
        "sa_agent.retrieval.qa.search_documents",
        lambda vector_store, query, k: documents,
    )

    # 模拟聊天模型，不调用真实聊天 API
    fake_model = MagicMock()
    fake_model.invoke.return_value.content = (
        "人工智能行业需要关注数据安全和模型风险。[资料 1]"
    )

    monkeypatch.setattr(
        "sa_agent.retrieval.qa.create_chat_model",
        lambda: fake_model,
    )

    result = answer_question(
        vector_store=object(),
        question="人工智能行业有哪些风险？",
    )

    assert "模型风险" in result["answer"]
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source"] == "risk.txt"
    assert result["sources"][0]["chunk_index"] == 0

    # 确认聊天模型确实收到了包含问题和参考资料的提示词
    prompt = fake_model.invoke.call_args.args[0]
    assert "人工智能行业有哪些风险？" in prompt
    assert "企业应关注数据安全和人工智能模型风险。" in prompt


def test_answer_question_rejects_empty_question():
    with pytest.raises(ValueError, match="问题不能为空"):
        answer_question(vector_store=None, question="   ")


def test_answer_question_returns_fallback_when_no_documents(monkeypatch):
    monkeypatch.setattr(
        "sa_agent.retrieval.qa.search_documents",
        lambda vector_store, query, k: [],
    )

    result = answer_question(
        vector_store=object(),
        question="一个有效的问题",
    )

    assert result["answer"] == "未检索到相关资料，暂时无法回答。"
    assert result["sources"] == []
