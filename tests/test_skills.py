
from langchain_core.documents import Document
import pytest

from sa_agent.skills import retrieve


def test_retrieval_success(monkeypatch):
    def fake_search_documents(vector_store, query, k):
        assert query == "如何处理重复数据？"
        assert k == 3
        return [
            Document(
                page_content="重复数据应根据业务标识识别。",
                metadata={
                    "source": "ai_industry_governance.txt",
                    "chunk_index": 0,
                },
            )
        ]

    monkeypatch.setattr(
        retrieve,
        "search_documents",
        fake_search_documents,
    )

    result = retrieve.run_retrieval(
        vector_store=object(),
        question="如何处理重复数据？",
    )

    assert result["question"] == "如何处理重复数据？"
    assert len(result["results"]) == 1
    assert result["results"][0]["source"] == "ai_industry_governance.txt"
    assert "重复数据" in result["results"][0]["content"]


@pytest.mark.parametrize(
    ("question", "k"),
    [
        ("   ", 3),
        ("如何处理重复数据？", 0),
        ("如何处理重复数据？", 11),
    ],
)
def test_retrieval_rejects_invalid_input(question, k):
    with pytest.raises(ValueError):
        retrieve.run_retrieval(
            vector_store=object(),
            question=question,
            k=k,
        )


def test_retrieval_empty_results(monkeypatch):
    monkeypatch.setattr(
        retrieve,
        "search_documents",
        lambda **kwargs: [],
    )

    result = retrieve.run_retrieval(
        vector_store=object(),
        question="资料中没有的问题",
    )

    assert result["results"] == []
