import pytest

from sa_agent.retrieval.store import search_documents

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from sa_agent.retrieval.store import search_documents

def test_search_rejects_empty_query():
    with pytest.raises(ValueError, match="检索问题不能为空"):
        search_documents(None, "   ")


def test_search_rejects_non_positive_k():
    with pytest.raises(ValueError, match="k 必须大于 0"):
        search_documents(None, "行业风险", k=0)





class FakeEmbeddings(Embeddings):
    """用于测试的模拟 Embedding，不调用真实模型 API。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        text = text.lower()

        # 用简单的关键词特征模拟向量，便于验证检索流程。
        return [
            float("行业" in text),
            float("风险" in text),
            float("增长" in text),
        ]


def test_search_returns_documents_with_metadata():
    embeddings = FakeEmbeddings()
    vector_store = InMemoryVectorStore(embedding=embeddings)

    documents = [
        Document(
            page_content="行业面临多种风险。",
            metadata={"source": "report.txt", "chunk_index": 0},
        ),
        Document(
            page_content="行业增长情况良好。",
            metadata={"source": "growth.txt", "chunk_index": 1},
        ),
    ]
    vector_store.add_documents(documents)

    results = search_documents(
        vector_store,
        query="行业风险",
        k=1,
    )

    assert len(results) == 1
    assert isinstance(results[0], Document)
    assert results[0].metadata["source"] == "report.txt"
    assert results[0].metadata["chunk_index"] == 0


def test_search_respects_k():
    embeddings = FakeEmbeddings()
    vector_store = InMemoryVectorStore(embedding=embeddings)

    documents = [
        Document(page_content="行业风险分析"),
        Document(page_content="行业增长分析"),
        Document(page_content="行业发展分析"),
    ]
    vector_store.add_documents(documents)

    results = search_documents(
        vector_store,
        query="行业",
        k=2,
    )

    assert len(results) == 2
