
import pytest

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from sa_agent.retrieval.store import (
    create_persistent_vector_store,
    search_documents,
)


class FakeEmbeddings(Embeddings):
    """测试专用的模拟 Embedding，不调用真实 API。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        text = text.lower()
        return [
            float("风险" in text),
            float("数据" in text),
            float("增长" in text),
        ]


def test_search_rejects_empty_query():
    with pytest.raises(ValueError, match="检索问题不能为空"):
        search_documents(None, "   ")


def test_search_rejects_non_positive_k():
    with pytest.raises(ValueError, match="k 必须大于 0"):
        search_documents(None, "行业风险", k=0)


def test_reimport_updates_existing_document(tmp_path):
    embeddings = FakeEmbeddings()
    persist_directory = tmp_path / "chroma"

    original = Document(
        page_content="企业需要关注数据风险。",
        metadata={"source": "report.txt", "chunk_index": 0},
    )

    store = create_persistent_vector_store(
        documents=[original],
        persist_directory=persist_directory,
        collection_name="dedup_test",
        embeddings=embeddings,
    )

    updated = Document(
        page_content="企业需要关注数据安全和隐私风险。",
        metadata={"source": "report.txt", "chunk_index": 0},
    )

    # 再次导入相同文档身份，但正文已经更新。
    store.add_documents(
        documents=[updated],
        ids=[
            __import__(
                "sa_agent.retrieval.store",
                fromlist=["_generate_document_id"],
            )._generate_document_id(updated)
        ],
    )

    records = store.get(where={"source": "report.txt"})

    assert len(records["ids"]) == 1
    assert records["documents"][0] == updated.page_content

def test_reimport_persists_without_duplicate_records(tmp_path):
    embeddings = FakeEmbeddings()
    persist_directory = tmp_path / "chroma"
    collection_name = "persistent_dedup_test"

    document = Document(
        page_content="企业需要关注数据风险。",
        metadata={"source": "report.txt", "chunk_index": 0},
    )

    # 第一次导入
    store = create_persistent_vector_store(
        documents=[document],
        persist_directory=persist_directory,
        collection_name=collection_name,
        embeddings=embeddings,
    )

    # 使用相同来源和片段序号，更新正文并再次导入
    updated_document = Document(
        page_content="企业需要关注数据安全和隐私风险。",
        metadata={"source": "report.txt", "chunk_index": 0},
    )

    from sa_agent.retrieval.store import _generate_document_id

    store.add_documents(
        documents=[updated_document],
        ids=[_generate_document_id(updated_document)],
    )

    # 重新打开数据库，不传入文档
    reopened_store = create_persistent_vector_store(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embeddings=embeddings,
    )

    records = reopened_store.get(
        where={"source": "report.txt"}
    )

    assert len(records["ids"]) == 1
    assert records["documents"][0] == updated_document.page_content
    assert records["metadatas"][0]["chunk_index"] == 0
