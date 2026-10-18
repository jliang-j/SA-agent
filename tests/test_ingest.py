
from langchain_core.embeddings import Embeddings

from sa_agent.retrieval.ingest import ingest_text_file
from sa_agent.retrieval.store import create_persistent_vector_store


class FakeEmbeddings(Embeddings):
    """用于测试的模拟向量模型，不调用外部 API。"""

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


def test_ingest_text_file(tmp_path):
    """验证文件可以正常导入向量库。"""
    file_path = tmp_path / "industry.txt"
    file_path.write_text("企业需要加强数据安全风险管理。", encoding="utf-8")

    embeddings = FakeEmbeddings()
    persist_directory = tmp_path / "chroma"

    result = ingest_text_file(
        file_path=file_path,
        persist_directory=persist_directory,
        collection_name="ingest_test",
        embeddings=embeddings,
    )

    vector_store = create_persistent_vector_store(
        persist_directory=persist_directory,
        collection_name="ingest_test",
        embeddings=embeddings,
    )

    assert result["chunks_count"] == 1
    assert vector_store._collection.count() == 1

    stored = vector_store.get(where={"source_path": str(file_path.resolve())})
    assert stored["metadatas"][0]["source"] == "industry.txt"
    assert stored["metadatas"][0]["chunk_index"] == 0


def test_repeated_ingest_does_not_duplicate_documents(tmp_path):
    """验证重复导入同一文件不会累积重复记录。"""
    file_path = tmp_path / "industry.txt"
    file_path.write_text("企业需要加强数据安全风险管理。", encoding="utf-8")

    embeddings = FakeEmbeddings()
    persist_directory = tmp_path / "chroma"

    for _ in range(2):
        ingest_text_file(
            file_path=file_path,
            persist_directory=persist_directory,
            collection_name="repeat_test",
            embeddings=embeddings,
        )

    vector_store = create_persistent_vector_store(
        persist_directory=persist_directory,
        collection_name="repeat_test",
        embeddings=embeddings,
    )

    assert vector_store._collection.count() == 1


def test_reingest_removes_stale_chunks(tmp_path):
    """验证文件变短后，之前多余的片段会被清理。"""
    file_path = tmp_path / "industry.txt"
    persist_directory = tmp_path / "chroma"
    embeddings = FakeEmbeddings()

    file_path.write_text("企业面临数据风险。 " * 100, encoding="utf-8")

    first_result = ingest_text_file(
        file_path=file_path,
        persist_directory=persist_directory,
        collection_name="update_test",
        embeddings=embeddings,
    )

    assert first_result["chunks_count"] > 1

    file_path.write_text("企业需要加强数据安全管理。", encoding="utf-8")

    second_result = ingest_text_file(
        file_path=file_path,
        persist_directory=persist_directory,
        collection_name="update_test",
        embeddings=embeddings,
    )

    vector_store = create_persistent_vector_store(
        persist_directory=persist_directory,
        collection_name="update_test",
        embeddings=embeddings,
    )

    assert second_result["chunks_count"] == 1
    assert vector_store._collection.count() == 1


def test_ingest_missing_file_raises_error(tmp_path):
    """验证文件不存在时会明确报错。"""
    import pytest

    with pytest.raises(FileNotFoundError):
        ingest_text_file(
            file_path=tmp_path / "missing.txt",
            persist_directory=tmp_path / "chroma",
            collection_name="missing_test",
            embeddings=FakeEmbeddings(),
        )


def test_ingest_empty_file_clears_old_chunks(tmp_path):
    """验证空文件重新导入时，旧片段也会被清理。"""
    file_path = tmp_path / "industry.txt"
    persist_directory = tmp_path / "chroma"
    embeddings = FakeEmbeddings()

    file_path.write_text("企业面临数据风险。 " * 100, encoding="utf-8")

    ingest_text_file(
        file_path=file_path,
        persist_directory=persist_directory,
        collection_name="empty_test",
        embeddings=embeddings,
    )

    file_path.write_text("", encoding="utf-8")

    result = ingest_text_file(
        file_path=file_path,
        persist_directory=persist_directory,
        collection_name="empty_test",
        embeddings=embeddings,
    )

    vector_store = create_persistent_vector_store(
        persist_directory=persist_directory,
        collection_name="empty_test",
        embeddings=embeddings,
    )

    assert result["chunks_count"] == 0
    assert vector_store._collection.count() == 0
