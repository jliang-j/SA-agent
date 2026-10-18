
from pathlib import Path

from langchain_core.embeddings import Embeddings

from sa_agent.retrieval.loader import load_and_split_document
from sa_agent.retrieval.store import create_persistent_vector_store


def ingest_text_file(
    file_path: str | Path,
    persist_directory: str | Path = "data/chroma",
    collection_name: str = "industry_documents",
    embeddings: Embeddings | None = None,
) -> dict:
    """读取文本文件并将切片写入持久化向量库。"""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"文档不存在：{path}")

    documents = load_and_split_document(path)
    source_path = str(path.resolve())

    for document in documents:
        document.metadata["source_path"] = source_path

    vector_store = create_persistent_vector_store(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embeddings=embeddings,
    )

    # 删除该文件上一次导入的片段，防止旧片段残留。
    vector_store.delete(where={"source_path": source_path})

    if documents:
        from sa_agent.retrieval.store import _generate_document_id

        ids = [_generate_document_id(document) for document in documents]
        vector_store.add_documents(documents=documents, ids=ids)

    return {
        "source": path.name,
        "chunks_count": len(documents),
        "persist_directory": str(persist_directory),
    }
