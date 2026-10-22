
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
    """兼容原有文本导入，保留空文件可清理旧片段的行为。"""
    return ingest_document_file(file_path, persist_directory, collection_name, embeddings, require_text=False)


def ingest_document_file(
    file_path: str | Path,
    persist_directory: str | Path = "data/chroma",
    collection_name: str = "industry_documents",
    embeddings: Embeddings | None = None,
    *, require_text: bool = True,
) -> dict:
    """解析并写入文档，先写新片段再清理旧片段；不提供事务级更新。"""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"文档不存在：{path}")

    documents = load_and_split_document(path)
    if require_text and not documents:
        raise ValueError("未提取到有效正文；文本请使用UTF-8，扫描PDF需要OCR后再导入")
    source_path = str(path.resolve())

    for document in documents:
        document.metadata["source_path"] = source_path

    vector_store = create_persistent_vector_store(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embeddings=embeddings,
    )

    if documents:
        from sa_agent.retrieval.store import _generate_document_id

        ids = [_generate_document_id(document) for document in documents]
        old = vector_store.get(where={"source_path": source_path})
        vector_store.add_documents(documents=documents, ids=ids)
        stale = list(set(old["ids"]) - set(ids))
        if stale:
            vector_store.delete(ids=stale)
    else:
        vector_store.delete(where={"source_path": source_path})

    return {
        "source": path.name,
        "chunks_count": len(documents),
        "persist_directory": str(persist_directory),
    }
