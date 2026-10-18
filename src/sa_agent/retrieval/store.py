import hashlib
import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from sa_agent.embeddings import create_embeddings

def create_vector_store(
    documents: list[Document],
) -> InMemoryVectorStore:
    """在内存中创建向量存储并添加文档。"""
    embeddings = create_embeddings()
    vector_store = InMemoryVectorStore(embedding=embeddings)

    if documents:
        vector_store.add_documents(documents)

    return vector_store


def _generate_document_id(document: Document) -> str:
    """根据文档身份生成稳定的唯一 ID。"""
    source = document.metadata.get("source")
    chunk_index = document.metadata.get("chunk_index")

    if source is not None and chunk_index is not None:
        # 同一来源的同一片段始终使用相同 ID。
        identity = {
            "source": str(source),
            "chunk_index": chunk_index,
        }
    else:
        # 缺少完整身份信息时，使用正文和元数据生成 ID。
        identity = {
            "source": source,
            "metadata": document.metadata,
            "page_content": document.page_content,
        }

    serialized = json.dumps(
        identity,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def create_persistent_vector_store(
    documents: list[Document] | None = None,
    persist_directory: str | Path = "data/chroma",
    collection_name: str = "industry_documents",
    embeddings: Embeddings | None = None,
) -> Chroma:
    """创建或加载 Chroma，并以稳定 ID 写入文档。"""
    if embeddings is None:
        embeddings = create_embeddings()

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )

    if documents:
        ids = [_generate_document_id(doc) for doc in documents]
        vector_store.add_documents(documents=documents, ids=ids)

    return vector_store


def search_documents(
    vector_store: InMemoryVectorStore | Chroma,
    query: str,
    k: int = 3,
) -> list[Document]:
    """根据问题检索最相关的文档片段。"""
    if not query.strip():
        raise ValueError("检索问题不能为空")
    if k <= 0:
        raise ValueError("k 必须大于 0")

    return vector_store.similarity_search(query, k=k)