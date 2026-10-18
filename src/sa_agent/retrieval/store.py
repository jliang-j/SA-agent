from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from sa_agent.embeddings import create_embeddings

def create_vector_store(
        documents: list[Document],
) -> InMemoryVectorStore:
    """在内存之中创建向量存储文档并添加文档"""
    embeddings = create_embeddings
    vector_store = InMemoryVectorStore(embedding=embeddings)

    if documents:
        vector_store.add_documents(documents)
    return vector_store

def create_persistent_vector_store(
        documents: list[Document] | None = None,
        persist_directory: str | Path = "data/chroma",
        collection_name: str = "industry_documents",

) -> Chroma:
    embeddings = create_embeddings()
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )

    if documents:
        vector_store.add_documents(documents)

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
