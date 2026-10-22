
from sa_agent.retrieval.store import search_documents


def run_retrieval(vector_store, question: str, k: int = 3) -> dict:
    """检索与问题相关的资料，并返回结构化结果。"""
    if not question.strip():
        raise ValueError("问题不能为空")

    if not 1 <= k <= 10:
        raise ValueError("k 必须在 1 到 10 之间")

    docs = search_documents(
        vector_store=vector_store,
        query=question,
        k=k,
    )

    results = [
        {
            "source": doc.metadata.get("source", "未知"),
            "chunk_index": doc.metadata.get("chunk_index"),
            "content": doc.page_content,
        }
        for doc in docs
    ]
    for item, doc in zip(results, docs):
        if "page" in doc.metadata:
            item["page"] = doc.metadata["page"]

    return {
        "question": question,
        "results": results,
    }
