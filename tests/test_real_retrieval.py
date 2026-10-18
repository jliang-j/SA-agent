
from langchain_core.documents import Document

from sa_agent.retrieval.store import create_vector_store, search_documents


def main():
    documents = [
        Document(
            page_content="人工智能产业正在推动制造业数字化转型。",
            metadata={"source": "industry.txt", "chunk_index": 0},
        ),
        Document(
            page_content="企业应关注数据安全、隐私保护和人工智能模型风险。",
            metadata={"source": "risk.txt", "chunk_index": 0},
        ),
        Document(
            page_content="新能源汽车市场持续发展，产业链竞争日趋激烈。",
            metadata={"source": "auto.txt", "chunk_index": 0},
        ),
    ]

    print("正在创建向量库，调用真实 Embedding API……")
    vector_store = create_vector_store(documents)

    query = "人工智能行业有哪些主要风险？"
    print(f"\n检索问题：{query}")

    results = search_documents(vector_store, query, k=2)

    print("\n检索结果：")
    for index, document in enumerate(results, start=1):
        print(f"\n结果 {index}")
        print(f"内容：{document.page_content}")
        print(f"来源：{document.metadata.get('source')}")
        print(f"片段序号：{document.metadata.get('chunk_index')}")


if __name__ == "__main__":
    main()
