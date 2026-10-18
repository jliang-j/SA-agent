
from sa_agent.embeddings import create_embeddings
from sa_agent.retrieval.qa import answer_question
from sa_agent.retrieval.store import (
    create_persistent_vector_store,
    search_documents,
)


def main():
    vector_store = create_persistent_vector_store(
        persist_directory="data/chroma",
        collection_name="industry_documents",
        embeddings=create_embeddings(),
    )

    questions = [
        "企业应该如何处理重复数据和缺失字段？",
        "DAG 工作流中的任务依赖关系是什么？",
        "现有资料提供的 2025 年人工智能行业市场规模是多少？",
    ]

    for question in questions:
        print("\n" + "=" * 60)
        print(f"问题：{question}")

        documents = search_documents(vector_store, question, k=3)

        print("\n检索结果：")
        for document in documents:
            print(
                f"- 来源：{document.metadata.get('source')}, "
                f"片段：{document.metadata.get('chunk_index')}"
            )
            print(document.page_content[:200])

        response = answer_question(vector_store, question, k=3)

        print("\n模型回答：")
        print(response["answer"])

        print("\n回答来源：")
        for source in response["sources"]:
            print(
                f"- {source['source']} "
                f"(chunk_index={source['chunk_index']})"
            )


if __name__ == "__main__":
    main()
