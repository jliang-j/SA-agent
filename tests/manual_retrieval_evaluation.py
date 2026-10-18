
from sa_agent.embeddings import create_embeddings
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

    cases = [
        {
            "question": "企业应该如何处理重复数据和缺失字段？",
            "expected_source": "ai_industry_governance.txt",
            "expected_keyword": "重复记录",
        },
        {
            "question": "DAG 工作流中的任务依赖关系是什么？",
            "expected_source": "ai_industry_governance.txt",
            "expected_keyword": "前置任务",
        },
        {
            "question": "现有资料提供的 2025 年人工智能行业市场规模是多少？",
            "expected_source": "ai_industry_governance.txt",
            "expected_keyword": "不提供经外部核验的行业统计数据",
        },
    ]

    passed = 0

    for case in cases:
        documents = search_documents(
            vector_store,
            query=case["question"],
            k=3,
        )

        # 检查正确来源是否出现在前三条结果中。
        source_found = any(
            doc.metadata.get("source") == case["expected_source"]
            for doc in documents
        )

        # 检查预期的关键内容是否出现在检索结果中。
        keyword_found = any(
            case["expected_keyword"] in doc.page_content
            for doc in documents
        )

        success = source_found and keyword_found

        print(f"\n问题：{case['question']}")
        print(f"预期来源：{case['expected_source']}")
        print(f"实际来源：{[doc.metadata.get('source') for doc in documents]}")
        print(f"关键内容命中：{keyword_found}")
        print(f"结果：{'通过' if success else '失败'}")

        if success:
            passed += 1

    print(f"\n评测结果：{passed}/{len(cases)} 通过")

    if passed != len(cases):
        raise AssertionError("检索评测未全部通过，请检查检索结果。")


if __name__ == "__main__":
    main()
