
from langchain_core.documents import Document

from sa_agent.retrieval.qa import answer_question
from sa_agent.retrieval.store import create_vector_store


def main():
    documents = [
        Document(
            page_content=(
                "人工智能行业面临数据安全、个人隐私保护、"
                "模型输出不准确以及训练数据质量等风险。"
            ),
            metadata={"source": "ai_risks.txt", "chunk_index": 0},
        ),
        Document(
            page_content=(
                "人工智能技术能够提升部分业务流程的自动化水平，"
                "但企业需要评估部署成本和实际应用效果。"
            ),
            metadata={"source": "ai_applications.txt", "chunk_index": 0},
        ),
        Document(
            page_content=(
                "新能源汽车产业涉及电池制造、整车生产和充电基础设施。"
            ),
            metadata={"source": "ev_industry.txt", "chunk_index": 0},
        ),
    ]

    print("正在创建向量库，将调用真实 Embedding API……")
    vector_store = create_vector_store(documents)

    question = "人工智能行业主要面临哪些风险？"

    print("\n正在调用 RAG 问答链路……")
    result = answer_question(
        vector_store=vector_store,
        question=question,
        k=2,
    )

    print("\n问题：", question)
    print("\n回答：")
    print(result["answer"])

    print("\n参考来源：")
    for source in result["sources"]:
        print(
            f"- {source['source']} "
            f"(片段 {source['chunk_index']})"
        )


if __name__ == "__main__":
    main()
