
from pathlib import Path
from langchain_core.documents import Document

from sa_agent.retrieval.qa import answer_question
from sa_agent.retrieval.store import create_persistent_vector_store

# 项目根目录下的持久化目录
PERSIST_DIRECTORY = Path("data/chroma")
COLLECTION_NAME = "real_qa_test"


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
                "人工智能可以提高业务自动化水平，"
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

    print("第一阶段：写入 Chroma，调用真实 Embedding API……")

    # 每次运行都使用独立的测试集合，避免旧数据干扰
    vector_store = create_persistent_vector_store(
        documents=documents,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
    )

    print("文档写入完成。")

    # 模拟程序重启：重新创建 Chroma 实例，不再传入文档
    print("\n第二阶段：重新打开持久化向量库……")
    reopened_store = create_persistent_vector_store(
        documents=None,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
    )

    question = "人工智能行业主要面临哪些风险？"

    print(f"\n第三阶段：检索问题：{question}")
    results = reopened_store.similarity_search(question, k=2)

    if not results:
        raise RuntimeError("持久化向量库没有检索到任何文档。")

    print("\n检索结果：")
    for index, document in enumerate(results, start=1):
        print(f"\n结果 {index}")
        print(f"内容：{document.page_content}")
        print(f"来源：{document.metadata.get('source')}")

    print("\n第四阶段：调用 qwen3.8-max 生成回答……")
    result = answer_question(
        vector_store=reopened_store,
        question=question,
        k=2,
    )

    print("\n回答：")
    print(result["answer"])

    print("\n参考来源：")
    for source in result["sources"]:
        print(
            f"- {source['source']} "
            f"(片段 {source['chunk_index']})"
        )
    print(f"\nChroma 数据目录：{PERSIST_DIRECTORY.resolve()}")
    print("\n测试流程执行完成。")

if __name__ == "__main__":
    main()
