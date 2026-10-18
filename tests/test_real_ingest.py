
from langchain_core.documents import Document

from sa_agent.embeddings import create_embeddings
from sa_agent.retrieval.ingest import ingest_text_file
from sa_agent.retrieval.qa import answer_question
from sa_agent.retrieval.store import create_persistent_vector_store


def test_real_document_ingest_and_qa(tmp_path):
    # 准备一份真实待导入的测试文档。
    document_path = tmp_path / "industry_risks.txt"
    document_path.write_text(
        "人工智能行业风险分析：\n"
        "企业在部署人工智能系统时，主要面临数据安全、"
        "隐私保护和模型输出不准确等风险。\n"
        "企业应建立数据访问权限管理机制，定期开展安全审计，"
        "并对重要模型输出进行人工复核。\n"
        "数据治理需要明确数据来源、数据质量标准和责任人，"
        "确保数据可追溯、可管理。",
        encoding="utf-8",
    )

    persist_directory = tmp_path / "chroma"
    collection_name = "real_ingest_qa_test"

    # 真实调用 Embedding API，将文档写入 Chroma。
    result = ingest_text_file(
        file_path=document_path,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    assert result["chunks_count"] > 0
    print("导入结果：", result)

    # 重新打开持久化向量库，验证数据确实已保存。
    vector_store = create_persistent_vector_store(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embeddings=create_embeddings(),
    )

    assert vector_store._collection.count() == result["chunks_count"]

    # 使用真实 Embedding 检索，并调用真实聊天模型回答问题。
    response = answer_question(
        vector_store,
        "企业部署人工智能系统时面临哪些风险？应采取什么措施？",
    )

    print("\n回答：")
    print(response["answer"])
    print("\n来源：")
    for source in response["sources"]:
        print(source)

    assert response["answer"].strip()
    assert response["sources"]
    assert any(
        source["source"] == "industry_risks.txt"
        for source in response["sources"]
    )
