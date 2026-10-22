import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import sa_agent.main as main
from sa_agent.embeddings import create_embeddings
from sa_agent.retrieval.store import create_persistent_vector_store


@pytest.mark.integration
@pytest.mark.parametrize("case,question", [
    ("supported", "人工智能应用有哪些主要风险？企业可以采取哪些治理措施？"),
    ("missing", "根据本地资料，2025年人工智能行业市场规模是多少？请给出具体数值和统计口径。"),
])
def test_real_agent_api(tmp_path, monkeypatch, case, question):
    """通过真实 API 处理函数验收正常研究及资料不足，模型和检索均不模拟。"""
    if not Path("data/chroma").exists():
        pytest.skip("本地向量库不存在")
    store = create_persistent_vector_store(
        persist_directory="data/chroma", collection_name="industry_documents",
        embeddings=create_embeddings(),
    )
    stored = store._collection.get(include=["metadatas"])
    if not stored["ids"]:
        pytest.skip("行业资料集合为空")
    sources = {item.get("source") for item in stored["metadatas"]}
    if case == "missing" and not sources <= {"ai_industry_risks.txt", "ai_industry_governance.txt"}:
        pytest.skip("资料库已扩充，需重新确认市场规模问题是否仍缺少依据")

    # 复用上面打开的真实向量库，只避免重复初始化，不替换检索或模型行为。
    monkeypatch.setattr(main, "create_persistent_vector_store", lambda **kwargs: store)
    with TestClient(main.app) as client:
        response = client.post("/agent/research", json={"question": question, "k": 3})
        output = tmp_path / f"agent_api_{case}.json"
        output.write_text(json.dumps(response.json(), ensure_ascii=False, indent=2), encoding="utf-8")
        print("\nResult file:", output)
        assert response.status_code == 200, response.json()
        assert client.get("/health").status_code == 200

    result = response.json()
    assert result["question"] == question
    calls = [item for item in result["trace"] if item["step"] == "research"]
    assert 1 <= len(calls) <= 2
    assert result["tasks"]
    assert all(task["status"] == "success" for task in result["tasks"])
    refs = {item["id"] for item in result["evidence"]}
    assert all(ref in refs for finding in result["report"]["findings"]
               for ref in finding["evidence_ids"])
    if case == "supported":
        assert result["report"]["findings"]
        assert result["evidence"]
    else:
        assert result["report"]["gaps"], "缺少资料时必须说明缺口"
        assert result["report"]["findings"] == [], "不得将不相关事实包装成市场规模结论"
    print("Case:", case, "Calls:", len(calls), "Findings:", len(result["report"]["findings"]))
