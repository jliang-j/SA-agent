import json
from pathlib import Path

import pytest

from sa_agent.embeddings import create_embeddings
from sa_agent.retrieval.store import create_persistent_vector_store
from sa_agent.workflows.agent import run_agent


@pytest.mark.integration
def test_real_agent(tmp_path):
    """验证真实模型组织任务，经 DAG 收集证据后回答，并保存结果供人工复核。"""
    store_dir = Path("data/chroma")
    if not store_dir.exists():
        pytest.skip("本地向量库不存在，请先导入示例文档")
    store = create_persistent_vector_store(
        persist_directory=store_dir, collection_name="industry_documents",
        embeddings=create_embeddings(),
    )
    if store._collection.count() == 0:
        pytest.skip("行业资料集合为空，请先导入示例文档")

    question = "人工智能应用有哪些主要风险？企业可以采取哪些治理措施？"
    result = run_agent(store, question, k=3)
    output = tmp_path / "agent_result.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nResult file:", output)
    print("Trace:", json.dumps(result["trace"], ensure_ascii=True))

    assert result["question"] == question
    assert result["answer"]
    assert result["report"]["findings"]
    evidence_ids = {item["id"] for item in result["evidence"]}
    assert evidence_ids
    assert all(ref in evidence_ids for finding in result["report"]["findings"]
               for ref in finding["evidence_ids"])
    calls = [entry for entry in result["trace"] if entry["step"] == "research"]
    assert 1 <= len(calls) <= 2
    assert 1 <= len(calls[0]["plan"]["tasks"]) <= 3
    if len(calls) == 2:
        assert len(calls[1]["plan"]["tasks"]) <= 2
    tasks = result["tasks"]
    assert len({task["id"] for task in tasks}) == len(tasks)
    assert all(task["status"] == "success" for task in tasks), result["trace"]

    passed = 0
    for task in tasks:
        assert set(task["nodes"].values()) == {"success"}
        facts = task["verification"]["facts"]
        assert task["verification"]["passed_count"] == sum(f["status"] == "passed" for f in facts)
        assert task["verification"]["failed_count"] == sum(f["status"] == "failed" for f in facts)
        for fact in facts:
            if fact["status"] == "passed":
                passed += 1
                assert any(
                    item["source"] == fact["source"]
                    and item["chunk_index"] == fact["chunk_index"]
                    and fact["evidence"] in item["content"]
                    for item in task["retrieval"]["results"]
                )
    assert passed > 0
    print("Tasks:", len(tasks), "Passed facts:", passed)
