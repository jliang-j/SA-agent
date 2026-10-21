
from pathlib import Path

import pytest

from sa_agent.embeddings import create_embeddings
from sa_agent.retrieval.store import create_persistent_vector_store
from sa_agent.workflows.research import run_research


@pytest.mark.integration
def test_real_research():
    store_dir = Path("data/chroma")
    if not store_dir.exists():
        pytest.skip("本地向量库不存在，请先导入行业文档")

    store = create_persistent_vector_store(
        persist_directory=store_dir,
        collection_name="industry_documents",
        embeddings=create_embeddings(),
    )

    question = "AI行业有哪些主要风险？如何进行治理？"
    result = run_research(
        vector_store=store,
        question=question,
        k=5,
    )

    assert result["question"] == question

    retrieved = result["retrieval"]["results"]
    facts = result["extraction"]["facts"]
    verification = result["verification"]
    verified_facts = verification["facts"]

    # 确认检索和事实提取确实产生了结果。
    assert retrieved
    assert facts

    # 每条事实都应有明确的核验状态。
    assert all(
        fact["status"] in {"passed", "failed"}
        for fact in verified_facts
    )

    # 核验统计必须与明细一致。
    passed = sum(
        fact["status"] == "passed"
        for fact in verified_facts
    )
    failed = sum(
        fact["status"] == "failed"
        for fact in verified_facts
    )

    assert verification["passed_count"] == passed
    assert verification["failed_count"] == failed
    assert passed + failed == len(verified_facts)
