from threading import Barrier

import pytest

from sa_agent.workflows import execute


def make_plan(count=2):
    return {"tasks": [
        {"id": f"t{i}", "goal": f"目标{i}", "query": f"查询{i}"}
        for i in range(1, count + 1)
    ]}


def mock_skills(monkeypatch, retrieve):
    monkeypatch.setattr(execute, "run_retrieval", retrieve)
    monkeypatch.setattr(execute, "run_extract", lambda data: {
        "question": data["question"], "facts": data["results"],
    })

    def verify(extracted, retrieved):
        assert extracted["facts"] == retrieved["results"]
        return {"facts": extracted["facts"]}

    monkeypatch.setattr(execute, "run_verify", verify)


def test_task_outputs(monkeypatch):
    store = object()

    def retrieve(vector_store, question, k):
        assert vector_store is store
        assert k == 4
        return {"question": question, "results": [question]}

    mock_skills(monkeypatch, retrieve)
    result = execute.run_tasks(store, make_plan(3), k=4)

    for i, task in enumerate(result["tasks"], start=1):
        assert task["id"] == f"t{i}"
        assert task["status"] == "success"
        assert task["error"] is None
        assert task["retrieval"]["question"] == f"查询{i}"
        assert task["extraction"]["question"] == f"目标{i}"
        assert task["verification"]["facts"] == [f"查询{i}"]
        assert set(task["nodes"].values()) == {"success"}


def test_parallel_tasks(monkeypatch):
    barrier = Barrier(2)

    def retrieve(vector_store, question, k):
        barrier.wait(timeout=5)
        return {"question": question, "results": []}

    mock_skills(monkeypatch, retrieve)
    result = execute.run_tasks(object(), make_plan())

    assert all(task["status"] == "success" for task in result["tasks"])


@pytest.mark.parametrize("stage", ["retrieve", "extract", "verify"])
def test_branch_failure(monkeypatch, stage):
    def retrieve(vector_store, question, k):
        if stage == "retrieve" and question == "查询1":
            raise RuntimeError("任务1失败")
        return {"question": question, "results": [question]}

    mock_skills(monkeypatch, retrieve)
    original = getattr(execute, f"run_{stage}" if stage != "retrieve" else "run_retrieval")
    if stage == "extract":
        def extract(data):
            if data["question"] == "目标1":
                raise RuntimeError("任务1失败")
            return original(data)
        monkeypatch.setattr(execute, "run_extract", extract)
    elif stage == "verify":
        def verify(extracted, retrieved):
            if extracted["question"] == "目标1":
                raise RuntimeError("任务1失败")
            return original(extracted, retrieved)
        monkeypatch.setattr(execute, "run_verify", verify)

    first, second = execute.run_tasks(object(), make_plan())["tasks"]

    assert first["status"] == "failed"
    assert first["error"] == {"type": "RuntimeError", "message": "任务1失败"}
    stages = ["retrieve", "extract", "verify"]
    index = stages.index(stage)
    assert list(first["nodes"].values()) == ["success"] * index + ["failed"] + ["skipped"] * (2 - index)
    assert second["status"] == "success"
    assert second["verification"]["facts"] == ["查询2"]


def test_invalid_input(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("非法输入不应执行检索")

    monkeypatch.setattr(execute, "run_retrieval", unexpected)
    with pytest.raises(ValueError):
        execute.run_tasks(object(), {"tasks": []})
    with pytest.raises(ValueError, match="k"):
        execute.run_tasks(object(), make_plan(), k=0)
