import json
from types import SimpleNamespace

import pytest

from sa_agent.skills import plan


def fake_model(monkeypatch, content):
    class FakeModel:
        def invoke(self, prompt):
            assert "人工智能" in prompt
            return SimpleNamespace(content=content)

    monkeypatch.setattr(plan, "create_chat_model", lambda: FakeModel())


@pytest.mark.parametrize("count", [1, 2, 3])
def test_plan_ok(monkeypatch, count):
    tasks = [
        {"id": f"t{i}", "goal": f"研究方面{i}", "query": f"查询{i}"}
        for i in range(1, count + 1)
    ]
    fake_model(monkeypatch, json.dumps({"tasks": tasks}, ensure_ascii=False))

    assert plan.run_plan("人工智能的风险与治理？") == {"tasks": tasks}


@pytest.mark.parametrize("question", ["", "   ", None, 123])
def test_bad_question(monkeypatch, question):
    def unexpected():
        pytest.fail("无效问题不应调用模型")

    monkeypatch.setattr(plan, "create_chat_model", unexpected)
    with pytest.raises(ValueError, match="问题"):
        plan.run_plan(question)


@pytest.mark.parametrize("output", [
    {"tasks": []},
    {"tasks": [{"id": f"t{i}", "goal": "目标", "query": "查询"} for i in range(1, 5)]},
    {"tasks": [{"id": "t1", "goal": "目标", "query": "查询"}] * 2},
    {"tasks": [{"id": "t1", "goal": "  ", "query": "查询"}]},
    {"tasks": [{"id": "t1", "goal": "目标", "query": "  "}]},
    {"tasks": [{"id": "t1", "goal": "目标"}]},
    {"tasks": [{"id": "bad", "goal": "目标", "query": "查询"}]},
    {"tasks": [{"id": "t1", "goal": "目标", "query": "查询", "tool": "web"}]},
    {"tasks": [{"id": "t1", "goal": "目标", "query": "查询"}], "action": "search"},
])
def test_bad_plan(monkeypatch, output):
    fake_model(monkeypatch, json.dumps(output, ensure_ascii=False))
    with pytest.raises(ValueError, match="计划格式"):
        plan.run_plan("人工智能的风险？")


@pytest.mark.parametrize("content", ["不是 JSON", "```json\n{}\n```", None, []])
def test_bad_json(monkeypatch, content):
    fake_model(monkeypatch, content)
    with pytest.raises(ValueError, match="计划格式"):
        plan.run_plan("人工智能的风险？")


def test_model_error(monkeypatch):
    error = RuntimeError("模型服务不可用")

    class FakeModel:
        def invoke(self, prompt):
            raise error

    monkeypatch.setattr(plan, "create_chat_model", lambda: FakeModel())
    with pytest.raises(RuntimeError) as caught:
        plan.run_plan("人工智能的风险？")
    assert caught.value is error
