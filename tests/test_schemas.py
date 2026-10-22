import pytest
from pydantic import ValidationError

from sa_agent.schemas import Plan


@pytest.mark.parametrize("count", [1, 2, 3])
def test_plan_ok(count):
    """现有工具接受一到三个格式正确的独立任务。"""
    data = {"tasks": [
        {"id": f"t{i}", "goal": f"目标{i}", "query": f"查询{i}"}
        for i in range(1, count + 1)
    ]}
    assert Plan.model_validate(data).model_dump() == data


@pytest.mark.parametrize("data", [
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
def test_bad_plan(data):
    """保留任务数量、唯一编号、非空字段和额外字段的校验覆盖。"""
    with pytest.raises(ValidationError):
        Plan.model_validate(data)


def test_strip_fields():
    """清理字段首尾空白后再执行结构校验。"""
    result = Plan.model_validate({"tasks": [{"id": " t1 ", "goal": " 目标 ", "query": " 查询 "}]})
    assert result.tasks[0].model_dump() == {"id": "t1", "goal": "目标", "query": "查询"}
