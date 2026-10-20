
import pytest

from sa_agent.workflows.dag import DAG, Node


def test_topological_sort_respects_dependencies():
    dag = DAG([
        Node("report", lambda: None, ["extract", "verify"]),
        Node("extract", lambda: None, ["retrieve"]),
        Node("verify", lambda: None, ["retrieve"]),
        Node("retrieve", lambda: None),
    ])

    order = dag.topological_sort()
    positions = {name: i for i, name in enumerate(order)}

    assert set(order) == {"retrieve", "extract", "verify", "report"}
    assert positions["retrieve"] < positions["extract"]
    assert positions["retrieve"] < positions["verify"]
    assert positions["extract"] < positions["report"]
    assert positions["verify"] < positions["report"]


def test_rejects_duplicate_node_names():
    with pytest.raises(ValueError, match="不能重复"):
        DAG([
            Node("retrieve", lambda: None),
            Node("retrieve", lambda: None),
        ])


def test_rejects_missing_dependency():
    with pytest.raises(ValueError, match="不存在"):
        DAG([
            Node("extract", lambda: None, ["retrieve"]),
        ])


def test_rejects_cycle():
    with pytest.raises(ValueError, match="循环依赖"):
        DAG([
            Node("a", lambda: None, ["b"]),
            Node("b", lambda: None, ["a"]),
        ])


def test_accepts_empty_dag():
    dag = DAG([])

    assert dag.topological_sort() == []
