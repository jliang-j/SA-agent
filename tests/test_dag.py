
import pytest

from sa_agent.workflows.dag import DAG, Node


def test_topo_sort():
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


def test_duplicate_nodes():
    with pytest.raises(ValueError, match="不能重复"):
        DAG([
            Node("retrieve", lambda: None),
            Node("retrieve", lambda: None),
        ])


def test_missing_dep():
    with pytest.raises(ValueError, match="不存在"):
        DAG([
            Node("extract", lambda: None, ["retrieve"]),
        ])


def test_cycle():
    with pytest.raises(ValueError, match="循环依赖"):
        DAG([
            Node("a", lambda: None, ["b"]),
            Node("b", lambda: None, ["a"]),
        ])


def test_empty_dag():
    dag = DAG([])

    assert dag.topological_sort() == []
    assert dag.run() == {}


def test_outputs():
    dag = DAG([
        Node("total", lambda left, right: left + right, ["left", "right"]),
        Node("left", lambda: 2),
        Node("right", lambda left: left * 3, ["left"]),
    ])

    results = dag.run()

    assert results["total"].output == 8
    assert all(result.status == "success" for result in results.values())
    assert all(result.error is None for result in results.values())


def test_failure():
    error = ValueError("模拟失败")

    def fail():
        raise error

    def unexpected(**kwargs):
        pytest.fail("失败节点的下游不应执行")

    dag = DAG([
        Node("bad", fail),
        Node("child", unexpected, ["bad"]),
        Node("leaf", unexpected, ["child"]),
        Node("other", lambda: 5),
        Node("next", lambda other: other + 1, ["other"]),
        Node("join", unexpected, ["bad", "next"]),
    ])

    results = dag.run()

    assert results["bad"].status == "failed"
    assert results["bad"].error is error
    assert results["bad"].output is None
    for name in ("child", "leaf", "join"):
        assert results[name].status == "skipped"
        assert results[name].output is None
        assert results[name].error is None
    assert results["next"].status == "success"
    assert results["next"].output == 6


def test_none_output():
    dag = DAG([
        Node("source", lambda: None),
        Node("check", lambda source: source is None, ["source"]),
    ])

    results = dag.run()

    assert results["source"].status == "success"
    assert results["check"].output is True


def test_repeat_run():
    calls = []

    def source():
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("首次失败")
        return 7

    dag = DAG([
        Node("source", source),
        Node("child", lambda source: source + 1, ["source"]),
    ])

    first = dag.run()
    second = dag.run()

    assert first["source"].status == "failed"
    assert first["child"].status == "skipped"
    assert second["source"].error is None
    assert second["child"].status == "success"
    assert second["child"].output == 8
