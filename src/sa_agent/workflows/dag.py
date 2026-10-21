from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable, Literal


@dataclass
class Node:
    """依赖节点的输出以同名关键字参数传给 run。"""
    name: str
    run: Callable[..., Any]
    deps: list[str] = field(default_factory=list)

#保存节点状态、输出和原始异常
@dataclass
class NodeResult:
    status: Literal["pending", "running", "success", "failed", "skipped"] = "pending"
    output: Any = None
    error: Exception | None = None


class DAG:
    def __init__(self, nodes: list[Node]):
        self.nodes = {node.name: node for node in nodes}

        if len(self.nodes) != len(nodes):
            raise ValueError("节点名称不能重复")

        self._validate()

    def _validate(self) -> None:
        for node in self.nodes.values():
            for dep in node.deps:
                if dep not in self.nodes:
                    raise ValueError(
                        f"节点 {node.name} 依赖不存在的节点 {dep}"
                    )

        self.topological_sort()
    # 按依赖就绪调度节点，独立节点可并行执行。
    def run(self, max_workers: int = 4) -> dict[str, NodeResult]:
        """并行执行就绪节点；节点应避免修改共享的上游输出。"""
        if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
            raise ValueError("max_workers 必须是正整数") #限制同时工作的线程数量
        order = self.topological_sort()
        results = {name: NodeResult() for name in self.nodes}
        pending = dict.fromkeys(order)
        running = {}

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            while pending or running:
                for name in list(pending):
                    node = self.nodes[name]
                    states = [results[dep].status for dep in node.deps]
                    if any(state in {"failed", "skipped"} for state in states):
                        results[name].status = "skipped"
                        del pending[name]
                    elif all(state == "success" for state in states) and len(running) < max_workers:
                        inputs = {dep: results[dep].output for dep in node.deps}
                        future = pool.submit(node.run, **inputs)
                        running[future] = name
                        results[name].status = "running"
                        del pending[name]

                if not running:
                    continue

                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    result = results[running.pop(future)]
                    try:
                        result.output = future.result()
                    except Exception as exc:
                        result.status = "failed"
                        result.error = exc
                    else:
                        result.status = "success"

        return results

    def topological_sort(self) -> list[str]:
        indegree = {
            name: len(set(node.deps))
            for name, node in self.nodes.items()
        }
        children = {name: [] for name in self.nodes}

        for node in self.nodes.values():
            for dep in set(node.deps):
                children[dep].append(node.name)

        ready = deque(
            name for name in self.nodes
            if indegree[name] == 0
        )
        order = []

        while ready:
            name = ready.popleft()
            order.append(name)

            for child in children[name]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)

        if len(order) != len(self.nodes):
            raise ValueError("DAG 存在循环依赖")

        return order
