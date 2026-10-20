from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Node:
    name: str
    run: Callable[..., Any]
    deps: list[str] = field(default_factory=list)


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
