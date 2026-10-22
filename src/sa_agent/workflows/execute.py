from sa_agent.skills.extract import run_extract
from sa_agent.schemas import Plan, Task
from sa_agent.skills.retrieve import run_retrieval
from sa_agent.skills.verify import run_verify
from sa_agent.workflows.dag import DAG, Node


def _task_nodes(vector_store, task: Task, k: int) -> list[Node]:
    """每次调用独立绑定一个任务，避免循环闭包串用任务数据。"""
    retrieved = f"{task.id}_retrieve"
    extracted = f"{task.id}_extract"
    verified = f"{task.id}_verify"

    def retrieve():
        return run_retrieval(vector_store, task.query, k)

    def extract(**inputs):
        # 检索保留查询词；提取依据完整研究目标，不修改上游结果。
        data = {**inputs[retrieved], "question": task.goal}
        return run_extract(data)

    def verify(**inputs):
        return run_verify(
            extracted=inputs[extracted], retrieved=inputs[retrieved],
        )

    return [
        Node(retrieved, retrieve),
        Node(extracted, extract, [retrieved]),
        Node(verified, verify, [extracted, retrieved]),
    ]


def run_tasks(vector_store, plan: dict, k: int = 3) -> dict:
    """执行一轮计划，最多并行两个节点；分支失败作为结果返回。"""
    checked = Plan.model_validate(plan)
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 10:
        raise ValueError("k 必须是 1 到 10 之间的整数")

    nodes = []
    for task in checked.tasks:
        nodes.extend(_task_nodes(vector_store, task, k))
    results = DAG(nodes).run(max_workers=2)

    tasks = []
    for task in checked.tasks:
        stages = {
            stage: results[f"{task.id}_{stage}"]
            for stage in ("retrieve", "extract", "verify")
        }
        error = next((item.error for item in stages.values() if item.error is not None), None)
        tasks.append({
            **task.model_dump(),
            "status": "failed" if error is not None else "success",
            "retrieval": stages["retrieve"].output,
            "extraction": stages["extract"].output,
            "verification": stages["verify"].output,
            "nodes": {stage: item.status for stage, item in stages.items()},
            "error": None if error is None else {
                "type": type(error).__name__, "message": str(error),
            },
        })

    return {"tasks": tasks}
