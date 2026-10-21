
from sa_agent.skills.extract import run_extract
from sa_agent.skills.retrieve import run_retrieval
from sa_agent.skills.verify import run_verify
from sa_agent.workflows.dag import DAG, Node


def run_research(vector_store, question: str, k: int = 3) -> dict:
    """通过 DAG 执行检索、提取和来源核验，保持原有返回结构。"""
    dag = DAG([
        Node("retrieved", lambda: run_retrieval(
            vector_store=vector_store, question=question, k=k,
        )),
        Node("extracted", lambda retrieved: run_extract(retrieved), ["retrieved"]),
        Node("verified", run_verify, ["extracted", "retrieved"]),
    ])
    results = dag.run()

    # 执行器记录失败，业务入口重新抛出原异常，保留调用方的错误处理方式。
    for result in results.values():
        if result.error is not None:
            raise result.error

    return {
        "question": question,
        "retrieval": results["retrieved"].output,
        "extraction": results["extracted"].output,
        "verification": results["verified"].output,
    }
