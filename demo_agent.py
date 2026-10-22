import argparse
import json
from pathlib import Path
from time import perf_counter

from sa_agent.embeddings import create_embeddings
from sa_agent.retrieval.store import create_persistent_vector_store
from sa_agent.workflows.agent import run_agent


def main():
    """手动运行真实研究 Agent，保存完整结果并打印执行摘要。"""
    parser = argparse.ArgumentParser(description="AI 行业研究助手手动测试")
    parser.add_argument("--question", help="研究问题，不传则在终端输入")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--output", default="outputs/agent_result.json")
    args = parser.parse_args()
    question = args.question if args.question is not None else input("请输入研究问题：")
    if not question.strip() or not 1 <= args.k <= 10:
        parser.error("问题不能为空，k 必须在1到10之间")
    if not Path("data/chroma").exists():
        parser.error("本地向量库不存在，请先导入示例文档")
    store = create_persistent_vector_store(
        persist_directory="data/chroma", collection_name="industry_documents",
        embeddings=create_embeddings(),
    )
    if store._collection.count() == 0:
        parser.error("行业资料集合为空，请先导入示例文档")
    print("正在研究，请等待模型完成……", flush=True)
    started = perf_counter()
    result = run_agent(store, question, k=args.k)
    result["elapsed_seconds"] = round(perf_counter() - started, 2)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["trace"], ensure_ascii=False, indent=2))
    print("\n最终回答：\n", result["answer"])
    print(f"\n耗时：{result['elapsed_seconds']} 秒；完整结果：{output.resolve()}")


if __name__ == "__main__":
    main()
