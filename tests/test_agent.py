import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from sa_agent.workflows import agent


class ToolModel(FakeMessagesListChatModel):
    """使用预设消息驱动真实框架。"""

    def bind_tools(self, tools, **kwargs):
        """模拟工具绑定，无网络请求。"""
        return self


def task(task_id="t1"):
    """构造研究任务。"""
    return {"id": task_id, "goal": f"目标{task_id}", "query": f"查询{task_id}"}


def call(tasks, call_id="c1"):
    """生成模型工具调用消息。"""
    return AIMessage(content="", tool_calls=[{
        "name": "research_tasks", "args": {"tasks": tasks}, "id": call_id, "type": "tool_call",
    }])


def final(text="结束"):
    """生成无证据场景下的合法简报消息。"""
    return AIMessage(content=json.dumps({
        "summary": text, "findings": [], "gaps": ["模拟资料不足"],
    }, ensure_ascii=False))


def setup_flow(monkeypatch, messages, failed=False):
    """模拟模型和研究执行，保留真实框架及工具校验。"""
    calls = []
    monkeypatch.setattr(agent, "create_chat_model", lambda: ToolModel(responses=messages))

    def execute(store, plan, k):
        """返回模拟研究结果并记录实际调用。"""
        calls.append(plan)
        return {"tasks": [
            {**item, "status": "failed" if failed else "success", "nodes": {},
             "verification": None if failed else {"facts": []}}
            for item in plan["tasks"]
        ]}

    monkeypatch.setattr(agent, "run_tasks", execute)
    return calls


def test_direct_finish(monkeypatch):
    """调用一次工具后返回回答。"""
    calls = setup_flow(monkeypatch, [call([task()]), final("研究回答")])
    result = agent.run_agent(object(), " 研究问题 ")
    assert len(calls) == 1
    assert result["question"] == "研究问题"
    assert result["report"]["summary"] == "研究回答"
    assert "研究回答" in result["answer"]
    assert result["tasks"][0]["id"] == "t1"


def test_followup(monkeypatch):
    """模型补查后保留累计结果。"""
    calls = setup_flow(monkeypatch, [call([task()]), call([task("t2")], "c2"), final()])
    result = agent.run_agent(object(), "研究问题")
    assert len(calls) == 2
    assert [t["id"] for t in result["tasks"]] == ["t1", "t2"]
    assert [t["call"] for t in result["trace"]] == [1, 2]


def test_limit(monkeypatch):
    """第三次调用不进入 DAG。"""
    calls = setup_flow(monkeypatch, [call([task()]), call([task("t2")], "c2"),
                                    call([task("t3")], "c3"), final()])
    result = agent.run_agent(object(), "研究问题")
    assert len(calls) == 2
    assert result["trace"][-1]["step"] == "rejected"


@pytest.mark.parametrize("followups", [[task()], [task("t2"), task("t3"), task("t4")]])
def test_bad_followup(monkeypatch, followups):
    """拒绝重复或过多的补查任务。"""
    calls = setup_flow(monkeypatch, [call([task()]), call(followups, "c2"), final()])
    agent.run_agent(object(), "研究问题")
    assert len(calls) == 1


def test_invalid_tool_args(monkeypatch):
    """模型能在收到格式错误后修正参数。"""
    calls = setup_flow(monkeypatch, [call([]), call([task()], "c2"), final()])
    agent.run_agent(object(), "研究问题")
    assert len(calls) == 1


def test_no_research(monkeypatch):
    """拒绝未研究就直接给出的回答。"""
    setup_flow(monkeypatch, [AIMessage(content="直接回答")])
    with pytest.raises(ValueError, match="未执行"):
        agent.run_agent(object(), "研究问题")


def test_request_isolation(monkeypatch):
    """两次请求不共享研究计数和结果。"""
    setup_flow(monkeypatch, [call([task()]), final()])
    first = agent.run_agent(object(), "研究问题")
    second = agent.run_agent(object(), "研究问题")
    assert len(first["tasks"]) == len(second["tasks"]) == 1
    assert second["trace"][0]["call"] == 1


def test_failed_tasks(monkeypatch):
    """保留失败任务供调用方检查。"""
    setup_flow(monkeypatch, [call([task()]), final("资料不足")], failed=True)
    result = agent.run_agent(object(), "研究问题")
    assert result["tasks"][0]["status"] == "failed"


def test_evidence_filter(monkeypatch):
    """工具只向模型返回核验通过的事实，不暴露原始错误和未通过事实。"""
    def execute(store, plan, k):
        """模拟混有通过和未通过事实的研究结果。"""
        return {"tasks": [{**task(), "status": "success", "verification": {"facts": [
            {"fact": "有效事实", "source": "demo.txt", "chunk_index": 0,
             "evidence": "原文", "status": "passed"},
            {"fact": "无效事实", "status": "failed"},
        ]}, "error": "内部信息"}]}

    monkeypatch.setattr(agent, "run_tasks", execute)
    history, trace = [], []
    research = agent._make_tool(object(), 3, history, trace)
    result = research.invoke({"tasks": [task()]})
    assert len(result["tasks"][0]["facts"]) == 1
    assert result["tasks"][0]["facts"][0]["fact"] == "有效事实"
    assert result["tasks"][0]["facts"][0]["id"] == "E1"
    assert "内部信息" not in str(result)
    assert result["remaining_calls"] == 1
    assert len(history[0]["verification"]["facts"]) == 2


def test_invalid_final(monkeypatch):
    """即使工具执行成功，也不能返回未通过校验的最终回答。"""
    setup_flow(monkeypatch, [call([task()]), AIMessage(content="未结构化的回答")])
    with pytest.raises(ValueError, match="简报格式"):
        agent.run_agent(object(), "研究问题")


@pytest.mark.parametrize("status,docs,extracted,verified,outcome", [
    ("success", [], [], [], "no_documents"),
    ("success", [{}], [], [], "no_facts"),
    ("success", [{}], [{}], [{"status": "failed"}], "no_verified_facts"),
    ("failed", [{}], [], [], "execution_failed"),
])
def test_outcome(status, docs, extracted, verified, outcome):
    """根据实际阶段计数区分检索为空、无事实、核验失败和执行失败。"""
    result = agent._summarize({
        **task(), "status": status, "retrieval": {"results": docs},
        "extraction": {"facts": extracted}, "verification": {"facts": verified},
    })
    assert result["outcome"] == outcome
    assert result["counts"]["retrieved"] == len(docs)
    assert result["counts"]["extracted"] == len(extracted)
    assert result["facts"] == []


def test_available_evidence():
    """有核验事实时提供证据状态，但不宣称资料完整。"""
    fact = {"fact": "事实", "source": "demo.txt", "chunk_index": 0,
            "evidence": "原文", "status": "passed"}
    result = agent._summarize({
        **task(), "status": "success", "retrieval": {"results": [{}]},
        "extraction": {"facts": [fact]}, "verification": {"facts": [fact]},
    })
    assert result["outcome"] == "evidence_available"
    assert result["counts"]["passed"] == 1


@pytest.mark.parametrize("question,k", [(" ", 3), (None, 3), ("研究问题", 0), ("研究问题", True)])
def test_invalid_input(question, k):
    """无效输入在调用模型前被拒绝。"""
    with pytest.raises(ValueError):
        agent.run_agent(object(), question, k)
