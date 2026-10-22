from uuid import uuid4

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage

from sa_agent.llm import create_chat_model


@pytest.mark.integration
def test_real_tool_agent():
    """验证现有模型经 create_agent 调用工具并使用返回结果。"""
    marker = f"probe-{uuid4().hex}"
    calls = []

    def lookup_demo(topic: str) -> str:
        """获取指定主题的演示记录及校验标记；此工具不查询真实行业资料。"""
        calls.append(topic)
        return f"演示主题：{topic}；校验标记：{marker}"

    agent = create_agent(
        model=create_chat_model(),
        tools=[lookup_demo],
        system_prompt=(
            "你正在执行工具调用兼容性测试。必须调用 lookup_demo 获取记录，"
            "然后在最终回答中逐字给出工具返回的校验标记，不得猜测标记。"
            "获取记录成功后不要重复调用。"
        ),
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "请查询主题为人工智能风险的演示记录，并返回校验标记。"}]},
        config={"recursion_limit": 6},
    )
    messages = result["messages"]
    tool_calls = [
        call for message in messages if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]
    tool_results = [message for message in messages if isinstance(message, ToolMessage)]

    assert calls, "模型未触发工具执行"
    assert all(isinstance(topic, str) and topic.strip() for topic in calls)
    assert any(call["name"] == "lookup_demo" for call in tool_calls)
    assert any(marker in str(message.content) for message in tool_results)
    assert isinstance(messages[-1], AIMessage)
    assert not messages[-1].tool_calls
    assert marker in str(messages[-1].content), "最终回答未使用工具返回的标记"

    print("\n工具调用：", tool_calls)
    print("最终回答：", messages[-1].content)
