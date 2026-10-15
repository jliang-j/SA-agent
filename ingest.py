import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv(encoding="utf-8")

modelqw = init_chat_model(
    model="qwen3.8-max",
    model_provider="openai",
    api_key=os.getenv("API_KEY"),
    base_url="https://maas.qianwenaiapi.com/compatible-mode/v1",
)

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"{city}一直天晴"

agent = create_agent(
    model=modelqw,
    tools=[get_weather],
    system_prompt="你是一名天气查询助手"
)
result = agent.invoke(
    {
        "messages":[{"role":"user","content":"请你告诉我长沙的天气"}]
    }
)
print(result["messages"][-1].content_blocks)