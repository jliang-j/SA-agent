import os
import urllib.error
import urllib.request

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv(encoding="utf-8")
print("hello")

checkpointer = InMemorySaver()

SYSTEM_PROMPT = """你是一个文档分析助手.

## 能力

- `fetch_text_from_url`: 从 URL 中加载文档文本并插入到对话中。
不要推理和猜测文档内容，必须以保存文件后的工具返回结果为准."""

modelqw = init_chat_model(
    model="qwen3.8-max",
    model_provider="openai",
    api_key=os.getenv("API_KEY"),
    base_url="https://maas.qianwenaiapi.com/compatible-mode/v1",
    temperature=0.5,
    timeout=300,
    max_tokens=25000,
    
)
'''
  
'''

@tool
def fetch_text_from_url(url: str) -> str:
    """从 URL 获取文档从 URL 获取文档.
    """
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; quickstart-research/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except urllib.error.URLError as e:
        return f"Fetch failed: {e}"
    text = raw.decode("utf-8", errors="replace")
    return text


agent = create_agent(
    model=modelqw,
    tools=[fetch_text_from_url],
    system_prompt=SYSTEM_PROMPT
)
result = agent.invoke(
    {
        "messages":[{"role":"user","content":"请你告诉我长沙的天气"}]
    }
)
print(result["messages"][-1].content_blocks)