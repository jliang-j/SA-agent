import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv(encoding="utf-8")
print("hello")



modelqw = init_chat_model(
    model="qwen3.8-max",
    model_provider="openai",
    api_key=os.getenv("API_KEY"),
    base_url="https://maas.qianwenaiapi.com/compatible-mode/v1",
    
)
'''
    temperature=0.5,
    timeout=300,
    max_tokens=25000,
'''
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
'''

from langchain_docling.loader import DoclingLoader

FILE_PATH = "https://arxiv.org/pdf/2408.09869"

loader = DoclingLoader(file_path=FILE_PATH)

# Load all documents
documents = loader.load()

# For large datasets, lazily load documents
for document in loader.lazy_load():
    print(document)


print(modelqw.invoke("请你用一句话告诉我你是谁").content)
client = OpenAI(
  api_key=os.getenv("API_KEY"),
  base_url="https://maas.qianwenaiapi.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
  model="qwen3.8-max",
  messages=[
    {"role": "system", "content": "请抽取用户的姓名与年龄信息，以JSON格式返回"},
    {"role": "user", "content": "大家好，我叫刘五，今年34岁，邮箱是liuwu@example.com，平时喜欢打篮球和旅游"},
  ],
  response_format={"type": "json_object"},
)
data = json.loads(json_string)
json_string = completion.choices[0].message.content
print(json_string)
'''