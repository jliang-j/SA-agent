
from langchain.chat_models import init_chat_model

from sa_agent.config import settings

def create_chat_model():
    return init_chat_model(
        model=settings.model,
        model_provider="openai",
        api_key=settings.api_key,
        base_url=settings.base_url,
    )