from langchain_openai import OpenAIEmbeddings

from sa_agent.config import settings


def create_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        check_embedding_ctx_length=False,
    )