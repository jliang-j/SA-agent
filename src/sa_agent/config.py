
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model: str = Field(validation_alias="MODEL_NAME")
    api_key: str = Field(validation_alias="API_KEY")
    base_url: str = Field(validation_alias="BASE_URL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    embedding_model: str = Field(
        validation_alias="EMBEDDING_MODEL",
    )

settings = Settings()
