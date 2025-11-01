from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', case_sensitive=False)

    # App
    environment: str = Field(default="dev")

    # Database
    database_url: str = Field(default="sqlite+aiosqlite:///./app.db")

    # Vector store
    vector_provider: str = Field(default="qdrant")  # qdrant|pinecone|weaviate|milvus
    qdrant_url: str = Field(default=":memory:")
    qdrant_api_key: str | None = Field(default=None)
    qdrant_collection: str = Field(default="documents")

    # Embeddings
    embedding_model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_dim: int = Field(default=384)

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0")
    chat_history_ttl_seconds: int = Field(default=60 * 60 * 24)
    chat_history_max_turns: int = Field(default=15)

    # LLM
    llm_provider: str = Field(default="gemini")  # gemini|openai|openrouter|local
    gemini_api_key: str | None = Field(default=None)
    openai_api_key: str | None = Field(default=None)
    openai_model: str = Field(default="gpt-4o-mini")
    # OpenRouter via OpenAI SDK
    openrouter_key: str | None = Field(default=None)
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    openrouter_model: str = Field(default="openrouter/auto")
    app_public_url: str | None = Field(default=None)  # for OpenRouter Referer header
    app_title: str = Field(default="PalmMind Backend")  # for OpenRouter X-Title header


settings = Settings()

