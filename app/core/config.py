from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Islamic AI Assistant API"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"

    LLM_BASE_URL: str
    LLM_MODEL: str

    # Embedding model
    EMBEDDING_MODEL: str = "nomic-embed-text"

    DATABASE_URL: str

    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 30

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()