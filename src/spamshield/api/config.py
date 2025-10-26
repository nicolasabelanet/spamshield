from pydantic_settings import BaseSettings, SettingsConfigDict

from pydantic import AnyHttpUrl


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env.dev")

    API_TITLE: str = "spamsheild-api"
    API_VERSION: str = "1.0.0"
    API_KEY: str = ""
    SECONDARY_API_KEY: str = ""
    REQUIRE_HMAC: bool = False
    MAX_TEXTS_PER_REQUEST: int = 64
    MAX_TEXT_LEN: int = 2000
    RATE_LIMIT: str = "60/minute"
    ENABLE_CORS: bool = False
    CORS_ORIGINS: list[AnyHttpUrl] = []
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = True


settings = Settings()
