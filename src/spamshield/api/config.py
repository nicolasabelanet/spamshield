from pydantic_settings import BaseSettings, SettingsConfigDict
import functools

from pydantic import AnyHttpUrl


class Settings(BaseSettings):
    """
    Central configuration for the SpamShield API service.

    This class defines all environment-driven settings that control the
    application’s runtime behavior - including model versioning, API metadata,
    authentication, rate limiting, CORS, and logging options.

    Values are automatically populated from environment variables using the
    prefix `SPAMSHIELD_`, or from the `.env.dev` file when present. This allows
    consistent configuration across local development, CI/CD, and production
    environments.

    Examples
    --------
    - `SPAMSHIELD_MODEL_VERSION=v1.0.0`
    - `SPAMSHIELD_REQUIRE_HMAC=true`
    - `SPAMSHIELD_LOG_LEVEL=DEBUG`

    Notes
    -----
    - Import and reuse the global `settings` instance rather than instantiating
      this class directly.
    """

    model_config = SettingsConfigDict(env_file=".env.dev", env_prefix="SPAMSHIELD_")

    # Version string of the active trained spam model
    MODEL_VERSION: str

    # Primary API key (stored as SHA-256 hash)
    API_KEY_HASH: str = ""

    # Optional shared secret (e.g. for HMAC signing)
    API_SECRET: str = "foo"

    # Backup key (allows rotation without downtime)
    SECONDARY_API_KEY_HASH: str = ""

    # Whether to enforce HMAC-signed requests
    REQUIRE_HMAC: bool = False

    # Max number of messages per /predict call
    MAX_TEXTS_PER_REQUEST: int = 64

    # Max length of each text sample (chars)
    MAX_TEXT_LEN: int = 2000

    # API rate limit (used by slowapi limiter)
    RATE_LIMIT: str = "60/minute"

    # Enable CORS middleware for browser clients
    ENABLE_CORS: bool = False

    # Allowed CORS origins (if enabled)
    CORS_ORIGINS: list[AnyHttpUrl] = []

    # Minimum log level (DEBUG, INFO, WARNING...)
    LOG_LEVEL: str = "INFO"

    # Emit logs in structured JSON format
    LOG_JSON: bool = True

    @classmethod
    def env(cls) -> Settings:
        """
        Load and validate settings from environment variables.

        Returns
        -------
        Settings
            A validated Settings instance with all fields populated from
            environment variables or defaults defined above.
        """
        return Settings.model_validate({})


@functools.cache
def get_settings() -> Settings:
    """
    Retrieve a cached global instance of the Settings.

    This function provides a lightweight singleton for accessing settings
    throughout the application. The instance is memoized using
    `functools.cache`, ensuring that all routes share the same
    settings instance unless explicitly overridden.

    Returns
    -------
    Settings
        The cached settings instance.
    """
    return Settings.env()
