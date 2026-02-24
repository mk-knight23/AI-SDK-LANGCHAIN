"""Application configuration management.

Implements immutable configuration patterns with environment-based loading.
"""
from dataclasses import dataclass, field
from functools import lru_cache
from os import getenv
from typing import Literal


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for a single LLM provider."""

    provider: Literal["openai", "anthropic", "openrouter", "perplexity"]
    api_key: str
    model: str
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60


@dataclass(frozen=True)
class DatabaseConfig:
    """PostgreSQL database configuration."""

    url: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    echo: bool = False


@dataclass(frozen=True)
class RedisConfig:
    """Redis caching configuration."""

    url: str
    default_ttl: int = 3600
    max_connections: int = 50
    socket_timeout: int = 5


@dataclass(frozen=True)
class WebSocketConfig:
    """WebSocket configuration."""

    heartbeat_interval: int = 30
    max_connections: int = 100
    message_queue_size: int = 100


@dataclass(frozen=True)
class AppConfig:
    """Main application configuration."""

    # Environment
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    # API
    api_title: str = "AI-SDK LangChain API"
    api_version: str = "1.0.0"
    cors_origins: list[str] = field(default_factory=lambda: ["*"])

    # LLM Providers
    openai: LLMConfig | None = None
    anthropic: LLMConfig | None = None
    openrouter: LLMConfig | None = None
    perplexity: LLMConfig | None = None

    # Default LLM provider
    default_llm_provider: Literal["openai", "anthropic", "openrouter", "perplexity"] = "openai"

    # Fallback chain (ordered list of providers to try)
    llm_fallback_chain: list[str] = field(default_factory=lambda: ["openai", "anthropic", "openrouter"])

    # Database
    database: DatabaseConfig = field(default_factory=lambda: DatabaseConfig(
        url=getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/ai_sdk_langchain")
    ))

    # Redis
    redis: RedisConfig = field(default_factory=lambda: RedisConfig(
        url=getenv("REDIS_URL", "redis://localhost:6379/0")
    ))

    # WebSocket
    websocket: WebSocketConfig = field(default_factory=lambda: WebSocketConfig())

    # LangSmith (tracing)
    langsmith_api_key: str | None = field(default_factory=lambda: getenv("LANGCHAIN_API_KEY"))
    langsmith_project: str = "ai-sdk-langchain"


@lru_cache
def get_config() -> AppConfig:
    """Get cached application configuration.

    Loads configuration from environment variables and returns immutable config.
    """
    env: Literal["development", "staging", "production"] = (
        getenv("ENVIRONMENT", "development")  # type: ignore
    )

    # OpenAI configuration
    openai_key = getenv("OPENAI_API_KEY")
    openai_config: LLMConfig | None = None
    if openai_key:
        openai_config = LLMConfig(
            provider="openai",
            api_key=openai_key,
            model=getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=float(getenv("OPENAI_TEMPERATURE", "0.7")),
        )

    # Anthropic configuration
    anthropic_key = getenv("ANTHROPIC_API_KEY")
    anthropic_config: LLMConfig | None = None
    if anthropic_key:
        anthropic_config = LLMConfig(
            provider="anthropic",
            api_key=anthropic_key,
            model=getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            temperature=float(getenv("ANTHROPIC_TEMPERATURE", "0.7")),
        )

    # OpenRouter configuration
    openrouter_key = getenv("OPENROUTER_API_KEY")
    openrouter_config: LLMConfig | None = None
    if openrouter_key:
        openrouter_config = LLMConfig(
            provider="openrouter",
            api_key=openrouter_key,
            model=getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4"),
            base_url="https://openrouter.ai/api/v1",
            temperature=float(getenv("OPENROUTER_TEMPERATURE", "0.7")),
        )

    # Perplexity configuration
    perplexity_key = getenv("PERPLEXITY_API_KEY")
    perplexity_config: LLMConfig | None = None
    if perplexity_key:
        perplexity_config = LLMConfig(
            provider="perplexity",
            api_key=perplexity_key,
            model=getenv("PERPLEXITY_MODEL", "llama-3.1-sonar-small-128k-online"),
            base_url="https://api.perplexity.ai",
            temperature=float(getenv("PERPLEXITY_TEMPERATURE", "0.2")),
        )

    # Database configuration
    database_url = getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/ai_sdk_langchain")
    database_config = DatabaseConfig(url=database_url)

    # Redis configuration
    redis_url = getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_config = RedisConfig(url=redis_url)

    return AppConfig(
        environment=env,
        debug=env == "development",
        openai=openai_config,
        anthropic=anthropic_config,
        openrouter=openrouter_config,
        perplexity=perplexity_config,
        database=database_config,
        redis=redis_config,
    )


def get_llm_config(provider: str) -> LLMConfig:
    """Get LLM configuration for a specific provider.

    Args:
        provider: The LLM provider name

    Returns:
        LLMConfig for the provider

    Raises:
        ValueError: If provider is not configured
    """
    config = get_config()

    providers = {
        "openai": config.openai,
        "anthropic": config.anthropic,
        "openrouter": config.openrouter,
        "perplexity": config.perplexity,
    }

    llm_config = providers.get(provider)
    if llm_config is None:
        available = [k for k, v in providers.items() if v is not None]
        raise ValueError(
            f"LLM provider '{provider}' not configured. "
            f"Available providers: {available or 'none'}"
        )

    return llm_config
