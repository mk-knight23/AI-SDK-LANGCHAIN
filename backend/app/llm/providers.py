"""LLM provider implementations.

This module provides a unified interface for multiple LLM providers including:
- OpenAI (GPT-4, GPT-4o)
- Anthropic (Claude Sonnet, Opus, Haiku)
- OpenRouter (multi-model routing)
- Perplexity (search augmentation)

All providers implement the LLMProvider interface for consistent usage.
"""
from abc import ABC, abstractmethod
from typing import Generator
import logging

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from app.config import LLMConfig, get_llm_config, get_config

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Exception raised when LLM provider operations fail.

    Attributes:
        message: Error message
        cause: Original exception that caused the error
    """

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.message = message
        self.cause = cause

    def __str__(self) -> str:
        if self.cause:
            return f"{self.message}: {self.cause}"
        return self.message


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    All LLM providers must implement this interface to ensure
    consistent behavior across different providers.
    """

    def __init__(self, config: LLMConfig) -> None:
        """Initialize the provider with configuration.

        Args:
            config: LLM configuration for this provider
        """
        self.config = config
        self._llm = self._create_llm()

    @abstractmethod
    def _create_llm(self):
        """Create and configure the underlying LLM instance."""
        pass

    def invoke(self, prompt: str, **kwargs):
        """Invoke the LLM with a prompt.

        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional arguments for the LLM

        Returns:
            LLM response object

        Raises:
            ProviderError: If invocation fails
        """
        try:
            return self._llm.invoke(prompt, **kwargs)
        except Exception as e:
            logger.error(f"{self.config.provider} invoke failed: {e}")
            raise ProviderError(
                f"{self.config.provider} provider invoke failed",
                cause=e,
            ) from e

    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Stream the LLM response token by token.

        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional arguments for the LLM

        Yields:
            Individual tokens/chunks from the LLM response

        Raises:
            ProviderError: If streaming fails
        """
        try:
            for chunk in self._llm.stream(prompt, **kwargs):
                yield chunk.content
        except Exception as e:
            logger.error(f"{self.config.provider} stream failed: {e}")
            raise ProviderError(
                f"{self.config.provider} provider stream failed",
                cause=e,
            ) from e

    def get_model(self) -> str:
        """Get the model name for this provider.

        Returns:
            Model identifier string
        """
        return self.config.model


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider using ChatOpenAI.

    Supports GPT-4, GPT-4o, GPT-4o-mini, and other OpenAI models.
    """

    def _create_llm(self) -> ChatOpenAI:
        """Create ChatOpenAI instance with configuration."""
        return ChatOpenAI(
            model=self.config.model,
            api_key=self.config.api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
        )


class AnthropicProvider(LLMProvider):
    """Anthropic LLM provider using ChatAnthropic.

    Supports Claude Sonnet, Opus, and Haiku models.
    """

    def _create_llm(self) -> ChatAnthropic:
        """Create ChatAnthropic instance with configuration."""
        return ChatAnthropic(
            model=self.config.model,
            api_key=self.config.api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
        )


class OpenRouterProvider(LLMProvider):
    """OpenRouter LLM provider for multi-model routing.

    OpenRouter provides access to multiple models through a single API.
    Uses ChatOpenAI with custom base_url for OpenRouter compatibility.
    """

    def _create_llm(self) -> ChatOpenAI:
        """Create ChatOpenAI instance with OpenRouter base URL."""
        return ChatOpenAI(
            model=self.config.model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
        )


class PerplexityProvider(LLMProvider):
    """Perplexity LLM provider with search augmentation.

    Perplexity models have built-in web search capabilities
    for up-to-date information retrieval.
    """

    def _create_llm(self) -> ChatOpenAI:
        """Create ChatOpenAI instance with Perplexity base URL."""
        return ChatOpenAI(
            model=self.config.model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
        )


def create_llm_provider(config: LLMConfig) -> LLMProvider:
    """Create an LLM provider instance based on configuration.

    Factory function that returns the appropriate provider implementation
    based on the provider type in the configuration.

    Args:
        config: LLM configuration specifying provider type

    Returns:
        Configured LLM provider instance

    Raises:
        ValueError: If provider type is unknown

    Examples:
        >>> config = LLMConfig(provider="openai", api_key="key", model="gpt-4")
        >>> provider = create_llm_provider(config)
        >>> response = provider.invoke("Hello!")
    """
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "openrouter": OpenRouterProvider,
        "perplexity": PerplexityProvider,
    }

    provider_class = providers.get(config.provider)
    if provider_class is None:
        raise ValueError(
            f"Unknown LLM provider: {config.provider}. "
            f"Available providers: {list(providers.keys())}"
        )

    return provider_class(config)


def create_llm_with_fallback(
    provider: str,
    prompt: str,
    **kwargs,
) -> object:
    """Create LLM provider and invoke with automatic fallback.

    Tries the primary provider first, then falls back to alternative
    providers configured in the fallback chain if the primary fails.

    Args:
        provider: Primary provider name to try first
        prompt: Prompt to send to the LLM
        **kwargs: Additional arguments for the LLM

    Returns:
        LLM response from the first successful provider

    Raises:
        ProviderError: If all providers in the fallback chain fail
        ValueError: If no providers are configured

    Examples:
        >>> response = create_llm_with_fallback("openai", "Explain AI")
        >>> # Falls back to anthropic, openrouter if openai fails
    """
    config = get_config()
    fallback_chain = config.llm_fallback_chain

    # Ensure the requested provider is first
    if provider in fallback_chain:
        fallback_chain.remove(provider)
    providers_to_try = [provider] + fallback_chain

    errors = []

    for provider_name in providers_to_try:
        try:
            llm_config = get_llm_config(provider_name)
            llm_provider = create_llm_provider(llm_config)
            logger.info(f"Using LLM provider: {provider_name}")
            return llm_provider.invoke(prompt, **kwargs)
        except ValueError as e:
            # Provider not configured, skip to next
            logger.warning(f"Provider {provider_name} not configured: {e}")
            continue
        except ProviderError as e:
            # Provider failed, save error and try next
            logger.error(f"Provider {provider_name} failed: {e}")
            errors.append((provider_name, e))
            continue

    # All providers failed
    raise ProviderError(
        f"All LLM providers failed. Tried: {', '.join([p for p, _ in errors])}",
    )
