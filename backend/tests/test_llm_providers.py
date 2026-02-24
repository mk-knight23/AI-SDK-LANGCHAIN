"""Tests for LLM provider implementations.

TDD approach: Tests written before implementation.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Generator

from app.llm.providers import (
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    OpenRouterProvider,
    PerplexityProvider,
    create_llm_provider,
    create_llm_with_fallback,
    ProviderError,
)
from app.config import LLMConfig, get_config


@pytest.fixture
def openai_config() -> LLMConfig:
    """Create OpenAI configuration for testing."""
    return LLMConfig(
        provider="openai",
        api_key="test-openai-key",
        model="gpt-4o-mini",
        temperature=0.7,
    )


@pytest.fixture
def anthropic_config() -> LLMConfig:
    """Create Anthropic configuration for testing."""
    return LLMConfig(
        provider="anthropic",
        api_key="test-anthropic-key",
        model="claude-sonnet-4-20250514",
        temperature=0.7,
    )


@pytest.fixture
def openrouter_config() -> LLMConfig:
    """Create OpenRouter configuration for testing."""
    return LLMConfig(
        provider="openrouter",
        api_key="test-openrouter-key",
        model="anthropic/claude-sonnet-4",
        base_url="https://openrouter.ai/api/v1",
        temperature=0.7,
    )


@pytest.fixture
def perplexity_config() -> LLMConfig:
    """Create Perplexity configuration for testing."""
    return LLMConfig(
        provider="perplexity",
        api_key="test-perplexity-key",
        model="llama-3.1-sonar-small-128k-online",
        base_url="https://api.perplexity.ai",
        temperature=0.2,
    )


class TestLLMProvider:
    """Base tests for LLM provider interface."""

    def test_provider_interface_has_required_methods(self):
        """Test that LLMProvider interface defines required methods."""
        assert hasattr(LLMProvider, "invoke")
        assert hasattr(LLMProvider, "stream")
        assert hasattr(LLMProvider, "get_model")


class TestOpenAIProvider:
    """Tests for OpenAI provider implementation."""

    @patch("app.llm.providers.ChatOpenAI")
    def test_openai_provider_initializes_with_config(self, mock_chat_openai, openai_config):
        """Test OpenAI provider initializes with correct configuration."""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm

        provider = OpenAIProvider(openai_config)

        mock_chat_openai.assert_called_once_with(
            model=openai_config.model,
            api_key=openai_config.api_key,
            temperature=openai_config.temperature,
            max_tokens=openai_config.max_tokens,
            timeout=openai_config.timeout,
        )
        assert provider.get_model() == openai_config.model

    @patch("app.llm.providers.ChatOpenAI")
    def test_openai_provider_invoke_calls_llm(self, mock_chat_openai, openai_config):
        """Test OpenAI provider invoke calls underlying LLM."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Test response"
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm

        provider = OpenAIProvider(openai_config)
        response = provider.invoke("Test prompt")

        mock_llm.invoke.assert_called_once()
        assert response.content == "Test response"

    @patch("app.llm.providers.ChatOpenAI")
    def test_openai_provider_stream_yields_tokens(self, mock_chat_openai, openai_config):
        """Test OpenAI provider stream yields tokens."""
        mock_llm = Mock()
        mock_chunk1 = Mock(content="Hello")
        mock_chunk2 = Mock(content=" world")
        mock_chunk3 = Mock(content="!")
        mock_llm.stream.return_value = [mock_chunk1, mock_chunk2, mock_chunk3]
        mock_chat_openai.return_value = mock_llm

        provider = OpenAIProvider(openai_config)
        tokens = list(provider.stream("Test prompt"))

        assert tokens == ["Hello", " world", "!"]
        mock_llm.stream.assert_called_once()

    @patch("app.llm.providers.ChatOpenAI")
    def test_openai_provider_handles_invoke_error(self, mock_chat_openai, openai_config):
        """Test OpenAI provider handles invoke errors gracefully."""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("API error")
        mock_chat_openai.return_value = mock_llm

        provider = OpenAIProvider(openai_config)

        with pytest.raises(ProviderError):
            provider.invoke("Test prompt")


class TestAnthropicProvider:
    """Tests for Anthropic provider implementation."""

    @patch("app.llm.providers.ChatAnthropic")
    def test_anthropic_provider_initializes_with_config(self, mock_chat_anthropic, anthropic_config):
        """Test Anthropic provider initializes with correct configuration."""
        mock_llm = Mock()
        mock_chat_anthropic.return_value = mock_llm

        provider = AnthropicProvider(anthropic_config)

        mock_chat_anthropic.assert_called_once()
        assert provider.get_model() == anthropic_config.model

    @patch("app.llm.providers.ChatAnthropic")
    def test_anthropic_provider_invoke_calls_llm(self, mock_chat_anthropic, anthropic_config):
        """Test Anthropic provider invoke calls underlying LLM."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Test response"
        mock_llm.invoke.return_value = mock_response
        mock_chat_anthropic.return_value = mock_llm

        provider = AnthropicProvider(anthropic_config)
        response = provider.invoke("Test prompt")

        mock_llm.invoke.assert_called_once()
        assert response.content == "Test response"


class TestOpenRouterProvider:
    """Tests for OpenRouter provider implementation."""

    @patch("app.llm.providers.ChatOpenAI")
    def test_openrouter_provider_initializes_with_base_url(self, mock_chat_openai, openrouter_config):
        """Test OpenRouter provider initializes with custom base URL."""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm

        provider = OpenRouterProvider(openrouter_config)

        mock_chat_openai.assert_called_once()
        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs["base_url"] == openrouter_config.base_url

    @patch("app.llm.providers.ChatOpenAI")
    def test_openrouter_provider_invoke_calls_llm(self, mock_chat_openai, openrouter_config):
        """Test OpenRouter provider invoke calls underlying LLM."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Test response"
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm

        provider = OpenRouterProvider(openrouter_config)
        response = provider.invoke("Test prompt")

        mock_llm.invoke.assert_called_once()
        assert response.content == "Test response"


class TestPerplexityProvider:
    """Tests for Perplexity provider implementation."""

    @patch("app.llm.providers.ChatOpenAI")
    def test_perplexity_provider_initializes_with_base_url(self, mock_chat_openai, perplexity_config):
        """Test Perplexity provider initializes with custom base URL."""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm

        provider = PerplexityProvider(perplexity_config)

        mock_chat_openai.assert_called_once()
        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs["base_url"] == perplexity_config.base_url

    @patch("app.llm.providers.ChatOpenAI")
    def test_perplexity_provider_invoke_calls_llm(self, mock_chat_openai, perplexity_config):
        """Test Perplexity provider invoke calls underlying LLM."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Test response"
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm

        provider = PerplexityProvider(perplexity_config)
        response = provider.invoke("Test prompt")

        mock_llm.invoke.assert_called_once()
        assert response.content == "Test response"


class TestCreateLLMProvider:
    """Tests for LLM provider factory function."""

    def test_create_openai_provider(self, openai_config):
        """Test factory creates OpenAI provider for 'openai' type."""
        with patch("app.llm.providers.OpenAIProvider") as mock_provider:
            provider = create_llm_provider(openai_config)
            mock_provider.assert_called_once_with(openai_config)

    def test_create_anthropic_provider(self, anthropic_config):
        """Test factory creates Anthropic provider for 'anthropic' type."""
        with patch("app.llm.providers.AnthropicProvider") as mock_provider:
            provider = create_llm_provider(anthropic_config)
            mock_provider.assert_called_once_with(anthropic_config)

    def test_create_openrouter_provider(self, openrouter_config):
        """Test factory creates OpenRouter provider for 'openrouter' type."""
        with patch("app.llm.providers.OpenRouterProvider") as mock_provider:
            provider = create_llm_provider(openrouter_config)
            mock_provider.assert_called_once_with(openrouter_config)

    def test_create_perplexity_provider(self, perplexity_config):
        """Test factory creates Perplexity provider for 'perplexity' type."""
        with patch("app.llm.providers.PerplexityProvider") as mock_provider:
            provider = create_llm_provider(perplexity_config)
            mock_provider.assert_called_once_with(perplexity_config)

    def test_create_provider_raises_for_unknown_type(self):
        """Test factory raises ValueError for unknown provider type."""
        from app.config import LLMConfig
        unknown_config = LLMConfig(
            provider="unknown",  # type: ignore
            api_key="test-key",
            model="test-model",
        )

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_provider(unknown_config)


class TestLLMProviderFallback:
    """Tests for LLM provider fallback logic."""

    @patch("app.llm.providers.create_llm_provider")
    @patch("app.config.get_llm_config")
    def test_fallback_tries_first_provider(self, mock_get_config, mock_create_provider):
        """Test fallback tries the primary provider first."""
        config = Mock()
        mock_get_config.return_value = config

        mock_provider = Mock()
        mock_response = Mock(content="Success")
        mock_provider.invoke.return_value = mock_response
        mock_create_provider.return_value = mock_provider

        with patch("app.config.get_config") as mock_app_config:
            mock_app_config.return_value = Mock(
                llm_fallback_chain=["openai", "anthropic"]
            )

            result = create_llm_with_fallback("openai", "Test prompt")

            assert result.content == "Success"
            mock_provider.invoke.assert_called_once()

    @patch("app.llm.providers.create_llm_provider")
    @patch("app.config.get_llm_config")
    def test_fallback_tries_next_provider_on_error(self, mock_get_config, mock_create_provider):
        """Test fallback tries next provider when primary fails."""
        config = Mock()
        mock_get_config.return_value = config

        # First provider fails
        failing_provider = Mock()
        failing_provider.invoke.side_effect = ProviderError("Primary failed")

        # Second provider succeeds
        success_provider = Mock()
        mock_response = Mock(content="Fallback success")
        success_provider.invoke.return_value = mock_response

        mock_create_provider.side_effect = [failing_provider, success_provider]

        with patch("app.config.get_config") as mock_app_config:
            mock_app_config.return_value = Mock(
                llm_fallback_chain=["openai", "anthropic"]
            )

            result = create_llm_with_fallback("openai", "Test prompt")

            assert result.content == "Fallback success"
            assert mock_create_provider.call_count == 2

    @patch("app.llm.providers.create_llm_provider")
    @patch("app.config.get_llm_config")
    def test_fallback_raises_when_all_providers_fail(self, mock_get_config, mock_create_provider):
        """Test fallback raises error when all providers fail."""
        config = Mock()
        mock_get_config.return_value = config

        mock_provider = Mock()
        mock_provider.invoke.side_effect = ProviderError("All failed")
        mock_create_provider.return_value = mock_provider

        with patch("app.config.get_config") as mock_app_config:
            mock_app_config.return_value = Mock(
                llm_fallback_chain=["openai"]
            )

            with pytest.raises(ProviderError, match="All LLM providers failed"):
                create_llm_with_fallback("openai", "Test prompt")


class TestProviderError:
    """Tests for ProviderError exception."""

    def test_provider_error_is_exception(self):
        """Test ProviderError is an Exception subclass."""
        error = ProviderError("Test error")
        assert isinstance(error, Exception)

    def test_provider_error_preserves_message(self):
        """Test ProviderError preserves error message."""
        message = "API connection failed"
        error = ProviderError(message)
        assert str(error) == message

    def test_provider_error_can_wrap_original_error(self):
        """Test ProviderError can wrap original exception."""
        original = ValueError("Original error")
        error = ProviderError("Wrapper message", cause=original)
        assert error.cause == original
