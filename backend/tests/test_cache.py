"""Tests for Redis caching implementation.

TDD approach: Tests written before implementation.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import timedelta

from app.cache.redis_cache import (
    RedisCache,
    CacheConfig,
    create_redis_cache,
    CacheError,
)
from app.config import get_config


@pytest.fixture
def redis_config():
    """Create Redis configuration for testing."""
    from app.cache.redis_cache import CacheConfig
    return CacheConfig(
        url="redis://localhost:6379/1",
        default_ttl=3600,
        max_connections=50,
    )


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis = Mock()
    redis.get = Mock(return_value=None)
    redis.set = Mock(return_value=True)
    redis.delete = Mock(return_value=1)
    redis.exists = Mock(return_value=1)
    redis.keys = Mock(return_value=[])
    redis.ping = Mock(return_value=True)
    return redis


class TestCacheConfig:
    """Tests for CacheConfig model."""

    def test_cache_config_creation(self):
        """Test CacheConfig creation with defaults."""
        from app.cache.redis_cache import CacheConfig
        config = CacheConfig(url="redis://localhost:6379/0")
        assert config.url == "redis://localhost:6379/0"
        assert config.default_ttl == 3600
        assert config.max_connections == 50

    def test_cache_config_with_custom_values(self):
        """Test CacheConfig with custom values."""
        from app.cache.redis_cache import CacheConfig
        config = CacheConfig(
            url="redis://custom:6380/2",
            default_ttl=7200,
            max_connections=100,
        )
        assert config.url == "redis://custom:6380/2"
        assert config.default_ttl == 7200
        assert config.max_connections == 100


class TestRedisCache:
    """Tests for RedisCache implementation."""

    @patch("app.cache.redis_cache.Redis")
    def test_redis_cache_initializes_with_config(self, mock_redis_class, redis_config):
        """Test RedisCache initializes with configuration."""
        mock_redis = Mock()
        mock_redis_class.return_value = mock_redis

        cache = RedisCache(redis_config)

        mock_redis_class.assert_called_once()
        assert cache._config == redis_config

    @patch("app.cache.redis_cache.Redis")
    def test_get_retrieves_value_from_cache(self, mock_redis_class, redis_config):
        """Test get retrieves value from cache."""
        mock_redis = Mock()
        mock_redis.get.return_value = b'"cached value"'
        mock_redis_class.return_value = mock_redis

        cache = RedisCache(redis_config)
        value = cache.get("test_key")

        mock_redis.get.assert_called_once_with("test_key")
        assert value == "cached value"

    @patch("app.cache.redis_cache.Redis")
    def test_get_returns_none_for_missing_key(self, mock_redis_class, redis_config):
        """Test get returns None when key doesn't exist."""
        mock_redis = Mock()
        mock_redis.get.return_value = None
        mock_redis_class.return_value = mock_redis

        cache = RedisCache(redis_config)
        value = cache.get("missing_key")

        assert value is None

    @patch("app.cache.redis_cache.Redis")
    def test_set_stores_value_with_default_ttl(self, mock_redis_class, redis_config):
        """Test set stores value with default TTL."""
        mock_redis = Mock()
        mock_redis.set.return_value = True
        mock_redis_class.return_value = mock_redis

        cache = RedisCache(redis_config)
        cache.set("test_key", {"data": "value"})

        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert call_args[0][0] == "test_key"
        assert call_args[1]["ex"] == redis_config.default_ttl

    @patch("app.cache.redis_mock.Redis")
    def test_set_stores_value_with_custom_ttl(self, mock_redis_class, redis_config):
        """Test set stores value with custom TTL."""
        mock_redis = Mock()
        mock_redis.set.return_value = True
        mock_redis_class.return_value = mock_redis

        cache = RedisCache(redis_config)
        cache.set("test_key", "value", ttl=7200)

        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert call_args[1]["ex"] == 7200

    @patch("app.cache.redis_cache.Redis")
    def test_delete_removes_key_from_cache(self, mock_redis_class, redis_config):
        """Test delete removes key from cache."""
        mock_redis = Mock()
        mock_redis.delete.return_value = 1
        mock_redis_class.return_value = mock_redis

        cache = RedisCache(redis_config)
        result = cache.delete("test_key")

        mock_redis.delete.assert_called_once_with("test_key")
        assert result is True

    @patch("app.cache.redis_cache.Redis")
    def test_exists_returns_true_for_existing_key(self, mock_redis_class, redis_config):
        """Test exists returns True when key exists."""
        mock_redis = Mock()
        mock_redis.exists.return_value = 1
        mock_redis_class.return_value = mock_redis

        cache = RedisCache(redis_config)
        result = cache.exists("test_key")

        assert result is True

    @patch("app.cache.redis_cache.Redis")
    def test_exists_returns_false_for_missing_key(self, mock_redis_class, redis_config):
        """Test exists returns False when key doesn't exist."""
        mock_redis = Mock()
        mock_redis.exists.return_value = 0
        mock_redis_class.return_value = mock_redis

        cache = RedisCache(redis_config)
        result = cache.exists("missing_key")

        assert result is False

    @patch("app.cache.redis_cache.Redis")
    def test_clear_pattern_deletes_matching_keys(self, mock_redis_class, redis_config):
        """Test clear_pattern deletes keys matching pattern."""
        mock_redis = Mock()
        mock_redis.keys.return_value = [b"key1", b"key2"]
        mock_redis.delete.return_value = 2
        mock_redis_class.return_value = mock_redis

        cache = RedisCache(redis_config)
        cache.clear_pattern("session:*")

        mock_redis.keys.assert_called_once_with("session:*")
        mock_redis.delete.assert_called_once()

    @patch("app.cache.redis_cache.Redis")
    def test_get_json_returns_parsed_object(self, mock_redis_class, redis_config):
        """Test get_json returns parsed JSON object."""
        mock_redis = Mock()
        mock_redis.get.return_value = b'{"key": "value"}'
        mock_redis_class.return_value = mock_redis

        cache = RedisCache(redis_config)
        value = cache.get_json("test_key")

        assert value == {"key": "value"}

    @patch("app.cache.redis_cache.Redis")
    def test_set_json_stores_serialized_object(self, mock_redis_class, redis_config):
        """Test set_json stores serialized JSON object."""
        mock_redis = Mock()
        mock_redis.set.return_value = True
        mock_redis_class.return_value = mock_redis

        cache = RedisCache(redis_config)
        cache.set_json("test_key", {"data": "complex"})

        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        stored_value = call_args[0][1]
        assert b'"data"' in stored_value
        assert b'"complex"' in stored_value

    @patch("app.cache.redis_cache.Redis")
    def test_ping_returns_true_when_connected(self, mock_redis_class, redis_config):
        """Test ping returns True when Redis is connected."""
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis_class.return_value = mock_redis

        cache = RedisCache(redis_config)
        result = cache.ping()

        assert result is True

    @patch("app.cache.redis_cache.Redis")
    def test_cache_llm_response_stores_llm_output(self, mock_redis_class, redis_config):
        """Test cache_llm_response stores LLM output with hash-based key."""
        mock_redis = Mock()
        mock_redis.set.return_value = True
        mock_redis_class.return_value = mock_redis

        cache = RedisCache(redis_config)
        cache.cache_llm_response("openai", "gpt-4", "test prompt", "response")

        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        key = call_args[0][0]
        assert "llm" in key
        assert "openai" in key

    @patch("app.cache.redis_cache.Redis")
    def test_get_cached_llm_response_retrieves_stored_response(self, mock_redis_class, redis_config):
        """Test get_cached_llm_response retrieves cached LLM response."""
        mock_redis = Mock()
        mock_redis.get.return_value = b'"cached response"'
        mock_redis_class.return_value = mock_redis

        cache = RedisCache(redis_config)
        response = cache.get_cached_llm_response("openai", "gpt-4", "test prompt")

        mock_redis.get.assert_called_once()
        assert response == "cached response"


class TestCacheError:
    """Tests for CacheError exception."""

    def test_cache_error_is_exception(self):
        """Test CacheError is an Exception subclass."""
        error = CacheError("Cache operation failed")
        assert isinstance(error, Exception)

    def test_cache_error_preserves_message(self):
        """Test CacheError preserves error message."""
        message = "Redis connection timeout"
        error = CacheError(message)
        assert str(error) == message


class TestCreateRedisCache:
    """Tests for create_redis_cache factory function."""

    @patch("app.cache.redis_cache.RedisCache")
    @patch("app.cache.redis_cache.get_config")
    def test_create_uses_app_config(self, mock_get_config, mock_cache_class):
        """Test factory uses app configuration."""
        config = Mock(
            redis=Mock(
                url="redis://app:6379/0",
                default_ttl=1800,
                max_connections=25,
            )
        )
        mock_get_config.return_value = config

        mock_instance = Mock()
        mock_cache_class.return_value = mock_instance

        cache = create_redis_cache()

        mock_cache_class.assert_called_once()
