"""Redis caching layer for LLM responses and session state.

Provides high-performance caching for:
- LLM responses (avoiding redundant API calls)
- Session state (faster agent resumption)
- Intermediate computation results

All cache operations are thread-safe and handle connection failures gracefully.
"""
from dataclasses import dataclass, field
from hashlib import sha256
import json
import logging
from typing import Any, Optional

try:
    from redis import Redis
    from redis.exceptions import RedisError
except ImportError:
    Redis = None
    RedisError = Exception

from app.config import get_config

logger = logging.getLogger(__name__)


class CacheError(Exception):
    """Exception raised when cache operations fail."""

    pass


@dataclass(frozen=True)
class CacheConfig:
    """Configuration for Redis cache.

    Attributes:
        url: Redis connection URL
        default_ttl: Default time-to-live for cache entries (seconds)
        max_connections: Maximum number of connections in pool
        socket_timeout: Socket timeout for operations
    """

    url: str
    default_ttl: int = 3600
    max_connections: int = 50
    socket_timeout: int = 5


class RedisCache:
    """Redis-based cache implementation.

    Provides caching for LLM responses and session state with
    automatic serialization and TTL management.

    Example:
        >>> cache = RedisCache(config)
        >>> cache.set("key", {"data": "value"})
        >>> value = cache.get("key")
        >>> cache.delete("key")
    """

    def __init__(self, config: CacheConfig) -> None:
        """Initialize the Redis cache.

        Args:
            config: Cache configuration

        Raises:
            ImportError: If redis package is not installed
        """
        if Redis is None:
            raise ImportError(
                "redis package is required. Install with: pip install redis"
            )

        self._config = config

        # Create Redis connection pool
        self._redis = Redis.from_url(
            config.url,
            max_connections=config.max_connections,
            socket_timeout=config.socket_timeout,
            decode_responses=False,  # We handle encoding ourselves
        )

        logger.info(f"RedisCache initialized with URL: {config.url}")

    def get(self, key: str) -> Any | None:
        """Get a value from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found

        Raises:
            CacheError: If Redis operation fails
        """
        try:
            value = self._redis.get(key)
            if value is None:
                return None

            return json.loads(value)
        except RedisError as e:
            logger.error(f"Redis get failed: {e}")
            raise CacheError(f"Cache get failed: {e}") from e
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode cached value: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set a value in the cache.

        Args:
            key: Cache key
            value: Value to cache (must be JSON-serializable)
            ttl: Time-to-live in seconds (uses default if not specified)

        Returns:
            True if successful, False otherwise

        Raises:
            CacheError: If Redis operation fails
        """
        try:
            ttl = ttl or self._config.default_ttl
            serialized = json.dumps(value)

            self._redis.set(key, serialized, ex=ttl)
            return True
        except RedisError as e:
            logger.error(f"Redis set failed: {e}")
            raise CacheError(f"Cache set failed: {e}") from e
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize value: {e}")
            raise CacheError(f"Value serialization failed: {e}") from e

    def delete(self, key: str) -> bool:
        """Delete a value from the cache.

        Args:
            key: Cache key

        Returns:
            True if key was deleted, False otherwise

        Raises:
            CacheError: If Redis operation fails
        """
        try:
            result = self._redis.delete(key)
            return result > 0
        except RedisError as e:
            logger.error(f"Redis delete failed: {e}")
            raise CacheError(f"Cache delete failed: {e}") from e

    def exists(self, key: str) -> bool:
        """Check if a key exists in the cache.

        Args:
            key: Cache key

        Returns:
            True if key exists, False otherwise

        Raises:
            CacheError: If Redis operation fails
        """
        try:
            return self._redis.exists(key) > 0
        except RedisError as e:
            logger.error(f"Redis exists failed: {e}")
            raise CacheError(f"Cache exists check failed: {e}") from e

    def clear_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern.

        Args:
            pattern: Key pattern (e.g., "session:*")

        Returns:
            Number of keys deleted

        Raises:
            CacheError: If Redis operation fails
        """
        try:
            keys = self._redis.keys(pattern)
            if keys:
                return self._redis.delete(*keys)
            return 0
        except RedisError as e:
            logger.error(f"Redis clear_pattern failed: {e}")
            raise CacheError(f"Cache clear pattern failed: {e}") from e

    def get_json(self, key: str) -> dict | None:
        """Get a JSON object from the cache.

        Convenience method that ensures the return type is a dict.

        Args:
            key: Cache key

        Returns:
            Dict or None if not found
        """
        value = self.get(key)
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        return None

    def set_json(self, key: str, value: dict, ttl: int | None = None) -> bool:
        """Store a JSON object in the cache.

        Convenience method for storing dict values.

        Args:
            key: Cache key
            value: Dict to store
            ttl: Optional TTL override

        Returns:
            True if successful
        """
        return self.set(key, value, ttl)

    def ping(self) -> bool:
        """Check if Redis is reachable.

        Returns:
            True if Redis responds to PING
        """
        try:
            return self._redis.ping()
        except RedisError:
            return False

    def cache_llm_response(
        self,
        provider: str,
        model: str,
        prompt: str,
        response: str,
        ttl: int = 86400,
    ) -> bool:
        """Cache an LLM response.

        Creates a deterministic key based on provider, model, and prompt hash.

        Args:
            provider: LLM provider name
            model: Model identifier
            prompt: The prompt that generated the response
            response: The LLM response to cache
            ttl: Time-to-live (default 24 hours)

        Returns:
            True if cached successfully
        """
        # Create deterministic key from inputs
        key = self._llm_cache_key(provider, model, prompt)
        return self.set(key, {"response": response}, ttl)

    def get_cached_llm_response(
        self, provider: str, model: str, prompt: str
    ) -> str | None:
        """Get a cached LLM response.

        Args:
            provider: LLM provider name
            model: Model identifier
            prompt: The prompt to look up

        Returns:
            Cached response or None if not found
        """
        key = self._llm_cache_key(provider, model, prompt)
        cached = self.get(key)
        return cached["response"] if cached else None

    def _llm_cache_key(self, provider: str, model: str, prompt: str) -> str:
        """Generate a cache key for LLM responses.

        Uses SHA256 hash of the prompt to create a deterministic key.

        Args:
            provider: LLM provider name
            model: Model identifier
            prompt: The prompt text

        Returns:
            Cache key string
        """
        # Hash the prompt for consistent key length
        prompt_hash = sha256(prompt.encode()).hexdigest()[:16]
        return f"llm:{provider}:{model}:{prompt_hash}"

    def close(self) -> None:
        """Close the Redis connection."""
        self._redis.close()
        logger.info("RedisCache connection closed")


def create_redis_cache(config: CacheConfig | None = None) -> RedisCache:
    """Factory function to create a Redis cache instance.

    Args:
        config: Optional cache configuration. Uses app config if not provided.

    Returns:
        Configured RedisCache instance
    """
    if config is None:
        app_config = get_config()
        config = CacheConfig(
            url=app_config.redis.url,
            default_ttl=app_config.redis.default_ttl,
            max_connections=app_config.redis.max_connections,
        )

    return RedisCache(config)
