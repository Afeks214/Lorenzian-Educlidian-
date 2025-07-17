"""
Redis compatibility layer for Python 3.12 and aioredis.
Provides a compatible interface that works around Python 3.12 TimeoutError issues.
"""

import asyncio
import sys
from typing import Any, Optional, Dict, List, Union

# Try to import aioredis with compatibility handling
try:
    import aioredis
    # Check if we're on Python 3.12+ and need to handle TimeoutError
    if sys.version_info >= (3, 12):
        # Create a compatibility wrapper
        original_timeout_error = aioredis.exceptions.TimeoutError
        
        class CompatTimeoutError(asyncio.TimeoutError):
            """Compatibility TimeoutError for Python 3.12+"""
            pass
        
        # Replace the problematic TimeoutError
        aioredis.exceptions.TimeoutError = CompatTimeoutError
        aioredis.TimeoutError = CompatTimeoutError
        
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None

class RedisClient:
    """Redis client wrapper with compatibility handling."""
    
    def __init__(self, url: str = "redis://localhost:6379", **kwargs):
        self.url = url
        self.kwargs = kwargs
        self._client = None
        self._available = REDIS_AVAILABLE
        
    async def connect(self):
        """Connect to Redis with error handling."""
        if not self._available:
            return None
            
        try:
            self._client = await aioredis.from_url(self.url, **self.kwargs)
            return self._client
        except Exception as e:
            print(f"Redis connection failed: {e}")
            self._available = False
            return None
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self._client:
            await self._client.close()
            self._client = None
    
    async def set(self, key: str, value: Any, **kwargs):
        """Set a key-value pair."""
        if not self._client:
            return False
        try:
            await self._client.set(key, value, **kwargs)
            return True
        except Exception:
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value by key."""
        if not self._client:
            return None
        try:
            return await self._client.get(key)
        except Exception:
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete a key."""
        if not self._client:
            return False
        try:
            result = await self._client.delete(key)
            return result > 0
        except Exception:
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        if not self._client:
            return False
        try:
            return await self._client.exists(key) > 0
        except Exception:
            return False
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching a pattern."""
        if not self._client:
            return []
        try:
            return await self._client.keys(pattern)
        except Exception:
            return []
    
    async def ping(self) -> bool:
        """Ping Redis to check connectivity."""
        if not self._client:
            return False
        try:
            await self._client.ping()
            return True
        except Exception:
            return False
    
    @property
    def available(self) -> bool:
        """Check if Redis is available."""
        return self._available and self._client is not None
    
    async def publish(self, channel: str, message: str):
        """Publish a message to a channel."""
        if not self._client:
            return False
        try:
            await self._client.publish(channel, message)
            return True
        except Exception:
            return False
    
    async def setex(self, key: str, seconds: int, value: Any):
        """Set a key with expiration."""
        if not self._client:
            return False
        try:
            await self._client.setex(key, seconds, value)
            return True
        except Exception:
            return False
    
    async def lpush(self, key: str, value: Any):
        """Push value to the left of a list."""
        if not self._client:
            return False
        try:
            await self._client.lpush(key, value)
            return True
        except Exception:
            return False
    
    async def ltrim(self, key: str, start: int, end: int):
        """Trim a list to specified range."""
        if not self._client:
            return False
        try:
            await self._client.ltrim(key, start, end)
            return True
        except Exception:
            return False
    
    async def lrange(self, key: str, start: int, end: int) -> List[Any]:
        """Get a range from a list."""
        if not self._client:
            return []
        try:
            return await self._client.lrange(key, start, end)
        except Exception:
            return []
    
    async def scan_iter(self, match: str = "*"):
        """Iterate over keys matching a pattern."""
        if not self._client:
            return []
        try:
            return await self._client.scan_iter(match=match)
        except Exception:
            return []
    
    def pubsub(self):
        """Get a pubsub instance."""
        if not self._client:
            return MockPubSub()
        try:
            return self._client.pubsub()
        except Exception:
            return MockPubSub()
    
    def close(self):
        """Close the Redis connection."""
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
    
    async def wait_closed(self):
        """Wait for the connection to close."""
        if self._client:
            try:
                await self._client.wait_closed()
            except Exception:
                pass

class MockPubSub:
    """Mock pub/sub for when Redis is not available."""
    
    def __init__(self):
        self.channels = {}
        self.closed = False
    
    async def subscribe(self, *channels):
        """Subscribe to channels."""
        for channel in channels:
            self.channels[channel] = True
    
    async def unsubscribe(self, *channels):
        """Unsubscribe from channels."""
        for channel in channels:
            self.channels.pop(channel, None)
    
    async def get_message(self, timeout=None):
        """Get a message (returns None for mock)."""
        return None
    
    async def close(self):
        """Close the pub/sub connection."""
        self.closed = True

# Global instance for easy access
redis_client = RedisClient()

def create_redis_pool(url: str, **kwargs):
    """Create a Redis connection pool (compatibility function)."""
    client = RedisClient(url, **kwargs)
    return client

async def create_redis_pool_async(url: str, **kwargs):
    """Create a Redis connection pool asynchronously."""
    client = RedisClient(url, **kwargs)
    await client.connect()
    return client