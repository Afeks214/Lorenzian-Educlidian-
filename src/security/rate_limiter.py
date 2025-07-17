"""
Enhanced rate limiting with granular control.
Supports per-user, per-IP, and per-endpoint rate limiting.
"""

import os
import time
import asyncio
from typing import Dict, Optional, Tuple, Callable
from collections import defaultdict
from datetime import datetime, timedelta
import redis.asyncio as aioredis
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

from src.monitoring.logger_config import get_logger
from src.monitoring.metrics_exporter import metrics_exporter

logger = get_logger(__name__)


class RateLimitRule:
    """Rate limit rule configuration."""
    
    def __init__(self, 
                 requests: int,
                 window_seconds: int,
                 burst_size: Optional[int] = None,
                 key_func: Optional[Callable] = None):
        """
        Initialize rate limit rule.
        
        Args:
            requests: Number of allowed requests
            window_seconds: Time window in seconds
            burst_size: Optional burst allowance
            key_func: Optional function to extract rate limit key
        """
        self.requests = requests
        self.window_seconds = window_seconds
        self.burst_size = burst_size or requests
        self.key_func = key_func or self._default_key_func
    
    @staticmethod
    def _default_key_func(request: Request) -> str:
        """Default key function using client IP."""
        return request.client.host if request.client else "unknown"


class RateLimiter:
    """
    Advanced rate limiter with Redis backend and multiple strategies.
    
    Supports:
    - Token bucket algorithm
    - Sliding window
    - Per-user and per-IP limits
    - Burst handling
    - Distributed rate limiting via Redis
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize rate limiter.
        
        Args:
            redis_url: Redis connection URL for distributed rate limiting
        """
        self.redis_url = redis_url
        self.redis_client: Optional[aioredis.Redis] = None
        self.local_buckets: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.rules: Dict[str, RateLimitRule] = {}
        
        # Default rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default rate limiting rules."""
        # Health endpoint - relaxed limits
        self.add_rule("health", RateLimitRule(
            requests=10,
            window_seconds=60,
            burst_size=20
        ))
        
        # Decision endpoint - strict limits
        self.add_rule("decide", RateLimitRule(
            requests=100,
            window_seconds=60,
            burst_size=10
        ))
        
        # Metrics endpoint - moderate limits
        self.add_rule("metrics", RateLimitRule(
            requests=30,
            window_seconds=60,
            burst_size=5
        ))
        
        # Global per-IP limit with DDoS protection
        self.add_rule("global_ip", RateLimitRule(
            requests=500,  # Reduced for DDoS protection
            window_seconds=60,
            burst_size=50   # Reduced burst size
        ))
        
        # DDoS protection - very restrictive
        self.add_rule("ddos_protection", RateLimitRule(
            requests=50,
            window_seconds=10,
            burst_size=10
        ))
        
        # Per-user limit (requires user ID in key func)
        self.add_rule("global_user", RateLimitRule(
            requests=5000,
            window_seconds=60,
            burst_size=500,
            key_func=self._user_key_func
        ))
    
    @staticmethod
    def _user_key_func(request: Request) -> str:
        """Extract user ID from request for rate limiting."""
        # In real implementation, extract from JWT token
        return getattr(request.state, "user_id", "anonymous")
    
    async def initialize(self):
        """Initialize Redis connection for distributed rate limiting."""
        if self.redis_url:
            try:
                self.redis_client = await aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                logger.info("Redis rate limiter initialized")
            except Exception as e:
                logger.error("Failed to connect to Redis for rate limiting", error=str(e))
                # Fall back to local rate limiting
                self.redis_client = None
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
    
    def add_rule(self, rule_name: str, rule: RateLimitRule):
        """Add a rate limiting rule."""
        self.rules[rule_name] = rule
        logger.info(
            "Rate limit rule added",
            rule_name=rule_name,
            requests=rule.requests,
            window=rule.window_seconds
        )
    
    async def check_rate_limit(self,
                             rule_name: str,
                             request: Request) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limits.
        
        Args:
            rule_name: Name of the rule to apply
            request: FastAPI request object
            
        Returns:
            Tuple of (allowed, metadata)
            metadata includes: limit, remaining, reset_time
        """
        if rule_name not in self.rules:
            logger.warning("Unknown rate limit rule", rule_name=rule_name)
            return True, {}
        
        rule = self.rules[rule_name]
        key = f"rate_limit:{rule_name}:{rule.key_func(request)}"
        
        # Use Redis if available, otherwise local
        if self.redis_client:
            return await self._check_redis_limit(key, rule)
        else:
            return await self._check_local_limit(key, rule)
    
    async def _check_redis_limit(self, 
                               key: str,
                               rule: RateLimitRule) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit using Redis backend."""
        try:
            current_time = time.time()
            window_start = current_time - rule.window_seconds
            
            # Use Redis sorted set for sliding window
            pipe = self.redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests in window
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(current_time): current_time})
            
            # Set expiry
            pipe.expire(key, rule.window_seconds + 1)
            
            # Execute pipeline
            results = await pipe.execute()
            current_count = results[1]
            
            # Check if within limits
            allowed = current_count < rule.requests
            
            metadata = {
                "limit": rule.requests,
                "remaining": max(0, rule.requests - current_count - 1),
                "reset": int(current_time + rule.window_seconds),
                "retry_after": rule.window_seconds if not allowed else None
            }
            
            # Update metrics
            metrics_exporter.record_rate_limit(
                endpoint=key.split(":")[1],
                allowed=allowed
            )
            
            return allowed, metadata
            
        except Exception as e:
            logger.error("Redis rate limit check failed", error=str(e))
            # Fail open - allow request on Redis failure
            return True, {}
    
    async def _check_local_limit(self,
                               key: str,
                               rule: RateLimitRule) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit using local memory."""
        current_time = time.time()
        
        # Initialize bucket if not exists
        if key not in self.local_buckets:
            self.local_buckets[key] = {
                "tokens": float(rule.burst_size),
                "last_update": current_time
            }
        
        bucket = self.local_buckets[key]
        
        # Calculate tokens to add based on time passed
        time_passed = current_time - bucket["last_update"]
        tokens_to_add = time_passed * (rule.requests / rule.window_seconds)
        
        # Update bucket
        bucket["tokens"] = min(rule.burst_size, bucket["tokens"] + tokens_to_add)
        bucket["last_update"] = current_time
        
        # Check if request allowed
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            allowed = True
        else:
            allowed = False
        
        # Calculate metadata
        tokens_per_second = rule.requests / rule.window_seconds
        seconds_until_token = (1 - bucket["tokens"]) / tokens_per_second if bucket["tokens"] < 1 else 0
        
        metadata = {
            "limit": rule.requests,
            "remaining": int(bucket["tokens"]),
            "reset": int(current_time + seconds_until_token),
            "retry_after": int(seconds_until_token) if not allowed else None
        }
        
        return allowed, metadata
    
    async def rate_limit_middleware(self, request: Request, call_next):
        """
        FastAPI middleware for rate limiting.
        
        Args:
            request: FastAPI request
            call_next: Next middleware/handler
            
        Returns:
            Response or rate limit error
        """
        # Extract endpoint from path
        path = request.url.path.strip("/").split("/")[0]
        
        # SECURITY: DDoS protection - check aggressive rate limiting first
        ddos_allowed, ddos_metadata = await self.check_rate_limit("ddos_protection", request)
        if not ddos_allowed:
            return self._rate_limit_exceeded_response(ddos_metadata)
        
        # Check endpoint-specific limit
        if path in self.rules:
            allowed, metadata = await self.check_rate_limit(path, request)
            if not allowed:
                return self._rate_limit_exceeded_response(metadata)
        
        # Check global limits
        for rule_name in ["global_ip", "global_user"]:
            if rule_name in self.rules:
                allowed, metadata = await self.check_rate_limit(rule_name, request)
                if not allowed:
                    return self._rate_limit_exceeded_response(metadata)
        
        # Add rate limit headers to response
        response = await call_next(request)
        
        # Add rate limit headers
        if "metadata" in locals():
            response.headers["X-RateLimit-Limit"] = str(metadata.get("limit", ""))
            response.headers["X-RateLimit-Remaining"] = str(metadata.get("remaining", ""))
            response.headers["X-RateLimit-Reset"] = str(metadata.get("reset", ""))
        
        return response
    
    def _rate_limit_exceeded_response(self, metadata: Dict[str, Any]) -> JSONResponse:
        """Create rate limit exceeded response."""
        logger.warning("Rate limit exceeded", metadata=metadata)
        
        response = JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "message": f"Too many requests. Please retry after {metadata.get('retry_after', 60)} seconds",
                "retry_after": metadata.get("retry_after")
            }
        )
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(metadata.get("limit", ""))
        response.headers["X-RateLimit-Remaining"] = "0"
        response.headers["X-RateLimit-Reset"] = str(metadata.get("reset", ""))
        response.headers["Retry-After"] = str(metadata.get("retry_after", ""))
        
        return response


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


async def get_rate_limiter() -> RateLimiter:
    """Get or create global rate limiter instance."""
    global _rate_limiter
    
    if _rate_limiter is None:
        redis_url = os.getenv("REDIS_URL")
        _rate_limiter = RateLimiter(redis_url)
        await _rate_limiter.initialize()
    
    return _rate_limiter


def create_rate_limit_dependency(rule_name: str):
    """
    Create a FastAPI dependency for rate limiting.
    
    Args:
        rule_name: Name of the rate limit rule
        
    Returns:
        FastAPI dependency function
    """
    async def rate_limit_check(request: Request):
        limiter = await get_rate_limiter()
        allowed, metadata = await limiter.check_rate_limit(rule_name, request)
        
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry after {metadata.get('retry_after', 60)} seconds",
                headers={
                    "X-RateLimit-Limit": str(metadata.get("limit", "")),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(metadata.get("reset", "")),
                    "Retry-After": str(metadata.get("retry_after", ""))
                }
            )
    
    return rate_limit_check