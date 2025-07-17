"""
Circuit Breaker Implementation for Tactical Service

Implements automated failure detection and recovery with exponential backoff
and P0 alert escalation when restart limits are exceeded.
"""

import asyncio
import logging
import time
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failure detected, service unavailable
    HALF_OPEN = "half_open"  # Testing if service has recovered

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    max_failures: int = 3
    timeout_seconds: int = 300  # 5 minutes
    backoff_base: float = 2.0
    alert_on_limit: bool = True
    redis_key_prefix: str = "tactical:circuit_breaker"

class TacticalCircuitBreaker:
    """
    Circuit breaker for tactical service with Redis-backed state persistence.
    
    Handles:
    - Failure counting and threshold detection
    - Exponential backoff for restart attempts
    - P0 alert escalation when restart limits exceeded
    - State persistence across service restarts
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """Initialize circuit breaker."""
        self.config = config or CircuitBreakerConfig()
        self._load_config_from_env()
        
        self.redis_client: Optional[redis.Redis] = None
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.next_retry_time = 0.0
        
        # Redis keys for state persistence
        self.state_key = f"{self.config.redis_key_prefix}:state"
        self.failure_count_key = f"{self.config.redis_key_prefix}:failure_count"
        self.last_failure_key = f"{self.config.redis_key_prefix}:last_failure"
        self.next_retry_key = f"{self.config.redis_key_prefix}:next_retry"
        
        logger.info(f"Circuit breaker initialized with config: {self.config}")
    
    def _load_config_from_env(self):
        """Load configuration from environment variables."""
        self.config.max_failures = int(os.getenv("TACTICAL_RESTART_MAX_ATTEMPTS", self.config.max_failures))
        self.config.timeout_seconds = int(os.getenv("TACTICAL_CIRCUIT_BREAKER_TIMEOUT", self.config.timeout_seconds))
        self.config.backoff_base = float(os.getenv("TACTICAL_RESTART_BACKOFF_BASE", self.config.backoff_base))
        self.config.alert_on_limit = os.getenv("TACTICAL_ALERT_ON_RESTART_LIMIT", "true").lower() == "true"
    
    async def initialize(self, redis_url: str = "redis://localhost:6379/2"):
        """Initialize Redis connection and restore state."""
        self.redis_client = redis.from_url(redis_url)
        await self.redis_client.ping()
        
        # Restore state from Redis
        await self._restore_state()
        
        logger.info(f"Circuit breaker initialized with state: {self.state}")
    
    async def _restore_state(self):
        """Restore circuit breaker state from Redis."""
        try:
            # Get persisted state
            state_data = await self.redis_client.hmget(
                self.state_key,
                ["state", "failure_count", "last_failure", "next_retry"]
            )
            
            if state_data[0]:  # state exists
                self.state = CircuitBreakerState(state_data[0].decode())
                self.failure_count = int(state_data[1] or 0)
                self.last_failure_time = float(state_data[2] or 0.0)
                self.next_retry_time = float(state_data[3] or 0.0)
                
                logger.info(f"Restored circuit breaker state: {self.state}, failures: {self.failure_count}")
            else:
                logger.info("No previous circuit breaker state found, starting fresh")
                
        except Exception as e:
            logger.error(f"Failed to restore circuit breaker state: {e}")
            # Start with default state
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
    
    async def _persist_state(self):
        """Persist circuit breaker state to Redis."""
        try:
            await self.redis_client.hmset(
                self.state_key,
                {
                    "state": self.state.value,
                    "failure_count": self.failure_count,
                    "last_failure": self.last_failure_time,
                    "next_retry": self.next_retry_time
                }
            )
            
            # Set expiration to prevent stale state
            await self.redis_client.expire(self.state_key, self.config.timeout_seconds * 2)
            
        except Exception as e:
            logger.error(f"Failed to persist circuit breaker state: {e}")
    
    async def record_success(self):
        """Record successful operation."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            # Service has recovered, reset to normal
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.last_failure_time = 0.0
            self.next_retry_time = 0.0
            
            await self._persist_state()
            logger.info("Circuit breaker reset to CLOSED after successful recovery")
        
        elif self.state == CircuitBreakerState.CLOSED:
            # Normal operation continues
            pass
    
    async def record_failure(self, error: Exception):
        """Record failure and update circuit breaker state."""
        current_time = time.time()
        self.failure_count += 1
        self.last_failure_time = current_time
        
        logger.warning(f"Circuit breaker recorded failure {self.failure_count}/{self.config.max_failures}: {error}")
        
        if self.failure_count >= self.config.max_failures:
            # Open circuit breaker
            self.state = CircuitBreakerState.OPEN
            
            # Calculate next retry time with exponential backoff
            backoff_seconds = self.config.backoff_base ** self.failure_count
            self.next_retry_time = current_time + backoff_seconds
            
            logger.error(f"Circuit breaker OPENED after {self.failure_count} failures. Next retry in {backoff_seconds}s")
            
            # Trigger P0 alert if configured
            if self.config.alert_on_limit:
                await self._trigger_p0_alert()
        
        await self._persist_state()
    
    async def can_execute(self) -> bool:
        """Check if operation can be executed based on circuit breaker state."""
        current_time = time.time()
        
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        elif self.state == CircuitBreakerState.OPEN:
            if current_time >= self.next_retry_time:
                # Try to recover
                self.state = CircuitBreakerState.HALF_OPEN
                await self._persist_state()
                logger.info("Circuit breaker moved to HALF_OPEN for recovery test")
                return True
            else:
                # Still in timeout
                remaining = self.next_retry_time - current_time
                logger.warning(f"Circuit breaker OPEN: {remaining:.1f}s remaining until retry")
                return False
        
        elif self.state == CircuitBreakerState.HALF_OPEN:
            # Allow single test operation
            return True
        
        return False
    
    async def _trigger_p0_alert(self):
        """Trigger P0 alert for restart limit exceeded."""
        try:
            alert_data = {
                "alert_type": "P0_CRITICAL",
                "service": "tactical-marl",
                "message": f"Circuit breaker OPEN: {self.failure_count} consecutive failures exceeded limit",
                "failure_count": self.failure_count,
                "max_failures": self.config.max_failures,
                "last_failure_time": self.last_failure_time,
                "next_retry_time": self.next_retry_time,
                "timestamp": time.time(),
                "escalation_required": True
            }
            
            # Publish alert to Redis stream
            await self.redis_client.xadd(
                "tactical_alerts",
                alert_data
            )
            
            logger.critical(f"P0 ALERT TRIGGERED: Circuit breaker opened after {self.failure_count} failures")
            
        except Exception as e:
            logger.error(f"Failed to trigger P0 alert: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "max_failures": self.config.max_failures,
            "last_failure_time": self.last_failure_time,
            "next_retry_time": self.next_retry_time,
            "can_execute": await self.can_execute(),
            "config": {
                "max_failures": self.config.max_failures,
                "timeout_seconds": self.config.timeout_seconds,
                "backoff_base": self.config.backoff_base,
                "alert_on_limit": self.config.alert_on_limit
            }
        }
    
    async def reset(self):
        """Reset circuit breaker to closed state (manual intervention)."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.next_retry_time = 0.0
        
        await self._persist_state()
        logger.info("Circuit breaker manually reset to CLOSED state")
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()