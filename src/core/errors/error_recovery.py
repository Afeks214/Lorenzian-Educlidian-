"""
Error Recovery and Graceful Degradation System

Provides comprehensive error recovery mechanisms including circuit breakers,
fallback strategies, and graceful degradation to maintain system resilience.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, deque

from .base_exceptions import (
    BaseGrandModelError, ErrorSeverity, ErrorCategory, 
    CircuitBreakerError, SystemError, DependencyError
)

logger = logging.getLogger(__name__)


class RecoveryAction(Enum):
    """Types of recovery actions."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    SYSTEM_RESTART = "system_restart"
    ESCALATE = "escalate"
    IGNORE = "ignore"


class SystemHealth(Enum):
    """System health states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILING = "failing"


@dataclass
class RecoveryConfig:
    """Configuration for recovery strategies."""
    max_retries: int = 3
    retry_delay: float = 1.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 30.0
    enable_fallback: bool = True
    enable_graceful_degradation: bool = True
    health_check_interval: float = 60.0
    recovery_timeout: float = 300.0


class RecoveryStrategy(ABC):
    """Abstract base class for recovery strategies."""
    
    @abstractmethod
    def can_recover(self, error: BaseGrandModelError) -> bool:
        """Check if this strategy can recover from the error."""
        pass
    
    @abstractmethod
    def recover(self, error: BaseGrandModelError, context: Dict[str, Any]) -> Any:
        """Attempt recovery from the error."""
        pass
    
    @abstractmethod
    def get_priority(self) -> int:
        """Get priority of this strategy (lower = higher priority)."""
        pass


class RetryStrategy(RecoveryStrategy):
    """Retry recovery strategy with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.retry_counts = defaultdict(int)
    
    def can_recover(self, error: BaseGrandModelError) -> bool:
        """Check if error is retryable."""
        return (
            error.recoverable and 
            error.category in [ErrorCategory.NETWORK, ErrorCategory.TIMEOUT, ErrorCategory.DEPENDENCY] and
            self.retry_counts[error.correlation_id] < self.max_retries
        )
    
    def recover(self, error: BaseGrandModelError, context: Dict[str, Any]) -> Any:
        """Retry the failed operation."""
        retry_count = self.retry_counts[error.correlation_id]
        self.retry_counts[error.correlation_id] += 1
        
        # Calculate delay with exponential backoff
        delay = self.base_delay * (2 ** retry_count)
        
        logger.info(
            f"Retrying operation after {delay:.2f}s (attempt {retry_count + 1}/{self.max_retries})",
            extra={
                'correlation_id': error.correlation_id,
                'retry_count': retry_count + 1,
                'max_retries': self.max_retries,
                'delay': delay
            }
        )
        
        time.sleep(delay)
        
        # Return indication to retry
        return {'action': 'retry', 'delay': delay}
    
    def get_priority(self) -> int:
        return 1


class FallbackStrategy(RecoveryStrategy):
    """Fallback recovery strategy."""
    
    def __init__(self):
        self.fallback_handlers: Dict[str, Callable] = {}
    
    def register_fallback(self, error_type: str, handler: Callable):
        """Register fallback handler for error type."""
        self.fallback_handlers[error_type] = handler
    
    def can_recover(self, error: BaseGrandModelError) -> bool:
        """Check if fallback is available."""
        return (
            error.recoverable and
            (error.category.value in self.fallback_handlers or 
             type(error).__name__ in self.fallback_handlers)
        )
    
    def recover(self, error: BaseGrandModelError, context: Dict[str, Any]) -> Any:
        """Execute fallback handler."""
        handler = None
        
        # Try specific error type first
        if type(error).__name__ in self.fallback_handlers:
            handler = self.fallback_handlers[type(error).__name__]
        elif error.category.value in self.fallback_handlers:
            handler = self.fallback_handlers[error.category.value]
        
        if handler:
            logger.info(
                f"Executing fallback for {type(error).__name__}",
                extra={
                    'correlation_id': error.correlation_id,
                    'error_type': type(error).__name__,
                    'category': error.category.value
                }
            )
            
            try:
                return handler(error, context)
            except Exception as fallback_error:
                logger.error(f"Fallback failed: {fallback_error}")
                raise fallback_error
        
        return None
    
    def get_priority(self) -> int:
        return 2


class CircuitBreakerStrategy(RecoveryStrategy):
    """Circuit breaker recovery strategy."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_counts = defaultdict(int)
        self.circuit_states = defaultdict(lambda: 'closed')
        self.last_failure_times = defaultdict(float)
    
    def can_recover(self, error: BaseGrandModelError) -> bool:
        """Check if circuit breaker should activate."""
        service_key = self._get_service_key(error)
        
        return (
            error.category in [ErrorCategory.DEPENDENCY, ErrorCategory.NETWORK] and
            self.failure_counts[service_key] >= self.failure_threshold
        )
    
    def recover(self, error: BaseGrandModelError, context: Dict[str, Any]) -> Any:
        """Activate circuit breaker."""
        service_key = self._get_service_key(error)
        current_time = time.time()
        
        # Update failure count
        self.failure_counts[service_key] += 1
        self.last_failure_times[service_key] = current_time
        
        # Check if should open circuit
        if self.failure_counts[service_key] >= self.failure_threshold:
            self.circuit_states[service_key] = 'open'
            
            logger.warning(
                f"Circuit breaker opened for {service_key}",
                extra={
                    'service_key': service_key,
                    'failure_count': self.failure_counts[service_key],
                    'threshold': self.failure_threshold
                }
            )
            
            return {
                'action': 'circuit_breaker',
                'state': 'open',
                'service_key': service_key,
                'retry_after': self.timeout
            }
        
        return None
    
    def _get_service_key(self, error: BaseGrandModelError) -> str:
        """Get service key from error."""
        return error.error_details.get('service_name', 'unknown_service')
    
    def get_priority(self) -> int:
        return 3


class GracefulDegradationStrategy(RecoveryStrategy):
    """Graceful degradation recovery strategy."""
    
    def __init__(self):
        self.degradation_levels = {
            ErrorSeverity.CRITICAL: 0.2,  # 20% functionality
            ErrorSeverity.HIGH: 0.5,      # 50% functionality
            ErrorSeverity.MEDIUM: 0.8,    # 80% functionality
            ErrorSeverity.LOW: 0.9        # 90% functionality
        }
    
    def can_recover(self, error: BaseGrandModelError) -> bool:
        """Check if graceful degradation is applicable."""
        return (
            error.recoverable and
            error.severity in self.degradation_levels
        )
    
    def recover(self, error: BaseGrandModelError, context: Dict[str, Any]) -> Any:
        """Apply graceful degradation."""
        degradation_level = self.degradation_levels[error.severity]
        
        logger.info(
            f"Applying graceful degradation at {degradation_level*100:.0f}% functionality",
            extra={
                'correlation_id': error.correlation_id,
                'degradation_level': degradation_level,
                'error_severity': error.severity.value
            }
        )
        
        return {
            'action': 'graceful_degradation',
            'functionality_level': degradation_level,
            'limited_features': self._get_limited_features(error),
            'alternative_actions': self._get_alternative_actions(error)
        }
    
    def _get_limited_features(self, error: BaseGrandModelError) -> List[str]:
        """Get list of features to limit."""
        # Implementation depends on specific system
        return ['non_critical_features', 'background_tasks']
    
    def _get_alternative_actions(self, error: BaseGrandModelError) -> List[str]:
        """Get alternative actions to suggest."""
        return ['use_cached_data', 'defer_processing', 'manual_intervention']
    
    def get_priority(self) -> int:
        return 4


class SystemHealthChecker:
    """Monitors system health and triggers recovery actions."""
    
    def __init__(self, config: RecoveryConfig):
        self.config = config
        self.health_status = SystemHealth.HEALTHY
        self.service_health: Dict[str, SystemHealth] = {}
        self.health_history: deque = deque(maxlen=100)
        self.running = False
        self.check_thread = None
        self._lock = threading.Lock()
    
    def start_monitoring(self):
        """Start health monitoring."""
        with self._lock:
            if not self.running:
                self.running = True
                self.check_thread = threading.Thread(target=self._monitoring_loop)
                self.check_thread.daemon = True
                self.check_thread.start()
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        with self._lock:
            self.running = False
            if self.check_thread:
                self.check_thread.join()
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._check_system_health()
                time.sleep(self.config.health_check_interval)
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                time.sleep(self.config.health_check_interval)
    
    def _check_system_health(self):
        """Check overall system health."""
        with self._lock:
            # Check individual services
            overall_health = SystemHealth.HEALTHY
            
            for service, health in self.service_health.items():
                if health == SystemHealth.FAILING:
                    overall_health = SystemHealth.FAILING
                    break
                elif health == SystemHealth.CRITICAL:
                    overall_health = SystemHealth.CRITICAL
                elif health == SystemHealth.DEGRADED and overall_health == SystemHealth.HEALTHY:
                    overall_health = SystemHealth.DEGRADED
            
            # Update health status
            if self.health_status != overall_health:
                logger.info(f"System health changed from {self.health_status.value} to {overall_health.value}")
                self.health_status = overall_health
            
            # Record health history
            self.health_history.append({
                'timestamp': time.time(),
                'health_status': overall_health.value,
                'service_health': dict(self.service_health)
            })
    
    def report_service_health(self, service_name: str, health: SystemHealth):
        """Report health status of a service."""
        with self._lock:
            self.service_health[service_name] = health
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        with self._lock:
            return {
                'overall_health': self.health_status.value,
                'service_health': dict(self.service_health),
                'last_check': time.time()
            }


class ServiceHealthTracker:
    """Tracks health of individual services."""
    
    def __init__(self, service_name: str, health_checker: SystemHealthChecker):
        self.service_name = service_name
        self.health_checker = health_checker
        self.error_count = 0
        self.success_count = 0
        self.last_error_time = 0.0
        self.health_status = SystemHealth.HEALTHY
    
    def record_success(self):
        """Record successful operation."""
        self.success_count += 1
        self._update_health()
    
    def record_error(self, error: BaseGrandModelError):
        """Record error operation."""
        self.error_count += 1
        self.last_error_time = time.time()
        self._update_health()
    
    def _update_health(self):
        """Update health status based on error rate."""
        total_operations = self.error_count + self.success_count
        
        if total_operations == 0:
            return
        
        error_rate = self.error_count / total_operations
        
        # Determine health status
        if error_rate >= 0.5:
            new_status = SystemHealth.FAILING
        elif error_rate >= 0.2:
            new_status = SystemHealth.CRITICAL
        elif error_rate >= 0.1:
            new_status = SystemHealth.DEGRADED
        else:
            new_status = SystemHealth.HEALTHY
        
        # Update if changed
        if self.health_status != new_status:
            self.health_status = new_status
            self.health_checker.report_service_health(self.service_name, new_status)
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get health metrics for this service."""
        total_operations = self.error_count + self.success_count
        error_rate = self.error_count / total_operations if total_operations > 0 else 0.0
        
        return {
            'service_name': self.service_name,
            'health_status': self.health_status.value,
            'error_count': self.error_count,
            'success_count': self.success_count,
            'error_rate': error_rate,
            'last_error_time': self.last_error_time
        }


class ErrorRecoveryManager:
    """Manages error recovery strategies and execution."""
    
    def __init__(self, config: Optional[RecoveryConfig] = None):
        self.config = config or RecoveryConfig()
        self.strategies: List[RecoveryStrategy] = []
        self.health_checker = SystemHealthChecker(self.config)
        self.service_trackers: Dict[str, ServiceHealthTracker] = {}
        self.recovery_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
        # Initialize default strategies
        self._initialize_default_strategies()
        
        # Start health monitoring
        self.health_checker.start_monitoring()
    
    def _initialize_default_strategies(self):
        """Initialize default recovery strategies."""
        self.strategies = [
            RetryStrategy(self.config.max_retries, self.config.retry_delay),
            FallbackStrategy(),
            CircuitBreakerStrategy(
                self.config.circuit_breaker_threshold,
                self.config.circuit_breaker_timeout
            ),
            GracefulDegradationStrategy()
        ]
        
        # Sort by priority
        self.strategies.sort(key=lambda s: s.get_priority())
    
    def add_strategy(self, strategy: RecoveryStrategy):
        """Add custom recovery strategy."""
        with self._lock:
            self.strategies.append(strategy)
            self.strategies.sort(key=lambda s: s.get_priority())
    
    def register_service(self, service_name: str) -> ServiceHealthTracker:
        """Register service for health tracking."""
        with self._lock:
            if service_name not in self.service_trackers:
                self.service_trackers[service_name] = ServiceHealthTracker(
                    service_name, self.health_checker
                )
            return self.service_trackers[service_name]
    
    def attempt_recovery(
        self, 
        error: BaseGrandModelError, 
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Any]:
        """Attempt to recover from error using available strategies."""
        context = context or {}
        
        with self._lock:
            # Record recovery attempt
            recovery_attempt = {
                'timestamp': time.time(),
                'error_type': type(error).__name__,
                'error_code': error.error_code,
                'correlation_id': error.correlation_id,
                'strategies_attempted': [],
                'recovery_successful': False,
                'recovery_result': None
            }
            
            # Try each strategy
            for strategy in self.strategies:
                if strategy.can_recover(error):
                    try:
                        logger.info(
                            f"Attempting recovery with {type(strategy).__name__}",
                            extra={
                                'strategy': type(strategy).__name__,
                                'error_type': type(error).__name__,
                                'correlation_id': error.correlation_id
                            }
                        )
                        
                        result = strategy.recover(error, context)
                        
                        recovery_attempt['strategies_attempted'].append({
                            'strategy': type(strategy).__name__,
                            'success': True,
                            'result': result
                        })
                        
                        recovery_attempt['recovery_successful'] = True
                        recovery_attempt['recovery_result'] = result
                        
                        self.recovery_history.append(recovery_attempt)
                        
                        # Update service health
                        service_name = error.error_details.get('service_name', 'unknown')
                        if service_name in self.service_trackers:
                            self.service_trackers[service_name].record_success()
                        
                        return True, result
                        
                    except Exception as recovery_error:
                        logger.error(
                            f"Recovery strategy {type(strategy).__name__} failed: {recovery_error}",
                            extra={
                                'strategy': type(strategy).__name__,
                                'error_type': type(error).__name__,
                                'correlation_id': error.correlation_id
                            }
                        )
                        
                        recovery_attempt['strategies_attempted'].append({
                            'strategy': type(strategy).__name__,
                            'success': False,
                            'error': str(recovery_error)
                        })
            
            # No recovery possible
            recovery_attempt['recovery_successful'] = False
            self.recovery_history.append(recovery_attempt)
            
            # Update service health
            service_name = error.error_details.get('service_name', 'unknown')
            if service_name in self.service_trackers:
                self.service_trackers[service_name].record_error(error)
            
            return False, None
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        with self._lock:
            total_attempts = len(self.recovery_history)
            successful_attempts = sum(1 for r in self.recovery_history if r['recovery_successful'])
            
            strategy_stats = defaultdict(lambda: {'attempts': 0, 'successes': 0})
            
            for attempt in self.recovery_history:
                for strategy_attempt in attempt['strategies_attempted']:
                    strategy_name = strategy_attempt['strategy']
                    strategy_stats[strategy_name]['attempts'] += 1
                    if strategy_attempt['success']:
                        strategy_stats[strategy_name]['successes'] += 1
            
            return {
                'total_recovery_attempts': total_attempts,
                'successful_recoveries': successful_attempts,
                'success_rate': successful_attempts / total_attempts if total_attempts > 0 else 0,
                'strategy_stats': dict(strategy_stats),
                'health_status': self.health_checker.get_health_status(),
                'service_health': {
                    name: tracker.get_health_metrics() 
                    for name, tracker in self.service_trackers.items()
                }
            }
    
    def shutdown(self):
        """Shutdown recovery manager."""
        self.health_checker.stop_monitoring()


# Global recovery manager instance
_global_recovery_manager = None
_recovery_lock = threading.Lock()


def get_global_recovery_manager() -> ErrorRecoveryManager:
    """Get global recovery manager instance."""
    global _global_recovery_manager
    
    if _global_recovery_manager is None:
        with _recovery_lock:
            if _global_recovery_manager is None:
                _global_recovery_manager = ErrorRecoveryManager()
    
    return _global_recovery_manager


def attempt_recovery(
    error: BaseGrandModelError, 
    context: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Any]:
    """Attempt recovery using global recovery manager."""
    return get_global_recovery_manager().attempt_recovery(error, context)


def register_service(service_name: str) -> ServiceHealthTracker:
    """Register service for health tracking."""
    return get_global_recovery_manager().register_service(service_name)