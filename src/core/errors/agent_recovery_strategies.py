"""
Enhanced error recovery strategies for different agent types.

Provides specialized recovery mechanisms for strategic, tactical, risk, and execution agents
with different failure modes and recovery patterns.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Callable, List, Union, Tuple
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
from contextlib import contextmanager

from .base_exceptions import (
    BaseGrandModelError, ErrorSeverity, ErrorCategory, ErrorContext,
    ValidationError, DataError, SystemError, TimeoutError, NetworkError,
    RecoverableError, NonRecoverableError, DependencyError
)
from .agent_error_decorators import AgentType

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Recovery strategy types"""
    IMMEDIATE_RETRY = "immediate_retry"
    GRADUAL_RECOVERY = "gradual_recovery"
    CIRCUIT_BREAKER = "circuit_breaker"
    FALLBACK_MODE = "fallback_mode"
    RESTART_AGENT = "restart_agent"
    ESCALATE_ERROR = "escalate_error"


class RecoveryStatus(Enum):
    """Recovery status"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    SUCCESSFUL = "successful"
    FAILED = "failed"
    ABANDONED = "abandoned"


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt"""
    strategy: RecoveryStrategy
    start_time: datetime
    end_time: Optional[datetime] = None
    status: RecoveryStatus = RecoveryStatus.NOT_STARTED
    error: Optional[Exception] = None
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class RecoveryConfig:
    """Configuration for recovery strategies"""
    max_recovery_attempts: int = 3
    recovery_timeout: float = 30.0
    gradual_recovery_steps: int = 3
    circuit_breaker_timeout: float = 60.0
    fallback_mode_timeout: float = 300.0
    restart_cooldown: float = 120.0
    escalation_threshold: int = 5


class BaseRecoveryStrategy(ABC):
    """Base class for recovery strategies"""
    
    def __init__(self, agent_type: AgentType, config: RecoveryConfig):
        self.agent_type = agent_type
        self.config = config
        self.recovery_history: List[RecoveryAttempt] = []
        self.last_recovery_time: Optional[datetime] = None
        self.recovery_lock = threading.Lock()
        
    @abstractmethod
    def can_recover(self, error: Exception, context: ErrorContext) -> bool:
        """Check if this strategy can recover from the error"""
        pass
    
    @abstractmethod
    def recover(self, error: Exception, context: ErrorContext, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt recovery from the error"""
        pass
    
    def record_recovery_attempt(self, attempt: RecoveryAttempt):
        """Record a recovery attempt"""
        with self.recovery_lock:
            self.recovery_history.append(attempt)
            self.last_recovery_time = attempt.start_time
            
            # Keep only recent attempts
            if len(self.recovery_history) > 100:
                self.recovery_history = self.recovery_history[-100:]
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics"""
        with self.recovery_lock:
            total_attempts = len(self.recovery_history)
            successful = len([a for a in self.recovery_history if a.status == RecoveryStatus.SUCCESSFUL])
            failed = len([a for a in self.recovery_history if a.status == RecoveryStatus.FAILED])
            
            return {
                'total_attempts': total_attempts,
                'successful': successful,
                'failed': failed,
                'success_rate': successful / total_attempts if total_attempts > 0 else 0.0,
                'last_recovery': self.last_recovery_time.isoformat() if self.last_recovery_time else None
            }


class ImmediateRetryStrategy(BaseRecoveryStrategy):
    """Immediate retry strategy for transient errors"""
    
    def can_recover(self, error: Exception, context: ErrorContext) -> bool:
        """Check if immediate retry is appropriate"""
        # Suitable for network errors, timeouts, temporary resource issues
        return isinstance(error, (NetworkError, TimeoutError, RecoverableError))
    
    def recover(self, error: Exception, context: ErrorContext, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt immediate retry"""
        attempt = RecoveryAttempt(
            strategy=RecoveryStrategy.IMMEDIATE_RETRY,
            start_time=datetime.utcnow(),
            status=RecoveryStatus.IN_PROGRESS
        )
        
        try:
            # Short delay before retry
            time.sleep(0.1)
            
            # Try to restore agent to last known good state
            if 'last_valid_state' in agent_state:
                restored_state = agent_state['last_valid_state'].copy()
                
                attempt.status = RecoveryStatus.SUCCESSFUL
                attempt.end_time = datetime.utcnow()
                attempt.details = {
                    'state_restored': True,
                    'retry_delay': 0.1
                }
                
                return {
                    'success': True,
                    'strategy': RecoveryStrategy.IMMEDIATE_RETRY,
                    'restored_state': restored_state,
                    'message': 'Immediate retry successful'
                }
            
            attempt.status = RecoveryStatus.FAILED
            attempt.error = ValueError("No valid state to restore")
            
            return {
                'success': False,
                'strategy': RecoveryStrategy.IMMEDIATE_RETRY,
                'message': 'No valid state available for immediate retry'
            }
            
        except Exception as recovery_error:
            attempt.status = RecoveryStatus.FAILED
            attempt.error = recovery_error
            logger.error(f"Immediate retry failed: {recovery_error}")
            
            return {
                'success': False,
                'strategy': RecoveryStrategy.IMMEDIATE_RETRY,
                'message': f'Immediate retry failed: {str(recovery_error)}'
            }
        
        finally:
            attempt.end_time = datetime.utcnow()
            self.record_recovery_attempt(attempt)


class GradualRecoveryStrategy(BaseRecoveryStrategy):
    """Gradual recovery strategy for complex errors"""
    
    def can_recover(self, error: Exception, context: ErrorContext) -> bool:
        """Check if gradual recovery is appropriate"""
        # Suitable for data errors, validation errors, system errors
        return isinstance(error, (DataError, ValidationError, SystemError))
    
    def recover(self, error: Exception, context: ErrorContext, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt gradual recovery"""
        attempt = RecoveryAttempt(
            strategy=RecoveryStrategy.GRADUAL_RECOVERY,
            start_time=datetime.utcnow(),
            status=RecoveryStatus.IN_PROGRESS
        )
        
        try:
            # Multi-step recovery process
            recovery_steps = self._get_recovery_steps(error, agent_state)
            successful_steps = 0
            
            for step_num, step_func in enumerate(recovery_steps, 1):
                try:
                    logger.info(f"Executing recovery step {step_num}/{len(recovery_steps)}")
                    step_func()
                    successful_steps += 1
                    time.sleep(1.0)  # Gradual approach
                    
                except Exception as step_error:
                    logger.error(f"Recovery step {step_num} failed: {step_error}")
                    break
            
            if successful_steps == len(recovery_steps):
                attempt.status = RecoveryStatus.SUCCESSFUL
                attempt.details = {
                    'steps_completed': successful_steps,
                    'total_steps': len(recovery_steps)
                }
                
                return {
                    'success': True,
                    'strategy': RecoveryStrategy.GRADUAL_RECOVERY,
                    'steps_completed': successful_steps,
                    'message': 'Gradual recovery successful'
                }
            else:
                attempt.status = RecoveryStatus.FAILED
                attempt.details = {
                    'steps_completed': successful_steps,
                    'total_steps': len(recovery_steps)
                }
                
                return {
                    'success': False,
                    'strategy': RecoveryStrategy.GRADUAL_RECOVERY,
                    'steps_completed': successful_steps,
                    'message': f'Gradual recovery partially successful ({successful_steps}/{len(recovery_steps)})'
                }
                
        except Exception as recovery_error:
            attempt.status = RecoveryStatus.FAILED
            attempt.error = recovery_error
            logger.error(f"Gradual recovery failed: {recovery_error}")
            
            return {
                'success': False,
                'strategy': RecoveryStrategy.GRADUAL_RECOVERY,
                'message': f'Gradual recovery failed: {str(recovery_error)}'
            }
        
        finally:
            attempt.end_time = datetime.utcnow()
            self.record_recovery_attempt(attempt)
    
    def _get_recovery_steps(self, error: Exception, agent_state: Dict[str, Any]) -> List[Callable]:
        """Get recovery steps based on error type and agent state"""
        steps = []
        
        # Step 1: Reset error state
        steps.append(lambda: self._reset_error_state(agent_state))
        
        # Step 2: Restore minimal functionality
        steps.append(lambda: self._restore_minimal_functionality(agent_state))
        
        # Step 3: Gradually restore full functionality
        steps.append(lambda: self._restore_full_functionality(agent_state))
        
        return steps
    
    def _reset_error_state(self, agent_state: Dict[str, Any]):
        """Reset error state"""
        agent_state['error_count'] = 0
        agent_state['last_error_time'] = None
        logger.info("Error state reset")
    
    def _restore_minimal_functionality(self, agent_state: Dict[str, Any]):
        """Restore minimal functionality"""
        # Implementation depends on agent type
        agent_state['status'] = 'degraded'
        logger.info("Minimal functionality restored")
    
    def _restore_full_functionality(self, agent_state: Dict[str, Any]):
        """Restore full functionality"""
        agent_state['status'] = 'active'
        logger.info("Full functionality restored")


class CircuitBreakerStrategy(BaseRecoveryStrategy):
    """Circuit breaker strategy for cascading failures"""
    
    def __init__(self, agent_type: AgentType, config: RecoveryConfig):
        super().__init__(agent_type, config)
        self.circuit_open_time: Optional[datetime] = None
        self.failure_count = 0
        self.circuit_state = "closed"  # closed, open, half-open
    
    def can_recover(self, error: Exception, context: ErrorContext) -> bool:
        """Check if circuit breaker is appropriate"""
        # Suitable for dependency errors, cascading failures
        return isinstance(error, (DependencyError, SystemError))
    
    def recover(self, error: Exception, context: ErrorContext, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt circuit breaker recovery"""
        attempt = RecoveryAttempt(
            strategy=RecoveryStrategy.CIRCUIT_BREAKER,
            start_time=datetime.utcnow(),
            status=RecoveryStatus.IN_PROGRESS
        )
        
        try:
            self.failure_count += 1
            
            if self.circuit_state == "closed":
                if self.failure_count >= 5:  # Threshold
                    self._open_circuit()
                
                return {
                    'success': False,
                    'strategy': RecoveryStrategy.CIRCUIT_BREAKER,
                    'circuit_state': self.circuit_state,
                    'message': 'Circuit breaker opened due to failures'
                }
            
            elif self.circuit_state == "open":
                if self._should_try_half_open():
                    self._half_open_circuit()
                    
                    return {
                        'success': True,
                        'strategy': RecoveryStrategy.CIRCUIT_BREAKER,
                        'circuit_state': self.circuit_state,
                        'message': 'Circuit breaker moved to half-open state'
                    }
                
                return {
                    'success': False,
                    'strategy': RecoveryStrategy.CIRCUIT_BREAKER,
                    'circuit_state': self.circuit_state,
                    'message': 'Circuit breaker still open'
                }
            
            elif self.circuit_state == "half-open":
                # Test if system is healthy
                if self._test_system_health():
                    self._close_circuit()
                    attempt.status = RecoveryStatus.SUCCESSFUL
                    
                    return {
                        'success': True,
                        'strategy': RecoveryStrategy.CIRCUIT_BREAKER,
                        'circuit_state': self.circuit_state,
                        'message': 'Circuit breaker closed - system healthy'
                    }
                else:
                    self._open_circuit()
                    
                    return {
                        'success': False,
                        'strategy': RecoveryStrategy.CIRCUIT_BREAKER,
                        'circuit_state': self.circuit_state,
                        'message': 'Circuit breaker reopened - system still unhealthy'
                    }
            
        except Exception as recovery_error:
            attempt.status = RecoveryStatus.FAILED
            attempt.error = recovery_error
            logger.error(f"Circuit breaker recovery failed: {recovery_error}")
            
            return {
                'success': False,
                'strategy': RecoveryStrategy.CIRCUIT_BREAKER,
                'message': f'Circuit breaker recovery failed: {str(recovery_error)}'
            }
        
        finally:
            attempt.end_time = datetime.utcnow()
            self.record_recovery_attempt(attempt)
    
    def _open_circuit(self):
        """Open the circuit breaker"""
        self.circuit_state = "open"
        self.circuit_open_time = datetime.utcnow()
        logger.warning("Circuit breaker opened")
    
    def _close_circuit(self):
        """Close the circuit breaker"""
        self.circuit_state = "closed"
        self.circuit_open_time = None
        self.failure_count = 0
        logger.info("Circuit breaker closed")
    
    def _half_open_circuit(self):
        """Move circuit breaker to half-open state"""
        self.circuit_state = "half-open"
        logger.info("Circuit breaker moved to half-open state")
    
    def _should_try_half_open(self) -> bool:
        """Check if should try half-open state"""
        if self.circuit_open_time is None:
            return False
        
        elapsed = datetime.utcnow() - self.circuit_open_time
        return elapsed.total_seconds() >= self.config.circuit_breaker_timeout
    
    def _test_system_health(self) -> bool:
        """Test if system is healthy"""
        # Simple health check - can be extended
        return True  # Placeholder


class FallbackModeStrategy(BaseRecoveryStrategy):
    """Fallback mode strategy for degraded operation"""
    
    def can_recover(self, error: Exception, context: ErrorContext) -> bool:
        """Check if fallback mode is appropriate"""
        # Suitable for most errors when graceful degradation is possible
        return not isinstance(error, NonRecoverableError)
    
    def recover(self, error: Exception, context: ErrorContext, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt fallback mode recovery"""
        attempt = RecoveryAttempt(
            strategy=RecoveryStrategy.FALLBACK_MODE,
            start_time=datetime.utcnow(),
            status=RecoveryStatus.IN_PROGRESS
        )
        
        try:
            # Enter fallback mode
            fallback_config = self._get_fallback_config()
            
            # Update agent state for fallback mode
            agent_state.update({
                'mode': 'fallback',
                'fallback_config': fallback_config,
                'fallback_start_time': datetime.utcnow(),
                'original_error': str(error)
            })
            
            attempt.status = RecoveryStatus.SUCCESSFUL
            attempt.details = {
                'fallback_config': fallback_config,
                'mode': 'fallback'
            }
            
            return {
                'success': True,
                'strategy': RecoveryStrategy.FALLBACK_MODE,
                'fallback_config': fallback_config,
                'message': 'Entered fallback mode successfully'
            }
            
        except Exception as recovery_error:
            attempt.status = RecoveryStatus.FAILED
            attempt.error = recovery_error
            logger.error(f"Fallback mode recovery failed: {recovery_error}")
            
            return {
                'success': False,
                'strategy': RecoveryStrategy.FALLBACK_MODE,
                'message': f'Fallback mode recovery failed: {str(recovery_error)}'
            }
        
        finally:
            attempt.end_time = datetime.utcnow()
            self.record_recovery_attempt(attempt)
    
    def _get_fallback_config(self) -> Dict[str, Any]:
        """Get fallback configuration based on agent type"""
        base_config = {
            'reduced_functionality': True,
            'conservative_mode': True,
            'timeout': self.config.fallback_mode_timeout
        }
        
        if self.agent_type == AgentType.STRATEGIC:
            base_config.update({
                'default_action': 'hold',
                'confidence_threshold': 0.8
            })
        elif self.agent_type == AgentType.TACTICAL:
            base_config.update({
                'signal_dampening': 0.5,
                'max_position_size': 0.1
            })
        elif self.agent_type == AgentType.RISK:
            base_config.update({
                'emergency_mode': True,
                'max_risk_limit': 0.01
            })
        elif self.agent_type == AgentType.EXECUTION:
            base_config.update({
                'order_size_limit': 0.1,
                'execution_delay': 2.0
            })
        
        return base_config


class AgentRecoveryManager:
    """Manages recovery strategies for an agent"""
    
    def __init__(self, agent_type: AgentType, config: Optional[RecoveryConfig] = None):
        self.agent_type = agent_type
        self.config = config or RecoveryConfig()
        self.strategies = self._initialize_strategies()
        self.recovery_history: List[Dict[str, Any]] = []
    
    def _initialize_strategies(self) -> List[BaseRecoveryStrategy]:
        """Initialize recovery strategies"""
        strategies = [
            ImmediateRetryStrategy(self.agent_type, self.config),
            GradualRecoveryStrategy(self.agent_type, self.config),
            CircuitBreakerStrategy(self.agent_type, self.config),
            FallbackModeStrategy(self.agent_type, self.config)
        ]
        
        return strategies
    
    def recover(self, error: Exception, context: ErrorContext, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt recovery using appropriate strategy"""
        recovery_start = datetime.utcnow()
        
        # Find suitable recovery strategy
        suitable_strategy = None
        for strategy in self.strategies:
            if strategy.can_recover(error, context):
                suitable_strategy = strategy
                break
        
        if suitable_strategy is None:
            logger.error(f"No suitable recovery strategy found for error: {error}")
            return {
                'success': False,
                'strategy': None,
                'message': 'No suitable recovery strategy found'
            }
        
        # Attempt recovery
        logger.info(f"Attempting recovery with {suitable_strategy.__class__.__name__}")
        result = suitable_strategy.recover(error, context, agent_state)
        
        # Record recovery attempt
        recovery_record = {
            'timestamp': recovery_start.isoformat(),
            'agent_type': self.agent_type.value,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'strategy': result.get('strategy'),
            'success': result.get('success', False),
            'details': result
        }
        
        self.recovery_history.append(recovery_record)
        
        # Keep only recent history
        if len(self.recovery_history) > 1000:
            self.recovery_history = self.recovery_history[-1000:]
        
        return result
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get comprehensive recovery statistics"""
        strategy_stats = {}
        for strategy in self.strategies:
            strategy_stats[strategy.__class__.__name__] = strategy.get_recovery_stats()
        
        total_attempts = len(self.recovery_history)
        successful = len([r for r in self.recovery_history if r['success']])
        
        return {
            'total_recovery_attempts': total_attempts,
            'successful_recoveries': successful,
            'recovery_success_rate': successful / total_attempts if total_attempts > 0 else 0.0,
            'strategy_stats': strategy_stats,
            'recent_recoveries': self.recovery_history[-10:]  # Last 10 recovery attempts
        }
    
    def reset_recovery_state(self):
        """Reset recovery state"""
        for strategy in self.strategies:
            strategy.recovery_history.clear()
            strategy.last_recovery_time = None
        
        self.recovery_history.clear()
        logger.info(f"Recovery state reset for {self.agent_type.value} agent")


# Factory functions for creating recovery managers
def create_strategic_recovery_manager(config: Optional[RecoveryConfig] = None) -> AgentRecoveryManager:
    """Create recovery manager for strategic agents"""
    return AgentRecoveryManager(AgentType.STRATEGIC, config)


def create_tactical_recovery_manager(config: Optional[RecoveryConfig] = None) -> AgentRecoveryManager:
    """Create recovery manager for tactical agents"""
    return AgentRecoveryManager(AgentType.TACTICAL, config)


def create_risk_recovery_manager(config: Optional[RecoveryConfig] = None) -> AgentRecoveryManager:
    """Create recovery manager for risk agents"""
    return AgentRecoveryManager(AgentType.RISK, config)


def create_execution_recovery_manager(config: Optional[RecoveryConfig] = None) -> AgentRecoveryManager:
    """Create recovery manager for execution agents"""
    return AgentRecoveryManager(AgentType.EXECUTION, config)