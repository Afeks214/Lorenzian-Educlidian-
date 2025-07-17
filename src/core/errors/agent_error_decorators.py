"""
Agent-specific error handling decorators and context managers.

Provides specialized error handling patterns for different types of agents
in the trading system, including strategic, tactical, risk, and execution agents.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Callable, Type, List, Union, Awaitable
from functools import wraps
from contextlib import contextmanager, asynccontextmanager
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

from .base_exceptions import (
    BaseGrandModelError, ErrorSeverity, ErrorCategory, ErrorContext,
    ValidationError, DataError, SystemError, TimeoutError, NetworkError,
    RecoverableError, NonRecoverableError
)
from .error_handler import ErrorHandler, RetryConfig, CircuitBreakerConfig

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of trading agents"""
    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    RISK = "risk"
    EXECUTION = "execution"
    MARL = "marl"
    MONITORING = "monitoring"


@dataclass
class AgentErrorConfig:
    """Configuration for agent-specific error handling"""
    agent_type: AgentType
    max_retries: int = 3
    retry_delay: float = 1.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 30.0
    silent_failure_detection: bool = True
    graceful_degradation: bool = True
    auto_recovery: bool = True
    fallback_mode: bool = True


class AgentErrorDecorator:
    """Decorator factory for agent-specific error handling"""
    
    def __init__(self, agent_type: AgentType, config: Optional[AgentErrorConfig] = None):
        self.agent_type = agent_type
        self.config = config or AgentErrorConfig(agent_type)
        self._setup_error_handling()
    
    def _setup_error_handling(self):
        """Setup error handling based on agent type"""
        retry_config = RetryConfig(
            max_attempts=self.config.max_retries,
            base_delay=self.config.retry_delay,
            retriable_exceptions=self._get_retriable_exceptions()
        )
        
        circuit_config = CircuitBreakerConfig(
            failure_threshold=self.config.circuit_breaker_threshold,
            recovery_timeout=self.config.circuit_breaker_timeout,
            name=f"{self.agent_type.value}_circuit"
        )
        
        self.error_handler = ErrorHandler(
            retry_config=retry_config,
            circuit_breaker_config=circuit_config
        )
    
    def _get_retriable_exceptions(self) -> List[Type[Exception]]:
        """Get retriable exceptions based on agent type"""
        common_retriable = [NetworkError, TimeoutError, RecoverableError]
        
        if self.agent_type == AgentType.STRATEGIC:
            return common_retriable + [DataError]
        elif self.agent_type == AgentType.TACTICAL:
            return common_retriable + [ValidationError]
        elif self.agent_type == AgentType.RISK:
            return common_retriable  # Risk agents should be more conservative
        elif self.agent_type == AgentType.EXECUTION:
            return common_retriable + [ValidationError]
        else:
            return common_retriable
    
    def observe(self, fallback_value: Any = None, validate_response: Optional[Callable] = None):
        """Decorator for observation methods"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    result = self.error_handler.execute_with_retry(func, *args, **kwargs)
                    
                    if validate_response and not validate_response(result):
                        raise ValidationError("Invalid observation response")
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Observation error in {self.agent_type.value} agent: {e}")
                    
                    if self.config.fallback_mode and fallback_value is not None:
                        return fallback_value
                    
                    # Re-raise if not in fallback mode
                    raise
            
            return wrapper
        return decorator
    
    def decide(self, fallback_action: str = "hold", validate_response: Optional[Callable] = None):
        """Decorator for decision methods"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    result = self.error_handler.execute_with_retry(func, *args, **kwargs)
                    
                    if validate_response and not validate_response(result):
                        raise ValidationError("Invalid decision response")
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Decision error in {self.agent_type.value} agent: {e}")
                    
                    if self.config.fallback_mode:
                        return self._get_safe_decision(fallback_action, str(e))
                    
                    raise
            
            return wrapper
        return decorator
    
    def act(self, validate_response: Optional[Callable] = None):
        """Decorator for action methods"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    result = self.error_handler.execute_with_retry(func, *args, **kwargs)
                    
                    if validate_response and not validate_response(result):
                        raise ValidationError("Invalid action response")
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Action error in {self.agent_type.value} agent: {e}")
                    
                    if self.config.fallback_mode:
                        return self._get_safe_action_result(str(e))
                    
                    raise
            
            return wrapper
        return decorator
    
    def risk_check(self, conservative_fallback: bool = True):
        """Decorator for risk checking methods"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    result = self.error_handler.execute_with_retry(func, *args, **kwargs)
                    
                    if not isinstance(result, bool):
                        raise ValidationError("Risk check must return boolean")
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Risk check error in {self.agent_type.value} agent: {e}")
                    
                    if self.config.fallback_mode:
                        # Conservative approach - reject in case of error
                        return not conservative_fallback
                    
                    raise
            
            return wrapper
        return decorator
    
    def position_update(self, rollback_on_error: bool = True):
        """Decorator for position update methods"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(self_ref, *args, **kwargs):
                # Backup current state
                original_positions = getattr(self_ref, 'positions', {}).copy()
                
                try:
                    result = self.error_handler.execute_with_retry(func, self_ref, *args, **kwargs)
                    return result
                    
                except Exception as e:
                    logger.error(f"Position update error in {self.agent_type.value} agent: {e}")
                    
                    if rollback_on_error and hasattr(self_ref, 'positions'):
                        # Rollback positions
                        self_ref.positions = original_positions
                        logger.info(f"Rolled back positions for {self.agent_type.value} agent")
                    
                    if self.config.fallback_mode:
                        return {
                            'status': 'error',
                            'message': f'Position update failed: {str(e)}',
                            'rollback_performed': rollback_on_error
                        }
                    
                    raise
            
            return wrapper
        return decorator
    
    def _get_safe_decision(self, fallback_action: str, error_message: str) -> Dict[str, Any]:
        """Get safe decision for fallback"""
        return {
            'action': fallback_action,
            'confidence': 0.0,
            'reasoning': f'Safe fallback decision due to error: {error_message}',
            'timestamp': datetime.utcnow().isoformat(),
            'fallback': True
        }
    
    def _get_safe_action_result(self, error_message: str) -> Dict[str, Any]:
        """Get safe action result for fallback"""
        return {
            'status': 'fallback',
            'action_taken': 'none',
            'message': f'Safe fallback action due to error: {error_message}',
            'timestamp': datetime.utcnow().isoformat(),
            'fallback': True
        }


class StrategicAgentErrorDecorator(AgentErrorDecorator):
    """Specialized decorator for strategic agents"""
    
    def __init__(self, config: Optional[AgentErrorConfig] = None):
        super().__init__(AgentType.STRATEGIC, config)
    
    def regime_analysis(self, fallback_regime: str = "unknown"):
        """Decorator for regime analysis methods"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    result = self.error_handler.execute_with_retry(func, *args, **kwargs)
                    
                    if not isinstance(result, dict) or 'regime' not in result:
                        raise ValidationError("Invalid regime analysis response")
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Regime analysis error: {e}")
                    
                    if self.config.fallback_mode:
                        return {
                            'regime': fallback_regime,
                            'confidence': 0.0,
                            'reasoning': f'Fallback regime due to error: {str(e)}',
                            'timestamp': datetime.utcnow().isoformat(),
                            'fallback': True
                        }
                    
                    raise
            
            return wrapper
        return decorator


class TacticalAgentErrorDecorator(AgentErrorDecorator):
    """Specialized decorator for tactical agents"""
    
    def __init__(self, config: Optional[AgentErrorConfig] = None):
        super().__init__(AgentType.TACTICAL, config)
    
    def signal_generation(self, fallback_signal: float = 0.0):
        """Decorator for signal generation methods"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    result = self.error_handler.execute_with_retry(func, *args, **kwargs)
                    
                    if not isinstance(result, (dict, float, int)):
                        raise ValidationError("Invalid signal generation response")
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Signal generation error: {e}")
                    
                    if self.config.fallback_mode:
                        return {
                            'signal': fallback_signal,
                            'confidence': 0.0,
                            'reasoning': f'Fallback signal due to error: {str(e)}',
                            'timestamp': datetime.utcnow().isoformat(),
                            'fallback': True
                        }
                    
                    raise
            
            return wrapper
        return decorator


class RiskAgentErrorDecorator(AgentErrorDecorator):
    """Specialized decorator for risk agents"""
    
    def __init__(self, config: Optional[AgentErrorConfig] = None):
        config = config or AgentErrorConfig(AgentType.RISK)
        config.max_retries = 2  # Risk agents should be more conservative
        config.graceful_degradation = False  # Risk agents should fail fast
        super().__init__(AgentType.RISK, config)
    
    def risk_assessment(self, conservative_fallback: bool = True):
        """Decorator for risk assessment methods"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    result = self.error_handler.execute_with_retry(func, *args, **kwargs)
                    
                    if not isinstance(result, dict) or 'risk_score' not in result:
                        raise ValidationError("Invalid risk assessment response")
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Risk assessment error: {e}")
                    
                    if self.config.fallback_mode and conservative_fallback:
                        return {
                            'risk_score': 1.0,  # Maximum risk
                            'recommendation': 'emergency_stop',
                            'reasoning': f'Conservative fallback due to error: {str(e)}',
                            'timestamp': datetime.utcnow().isoformat(),
                            'fallback': True
                        }
                    
                    raise
            
            return wrapper
        return decorator


class ExecutionAgentErrorDecorator(AgentErrorDecorator):
    """Specialized decorator for execution agents"""
    
    def __init__(self, config: Optional[AgentErrorConfig] = None):
        super().__init__(AgentType.EXECUTION, config)
    
    def order_execution(self, retry_on_partial_fill: bool = True):
        """Decorator for order execution methods"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    result = self.error_handler.execute_with_retry(func, *args, **kwargs)
                    
                    if not isinstance(result, dict) or 'status' not in result:
                        raise ValidationError("Invalid order execution response")
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Order execution error: {e}")
                    
                    if self.config.fallback_mode:
                        return {
                            'status': 'failed',
                            'filled_quantity': 0,
                            'remaining_quantity': kwargs.get('quantity', 0),
                            'error': str(e),
                            'timestamp': datetime.utcnow().isoformat(),
                            'fallback': True
                        }
                    
                    raise
            
            return wrapper
        return decorator


@contextmanager
def agent_operation_context(agent_type: AgentType, operation_name: str):
    """Context manager for agent operations"""
    start_time = time.time()
    logger.info(f"Starting {operation_name} for {agent_type.value} agent")
    
    try:
        yield
        duration = time.time() - start_time
        logger.info(f"Completed {operation_name} for {agent_type.value} agent in {duration:.2f}s")
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Failed {operation_name} for {agent_type.value} agent after {duration:.2f}s: {e}")
        raise


@asynccontextmanager
async def async_agent_operation_context(agent_type: AgentType, operation_name: str):
    """Async context manager for agent operations"""
    start_time = time.time()
    logger.info(f"Starting async {operation_name} for {agent_type.value} agent")
    
    try:
        yield
        duration = time.time() - start_time
        logger.info(f"Completed async {operation_name} for {agent_type.value} agent in {duration:.2f}s")
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Failed async {operation_name} for {agent_type.value} agent after {duration:.2f}s: {e}")
        raise


def with_agent_error_handling(
    agent_type: AgentType,
    operation_name: str,
    fallback_value: Any = None,
    validate_response: Optional[Callable] = None
):
    """General decorator for agent error handling"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with agent_operation_context(agent_type, operation_name):
                try:
                    result = func(*args, **kwargs)
                    
                    if validate_response and not validate_response(result):
                        raise ValidationError(f"Invalid response for {operation_name}")
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Error in {operation_name} for {agent_type.value} agent: {e}")
                    
                    if fallback_value is not None:
                        return fallback_value
                    
                    raise
        
        return wrapper
    return decorator


def async_with_agent_error_handling(
    agent_type: AgentType,
    operation_name: str,
    fallback_value: Any = None,
    validate_response: Optional[Callable] = None
):
    """Async decorator for agent error handling"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with async_agent_operation_context(agent_type, operation_name):
                try:
                    result = await func(*args, **kwargs)
                    
                    if validate_response and not validate_response(result):
                        raise ValidationError(f"Invalid response for {operation_name}")
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Error in async {operation_name} for {agent_type.value} agent: {e}")
                    
                    if fallback_value is not None:
                        return fallback_value
                    
                    raise
        
        return wrapper
    return decorator


# Convenience factory functions
def strategic_agent_decorator(config: Optional[AgentErrorConfig] = None) -> StrategicAgentErrorDecorator:
    """Create strategic agent error decorator"""
    return StrategicAgentErrorDecorator(config)


def tactical_agent_decorator(config: Optional[AgentErrorConfig] = None) -> TacticalAgentErrorDecorator:
    """Create tactical agent error decorator"""
    return TacticalAgentErrorDecorator(config)


def risk_agent_decorator(config: Optional[AgentErrorConfig] = None) -> RiskAgentErrorDecorator:
    """Create risk agent error decorator"""
    return RiskAgentErrorDecorator(config)


def execution_agent_decorator(config: Optional[AgentErrorConfig] = None) -> ExecutionAgentErrorDecorator:
    """Create execution agent error decorator"""
    return ExecutionAgentErrorDecorator(config)