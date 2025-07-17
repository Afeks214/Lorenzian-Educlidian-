#!/usr/bin/env python3
"""
AGENT 7: Trading Error Integration
Integration between trading decision logging and error handling systems.
"""

import functools
import asyncio
import time
from typing import Dict, Any, Optional, Callable, Union
from datetime import datetime, timezone

from src.monitoring.trading_decision_logger import (
    TradingDecisionLogger,
    TradingDecisionType,
    TradingDecisionOutcome,
    TradingDecisionMetrics,
    get_trading_decision_logger
)
from src.core.errors.error_handler import get_error_handler
from src.core.errors.base_exceptions import BaseGrandModelError, ErrorContext, SystemError
from src.monitoring.structured_logging import get_logger, correlation_context, LogComponent


class TradingErrorIntegrator:
    """
    Integrates trading decision logging with error handling to eliminate silent failures
    and provide comprehensive audit trails.
    """
    
    def __init__(self):
        self.trading_logger = get_trading_decision_logger()
        self.error_handler = get_error_handler()
        self.logger = get_logger("trading_error_integration")
        
        # Register critical trading functions as mandatory response functions
        self._register_critical_functions()
    
    def _register_critical_functions(self):
        """Register critical trading functions that must not fail silently."""
        
        critical_functions = [
            "calculate_var",
            "calculate_position_size",
            "generate_trading_signal",
            "execute_order",
            "risk_assessment",
            "portfolio_rebalancing",
            "stop_loss_placement",
            "emergency_action"
        ]
        
        for func_name in critical_functions:
            # Register with error handler
            self.error_handler.register_mandatory_response_function(
                func_name,
                validator=self._create_response_validator(func_name)
            )
            
            self.logger.info(f"Registered critical trading function: {func_name}")
    
    def _create_response_validator(self, function_name: str) -> Callable:
        """Create response validator for specific function type."""
        
        def validate_response(response: Any) -> bool:
            """Validate response based on function type."""
            
            if response is None:
                return False
            
            # VaR calculation validation
            if "var" in function_name.lower():
                return (hasattr(response, 'portfolio_var') and 
                       response.portfolio_var is not None and 
                       response.portfolio_var > 0)
            
            # Position sizing validation
            if "position" in function_name.lower():
                return (isinstance(response, (int, float)) and 
                       response > 0)
            
            # Signal generation validation
            if "signal" in function_name.lower():
                return (hasattr(response, 'signal_strength') or 
                       isinstance(response, dict) and 'signal' in response)
            
            # Order execution validation
            if "execute" in function_name.lower():
                return (hasattr(response, 'order_id') or 
                       isinstance(response, dict) and 'status' in response)
            
            # Risk assessment validation
            if "risk" in function_name.lower():
                return (hasattr(response, 'risk_score') or 
                       isinstance(response, dict) and 'risk' in response)
            
            # Default validation - just check not None
            return response is not None
        
        return validate_response
    
    def trading_function(
        self,
        decision_type: TradingDecisionType,
        agent_id: str,
        strategy_id: str,
        mandatory_response: bool = True,
        timeout_seconds: float = 30.0,
        **kwargs
    ):
        """
        Decorator for trading functions with integrated error handling and logging.
        
        Args:
            decision_type: Type of trading decision
            agent_id: Agent making the decision
            strategy_id: Strategy being executed
            mandatory_response: Whether function must return valid response
            timeout_seconds: Function timeout
            **kwargs: Additional context for decision logging
        """
        
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **func_kwargs):
                return self._execute_trading_function(
                    func, decision_type, agent_id, strategy_id,
                    mandatory_response, timeout_seconds, kwargs,
                    *args, **func_kwargs
                )
            return wrapper
        return decorator
    
    def async_trading_function(
        self,
        decision_type: TradingDecisionType,
        agent_id: str,
        strategy_id: str,
        mandatory_response: bool = True,
        timeout_seconds: float = 30.0,
        **kwargs
    ):
        """
        Async decorator for trading functions with integrated error handling and logging.
        """
        
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **func_kwargs):
                return await self._execute_async_trading_function(
                    func, decision_type, agent_id, strategy_id,
                    mandatory_response, timeout_seconds, kwargs,
                    *args, **func_kwargs
                )
            return wrapper
        return decorator
    
    def _execute_trading_function(
        self,
        func: Callable,
        decision_type: TradingDecisionType,
        agent_id: str,
        strategy_id: str,
        mandatory_response: bool,
        timeout_seconds: float,
        context_kwargs: Dict[str, Any],
        *args,
        **kwargs
    ):
        """Execute trading function with comprehensive error handling and logging."""
        
        function_name = f"{func.__module__}.{func.__name__}"
        
        # Set up correlation context
        with correlation_context.context(
            component=LogComponent.TRADING_ENGINE,
            strategy_id=strategy_id
        ):
            # Use trading decision logger context
            with self.trading_logger.decision_context(
                decision_type=decision_type,
                agent_id=agent_id,
                strategy_id=strategy_id,
                **context_kwargs
            ) as tracker:
                
                # Track function call
                self.error_handler.track_function_call(function_name)
                
                # Set decision logic
                logic = f"{func.__name__}: {func.__doc__ or 'No description available'}"
                tracker.set_decision_logic(logic)
                
                # Capture inputs
                inputs = {"args": args, "kwargs": kwargs}
                tracker.set_inputs(inputs)
                
                start_time = time.time()
                
                try:
                    # Execute with timeout if specified
                    if timeout_seconds > 0:
                        import signal
                        
                        def timeout_handler(signum, frame):
                            raise TimeoutError(f"Function {function_name} timed out after {timeout_seconds} seconds")
                        
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(int(timeout_seconds))
                    
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    if timeout_seconds > 0:
                        signal.alarm(0)  # Cancel alarm
                    
                    execution_time = (time.time() - start_time) * 1000
                    
                    # Validate response
                    if mandatory_response and not self.error_handler.validate_response(function_name, result):
                        tracker.set_outcome(TradingDecisionOutcome.FAILURE)
                        
                        # Create silent failure error
                        error = SystemError(
                            message=f"Silent failure detected in {function_name}: invalid response",
                            context=ErrorContext(
                                additional_data={
                                    "function_name": function_name,
                                    "response": str(result),
                                    "mandatory_response": mandatory_response
                                }
                            )
                        )
                        
                        self.error_handler.handle_exception(error, function_name=function_name)
                        return None
                    
                    # Record successful execution
                    tracker.set_outputs({"result": result})
                    tracker.update_metrics(
                        decision_latency_ms=execution_time,
                        confidence_score=0.9,  # High confidence for successful execution
                        risk_score=0.1,  # Low risk for successful execution
                        expected_return=0.0,  # Default values
                        expected_risk=0.0
                    )
                    tracker.set_outcome(TradingDecisionOutcome.SUCCESS)
                    
                    return result
                
                except Exception as e:
                    if timeout_seconds > 0:
                        signal.alarm(0)  # Cancel alarm
                    
                    execution_time = (time.time() - start_time) * 1000
                    
                    # Update metrics with error information
                    tracker.update_metrics(
                        decision_latency_ms=execution_time,
                        confidence_score=0.0,  # No confidence on error
                        risk_score=1.0,  # High risk on error
                        expected_return=0.0,
                        expected_risk=1.0
                    )
                    
                    # Set failure outcome
                    tracker.set_outcome(TradingDecisionOutcome.FAILURE)
                    
                    # Create error context
                    error_context = ErrorContext(
                        additional_data={
                            "function_name": function_name,
                            "decision_type": decision_type.value,
                            "agent_id": agent_id,
                            "strategy_id": strategy_id,
                            "execution_time_ms": execution_time,
                            "inputs": inputs
                        }
                    )
                    
                    # Handle error
                    if isinstance(e, BaseGrandModelError):
                        result = self.error_handler.handle_exception(e, error_context, function_name=function_name)
                    else:
                        # Convert to system error
                        system_error = SystemError(
                            message=f"Trading function {function_name} failed: {str(e)}",
                            context=error_context,
                            cause=e,
                            recoverable=True
                        )
                        result = self.error_handler.handle_exception(system_error, error_context, function_name=function_name)
                    
                    # If error handling provided a result, return it
                    if result is not None:
                        tracker.set_outcome(TradingDecisionOutcome.PARTIAL_SUCCESS)
                        return result
                    
                    # Re-raise if no recovery
                    raise
    
    async def _execute_async_trading_function(
        self,
        func: Callable,
        decision_type: TradingDecisionType,
        agent_id: str,
        strategy_id: str,
        mandatory_response: bool,
        timeout_seconds: float,
        context_kwargs: Dict[str, Any],
        *args,
        **kwargs
    ):
        """Execute async trading function with comprehensive error handling and logging."""
        
        function_name = f"{func.__module__}.{func.__name__}"
        
        # Set up correlation context
        with correlation_context.context(
            component=LogComponent.TRADING_ENGINE,
            strategy_id=strategy_id
        ):
            # Use trading decision logger context
            with self.trading_logger.decision_context(
                decision_type=decision_type,
                agent_id=agent_id,
                strategy_id=strategy_id,
                **context_kwargs
            ) as tracker:
                
                # Track function call
                self.error_handler.track_function_call(function_name)
                
                # Set decision logic
                logic = f"{func.__name__}: {func.__doc__ or 'No description available'}"
                tracker.set_decision_logic(logic)
                
                # Capture inputs
                inputs = {"args": args, "kwargs": kwargs}
                tracker.set_inputs(inputs)
                
                start_time = time.time()
                
                try:
                    # Execute with timeout
                    if timeout_seconds > 0:
                        result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
                    else:
                        result = await func(*args, **kwargs)
                    
                    execution_time = (time.time() - start_time) * 1000
                    
                    # Validate response
                    if mandatory_response and not self.error_handler.validate_response(function_name, result):
                        tracker.set_outcome(TradingDecisionOutcome.FAILURE)
                        
                        # Create silent failure error
                        error = SystemError(
                            message=f"Silent failure detected in {function_name}: invalid response",
                            context=ErrorContext(
                                additional_data={
                                    "function_name": function_name,
                                    "response": str(result),
                                    "mandatory_response": mandatory_response
                                }
                            )
                        )
                        
                        self.error_handler.handle_exception(error, function_name=function_name)
                        return None
                    
                    # Record successful execution
                    tracker.set_outputs({"result": result})
                    tracker.update_metrics(
                        decision_latency_ms=execution_time,
                        confidence_score=0.9,
                        risk_score=0.1,
                        expected_return=0.0,
                        expected_risk=0.0
                    )
                    tracker.set_outcome(TradingDecisionOutcome.SUCCESS)
                    
                    return result
                
                except Exception as e:
                    execution_time = (time.time() - start_time) * 1000
                    
                    # Update metrics with error information
                    tracker.update_metrics(
                        decision_latency_ms=execution_time,
                        confidence_score=0.0,
                        risk_score=1.0,
                        expected_return=0.0,
                        expected_risk=1.0
                    )
                    
                    # Set failure outcome
                    tracker.set_outcome(TradingDecisionOutcome.FAILURE)
                    
                    # Create error context
                    error_context = ErrorContext(
                        additional_data={
                            "function_name": function_name,
                            "decision_type": decision_type.value,
                            "agent_id": agent_id,
                            "strategy_id": strategy_id,
                            "execution_time_ms": execution_time,
                            "inputs": inputs
                        }
                    )
                    
                    # Handle error
                    if isinstance(e, BaseGrandModelError):
                        result = self.error_handler.handle_exception(e, error_context, function_name=function_name)
                    else:
                        # Convert to system error
                        system_error = SystemError(
                            message=f"Async trading function {function_name} failed: {str(e)}",
                            context=error_context,
                            cause=e,
                            recoverable=True
                        )
                        result = self.error_handler.handle_exception(system_error, error_context, function_name=function_name)
                    
                    # If error handling provided a result, return it
                    if result is not None:
                        tracker.set_outcome(TradingDecisionOutcome.PARTIAL_SUCCESS)
                        return result
                    
                    # Re-raise if no recovery
                    raise
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get integration statistics combining trading and error data."""
        
        trading_stats = self.trading_logger.get_decision_statistics()
        error_stats = self.error_handler.get_error_statistics()
        
        return {
            "trading_decisions": trading_stats,
            "error_handling": error_stats,
            "integration_health": {
                "total_functions_tracked": len(self.error_handler.mandatory_response_functions),
                "silent_failures_prevented": error_stats.get("silent_failures", 0),
                "successful_recoveries": error_stats.get("recovery_successes", 0),
                "error_to_decision_ratio": (
                    error_stats.get("total_errors", 0) / max(trading_stats.get("total_decisions", 1), 1)
                )
            }
        }


# Global integrator instance
_trading_error_integrator = None


def get_trading_error_integrator() -> TradingErrorIntegrator:
    """Get global trading error integrator instance."""
    global _trading_error_integrator
    if _trading_error_integrator is None:
        _trading_error_integrator = TradingErrorIntegrator()
    return _trading_error_integrator


# Convenience decorators using global integrator
def trading_function(
    decision_type: TradingDecisionType,
    agent_id: str,
    strategy_id: str,
    mandatory_response: bool = True,
    timeout_seconds: float = 30.0,
    **kwargs
):
    """Convenience decorator for trading functions."""
    integrator = get_trading_error_integrator()
    return integrator.trading_function(
        decision_type, agent_id, strategy_id, mandatory_response, timeout_seconds, **kwargs
    )


def async_trading_function(
    decision_type: TradingDecisionType,
    agent_id: str,
    strategy_id: str,
    mandatory_response: bool = True,
    timeout_seconds: float = 30.0,
    **kwargs
):
    """Convenience decorator for async trading functions."""
    integrator = get_trading_error_integrator()
    return integrator.async_trading_function(
        decision_type, agent_id, strategy_id, mandatory_response, timeout_seconds, **kwargs
    )


if __name__ == "__main__":
    # Demo usage
    integrator = TradingErrorIntegrator()
    
    @integrator.trading_function(
        decision_type=TradingDecisionType.POSITION_SIZING,
        agent_id="risk_agent",
        strategy_id="momentum_strategy",
        symbol="BTCUSD"
    )
    def calculate_position_size(symbol: str, portfolio_value: float, risk_percentage: float) -> float:
        """Calculate position size based on risk parameters."""
        if risk_percentage <= 0:
            raise ValueError("Risk percentage must be positive")
        
        return portfolio_value * risk_percentage
    
    # Test the function
    try:
        position_size = calculate_position_size("BTCUSD", 100000.0, 0.02)
        print(f"Position size: ${position_size}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Get integration statistics
    stats = integrator.get_integration_statistics()
    print("Integration Statistics:", stats)