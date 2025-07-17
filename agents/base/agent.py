from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
import logging
from datetime import datetime
import uuid
import time
import traceback
from contextlib import contextmanager
from functools import wraps
from enum import Enum

from src.core.errors.error_handler import ErrorHandler, RetryConfig, CircuitBreakerConfig, FallbackManager
from src.core.errors.base_exceptions import (
    BaseGrandModelError, ErrorSeverity, ErrorCategory, ErrorContext,
    ValidationError, DataError, SystemError, TimeoutError, NetworkError,
    RecoverableError, NonRecoverableError
)

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent operational status"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    DISABLED = "disabled"
    FAILED = "failed"


class BaseAgent(ABC):
    """Abstract base class for all trading agents with enhanced error handling"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.agent_id = str(uuid.uuid4())
        self.config = config
        self.state = {}
        self.status = AgentStatus.INITIALIZING
        self.created_at = datetime.utcnow()
        self.last_action_time = None
        self.last_error_time = None
        self.error_count = 0
        self.recovery_attempts = 0
        self.max_recovery_attempts = config.get('max_recovery_attempts', 3)
        
        # Initialize error handling
        self._setup_error_handling(config)
        
        # Health monitoring
        self.health_metrics = {
            'total_actions': 0,
            'successful_actions': 0,
            'failed_actions': 0,
            'recovery_successes': 0,
            'uptime_start': datetime.utcnow()
        }
        
        logger.info(f"Initialized agent {self.name} with ID {self.agent_id}")
    
    def _setup_error_handling(self, config: Dict[str, Any]):
        """Setup comprehensive error handling for the agent"""
        # Configure retry behavior
        retry_config = RetryConfig(
            max_attempts=config.get('max_retry_attempts', 3),
            base_delay=config.get('base_retry_delay', 1.0),
            max_delay=config.get('max_retry_delay', 30.0),
            backoff_multiplier=config.get('backoff_multiplier', 2.0),
            jitter=config.get('retry_jitter', True),
            retriable_exceptions=[NetworkError, TimeoutError, RecoverableError]
        )
        
        # Configure circuit breaker
        circuit_config = CircuitBreakerConfig(
            failure_threshold=config.get('circuit_failure_threshold', 5),
            recovery_timeout=config.get('circuit_recovery_timeout', 30.0),
            name=f"{self.name}_circuit"
        )
        
        # Initialize fallback manager
        fallback_manager = FallbackManager()
        self._register_fallbacks(fallback_manager)
        
        # Create error handler
        self.error_handler = ErrorHandler(
            retry_config=retry_config,
            circuit_breaker_config=circuit_config,
            fallback_manager=fallback_manager
        )
        
        # Register mandatory response functions
        self._register_mandatory_responses()
    
    def _register_fallbacks(self, fallback_manager: FallbackManager):
        """Register fallback mechanisms for agent operations"""
        # Default fallback for observations
        fallback_manager.register_fallback(
            'observe_fallback',
            lambda: self._get_safe_default_observation()
        )
        
        # Default fallback for decisions
        fallback_manager.register_fallback(
            'decide_fallback',
            lambda: self._get_safe_default_decision()
        )
        
        # Default fallback for actions
        fallback_manager.register_fallback(
            'act_fallback',
            lambda decision: self._get_safe_default_action(decision)
        )
    
    def _register_mandatory_responses(self):
        """Register functions that must return valid responses"""
        self.error_handler.register_mandatory_response_function(
            'observe',
            lambda response: response is not None
        )
        self.error_handler.register_mandatory_response_function(
            'decide',
            lambda response: response is not None
        )
        self.error_handler.register_mandatory_response_function(
            'act',
            lambda response: response is not None and 'status' in response
        )
    
    def _get_safe_default_observation(self) -> Dict[str, Any]:
        """Get safe default observation for fallback"""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'fallback',
            'data': {},
            'error': 'Using fallback observation due to error'
        }
    
    def _get_safe_default_decision(self) -> Dict[str, Any]:
        """Get safe default decision for fallback"""
        return {
            'action': 'hold',
            'confidence': 0.0,
            'reasoning': 'Safe default decision due to error',
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _get_safe_default_action(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Get safe default action for fallback"""
        return {
            'status': 'success',
            'action_taken': 'hold',
            'message': 'Safe default action executed due to error',
            'timestamp': datetime.utcnow().isoformat()
        }
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize agent resources and connections"""
        pass
    
    @abstractmethod
    def _observe_impl(self, market_data: Dict[str, Any]) -> None:
        """Internal observation implementation - to be overridden by subclasses"""
        pass
    
    @abstractmethod
    def _decide_impl(self) -> Optional[Dict[str, Any]]:
        """Internal decision implementation - to be overridden by subclasses"""
        pass
    
    @abstractmethod
    def _act_impl(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Internal action implementation - to be overridden by subclasses"""
        pass
    
    def observe(self, market_data: Dict[str, Any]) -> None:
        """Process incoming market data with error handling"""
        try:
            self.health_metrics['total_actions'] += 1
            self.error_handler.track_function_call('observe')
            
            # Validate input
            if not self._validate_market_data(market_data):
                raise ValidationError(
                    "Invalid market data format",
                    field="market_data",
                    value=market_data
                )
            
            # Execute with retry logic
            result = self.error_handler.execute_with_retry(
                self._observe_impl,
                market_data
            )
            
            # Validate response
            if not self.error_handler.validate_response('observe', result):
                raise ValidationError("Invalid observation response")
            
            self.health_metrics['successful_actions'] += 1
            self.last_action_time = datetime.utcnow()
            
            # Update status if recovering
            if self.status == AgentStatus.RECOVERING:
                self.status = AgentStatus.ACTIVE
                self.recovery_attempts = 0
                self.health_metrics['recovery_successes'] += 1
                logger.info(f"Agent {self.name} successfully recovered")
            
        except Exception as e:
            self._handle_operation_error(e, 'observe', fallback_name='observe_fallback')
    
    def decide(self) -> Optional[Dict[str, Any]]:
        """Make trading decision based on observations with error handling"""
        try:
            self.health_metrics['total_actions'] += 1
            self.error_handler.track_function_call('decide')
            
            # Execute with retry logic
            result = self.error_handler.execute_with_retry(self._decide_impl)
            
            # Validate response
            if not self.error_handler.validate_response('decide', result):
                raise ValidationError("Invalid decision response")
            
            self.health_metrics['successful_actions'] += 1
            self.last_action_time = datetime.utcnow()
            
            # Update status if recovering
            if self.status == AgentStatus.RECOVERING:
                self.status = AgentStatus.ACTIVE
                self.recovery_attempts = 0
                self.health_metrics['recovery_successes'] += 1
                logger.info(f"Agent {self.name} successfully recovered")
            
            return result
            
        except Exception as e:
            return self._handle_operation_error(e, 'decide', fallback_name='decide_fallback')
    
    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trading action based on decision with error handling"""
        try:
            self.health_metrics['total_actions'] += 1
            self.error_handler.track_function_call('act')
            
            # Validate input
            if not self._validate_decision(decision):
                raise ValidationError(
                    "Invalid decision format",
                    field="decision",
                    value=decision
                )
            
            # Execute with retry logic
            result = self.error_handler.execute_with_retry(
                self._act_impl,
                decision
            )
            
            # Validate response
            if not self.error_handler.validate_response('act', result):
                raise ValidationError("Invalid action response")
            
            self.health_metrics['successful_actions'] += 1
            self.last_action_time = datetime.utcnow()
            
            # Update status if recovering
            if self.status == AgentStatus.RECOVERING:
                self.status = AgentStatus.ACTIVE
                self.recovery_attempts = 0
                self.health_metrics['recovery_successes'] += 1
                logger.info(f"Agent {self.name} successfully recovered")
            
            return result
            
        except Exception as e:
            return self._handle_operation_error(e, 'act', fallback_name='act_fallback', decision=decision)
    
    def _validate_market_data(self, market_data: Dict[str, Any]) -> bool:
        """Validate market data input"""
        if market_data is None:
            return False
        if not isinstance(market_data, dict):
            return False
        # Add more specific validation based on requirements
        return True
    
    def _validate_decision(self, decision: Dict[str, Any]) -> bool:
        """Validate decision input"""
        if decision is None:
            return False
        if not isinstance(decision, dict):
            return False
        # Add more specific validation based on requirements
        return True
    
    def _handle_operation_error(self, error: Exception, operation: str, fallback_name: Optional[str] = None, **kwargs):
        """Handle operation errors with recovery and fallback"""
        self.error_count += 1
        self.last_error_time = datetime.utcnow()
        self.health_metrics['failed_actions'] += 1
        
        # Create error context
        context = ErrorContext(
            service_name=self.name,
            additional_data={
                'operation': operation,
                'agent_id': self.agent_id,
                'error_count': self.error_count,
                'status': self.status.value
            }
        )
        
        # Try to recover
        recovery_result = self._attempt_recovery(error, operation, context)
        if recovery_result is not None:
            return recovery_result
        
        # Handle with error handler (includes fallback)
        return self.error_handler.handle_exception(
            error,
            context=context,
            fallback_name=fallback_name,
            function_name=operation,
            **kwargs
        )
    
    def _attempt_recovery(self, error: Exception, operation: str, context: ErrorContext):
        """Attempt to recover from error"""
        if self.recovery_attempts >= self.max_recovery_attempts:
            logger.error(f"Maximum recovery attempts reached for {self.name}")
            self.status = AgentStatus.FAILED
            return None
        
        # Only attempt recovery for certain types of errors
        if not isinstance(error, (NetworkError, TimeoutError, RecoverableError)):
            return None
        
        self.recovery_attempts += 1
        self.status = AgentStatus.RECOVERING
        
        logger.info(f"Attempting recovery for {self.name} (attempt {self.recovery_attempts}/{self.max_recovery_attempts})")
        
        try:
            # Give the system time to recover
            time.sleep(min(self.recovery_attempts * 2, 10))
            
            # Try to reinitialize if needed
            if hasattr(self, 'reinitialize'):
                self.reinitialize()
            
            return None  # Let the normal retry mechanism handle it
            
        except Exception as recovery_error:
            logger.error(f"Recovery attempt failed for {self.name}: {recovery_error}")
            return None
    
    def update_state(self, key: str, value: Any) -> None:
        """Update agent's internal state"""
        self.state[key] = value
        logger.debug(f"Agent {self.name} state updated: {key} = {value}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state"""
        return self.state.copy()
    
    def activate(self) -> None:
        """Activate the agent"""
        if self.status != AgentStatus.FAILED:
            self.status = AgentStatus.ACTIVE
            logger.info(f"Agent {self.name} activated")
        else:
            logger.warning(f"Cannot activate failed agent {self.name}")
    
    def deactivate(self) -> None:
        """Deactivate the agent"""
        self.status = AgentStatus.DISABLED
        logger.info(f"Agent {self.name} deactivated")
    
    def is_active(self) -> bool:
        """Check if agent is active"""
        return self.status == AgentStatus.ACTIVE
    
    def is_healthy(self) -> bool:
        """Check if agent is healthy"""
        return self.status in [AgentStatus.ACTIVE, AgentStatus.RECOVERING]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive agent performance metrics"""
        uptime = datetime.utcnow() - self.health_metrics['uptime_start']
        
        success_rate = (
            self.health_metrics['successful_actions'] / self.health_metrics['total_actions']
            if self.health_metrics['total_actions'] > 0 else 0.0
        )
        
        error_rate = (
            self.health_metrics['failed_actions'] / self.health_metrics['total_actions']
            if self.health_metrics['total_actions'] > 0 else 0.0
        )
        
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'last_action_time': self.last_action_time.isoformat() if self.last_action_time else None,
            'last_error_time': self.last_error_time.isoformat() if self.last_error_time else None,
            'uptime_seconds': uptime.total_seconds(),
            'health_metrics': self.health_metrics.copy(),
            'success_rate': success_rate,
            'error_rate': error_rate,
            'error_count': self.error_count,
            'recovery_attempts': self.recovery_attempts,
            'max_recovery_attempts': self.max_recovery_attempts,
            'error_handler_stats': self.error_handler.get_error_statistics(),
            'circuit_breaker_state': (
                self.error_handler.circuit_breaker.get_state() 
                if self.error_handler.circuit_breaker else None
            )
        }
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        metrics = self.get_metrics()
        error_handler_health = self.error_handler.get_health_report()
        
        # Calculate overall health score
        health_score = 100
        
        # Deduct for errors
        if metrics['error_rate'] > 0.1:  # 10% error rate threshold
            health_score -= min(metrics['error_rate'] * 100, 40)
        
        # Deduct for status issues
        if self.status == AgentStatus.DEGRADED:
            health_score -= 20
        elif self.status == AgentStatus.RECOVERING:
            health_score -= 30
        elif self.status == AgentStatus.FAILED:
            health_score = 0
        
        # Factor in error handler health
        health_score = (health_score * 0.7) + (error_handler_health['health_score'] * 0.3)
        
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'health_score': max(0, min(100, health_score)),
            'status': self.status.value,
            'metrics': metrics,
            'error_handler_health': error_handler_health,
            'recommendations': self._generate_health_recommendations(metrics)
        }
    
    def _generate_health_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate health recommendations"""
        recommendations = []
        
        if metrics['error_rate'] > 0.1:
            recommendations.append(f"High error rate ({metrics['error_rate']:.1%}) - investigate error patterns")
        
        if self.recovery_attempts > 0:
            recommendations.append(f"Agent has attempted recovery {self.recovery_attempts} times - monitor stability")
        
        if metrics['success_rate'] < 0.8:
            recommendations.append(f"Low success rate ({metrics['success_rate']:.1%}) - review operation logic")
        
        if self.status == AgentStatus.DEGRADED:
            recommendations.append("Agent is in degraded state - consider restart or reconfiguration")
        
        if self.error_count > 50:
            recommendations.append("High error count - investigate root causes")
        
        return recommendations
    
    @contextmanager
    def operation_context(self, operation_name: str):
        """Context manager for operation-level error handling"""
        start_time = time.time()
        try:
            logger.debug(f"Starting operation {operation_name} for agent {self.name}")
            yield
            logger.debug(f"Completed operation {operation_name} for agent {self.name}")
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Operation {operation_name} failed for agent {self.name} after {duration:.2f}s: {e}")
            raise
    
    def with_error_handling(self, operation_name: str, fallback_name: Optional[str] = None):
        """Decorator factory for adding error handling to methods"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    return self._handle_operation_error(
                        e, operation_name, fallback_name=fallback_name, *args, **kwargs
                    )
            return wrapper
        return decorator
    
    def reset_error_state(self):
        """Reset error state for recovery"""
        self.error_count = 0
        self.recovery_attempts = 0
        self.last_error_time = None
        if self.status == AgentStatus.FAILED:
            self.status = AgentStatus.ACTIVE
        logger.info(f"Error state reset for agent {self.name}")
    
    def force_recovery(self):
        """Force agent recovery"""
        try:
            self.status = AgentStatus.RECOVERING
            self.recovery_attempts = 0
            
            # Reset error handler state
            if hasattr(self.error_handler, 'circuit_breaker') and self.error_handler.circuit_breaker:
                self.error_handler.circuit_breaker.failure_count = 0
                self.error_handler.circuit_breaker.state = self.error_handler.circuit_breaker.CircuitBreakerState.CLOSED
            
            # Try to reinitialize
            if hasattr(self, 'reinitialize'):
                self.reinitialize()
            else:
                self.initialize()
            
            self.status = AgentStatus.ACTIVE
            self.health_metrics['recovery_successes'] += 1
            logger.info(f"Force recovery successful for agent {self.name}")
            
        except Exception as e:
            self.status = AgentStatus.FAILED
            logger.error(f"Force recovery failed for agent {self.name}: {e}")
            raise


class TradingAgent(BaseAgent):
    """Base class for trading-specific agents with enhanced error handling"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.positions = {}
        self.orders = []
        self.pnl = 0.0
        self.risk_limits = config.get('risk_limits', {})
        self.last_valid_positions = {}  # Backup for recovery
        self.position_lock = threading.Lock()
        
        # Trading-specific error handling
        self._register_trading_fallbacks()
        self._register_trading_mandatory_responses()
    
    def _register_trading_fallbacks(self):
        """Register trading-specific fallbacks"""
        self.error_handler.fallback_manager.register_fallback(
            'risk_check_fallback',
            lambda order: self._safe_risk_check(order)
        )
        
        self.error_handler.fallback_manager.register_fallback(
            'position_update_fallback',
            lambda fills: self._safe_position_update(fills)
        )
    
    def _register_trading_mandatory_responses(self):
        """Register trading-specific mandatory response functions"""
        self.error_handler.register_mandatory_response_function(
            'check_risk_limits',
            lambda response: isinstance(response, bool)
        )
        
        self.error_handler.register_mandatory_response_function(
            'update_positions',
            lambda response: response is not None
        )
    
    def _safe_risk_check(self, order: Dict[str, Any]) -> bool:
        """Safe fallback for risk checking"""
        logger.warning(f"Using fallback risk check for order: {order}")
        # Conservative approach - reject all orders in fallback mode
        return False
    
    def _safe_position_update(self, fills: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Safe fallback for position updates"""
        logger.warning(f"Using fallback position update for fills: {fills}")
        return {
            'status': 'fallback',
            'message': 'Position update fallback executed',
            'positions_updated': 0
        }
    
    def check_risk_limits(self, order: Dict[str, Any]) -> bool:
        """Check if order complies with risk limits with error handling"""
        try:
            self.error_handler.track_function_call('check_risk_limits')
            
            # Validate order format
            if not self._validate_order_format(order):
                raise ValidationError("Invalid order format", field="order", value=order)
            
            # Execute risk check with retry
            result = self.error_handler.execute_with_retry(
                self._check_risk_limits_impl,
                order
            )
            
            # Validate response
            if not self.error_handler.validate_response('check_risk_limits', result):
                raise ValidationError("Invalid risk check response")
            
            return result
            
        except Exception as e:
            return self.error_handler.handle_exception(
                e,
                fallback_name='risk_check_fallback',
                function_name='check_risk_limits',
                order=order
            )
    
    def _check_risk_limits_impl(self, order: Dict[str, Any]) -> bool:
        """Internal implementation of risk limit checking"""
        max_position = self.risk_limits.get('max_position_size', float('inf'))
        max_loss = self.risk_limits.get('max_daily_loss', float('inf'))
        max_order_value = self.risk_limits.get('max_order_value', float('inf'))
        
        # Basic checks
        order_quantity = abs(order.get('quantity', 0))
        order_price = order.get('price', 0)
        order_value = order_quantity * order_price
        
        if order_quantity > max_position:
            logger.warning(f"Order exceeds max position size: {order}")
            return False
        
        if order_value > max_order_value:
            logger.warning(f"Order exceeds max order value: {order}")
            return False
        
        if self.pnl < -max_loss:
            logger.warning(f"Daily loss limit reached: {self.pnl}")
            return False
        
        # Check position limits
        symbol = order.get('symbol')
        if symbol and symbol in self.positions:
            current_quantity = self.positions[symbol]['quantity']
            new_quantity = current_quantity + order.get('quantity', 0)
            
            if abs(new_quantity) > max_position:
                logger.warning(f"Position limit would be exceeded for {symbol}: {new_quantity}")
                return False
        
        return True
    
    def _validate_order_format(self, order: Dict[str, Any]) -> bool:
        """Validate order format"""
        required_fields = ['symbol', 'quantity', 'price']
        
        if not isinstance(order, dict):
            return False
        
        for field in required_fields:
            if field not in order:
                return False
        
        # Validate types
        if not isinstance(order['symbol'], str):
            return False
        
        if not isinstance(order['quantity'], (int, float)):
            return False
        
        if not isinstance(order['price'], (int, float)):
            return False
        
        # Validate values
        if order['quantity'] == 0:
            return False
        
        if order['price'] <= 0:
            return False
        
        return True
    
    def update_positions(self, fills: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update positions based on filled orders with error handling"""
        try:
            self.error_handler.track_function_call('update_positions')
            
            # Validate fills format
            if not self._validate_fills_format(fills):
                raise ValidationError("Invalid fills format", field="fills", value=fills)
            
            # Execute position update with retry
            result = self.error_handler.execute_with_retry(
                self._update_positions_impl,
                fills
            )
            
            # Validate response
            if not self.error_handler.validate_response('update_positions', result):
                raise ValidationError("Invalid position update response")
            
            return result
            
        except Exception as e:
            return self.error_handler.handle_exception(
                e,
                fallback_name='position_update_fallback',
                function_name='update_positions',
                fills=fills
            )
    
    def _update_positions_impl(self, fills: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Internal implementation of position updates"""
        with self.position_lock:
            # Backup current positions
            self.last_valid_positions = self.positions.copy()
            
            updated_symbols = set()
            
            for fill in fills:
                symbol = fill['symbol']
                quantity = fill['quantity']
                price = fill['price']
                
                if symbol not in self.positions:
                    self.positions[symbol] = {'quantity': 0, 'avg_price': 0}
                
                current_pos = self.positions[symbol]
                new_quantity = current_pos['quantity'] + quantity
                
                if new_quantity == 0:
                    self.positions[symbol] = {'quantity': 0, 'avg_price': 0}
                else:
                    # Update average price
                    total_cost = (current_pos['quantity'] * current_pos['avg_price'] + 
                                 quantity * price)
                    self.positions[symbol] = {
                        'quantity': new_quantity,
                        'avg_price': total_cost / new_quantity
                    }
                
                updated_symbols.add(symbol)
            
            logger.info(f"Updated positions for {self.name}: {self.positions}")
            
            return {
                'status': 'success',
                'updated_symbols': list(updated_symbols),
                'positions_updated': len(updated_symbols),
                'total_positions': len(self.positions)
            }
    
    def _validate_fills_format(self, fills: List[Dict[str, Any]]) -> bool:
        """Validate fills format"""
        if not isinstance(fills, list):
            return False
        
        required_fields = ['symbol', 'quantity', 'price']
        
        for fill in fills:
            if not isinstance(fill, dict):
                return False
            
            for field in required_fields:
                if field not in fill:
                    return False
            
            # Validate types
            if not isinstance(fill['symbol'], str):
                return False
            
            if not isinstance(fill['quantity'], (int, float)):
                return False
            
            if not isinstance(fill['price'], (int, float)):
                return False
            
            # Validate values
            if fill['quantity'] == 0:
                return False
            
            if fill['price'] <= 0:
                return False
        
        return True
    
    def recover_positions(self):
        """Recover positions from backup"""
        try:
            with self.position_lock:
                if self.last_valid_positions:
                    self.positions = self.last_valid_positions.copy()
                    logger.info(f"Recovered positions for {self.name}: {self.positions}")
                else:
                    logger.warning(f"No valid position backup available for {self.name}")
        except Exception as e:
            logger.error(f"Position recovery failed for {self.name}: {e}")
    
    def get_trading_metrics(self) -> Dict[str, Any]:
        """Get trading-specific metrics"""
        base_metrics = self.get_metrics()
        
        return {
            **base_metrics,
            'positions': self.positions.copy(),
            'total_positions': len(self.positions),
            'active_positions': len([p for p in self.positions.values() if p['quantity'] != 0]),
            'pnl': self.pnl,
            'orders_count': len(self.orders),
            'risk_limits': self.risk_limits.copy(),
            'has_position_backup': bool(self.last_valid_positions)
        }


class MultiTimeframeAgent(TradingAgent):
    """Agent that operates on multiple timeframes"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.timeframes = config.get('timeframes', ['5m', '30m', '1h'])
        self.data_buffers = {tf: [] for tf in self.timeframes}
        self.indicators = {tf: {} for tf in self.timeframes}
    
    def aggregate_signals(self) -> Dict[str, float]:
        """Aggregate signals from multiple timeframes"""
        # Implement signal aggregation logic
        signals = {}
        for tf in self.timeframes:
            tf_indicators = self.indicators.get(tf, {})
            # Process indicators for each timeframe
            # This is a placeholder - implement actual logic
            signals[tf] = 0.0
        
        return signals