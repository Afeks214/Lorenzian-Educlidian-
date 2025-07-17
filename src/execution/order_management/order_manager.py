"""
Order Manager

Ultra-high performance order management system designed for <500μs order placement
and >99.8% fill rate requirements. Handles complete order lifecycle with comprehensive
risk controls and execution tracking.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading
import structlog
from functools import wraps

from ..core.events import EventBus, Event, EventType
from .order_types import Order, OrderRequest, OrderUpdate, OrderStatus, OrderType
from .order_validator import OrderValidator, ValidationConfig
from .execution_tracker import ExecutionTracker
from ..routing.smart_router import SmartOrderRouter
from ..analytics.risk_manager import PreTradeRiskManager
from ..operations.operational_controls import OperationalControls
from ..safety.kill_switch import get_kill_switch

logger = structlog.get_logger()


def require_system_active(func):
    """
    Decorator to ensure system is active before executing trading functions.
    
    Checks both kill switch and operational controls to ensure system safety.
    Blocks execution if system is in emergency stop or maintenance mode.
    """
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        # Check kill switch first
        kill_switch = get_kill_switch()
        if kill_switch and kill_switch.is_active():
            logger.error(f"BLOCKED: {func.__name__} - Kill switch is active")
            # Don't execute the function, just return None or appropriate failure response
            if func.__name__ == 'submit_order':
                raise OrderSubmissionError("System is in emergency shutdown - kill switch active")
            return False
        
        # Check operational controls
        if hasattr(self, 'operational_controls') and self.operational_controls:
            if self.operational_controls.emergency_stop:
                logger.error(f"BLOCKED: {func.__name__} - Emergency stop is active")
                if func.__name__ == 'submit_order':
                    raise OrderSubmissionError("System is in emergency stop mode")
                return False
            
            if self.operational_controls.maintenance_mode:
                logger.warning(f"BLOCKED: {func.__name__} - System is in maintenance mode")
                if func.__name__ == 'submit_order':
                    raise OrderSubmissionError("System is in maintenance mode")
                return False
        
        # System is active, proceed with execution
        return await func(self, *args, **kwargs)
    
    return wrapper


@dataclass
class OrderManagerConfig:
    """Configuration for order manager"""
    
    # Performance settings
    max_concurrent_orders: int = 10000
    order_queue_size: int = 50000
    worker_threads: int = 8
    enable_fast_path: bool = True
    
    # Validation settings
    enable_pre_trade_risk: bool = True
    enable_real_time_validation: bool = True
    validation_timeout_ms: int = 10
    
    # Execution settings
    default_routing_strategy: str = "smart_router"
    enable_execution_tracking: bool = True
    auto_cancel_timeout_minutes: int = 1440  # 24 hours
    
    # Risk settings
    max_order_value: float = 10_000_000  # $10M
    daily_order_limit: int = 50000
    position_limit_check: bool = True
    
    # Monitoring
    enable_performance_monitoring: bool = True
    latency_alert_threshold_us: int = 1000
    fill_rate_alert_threshold: float = 0.95


class OrderManagerError(Exception):
    """Base exception for order manager errors"""
    pass


class OrderValidationError(OrderManagerError):
    """Order validation failed"""
    pass


class OrderSubmissionError(OrderManagerError):
    """Order submission failed"""
    pass


class OrderManager:
    """
    Ultra-high performance order manager for institutional trading.
    
    Designed to meet aggressive performance targets:
    - <500μs order placement latency
    - >99.8% fill rate achievement
    - Support for 10,000+ concurrent orders
    """
    
    def __init__(
        self,
        config: OrderManagerConfig,
        event_bus: EventBus,
        smart_router: SmartOrderRouter,
        validator: Optional[OrderValidator] = None,
        risk_manager: Optional[PreTradeRiskManager] = None,
        operational_controls: Optional[OperationalControls] = None
    ):
        self.config = config
        self.event_bus = event_bus
        self.smart_router = smart_router
        
        # Initialize safety controls
        self.operational_controls = operational_controls
        
        # Initialize components
        self.validator = validator or self._create_default_validator()
        self.risk_manager = risk_manager or self._create_default_risk_manager()
        self.execution_tracker = ExecutionTracker() if config.enable_execution_tracking else None
        
        # Order storage
        self.orders: Dict[str, Order] = {}
        self.pending_orders: Dict[str, Order] = {}
        self.active_orders: Dict[str, Order] = {}
        
        # Performance optimization
        self.order_queue = asyncio.Queue(maxsize=config.order_queue_size)
        self.executor = ThreadPoolExecutor(
            max_workers=config.worker_threads,
            thread_name_prefix="order_manager"
        )
        
        # Threading and synchronization
        self.lock = threading.RLock()
        self.shutdown_event = threading.Event()
        
        # Performance tracking
        self.performance_metrics = {
            'orders_submitted': 0,
            'orders_filled': 0,
            'orders_rejected': 0,
            'avg_submission_latency_us': 0.0,
            'avg_fill_latency_us': 0.0,
            'total_processing_time': 0.0
        }
        
        # Event subscriptions
        self._setup_event_subscriptions()
        
        # Background tasks
        self._start_background_tasks()
        
        logger.info(
            "OrderManager initialized with safety controls",
            max_concurrent_orders=config.max_concurrent_orders,
            worker_threads=config.worker_threads,
            enable_fast_path=config.enable_fast_path
        )
    
    def _create_default_validator(self) -> OrderValidator:
        """Create default order validator"""
        validation_config = ValidationConfig(
            max_order_value={'default': self.config.max_order_value},
            daily_order_limit=self.config.daily_order_limit,
            enable_risk_checks=self.config.enable_pre_trade_risk
        )
        return OrderValidator(validation_config)
    
    def _create_default_risk_manager(self) -> PreTradeRiskManager:
        """Create default risk manager"""
        from ..analytics.risk_manager import PreTradeRiskManager
        return PreTradeRiskManager({'max_order_value': self.config.max_order_value})
    
    def _setup_event_subscriptions(self) -> None:
        """Setup event bus subscriptions"""
        self.event_bus.subscribe(EventType.ORDER_FILLED, self._handle_order_filled)
        self.event_bus.subscribe(EventType.ORDER_CANCELLED, self._handle_order_cancelled)
        self.event_bus.subscribe(EventType.SYSTEM_SHUTDOWN, self._handle_system_shutdown)
    
    def _start_background_tasks(self) -> None:
        """Start background processing tasks"""
        if self.config.enable_performance_monitoring:
            self.executor.submit(self._performance_monitoring_loop)
        
        # Start order processing loop
        self.executor.submit(asyncio.run, self._order_processing_loop())
    
    @require_system_active
    async def submit_order(self, order_request: OrderRequest) -> str:
        """
        Submit order for execution with ultra-low latency path.
        
        Target: <500μs total execution time including validation and routing.
        """
        start_time = time.perf_counter()
        
        try:
            # Convert request to order
            order = order_request.to_order()
            
            # Fast path for high-priority orders
            if self.config.enable_fast_path and order.priority.value >= 3:
                return await self._fast_path_submission(order, start_time)
            
            # Standard submission path
            return await self._standard_submission(order, start_time)
            
        except Exception as e:
            submission_time = (time.perf_counter() - start_time) * 1_000_000
            logger.error(
                "Order submission failed",
                error=str(e),
                submission_time_us=submission_time
            )
            raise OrderSubmissionError(f"Order submission failed: {str(e)}")
    
    async def _fast_path_submission(self, order: Order, start_time: float) -> str:
        """Ultra-fast submission path for high-priority orders"""
        
        # Minimal validation for speed
        if not order.symbol or order.quantity <= 0:
            raise OrderValidationError("Invalid order: missing symbol or quantity")
        
        # Skip detailed validation for speed
        order.risk_checked = False
        order.compliance_checked = False
        
        # Immediate routing
        routing_result = await self.smart_router.route_order_fast(order)
        
        # Track submission
        if self.execution_tracker:
            self.execution_tracker.track_order_submission(order)
        
        # Store order
        with self.lock:
            self.orders[order.order_id] = order
            self.active_orders[order.order_id] = order
            order.update_status(OrderStatus.SUBMITTED)
        
        # Submit to venue
        await self._submit_to_venue(order, routing_result)
        
        # Calculate and log latency
        submission_latency = (time.perf_counter() - start_time) * 1_000_000
        order.order_placement_latency = submission_latency
        
        # Update performance metrics
        self._update_performance_metrics(submission_latency, fast_path=True)
        
        # Publish event
        self.event_bus.publish(
            Event(
                event_type=EventType.ORDER_SUBMITTED,
                timestamp=datetime.now(),
                payload={'order_id': order.order_id, 'fast_path': True},
                source='order_manager'
            )
        )
        
        logger.info(
            "Fast path order submitted",
            order_id=order.order_id,
            symbol=order.symbol,
            latency_us=submission_latency
        )
        
        return order.order_id
    
    async def _standard_submission(self, order: Order, start_time: float) -> str:
        """Standard submission path with full validation"""
        
        # Full order validation
        validation_start = time.perf_counter()
        validation_result = self.validator.validate_order(order)
        validation_time = (time.perf_counter() - validation_start) * 1_000_000
        
        if not validation_result.is_valid:
            raise OrderValidationError(f"Order validation failed: {', '.join(validation_result.errors)}")
        
        # Pre-trade risk check
        if self.config.enable_pre_trade_risk:
            risk_start = time.perf_counter()
            risk_result = await self.risk_manager.check_order_risk(order)
            risk_time = (time.perf_counter() - risk_start) * 1_000_000
            
            if not risk_result.approved:
                raise OrderValidationError(f"Risk check failed: {', '.join(risk_result.failed_checks)}")
            
            order.risk_checked = True
        
        # Smart routing
        routing_start = time.perf_counter()
        routing_result = await self.smart_router.route_order(order)
        routing_time = (time.perf_counter() - routing_start) * 1_000_000
        
        # Track submission
        if self.execution_tracker:
            self.execution_tracker.track_order_submission(order)
        
        # Store order
        with self.lock:
            self.orders[order.order_id] = order
            self.active_orders[order.order_id] = order
            order.update_status(OrderStatus.SUBMITTED)
        
        # Submit to venue
        submission_start = time.perf_counter()
        await self._submit_to_venue(order, routing_result)
        venue_time = (time.perf_counter() - submission_start) * 1_000_000
        
        # Calculate total latency
        total_latency = (time.perf_counter() - start_time) * 1_000_000
        order.order_placement_latency = total_latency
        
        # Update performance metrics
        self._update_performance_metrics(total_latency, fast_path=False)
        
        # Check latency target
        if total_latency > self.config.latency_alert_threshold_us:
            logger.warning(
                "Order submission latency exceeded threshold",
                order_id=order.order_id,
                latency_us=total_latency,
                threshold_us=self.config.latency_alert_threshold_us,
                breakdown={
                    'validation_us': validation_time,
                    'risk_check_us': risk_time if self.config.enable_pre_trade_risk else 0,
                    'routing_us': routing_time,
                    'venue_submission_us': venue_time
                }
            )
        
        # Publish event
        self.event_bus.publish(
            Event(
                event_type=EventType.ORDER_SUBMITTED,
                timestamp=datetime.now(),
                payload={
                    'order_id': order.order_id,
                    'latency_us': total_latency,
                    'validation_result': validation_result,
                    'routing_result': routing_result
                },
                source='order_manager'
            )
        )
        
        logger.info(
            "Standard order submitted",
            order_id=order.order_id,
            symbol=order.symbol,
            latency_us=total_latency,
            venue=routing_result.venue if hasattr(routing_result, 'venue') else 'unknown'
        )
        
        return order.order_id
    
    async def _submit_to_venue(self, order: Order, routing_result: Any) -> None:
        """Submit order to selected venue"""
        try:
            # This would integrate with actual venue APIs in production
            # For now, simulate venue submission
            await asyncio.sleep(0.001)  # Simulate 1ms venue latency
            
            # Update order with venue information
            order.actual_venue = getattr(routing_result, 'venue', 'SIMULATED')
            order.routing_strategy = getattr(routing_result, 'strategy', 'default')
            
        except Exception as e:
            logger.error(
                "Venue submission failed",
                order_id=order.order_id,
                error=str(e)
            )
            raise
    
    @require_system_active
    async def cancel_order(self, order_id: str, reason: str = "User requested") -> bool:
        """Cancel an active order"""
        with self.lock:
            order = self.active_orders.get(order_id)
            if not order:
                logger.warning("Cannot cancel order - not found or not active", order_id=order_id)
                return False
            
            if not order.is_active:
                logger.warning("Cannot cancel order - not in active state", order_id=order_id, status=order.status)
                return False
        
        try:
            # Cancel at venue (simulate)
            await asyncio.sleep(0.0005)  # Simulate 0.5ms cancellation latency
            
            # Update order status
            order.update_status(OrderStatus.CANCELLED)
            
            # Move from active to completed
            with self.lock:
                del self.active_orders[order_id]
            
            # Track completion
            if self.execution_tracker:
                self.execution_tracker.track_order_completion(order_id, OrderStatus.CANCELLED)
            
            # Publish event
            self.event_bus.publish(
                Event(
                    event_type=EventType.ORDER_CANCELLED,
                    timestamp=datetime.now(),
                    payload={'order_id': order_id, 'reason': reason},
                    source='order_manager'
                )
            )
            
            logger.info("Order cancelled", order_id=order_id, reason=reason)
            return True
            
        except Exception as e:
            logger.error("Order cancellation failed", order_id=order_id, error=str(e))
            return False
    
    @require_system_active
    async def modify_order(self, order_update: OrderUpdate) -> bool:
        """Modify an existing order"""
        with self.lock:
            order = self.active_orders.get(order_update.order_id)
            if not order:
                logger.warning("Cannot modify order - not found or not active", order_id=order_update.order_id)
                return False
        
        try:
            # Validate modification
            if order_update.new_quantity is not None:
                if order_update.new_quantity <= 0:
                    raise OrderValidationError("Invalid quantity for modification")
                order.quantity = order_update.new_quantity
                order.remaining_quantity = order_update.new_quantity - order.filled_quantity
            
            if order_update.new_price is not None:
                if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                    order.price = order_update.new_price
                else:
                    logger.warning("Price modification not applicable for order type", 
                                 order_type=order.order_type)
            
            # Submit modification to venue (simulate)
            await asyncio.sleep(0.0008)  # Simulate 0.8ms modification latency
            
            order.last_updated = datetime.now()
            
            logger.info(
                "Order modified",
                order_id=order_update.order_id,
                new_quantity=order_update.new_quantity,
                new_price=order_update.new_price
            )
            
            return True
            
        except Exception as e:
            logger.error("Order modification failed", order_id=order_update.order_id, error=str(e))
            return False
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        with self.lock:
            return self.orders.get(order_id)
    
    def get_active_orders(self, symbol: str = None) -> List[Order]:
        """Get all active orders, optionally filtered by symbol"""
        with self.lock:
            orders = list(self.active_orders.values())
            if symbol:
                orders = [o for o in orders if o.symbol == symbol]
            return orders
    
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed order status"""
        order = self.get_order(order_id)
        if not order:
            return None
        
        return {
            'order_id': order.order_id,
            'symbol': order.symbol,
            'status': order.status.value,
            'quantity': order.quantity,
            'filled_quantity': order.filled_quantity,
            'remaining_quantity': order.remaining_quantity,
            'average_fill_price': order.average_fill_price,
            'created_timestamp': order.created_timestamp.isoformat(),
            'last_updated': order.last_updated.isoformat(),
            'performance_metrics': order.get_performance_metrics(),
            'executions': [
                {
                    'execution_id': ex.execution_id,
                    'timestamp': ex.timestamp.isoformat(),
                    'price': ex.price,
                    'quantity': ex.quantity,
                    'venue': ex.venue
                }
                for ex in order.executions
            ]
        }
    
    async def _order_processing_loop(self) -> None:
        """Background order processing loop"""
        while not self.shutdown_event.is_set():
            try:
                # Process any queued orders
                await asyncio.sleep(0.001)  # 1ms processing cycle
                
                # Check for expired orders
                await self._check_expired_orders()
                
                # Cleanup completed orders older than 1 hour
                await self._cleanup_old_orders()
                
            except Exception as e:
                logger.error("Error in order processing loop", error=str(e))
                await asyncio.sleep(1.0)
    
    async def _check_expired_orders(self) -> None:
        """Check for and cancel expired orders"""
        current_time = datetime.now()
        expired_orders = []
        
        with self.lock:
            for order in self.active_orders.values():
                if order.expiry_time and current_time >= order.expiry_time:
                    expired_orders.append(order.order_id)
                # Auto-cancel old orders
                elif (current_time - order.created_timestamp).total_seconds() > self.config.auto_cancel_timeout_minutes * 60:
                    expired_orders.append(order.order_id)
        
        # Cancel expired orders
        for order_id in expired_orders:
            await self.cancel_order(order_id, "Expired")
    
    async def _cleanup_old_orders(self) -> None:
        """Cleanup old completed orders to manage memory"""
        cutoff_time = datetime.now() - asyncio.get_event_loop().time() + 3600  # 1 hour ago
        
        with self.lock:
            old_orders = [
                order_id for order_id, order in self.orders.items()
                if order.is_terminal and order.last_updated.timestamp() < cutoff_time
            ]
            
            for order_id in old_orders:
                del self.orders[order_id]
        
        if old_orders:
            logger.debug(f"Cleaned up {len(old_orders)} old orders")
    
    def _performance_monitoring_loop(self) -> None:
        """Background performance monitoring"""
        while not self.shutdown_event.is_set():
            try:
                # Get current metrics
                if self.execution_tracker:
                    metrics = self.execution_tracker.get_real_time_metrics()
                    
                    # Check performance targets
                    if not metrics['performance_targets']['meeting_latency_target']:
                        logger.warning("Latency target not being met", metrics=metrics['latency'])
                    
                    if not metrics['performance_targets']['meeting_fill_target']:
                        logger.warning("Fill rate target not being met", metrics=metrics['fill_rates'])
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error("Error in performance monitoring", error=str(e))
                time.sleep(30)
    
    def _update_performance_metrics(self, latency_us: float, fast_path: bool = False) -> None:
        """Update internal performance metrics"""
        with self.lock:
            self.performance_metrics['orders_submitted'] += 1
            
            # Update average latency
            current_avg = self.performance_metrics['avg_submission_latency_us']
            count = self.performance_metrics['orders_submitted']
            self.performance_metrics['avg_submission_latency_us'] = (
                (current_avg * (count - 1) + latency_us) / count
            )
    
    def _handle_order_filled(self, event: Event) -> None:
        """Handle order filled event"""
        order_id = event.payload.get('order_id')
        execution_data = event.payload.get('execution')
        
        if order_id and execution_data:
            if self.execution_tracker:
                from .order_types import OrderExecution
                execution = OrderExecution(
                    execution_id=execution_data['execution_id'],
                    timestamp=datetime.fromisoformat(execution_data['timestamp']),
                    price=execution_data['price'],
                    quantity=execution_data['quantity'],
                    venue=execution_data['venue']
                )
                self.execution_tracker.track_order_execution(order_id, execution)
    
    def _handle_order_cancelled(self, event: Event) -> None:
        """Handle order cancelled event"""
        order_id = event.payload.get('order_id')
        if order_id and self.execution_tracker:
            self.execution_tracker.track_order_completion(order_id, OrderStatus.CANCELLED)
    
    def _handle_system_shutdown(self, event: Event) -> None:
        """Handle system shutdown event"""
        logger.info("Received shutdown signal")
        self.shutdown()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        with self.lock:
            # Check system state
            system_state = "active"
            kill_switch = get_kill_switch()
            if kill_switch and kill_switch.is_active():
                system_state = "kill_switch_active"
            elif self.operational_controls and self.operational_controls.emergency_stop:
                system_state = "emergency_stop"
            elif self.operational_controls and self.operational_controls.maintenance_mode:
                system_state = "maintenance_mode"
            
            summary = {
                'system_state': system_state,
                'order_counts': {
                    'total_orders': len(self.orders),
                    'active_orders': len(self.active_orders),
                    'orders_submitted': self.performance_metrics['orders_submitted']
                },
                'latency_metrics': {
                    'avg_submission_latency_us': self.performance_metrics['avg_submission_latency_us'],
                    'target_latency_us': 500,
                    'meeting_target': self.performance_metrics['avg_submission_latency_us'] <= 500
                }
            }
            
            # Add execution tracker metrics if available
            if self.execution_tracker:
                tracker_metrics = self.execution_tracker.get_real_time_metrics()
                summary['execution_metrics'] = tracker_metrics
            
            return summary
    
    def shutdown(self) -> None:
        """Shutdown order manager"""
        logger.info("OrderManager shutdown initiated")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Cancel all active orders
        active_order_ids = list(self.active_orders.keys())
        for order_id in active_order_ids:
            try:
                asyncio.run(self.cancel_order(order_id, "System shutdown"))
            except Exception as e:
                logger.error("Error cancelling order during shutdown", order_id=order_id, error=str(e))
        
        # Shutdown components
        if self.execution_tracker:
            self.execution_tracker.shutdown()
        
        self.executor.shutdown(wait=True)
        
        logger.info("OrderManager shutdown complete")