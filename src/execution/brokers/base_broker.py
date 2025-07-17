"""
Base Broker Client

Abstract base class for broker API integrations providing common interface
and functionality across different broker implementations.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import structlog

logger = structlog.get_logger()


class ConnectionStatus(Enum):
    """Broker connection status"""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    ERROR = "ERROR"


class OrderStatus(Enum):
    """Broker order status"""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class BrokerConnection:
    """Broker connection information"""
    
    broker_name: str
    status: ConnectionStatus
    connected_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    connection_latency_ms: float = 0.0
    error_message: Optional[str] = None
    reconnect_attempts: int = 0
    max_reconnect_attempts: int = 5


@dataclass
class BrokerOrder:
    """Broker order representation"""
    
    broker_order_id: str
    client_order_id: str
    symbol: str
    side: str  # BUY/SELL
    quantity: int
    order_type: str  # MARKET/LIMIT/STOP
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    remaining_quantity: int = 0
    average_fill_price: float = 0.0
    commission: float = 0.0
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = self.created_at
        if self.remaining_quantity == 0:
            self.remaining_quantity = self.quantity


@dataclass
class BrokerExecution:
    """Broker execution report"""
    
    execution_id: str
    broker_order_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    timestamp: datetime
    commission: float = 0.0
    fees: float = 0.0
    liquidity_flag: str = ""  # ADDED/REMOVED
    venue: str = ""


@dataclass
class BrokerPosition:
    """Broker position information"""
    
    symbol: str
    quantity: int
    average_cost: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    last_updated: datetime


@dataclass
class BrokerAccount:
    """Broker account information"""
    
    account_id: str
    total_value: float
    available_cash: float
    buying_power: float
    day_trading_buying_power: float
    maintenance_margin: float
    initial_margin: float
    currency: str = "USD"
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()


class BaseBrokerClient(ABC):
    """
    Abstract base class for broker API clients.
    
    Provides common interface and functionality for all broker integrations
    including connection management, order handling, and data streaming.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection = BrokerConnection(
            broker_name=self.__class__.__name__,
            status=ConnectionStatus.DISCONNECTED
        )
        
        # Event callbacks
        self.order_callback: Optional[Callable] = None
        self.execution_callback: Optional[Callable] = None
        self.position_callback: Optional[Callable] = None
        self.error_callback: Optional[Callable] = None
        
        # Internal state
        self.orders: Dict[str, BrokerOrder] = {}
        self.positions: Dict[str, BrokerPosition] = {}
        self.account: Optional[BrokerAccount] = None
        
        # Performance tracking
        self.order_latencies: List[float] = []
        self.connection_metrics = {
            'total_connections': 0,
            'successful_connections': 0,
            'connection_errors': 0,
            'total_orders': 0,
            'successful_orders': 0,
            'order_errors': 0
        }
        
        # Background tasks
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.data_stream_task: Optional[asyncio.Task] = None
        
        logger.info(f"{self.__class__.__name__} initialized")
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker API"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from broker API"""
        pass
    
    @abstractmethod
    async def submit_order(self, order_data: Dict[str, Any]) -> BrokerOrder:
        """Submit order to broker"""
        pass
    
    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel order at broker"""
        pass
    
    @abstractmethod
    async def modify_order(
        self,
        broker_order_id: str,
        modifications: Dict[str, Any]
    ) -> bool:
        """Modify order at broker"""
        pass
    
    @abstractmethod
    async def get_order_status(self, broker_order_id: str) -> Optional[BrokerOrder]:
        """Get order status from broker"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[BrokerPosition]:
        """Get current positions from broker"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> BrokerAccount:
        """Get account information from broker"""
        pass
    
    def set_order_callback(self, callback: Callable[[BrokerOrder], None]) -> None:
        """Set callback for order updates"""
        self.order_callback = callback
    
    def set_execution_callback(self, callback: Callable[[BrokerExecution], None]) -> None:
        """Set callback for execution reports"""
        self.execution_callback = callback
    
    def set_position_callback(self, callback: Callable[[BrokerPosition], None]) -> None:
        """Set callback for position updates"""
        self.position_callback = callback
    
    def set_error_callback(self, callback: Callable[[str, Exception], None]) -> None:
        """Set callback for error notifications"""
        self.error_callback = callback
    
    async def _update_connection_status(self, status: ConnectionStatus, error: str = None) -> None:
        """Update connection status"""
        old_status = self.connection.status
        self.connection.status = status
        
        if status == ConnectionStatus.CONNECTED:
            self.connection.connected_at = datetime.now()
            self.connection.reconnect_attempts = 0
            self.connection.error_message = None
        elif status == ConnectionStatus.ERROR:
            self.connection.error_message = error
        
        if old_status != status:
            logger.info(
                "Connection status changed",
                broker=self.connection.broker_name,
                old_status=old_status.value,
                new_status=status.value,
                error=error
            )
    
    async def _handle_order_update(self, order: BrokerOrder) -> None:
        """Handle order status update"""
        self.orders[order.broker_order_id] = order
        
        if self.order_callback:
            try:
                self.order_callback(order)
            except Exception as e:
                logger.error(f"Error in order callback: {str(e)}")
    
    async def _handle_execution_report(self, execution: BrokerExecution) -> None:
        """Handle execution report"""
        # Update order if exists
        if execution.broker_order_id in self.orders:
            order = self.orders[execution.broker_order_id]
            order.filled_quantity += execution.quantity
            order.remaining_quantity = order.quantity - order.filled_quantity
            order.commission += execution.commission
            order.updated_at = execution.timestamp
            
            # Update average fill price
            if order.filled_quantity > 0:
                total_value = (order.average_fill_price * (order.filled_quantity - execution.quantity) +
                              execution.price * execution.quantity)
                order.average_fill_price = total_value / order.filled_quantity
            
            # Update status
            if order.remaining_quantity == 0:
                order.status = OrderStatus.FILLED
            elif order.filled_quantity > 0:
                order.status = OrderStatus.PARTIALLY_FILLED
        
        if self.execution_callback:
            try:
                self.execution_callback(execution)
            except Exception as e:
                logger.error(f"Error in execution callback: {str(e)}")
    
    async def _handle_position_update(self, position: BrokerPosition) -> None:
        """Handle position update"""
        self.positions[position.symbol] = position
        
        if self.position_callback:
            try:
                self.position_callback(position)
            except Exception as e:
                logger.error(f"Error in position callback: {str(e)}")
    
    async def _handle_error(self, context: str, error: Exception) -> None:
        """Handle error"""
        logger.error(f"Broker error in {context}: {str(error)}")
        
        if self.error_callback:
            try:
                self.error_callback(context, error)
            except Exception as e:
                logger.error(f"Error in error callback: {str(e)}")
    
    async def _start_heartbeat(self, interval_seconds: int = 30) -> None:
        """Start heartbeat monitoring"""
        if self.heartbeat_task and not self.heartbeat_task.done():
            return
        
        self.heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(interval_seconds)
        )
    
    async def _heartbeat_loop(self, interval_seconds: int) -> None:
        """Heartbeat monitoring loop"""
        while self.connection.status == ConnectionStatus.CONNECTED:
            try:
                # Perform heartbeat check
                start_time = time.perf_counter()
                heartbeat_success = await self._perform_heartbeat()
                latency = (time.perf_counter() - start_time) * 1000
                
                if heartbeat_success:
                    self.connection.last_heartbeat = datetime.now()
                    self.connection.connection_latency_ms = latency
                else:
                    logger.warning("Heartbeat failed")
                    await self._update_connection_status(
                        ConnectionStatus.ERROR,
                        "Heartbeat failed"
                    )
                    break
                
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {str(e)}")
                await asyncio.sleep(interval_seconds)
    
    async def _perform_heartbeat(self) -> bool:
        """Perform broker-specific heartbeat check"""
        # Default implementation - override in subclasses
        return self.connection.status == ConnectionStatus.CONNECTED
    
    async def _auto_reconnect(self) -> bool:
        """Attempt automatic reconnection"""
        if self.connection.reconnect_attempts >= self.connection.max_reconnect_attempts:
            logger.error("Maximum reconnection attempts reached")
            return False
        
        self.connection.reconnect_attempts += 1
        await self._update_connection_status(ConnectionStatus.RECONNECTING)
        
        logger.info(
            f"Attempting reconnection {self.connection.reconnect_attempts}/"
            f"{self.connection.max_reconnect_attempts}"
        )
        
        # Exponential backoff
        delay = min(300, 2 ** self.connection.reconnect_attempts)  # Max 5 minutes
        await asyncio.sleep(delay)
        
        return await self.connect()
    
    def _update_performance_metrics(self, metric_type: str, success: bool = True) -> None:
        """Update performance metrics"""
        if metric_type == 'connection':
            self.connection_metrics['total_connections'] += 1
            if success:
                self.connection_metrics['successful_connections'] += 1
            else:
                self.connection_metrics['connection_errors'] += 1
        
        elif metric_type == 'order':
            self.connection_metrics['total_orders'] += 1
            if success:
                self.connection_metrics['successful_orders'] += 1
            else:
                self.connection_metrics['order_errors'] += 1
    
    def _track_order_latency(self, latency_ms: float) -> None:
        """Track order submission latency"""
        self.order_latencies.append(latency_ms)
        
        # Keep only recent latencies
        if len(self.order_latencies) > 1000:
            self.order_latencies = self.order_latencies[-1000:]
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information"""
        return {
            'broker_name': self.connection.broker_name,
            'status': self.connection.status.value,
            'connected_at': self.connection.connected_at.isoformat() if self.connection.connected_at else None,
            'last_heartbeat': self.connection.last_heartbeat.isoformat() if self.connection.last_heartbeat else None,
            'connection_latency_ms': self.connection.connection_latency_ms,
            'error_message': self.connection.error_message,
            'reconnect_attempts': self.connection.reconnect_attempts
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        avg_latency = sum(self.order_latencies) / len(self.order_latencies) if self.order_latencies else 0.0
        
        return {
            'connection_metrics': self.connection_metrics.copy(),
            'order_latency': {
                'avg_latency_ms': avg_latency,
                'min_latency_ms': min(self.order_latencies) if self.order_latencies else 0.0,
                'max_latency_ms': max(self.order_latencies) if self.order_latencies else 0.0,
                'sample_count': len(self.order_latencies)
            },
            'current_state': {
                'active_orders': len([o for o in self.orders.values() if o.status in [
                    OrderStatus.SUBMITTED, OrderStatus.ACKNOWLEDGED, OrderStatus.PARTIALLY_FILLED
                ]]),
                'total_orders': len(self.orders),
                'positions_count': len(self.positions)
            }
        }
    
    def get_orders(self, status_filter: Optional[OrderStatus] = None) -> List[BrokerOrder]:
        """Get orders with optional status filter"""
        orders = list(self.orders.values())
        
        if status_filter:
            orders = [order for order in orders if order.status == status_filter]
        
        return orders
    
    def get_order_by_client_id(self, client_order_id: str) -> Optional[BrokerOrder]:
        """Get order by client order ID"""
        for order in self.orders.values():
            if order.client_order_id == client_order_id:
                return order
        return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_status = {
            'broker': self.connection.broker_name,
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        # Connection check
        connection_healthy = self.connection.status == ConnectionStatus.CONNECTED
        health_status['checks']['connection'] = {
            'status': 'pass' if connection_healthy else 'fail',
            'connection_status': self.connection.status.value,
            'last_heartbeat': self.connection.last_heartbeat.isoformat() if self.connection.last_heartbeat else None
        }
        
        # Latency check
        recent_latencies = self.order_latencies[-10:] if self.order_latencies else []
        avg_recent_latency = sum(recent_latencies) / len(recent_latencies) if recent_latencies else 0.0
        latency_healthy = avg_recent_latency < 1000  # <1 second
        
        health_status['checks']['latency'] = {
            'status': 'pass' if latency_healthy else 'fail',
            'avg_latency_ms': avg_recent_latency,
            'sample_count': len(recent_latencies)
        }
        
        # Error rate check
        total_orders = self.connection_metrics['total_orders']
        error_orders = self.connection_metrics['order_errors']
        error_rate = error_orders / total_orders if total_orders > 0 else 0.0
        error_rate_healthy = error_rate < 0.05  # <5% error rate
        
        health_status['checks']['error_rate'] = {
            'status': 'pass' if error_rate_healthy else 'fail',
            'error_rate': error_rate,
            'total_orders': total_orders,
            'error_orders': error_orders
        }
        
        # Overall status
        all_checks_pass = all(
            check['status'] == 'pass' 
            for check in health_status['checks'].values()
        )
        
        if not all_checks_pass:
            health_status['overall_status'] = 'degraded' if connection_healthy else 'unhealthy'
        
        return health_status
    
    async def shutdown(self) -> None:
        """Shutdown broker client"""
        logger.info(f"Shutting down {self.__class__.__name__}")
        
        # Cancel background tasks
        if self.heartbeat_task and not self.heartbeat_task.done():
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
        
        if self.data_stream_task and not self.data_stream_task.done():
            self.data_stream_task.cancel()
            try:
                await self.data_stream_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect
        await self.disconnect()
        
        logger.info(f"{self.__class__.__name__} shutdown complete")