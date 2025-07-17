"""
Order Types and Enums

Defines all order-related data structures optimized for high-frequency execution.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
import uuid
import time


class OrderType(Enum):
    """Order types supported by the execution engine"""
    MARKET = "MARKET"
    LIMIT = "LIMIT" 
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    ICEBERG = "ICEBERG"
    TWAP = "TWAP"
    VWAP = "VWAP"
    IMPLEMENTATION_SHORTFALL = "IMPLEMENTATION_SHORTFALL"


class OrderSide(Enum):
    """Order side (buy/sell)"""
    BUY = "BUY"
    SELL = "SELL"


class TimeInForce(Enum):
    """Time in force options"""
    DAY = "DAY"           # Good for day
    GTC = "GTC"           # Good till canceled
    IOC = "IOC"           # Immediate or cancel
    FOK = "FOK"           # Fill or kill
    GTD = "GTD"           # Good till date


class OrderStatus(Enum):
    """Order status lifecycle"""
    PENDING = "PENDING"               # Order created, not yet submitted
    SUBMITTED = "SUBMITTED"           # Order submitted to venue
    ACKNOWLEDGED = "ACKNOWLEDGED"     # Order acknowledged by venue
    PARTIALLY_FILLED = "PARTIALLY_FILLED"  # Order partially executed
    FILLED = "FILLED"                 # Order completely filled
    CANCELLED = "CANCELLED"           # Order cancelled
    REJECTED = "REJECTED"             # Order rejected
    EXPIRED = "EXPIRED"               # Order expired


class OrderPriority(Enum):
    """Order priority levels for execution"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class OrderExecution:
    """Individual execution record"""
    execution_id: str
    timestamp: datetime
    price: float
    quantity: int
    venue: str
    commission: float = 0.0
    fees: float = 0.0
    liquidity_flag: str = ""  # Added/Removed liquidity


@dataclass
class Order:
    """
    Ultra-optimized order structure for high-frequency execution.
    Uses slots for memory efficiency and fast access.
    """
    
    # Core identification
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    client_order_id: Optional[str] = None
    symbol: str = ""
    
    # Order details
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: int = 0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    
    # Status and timing
    status: OrderStatus = OrderStatus.PENDING
    created_timestamp: datetime = field(default_factory=datetime.now)
    submitted_timestamp: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)
    expiry_time: Optional[datetime] = None
    
    # Execution tracking
    filled_quantity: int = 0
    remaining_quantity: int = 0
    average_fill_price: float = 0.0
    total_commission: float = 0.0
    total_fees: float = 0.0
    executions: list = field(default_factory=list)
    
    # Routing and venue info
    target_venue: Optional[str] = None
    actual_venue: Optional[str] = None
    routing_strategy: Optional[str] = None
    
    # Algorithm parameters (for algo orders)
    algorithm_params: Dict[str, Any] = field(default_factory=dict)
    
    # Risk and compliance
    risk_checked: bool = False
    compliance_checked: bool = False
    priority: OrderPriority = OrderPriority.NORMAL
    
    # Performance tracking
    order_placement_latency: Optional[float] = None  # μs
    acknowledgement_latency: Optional[float] = None  # μs
    fill_latency: Optional[float] = None  # μs
    
    # Metadata
    source: str = "MARL_SYSTEM"
    tags: Dict[str, str] = field(default_factory=dict)
    notes: str = ""
    
    def __post_init__(self):
        """Initialize computed fields"""
        if self.remaining_quantity == 0:
            self.remaining_quantity = self.quantity
            
        # Generate client order ID if not provided
        if not self.client_order_id:
            self.client_order_id = f"MARL_{int(time.time() * 1000000)}"
    
    @property
    def is_buy(self) -> bool:
        """Check if order is a buy order"""
        return self.side == OrderSide.BUY
    
    @property
    def is_sell(self) -> bool:
        """Check if order is a sell order"""
        return self.side == OrderSide.SELL
    
    @property
    def signed_quantity(self) -> int:
        """Get signed quantity (positive for buy, negative for sell)"""
        return self.quantity if self.is_buy else -self.quantity
    
    @property
    def fill_ratio(self) -> float:
        """Get fill ratio (0.0 to 1.0)"""
        return self.filled_quantity / self.quantity if self.quantity > 0 else 0.0
    
    @property
    def is_complete(self) -> bool:
        """Check if order is completely filled"""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active (can be filled)"""
        return self.status in [
            OrderStatus.SUBMITTED,
            OrderStatus.ACKNOWLEDGED, 
            OrderStatus.PARTIALLY_FILLED
        ]
    
    @property
    def is_terminal(self) -> bool:
        """Check if order is in terminal state"""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        ]
    
    @property
    def notional_value(self) -> float:
        """Calculate notional value of order"""
        if self.price is not None:
            return abs(self.quantity * self.price)
        return 0.0
    
    @property
    def filled_notional(self) -> float:
        """Calculate filled notional value"""
        return abs(self.filled_quantity * self.average_fill_price)
    
    def add_execution(self, execution: OrderExecution) -> None:
        """Add execution to order and update fill statistics"""
        self.executions.append(execution)
        
        # Update fill statistics
        old_filled = self.filled_quantity
        self.filled_quantity += execution.quantity
        self.remaining_quantity = self.quantity - self.filled_quantity
        
        # Update average fill price (quantity-weighted)
        if self.filled_quantity > 0:
            total_value = (old_filled * self.average_fill_price + 
                          execution.quantity * execution.price)
            self.average_fill_price = total_value / self.filled_quantity
        
        # Update costs
        self.total_commission += execution.commission
        self.total_fees += execution.fees
        
        # Update status
        if self.remaining_quantity == 0:
            self.status = OrderStatus.FILLED
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED
            
        self.last_updated = datetime.now()
    
    def update_status(self, new_status: OrderStatus, timestamp: Optional[datetime] = None) -> None:
        """Update order status with timestamp"""
        old_status = self.status
        self.status = new_status
        self.last_updated = timestamp or datetime.now()
        
        # Set submitted timestamp when order is first submitted
        if old_status == OrderStatus.PENDING and new_status == OrderStatus.SUBMITTED:
            self.submitted_timestamp = self.last_updated
    
    def calculate_implementation_shortfall(self, decision_price: float) -> float:
        """Calculate implementation shortfall vs decision price"""
        if self.filled_quantity == 0:
            return 0.0
            
        shortfall = (self.average_fill_price - decision_price) / decision_price
        
        # Adjust sign for sell orders
        if self.is_sell:
            shortfall = -shortfall
            
        return shortfall
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get comprehensive performance metrics"""
        metrics = {
            'fill_ratio': self.fill_ratio,
            'filled_notional': self.filled_notional,
            'total_costs': self.total_commission + self.total_fees,
            'cost_per_share': (self.total_commission + self.total_fees) / max(self.filled_quantity, 1),
        }
        
        # Add latency metrics if available
        if self.order_placement_latency is not None:
            metrics['placement_latency_us'] = self.order_placement_latency
        if self.acknowledgement_latency is not None:
            metrics['ack_latency_us'] = self.acknowledgement_latency
        if self.fill_latency is not None:
            metrics['fill_latency_us'] = self.fill_latency
            
        return metrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary for serialization"""
        return {
            'order_id': self.order_id,
            'client_order_id': self.client_order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force.value,
            'status': self.status.value,
            'created_timestamp': self.created_timestamp.isoformat(),
            'submitted_timestamp': self.submitted_timestamp.isoformat() if self.submitted_timestamp else None,
            'filled_quantity': self.filled_quantity,
            'remaining_quantity': self.remaining_quantity,
            'average_fill_price': self.average_fill_price,
            'total_commission': self.total_commission,
            'total_fees': self.total_fees,
            'target_venue': self.target_venue,
            'actual_venue': self.actual_venue,
            'routing_strategy': self.routing_strategy,
            'algorithm_params': self.algorithm_params,
            'priority': self.priority.value,
            'performance_metrics': self.get_performance_metrics()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """Create order from dictionary"""
        order = cls()
        
        # Basic fields
        order.order_id = data.get('order_id', order.order_id)
        order.client_order_id = data.get('client_order_id')
        order.symbol = data.get('symbol', '')
        order.side = OrderSide(data.get('side', 'BUY'))
        order.order_type = OrderType(data.get('order_type', 'MARKET'))
        order.quantity = data.get('quantity', 0)
        order.price = data.get('price')
        order.stop_price = data.get('stop_price')
        order.time_in_force = TimeInForce(data.get('time_in_force', 'DAY'))
        order.status = OrderStatus(data.get('status', 'PENDING'))
        
        # Timestamps
        if data.get('created_timestamp'):
            order.created_timestamp = datetime.fromisoformat(data['created_timestamp'])
        if data.get('submitted_timestamp'):
            order.submitted_timestamp = datetime.fromisoformat(data['submitted_timestamp'])
            
        # Execution data
        order.filled_quantity = data.get('filled_quantity', 0)
        order.remaining_quantity = data.get('remaining_quantity', order.quantity)
        order.average_fill_price = data.get('average_fill_price', 0.0)
        order.total_commission = data.get('total_commission', 0.0)
        order.total_fees = data.get('total_fees', 0.0)
        
        # Routing info
        order.target_venue = data.get('target_venue')
        order.actual_venue = data.get('actual_venue')
        order.routing_strategy = data.get('routing_strategy')
        order.algorithm_params = data.get('algorithm_params', {})
        
        # Priority
        if 'priority' in data:
            order.priority = OrderPriority(data['priority'])
        
        return order


@dataclass
class OrderRequest:
    """Request to create a new order"""
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    priority: OrderPriority = OrderPriority.NORMAL
    algorithm_params: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    source: str = "MARL_SYSTEM"
    
    def to_order(self) -> Order:
        """Convert request to order object"""
        return Order(
            symbol=self.symbol,
            side=self.side,
            quantity=self.quantity,
            order_type=self.order_type,
            price=self.price,
            stop_price=self.stop_price,
            time_in_force=self.time_in_force,
            priority=self.priority,
            algorithm_params=self.algorithm_params,
            tags=self.tags,
            source=self.source
        )


@dataclass 
class OrderUpdate:
    """Update to existing order"""
    order_id: str
    action: str  # 'modify', 'cancel', 'status_update'
    new_quantity: Optional[int] = None
    new_price: Optional[float] = None
    new_stop_price: Optional[float] = None
    new_status: Optional[OrderStatus] = None
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)