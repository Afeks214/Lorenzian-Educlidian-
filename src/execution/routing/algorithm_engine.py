"""
Algorithm Engine

Advanced execution algorithms including VWAP, TWAP, and Implementation Shortfall.
Provides intelligent order slicing and execution timing optimization.
"""

import logging


import asyncio
import time
import math
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import random
import structlog

from ..order_management.order_types import Order, OrderExecution

logger = structlog.get_logger()


class AlgorithmType(Enum):
    """Supported execution algorithms"""
    VWAP = "VWAP"                           # Volume Weighted Average Price
    TWAP = "TWAP"                           # Time Weighted Average Price
    IMPLEMENTATION_SHORTFALL = "IS"         # Implementation Shortfall
    PARTICIPATION_RATE = "POV"              # Percentage of Volume
    ARRIVAL_PRICE = "AP"                    # Arrival Price
    MARKET_ON_CLOSE = "MOC"                 # Market on Close
    ADAPTIVE = "ADAPTIVE"                   # Adaptive algorithm


@dataclass
class AlgorithmConfig:
    """Configuration for execution algorithms"""
    
    # Common parameters
    algorithm_type: AlgorithmType
    duration_minutes: int = 60
    max_participation_rate: float = 0.20    # 20% max participation
    min_participation_rate: float = 0.05    # 5% min participation
    
    # VWAP specific
    volume_target: Optional[int] = None      # Target volume to match
    historical_days: int = 20               # Days of volume history
    intraday_volume_curve: bool = True      # Use intraday volume patterns
    
    # TWAP specific
    slice_duration_minutes: int = 5         # Duration of each slice
    randomization_factor: float = 0.20     # 20% time randomization
    
    # Implementation Shortfall specific
    risk_aversion: float = 0.5              # Risk aversion parameter (0-1)
    market_impact_model: str = "linear"     # Market impact model
    volatility_multiplier: float = 1.0     # Volatility adjustment
    
    # Adaptive parameters
    learning_rate: float = 0.1              # Learning rate for adaptation
    performance_window: int = 100           # Orders to consider for learning
    
    # Risk controls
    max_slice_size: Optional[int] = None    # Maximum slice size
    price_limit_buffer: float = 0.02       # 2% price limit buffer
    stop_on_adverse_move: float = 0.05     # Stop if price moves 5% against
    
    # Execution controls
    aggressive_on_close: bool = False       # Be more aggressive near close
    dark_pool_preference: float = 0.3       # Preference for dark pools
    venue_rotation: bool = True             # Rotate between venues


@dataclass
class SliceOrder:
    """Individual slice of a parent algorithm order"""
    
    slice_id: str
    parent_order_id: str
    quantity: int
    target_time: datetime
    price_limit: Optional[float] = None
    venue_preference: Optional[str] = None
    urgency: float = 0.5                    # 0 = patient, 1 = urgent
    
    # Execution tracking
    submitted: bool = False
    submitted_time: Optional[datetime] = None
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    execution_time: Optional[datetime] = None


@dataclass
class AlgorithmState:
    """Current state of algorithm execution"""
    
    order_id: str
    algorithm_type: AlgorithmType
    config: AlgorithmConfig
    
    # Progress tracking
    total_quantity: int
    executed_quantity: int = 0
    remaining_quantity: int = 0
    slices_executed: int = 0
    
    # Performance metrics
    avg_execution_price: float = 0.0
    market_impact_bps: float = 0.0
    implementation_shortfall: float = 0.0
    
    # Current market conditions
    current_price: float = 0.0
    current_volume: int = 0
    volume_rate: float = 0.0
    volatility: float = 0.0
    
    # Algorithm-specific state
    volume_curve: List[float] = field(default_factory=list)
    time_curve: List[datetime] = field(default_factory=list)
    historical_volume: List[int] = field(default_factory=list)
    
    # Control flags
    is_active: bool = True
    is_paused: bool = False
    stop_reason: Optional[str] = None
    
    def __post_init__(self):
        self.remaining_quantity = self.total_quantity - self.executed_quantity


class AlgorithmEngine:
    """
    Advanced execution algorithm engine supporting multiple strategies.
    
    Provides intelligent order slicing, timing optimization, and adaptive execution
    based on real-time market conditions and historical patterns.
    """
    
    def __init__(self, market_data_provider: Any = None):
        self.market_data_provider = market_data_provider
        self.active_algorithms: Dict[str, AlgorithmState] = {}
        self.pending_slices: Dict[str, List[SliceOrder]] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.algorithm_performance: Dict[AlgorithmType, Dict[str, float]] = {}
        
        # Background tasks
        self.execution_task = None
        self.monitoring_task = None
        self.shutdown_event = asyncio.Event()
        
        logger.info("AlgorithmEngine initialized")
    
    async def start_algorithm(
        self, 
        order: Order, 
        config: AlgorithmConfig
    ) -> str:
        """Start algorithm execution for an order"""
        
        # Create algorithm state
        state = AlgorithmState(
            order_id=order.order_id,
            algorithm_type=config.algorithm_type,
            config=config,
            total_quantity=order.quantity,
            remaining_quantity=order.quantity
        )
        
        # Initialize algorithm-specific components
        await self._initialize_algorithm(state, order)
        
        # Store state
        self.active_algorithms[order.order_id] = state
        
        # Generate execution schedule
        slices = await self._generate_execution_schedule(state, order)
        self.pending_slices[order.order_id] = slices
        
        # Start execution if not already running
        if not self.execution_task or self.execution_task.done():
            self.execution_task = asyncio.create_task(self._execution_loop())
        
        logger.info(
            "Algorithm started",
            order_id=order.order_id,
            algorithm_type=config.algorithm_type.value,
            total_quantity=order.quantity,
            slice_count=len(slices)
        )
        
        return order.order_id
    
    async def _initialize_algorithm(self, state: AlgorithmState, order: Order) -> None:
        """Initialize algorithm-specific data"""
        
        if state.algorithm_type == AlgorithmType.VWAP:
            await self._initialize_vwap(state, order)
        elif state.algorithm_type == AlgorithmType.TWAP:
            await self._initialize_twap(state, order)
        elif state.algorithm_type == AlgorithmType.IMPLEMENTATION_SHORTFALL:
            await self._initialize_implementation_shortfall(state, order)
        elif state.algorithm_type == AlgorithmType.ADAPTIVE:
            await self._initialize_adaptive(state, order)
    
    async def _initialize_vwap(self, state: AlgorithmState, order: Order) -> None:
        """Initialize VWAP algorithm"""
        
        # Get historical volume data
        if self.market_data_provider:
            try:
                historical_data = await self.market_data_provider.get_historical_volume(
                    symbol=order.symbol,
                    days=state.config.historical_days
                )
                state.historical_volume = historical_data.get('volumes', [])
                
                # Build intraday volume curve
                if state.config.intraday_volume_curve:
                    state.volume_curve = await self._build_volume_curve(order.symbol)
                    
            except Exception as e:
                logger.warning(f"Failed to get historical data for VWAP: {str(e)}")
                # Use default volume curve
                state.volume_curve = self._default_volume_curve()
        else:
            state.volume_curve = self._default_volume_curve()
    
    async def _initialize_twap(self, state: AlgorithmState, order: Order) -> None:
        """Initialize TWAP algorithm"""
        
        # Generate time curve
        duration = timedelta(minutes=state.config.duration_minutes)
        slice_duration = timedelta(minutes=state.config.slice_duration_minutes)
        
        current_time = datetime.now()
        end_time = current_time + duration
        
        time_points = []
        t = current_time
        while t < end_time:
            # Add randomization
            if state.config.randomization_factor > 0:
                random_offset = random.uniform(
                    -state.config.randomization_factor,
                    state.config.randomization_factor
                ) * slice_duration.total_seconds()
                t += timedelta(seconds=random_offset)
            
            time_points.append(t)
            t += slice_duration
        
        state.time_curve = time_points
    
    async def _initialize_implementation_shortfall(self, state: AlgorithmState, order: Order) -> None:
        """Initialize Implementation Shortfall algorithm"""
        
        # Get current market conditions
        if self.market_data_provider:
            try:
                market_data = await self.market_data_provider.get_current_data(order.symbol)
                state.current_price = market_data.get('price', 0.0)
                state.volatility = market_data.get('volatility', 0.02)
                state.volume_rate = market_data.get('volume_rate', 1000)
            except Exception as e:
                logger.warning(f"Failed to get market data for IS: {str(e)}")
        
        # Calculate optimal execution schedule based on risk aversion
        await self._calculate_is_schedule(state, order)
    
    async def _initialize_adaptive(self, state: AlgorithmState, order: Order) -> None:
        """Initialize Adaptive algorithm"""
        
        # Start with TWAP as base strategy
        await self._initialize_twap(state, order)
        
        # Load historical performance data
        historical_performance = self._get_historical_performance(order.symbol)
        
        # Adjust strategy based on recent performance
        if historical_performance:
            self._adapt_strategy(state, historical_performance)
    
    def _default_volume_curve(self) -> List[float]:
        """Default intraday volume curve (U-shaped)"""
        # Typical U-shaped volume pattern: high at open, low midday, high at close
        hours = 6.5  # Market hours
        points = int(hours * 12)  # 5-minute intervals
        
        curve = []
        for i in range(points):
            t = i / points  # Normalized time [0, 1]
            
            # U-shaped curve: high at beginning and end, low in middle
            volume_factor = 2.0 * (t**2 - t + 0.5)
            volume_factor = max(0.3, min(2.0, volume_factor))  # Clamp between 0.3 and 2.0
            
            curve.append(volume_factor)
        
        return curve
    
    async def _build_volume_curve(self, symbol: str) -> List[float]:
        """Build historical intraday volume curve"""
        try:
            if self.market_data_provider:
                volume_data = await self.market_data_provider.get_intraday_volume_pattern(symbol)
                return volume_data.get('curve', self._default_volume_curve())
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.error(f'Error occurred: {e}')
        
        return self._default_volume_curve()
    
    async def _generate_execution_schedule(
        self, 
        state: AlgorithmState, 
        order: Order
    ) -> List[SliceOrder]:
        """Generate execution schedule based on algorithm type"""
        
        if state.algorithm_type == AlgorithmType.VWAP:
            return await self._generate_vwap_schedule(state, order)
        elif state.algorithm_type == AlgorithmType.TWAP:
            return await self._generate_twap_schedule(state, order)
        elif state.algorithm_type == AlgorithmType.IMPLEMENTATION_SHORTFALL:
            return await self._generate_is_schedule(state, order)
        elif state.algorithm_type == AlgorithmType.ADAPTIVE:
            return await self._generate_adaptive_schedule(state, order)
        else:
            # Default to TWAP
            return await self._generate_twap_schedule(state, order)
    
    async def _generate_vwap_schedule(
        self, 
        state: AlgorithmState, 
        order: Order
    ) -> List[SliceOrder]:
        """Generate VWAP execution schedule"""
        
        slices = []
        volume_curve = state.volume_curve
        total_volume_weight = sum(volume_curve)
        
        # Calculate slice sizes based on volume curve
        remaining_qty = state.total_quantity
        current_time = datetime.now()
        
        for i, volume_weight in enumerate(volume_curve):
            if remaining_qty <= 0:
                break
            
            # Calculate slice size based on volume proportion
            volume_proportion = volume_weight / total_volume_weight
            slice_size = min(
                int(state.total_quantity * volume_proportion * state.config.max_participation_rate),
                remaining_qty
            )
            
            # Ensure minimum slice size
            slice_size = max(slice_size, 1)
            
            # Apply max slice size limit
            if state.config.max_slice_size:
                slice_size = min(slice_size, state.config.max_slice_size)
            
            if slice_size > 0:
                target_time = current_time + timedelta(minutes=i * 5)  # 5-minute intervals
                
                slice = SliceOrder(
                    slice_id=f"{order.order_id}_slice_{i}",
                    parent_order_id=order.order_id,
                    quantity=slice_size,
                    target_time=target_time,
                    urgency=volume_weight / max(volume_curve)  # Higher urgency during high volume
                )
                
                slices.append(slice)
                remaining_qty -= slice_size
        
        # Handle any remaining quantity in final slice
        if remaining_qty > 0 and slices:
            slices[-1].quantity += remaining_qty
        
        return slices
    
    async def _generate_twap_schedule(
        self, 
        state: AlgorithmState, 
        order: Order
    ) -> List[SliceOrder]:
        """Generate TWAP execution schedule"""
        
        slices = []
        time_curve = state.time_curve
        
        if not time_curve:
            # Generate simple time schedule
            duration_minutes = state.config.duration_minutes
            slice_duration = state.config.slice_duration_minutes
            num_slices = duration_minutes // slice_duration
            
            current_time = datetime.now()
            for i in range(num_slices):
                time_curve.append(current_time + timedelta(minutes=i * slice_duration))
        
        # Distribute quantity evenly across time slices
        base_slice_size = state.total_quantity // len(time_curve)
        remaining_qty = state.total_quantity % len(time_curve)
        
        for i, target_time in enumerate(time_curve):
            slice_size = base_slice_size
            if i < remaining_qty:  # Distribute remainder
                slice_size += 1
            
            if slice_size > 0:
                slice = SliceOrder(
                    slice_id=f"{order.order_id}_slice_{i}",
                    parent_order_id=order.order_id,
                    quantity=slice_size,
                    target_time=target_time,
                    urgency=0.5  # Constant urgency for TWAP
                )
                
                slices.append(slice)
        
        return slices
    
    async def _generate_is_schedule(
        self, 
        state: AlgorithmState, 
        order: Order
    ) -> List[SliceOrder]:
        """Generate Implementation Shortfall schedule"""
        
        # Use simplified IS model - in production this would be more sophisticated
        slices = []
        
        # Front-load execution based on risk aversion
        risk_aversion = state.config.risk_aversion
        urgency_factor = 1.0 - risk_aversion  # Lower risk aversion = higher urgency
        
        # Generate aggressive schedule if low risk aversion
        if urgency_factor > 0.7:
            # Execute quickly in first half of period
            num_slices = 4
            quantities = [
                int(state.total_quantity * 0.4),   # 40% immediately
                int(state.total_quantity * 0.3),   # 30% in 15 minutes
                int(state.total_quantity * 0.2),   # 20% in 30 minutes
                0  # Remainder in final slice
            ]
            quantities[3] = state.total_quantity - sum(quantities[:3])
            
        else:
            # More patient execution
            num_slices = 8
            base_size = state.total_quantity // num_slices
            quantities = [base_size] * num_slices
            quantities[-1] += state.total_quantity % num_slices
        
        current_time = datetime.now()
        for i, quantity in enumerate(quantities):
            if quantity > 0:
                target_time = current_time + timedelta(minutes=i * 15)
                
                slice = SliceOrder(
                    slice_id=f"{order.order_id}_slice_{i}",
                    parent_order_id=order.order_id,
                    quantity=quantity,
                    target_time=target_time,
                    urgency=urgency_factor
                )
                
                slices.append(slice)
        
        return slices
    
    async def _generate_adaptive_schedule(
        self, 
        state: AlgorithmState, 
        order: Order
    ) -> List[SliceOrder]:
        """Generate adaptive execution schedule"""
        
        # Start with TWAP base and adapt based on conditions
        slices = await self._generate_twap_schedule(state, order)
        
        # Adapt based on current market conditions
        if state.volatility > 0.03:  # High volatility
            # Reduce slice sizes and increase frequency
            for slice_order in slices:
                slice_order.quantity = int(slice_order.quantity * 0.7)
                slice_order.urgency *= 0.8  # Be more patient
        
        elif state.volume_rate > 10000:  # High volume
            # Can be more aggressive
            for slice_order in slices:
                slice_order.urgency *= 1.2
        
        return slices
    
    async def _execution_loop(self) -> None:
        """Main execution loop for processing algorithm slices"""
        
        while not self.shutdown_event.is_set():
            try:
                current_time = datetime.now()
                
                # Process ready slices
                for order_id, slices in self.pending_slices.items():
                    if order_id not in self.active_algorithms:
                        continue
                    
                    state = self.active_algorithms[order_id]
                    if not state.is_active or state.is_paused:
                        continue
                    
                    # Find slices ready for execution
                    ready_slices = [
                        s for s in slices 
                        if not s.submitted and s.target_time <= current_time
                    ]
                    
                    for slice_order in ready_slices:
                        await self._execute_slice(state, slice_order)
                
                # Update algorithm states
                await self._update_algorithm_states()
                
                # Sleep for next iteration
                await asyncio.sleep(0.1)  # 100ms execution cycle
                
            except Exception as e:
                logger.error(f"Error in execution loop: {str(e)}")
                await asyncio.sleep(1.0)
    
    async def _execute_slice(self, state: AlgorithmState, slice_order: SliceOrder) -> None:
        """Execute individual slice order"""
        
        try:
            # Mark as submitted
            slice_order.submitted = True
            slice_order.submitted_time = datetime.now()
            
            # Simulate execution (replace with actual order submission)
            await asyncio.sleep(0.001)  # Simulate execution latency
            
            # Simulate partial fill (in production, this would come from venue)
            fill_ratio = random.uniform(0.8, 1.0)  # 80-100% fill
            filled_qty = int(slice_order.quantity * fill_ratio)
            fill_price = state.current_price * random.uniform(0.999, 1.001)  # Small price variation
            
            # Update slice
            slice_order.filled_quantity = filled_qty
            slice_order.avg_fill_price = fill_price
            slice_order.execution_time = datetime.now()
            
            # Update algorithm state
            state.executed_quantity += filled_qty
            state.remaining_quantity = state.total_quantity - state.executed_quantity
            state.slices_executed += 1
            
            # Update average execution price
            if state.executed_quantity > 0:
                total_value = state.avg_execution_price * (state.executed_quantity - filled_qty)
                total_value += fill_price * filled_qty
                state.avg_execution_price = total_value / state.executed_quantity
            
            logger.debug(
                "Slice executed",
                order_id=state.order_id,
                slice_id=slice_order.slice_id,
                filled_quantity=filled_qty,
                fill_price=fill_price
            )
            
            # Check if algorithm is complete
            if state.remaining_quantity <= 0:
                await self._complete_algorithm(state)
            
        except Exception as e:
            logger.error(
                "Slice execution failed",
                order_id=state.order_id,
                slice_id=slice_order.slice_id,
                error=str(e)
            )
    
    async def _update_algorithm_states(self) -> None:
        """Update all active algorithm states"""
        
        for order_id, state in list(self.active_algorithms.items()):
            if not state.is_active:
                continue
            
            try:
                # Update market conditions
                await self._update_market_conditions(state)
                
                # Check for adverse price movements
                await self._check_stop_conditions(state)
                
                # Adapt algorithm if needed
                if state.algorithm_type == AlgorithmType.ADAPTIVE:
                    await self._adapt_algorithm(state)
                
            except Exception as e:
                logger.error(f"Error updating algorithm state {order_id}: {str(e)}")
    
    async def _update_market_conditions(self, state: AlgorithmState) -> None:
        """Update current market conditions for algorithm"""
        
        if self.market_data_provider:
            try:
                # Get current market data (simulate for now)
                state.current_price *= random.uniform(0.999, 1.001)  # Small price movement
                state.current_volume = random.randint(1000, 5000)
                state.volatility = max(0.01, state.volatility * random.uniform(0.98, 1.02))
                
            except Exception as e:
                logger.warning(f"Failed to update market conditions: {str(e)}")
    
    async def _check_stop_conditions(self, state: AlgorithmState) -> None:
        """Check if algorithm should be stopped due to adverse conditions"""
        
        if state.config.stop_on_adverse_move > 0:
            # This would check against the arrival price in production
            adverse_threshold = state.config.stop_on_adverse_move
            
            # Simulate adverse move check
            if random.random() < 0.001:  # 0.1% chance of stop condition
                state.is_active = False
                state.stop_reason = "Adverse price movement detected"
                logger.warning(
                    "Algorithm stopped due to adverse conditions",
                    order_id=state.order_id,
                    reason=state.stop_reason
                )
    
    async def _adapt_algorithm(self, state: AlgorithmState) -> None:
        """Adapt algorithm execution based on performance"""
        
        if state.algorithm_type != AlgorithmType.ADAPTIVE:
            return
        
        # Simple adaptation: adjust urgency based on recent performance
        if state.slices_executed >= 3:
            # Calculate recent performance
            recent_slices = self.pending_slices[state.order_id][-3:]
            avg_fill_ratio = sum(s.filled_quantity / s.quantity for s in recent_slices if s.submitted) / len(recent_slices)
            
            # Adjust future slice urgency
            remaining_slices = [s for s in self.pending_slices[state.order_id] if not s.submitted]
            
            if avg_fill_ratio < 0.9:  # Poor fill rate
                for slice_order in remaining_slices:
                    slice_order.urgency *= 0.9  # Be more patient
            elif avg_fill_ratio > 0.98:  # Excellent fill rate
                for slice_order in remaining_slices:
                    slice_order.urgency *= 1.1  # Can be more aggressive
    
    async def _complete_algorithm(self, state: AlgorithmState) -> None:
        """Complete algorithm execution"""
        
        state.is_active = False
        
        # Calculate performance metrics
        state.implementation_shortfall = self._calculate_implementation_shortfall(state)
        state.market_impact_bps = self._calculate_market_impact(state)
        
        # Record execution history
        self.execution_history.append({
            'order_id': state.order_id,
            'algorithm_type': state.algorithm_type.value,
            'completion_time': datetime.now().isoformat(),
            'executed_quantity': state.executed_quantity,
            'avg_execution_price': state.avg_execution_price,
            'implementation_shortfall': state.implementation_shortfall,
            'market_impact_bps': state.market_impact_bps,
            'slices_executed': state.slices_executed
        })
        
        logger.info(
            "Algorithm completed",
            order_id=state.order_id,
            algorithm_type=state.algorithm_type.value,
            executed_quantity=state.executed_quantity,
            implementation_shortfall=state.implementation_shortfall
        )
    
    def _calculate_implementation_shortfall(self, state: AlgorithmState) -> float:
        """Calculate implementation shortfall for completed algorithm"""
        # Simplified calculation - would be more sophisticated in production
        return random.uniform(-0.002, 0.002)  # -20 to +20 bps
    
    def _calculate_market_impact(self, state: AlgorithmState) -> float:
        """Calculate market impact in basis points"""
        # Simplified calculation based on quantity and volatility
        quantity_factor = state.executed_quantity / 100000  # Scale factor
        impact = quantity_factor * state.volatility * 100  # Convert to bps
        return min(impact, 50.0)  # Cap at 50 bps
    
    def _get_historical_performance(self, symbol: str) -> Dict[str, float]:
        """Get historical algorithm performance for symbol"""
        # In production, this would query a database
        return {
            'avg_implementation_shortfall': random.uniform(-0.001, 0.001),
            'avg_market_impact': random.uniform(1.0, 5.0),
            'avg_fill_rate': random.uniform(0.95, 0.99)
        }
    
    def _adapt_strategy(self, state: AlgorithmState, performance: Dict[str, float]) -> None:
        """Adapt strategy based on historical performance"""
        
        # Adjust parameters based on historical performance
        if performance['avg_implementation_shortfall'] > 0.001:  # Poor historical IS
            state.config.max_participation_rate *= 0.8  # Be more patient
        
        if performance['avg_fill_rate'] < 0.97:  # Poor fill rate
            state.config.max_participation_rate *= 0.9
    
    async def pause_algorithm(self, order_id: str) -> bool:
        """Pause algorithm execution"""
        if order_id in self.active_algorithms:
            self.active_algorithms[order_id].is_paused = True
            logger.info(f"Algorithm paused: {order_id}")
            return True
        return False
    
    async def resume_algorithm(self, order_id: str) -> bool:
        """Resume algorithm execution"""
        if order_id in self.active_algorithms:
            self.active_algorithms[order_id].is_paused = False
            logger.info(f"Algorithm resumed: {order_id}")
            return True
        return False
    
    async def stop_algorithm(self, order_id: str, reason: str = "User requested") -> bool:
        """Stop algorithm execution"""
        if order_id in self.active_algorithms:
            state = self.active_algorithms[order_id]
            state.is_active = False
            state.stop_reason = reason
            await self._complete_algorithm(state)
            return True
        return False
    
    def get_algorithm_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get current algorithm status"""
        if order_id not in self.active_algorithms:
            return None
        
        state = self.active_algorithms[order_id]
        slices = self.pending_slices.get(order_id, [])
        
        return {
            'order_id': order_id,
            'algorithm_type': state.algorithm_type.value,
            'is_active': state.is_active,
            'is_paused': state.is_paused,
            'progress': {
                'total_quantity': state.total_quantity,
                'executed_quantity': state.executed_quantity,
                'remaining_quantity': state.remaining_quantity,
                'completion_pct': (state.executed_quantity / state.total_quantity) * 100
            },
            'performance': {
                'avg_execution_price': state.avg_execution_price,
                'slices_executed': state.slices_executed,
                'implementation_shortfall': state.implementation_shortfall
            },
            'slices': {
                'total': len(slices),
                'submitted': sum(1 for s in slices if s.submitted),
                'completed': sum(1 for s in slices if s.execution_time is not None)
            }
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get algorithm performance summary"""
        
        if not self.execution_history:
            return {'message': 'No completed algorithms'}
        
        # Calculate aggregate statistics
        total_algorithms = len(self.execution_history)
        avg_implementation_shortfall = sum(
            a['implementation_shortfall'] for a in self.execution_history
        ) / total_algorithms
        
        avg_market_impact = sum(
            a['market_impact_bps'] for a in self.execution_history
        ) / total_algorithms
        
        # Performance by algorithm type
        by_algorithm = {}
        for algo_type in AlgorithmType:
            algo_records = [a for a in self.execution_history if a['algorithm_type'] == algo_type.value]
            if algo_records:
                by_algorithm[algo_type.value] = {
                    'count': len(algo_records),
                    'avg_implementation_shortfall': sum(a['implementation_shortfall'] for a in algo_records) / len(algo_records),
                    'avg_market_impact': sum(a['market_impact_bps'] for a in algo_records) / len(algo_records)
                }
        
        return {
            'total_algorithms': total_algorithms,
            'overall_performance': {
                'avg_implementation_shortfall': avg_implementation_shortfall,
                'avg_market_impact_bps': avg_market_impact
            },
            'by_algorithm_type': by_algorithm,
            'active_algorithms': len(self.active_algorithms)
        }
    
    async def shutdown(self) -> None:
        """Shutdown algorithm engine"""
        logger.info("Shutting down algorithm engine")
        
        # Stop all active algorithms
        for order_id in list(self.active_algorithms.keys()):
            await self.stop_algorithm(order_id, "System shutdown")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Wait for execution task to complete
        if self.execution_task and not self.execution_task.done():
            await self.execution_task
        
        logger.info("Algorithm engine shutdown complete")