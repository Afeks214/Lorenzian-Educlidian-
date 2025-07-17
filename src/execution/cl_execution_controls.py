"""
CL Execution Controls System
===========================

Advanced execution controls for CL crude oil trading with realistic market impact modeling.
Implements execution algorithms, slippage models, and market impact analysis
specifically designed for commodity futures markets.

Key Features:
- Realistic market impact modeling for CL futures
- Volume-based execution algorithms
- Slippage and fill quality analysis
- Execution cost estimation
- Optimal order placement strategies
- Real-time execution monitoring

Author: Agent 4 - Risk Management Mission
Date: 2025-07-17
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"

class ExecutionAlgorithm(Enum):
    """Execution algorithm types"""
    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"
    BALANCED = "balanced"
    STEALTH = "stealth"
    ADAPTIVE = "adaptive"

class FillQuality(Enum):
    """Fill quality assessment"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    VERY_POOR = "very_poor"

@dataclass
class MarketImpactModel:
    """Market impact model parameters"""
    linear_coefficient: float = 0.1
    square_root_coefficient: float = 0.5
    temporary_impact_decay: float = 0.7
    permanent_impact_ratio: float = 0.3
    volatility_adjustment: float = 1.0
    liquidity_adjustment: float = 1.0

@dataclass
class ExecutionOrder:
    """Execution order data structure"""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    algorithm: ExecutionAlgorithm = ExecutionAlgorithm.BALANCED
    urgency: float = 0.5  # 0-1 scale
    participation_rate: float = 0.1  # Max 10% of volume
    min_fill_size: float = 1.0
    max_fill_size: Optional[float] = None
    time_in_force: str = "DAY"
    created_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionFill:
    """Execution fill data structure"""
    fill_id: str
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    execution_venue: str = "NYMEX"
    slippage: float = 0.0
    market_impact: float = 0.0
    execution_cost: float = 0.0
    fill_quality: FillQuality = FillQuality.GOOD
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionMetrics:
    """Execution performance metrics"""
    total_orders: int = 0
    total_fills: int = 0
    total_volume: float = 0.0
    total_cost: float = 0.0
    average_slippage: float = 0.0
    average_market_impact: float = 0.0
    fill_rate: float = 0.0
    execution_shortfall: float = 0.0
    implementation_shortfall: float = 0.0
    arrival_price_performance: float = 0.0

class CLExecutionEngine:
    """
    Advanced execution engine for CL crude oil trading
    
    Implements sophisticated execution algorithms with realistic market impact
    modeling and cost analysis specifically designed for commodity futures.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CL Execution Engine
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # CL-specific parameters
        self.cl_contract_size = config.get('cl_contract_size', 1000)  # barrels
        self.cl_tick_size = config.get('cl_tick_size', 0.01)
        self.cl_tick_value = config.get('cl_tick_value', 10.0)
        self.cl_average_volume = config.get('cl_average_volume', 500000)  # daily volume
        
        # Market impact model
        self.impact_model = MarketImpactModel(**config.get('market_impact_model', {}))
        
        # Execution parameters
        self.execution_config = config.get('execution_parameters', {})
        self.max_participation_rate = self.execution_config.get('max_participation_rate', 0.15)
        self.min_order_size = self.execution_config.get('min_order_size', 1.0)
        self.max_order_size = self.execution_config.get('max_order_size', 100.0)
        
        # Slippage model
        self.slippage_config = config.get('slippage_model', {})
        self.base_slippage = self.slippage_config.get('base_slippage', 0.0005)
        self.volatility_multiplier = self.slippage_config.get('volatility_multiplier', 2.0)
        
        # Order tracking
        self.active_orders: Dict[str, ExecutionOrder] = {}
        self.completed_orders: Dict[str, ExecutionOrder] = {}
        self.execution_fills: List[ExecutionFill] = []
        
        # Performance tracking
        self.execution_metrics = ExecutionMetrics()
        
        # Market data cache
        self.market_data_cache = {}
        self.order_book_cache = {}
        self.volume_profile_cache = {}
        
        # Real-time monitoring
        self.execution_alerts = []
        self.performance_degradation_threshold = 0.1  # 10 bps
        
        logger.info("✅ CL Execution Engine initialized")
        logger.info(f"   📊 Contract Size: {self.cl_contract_size} barrels")
        logger.info(f"   📊 Tick Size: ${self.cl_tick_size}")
        logger.info(f"   📊 Max Participation Rate: {self.max_participation_rate:.1%}")
        logger.info(f"   📊 Base Slippage: {self.base_slippage:.4f}")
    
    async def submit_order(self, order: ExecutionOrder) -> Dict[str, Any]:
        """
        Submit order for execution
        
        Args:
            order: Execution order
            
        Returns:
            Order submission result
        """
        try:
            # Validate order
            validation_result = await self._validate_order(order)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'order_id': order.order_id,
                    'error': validation_result['error'],
                    'timestamp': datetime.now().isoformat()
                }
            
            # Calculate execution parameters
            execution_params = await self._calculate_execution_parameters(order)
            
            # Estimate execution cost
            cost_estimate = await self._estimate_execution_cost(order, execution_params)
            
            # Add to active orders
            self.active_orders[order.order_id] = order
            
            # Start execution process
            execution_result = await self._execute_order(order, execution_params, cost_estimate)
            
            # Update metrics
            self.execution_metrics.total_orders += 1
            
            logger.info(f"✅ Order submitted: {order.order_id} {order.side} {order.quantity} {order.symbol}")
            
            return {
                'success': True,
                'order_id': order.order_id,
                'execution_params': execution_params,
                'cost_estimate': cost_estimate,
                'execution_result': execution_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error submitting order {order.order_id}: {e}")
            return {
                'success': False,
                'order_id': order.order_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _validate_order(self, order: ExecutionOrder) -> Dict[str, Any]:
        """Validate order parameters"""
        try:
            # Check minimum order size
            if order.quantity < self.min_order_size:
                return {
                    'valid': False,
                    'error': f"Order size {order.quantity} below minimum {self.min_order_size}"
                }
            
            # Check maximum order size
            if order.quantity > self.max_order_size:
                return {
                    'valid': False,
                    'error': f"Order size {order.quantity} above maximum {self.max_order_size}"
                }
            
            # Check symbol format
            if not order.symbol.startswith('CL'):
                return {
                    'valid': False,
                    'error': f"Invalid symbol format: {order.symbol}"
                }
            
            # Check side
            if order.side not in ['buy', 'sell']:
                return {
                    'valid': False,
                    'error': f"Invalid side: {order.side}"
                }
            
            # Check participation rate
            if order.participation_rate > self.max_participation_rate:
                return {
                    'valid': False,
                    'error': f"Participation rate {order.participation_rate:.1%} exceeds maximum {self.max_participation_rate:.1%}"
                }
            
            # Check limit price for limit orders
            if order.order_type == OrderType.LIMIT and order.limit_price is None:
                return {
                    'valid': False,
                    'error': "Limit price required for limit orders"
                }
            
            # Check stop price for stop orders
            if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price is None:
                return {
                    'valid': False,
                    'error': "Stop price required for stop orders"
                }
            
            return {'valid': True}
            
        except Exception as e:
            return {
                'valid': False,
                'error': f"Order validation error: {e}"
            }
    
    async def _calculate_execution_parameters(self, order: ExecutionOrder) -> Dict[str, Any]:
        """Calculate execution parameters based on order and market conditions"""
        try:
            # Get current market data
            market_data = await self._get_market_data(order.symbol)
            
            # Calculate current liquidity
            current_liquidity = await self._calculate_current_liquidity(order.symbol, market_data)
            
            # Calculate optimal slice size
            optimal_slice_size = await self._calculate_optimal_slice_size(order, market_data, current_liquidity)
            
            # Calculate timing parameters
            timing_params = await self._calculate_timing_parameters(order, market_data)
            
            # Select execution venue
            execution_venue = await self._select_execution_venue(order, market_data)
            
            return {
                'optimal_slice_size': optimal_slice_size,
                'timing_params': timing_params,
                'execution_venue': execution_venue,
                'current_liquidity': current_liquidity,
                'market_data': market_data,
                'estimated_duration': timing_params.get('total_duration', 0)
            }
            
        except Exception as e:
            logger.error(f"Error calculating execution parameters: {e}")
            return {
                'optimal_slice_size': min(order.quantity, 10.0),
                'timing_params': {'interval': 30, 'total_duration': 300},
                'execution_venue': 'NYMEX',
                'current_liquidity': 0.5,
                'error': str(e)
            }
    
    async def _get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get current market data for symbol"""
        try:
            # Simulate market data retrieval
            # In production, this would fetch from real market data feed
            
            current_time = datetime.now()
            
            # Generate realistic CL market data
            base_price = 75.0  # Base CL price
            volatility = 0.02  # Daily volatility
            
            # Add some random variation
            price_change = np.random.normal(0, volatility)
            current_price = base_price * (1 + price_change)
            
            # Generate bid/ask spread
            spread = max(0.01, np.random.normal(0.02, 0.005))  # 2 bps average spread
            
            # Generate volume
            session_volume = np.random.lognormal(np.log(self.cl_average_volume), 0.3)
            
            market_data = {
                'symbol': symbol,
                'timestamp': current_time.isoformat(),
                'bid': current_price - spread / 2,
                'ask': current_price + spread / 2,
                'last': current_price,
                'volume': session_volume,
                'spread': spread,
                'volatility': volatility,
                'vwap': current_price * (1 + np.random.normal(0, 0.001)),
                'twap': current_price * (1 + np.random.normal(0, 0.001))
            }
            
            # Cache market data
            self.market_data_cache[symbol] = market_data
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'bid': 75.0,
                'ask': 75.02,
                'last': 75.01,
                'volume': self.cl_average_volume,
                'spread': 0.02,
                'volatility': 0.02,
                'error': str(e)
            }
    
    async def _calculate_current_liquidity(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """Calculate current market liquidity"""
        try:
            # Base liquidity on volume and spread
            volume = market_data.get('volume', self.cl_average_volume)
            spread = market_data.get('spread', 0.02)
            
            # Normalize volume (higher volume = higher liquidity)
            volume_factor = volume / self.cl_average_volume
            
            # Normalize spread (lower spread = higher liquidity)
            spread_factor = 0.02 / spread
            
            # Session adjustment
            current_hour = datetime.now().hour
            if 9 <= current_hour <= 16:  # US session
                session_factor = 1.0
            elif 2 <= current_hour <= 9:  # European session
                session_factor = 0.8
            else:  # Asian/overnight
                session_factor = 0.6
            
            # Combined liquidity score
            liquidity_score = (volume_factor * 0.6 + spread_factor * 0.4) * session_factor
            
            return min(liquidity_score, 2.0)  # Cap at 2.0
            
        except Exception as e:
            logger.error(f"Error calculating liquidity: {e}")
            return 0.5  # Default moderate liquidity
    
    async def _calculate_optimal_slice_size(self, 
                                          order: ExecutionOrder,
                                          market_data: Dict[str, Any],
                                          liquidity: float) -> float:
        """Calculate optimal slice size for order execution"""
        try:
            volume = market_data.get('volume', self.cl_average_volume)
            
            # Base slice size on participation rate
            max_slice_by_participation = volume * order.participation_rate
            
            # Adjust for urgency
            urgency_multiplier = 1.0 + (order.urgency - 0.5) * 0.5
            
            # Adjust for liquidity
            liquidity_multiplier = min(liquidity, 1.5)
            
            # Adjust for algorithm type
            algorithm_multipliers = {
                ExecutionAlgorithm.AGGRESSIVE: 1.5,
                ExecutionAlgorithm.PASSIVE: 0.5,
                ExecutionAlgorithm.BALANCED: 1.0,
                ExecutionAlgorithm.STEALTH: 0.3,
                ExecutionAlgorithm.ADAPTIVE: 1.0
            }
            
            algorithm_multiplier = algorithm_multipliers.get(order.algorithm, 1.0)
            
            # Calculate optimal slice size
            optimal_slice = (max_slice_by_participation * urgency_multiplier * 
                           liquidity_multiplier * algorithm_multiplier)
            
            # Apply min/max constraints
            if order.min_fill_size:
                optimal_slice = max(optimal_slice, order.min_fill_size)
            
            if order.max_fill_size:
                optimal_slice = min(optimal_slice, order.max_fill_size)
            
            # Don't exceed remaining quantity
            optimal_slice = min(optimal_slice, order.quantity)
            
            return optimal_slice
            
        except Exception as e:
            logger.error(f"Error calculating optimal slice size: {e}")
            return min(order.quantity, 10.0)  # Default to reasonable slice
    
    async def _calculate_timing_parameters(self, 
                                         order: ExecutionOrder,
                                         market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal timing parameters"""
        try:
            # Base timing on order quantity and market conditions
            volume = market_data.get('volume', self.cl_average_volume)
            
            # Estimate total execution time
            participation_volume = volume * order.participation_rate
            estimated_duration = (order.quantity / participation_volume) * 3600  # seconds
            
            # Adjust for urgency
            urgency_adjustment = 1.0 / (order.urgency + 0.1)
            adjusted_duration = estimated_duration * urgency_adjustment
            
            # Calculate slice interval
            optimal_slice_size = await self._calculate_optimal_slice_size(order, market_data, 1.0)
            num_slices = max(1, order.quantity / optimal_slice_size)
            slice_interval = adjusted_duration / num_slices
            
            # Apply minimum and maximum intervals
            slice_interval = max(slice_interval, 5.0)   # Minimum 5 seconds
            slice_interval = min(slice_interval, 300.0)  # Maximum 5 minutes
            
            return {
                'total_duration': adjusted_duration,
                'slice_interval': slice_interval,
                'num_slices': int(num_slices),
                'participation_volume': participation_volume,
                'estimated_completion': (datetime.now() + timedelta(seconds=adjusted_duration)).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating timing parameters: {e}")
            return {
                'total_duration': 300,
                'slice_interval': 30,
                'num_slices': 10,
                'participation_volume': 1000,
                'error': str(e)
            }
    
    async def _select_execution_venue(self, 
                                    order: ExecutionOrder,
                                    market_data: Dict[str, Any]) -> str:
        """Select optimal execution venue"""
        try:
            # For CL, primary venue is NYMEX
            # In production, this could consider multiple venues
            
            volume = market_data.get('volume', self.cl_average_volume)
            spread = market_data.get('spread', 0.02)
            
            # Simple venue selection logic
            if volume > self.cl_average_volume * 1.5 and spread < 0.015:
                return "NYMEX_ELECTRONIC"
            elif order.quantity > 50.0:
                return "NYMEX_BLOCK"
            else:
                return "NYMEX"
            
        except Exception as e:
            logger.error(f"Error selecting execution venue: {e}")
            return "NYMEX"
    
    async def _estimate_execution_cost(self, 
                                     order: ExecutionOrder,
                                     execution_params: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate execution cost components"""
        try:
            market_data = execution_params.get('market_data', {})
            current_price = market_data.get('last', 75.0)
            volume = market_data.get('volume', self.cl_average_volume)
            spread = market_data.get('spread', 0.02)
            volatility = market_data.get('volatility', 0.02)
            
            # Calculate market impact
            market_impact = await self._calculate_market_impact(order, market_data)
            
            # Calculate slippage
            slippage = await self._calculate_expected_slippage(order, market_data)
            
            # Calculate timing cost
            timing_cost = await self._calculate_timing_cost(order, execution_params)
            
            # Calculate commission
            commission = self._calculate_commission(order)
            
            # Calculate opportunity cost
            opportunity_cost = await self._calculate_opportunity_cost(order, execution_params)
            
            # Total cost
            total_cost = market_impact + slippage + timing_cost + commission + opportunity_cost
            
            # Cost in basis points
            cost_bps = (total_cost / current_price) * 10000
            
            return {
                'market_impact': market_impact,
                'slippage': slippage,
                'timing_cost': timing_cost,
                'commission': commission,
                'opportunity_cost': opportunity_cost,
                'total_cost': total_cost,
                'cost_bps': cost_bps,
                'cost_percentage': total_cost / current_price,
                'breakdown': {
                    'market_impact_bps': (market_impact / current_price) * 10000,
                    'slippage_bps': (slippage / current_price) * 10000,
                    'timing_cost_bps': (timing_cost / current_price) * 10000,
                    'commission_bps': (commission / current_price) * 10000,
                    'opportunity_cost_bps': (opportunity_cost / current_price) * 10000
                }
            }
            
        except Exception as e:
            logger.error(f"Error estimating execution cost: {e}")
            return {
                'total_cost': 0.0,
                'cost_bps': 0.0,
                'error': str(e)
            }
    
    async def _calculate_market_impact(self, 
                                     order: ExecutionOrder,
                                     market_data: Dict[str, Any]) -> float:
        """Calculate market impact using impact model"""
        try:
            volume = market_data.get('volume', self.cl_average_volume)
            current_price = market_data.get('last', 75.0)
            volatility = market_data.get('volatility', 0.02)
            
            # Participation rate
            participation_rate = order.quantity / volume
            
            # Linear impact component
            linear_impact = (self.impact_model.linear_coefficient * 
                           participation_rate * current_price)
            
            # Square root impact component
            sqrt_impact = (self.impact_model.square_root_coefficient * 
                          np.sqrt(participation_rate) * current_price)
            
            # Volatility adjustment
            volatility_adjustment = self.impact_model.volatility_adjustment * volatility
            
            # Total temporary impact
            temporary_impact = (linear_impact + sqrt_impact) * (1 + volatility_adjustment)
            
            # Permanent impact
            permanent_impact = temporary_impact * self.impact_model.permanent_impact_ratio
            
            # Total market impact
            total_impact = temporary_impact + permanent_impact
            
            return total_impact
            
        except Exception as e:
            logger.error(f"Error calculating market impact: {e}")
            return 0.0
    
    async def _calculate_expected_slippage(self, 
                                         order: ExecutionOrder,
                                         market_data: Dict[str, Any]) -> float:
        """Calculate expected slippage"""
        try:
            current_price = market_data.get('last', 75.0)
            volatility = market_data.get('volatility', 0.02)
            spread = market_data.get('spread', 0.02)
            
            # Base slippage
            base_slippage = self.base_slippage * current_price
            
            # Volatility adjustment
            volatility_slippage = volatility * self.volatility_multiplier * current_price
            
            # Spread component
            spread_slippage = spread * 0.5  # Half spread on average
            
            # Order type adjustment
            order_type_multipliers = {
                OrderType.MARKET: 1.0,
                OrderType.LIMIT: 0.3,
                OrderType.STOP: 1.2,
                OrderType.STOP_LIMIT: 0.8,
                OrderType.ICEBERG: 0.6,
                OrderType.TWAP: 0.5,
                OrderType.VWAP: 0.4
            }
            
            order_type_multiplier = order_type_multipliers.get(order.order_type, 1.0)
            
            # Urgency adjustment
            urgency_multiplier = 1.0 + (order.urgency - 0.5) * 0.3
            
            # Total slippage
            total_slippage = ((base_slippage + volatility_slippage + spread_slippage) * 
                            order_type_multiplier * urgency_multiplier)
            
            return total_slippage
            
        except Exception as e:
            logger.error(f"Error calculating slippage: {e}")
            return 0.0
    
    async def _calculate_timing_cost(self, 
                                   order: ExecutionOrder,
                                   execution_params: Dict[str, Any]) -> float:
        """Calculate timing cost"""
        try:
            market_data = execution_params.get('market_data', {})
            current_price = market_data.get('last', 75.0)
            volatility = market_data.get('volatility', 0.02)
            
            # Execution duration
            duration = execution_params.get('timing_params', {}).get('total_duration', 300)
            duration_hours = duration / 3600
            
            # Timing cost based on volatility and duration
            timing_cost = volatility * np.sqrt(duration_hours) * current_price * 0.5
            
            return timing_cost
            
        except Exception as e:
            logger.error(f"Error calculating timing cost: {e}")
            return 0.0
    
    def _calculate_commission(self, order: ExecutionOrder) -> float:
        """Calculate commission cost"""
        try:
            # CL commission structure (simplified)
            base_commission = 2.50  # Base commission per contract
            
            # Volume discounts
            if order.quantity >= 50:
                discount = 0.2
            elif order.quantity >= 20:
                discount = 0.1
            else:
                discount = 0.0
            
            commission_per_contract = base_commission * (1 - discount)
            total_commission = commission_per_contract * order.quantity
            
            return total_commission
            
        except Exception as e:
            logger.error(f"Error calculating commission: {e}")
            return 0.0
    
    async def _calculate_opportunity_cost(self, 
                                        order: ExecutionOrder,
                                        execution_params: Dict[str, Any]) -> float:
        """Calculate opportunity cost"""
        try:
            market_data = execution_params.get('market_data', {})
            current_price = market_data.get('last', 75.0)
            
            # Opportunity cost based on delay
            duration = execution_params.get('timing_params', {}).get('total_duration', 300)
            duration_hours = duration / 3600
            
            # Simplified opportunity cost calculation
            # Based on expected price drift during execution
            opportunity_cost = current_price * 0.001 * duration_hours  # 0.1% per hour
            
            return opportunity_cost * order.quantity
            
        except Exception as e:
            logger.error(f"Error calculating opportunity cost: {e}")
            return 0.0
    
    async def _execute_order(self, 
                           order: ExecutionOrder,
                           execution_params: Dict[str, Any],
                           cost_estimate: Dict[str, Any]) -> Dict[str, Any]:
        """Execute order using specified algorithm"""
        try:
            logger.info(f"Executing order {order.order_id} using {order.algorithm.value} algorithm")
            
            # Simulate order execution
            fills = []
            remaining_quantity = order.quantity
            total_cost = 0.0
            
            # Get execution parameters
            optimal_slice_size = execution_params.get('optimal_slice_size', 10.0)
            slice_interval = execution_params.get('timing_params', {}).get('slice_interval', 30)
            
            # Execute in slices
            slice_count = 0
            while remaining_quantity > 0 and slice_count < 100:  # Safety limit
                # Calculate slice size
                slice_size = min(remaining_quantity, optimal_slice_size)
                
                # Simulate fill
                fill = await self._simulate_fill(order, slice_size, execution_params)
                fills.append(fill)
                
                # Update remaining quantity
                remaining_quantity -= fill.quantity
                total_cost += fill.execution_cost
                
                # Update metrics
                self.execution_metrics.total_fills += 1
                self.execution_metrics.total_volume += fill.quantity
                
                slice_count += 1
                
                # Break if fully filled
                if remaining_quantity <= 0:
                    break
                
                # Wait for next slice (in simulation, we skip the wait)
                # await asyncio.sleep(slice_interval)
            
            # Calculate execution performance
            execution_performance = await self._calculate_execution_performance(
                order, fills, cost_estimate
            )
            
            # Move order to completed
            self.completed_orders[order.order_id] = order
            if order.order_id in self.active_orders:
                del self.active_orders[order.order_id]
            
            # Store fills
            self.execution_fills.extend(fills)
            
            # Update metrics
            self.execution_metrics.total_cost += total_cost
            self.execution_metrics.fill_rate = len(fills) / max(slice_count, 1)
            
            logger.info(f"✅ Order {order.order_id} executed: {len(fills)} fills, total cost: ${total_cost:.2f}")
            
            return {
                'order_id': order.order_id,
                'status': 'completed',
                'fills': [fill.__dict__ for fill in fills],
                'total_fills': len(fills),
                'total_quantity': sum(fill.quantity for fill in fills),
                'average_price': sum(fill.price * fill.quantity for fill in fills) / sum(fill.quantity for fill in fills),
                'total_cost': total_cost,
                'execution_performance': execution_performance,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error executing order {order.order_id}: {e}")
            return {
                'order_id': order.order_id,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _simulate_fill(self, 
                           order: ExecutionOrder,
                           slice_size: float,
                           execution_params: Dict[str, Any]) -> ExecutionFill:
        """Simulate order fill"""
        try:
            market_data = execution_params.get('market_data', {})
            current_price = market_data.get('last', 75.0)
            spread = market_data.get('spread', 0.02)
            
            # Simulate fill price
            if order.side == 'buy':
                base_price = market_data.get('ask', current_price + spread / 2)
            else:
                base_price = market_data.get('bid', current_price - spread / 2)
            
            # Add random slippage
            slippage = np.random.normal(0, self.base_slippage * current_price)
            fill_price = base_price + slippage
            
            # Calculate market impact for this slice
            market_impact = await self._calculate_market_impact(
                ExecutionOrder(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=slice_size,
                    order_type=order.order_type
                ),
                market_data
            )
            
            # Adjust fill price for market impact
            if order.side == 'buy':
                fill_price += market_impact
            else:
                fill_price -= market_impact
            
            # Calculate execution cost
            execution_cost = self._calculate_commission(
                ExecutionOrder(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=slice_size,
                    order_type=order.order_type
                )
            )
            
            # Assess fill quality
            fill_quality = self._assess_fill_quality(fill_price, current_price, market_impact, slippage)
            
            # Create fill
            fill = ExecutionFill(
                fill_id=f"fill_{order.order_id}_{len(self.execution_fills)}",
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=slice_size,
                price=fill_price,
                timestamp=datetime.now(),
                execution_venue=execution_params.get('execution_venue', 'NYMEX'),
                slippage=slippage,
                market_impact=market_impact,
                execution_cost=execution_cost,
                fill_quality=fill_quality
            )
            
            return fill
            
        except Exception as e:
            logger.error(f"Error simulating fill: {e}")
            return ExecutionFill(
                fill_id=f"fill_{order.order_id}_error",
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=slice_size,
                price=75.0,
                timestamp=datetime.now(),
                execution_cost=0.0,
                fill_quality=FillQuality.POOR
            )
    
    def _assess_fill_quality(self, 
                           fill_price: float,
                           reference_price: float,
                           market_impact: float,
                           slippage: float) -> FillQuality:
        """Assess fill quality"""
        try:
            # Calculate performance relative to reference
            relative_performance = abs(fill_price - reference_price) / reference_price
            
            # Quality thresholds
            if relative_performance < 0.001:  # < 10 bps
                return FillQuality.EXCELLENT
            elif relative_performance < 0.002:  # < 20 bps
                return FillQuality.GOOD
            elif relative_performance < 0.005:  # < 50 bps
                return FillQuality.FAIR
            elif relative_performance < 0.01:  # < 100 bps
                return FillQuality.POOR
            else:
                return FillQuality.VERY_POOR
                
        except Exception as e:
            logger.error(f"Error assessing fill quality: {e}")
            return FillQuality.FAIR
    
    async def _calculate_execution_performance(self, 
                                             order: ExecutionOrder,
                                             fills: List[ExecutionFill],
                                             cost_estimate: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate execution performance metrics"""
        try:
            if not fills:
                return {'error': 'No fills to analyze'}
            
            # Calculate average fill price
            total_quantity = sum(fill.quantity for fill in fills)
            average_price = sum(fill.price * fill.quantity for fill in fills) / total_quantity
            
            # Calculate total costs
            total_slippage = sum(fill.slippage for fill in fills)
            total_market_impact = sum(fill.market_impact for fill in fills)
            total_execution_cost = sum(fill.execution_cost for fill in fills)
            
            # Calculate vs. estimate
            estimated_cost = cost_estimate.get('total_cost', 0.0)
            cost_variance = total_execution_cost - estimated_cost
            
            # Calculate fill rate
            fill_rate = len(fills) / max(1, order.quantity / 10)  # Assuming 10 contracts per expected fill
            
            # Calculate implementation shortfall
            # (Simplified - would need arrival price in production)
            implementation_shortfall = total_market_impact + total_slippage
            
            return {
                'total_fills': len(fills),
                'total_quantity': total_quantity,
                'average_price': average_price,
                'total_slippage': total_slippage,
                'total_market_impact': total_market_impact,
                'total_execution_cost': total_execution_cost,
                'estimated_cost': estimated_cost,
                'cost_variance': cost_variance,
                'fill_rate': fill_rate,
                'implementation_shortfall': implementation_shortfall,
                'average_fill_quality': np.mean([fill.fill_quality.value for fill in fills]),
                'execution_time': (fills[-1].timestamp - fills[0].timestamp).total_seconds() if len(fills) > 1 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating execution performance: {e}")
            return {'error': str(e)}
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get comprehensive execution summary"""
        try:
            # Calculate performance metrics
            if self.execution_metrics.total_volume > 0:
                avg_slippage = self.execution_metrics.total_cost / self.execution_metrics.total_volume
            else:
                avg_slippage = 0.0
            
            # Fill quality distribution
            fill_quality_dist = {}
            for fill in self.execution_fills:
                quality = fill.fill_quality.value
                fill_quality_dist[quality] = fill_quality_dist.get(quality, 0) + 1
            
            # Recent performance
            recent_fills = self.execution_fills[-100:] if len(self.execution_fills) > 100 else self.execution_fills
            recent_avg_cost = np.mean([fill.execution_cost for fill in recent_fills]) if recent_fills else 0.0
            
            return {
                'timestamp': datetime.now().isoformat(),
                'total_orders': self.execution_metrics.total_orders,
                'total_fills': self.execution_metrics.total_fills,
                'total_volume': self.execution_metrics.total_volume,
                'total_cost': self.execution_metrics.total_cost,
                'average_slippage': avg_slippage,
                'fill_rate': self.execution_metrics.fill_rate,
                'active_orders': len(self.active_orders),
                'completed_orders': len(self.completed_orders),
                'fill_quality_distribution': fill_quality_dist,
                'recent_performance': {
                    'recent_fills': len(recent_fills),
                    'recent_avg_cost': recent_avg_cost,
                    'recent_avg_slippage': np.mean([fill.slippage for fill in recent_fills]) if recent_fills else 0.0
                },
                'execution_venues': list(set(fill.execution_venue for fill in self.execution_fills)),
                'performance_alerts': len(self.execution_alerts)
            }
            
        except Exception as e:
            logger.error(f"Error generating execution summary: {e}")
            return {'error': str(e)}
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get status of specific order"""
        try:
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                status = 'active'
            elif order_id in self.completed_orders:
                order = self.completed_orders[order_id]
                status = 'completed'
            else:
                return {'error': f'Order {order_id} not found'}
            
            # Get fills for this order
            order_fills = [fill for fill in self.execution_fills if fill.order_id == order_id]
            
            return {
                'order_id': order_id,
                'status': status,
                'order': order.__dict__,
                'fills': [fill.__dict__ for fill in order_fills],
                'total_fills': len(order_fills),
                'filled_quantity': sum(fill.quantity for fill in order_fills),
                'remaining_quantity': order.quantity - sum(fill.quantity for fill in order_fills),
                'average_fill_price': sum(fill.price * fill.quantity for fill in order_fills) / sum(fill.quantity for fill in order_fills) if order_fills else 0,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return {'error': str(e)}
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel active order"""
        try:
            if order_id not in self.active_orders:
                return {
                    'success': False,
                    'error': f'Order {order_id} not found or not active'
                }
            
            order = self.active_orders[order_id]
            
            # Move to completed orders
            self.completed_orders[order_id] = order
            del self.active_orders[order_id]
            
            # Get fills so far
            order_fills = [fill for fill in self.execution_fills if fill.order_id == order_id]
            filled_quantity = sum(fill.quantity for fill in order_fills)
            
            logger.info(f"Order {order_id} cancelled: {filled_quantity}/{order.quantity} filled")
            
            return {
                'success': True,
                'order_id': order_id,
                'filled_quantity': filled_quantity,
                'remaining_quantity': order.quantity - filled_quantity,
                'fills': [fill.__dict__ for fill in order_fills],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def reset_daily_metrics(self):
        """Reset daily execution metrics"""
        try:
            self.execution_metrics = ExecutionMetrics()
            
            # Clear old fills (keep last 1000)
            if len(self.execution_fills) > 1000:
                self.execution_fills = self.execution_fills[-1000:]
            
            # Clear old completed orders (keep last 100)
            if len(self.completed_orders) > 100:
                order_ids = list(self.completed_orders.keys())[-100:]
                self.completed_orders = {k: v for k, v in self.completed_orders.items() if k in order_ids}
            
            # Clear alerts
            self.execution_alerts = []
            
            logger.info("Daily execution metrics reset")
            
        except Exception as e:
            logger.error(f"Error resetting daily metrics: {e}")