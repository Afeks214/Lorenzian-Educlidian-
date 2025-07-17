"""
Execution Strategies Testing Framework

This module provides comprehensive testing for execution strategies including
TWAP, VWAP, POV algorithms, market impact minimization, and execution optimization.

Key Features:
- TWAP (Time-Weighted Average Price) execution testing
- VWAP (Volume-Weighted Average Price) execution testing
- POV (Percentage of Volume) execution testing
- Market impact minimization strategies
- Dark pool and lit market execution optimization
- Slippage and transaction cost analysis
- Execution quality metrics (Implementation Shortfall, VWAP tracking)
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from unittest.mock import Mock, patch
import asyncio
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Types of execution strategies"""
    TWAP = "twap"
    VWAP = "vwap"
    POV = "pov"
    IS = "implementation_shortfall"
    MARKET = "market"
    LIMIT = "limit"
    ICEBERG = "iceberg"
    DARK_POOL = "dark_pool"


class OrderType(Enum):
    """Order types"""
    BUY = "buy"
    SELL = "sell"


class VenueType(Enum):
    """Venue types"""
    LIT = "lit"
    DARK = "dark"
    CROSSING = "crossing"


@dataclass
class MarketData:
    """Real-time market data"""
    timestamp: datetime
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    vwap: float
    spread: float
    market_impact: float


@dataclass
class ExecutionOrder:
    """Order for execution"""
    order_id: str
    symbol: str
    order_type: OrderType
    quantity: float
    strategy: ExecutionStrategy
    start_time: datetime
    end_time: datetime
    price_limit: Optional[float] = None
    strategy_params: Optional[Dict[str, Any]] = None


@dataclass
class ExecutionFill:
    """Execution fill details"""
    fill_id: str
    order_id: str
    timestamp: datetime
    quantity: float
    price: float
    venue: VenueType
    fees: float
    market_impact: float


@dataclass
class ExecutionResult:
    """Execution result metrics"""
    order_id: str
    total_filled: float
    avg_fill_price: float
    implementation_shortfall: float
    vwap_tracking_error: float
    market_impact: float
    total_fees: float
    fill_rate: float
    time_to_complete: float
    venue_breakdown: Dict[VenueType, float]
    slippage: float
    execution_quality_score: float


class ExecutionEngine:
    """
    Execution engine for testing various execution strategies.
    """
    
    def __init__(self):
        self.orders: Dict[str, ExecutionOrder] = {}
        self.fills: List[ExecutionFill] = []
        self.market_data: Dict[str, MarketData] = {}
        self.execution_results: Dict[str, ExecutionResult] = {}
        
        # Market simulation parameters
        self.market_impact_model = MarketImpactModel()
        self.slippage_model = SlippageModel()
        self.venue_manager = VenueManager()
        
    def submit_order(self, order: ExecutionOrder) -> str:
        """Submit order for execution"""
        self.orders[order.order_id] = order
        return order.order_id
    
    def execute_twap(
        self,
        order: ExecutionOrder,
        market_data: pd.DataFrame,
        num_slices: int = 10
    ) -> ExecutionResult:
        """
        Execute order using TWAP (Time-Weighted Average Price) strategy.
        
        Args:
            order: Order to execute
            market_data: Historical market data
            num_slices: Number of time slices
            
        Returns:
            ExecutionResult with execution metrics
        """
        
        # Calculate time slices
        total_duration = (order.end_time - order.start_time).total_seconds()
        slice_duration = total_duration / num_slices
        slice_quantity = order.quantity / num_slices
        
        fills = []
        current_time = order.start_time
        
        for i in range(num_slices):
            # Find market data for current time slice
            slice_market_data = self._get_market_data_at_time(market_data, current_time)
            
            if slice_market_data is None:
                current_time += timedelta(seconds=slice_duration)
                continue
            
            # Calculate execution price with market impact
            execution_price = self._calculate_execution_price(
                slice_market_data,
                slice_quantity,
                order.order_type,
                ExecutionStrategy.TWAP
            )
            
            # Create fill
            fill = ExecutionFill(
                fill_id=f"{order.order_id}_fill_{i}",
                order_id=order.order_id,
                timestamp=current_time,
                quantity=slice_quantity,
                price=execution_price,
                venue=VenueType.LIT,
                fees=self._calculate_fees(slice_quantity, execution_price),
                market_impact=self.market_impact_model.calculate_impact(
                    slice_quantity, slice_market_data['volume']
                )
            )
            
            fills.append(fill)
            current_time += timedelta(seconds=slice_duration)
        
        # Calculate execution metrics
        result = self._calculate_execution_metrics(order, fills, market_data)
        
        self.fills.extend(fills)
        self.execution_results[order.order_id] = result
        
        return result
    
    def execute_vwap(
        self,
        order: ExecutionOrder,
        market_data: pd.DataFrame,
        participation_rate: float = 0.1
    ) -> ExecutionResult:
        """
        Execute order using VWAP (Volume-Weighted Average Price) strategy.
        
        Args:
            order: Order to execute
            market_data: Historical market data
            participation_rate: Target participation rate (0-1)
            
        Returns:
            ExecutionResult with execution metrics
        """
        
        # Calculate volume profile
        volume_profile = self._calculate_volume_profile(market_data, order.start_time, order.end_time)
        
        fills = []
        remaining_quantity = order.quantity
        
        for timestamp, expected_volume in volume_profile.items():
            if remaining_quantity <= 0:
                break
            
            # Calculate slice quantity based on volume profile
            slice_quantity = min(
                remaining_quantity,
                expected_volume * participation_rate
            )
            
            if slice_quantity <= 0:
                continue
            
            # Find market data for current time
            slice_market_data = self._get_market_data_at_time(market_data, timestamp)
            
            if slice_market_data is None:
                continue
            
            # Calculate execution price
            execution_price = self._calculate_execution_price(
                slice_market_data,
                slice_quantity,
                order.order_type,
                ExecutionStrategy.VWAP
            )
            
            # Create fill
            fill = ExecutionFill(
                fill_id=f"{order.order_id}_vwap_{len(fills)}",
                order_id=order.order_id,
                timestamp=timestamp,
                quantity=slice_quantity,
                price=execution_price,
                venue=VenueType.LIT,
                fees=self._calculate_fees(slice_quantity, execution_price),
                market_impact=self.market_impact_model.calculate_impact(
                    slice_quantity, slice_market_data['volume']
                )
            )
            
            fills.append(fill)
            remaining_quantity -= slice_quantity
        
        # Calculate execution metrics
        result = self._calculate_execution_metrics(order, fills, market_data)
        
        self.fills.extend(fills)
        self.execution_results[order.order_id] = result
        
        return result
    
    def execute_pov(
        self,
        order: ExecutionOrder,
        market_data: pd.DataFrame,
        target_participation: float = 0.15
    ) -> ExecutionResult:
        """
        Execute order using POV (Percentage of Volume) strategy.
        
        Args:
            order: Order to execute
            market_data: Historical market data
            target_participation: Target participation rate (0-1)
            
        Returns:
            ExecutionResult with execution metrics
        """
        
        fills = []
        remaining_quantity = order.quantity
        
        # Filter market data for execution period
        execution_data = market_data[
            (market_data.index >= order.start_time) &
            (market_data.index <= order.end_time)
        ]
        
        for timestamp, row in execution_data.iterrows():
            if remaining_quantity <= 0:
                break
            
            # Calculate slice quantity based on market volume
            market_volume = row.get('volume', 1000)
            slice_quantity = min(
                remaining_quantity,
                market_volume * target_participation
            )
            
            if slice_quantity <= 0:
                continue
            
            # Calculate execution price
            execution_price = self._calculate_execution_price(
                row,
                slice_quantity,
                order.order_type,
                ExecutionStrategy.POV
            )
            
            # Create fill
            fill = ExecutionFill(
                fill_id=f"{order.order_id}_pov_{len(fills)}",
                order_id=order.order_id,
                timestamp=timestamp,
                quantity=slice_quantity,
                price=execution_price,
                venue=VenueType.LIT,
                fees=self._calculate_fees(slice_quantity, execution_price),
                market_impact=self.market_impact_model.calculate_impact(
                    slice_quantity, market_volume
                )
            )
            
            fills.append(fill)
            remaining_quantity -= slice_quantity
        
        # Calculate execution metrics
        result = self._calculate_execution_metrics(order, fills, market_data)
        
        self.fills.extend(fills)
        self.execution_results[order.order_id] = result
        
        return result
    
    def execute_implementation_shortfall(
        self,
        order: ExecutionOrder,
        market_data: pd.DataFrame,
        risk_aversion: float = 0.5
    ) -> ExecutionResult:
        """
        Execute order using Implementation Shortfall strategy.
        
        Args:
            order: Order to execute
            market_data: Historical market data
            risk_aversion: Risk aversion parameter (0-1)
            
        Returns:
            ExecutionResult with execution metrics
        """
        
        # Calculate optimal execution schedule
        execution_schedule = self._calculate_is_schedule(
            order, market_data, risk_aversion
        )
        
        fills = []
        
        for timestamp, slice_quantity in execution_schedule.items():
            if slice_quantity <= 0:
                continue
            
            # Find market data for current time
            slice_market_data = self._get_market_data_at_time(market_data, timestamp)
            
            if slice_market_data is None:
                continue
            
            # Calculate execution price
            execution_price = self._calculate_execution_price(
                slice_market_data,
                slice_quantity,
                order.order_type,
                ExecutionStrategy.IS
            )
            
            # Create fill
            fill = ExecutionFill(
                fill_id=f"{order.order_id}_is_{len(fills)}",
                order_id=order.order_id,
                timestamp=timestamp,
                quantity=slice_quantity,
                price=execution_price,
                venue=VenueType.LIT,
                fees=self._calculate_fees(slice_quantity, execution_price),
                market_impact=self.market_impact_model.calculate_impact(
                    slice_quantity, slice_market_data['volume']
                )
            )
            
            fills.append(fill)
        
        # Calculate execution metrics
        result = self._calculate_execution_metrics(order, fills, market_data)
        
        self.fills.extend(fills)
        self.execution_results[order.order_id] = result
        
        return result
    
    def execute_iceberg(
        self,
        order: ExecutionOrder,
        market_data: pd.DataFrame,
        display_size: float = 1000,
        refresh_threshold: float = 0.5
    ) -> ExecutionResult:
        """
        Execute order using Iceberg strategy.
        
        Args:
            order: Order to execute
            market_data: Historical market data
            display_size: Size of each iceberg slice
            refresh_threshold: Threshold for refreshing display
            
        Returns:
            ExecutionResult with execution metrics
        """
        
        fills = []
        remaining_quantity = order.quantity
        
        while remaining_quantity > 0:
            # Calculate current display size
            current_display = min(display_size, remaining_quantity)
            
            # Find current market data
            current_time = order.start_time
            slice_market_data = self._get_market_data_at_time(market_data, current_time)
            
            if slice_market_data is None:
                break
            
            # Calculate execution price
            execution_price = self._calculate_execution_price(
                slice_market_data,
                current_display,
                order.order_type,
                ExecutionStrategy.ICEBERG
            )
            
            # Create fill
            fill = ExecutionFill(
                fill_id=f"{order.order_id}_iceberg_{len(fills)}",
                order_id=order.order_id,
                timestamp=current_time,
                quantity=current_display,
                price=execution_price,
                venue=VenueType.LIT,
                fees=self._calculate_fees(current_display, execution_price),
                market_impact=self.market_impact_model.calculate_impact(
                    current_display, slice_market_data['volume']
                )
            )
            
            fills.append(fill)
            remaining_quantity -= current_display
            
            # Update start time for next slice
            order.start_time += timedelta(seconds=30)  # 30 second intervals
        
        # Calculate execution metrics
        result = self._calculate_execution_metrics(order, fills, market_data)
        
        self.fills.extend(fills)
        self.execution_results[order.order_id] = result
        
        return result
    
    def execute_dark_pool(
        self,
        order: ExecutionOrder,
        market_data: pd.DataFrame,
        dark_pool_probability: float = 0.3
    ) -> ExecutionResult:
        """
        Execute order using Dark Pool strategy.
        
        Args:
            order: Order to execute
            market_data: Historical market data
            dark_pool_probability: Probability of dark pool execution
            
        Returns:
            ExecutionResult with execution metrics
        """
        
        fills = []
        remaining_quantity = order.quantity
        
        # Execute over time with dark pool/lit market routing
        execution_data = market_data[
            (market_data.index >= order.start_time) &
            (market_data.index <= order.end_time)
        ]
        
        for timestamp, row in execution_data.iterrows():
            if remaining_quantity <= 0:
                break
            
            # Determine venue (dark pool vs lit market)
            is_dark_pool = np.random.random() < dark_pool_probability
            venue = VenueType.DARK if is_dark_pool else VenueType.LIT
            
            # Calculate slice quantity
            slice_quantity = min(remaining_quantity, 500)  # Smaller slices for dark pool
            
            # Calculate execution price
            execution_price = self._calculate_dark_pool_price(
                row, slice_quantity, order.order_type, venue
            )
            
            # Create fill
            fill = ExecutionFill(
                fill_id=f"{order.order_id}_dark_{len(fills)}",
                order_id=order.order_id,
                timestamp=timestamp,
                quantity=slice_quantity,
                price=execution_price,
                venue=venue,
                fees=self._calculate_fees(slice_quantity, execution_price),
                market_impact=self.market_impact_model.calculate_impact(
                    slice_quantity, row.get('volume', 1000)
                ) * (0.5 if venue == VenueType.DARK else 1.0)  # Lower impact in dark pools
            )
            
            fills.append(fill)
            remaining_quantity -= slice_quantity
        
        # Calculate execution metrics
        result = self._calculate_execution_metrics(order, fills, market_data)
        
        self.fills.extend(fills)
        self.execution_results[order.order_id] = result
        
        return result
    
    def _get_market_data_at_time(self, market_data: pd.DataFrame, timestamp: datetime) -> Optional[pd.Series]:
        """Get market data at specific timestamp"""
        
        # Find closest timestamp
        if timestamp in market_data.index:
            return market_data.loc[timestamp]
        
        # Find nearest timestamp
        time_diff = np.abs(market_data.index - timestamp)
        nearest_idx = time_diff.argmin()
        
        if time_diff.iloc[nearest_idx] < timedelta(minutes=5):  # Within 5 minutes
            return market_data.iloc[nearest_idx]
        
        return None
    
    def _calculate_execution_price(
        self,
        market_data: pd.Series,
        quantity: float,
        order_type: OrderType,
        strategy: ExecutionStrategy
    ) -> float:
        """Calculate execution price with market impact"""
        
        # Base price
        if order_type == OrderType.BUY:
            base_price = market_data.get('ask', market_data.get('price', 100.0))
        else:
            base_price = market_data.get('bid', market_data.get('price', 100.0))
        
        # Calculate market impact
        market_volume = market_data.get('volume', 1000)
        market_impact = self.market_impact_model.calculate_impact(quantity, market_volume)
        
        # Apply strategy-specific adjustments
        if strategy == ExecutionStrategy.TWAP:
            # TWAP tries to minimize market impact
            market_impact *= 0.8
        elif strategy == ExecutionStrategy.VWAP:
            # VWAP follows market volume
            market_impact *= 0.9
        elif strategy == ExecutionStrategy.POV:
            # POV may have higher impact
            market_impact *= 1.1
        
        # Apply market impact
        if order_type == OrderType.BUY:
            execution_price = base_price * (1 + market_impact)
        else:
            execution_price = base_price * (1 - market_impact)
        
        return execution_price
    
    def _calculate_dark_pool_price(
        self,
        market_data: pd.Series,
        quantity: float,
        order_type: OrderType,
        venue: VenueType
    ) -> float:
        """Calculate execution price for dark pool"""
        
        # Mid-point price for dark pools
        bid = market_data.get('bid', market_data.get('price', 100.0))
        ask = market_data.get('ask', market_data.get('price', 100.0))
        mid_price = (bid + ask) / 2
        
        if venue == VenueType.DARK:
            # Dark pools typically execute at mid-point
            return mid_price
        else:
            # Lit market execution
            return self._calculate_execution_price(
                market_data, quantity, order_type, ExecutionStrategy.MARKET
            )
    
    def _calculate_fees(self, quantity: float, price: float) -> float:
        """Calculate execution fees"""
        
        # Simplified fee model
        value = quantity * price
        fee_rate = 0.0005  # 5 basis points
        return value * fee_rate
    
    def _calculate_volume_profile(
        self,
        market_data: pd.DataFrame,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[datetime, float]:
        """Calculate expected volume profile"""
        
        # Filter data for execution period
        execution_data = market_data[
            (market_data.index >= start_time) &
            (market_data.index <= end_time)
        ]
        
        if execution_data.empty:
            return {}
        
        # Calculate volume profile
        volume_profile = {}
        for timestamp, row in execution_data.iterrows():
            volume_profile[timestamp] = row.get('volume', 1000)
        
        return volume_profile
    
    def _calculate_is_schedule(
        self,
        order: ExecutionOrder,
        market_data: pd.DataFrame,
        risk_aversion: float
    ) -> Dict[datetime, float]:
        """Calculate Implementation Shortfall execution schedule"""
        
        # Simplified IS schedule calculation
        execution_data = market_data[
            (market_data.index >= order.start_time) &
            (market_data.index <= order.end_time)
        ]
        
        if execution_data.empty:
            return {}
        
        # Calculate optimal execution rate
        total_periods = len(execution_data)
        base_rate = order.quantity / total_periods
        
        schedule = {}
        for i, (timestamp, row) in enumerate(execution_data.iterrows()):
            # Higher execution rate early (front-loaded)
            time_weight = (total_periods - i) / total_periods
            execution_rate = base_rate * (1 + risk_aversion * time_weight)
            schedule[timestamp] = execution_rate
        
        return schedule
    
    def _calculate_execution_metrics(
        self,
        order: ExecutionOrder,
        fills: List[ExecutionFill],
        market_data: pd.DataFrame
    ) -> ExecutionResult:
        """Calculate comprehensive execution metrics"""
        
        if not fills:
            return ExecutionResult(
                order_id=order.order_id,
                total_filled=0,
                avg_fill_price=0,
                implementation_shortfall=0,
                vwap_tracking_error=0,
                market_impact=0,
                total_fees=0,
                fill_rate=0,
                time_to_complete=0,
                venue_breakdown={},
                slippage=0,
                execution_quality_score=0
            )
        
        # Basic metrics
        total_filled = sum(fill.quantity for fill in fills)
        total_value = sum(fill.quantity * fill.price for fill in fills)
        avg_fill_price = total_value / total_filled if total_filled > 0 else 0
        
        # Calculate benchmark prices
        arrival_price = self._get_arrival_price(order, market_data)
        market_vwap = self._get_market_vwap(order, market_data)
        
        # Implementation Shortfall
        if order.order_type == OrderType.BUY:
            implementation_shortfall = (avg_fill_price - arrival_price) / arrival_price
        else:
            implementation_shortfall = (arrival_price - avg_fill_price) / arrival_price
        
        # VWAP tracking error
        vwap_tracking_error = abs(avg_fill_price - market_vwap) / market_vwap if market_vwap > 0 else 0
        
        # Market impact
        total_market_impact = sum(fill.market_impact for fill in fills)
        market_impact = total_market_impact / len(fills) if fills else 0
        
        # Fees
        total_fees = sum(fill.fees for fill in fills)
        
        # Fill rate
        fill_rate = total_filled / order.quantity if order.quantity > 0 else 0
        
        # Time to complete
        if fills:
            time_to_complete = (fills[-1].timestamp - order.start_time).total_seconds()
        else:
            time_to_complete = 0
        
        # Venue breakdown
        venue_breakdown = {}
        for venue in VenueType:
            venue_fills = [f for f in fills if f.venue == venue]
            if venue_fills:
                venue_breakdown[venue] = sum(f.quantity for f in venue_fills)
        
        # Slippage
        slippage = implementation_shortfall  # Simplified
        
        # Execution quality score (0-100)
        execution_quality_score = self._calculate_execution_quality_score(
            implementation_shortfall, vwap_tracking_error, market_impact, fill_rate
        )
        
        return ExecutionResult(
            order_id=order.order_id,
            total_filled=total_filled,
            avg_fill_price=avg_fill_price,
            implementation_shortfall=implementation_shortfall,
            vwap_tracking_error=vwap_tracking_error,
            market_impact=market_impact,
            total_fees=total_fees,
            fill_rate=fill_rate,
            time_to_complete=time_to_complete,
            venue_breakdown=venue_breakdown,
            slippage=slippage,
            execution_quality_score=execution_quality_score
        )
    
    def _get_arrival_price(self, order: ExecutionOrder, market_data: pd.DataFrame) -> float:
        """Get arrival price (price at order submission)"""
        
        # Find price at order start time
        arrival_data = self._get_market_data_at_time(market_data, order.start_time)
        
        if arrival_data is not None:
            return arrival_data.get('price', 100.0)
        
        return 100.0  # Default price
    
    def _get_market_vwap(self, order: ExecutionOrder, market_data: pd.DataFrame) -> float:
        """Get market VWAP during execution period"""
        
        execution_data = market_data[
            (market_data.index >= order.start_time) &
            (market_data.index <= order.end_time)
        ]
        
        if execution_data.empty:
            return 100.0
        
        # Calculate VWAP
        if 'volume' in execution_data.columns and 'price' in execution_data.columns:
            total_value = (execution_data['price'] * execution_data['volume']).sum()
            total_volume = execution_data['volume'].sum()
            return total_value / total_volume if total_volume > 0 else 100.0
        
        return execution_data['price'].mean() if 'price' in execution_data.columns else 100.0
    
    def _calculate_execution_quality_score(
        self,
        implementation_shortfall: float,
        vwap_tracking_error: float,
        market_impact: float,
        fill_rate: float
    ) -> float:
        """Calculate execution quality score (0-100)"""
        
        # Normalize metrics (lower is better for most metrics)
        is_score = max(0, 100 - abs(implementation_shortfall) * 10000)
        vwap_score = max(0, 100 - vwap_tracking_error * 10000)
        impact_score = max(0, 100 - market_impact * 10000)
        fill_score = fill_rate * 100
        
        # Weighted combination
        quality_score = (
            0.3 * is_score +
            0.25 * vwap_score +
            0.25 * impact_score +
            0.2 * fill_score
        )
        
        return min(100, max(0, quality_score))


class MarketImpactModel:
    """Model for calculating market impact"""
    
    def __init__(self):
        self.impact_coefficient = 0.01  # 1% impact per 1% of volume
        self.temporary_impact_decay = 0.5  # 50% decay rate
    
    def calculate_impact(self, quantity: float, market_volume: float) -> float:
        """Calculate market impact"""
        
        if market_volume <= 0:
            return 0.01  # Default 1% impact
        
        # Calculate participation rate
        participation_rate = quantity / market_volume
        
        # Square root model
        impact = self.impact_coefficient * np.sqrt(participation_rate)
        
        return min(0.1, max(0, impact))  # Cap at 10%
    
    def calculate_temporary_impact(self, permanent_impact: float) -> float:
        """Calculate temporary impact component"""
        return permanent_impact * self.temporary_impact_decay


class SlippageModel:
    """Model for calculating slippage"""
    
    def __init__(self):
        self.base_slippage = 0.0005  # 5 basis points
        self.volatility_factor = 0.1
    
    def calculate_slippage(self, quantity: float, volatility: float) -> float:
        """Calculate slippage"""
        
        # Size-dependent slippage
        size_factor = np.sqrt(quantity / 1000)  # Normalized to 1000 shares
        
        # Volatility-dependent slippage
        vol_factor = volatility * self.volatility_factor
        
        slippage = self.base_slippage * size_factor * (1 + vol_factor)
        
        return min(0.01, max(0, slippage))  # Cap at 1%


class VenueManager:
    """Manager for different execution venues"""
    
    def __init__(self):
        self.venues = {
            VenueType.LIT: {
                'fee_rate': 0.0005,
                'rebate_rate': 0.0003,
                'market_impact_factor': 1.0
            },
            VenueType.DARK: {
                'fee_rate': 0.0003,
                'rebate_rate': 0.0,
                'market_impact_factor': 0.5
            },
            VenueType.CROSSING: {
                'fee_rate': 0.0001,
                'rebate_rate': 0.0,
                'market_impact_factor': 0.3
            }
        }
    
    def get_venue_characteristics(self, venue: VenueType) -> Dict[str, float]:
        """Get venue characteristics"""
        return self.venues.get(venue, self.venues[VenueType.LIT])
    
    def select_optimal_venue(
        self,
        order_size: float,
        market_conditions: Dict[str, float]
    ) -> VenueType:
        """Select optimal venue based on order size and market conditions"""
        
        # Simplified venue selection logic
        if order_size < 1000:
            return VenueType.LIT
        elif order_size < 5000:
            return VenueType.DARK
        else:
            return VenueType.CROSSING


# Test fixtures
@pytest.fixture
def sample_execution_data():
    """Generate sample execution market data"""
    
    np.random.seed(42)
    dates = pd.date_range('2023-01-01 09:30:00', '2023-01-01 16:00:00', freq='1min')
    
    # Generate realistic intraday data
    base_price = 100.0
    prices = []
    volumes = []
    
    for i, date in enumerate(dates):
        # Add intraday patterns
        hour = date.hour
        minute = date.minute
        
        # U-shaped volume pattern
        volume_factor = 1.0
        if hour < 11 or hour > 15:
            volume_factor = 2.0
        elif 11 <= hour <= 15:
            volume_factor = 0.5
        
        # Random walk with mean reversion
        price_change = np.random.normal(0, 0.001)
        base_price += price_change
        
        prices.append(base_price)
        volumes.append(np.random.lognormal(8, 1) * volume_factor)
    
    data = pd.DataFrame({
        'price': prices,
        'volume': volumes,
        'bid': np.array(prices) * 0.999,
        'ask': np.array(prices) * 1.001,
        'spread': np.array(prices) * 0.002
    }, index=dates)
    
    return data


@pytest.fixture
def execution_engine():
    """Create execution engine instance"""
    return ExecutionEngine()


@pytest.fixture
def sample_order():
    """Create sample execution order"""
    return ExecutionOrder(
        order_id="TEST_ORDER_001",
        symbol="AAPL",
        order_type=OrderType.BUY,
        quantity=10000,
        strategy=ExecutionStrategy.TWAP,
        start_time=datetime(2023, 1, 1, 9, 30, 0),
        end_time=datetime(2023, 1, 1, 16, 0, 0),
        strategy_params={}
    )


# Comprehensive test suite
@pytest.mark.asyncio
class TestExecutionStrategies:
    """Comprehensive execution strategies tests"""
    
    def test_twap_execution(self, execution_engine, sample_order, sample_execution_data):
        """Test TWAP execution strategy"""
        
        result = execution_engine.execute_twap(
            sample_order,
            sample_execution_data,
            num_slices=10
        )
        
        assert result.order_id == sample_order.order_id
        assert result.total_filled > 0
        assert result.avg_fill_price > 0
        assert result.fill_rate > 0
        assert result.time_to_complete > 0
        assert result.execution_quality_score >= 0
        
        # TWAP should have relatively low market impact
        assert result.market_impact < 0.05  # Less than 5%
    
    def test_vwap_execution(self, execution_engine, sample_order, sample_execution_data):
        """Test VWAP execution strategy"""
        
        result = execution_engine.execute_vwap(
            sample_order,
            sample_execution_data,
            participation_rate=0.1
        )
        
        assert result.order_id == sample_order.order_id
        assert result.total_filled > 0
        assert result.avg_fill_price > 0
        assert result.fill_rate > 0
        
        # VWAP should have good tracking error
        assert result.vwap_tracking_error < 0.01  # Less than 1%
    
    def test_pov_execution(self, execution_engine, sample_order, sample_execution_data):
        """Test POV execution strategy"""
        
        result = execution_engine.execute_pov(
            sample_order,
            sample_execution_data,
            target_participation=0.15
        )
        
        assert result.order_id == sample_order.order_id
        assert result.total_filled > 0
        assert result.avg_fill_price > 0
        assert result.fill_rate > 0
        
        # POV should maintain consistent participation
        assert result.execution_quality_score > 50  # Reasonable quality
    
    def test_implementation_shortfall_execution(self, execution_engine, sample_order, sample_execution_data):
        """Test Implementation Shortfall execution strategy"""
        
        result = execution_engine.execute_implementation_shortfall(
            sample_order,
            sample_execution_data,
            risk_aversion=0.5
        )
        
        assert result.order_id == sample_order.order_id
        assert result.total_filled > 0
        assert result.avg_fill_price > 0
        
        # IS should optimize implementation shortfall
        assert abs(result.implementation_shortfall) < 0.02  # Less than 2%
    
    def test_iceberg_execution(self, execution_engine, sample_order, sample_execution_data):
        """Test Iceberg execution strategy"""
        
        result = execution_engine.execute_iceberg(
            sample_order,
            sample_execution_data,
            display_size=1000
        )
        
        assert result.order_id == sample_order.order_id
        assert result.total_filled > 0
        assert result.avg_fill_price > 0
        
        # Iceberg should have lower market impact
        assert result.market_impact < 0.03  # Less than 3%
    
    def test_dark_pool_execution(self, execution_engine, sample_order, sample_execution_data):
        """Test Dark Pool execution strategy"""
        
        result = execution_engine.execute_dark_pool(
            sample_order,
            sample_execution_data,
            dark_pool_probability=0.3
        )
        
        assert result.order_id == sample_order.order_id
        assert result.total_filled > 0
        assert result.avg_fill_price > 0
        
        # Should have venue breakdown
        assert len(result.venue_breakdown) > 0
        
        # Dark pool should have lower fees
        assert result.total_fees < sample_order.quantity * 100 * 0.001  # Less than 10 bps
    
    def test_market_impact_model(self):
        """Test market impact model"""
        
        impact_model = MarketImpactModel()
        
        # Test different order sizes
        small_impact = impact_model.calculate_impact(100, 10000)
        large_impact = impact_model.calculate_impact(1000, 10000)
        
        assert small_impact >= 0
        assert large_impact >= 0
        assert large_impact > small_impact
        
        # Test temporary impact
        temporary = impact_model.calculate_temporary_impact(0.01)
        assert 0 <= temporary <= 0.01
    
    def test_slippage_model(self):
        """Test slippage model"""
        
        slippage_model = SlippageModel()
        
        # Test different conditions
        low_vol_slippage = slippage_model.calculate_slippage(1000, 0.01)
        high_vol_slippage = slippage_model.calculate_slippage(1000, 0.05)
        
        assert low_vol_slippage >= 0
        assert high_vol_slippage >= 0
        assert high_vol_slippage > low_vol_slippage
    
    def test_venue_manager(self):
        """Test venue manager"""
        
        venue_manager = VenueManager()
        
        # Test venue characteristics
        lit_chars = venue_manager.get_venue_characteristics(VenueType.LIT)
        dark_chars = venue_manager.get_venue_characteristics(VenueType.DARK)
        
        assert 'fee_rate' in lit_chars
        assert 'market_impact_factor' in lit_chars
        assert dark_chars['market_impact_factor'] < lit_chars['market_impact_factor']
        
        # Test venue selection
        small_venue = venue_manager.select_optimal_venue(500, {})
        large_venue = venue_manager.select_optimal_venue(10000, {})
        
        assert isinstance(small_venue, VenueType)
        assert isinstance(large_venue, VenueType)
    
    def test_execution_quality_metrics(self, execution_engine, sample_order, sample_execution_data):
        """Test execution quality metrics calculation"""
        
        result = execution_engine.execute_twap(sample_order, sample_execution_data)
        
        # Test metric ranges
        assert 0 <= result.execution_quality_score <= 100
        assert result.fill_rate <= 1.0
        assert result.market_impact >= 0
        assert result.total_fees >= 0
        
        # Test implementation shortfall calculation
        assert abs(result.implementation_shortfall) < 0.1  # Should be reasonable
    
    def test_strategy_comparison(self, execution_engine, sample_execution_data):
        """Test comparison between different execution strategies"""
        
        # Create identical orders for different strategies
        strategies = [
            ExecutionStrategy.TWAP,
            ExecutionStrategy.VWAP,
            ExecutionStrategy.POV,
            ExecutionStrategy.IS
        ]
        
        results = {}
        
        for strategy in strategies:
            order = ExecutionOrder(
                order_id=f"ORDER_{strategy.value}",
                symbol="AAPL",
                order_type=OrderType.BUY,
                quantity=5000,
                strategy=strategy,
                start_time=datetime(2023, 1, 1, 10, 0, 0),
                end_time=datetime(2023, 1, 1, 15, 0, 0)
            )
            
            if strategy == ExecutionStrategy.TWAP:
                result = execution_engine.execute_twap(order, sample_execution_data)
            elif strategy == ExecutionStrategy.VWAP:
                result = execution_engine.execute_vwap(order, sample_execution_data)
            elif strategy == ExecutionStrategy.POV:
                result = execution_engine.execute_pov(order, sample_execution_data)
            elif strategy == ExecutionStrategy.IS:
                result = execution_engine.execute_implementation_shortfall(order, sample_execution_data)
            
            results[strategy] = result
        
        # Compare strategies
        for strategy, result in results.items():
            assert result.total_filled > 0
            assert result.execution_quality_score > 0
            assert result.fill_rate > 0
        
        # VWAP should have good tracking error
        assert results[ExecutionStrategy.VWAP].vwap_tracking_error < 0.02
        
        # IS should have good implementation shortfall
        assert abs(results[ExecutionStrategy.IS].implementation_shortfall) < 0.02
    
    def test_order_size_impact(self, execution_engine, sample_execution_data):
        """Test impact of order size on execution"""
        
        order_sizes = [1000, 5000, 10000, 50000]
        results = {}
        
        for size in order_sizes:
            order = ExecutionOrder(
                order_id=f"ORDER_{size}",
                symbol="AAPL",
                order_type=OrderType.BUY,
                quantity=size,
                strategy=ExecutionStrategy.TWAP,
                start_time=datetime(2023, 1, 1, 10, 0, 0),
                end_time=datetime(2023, 1, 1, 15, 0, 0)
            )
            
            result = execution_engine.execute_twap(order, sample_execution_data)
            results[size] = result
        
        # Larger orders should have higher market impact
        impact_values = [results[size].market_impact for size in order_sizes]
        
        # Impact should generally increase with size
        for i in range(1, len(impact_values)):
            assert impact_values[i] >= impact_values[i-1] * 0.8  # Allow some variation
    
    def test_execution_timing(self, execution_engine, sample_execution_data):
        """Test execution timing effects"""
        
        # Test different execution periods
        periods = [
            (datetime(2023, 1, 1, 9, 30, 0), datetime(2023, 1, 1, 10, 30, 0)),  # Open
            (datetime(2023, 1, 1, 12, 0, 0), datetime(2023, 1, 1, 13, 0, 0)),   # Lunch
            (datetime(2023, 1, 1, 15, 0, 0), datetime(2023, 1, 1, 16, 0, 0))    # Close
        ]
        
        results = {}
        
        for i, (start, end) in enumerate(periods):
            order = ExecutionOrder(
                order_id=f"ORDER_TIMING_{i}",
                symbol="AAPL",
                order_type=OrderType.BUY,
                quantity=5000,
                strategy=ExecutionStrategy.VWAP,
                start_time=start,
                end_time=end
            )
            
            result = execution_engine.execute_vwap(order, sample_execution_data)
            results[i] = result
        
        # All executions should be successful
        for result in results.values():
            assert result.total_filled > 0
            assert result.execution_quality_score > 0
    
    def test_market_conditions_adaptation(self, execution_engine):
        """Test adaptation to different market conditions"""
        
        # Create different market conditions
        conditions = [
            "low_volatility",
            "high_volatility",
            "low_volume",
            "high_volume"
        ]
        
        results = {}
        
        for condition in conditions:
            # Generate market data based on condition
            dates = pd.date_range('2023-01-01 09:30:00', '2023-01-01 16:00:00', freq='1min')
            
            if condition == "low_volatility":
                price_std = 0.0005
                volume_base = 1000
            elif condition == "high_volatility":
                price_std = 0.005
                volume_base = 1000
            elif condition == "low_volume":
                price_std = 0.001
                volume_base = 100
            else:  # high_volume
                price_std = 0.001
                volume_base = 5000
            
            # Generate data
            base_price = 100.0
            prices = []
            volumes = []
            
            for _ in dates:
                price_change = np.random.normal(0, price_std)
                base_price += price_change
                prices.append(base_price)
                volumes.append(np.random.lognormal(np.log(volume_base), 0.5))
            
            market_data = pd.DataFrame({
                'price': prices,
                'volume': volumes,
                'bid': np.array(prices) * 0.999,
                'ask': np.array(prices) * 1.001
            }, index=dates)
            
            # Execute order
            order = ExecutionOrder(
                order_id=f"ORDER_{condition}",
                symbol="AAPL",
                order_type=OrderType.BUY,
                quantity=5000,
                strategy=ExecutionStrategy.TWAP,
                start_time=dates[0],
                end_time=dates[-1]
            )
            
            result = execution_engine.execute_twap(order, market_data)
            results[condition] = result
        
        # High volatility should have higher impact
        assert results["high_volatility"].market_impact > results["low_volatility"].market_impact
        
        # Low volume should have higher impact
        assert results["low_volume"].market_impact > results["high_volume"].market_impact
    
    def test_buy_vs_sell_execution(self, execution_engine, sample_execution_data):
        """Test buy vs sell execution differences"""
        
        # Create buy and sell orders
        buy_order = ExecutionOrder(
            order_id="BUY_ORDER",
            symbol="AAPL",
            order_type=OrderType.BUY,
            quantity=5000,
            strategy=ExecutionStrategy.TWAP,
            start_time=datetime(2023, 1, 1, 10, 0, 0),
            end_time=datetime(2023, 1, 1, 15, 0, 0)
        )
        
        sell_order = ExecutionOrder(
            order_id="SELL_ORDER",
            symbol="AAPL",
            order_type=OrderType.SELL,
            quantity=5000,
            strategy=ExecutionStrategy.TWAP,
            start_time=datetime(2023, 1, 1, 10, 0, 0),
            end_time=datetime(2023, 1, 1, 15, 0, 0)
        )
        
        buy_result = execution_engine.execute_twap(buy_order, sample_execution_data)
        sell_result = execution_engine.execute_twap(sell_order, sample_execution_data)
        
        # Both should execute successfully
        assert buy_result.total_filled > 0
        assert sell_result.total_filled > 0
        
        # Buy should have higher average price than sell (due to spread)
        assert buy_result.avg_fill_price > sell_result.avg_fill_price
    
    def test_execution_stress_scenarios(self, execution_engine):
        """Test execution under stress scenarios"""
        
        # Create stressed market data (gap, high volatility)
        dates = pd.date_range('2023-01-01 09:30:00', '2023-01-01 16:00:00', freq='1min')
        
        # Create gap and high volatility
        base_price = 100.0
        prices = []
        volumes = []
        
        for i, date in enumerate(dates):
            if i == 100:  # Create gap
                base_price *= 1.05  # 5% gap up
            
            # High volatility
            price_change = np.random.normal(0, 0.01)
            base_price += price_change
            prices.append(base_price)
            volumes.append(np.random.lognormal(6, 1))  # Lower volume
        
        stressed_data = pd.DataFrame({
            'price': prices,
            'volume': volumes,
            'bid': np.array(prices) * 0.995,  # Wider spread
            'ask': np.array(prices) * 1.005
        }, index=dates)
        
        # Execute order in stressed conditions
        order = ExecutionOrder(
            order_id="STRESS_ORDER",
            symbol="AAPL",
            order_type=OrderType.BUY,
            quantity=5000,
            strategy=ExecutionStrategy.TWAP,
            start_time=dates[0],
            end_time=dates[-1]
        )
        
        result = execution_engine.execute_twap(order, stressed_data)
        
        # Should still execute but with higher impact
        assert result.total_filled > 0
        assert result.market_impact > 0.02  # Higher impact in stressed conditions
        assert result.execution_quality_score < 80  # Lower quality score
    
    def test_partial_fill_handling(self, execution_engine, sample_execution_data):
        """Test handling of partial fills"""
        
        # Create large order that may not fill completely
        large_order = ExecutionOrder(
            order_id="LARGE_ORDER",
            symbol="AAPL",
            order_type=OrderType.BUY,
            quantity=100000,  # Very large order
            strategy=ExecutionStrategy.POV,
            start_time=datetime(2023, 1, 1, 10, 0, 0),
            end_time=datetime(2023, 1, 1, 11, 0, 0),  # Short execution window
            strategy_params={'target_participation': 0.05}  # Low participation
        )
        
        result = execution_engine.execute_pov(large_order, sample_execution_data, target_participation=0.05)
        
        # Should have partial fill
        assert result.total_filled > 0
        assert result.fill_rate < 1.0  # Not completely filled
        assert result.execution_quality_score > 0
    
    def test_execution_analytics(self, execution_engine, sample_order, sample_execution_data):
        """Test execution analytics and reporting"""
        
        # Execute order
        result = execution_engine.execute_twap(sample_order, sample_execution_data)
        
        # Test analytics
        assert len(execution_engine.fills) > 0
        assert sample_order.order_id in execution_engine.execution_results
        
        # Check fill details
        for fill in execution_engine.fills:
            assert fill.order_id == sample_order.order_id
            assert fill.quantity > 0
            assert fill.price > 0
            assert fill.fees >= 0
            assert fill.market_impact >= 0
        
        # Check result completeness
        assert result.total_filled > 0
        assert result.avg_fill_price > 0
        assert result.execution_quality_score >= 0