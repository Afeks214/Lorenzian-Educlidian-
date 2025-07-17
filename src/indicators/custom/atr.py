"""
Average True Range (ATR) Indicator

Implements the Average True Range technical indicator for volatility measurement
and stop-loss/take-profit level calculation in the Stop/Target Agent system.

The ATR is a key component for dynamic stop-loss and take-profit level calculation,
providing volatility-adjusted position sizing and risk management.

Mathematical Formula:
True Range = max(High - Low, abs(High - Close_prev), abs(Low - Close_prev))
ATR = Simple Moving Average of True Range over N periods

Author: Agent 3 - Stop/Target Agent Developer
Version: 1.0
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import structlog

from ..base import BaseIndicator
from src.core.minimal_dependencies import EventBus, Event, EventType, BarData

logger = structlog.get_logger()


@dataclass
class ATRReading:
    """ATR indicator reading"""
    timestamp: datetime
    atr_value: float
    true_range: float
    volatility_percentile: float
    trend_strength: float


class ATRIndicator(BaseIndicator):
    """
    Average True Range (ATR) Indicator
    
    Calculates volatility-based metrics for dynamic stop-loss and take-profit
    level determination. Provides both current ATR values and volatility regime
    classification for the Stop/Target Agent.
    
    Features:
    - Standard ATR calculation with configurable period
    - Volatility percentile ranking
    - Trend strength measurement
    - Real-time ATR updates via event system
    - Historical ATR distribution analysis
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: EventBus):
        """
        Initialize ATR indicator
        
        Args:
            config: ATR configuration parameters
            event_bus: System event bus
        """
        super().__init__(config, event_bus)
        
        # ATR parameters
        self.period = config.get('period', 14)
        self.smoothing_factor = config.get('smoothing_factor', 2.0 / (self.period + 1))
        self.min_periods = config.get('min_periods', 2)
        
        # Volatility regime thresholds
        self.volatility_thresholds = {
            'low': config.get('vol_low_percentile', 25),
            'medium': config.get('vol_medium_percentile', 50),
            'high': config.get('vol_high_percentile', 75)
        }
        
        # Data storage
        self.price_data: List[BarData] = []
        self.true_ranges: List[float] = []
        self.atr_values: List[float] = []
        self.atr_readings: List[ATRReading] = []
        
        # Current state
        self.current_atr: Optional[float] = None
        self.current_volatility_percentile: Optional[float] = None
        self.ewma_atr: Optional[float] = None
        
        # Performance tracking
        self.calculations_performed = 0
        self.last_update_time: Optional[datetime] = None
        
        self.logger = logger.bind(indicator="ATR", period=self.period)
        self.logger.info("ATR indicator initialized")
    
    def calculate_true_range(self, current_bar: BarData, previous_close: float) -> float:
        """
        Calculate True Range for a single bar
        
        Args:
            current_bar: Current price bar
            previous_close: Previous bar's close price
            
        Returns:
            True Range value
        """
        tr1 = current_bar.high - current_bar.low
        tr2 = abs(current_bar.high - previous_close)
        tr3 = abs(current_bar.low - previous_close)
        
        return max(tr1, tr2, tr3)
    
    def calculate_atr(self, true_ranges: List[float]) -> float:
        """
        Calculate ATR from True Range values
        
        Args:
            true_ranges: List of True Range values
            
        Returns:
            Average True Range
        """
        if len(true_ranges) < self.min_periods:
            return 0.0
        
        if len(true_ranges) >= self.period:
            # Use simple moving average for full period
            return np.mean(true_ranges[-self.period:])
        else:
            # Use all available data for partial period
            return np.mean(true_ranges)
    
    def calculate_ewma_atr(self, new_true_range: float) -> float:
        """
        Calculate Exponentially Weighted Moving Average ATR
        
        Args:
            new_true_range: Latest True Range value
            
        Returns:
            EWMA ATR value
        """
        if self.ewma_atr is None:
            self.ewma_atr = new_true_range
        else:
            self.ewma_atr = (self.smoothing_factor * new_true_range + 
                           (1 - self.smoothing_factor) * self.ewma_atr)
        
        return self.ewma_atr
    
    def calculate_volatility_percentile(self, current_atr: float) -> float:
        """
        Calculate volatility percentile ranking
        
        Args:
            current_atr: Current ATR value
            
        Returns:
            Volatility percentile (0-100)
        """
        if len(self.atr_values) < 20:  # Need sufficient history
            return 50.0  # Neutral if insufficient data
        
        # Use last 100 ATR values for percentile calculation
        recent_atr_values = self.atr_values[-100:]
        
        if current_atr <= 0:
            return 0.0
        
        percentile = (np.searchsorted(sorted(recent_atr_values), current_atr) / 
                     len(recent_atr_values)) * 100
        
        return np.clip(percentile, 0.0, 100.0)
    
    def calculate_trend_strength(self, bars: List[BarData]) -> float:
        """
        Calculate trend strength based on price movement vs ATR
        
        Args:
            bars: Recent price bars
            
        Returns:
            Trend strength (0-1, higher = stronger trend)
        """
        if len(bars) < 3 or self.current_atr is None or self.current_atr <= 0:
            return 0.0
        
        # Calculate price change over last 3 bars
        price_change = abs(bars[-1].close - bars[-3].close)
        
        # Normalize by ATR to get trend strength
        trend_strength = min(price_change / (self.current_atr * 2), 1.0)
        
        return trend_strength
    
    def update(self, bar_data: BarData) -> Optional[ATRReading]:
        """
        Update ATR with new price data
        
        Args:
            bar_data: New price bar
            
        Returns:
            ATR reading if calculation successful
        """
        try:
            start_time = datetime.now()
            
            # Store price data
            self.price_data.append(bar_data)
            
            # Keep only necessary history (period * 3 for percentile calc)
            max_history = max(self.period * 3, 100)
            if len(self.price_data) > max_history:
                self.price_data = self.price_data[-max_history:]
            
            # Need at least 2 bars for True Range calculation
            if len(self.price_data) < 2:
                return None
            
            # Calculate True Range
            previous_close = self.price_data[-2].close
            true_range = self.calculate_true_range(bar_data, previous_close)
            self.true_ranges.append(true_range)
            
            # Keep True Range history
            if len(self.true_ranges) > max_history:
                self.true_ranges = self.true_ranges[-max_history:]
            
            # Calculate ATR
            atr_value = self.calculate_atr(self.true_ranges)
            if atr_value <= 0:
                return None
            
            self.current_atr = atr_value
            self.atr_values.append(atr_value)
            
            # Keep ATR history
            if len(self.atr_values) > max_history:
                self.atr_values = self.atr_values[-max_history:]
            
            # Calculate EWMA ATR for smoothed values
            ewma_atr = self.calculate_ewma_atr(true_range)
            
            # Calculate volatility percentile
            volatility_percentile = self.calculate_volatility_percentile(atr_value)
            self.current_volatility_percentile = volatility_percentile
            
            # Calculate trend strength
            trend_strength = self.calculate_trend_strength(self.price_data)
            
            # Create ATR reading
            atr_reading = ATRReading(
                timestamp=bar_data.timestamp,
                atr_value=atr_value,
                true_range=true_range,
                volatility_percentile=volatility_percentile,
                trend_strength=trend_strength
            )
            
            self.atr_readings.append(atr_reading)
            if len(self.atr_readings) > 1000:
                self.atr_readings = self.atr_readings[-1000:]
            
            # Performance tracking
            self.calculations_performed += 1
            self.last_update_time = datetime.now()
            calculation_time = (self.last_update_time - start_time).total_seconds() * 1000
            
            # Publish ATR update event
            self.event_bus.publish(
                self.event_bus.create_event(
                    EventType.INDICATOR_UPDATE,
                    {
                        'indicator': 'ATR',
                        'reading': atr_reading,
                        'atr_value': atr_value,
                        'ewma_atr': ewma_atr,
                        'volatility_percentile': volatility_percentile,
                        'trend_strength': trend_strength,
                        'calculation_time_ms': calculation_time
                    },
                    'ATRIndicator'
                )
            )
            
            if calculation_time > 5.0:  # Log slow calculations
                self.logger.warning("Slow ATR calculation",
                                  calculation_time_ms=calculation_time)
            
            return atr_reading
            
        except Exception as e:
            self.logger.error("Error updating ATR", error=str(e))
            return None
    
    def get_current_atr(self) -> Optional[float]:
        """Get current ATR value"""
        return self.current_atr
    
    def get_volatility_regime(self) -> str:
        """
        Get current volatility regime classification
        
        Returns:
            Volatility regime: 'low', 'medium', 'high', or 'extreme'
        """
        if self.current_volatility_percentile is None:
            return 'medium'
        
        percentile = self.current_volatility_percentile
        
        if percentile <= self.volatility_thresholds['low']:
            return 'low'
        elif percentile <= self.volatility_thresholds['medium']:
            return 'medium'
        elif percentile <= self.volatility_thresholds['high']:
            return 'high'
        else:
            return 'extreme'
    
    def get_stop_distance_recommendation(self, risk_multiplier: float = 1.5) -> Optional[float]:
        """
        Get recommended stop distance based on ATR
        
        Args:
            risk_multiplier: Risk multiplier for stop distance
            
        Returns:
            Recommended stop distance in price units
        """
        if self.current_atr is None:
            return None
        
        return self.current_atr * risk_multiplier
    
    def get_target_distance_recommendation(self, reward_multiplier: float = 2.0) -> Optional[float]:
        """
        Get recommended target distance based on ATR
        
        Args:
            reward_multiplier: Reward multiplier for target distance
            
        Returns:
            Recommended target distance in price units
        """
        if self.current_atr is None:
            return None
        
        return self.current_atr * reward_multiplier
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get ATR calculation statistics"""
        if not self.atr_values:
            return {}
        
        recent_atr = self.atr_values[-20:] if len(self.atr_values) >= 20 else self.atr_values
        
        return {
            'current_atr': self.current_atr,
            'ewma_atr': self.ewma_atr,
            'volatility_percentile': self.current_volatility_percentile,
            'volatility_regime': self.get_volatility_regime(),
            'calculations_performed': self.calculations_performed,
            'last_update': self.last_update_time,
            'atr_statistics': {
                'mean': np.mean(recent_atr),
                'std': np.std(recent_atr),
                'min': np.min(recent_atr),
                'max': np.max(recent_atr),
                'latest': recent_atr[-1] if recent_atr else None
            }
        }
    
    def reset(self) -> None:
        """Reset ATR indicator state"""
        self.price_data = []
        self.true_ranges = []
        self.atr_values = []
        self.atr_readings = []
        self.current_atr = None
        self.current_volatility_percentile = None
        self.ewma_atr = None
        self.calculations_performed = 0
        self.last_update_time = None
        
        self.logger.info("ATR indicator reset")
    
    def validate_inputs(self, bar_data: BarData) -> bool:
        """
        Validate input data
        
        Args:
            bar_data: Price bar to validate
            
        Returns:
            True if valid, False otherwise
        """
        if bar_data is None:
            return False
        
        if any(val is None or np.isnan(val) or np.isinf(val) 
               for val in [bar_data.high, bar_data.low, bar_data.close]):
            return False
        
        if bar_data.high < bar_data.low:
            return False
        
        if bar_data.close < 0:
            return False
        
        return True
    
    def __str__(self) -> str:
        """String representation"""
        return f"ATR(period={self.period}, current={self.current_atr:.6f})"
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return (f"ATRIndicator(period={self.period}, "
                f"calculations={self.calculations_performed}, "
                f"current_atr={self.current_atr}, "
                f"vol_regime={self.get_volatility_regime()})")