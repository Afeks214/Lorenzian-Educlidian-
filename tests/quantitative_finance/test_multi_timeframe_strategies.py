"""
Multi-Timeframe Strategies Testing Framework

This module provides comprehensive testing for multi-timeframe trading strategies,
including strategic (30m) and tactical (5m) signal integration, signal aggregation,
conflict resolution, and regime-dependent strategy selection.

Key Features:
- Strategic (30m) and tactical (5m) signal integration
- Signal aggregation and conflict resolution
- Regime-dependent strategy selection
- Timeframe hierarchy and priority management
- Cross-timeframe correlation analysis
- Dynamic timeframe switching
- Performance attribution by timeframe
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
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class Timeframe(Enum):
    """Trading timeframes"""
    MINUTE_1 = "1min"
    MINUTE_5 = "5min"
    MINUTE_15 = "15min"
    MINUTE_30 = "30min"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"


class SignalType(Enum):
    """Signal types"""
    ENTRY = "entry"
    EXIT = "exit"
    FILTER = "filter"
    CONFIRMATION = "confirmation"
    RISK_MANAGEMENT = "risk_management"


class SignalStrength(Enum):
    """Signal strength levels"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class ConflictResolution(Enum):
    """Conflict resolution methods"""
    HIGHER_TIMEFRAME = "higher_timeframe"
    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTE = "majority_vote"
    STRENGTH_BASED = "strength_based"
    RECENT_SIGNAL = "recent_signal"


@dataclass
class TimeframeSignal:
    """Signal from specific timeframe"""
    timeframe: Timeframe
    signal_type: SignalType
    direction: int  # -1, 0, 1 for sell, neutral, buy
    strength: SignalStrength
    confidence: float
    timestamp: datetime
    price: float
    volume: float
    indicators: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class AggregatedSignal:
    """Aggregated signal from multiple timeframes"""
    final_direction: int
    final_strength: SignalStrength
    final_confidence: float
    contributing_signals: List[TimeframeSignal]
    conflict_resolution_method: ConflictResolution
    timeframe_weights: Dict[Timeframe, float]
    timestamp: datetime
    reasoning: str


@dataclass
class StrategyPerformance:
    """Performance metrics by timeframe"""
    timeframe: Timeframe
    total_signals: int
    correct_signals: int
    accuracy: float
    profit_factor: float
    avg_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float


class TimeframeAnalyzer:
    """
    Analyzer for multi-timeframe signal relationships and correlations.
    """
    
    def __init__(self):
        self.timeframe_hierarchy = {
            Timeframe.MINUTE_1: 1,
            Timeframe.MINUTE_5: 5,
            Timeframe.MINUTE_15: 15,
            Timeframe.MINUTE_30: 30,
            Timeframe.HOUR_1: 60,
            Timeframe.HOUR_4: 240,
            Timeframe.DAILY: 1440
        }
        
    def analyze_timeframe_correlation(
        self,
        signals: Dict[Timeframe, List[TimeframeSignal]],
        window: int = 50
    ) -> pd.DataFrame:
        """
        Analyze correlation between different timeframes.
        
        Args:
            signals: Dictionary of signals by timeframe
            window: Rolling window for correlation calculation
            
        Returns:
            Correlation matrix between timeframes
        """
        
        # Create signal series for each timeframe
        timeframe_series = {}
        
        for timeframe, signal_list in signals.items():
            if not signal_list:
                continue
                
            # Create time series of signal directions
            signal_df = pd.DataFrame([
                {
                    'timestamp': signal.timestamp,
                    'direction': signal.direction,
                    'strength': self._strength_to_numeric(signal.strength),
                    'confidence': signal.confidence
                }
                for signal in signal_list
            ])
            
            if not signal_df.empty:
                signal_df.set_index('timestamp', inplace=True)
                # Combine direction and strength
                signal_df['signal_value'] = signal_df['direction'] * signal_df['strength'] * signal_df['confidence']
                timeframe_series[timeframe] = signal_df['signal_value']
        
        if len(timeframe_series) < 2:
            return pd.DataFrame()
        
        # Align all series to common timestamps
        combined_df = pd.DataFrame(timeframe_series)
        combined_df = combined_df.dropna()
        
        if len(combined_df) < window:
            return combined_df.corr()
        
        # Calculate rolling correlation
        rolling_corr = combined_df.rolling(window=window).corr()
        
        # Return average correlation matrix
        return rolling_corr.groupby(level=1).mean()
    
    def _strength_to_numeric(self, strength: SignalStrength) -> float:
        """Convert signal strength to numeric value"""
        strength_map = {
            SignalStrength.WEAK: 0.25,
            SignalStrength.MODERATE: 0.5,
            SignalStrength.STRONG: 0.75,
            SignalStrength.VERY_STRONG: 1.0
        }
        return strength_map.get(strength, 0.5)
    
    def analyze_timeframe_lead_lag(
        self,
        signals: Dict[Timeframe, List[TimeframeSignal]],
        max_lag: int = 10
    ) -> Dict[Tuple[Timeframe, Timeframe], int]:
        """
        Analyze lead-lag relationships between timeframes.
        
        Args:
            signals: Dictionary of signals by timeframe
            max_lag: Maximum lag to test
            
        Returns:
            Dictionary of optimal lag between timeframe pairs
        """
        
        timeframe_series = {}
        
        # Convert signals to time series
        for timeframe, signal_list in signals.items():
            if not signal_list:
                continue
                
            signal_df = pd.DataFrame([
                {
                    'timestamp': signal.timestamp,
                    'direction': signal.direction
                }
                for signal in signal_list
            ])
            
            if not signal_df.empty:
                signal_df.set_index('timestamp', inplace=True)
                timeframe_series[timeframe] = signal_df['direction']
        
        lead_lag_results = {}
        
        # Test all pairs of timeframes
        for tf1 in timeframe_series:
            for tf2 in timeframe_series:
                if tf1 == tf2:
                    continue
                
                series1 = timeframe_series[tf1]
                series2 = timeframe_series[tf2]
                
                # Align series
                aligned_df = pd.DataFrame({'tf1': series1, 'tf2': series2}).dropna()
                
                if len(aligned_df) < max_lag * 2:
                    continue
                
                # Calculate cross-correlation at different lags
                best_lag = 0
                best_corr = 0
                
                for lag in range(-max_lag, max_lag + 1):
                    if lag == 0:
                        corr = aligned_df['tf1'].corr(aligned_df['tf2'])
                    elif lag > 0:
                        # tf1 leads tf2
                        corr = aligned_df['tf1'].iloc[:-lag].corr(aligned_df['tf2'].iloc[lag:])
                    else:
                        # tf2 leads tf1
                        corr = aligned_df['tf1'].iloc[-lag:].corr(aligned_df['tf2'].iloc[:lag])
                    
                    if abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_lag = lag
                
                lead_lag_results[(tf1, tf2)] = best_lag
        
        return lead_lag_results
    
    def calculate_timeframe_efficiency(
        self,
        signals: Dict[Timeframe, List[TimeframeSignal]],
        returns: pd.Series,
        holding_period: int = 5
    ) -> Dict[Timeframe, float]:
        """
        Calculate efficiency of each timeframe for generating profitable signals.
        
        Args:
            signals: Dictionary of signals by timeframe
            returns: Price returns series
            holding_period: Holding period for signal evaluation
            
        Returns:
            Dictionary of efficiency scores by timeframe
        """
        
        efficiency_scores = {}
        
        for timeframe, signal_list in signals.items():
            if not signal_list:
                efficiency_scores[timeframe] = 0.0
                continue
            
            signal_returns = []
            
            for signal in signal_list:
                # Find returns after signal
                signal_time = signal.timestamp
                
                # Get returns for holding period
                future_returns = returns[returns.index >= signal_time][:holding_period]
                
                if len(future_returns) >= holding_period:
                    cumulative_return = (1 + future_returns).prod() - 1
                    
                    # Adjust return by signal direction
                    adjusted_return = cumulative_return * signal.direction
                    signal_returns.append(adjusted_return)
            
            if signal_returns:
                # Calculate efficiency as Sharpe ratio
                mean_return = np.mean(signal_returns)
                std_return = np.std(signal_returns)
                
                efficiency = mean_return / std_return if std_return > 0 else 0
                efficiency_scores[timeframe] = efficiency
            else:
                efficiency_scores[timeframe] = 0.0
        
        return efficiency_scores


class SignalAggregator:
    """
    Aggregator for combining signals from multiple timeframes.
    """
    
    def __init__(self):
        self.default_weights = {
            Timeframe.MINUTE_1: 0.1,
            Timeframe.MINUTE_5: 0.15,
            Timeframe.MINUTE_15: 0.2,
            Timeframe.MINUTE_30: 0.25,
            Timeframe.HOUR_1: 0.2,
            Timeframe.HOUR_4: 0.1,
            Timeframe.DAILY: 0.05
        }
        
    def aggregate_signals(
        self,
        signals: List[TimeframeSignal],
        method: ConflictResolution = ConflictResolution.WEIGHTED_AVERAGE,
        custom_weights: Optional[Dict[Timeframe, float]] = None
    ) -> AggregatedSignal:
        """
        Aggregate signals from multiple timeframes.
        
        Args:
            signals: List of signals from different timeframes
            method: Conflict resolution method
            custom_weights: Custom weights for timeframes
            
        Returns:
            Aggregated signal
        """
        
        if not signals:
            return AggregatedSignal(
                final_direction=0,
                final_strength=SignalStrength.WEAK,
                final_confidence=0.0,
                contributing_signals=[],
                conflict_resolution_method=method,
                timeframe_weights={},
                timestamp=datetime.now(),
                reasoning="No signals to aggregate"
            )
        
        # Use custom weights or default
        weights = custom_weights or self.default_weights
        
        # Apply conflict resolution method
        if method == ConflictResolution.HIGHER_TIMEFRAME:
            final_signal = self._resolve_higher_timeframe(signals)
        elif method == ConflictResolution.WEIGHTED_AVERAGE:
            final_signal = self._resolve_weighted_average(signals, weights)
        elif method == ConflictResolution.MAJORITY_VOTE:
            final_signal = self._resolve_majority_vote(signals)
        elif method == ConflictResolution.STRENGTH_BASED:
            final_signal = self._resolve_strength_based(signals)
        elif method == ConflictResolution.RECENT_SIGNAL:
            final_signal = self._resolve_recent_signal(signals)
        else:
            final_signal = self._resolve_weighted_average(signals, weights)
        
        return final_signal
    
    def _resolve_higher_timeframe(self, signals: List[TimeframeSignal]) -> AggregatedSignal:
        """Resolve conflicts by prioritizing higher timeframes"""
        
        # Sort by timeframe hierarchy (higher timeframes first)
        timeframe_order = {
            Timeframe.DAILY: 7,
            Timeframe.HOUR_4: 6,
            Timeframe.HOUR_1: 5,
            Timeframe.MINUTE_30: 4,
            Timeframe.MINUTE_15: 3,
            Timeframe.MINUTE_5: 2,
            Timeframe.MINUTE_1: 1
        }
        
        sorted_signals = sorted(signals, key=lambda x: timeframe_order.get(x.timeframe, 0), reverse=True)
        
        # Use highest timeframe signal
        primary_signal = sorted_signals[0]
        
        return AggregatedSignal(
            final_direction=primary_signal.direction,
            final_strength=primary_signal.strength,
            final_confidence=primary_signal.confidence,
            contributing_signals=signals,
            conflict_resolution_method=ConflictResolution.HIGHER_TIMEFRAME,
            timeframe_weights={primary_signal.timeframe: 1.0},
            timestamp=primary_signal.timestamp,
            reasoning=f"Higher timeframe priority: {primary_signal.timeframe.value}"
        )
    
    def _resolve_weighted_average(
        self,
        signals: List[TimeframeSignal],
        weights: Dict[Timeframe, float]
    ) -> AggregatedSignal:
        """Resolve conflicts using weighted average"""
        
        weighted_direction = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for signal in signals:
            weight = weights.get(signal.timeframe, 0.0)
            strength_multiplier = self._strength_to_numeric(signal.strength)
            
            effective_weight = weight * strength_multiplier
            
            weighted_direction += signal.direction * effective_weight
            weighted_confidence += signal.confidence * effective_weight
            total_weight += effective_weight
        
        if total_weight > 0:
            final_direction = weighted_direction / total_weight
            final_confidence = weighted_confidence / total_weight
        else:
            final_direction = 0.0
            final_confidence = 0.0
        
        # Convert to discrete direction
        if final_direction > 0.33:
            discrete_direction = 1
        elif final_direction < -0.33:
            discrete_direction = -1
        else:
            discrete_direction = 0
        
        # Determine final strength
        final_strength = self._numeric_to_strength(abs(final_direction))
        
        return AggregatedSignal(
            final_direction=discrete_direction,
            final_strength=final_strength,
            final_confidence=final_confidence,
            contributing_signals=signals,
            conflict_resolution_method=ConflictResolution.WEIGHTED_AVERAGE,
            timeframe_weights=weights,
            timestamp=max(signal.timestamp for signal in signals),
            reasoning=f"Weighted average: {final_direction:.2f}"
        )
    
    def _resolve_majority_vote(self, signals: List[TimeframeSignal]) -> AggregatedSignal:
        """Resolve conflicts using majority vote"""
        
        # Count votes for each direction
        votes = {-1: 0, 0: 0, 1: 0}
        
        for signal in signals:
            votes[signal.direction] += 1
        
        # Find majority direction
        majority_direction = max(votes, key=votes.get)
        
        # Calculate average confidence of majority signals
        majority_signals = [s for s in signals if s.direction == majority_direction]
        avg_confidence = np.mean([s.confidence for s in majority_signals])
        
        # Determine strength based on majority size
        majority_ratio = votes[majority_direction] / len(signals)
        if majority_ratio >= 0.8:
            final_strength = SignalStrength.VERY_STRONG
        elif majority_ratio >= 0.6:
            final_strength = SignalStrength.STRONG
        elif majority_ratio >= 0.5:
            final_strength = SignalStrength.MODERATE
        else:
            final_strength = SignalStrength.WEAK
        
        return AggregatedSignal(
            final_direction=majority_direction,
            final_strength=final_strength,
            final_confidence=avg_confidence,
            contributing_signals=signals,
            conflict_resolution_method=ConflictResolution.MAJORITY_VOTE,
            timeframe_weights={},
            timestamp=max(signal.timestamp for signal in signals),
            reasoning=f"Majority vote: {votes[majority_direction]}/{len(signals)}"
        )
    
    def _resolve_strength_based(self, signals: List[TimeframeSignal]) -> AggregatedSignal:
        """Resolve conflicts based on signal strength"""
        
        # Find strongest signal
        strongest_signal = max(signals, key=lambda x: (
            self._strength_to_numeric(x.strength) * x.confidence
        ))
        
        return AggregatedSignal(
            final_direction=strongest_signal.direction,
            final_strength=strongest_signal.strength,
            final_confidence=strongest_signal.confidence,
            contributing_signals=signals,
            conflict_resolution_method=ConflictResolution.STRENGTH_BASED,
            timeframe_weights={strongest_signal.timeframe: 1.0},
            timestamp=strongest_signal.timestamp,
            reasoning=f"Strongest signal: {strongest_signal.timeframe.value}"
        )
    
    def _resolve_recent_signal(self, signals: List[TimeframeSignal]) -> AggregatedSignal:
        """Resolve conflicts based on most recent signal"""
        
        # Find most recent signal
        most_recent = max(signals, key=lambda x: x.timestamp)
        
        return AggregatedSignal(
            final_direction=most_recent.direction,
            final_strength=most_recent.strength,
            final_confidence=most_recent.confidence,
            contributing_signals=signals,
            conflict_resolution_method=ConflictResolution.RECENT_SIGNAL,
            timeframe_weights={most_recent.timeframe: 1.0},
            timestamp=most_recent.timestamp,
            reasoning=f"Most recent signal: {most_recent.timeframe.value}"
        )
    
    def _strength_to_numeric(self, strength: SignalStrength) -> float:
        """Convert signal strength to numeric value"""
        strength_map = {
            SignalStrength.WEAK: 0.25,
            SignalStrength.MODERATE: 0.5,
            SignalStrength.STRONG: 0.75,
            SignalStrength.VERY_STRONG: 1.0
        }
        return strength_map.get(strength, 0.5)
    
    def _numeric_to_strength(self, value: float) -> SignalStrength:
        """Convert numeric value to signal strength"""
        if value >= 0.75:
            return SignalStrength.VERY_STRONG
        elif value >= 0.5:
            return SignalStrength.STRONG
        elif value >= 0.25:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK


class RegimeBasedStrategy:
    """
    Strategy that adapts timeframe selection based on market regime.
    """
    
    def __init__(self):
        self.regime_timeframe_preferences = {
            'trending': {
                Timeframe.MINUTE_30: 0.4,
                Timeframe.HOUR_1: 0.3,
                Timeframe.MINUTE_15: 0.2,
                Timeframe.MINUTE_5: 0.1
            },
            'ranging': {
                Timeframe.MINUTE_5: 0.4,
                Timeframe.MINUTE_15: 0.3,
                Timeframe.MINUTE_30: 0.2,
                Timeframe.HOUR_1: 0.1
            },
            'volatile': {
                Timeframe.MINUTE_1: 0.3,
                Timeframe.MINUTE_5: 0.4,
                Timeframe.MINUTE_15: 0.2,
                Timeframe.MINUTE_30: 0.1
            },
            'low_volatility': {
                Timeframe.HOUR_1: 0.4,
                Timeframe.MINUTE_30: 0.3,
                Timeframe.HOUR_4: 0.2,
                Timeframe.DAILY: 0.1
            }
        }
    
    def detect_market_regime(
        self,
        price_data: pd.DataFrame,
        window: int = 50
    ) -> str:
        """
        Detect current market regime.
        
        Args:
            price_data: Price data with OHLC
            window: Lookback window for regime detection
            
        Returns:
            Detected regime ('trending', 'ranging', 'volatile', 'low_volatility')
        """
        
        if len(price_data) < window:
            return 'ranging'  # Default regime
        
        # Calculate indicators
        closes = price_data['close'].iloc[-window:]
        
        # Trend strength (using linear regression slope)
        x = np.arange(len(closes))
        slope, _, r_value, _, _ = stats.linregress(x, closes)
        trend_strength = abs(r_value)
        
        # Volatility
        returns = closes.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Regime classification
        if trend_strength > 0.7:
            return 'trending'
        elif volatility > 0.3:
            return 'volatile'
        elif volatility < 0.1:
            return 'low_volatility'
        else:
            return 'ranging'
    
    def get_regime_weights(self, regime: str) -> Dict[Timeframe, float]:
        """Get timeframe weights for specific regime"""
        return self.regime_timeframe_preferences.get(regime, {})
    
    def adapt_strategy_to_regime(
        self,
        signals: List[TimeframeSignal],
        market_regime: str,
        aggregator: SignalAggregator
    ) -> AggregatedSignal:
        """
        Adapt strategy based on market regime.
        
        Args:
            signals: List of timeframe signals
            market_regime: Current market regime
            aggregator: Signal aggregator
            
        Returns:
            Regime-adapted aggregated signal
        """
        
        # Get regime-specific weights
        regime_weights = self.get_regime_weights(market_regime)
        
        # Aggregate signals with regime weights
        return aggregator.aggregate_signals(
            signals,
            method=ConflictResolution.WEIGHTED_AVERAGE,
            custom_weights=regime_weights
        )


class MultiTimeframeStrategy:
    """
    Comprehensive multi-timeframe trading strategy.
    """
    
    def __init__(self):
        self.timeframe_analyzer = TimeframeAnalyzer()
        self.signal_aggregator = SignalAggregator()
        self.regime_strategy = RegimeBasedStrategy()
        
        # Strategy configuration
        self.active_timeframes = [
            Timeframe.MINUTE_5,
            Timeframe.MINUTE_15,
            Timeframe.MINUTE_30,
            Timeframe.HOUR_1
        ]
        
        # Performance tracking
        self.performance_history: Dict[Timeframe, List[float]] = {}
        self.signal_history: List[AggregatedSignal] = []
        
    def generate_timeframe_signals(
        self,
        price_data: Dict[Timeframe, pd.DataFrame],
        timestamp: datetime
    ) -> List[TimeframeSignal]:
        """
        Generate signals for all active timeframes.
        
        Args:
            price_data: Price data for each timeframe
            timestamp: Current timestamp
            
        Returns:
            List of timeframe signals
        """
        
        signals = []
        
        for timeframe in self.active_timeframes:
            if timeframe not in price_data:
                continue
            
            tf_data = price_data[timeframe]
            
            # Generate signal for this timeframe
            signal = self._generate_single_timeframe_signal(tf_data, timeframe, timestamp)
            
            if signal:
                signals.append(signal)
        
        return signals
    
    def _generate_single_timeframe_signal(
        self,
        data: pd.DataFrame,
        timeframe: Timeframe,
        timestamp: datetime
    ) -> Optional[TimeframeSignal]:
        """Generate signal for single timeframe"""
        
        if len(data) < 20:
            return None
        
        # Calculate indicators
        closes = data['close']
        
        # Moving averages
        ma_fast = closes.rolling(window=10).mean()
        ma_slow = closes.rolling(window=20).mean()
        
        # RSI
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        ema_fast = closes.ewm(span=12).mean()
        ema_slow = closes.ewm(span=26).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=9).mean()
        
        # Current values
        current_close = closes.iloc[-1]
        current_ma_fast = ma_fast.iloc[-1]
        current_ma_slow = ma_slow.iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_macd = macd.iloc[-1]
        current_macd_signal = macd_signal.iloc[-1]
        
        # Generate signal
        signal_direction = 0
        signal_strength = SignalStrength.WEAK
        confidence = 0.5
        
        # MA crossover
        if current_ma_fast > current_ma_slow:
            signal_direction = 1
            confidence += 0.2
        elif current_ma_fast < current_ma_slow:
            signal_direction = -1
            confidence += 0.2
        
        # RSI conditions
        if current_rsi < 30:
            signal_direction = 1
            confidence += 0.15
        elif current_rsi > 70:
            signal_direction = -1
            confidence += 0.15
        
        # MACD conditions
        if current_macd > current_macd_signal:
            signal_direction = 1
            confidence += 0.1
        elif current_macd < current_macd_signal:
            signal_direction = -1
            confidence += 0.1
        
        # Determine strength based on confidence
        if confidence >= 0.8:
            signal_strength = SignalStrength.VERY_STRONG
        elif confidence >= 0.6:
            signal_strength = SignalStrength.STRONG
        elif confidence >= 0.4:
            signal_strength = SignalStrength.MODERATE
        else:
            signal_strength = SignalStrength.WEAK
        
        # Volume (if available)
        volume = data['volume'].iloc[-1] if 'volume' in data.columns else 1000
        
        return TimeframeSignal(
            timeframe=timeframe,
            signal_type=SignalType.ENTRY,
            direction=signal_direction,
            strength=signal_strength,
            confidence=min(1.0, confidence),
            timestamp=timestamp,
            price=current_close,
            volume=volume,
            indicators={
                'ma_fast': current_ma_fast,
                'ma_slow': current_ma_slow,
                'rsi': current_rsi,
                'macd': current_macd,
                'macd_signal': current_macd_signal
            },
            metadata={
                'timeframe': timeframe.value,
                'data_length': len(data)
            }
        )
    
    def execute_strategy(
        self,
        price_data: Dict[Timeframe, pd.DataFrame],
        timestamp: datetime,
        conflict_resolution: ConflictResolution = ConflictResolution.WEIGHTED_AVERAGE
    ) -> AggregatedSignal:
        """
        Execute multi-timeframe strategy.
        
        Args:
            price_data: Price data for each timeframe
            timestamp: Current timestamp
            conflict_resolution: Method for resolving conflicts
            
        Returns:
            Aggregated trading signal
        """
        
        # Generate signals for all timeframes
        timeframe_signals = self.generate_timeframe_signals(price_data, timestamp)
        
        if not timeframe_signals:
            return AggregatedSignal(
                final_direction=0,
                final_strength=SignalStrength.WEAK,
                final_confidence=0.0,
                contributing_signals=[],
                conflict_resolution_method=conflict_resolution,
                timeframe_weights={},
                timestamp=timestamp,
                reasoning="No timeframe signals generated"
            )
        
        # Detect market regime
        if Timeframe.MINUTE_30 in price_data:
            market_regime = self.regime_strategy.detect_market_regime(
                price_data[Timeframe.MINUTE_30]
            )
            
            # Use regime-based aggregation
            aggregated_signal = self.regime_strategy.adapt_strategy_to_regime(
                timeframe_signals, market_regime, self.signal_aggregator
            )
            
            aggregated_signal.reasoning += f" (Regime: {market_regime})"
        else:
            # Use standard aggregation
            aggregated_signal = self.signal_aggregator.aggregate_signals(
                timeframe_signals, conflict_resolution
            )
        
        # Store signal history
        self.signal_history.append(aggregated_signal)
        
        return aggregated_signal
    
    def analyze_strategy_performance(
        self,
        returns: pd.Series,
        lookback_period: int = 100
    ) -> Dict[Timeframe, StrategyPerformance]:
        """
        Analyze strategy performance by timeframe.
        
        Args:
            returns: Price returns series
            lookback_period: Lookback period for analysis
            
        Returns:
            Performance metrics by timeframe
        """
        
        performance_metrics = {}
        
        # Get recent signals
        recent_signals = self.signal_history[-lookback_period:] if len(self.signal_history) >= lookback_period else self.signal_history
        
        if not recent_signals:
            return performance_metrics
        
        # Analyze performance by timeframe
        for timeframe in self.active_timeframes:
            # Get signals that involved this timeframe
            timeframe_contributions = []
            
            for agg_signal in recent_signals:
                for contrib_signal in agg_signal.contributing_signals:
                    if contrib_signal.timeframe == timeframe:
                        timeframe_contributions.append({
                            'signal': contrib_signal,
                            'timestamp': agg_signal.timestamp,
                            'final_direction': agg_signal.final_direction
                        })
            
            if not timeframe_contributions:
                continue
            
            # Calculate performance metrics
            signal_returns = []
            correct_signals = 0
            
            for contrib in timeframe_contributions:
                signal_time = contrib['timestamp']
                signal_direction = contrib['signal'].direction
                
                # Get future returns
                future_returns = returns[returns.index > signal_time][:5]  # Next 5 periods
                
                if len(future_returns) >= 5:
                    cumulative_return = (1 + future_returns).prod() - 1
                    adjusted_return = cumulative_return * signal_direction
                    signal_returns.append(adjusted_return)
                    
                    # Check if signal was correct
                    if (signal_direction > 0 and cumulative_return > 0) or \
                       (signal_direction < 0 and cumulative_return < 0):
                        correct_signals += 1
            
            if signal_returns:
                total_signals = len(signal_returns)
                accuracy = correct_signals / total_signals
                avg_return = np.mean(signal_returns)
                volatility = np.std(signal_returns)
                
                # Profit factor
                positive_returns = [r for r in signal_returns if r > 0]
                negative_returns = [r for r in signal_returns if r < 0]
                
                gross_profit = sum(positive_returns) if positive_returns else 0
                gross_loss = abs(sum(negative_returns)) if negative_returns else 0
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
                
                # Sharpe ratio
                sharpe_ratio = avg_return / volatility if volatility > 0 else 0
                
                # Max drawdown (simplified)
                cumulative_returns = np.cumsum(signal_returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = cumulative_returns - running_max
                max_drawdown = np.min(drawdown)
                
                # Win rate
                win_rate = len(positive_returns) / total_signals
                
                performance_metrics[timeframe] = StrategyPerformance(
                    timeframe=timeframe,
                    total_signals=total_signals,
                    correct_signals=correct_signals,
                    accuracy=accuracy,
                    profit_factor=profit_factor,
                    avg_return=avg_return,
                    volatility=volatility,
                    sharpe_ratio=sharpe_ratio,
                    max_drawdown=max_drawdown,
                    win_rate=win_rate
                )
        
        return performance_metrics
    
    def optimize_timeframe_weights(
        self,
        performance_metrics: Dict[Timeframe, StrategyPerformance]
    ) -> Dict[Timeframe, float]:
        """
        Optimize timeframe weights based on performance.
        
        Args:
            performance_metrics: Performance metrics by timeframe
            
        Returns:
            Optimized weights for each timeframe
        """
        
        if not performance_metrics:
            return self.signal_aggregator.default_weights
        
        # Calculate weight based on Sharpe ratio and accuracy
        weights = {}
        total_score = 0
        
        for timeframe, performance in performance_metrics.items():
            # Combined score: Sharpe ratio + accuracy
            score = performance.sharpe_ratio + performance.accuracy
            weights[timeframe] = max(0, score)  # Ensure non-negative
            total_score += weights[timeframe]
        
        # Normalize weights
        if total_score > 0:
            for timeframe in weights:
                weights[timeframe] /= total_score
        else:
            # Fall back to equal weights
            n_timeframes = len(performance_metrics)
            weights = {tf: 1.0 / n_timeframes for tf in performance_metrics}
        
        return weights


# Test fixtures
@pytest.fixture
def sample_multi_timeframe_data():
    """Generate sample multi-timeframe data"""
    
    np.random.seed(42)
    
    # Generate base daily data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    base_price = 100.0
    
    # Create data for different timeframes
    timeframe_data = {}
    
    # Daily data
    daily_prices = []
    for _ in dates:
        daily_return = np.random.normal(0.001, 0.02)
        base_price *= (1 + daily_return)
        daily_prices.append(base_price)
    
    # Create intraday data by interpolating daily data
    for timeframe in [Timeframe.MINUTE_5, Timeframe.MINUTE_15, Timeframe.MINUTE_30, Timeframe.HOUR_1]:
        # Determine frequency
        if timeframe == Timeframe.MINUTE_5:
            freq = '5min'
            periods_per_day = 288  # 24 * 60 / 5
        elif timeframe == Timeframe.MINUTE_15:
            freq = '15min'
            periods_per_day = 96   # 24 * 60 / 15
        elif timeframe == Timeframe.MINUTE_30:
            freq = '30min'
            periods_per_day = 48   # 24 * 60 / 30
        else:  # HOUR_1
            freq = '1h'
            periods_per_day = 24
        
        # Generate intraday timestamps
        intraday_dates = pd.date_range(
            start=dates[0],
            end=dates[-1],
            freq=freq
        )
        
        # Generate intraday prices
        intraday_prices = []
        volumes = []
        
        for i, date in enumerate(intraday_dates):
            # Add intraday noise
            noise = np.random.normal(0, 0.005)
            day_idx = min(i // periods_per_day, len(daily_prices) - 1)
            
            price = daily_prices[day_idx] * (1 + noise)
            intraday_prices.append(price)
            
            # Generate volume
            volume = np.random.lognormal(10, 1)
            volumes.append(volume)
        
        # Create OHLC data
        ohlc_data = []
        for i in range(0, len(intraday_prices), 4):
            chunk = intraday_prices[i:i+4]
            if len(chunk) < 4:
                chunk.extend([chunk[-1]] * (4 - len(chunk)))
            
            ohlc_data.append({
                'open': chunk[0],
                'high': max(chunk),
                'low': min(chunk),
                'close': chunk[-1],
                'volume': np.mean(volumes[i:i+4])
            })
        
        # Create DataFrame
        ohlc_dates = intraday_dates[::4][:len(ohlc_data)]
        timeframe_data[timeframe] = pd.DataFrame(ohlc_data, index=ohlc_dates)
    
    return timeframe_data


@pytest.fixture
def timeframe_analyzer():
    """Create timeframe analyzer instance"""
    return TimeframeAnalyzer()


@pytest.fixture
def signal_aggregator():
    """Create signal aggregator instance"""
    return SignalAggregator()


@pytest.fixture
def regime_strategy():
    """Create regime-based strategy instance"""
    return RegimeBasedStrategy()


@pytest.fixture
def multi_timeframe_strategy():
    """Create multi-timeframe strategy instance"""
    return MultiTimeframeStrategy()


# Comprehensive test suite
@pytest.mark.asyncio
class TestMultiTimeframeStrategies:
    """Comprehensive multi-timeframe strategies tests"""
    
    def test_timeframe_signal_generation(self, multi_timeframe_strategy, sample_multi_timeframe_data):
        """Test signal generation for multiple timeframes"""
        
        timestamp = datetime.now()
        signals = multi_timeframe_strategy.generate_timeframe_signals(
            sample_multi_timeframe_data, timestamp
        )
        
        assert len(signals) > 0
        assert all(isinstance(signal, TimeframeSignal) for signal in signals)
        
        # Check signal properties
        for signal in signals:
            assert signal.timeframe in multi_timeframe_strategy.active_timeframes
            assert signal.direction in [-1, 0, 1]
            assert isinstance(signal.strength, SignalStrength)
            assert 0 <= signal.confidence <= 1
            assert signal.timestamp == timestamp
            assert signal.price > 0
            assert len(signal.indicators) > 0
    
    def test_signal_aggregation_weighted_average(self, signal_aggregator):
        """Test weighted average signal aggregation"""
        
        # Create test signals
        signals = [
            TimeframeSignal(
                timeframe=Timeframe.MINUTE_5,
                signal_type=SignalType.ENTRY,
                direction=1,
                strength=SignalStrength.STRONG,
                confidence=0.8,
                timestamp=datetime.now(),
                price=100.0,
                volume=1000,
                indicators={},
                metadata={}
            ),
            TimeframeSignal(
                timeframe=Timeframe.MINUTE_30,
                signal_type=SignalType.ENTRY,
                direction=-1,
                strength=SignalStrength.MODERATE,
                confidence=0.6,
                timestamp=datetime.now(),
                price=100.0,
                volume=1000,
                indicators={},
                metadata={}
            )
        ]
        
        aggregated = signal_aggregator.aggregate_signals(
            signals, ConflictResolution.WEIGHTED_AVERAGE
        )
        
        assert isinstance(aggregated, AggregatedSignal)
        assert aggregated.final_direction in [-1, 0, 1]
        assert isinstance(aggregated.final_strength, SignalStrength)
        assert 0 <= aggregated.final_confidence <= 1
        assert len(aggregated.contributing_signals) == 2
        assert aggregated.conflict_resolution_method == ConflictResolution.WEIGHTED_AVERAGE
    
    def test_signal_aggregation_higher_timeframe(self, signal_aggregator):
        """Test higher timeframe priority aggregation"""
        
        signals = [
            TimeframeSignal(
                timeframe=Timeframe.MINUTE_5,
                signal_type=SignalType.ENTRY,
                direction=1,
                strength=SignalStrength.VERY_STRONG,
                confidence=0.9,
                timestamp=datetime.now(),
                price=100.0,
                volume=1000,
                indicators={},
                metadata={}
            ),
            TimeframeSignal(
                timeframe=Timeframe.HOUR_1,
                signal_type=SignalType.ENTRY,
                direction=-1,
                strength=SignalStrength.MODERATE,
                confidence=0.6,
                timestamp=datetime.now(),
                price=100.0,
                volume=1000,
                indicators={},
                metadata={}
            )
        ]
        
        aggregated = signal_aggregator.aggregate_signals(
            signals, ConflictResolution.HIGHER_TIMEFRAME
        )
        
        # Should prioritize the higher timeframe (HOUR_1)
        assert aggregated.final_direction == -1
        assert aggregated.final_strength == SignalStrength.MODERATE
        assert aggregated.conflict_resolution_method == ConflictResolution.HIGHER_TIMEFRAME
    
    def test_signal_aggregation_majority_vote(self, signal_aggregator):
        """Test majority vote aggregation"""
        
        signals = [
            TimeframeSignal(
                timeframe=Timeframe.MINUTE_5,
                signal_type=SignalType.ENTRY,
                direction=1,
                strength=SignalStrength.MODERATE,
                confidence=0.7,
                timestamp=datetime.now(),
                price=100.0,
                volume=1000,
                indicators={},
                metadata={}
            ),
            TimeframeSignal(
                timeframe=Timeframe.MINUTE_15,
                signal_type=SignalType.ENTRY,
                direction=1,
                strength=SignalStrength.STRONG,
                confidence=0.8,
                timestamp=datetime.now(),
                price=100.0,
                volume=1000,
                indicators={},
                metadata={}
            ),
            TimeframeSignal(
                timeframe=Timeframe.MINUTE_30,
                signal_type=SignalType.ENTRY,
                direction=-1,
                strength=SignalStrength.WEAK,
                confidence=0.5,
                timestamp=datetime.now(),
                price=100.0,
                volume=1000,
                indicators={},
                metadata={}
            )
        ]
        
        aggregated = signal_aggregator.aggregate_signals(
            signals, ConflictResolution.MAJORITY_VOTE
        )
        
        # Should choose majority direction (1)
        assert aggregated.final_direction == 1
        assert aggregated.conflict_resolution_method == ConflictResolution.MAJORITY_VOTE
    
    def test_timeframe_correlation_analysis(self, timeframe_analyzer):
        """Test timeframe correlation analysis"""
        
        # Create test signals
        signals = {
            Timeframe.MINUTE_5: [
                TimeframeSignal(
                    timeframe=Timeframe.MINUTE_5,
                    signal_type=SignalType.ENTRY,
                    direction=1,
                    strength=SignalStrength.STRONG,
                    confidence=0.8,
                    timestamp=datetime.now() - timedelta(minutes=i),
                    price=100.0,
                    volume=1000,
                    indicators={},
                    metadata={}
                )
                for i in range(100)
            ],
            Timeframe.MINUTE_30: [
                TimeframeSignal(
                    timeframe=Timeframe.MINUTE_30,
                    signal_type=SignalType.ENTRY,
                    direction=1 if i % 2 == 0 else -1,
                    strength=SignalStrength.MODERATE,
                    confidence=0.6,
                    timestamp=datetime.now() - timedelta(minutes=i*30),
                    price=100.0,
                    volume=1000,
                    indicators={},
                    metadata={}
                )
                for i in range(20)
            ]
        }
        
        correlation_matrix = timeframe_analyzer.analyze_timeframe_correlation(signals)
        
        # Should return correlation matrix
        if not correlation_matrix.empty:
            assert isinstance(correlation_matrix, pd.DataFrame)
            assert correlation_matrix.shape[0] == correlation_matrix.shape[1]
            # Correlation values should be between -1 and 1
            assert (correlation_matrix >= -1).all().all()
            assert (correlation_matrix <= 1).all().all()
    
    def test_timeframe_efficiency_analysis(self, timeframe_analyzer):
        """Test timeframe efficiency analysis"""
        
        # Create test signals
        signals = {
            Timeframe.MINUTE_5: [
                TimeframeSignal(
                    timeframe=Timeframe.MINUTE_5,
                    signal_type=SignalType.ENTRY,
                    direction=1,
                    strength=SignalStrength.STRONG,
                    confidence=0.8,
                    timestamp=datetime.now() - timedelta(minutes=i),
                    price=100.0,
                    volume=1000,
                    indicators={},
                    metadata={}
                )
                for i in range(50)
            ]
        }
        
        # Create test returns
        returns = pd.Series(
            np.random.normal(0.001, 0.02, 200),
            index=pd.date_range('2023-01-01', periods=200, freq='5min')
        )
        
        efficiency_scores = timeframe_analyzer.calculate_timeframe_efficiency(
            signals, returns
        )
        
        assert isinstance(efficiency_scores, dict)
        assert Timeframe.MINUTE_5 in efficiency_scores
        assert isinstance(efficiency_scores[Timeframe.MINUTE_5], float)
    
    def test_regime_detection(self, regime_strategy):
        """Test market regime detection"""
        
        # Create test data for different regimes
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Trending market
        trending_prices = [100 + i * 0.5 for i in range(100)]
        trending_data = pd.DataFrame({
            'open': trending_prices,
            'high': [p * 1.02 for p in trending_prices],
            'low': [p * 0.98 for p in trending_prices],
            'close': trending_prices,
            'volume': [1000] * 100
        }, index=dates)
        
        regime = regime_strategy.detect_market_regime(trending_data)
        assert regime in ['trending', 'ranging', 'volatile', 'low_volatility']
        
        # Volatile market
        volatile_prices = [100 + np.random.normal(0, 5) for _ in range(100)]
        volatile_data = pd.DataFrame({
            'open': volatile_prices,
            'high': [p * 1.05 for p in volatile_prices],
            'low': [p * 0.95 for p in volatile_prices],
            'close': volatile_prices,
            'volume': [1000] * 100
        }, index=dates)
        
        regime = regime_strategy.detect_market_regime(volatile_data)
        assert regime in ['trending', 'ranging', 'volatile', 'low_volatility']
    
    def test_regime_based_strategy_adaptation(self, regime_strategy, signal_aggregator):
        """Test regime-based strategy adaptation"""
        
        # Create test signals
        signals = [
            TimeframeSignal(
                timeframe=Timeframe.MINUTE_5,
                signal_type=SignalType.ENTRY,
                direction=1,
                strength=SignalStrength.STRONG,
                confidence=0.8,
                timestamp=datetime.now(),
                price=100.0,
                volume=1000,
                indicators={},
                metadata={}
            ),
            TimeframeSignal(
                timeframe=Timeframe.MINUTE_30,
                signal_type=SignalType.ENTRY,
                direction=-1,
                strength=SignalStrength.MODERATE,
                confidence=0.6,
                timestamp=datetime.now(),
                price=100.0,
                volume=1000,
                indicators={},
                metadata={}
            )
        ]
        
        # Test different regimes
        for regime in ['trending', 'ranging', 'volatile', 'low_volatility']:
            adapted_signal = regime_strategy.adapt_strategy_to_regime(
                signals, regime, signal_aggregator
            )
            
            assert isinstance(adapted_signal, AggregatedSignal)
            assert adapted_signal.final_direction in [-1, 0, 1]
            assert regime in adapted_signal.reasoning
    
    def test_complete_multi_timeframe_strategy(self, multi_timeframe_strategy, sample_multi_timeframe_data):
        """Test complete multi-timeframe strategy execution"""
        
        timestamp = datetime.now()
        
        aggregated_signal = multi_timeframe_strategy.execute_strategy(
            sample_multi_timeframe_data,
            timestamp,
            ConflictResolution.WEIGHTED_AVERAGE
        )
        
        assert isinstance(aggregated_signal, AggregatedSignal)
        assert aggregated_signal.final_direction in [-1, 0, 1]
        assert isinstance(aggregated_signal.final_strength, SignalStrength)
        assert 0 <= aggregated_signal.final_confidence <= 1
        assert len(aggregated_signal.contributing_signals) > 0
        assert aggregated_signal.timestamp == timestamp
        
        # Check that signal was stored in history
        assert len(multi_timeframe_strategy.signal_history) > 0
        assert multi_timeframe_strategy.signal_history[-1] == aggregated_signal
    
    def test_strategy_performance_analysis(self, multi_timeframe_strategy):
        """Test strategy performance analysis"""
        
        # Create dummy signal history
        for i in range(50):
            dummy_signal = AggregatedSignal(
                final_direction=1 if i % 2 == 0 else -1,
                final_strength=SignalStrength.MODERATE,
                final_confidence=0.6,
                contributing_signals=[
                    TimeframeSignal(
                        timeframe=Timeframe.MINUTE_5,
                        signal_type=SignalType.ENTRY,
                        direction=1 if i % 2 == 0 else -1,
                        strength=SignalStrength.MODERATE,
                        confidence=0.6,
                        timestamp=datetime.now() - timedelta(hours=i),
                        price=100.0,
                        volume=1000,
                        indicators={},
                        metadata={}
                    )
                ],
                conflict_resolution_method=ConflictResolution.WEIGHTED_AVERAGE,
                timeframe_weights={},
                timestamp=datetime.now() - timedelta(hours=i),
                reasoning="Test signal"
            )
            multi_timeframe_strategy.signal_history.append(dummy_signal)
        
        # Create dummy returns
        returns = pd.Series(
            np.random.normal(0.001, 0.02, 200),
            index=pd.date_range('2023-01-01', periods=200, freq='h')
        )
        
        performance = multi_timeframe_strategy.analyze_strategy_performance(returns)
        
        assert isinstance(performance, dict)
        
        for timeframe, metrics in performance.items():
            assert isinstance(metrics, StrategyPerformance)
            assert metrics.timeframe == timeframe
            assert metrics.total_signals > 0
            assert 0 <= metrics.accuracy <= 1
            assert 0 <= metrics.win_rate <= 1
    
    def test_timeframe_weight_optimization(self, multi_timeframe_strategy):
        """Test timeframe weight optimization"""
        
        # Create dummy performance metrics
        performance_metrics = {
            Timeframe.MINUTE_5: StrategyPerformance(
                timeframe=Timeframe.MINUTE_5,
                total_signals=50,
                correct_signals=30,
                accuracy=0.6,
                profit_factor=1.5,
                avg_return=0.002,
                volatility=0.01,
                sharpe_ratio=0.2,
                max_drawdown=-0.05,
                win_rate=0.6
            ),
            Timeframe.MINUTE_30: StrategyPerformance(
                timeframe=Timeframe.MINUTE_30,
                total_signals=30,
                correct_signals=25,
                accuracy=0.83,
                profit_factor=2.0,
                avg_return=0.003,
                volatility=0.008,
                sharpe_ratio=0.375,
                max_drawdown=-0.03,
                win_rate=0.83
            )
        }
        
        optimized_weights = multi_timeframe_strategy.optimize_timeframe_weights(
            performance_metrics
        )
        
        assert isinstance(optimized_weights, dict)
        assert len(optimized_weights) == len(performance_metrics)
        
        # Weights should sum to 1
        assert abs(sum(optimized_weights.values()) - 1.0) < 1e-6
        
        # Higher performing timeframe should have higher weight
        assert optimized_weights[Timeframe.MINUTE_30] > optimized_weights[Timeframe.MINUTE_5]
    
    def test_conflicting_signals_resolution(self, signal_aggregator):
        """Test resolution of conflicting signals"""
        
        # Create strongly conflicting signals
        conflicting_signals = [
            TimeframeSignal(
                timeframe=Timeframe.MINUTE_5,
                signal_type=SignalType.ENTRY,
                direction=1,
                strength=SignalStrength.VERY_STRONG,
                confidence=0.9,
                timestamp=datetime.now(),
                price=100.0,
                volume=1000,
                indicators={},
                metadata={}
            ),
            TimeframeSignal(
                timeframe=Timeframe.MINUTE_30,
                signal_type=SignalType.ENTRY,
                direction=-1,
                strength=SignalStrength.VERY_STRONG,
                confidence=0.9,
                timestamp=datetime.now(),
                price=100.0,
                volume=1000,
                indicators={},
                metadata={}
            )
        ]
        
        # Test different conflict resolution methods
        methods = [
            ConflictResolution.HIGHER_TIMEFRAME,
            ConflictResolution.WEIGHTED_AVERAGE,
            ConflictResolution.MAJORITY_VOTE,
            ConflictResolution.STRENGTH_BASED,
            ConflictResolution.RECENT_SIGNAL
        ]
        
        for method in methods:
            result = signal_aggregator.aggregate_signals(conflicting_signals, method)
            
            assert isinstance(result, AggregatedSignal)
            assert result.final_direction in [-1, 0, 1]
            assert result.conflict_resolution_method == method
            assert len(result.contributing_signals) == 2
    
    def test_signal_strength_consistency(self, signal_aggregator):
        """Test signal strength consistency across aggregation methods"""
        
        # Create signals with varying strengths
        signals = [
            TimeframeSignal(
                timeframe=Timeframe.MINUTE_5,
                signal_type=SignalType.ENTRY,
                direction=1,
                strength=SignalStrength.WEAK,
                confidence=0.3,
                timestamp=datetime.now(),
                price=100.0,
                volume=1000,
                indicators={},
                metadata={}
            ),
            TimeframeSignal(
                timeframe=Timeframe.MINUTE_30,
                signal_type=SignalType.ENTRY,
                direction=1,
                strength=SignalStrength.VERY_STRONG,
                confidence=0.95,
                timestamp=datetime.now(),
                price=100.0,
                volume=1000,
                indicators={},
                metadata={}
            )
        ]
        
        # Test strength-based aggregation
        result = signal_aggregator.aggregate_signals(
            signals, ConflictResolution.STRENGTH_BASED
        )
        
        # Should choose the stronger signal
        assert result.final_direction == 1
        assert result.final_strength == SignalStrength.VERY_STRONG
    
    def test_empty_signal_handling(self, signal_aggregator, multi_timeframe_strategy):
        """Test handling of empty or invalid signals"""
        
        # Test empty signal list
        empty_result = signal_aggregator.aggregate_signals([])
        assert empty_result.final_direction == 0
        assert empty_result.final_confidence == 0.0
        assert len(empty_result.contributing_signals) == 0
        
        # Test strategy with no data
        empty_strategy_result = multi_timeframe_strategy.execute_strategy(
            {}, datetime.now()
        )
        assert empty_strategy_result.final_direction == 0
        assert empty_strategy_result.final_confidence == 0.0
    
    def test_signal_timing_consistency(self, multi_timeframe_strategy, sample_multi_timeframe_data):
        """Test signal timing consistency across timeframes"""
        
        timestamp = datetime.now()
        
        # Execute strategy multiple times with same data
        results = []
        for _ in range(5):
            result = multi_timeframe_strategy.execute_strategy(
                sample_multi_timeframe_data, timestamp
            )
            results.append(result)
        
        # Results should be consistent for same timestamp and data
        first_result = results[0]
        for result in results[1:]:
            assert result.final_direction == first_result.final_direction
            assert result.final_strength == first_result.final_strength
            assert abs(result.final_confidence - first_result.final_confidence) < 0.1
    
    def test_large_scale_signal_processing(self, multi_timeframe_strategy):
        """Test processing of large numbers of signals"""
        
        # Create large number of signals
        large_signal_list = []
        for i in range(1000):
            signal = TimeframeSignal(
                timeframe=Timeframe.MINUTE_5,
                signal_type=SignalType.ENTRY,
                direction=1 if i % 2 == 0 else -1,
                strength=SignalStrength.MODERATE,
                confidence=0.5,
                timestamp=datetime.now() - timedelta(minutes=i),
                price=100.0,
                volume=1000,
                indicators={},
                metadata={}
            )
            large_signal_list.append(signal)
        
        # Add to strategy history
        for signal in large_signal_list[:500]:  # Add first 500 as aggregated signals
            agg_signal = AggregatedSignal(
                final_direction=signal.direction,
                final_strength=signal.strength,
                final_confidence=signal.confidence,
                contributing_signals=[signal],
                conflict_resolution_method=ConflictResolution.WEIGHTED_AVERAGE,
                timeframe_weights={},
                timestamp=signal.timestamp,
                reasoning="Test signal"
            )
            multi_timeframe_strategy.signal_history.append(agg_signal)
        
        # Test performance analysis with large dataset
        returns = pd.Series(
            np.random.normal(0.001, 0.02, 2000),
            index=pd.date_range('2023-01-01', periods=2000, freq='5min')
        )
        
        performance = multi_timeframe_strategy.analyze_strategy_performance(returns)
        
        # Should handle large dataset without issues
        assert isinstance(performance, dict)
        if performance:
            for metrics in performance.values():
                assert metrics.total_signals > 0
                assert 0 <= metrics.accuracy <= 1
    
    def test_dynamic_timeframe_switching(self, multi_timeframe_strategy, sample_multi_timeframe_data):
        """Test dynamic switching between timeframes based on conditions"""
        
        timestamp = datetime.now()
        
        # Test different market conditions
        conditions = ['normal', 'volatile', 'trending']
        
        results = {}
        for condition in conditions:
            # Modify data based on condition
            modified_data = sample_multi_timeframe_data.copy()
            
            if condition == 'volatile':
                # Increase volatility
                for timeframe in modified_data:
                    modified_data[timeframe] = modified_data[timeframe] * (1 + np.random.normal(0, 0.1, len(modified_data[timeframe])))
            
            elif condition == 'trending':
                # Add trend
                for timeframe in modified_data:
                    trend = np.linspace(0, 0.1, len(modified_data[timeframe]))
                    modified_data[timeframe] = modified_data[timeframe] * (1 + trend.reshape(-1, 1))
            
            result = multi_timeframe_strategy.execute_strategy(
                modified_data, timestamp
            )
            results[condition] = result
        
        # Results should adapt to different conditions
        for condition, result in results.items():
            assert isinstance(result, AggregatedSignal)
            assert result.final_direction in [-1, 0, 1]
            assert len(result.contributing_signals) > 0