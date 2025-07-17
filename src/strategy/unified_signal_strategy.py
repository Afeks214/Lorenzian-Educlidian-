"""
Unified Signal Strategy with Proper Timeframe Alignment

This strategy implementation uses the new signal alignment system to properly
synchronize signals across different timeframes without the mapping_indices
calculation errors found in the original notebook.

Key Features:
1. Proper signal interpolation instead of naive index mapping
2. Synchronized signal processing across timeframes
3. Deterministic signal ordering
4. Signal validation and confidence scoring
5. Temporal consistency enforcement

Author: Claude (Anthropic)
Date: 2025-01-17
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass

from src.core.signal_alignment import (
    SignalAlignmentEngine, SignalType, SignalDirection,
    create_signal_alignment_engine
)
from src.indicators.custom.nwrqk import NWRQKCalculator
from src.indicators.custom.mlmi import MLMICalculator
from src.indicators.custom.fvg import FVGDetector
from src.core.minimal_dependencies import EventBus, BarData

logger = structlog.get_logger(__name__)


@dataclass
class StrategyConfig:
    """Configuration for the unified signal strategy"""
    # Signal alignment settings
    signal_alignment_config: Dict[str, Any]
    
    # Indicator configurations
    nwrqk_config: Dict[str, Any]
    mlmi_config: Dict[str, Any]
    fvg_config: Dict[str, Any]
    
    # Strategy settings
    min_signal_confidence: float = 0.6
    signal_timeout_minutes: int = 30
    max_concurrent_signals: int = 5
    
    # Synergy patterns
    synergy_patterns: List[List[SignalType]] = None
    
    def __post_init__(self):
        if self.synergy_patterns is None:
            self.synergy_patterns = [
                [SignalType.MLMI, SignalType.FVG, SignalType.NWRQK],
                [SignalType.MLMI, SignalType.NWRQK, SignalType.FVG],
                [SignalType.NWRQK, SignalType.MLMI, SignalType.FVG],
                [SignalType.NWRQK, SignalType.FVG, SignalType.MLMI]
            ]


class UnifiedSignalStrategy:
    """
    Unified strategy that properly handles signal alignment across timeframes
    """
    
    def __init__(self, config: StrategyConfig, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.logger = structlog.get_logger(self.__class__.__name__)
        
        # Initialize signal alignment engine
        self.signal_engine = create_signal_alignment_engine(config.signal_alignment_config)
        
        # Initialize indicators with signal alignment
        self.nwrqk = NWRQKCalculator(config.nwrqk_config, event_bus)
        self.mlmi = MLMICalculator(config.mlmi_config, event_bus)
        self.fvg = FVGDetector(config.fvg_config, event_bus)
        
        # Strategy state
        self.current_position = 0  # -1: short, 0: neutral, 1: long
        self.entry_price = 0.0
        self.entry_timestamp = None
        self.current_synergy_pattern = None
        self.synergy_state = {}
        
        # Performance tracking
        self.trades_executed = 0
        self.signals_processed = 0
        self.synergy_matches = 0
        
        # Signal history for analysis
        self.signal_history = []
        self.trade_history = []
        
        self.logger.info("Unified Signal Strategy initialized")
    
    def process_5m_bar(self, bar: BarData) -> Dict[str, Any]:
        """
        Process a 5-minute bar through the complete signal alignment pipeline
        
        Args:
            bar: 5-minute bar data
            
        Returns:
            Dictionary containing strategy signals and state
        """
        try:
            # Process bar through FVG detector (5m only)
            fvg_results = self.fvg.calculate_5m(bar)
            
            # Get synchronized signals from all timeframes
            synchronized_signals = self.signal_engine.get_synchronized_signals(bar.timestamp)
            
            # Analyze synergy patterns
            synergy_result = self._analyze_synergy_patterns(synchronized_signals, bar)
            
            # Generate trading signals
            trading_signals = self._generate_trading_signals(synergy_result, bar)
            
            # Update strategy state
            self._update_strategy_state(trading_signals, bar)
            
            # Clean up old signals
            self._cleanup_old_signals(bar.timestamp)
            
            self.signals_processed += 1
            
            return {
                'timestamp': bar.timestamp,
                'current_position': self.current_position,
                'entry_price': self.entry_price,
                'fvg_results': fvg_results,
                'synchronized_signals': {k.value: v for k, v in synchronized_signals.items()},
                'synergy_result': synergy_result,
                'trading_signals': trading_signals,
                'strategy_stats': self._get_strategy_stats()
            }
            
        except Exception as e:
            self.logger.error("Error processing 5m bar", error=str(e))
            return {'error': str(e)}
    
    def process_30m_bar(self, bar: BarData) -> Dict[str, Any]:
        """
        Process a 30-minute bar through MLMI and NW-RQK indicators
        
        Args:
            bar: 30-minute bar data
            
        Returns:
            Dictionary containing indicator results
        """
        try:
            # Process bar through 30m indicators
            nwrqk_results = self.nwrqk.calculate_30m(bar)
            mlmi_results = self.mlmi.calculate_30m(bar)
            
            # The signals are automatically processed through the alignment engine
            # via the indicator's internal signal processing
            
            return {
                'timestamp': bar.timestamp,
                'nwrqk_results': nwrqk_results,
                'mlmi_results': mlmi_results,
                'timeframe': '30m'
            }
            
        except Exception as e:
            self.logger.error("Error processing 30m bar", error=str(e))
            return {'error': str(e)}
    
    def _analyze_synergy_patterns(self, signals: Dict[SignalType, Any], bar: BarData) -> Dict[str, Any]:
        """
        Analyze synergy patterns across different signal types
        
        Args:
            signals: Synchronized signals from all indicators
            bar: Current bar data
            
        Returns:
            Synergy analysis results
        """
        synergy_results = {
            'active_patterns': [],
            'pattern_strengths': {},
            'best_pattern': None,
            'confidence': 0.0
        }
        
        # Check each synergy pattern
        for i, pattern in enumerate(self.config.synergy_patterns):
            pattern_name = f"TYPE_{i+1}_{pattern[0].value.upper()}_{pattern[1].value.upper()}_{pattern[2].value.upper()}"
            
            # Check if all signals in pattern are available and valid
            pattern_signals = []
            pattern_strength = 0.0
            pattern_confidence = 0.0
            
            for signal_type in pattern:
                if signal_type in signals:
                    signal = signals[signal_type]
                    if signal.confidence >= self.config.min_signal_confidence:
                        pattern_signals.append(signal)
                        pattern_strength += signal.strength
                        pattern_confidence += signal.confidence
                    else:
                        # Pattern broken due to low confidence
                        break
            
            # If all signals in pattern are present and valid
            if len(pattern_signals) == len(pattern):
                avg_strength = pattern_strength / len(pattern)
                avg_confidence = pattern_confidence / len(pattern)
                
                synergy_results['active_patterns'].append(pattern_name)
                synergy_results['pattern_strengths'][pattern_name] = avg_strength
                
                # Check if this is the best pattern so far
                if avg_confidence > synergy_results['confidence']:
                    synergy_results['best_pattern'] = pattern_name
                    synergy_results['confidence'] = avg_confidence
                    
                    # Check signal alignment for synergy
                    if self._check_signal_alignment(pattern_signals):
                        self.synergy_matches += 1
        
        return synergy_results
    
    def _check_signal_alignment(self, signals: List[Any]) -> bool:
        """
        Check if signals are properly aligned for synergy
        
        Args:
            signals: List of signals to check
            
        Returns:
            True if signals are aligned, False otherwise
        """
        if len(signals) < 2:
            return False
        
        # Check if all signals have the same direction
        first_direction = signals[0].direction
        for signal in signals[1:]:
            if signal.direction != first_direction:
                return False
        
        # Check if signals are within reasonable time window
        max_time_diff = timedelta(minutes=self.config.signal_timeout_minutes)
        base_time = signals[0].timestamp
        
        for signal in signals[1:]:
            if abs(signal.timestamp - base_time) > max_time_diff:
                return False
        
        return True
    
    def _generate_trading_signals(self, synergy_result: Dict[str, Any], bar: BarData) -> Dict[str, Any]:
        """
        Generate trading signals based on synergy analysis
        
        Args:
            synergy_result: Results from synergy analysis
            bar: Current bar data
            
        Returns:
            Trading signals
        """
        signals = {
            'long_entry': False,
            'short_entry': False,
            'long_exit': False,
            'short_exit': False,
            'signal_strength': 0.0,
            'signal_confidence': 0.0,
            'pattern_used': None
        }
        
        # Generate signals based on best synergy pattern
        if synergy_result['best_pattern'] and synergy_result['confidence'] >= self.config.min_signal_confidence:
            pattern_name = synergy_result['best_pattern']
            pattern_strength = synergy_result['pattern_strengths'][pattern_name]
            
            # Get synchronized signals for direction determination
            synchronized_signals = self.signal_engine.get_synchronized_signals(bar.timestamp)
            
            # Determine signal direction based on majority vote
            bullish_count = sum(1 for s in synchronized_signals.values() if s.direction == SignalDirection.BULLISH)
            bearish_count = sum(1 for s in synchronized_signals.values() if s.direction == SignalDirection.BEARISH)
            
            if bullish_count > bearish_count and self.current_position <= 0:
                signals['long_entry'] = True
                signals['short_exit'] = True
                signals['signal_strength'] = pattern_strength
                signals['signal_confidence'] = synergy_result['confidence']
                signals['pattern_used'] = pattern_name
                
            elif bearish_count > bullish_count and self.current_position >= 0:
                signals['short_entry'] = True
                signals['long_exit'] = True
                signals['signal_strength'] = pattern_strength
                signals['signal_confidence'] = synergy_result['confidence']
                signals['pattern_used'] = pattern_name
        
        return signals
    
    def _update_strategy_state(self, trading_signals: Dict[str, Any], bar: BarData):
        """
        Update strategy state based on trading signals
        
        Args:
            trading_signals: Generated trading signals
            bar: Current bar data
        """
        # Handle position changes
        if trading_signals['long_entry'] and self.current_position != 1:
            self._execute_trade('long', bar.close, bar.timestamp, trading_signals)
            
        elif trading_signals['short_entry'] and self.current_position != -1:
            self._execute_trade('short', bar.close, bar.timestamp, trading_signals)
            
        elif trading_signals['long_exit'] and self.current_position == 1:
            self._execute_trade('close', bar.close, bar.timestamp, trading_signals)
            
        elif trading_signals['short_exit'] and self.current_position == -1:
            self._execute_trade('close', bar.close, bar.timestamp, trading_signals)
    
    def _execute_trade(self, action: str, price: float, timestamp: datetime, signals: Dict[str, Any]):
        """
        Execute a trade and update position
        
        Args:
            action: Trade action ('long', 'short', 'close')
            price: Execution price
            timestamp: Trade timestamp
            signals: Trading signals that triggered the trade
        """
        old_position = self.current_position
        
        if action == 'long':
            self.current_position = 1
            self.entry_price = price
            self.entry_timestamp = timestamp
            
        elif action == 'short':
            self.current_position = -1
            self.entry_price = price
            self.entry_timestamp = timestamp
            
        elif action == 'close':
            # Calculate P&L
            if self.entry_price > 0:
                if old_position == 1:
                    pnl = (price - self.entry_price) / self.entry_price
                else:
                    pnl = (self.entry_price - price) / self.entry_price
                
                # Record trade
                self.trade_history.append({
                    'entry_time': self.entry_timestamp,
                    'exit_time': timestamp,
                    'entry_price': self.entry_price,
                    'exit_price': price,
                    'position': old_position,
                    'pnl': pnl,
                    'pattern': signals.get('pattern_used'),
                    'confidence': signals.get('signal_confidence', 0.0)
                })
                
                self.trades_executed += 1
            
            self.current_position = 0
            self.entry_price = 0.0
            self.entry_timestamp = None
        
        self.logger.info(
            "Trade executed",
            action=action,
            price=price,
            position=self.current_position,
            pattern=signals.get('pattern_used')
        )
    
    def _cleanup_old_signals(self, current_time: datetime):
        """
        Clean up old signals to prevent memory issues
        
        Args:
            current_time: Current timestamp
        """
        cutoff_time = current_time - timedelta(hours=1)
        self.signal_engine.clear_old_signals(cutoff_time)
    
    def _get_strategy_stats(self) -> Dict[str, Any]:
        """Get current strategy statistics"""
        return {
            'trades_executed': self.trades_executed,
            'signals_processed': self.signals_processed,
            'synergy_matches': self.synergy_matches,
            'current_position': self.current_position,
            'signal_engine_stats': self.signal_engine.get_stats()
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        
        Returns:
            Performance analysis report
        """
        if not self.trade_history:
            return {'error': 'No trades executed yet'}
        
        trades_df = pd.DataFrame(self.trade_history)
        
        # Calculate basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate returns
        total_return = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Calculate other metrics
        max_win = trades_df['pnl'].max()
        max_loss = trades_df['pnl'].min()
        volatility = trades_df['pnl'].std()
        
        # Pattern analysis
        pattern_performance = trades_df.groupby('pattern')['pnl'].agg(['count', 'mean', 'sum']).to_dict()
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_win': max_win,
            'max_loss': max_loss,
            'volatility': volatility,
            'pattern_performance': pattern_performance,
            'strategy_stats': self._get_strategy_stats()
        }
    
    def reset(self):
        """Reset strategy state"""
        self.current_position = 0
        self.entry_price = 0.0
        self.entry_timestamp = None
        self.current_synergy_pattern = None
        self.synergy_state = {}
        
        self.trades_executed = 0
        self.signals_processed = 0
        self.synergy_matches = 0
        
        self.signal_history.clear()
        self.trade_history.clear()
        
        self.signal_engine.reset()
        
        self.logger.info("Strategy reset")


def create_default_strategy_config() -> StrategyConfig:
    """Create default strategy configuration"""
    return StrategyConfig(
        signal_alignment_config={
            'max_signal_age_minutes': 30,
            'confidence_decay_rate': 0.1,
            'interpolation_method': 'forward_fill'
        },
        nwrqk_config={
            'h': 8.0,
            'r': 8.0,
            'x_0': 25,
            'lag': 2,
            'smooth_colors': False,
            'max_history_length': 1000
        },
        mlmi_config={
            'num_neighbors': 200,
            'momentum_window': 20,
            'ma_quick_period': 5,
            'ma_slow_period': 20,
            'rsi_quick_period': 5,
            'rsi_slow_period': 20,
            'max_history_length': 1000
        },
        fvg_config={
            'threshold': 0.001,
            'lookback_period': 10,
            'body_multiplier': 1.5,
            'max_history_length': 1000
        },
        min_signal_confidence=0.6,
        signal_timeout_minutes=30,
        max_concurrent_signals=5
    )


def create_unified_strategy(config: StrategyConfig = None, event_bus: EventBus = None) -> UnifiedSignalStrategy:
    """
    Create a unified signal strategy with proper configuration
    
    Args:
        config: Strategy configuration (uses default if None)
        event_bus: Event bus for communication (creates new if None)
        
    Returns:
        Configured UnifiedSignalStrategy instance
    """
    if config is None:
        config = create_default_strategy_config()
    
    if event_bus is None:
        event_bus = EventBus()
    
    return UnifiedSignalStrategy(config, event_bus)