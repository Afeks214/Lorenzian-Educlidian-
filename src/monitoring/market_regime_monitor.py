#!/usr/bin/env python3
"""
AGENT 6: Market Regime Detection and Defensive Alerts
Advanced market regime detection with bear market defense system,
regime-aware position sizing alerts, and defensive trading mode activation.
"""

import asyncio
import time
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import redis
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from prometheus_client import Counter, Histogram, Gauge, Summary

# Import existing monitoring components
from .enhanced_alerting import EnhancedAlertingSystem, EnhancedAlert, AlertPriority, AlertStatus
from .real_time_performance_monitor import MarketRegime, DefenseMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Market regime metrics
REGIME_DETECTION_ACCURACY = Gauge(
    'regime_detection_accuracy',
    'Market regime detection accuracy',
    ['regime_type', 'timeframe']
)

REGIME_TRANSITIONS = Counter(
    'regime_transitions_total',
    'Total regime transitions',
    ['from_regime', 'to_regime']
)

REGIME_CONFIDENCE_SCORE = Gauge(
    'regime_confidence_score',
    'Market regime confidence score',
    ['regime_type', 'detection_method']
)

DEFENSIVE_ACTIONS_TRIGGERED = Counter(
    'defensive_actions_triggered_total',
    'Total defensive actions triggered',
    ['action_type', 'trigger_reason']
)

POSITION_SIZE_ADJUSTMENTS = Counter(
    'position_size_adjustments_total',
    'Total position size adjustments',
    ['adjustment_type', 'regime', 'reason']
)

VOLATILITY_ALERTS = Counter(
    'volatility_alerts_total',
    'Total volatility alerts',
    ['alert_type', 'severity']
)

CORRELATION_BREAKDOWN_ALERTS = Counter(
    'correlation_breakdown_alerts_total',
    'Total correlation breakdown alerts',
    ['asset_pair', 'severity']
)

class RegimeDetectionMethod(Enum):
    """Market regime detection methods."""
    VOLATILITY_CLUSTERING = "volatility_clustering"
    TREND_ANALYSIS = "trend_analysis"
    VOLUME_ANALYSIS = "volume_analysis"
    CORRELATION_ANALYSIS = "correlation_analysis"
    MACHINE_LEARNING = "machine_learning"
    COMPOSITE = "composite"

class DefensiveAction(Enum):
    """Defensive actions that can be triggered."""
    REDUCE_POSITION_SIZE = "reduce_position_size"
    INCREASE_CASH_BUFFER = "increase_cash_buffer"
    TIGHTEN_STOP_LOSSES = "tighten_stop_losses"
    INCREASE_DIVERSIFICATION = "increase_diversification"
    ACTIVATE_HEDGING = "activate_hedging"
    PAUSE_TRADING = "pause_trading"
    EMERGENCY_LIQUIDATION = "emergency_liquidation"

class VolatilityRegime(Enum):
    """Volatility regime types."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class MarketRegimeSignal:
    """Market regime detection signal."""
    regime: MarketRegime
    confidence: float
    method: RegimeDetectionMethod
    timestamp: datetime
    supporting_evidence: Dict[str, Any]
    volatility_regime: VolatilityRegime
    trend_strength: float
    correlation_breakdown: bool = False
    
@dataclass
class DefensiveParameters:
    """Defensive trading parameters."""
    max_position_size: float
    cash_buffer_target: float
    stop_loss_tightness: float
    diversification_threshold: float
    hedging_ratio: float
    trading_pause_duration: timedelta
    
@dataclass
class RegimeTransition:
    """Market regime transition record."""
    from_regime: MarketRegime
    to_regime: MarketRegime
    timestamp: datetime
    confidence: float
    triggers: List[str]
    duration_in_previous: timedelta

class VolatilityAnalyzer:
    """Analyzes market volatility patterns."""
    
    def __init__(self, lookback_period: int = 252):
        self.lookback_period = lookback_period
        self.returns_history = deque(maxlen=lookback_period)
        self.volatility_history = deque(maxlen=lookback_period)
        self.volatility_regimes = deque(maxlen=100)
        
    def add_return(self, return_value: float):
        """Add new return to analysis."""
        self.returns_history.append(return_value)
        
        if len(self.returns_history) >= 20:
            # Calculate rolling volatility
            recent_returns = np.array(list(self.returns_history)[-20:])
            volatility = np.std(recent_returns) * np.sqrt(252)  # Annualized
            self.volatility_history.append(volatility)
            
            # Classify volatility regime
            regime = self._classify_volatility_regime(volatility)
            self.volatility_regimes.append(regime)
            
    def _classify_volatility_regime(self, current_vol: float) -> VolatilityRegime:
        """Classify current volatility regime."""
        if len(self.volatility_history) < 50:
            return VolatilityRegime.NORMAL
            
        vol_history = np.array(list(self.volatility_history))
        
        # Calculate percentiles
        p25 = np.percentile(vol_history, 25)
        p50 = np.percentile(vol_history, 50)
        p75 = np.percentile(vol_history, 75)
        p90 = np.percentile(vol_history, 90)
        
        if current_vol > p90:
            return VolatilityRegime.EXTREME
        elif current_vol > p75:
            return VolatilityRegime.HIGH
        elif current_vol < p25:
            return VolatilityRegime.LOW
        else:
            return VolatilityRegime.NORMAL
            
    def get_volatility_persistence(self) -> float:
        """Calculate volatility persistence (clustering)."""
        if len(self.volatility_history) < 50:
            return 0.5
            
        vol_changes = np.diff(np.array(list(self.volatility_history)[-50:]))
        # Calculate autocorrelation at lag 1
        if len(vol_changes) > 1:
            correlation = np.corrcoef(vol_changes[:-1], vol_changes[1:])[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0.5
        return 0.5
        
    def detect_volatility_breakout(self) -> bool:
        """Detect volatility breakout."""
        if len(self.volatility_history) < 10:
            return False
            
        recent_vol = list(self.volatility_history)[-5:]
        historical_vol = list(self.volatility_history)[:-5]
        
        if len(historical_vol) < 20:
            return False
            
        recent_mean = np.mean(recent_vol)
        historical_mean = np.mean(historical_vol)
        historical_std = np.std(historical_vol)
        
        # Check if recent volatility is significantly higher
        z_score = (recent_mean - historical_mean) / historical_std
        return z_score > 2.0

class TrendAnalyzer:
    """Analyzes market trends."""
    
    def __init__(self, lookback_period: int = 100):
        self.lookback_period = lookback_period
        self.price_history = deque(maxlen=lookback_period)
        self.trend_history = deque(maxlen=100)
        
    def add_price(self, price: float):
        """Add new price to analysis."""
        self.price_history.append(price)
        
        if len(self.price_history) >= 20:
            trend_strength = self._calculate_trend_strength()
            self.trend_history.append(trend_strength)
            
    def _calculate_trend_strength(self) -> float:
        """Calculate trend strength using linear regression."""
        if len(self.price_history) < 20:
            return 0.0
            
        prices = np.array(list(self.price_history)[-20:])
        x = np.arange(len(prices))
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
        
        # Normalize slope by price level
        normalized_slope = slope / np.mean(prices)
        
        # Weight by R-squared (goodness of fit)
        trend_strength = normalized_slope * (r_value ** 2)
        
        return trend_strength
        
    def detect_trend_reversal(self) -> bool:
        """Detect trend reversal."""
        if len(self.trend_history) < 10:
            return False
            
        recent_trends = list(self.trend_history)[-5:]
        previous_trends = list(self.trend_history)[-10:-5]
        
        recent_mean = np.mean(recent_trends)
        previous_mean = np.mean(previous_trends)
        
        # Check for sign change and magnitude
        return (np.sign(recent_mean) != np.sign(previous_mean) and 
                abs(recent_mean - previous_mean) > 0.01)

class CorrelationAnalyzer:
    """Analyzes asset correlations."""
    
    def __init__(self, lookback_period: int = 50):
        self.lookback_period = lookback_period
        self.asset_returns = defaultdict(lambda: deque(maxlen=lookback_period))
        self.correlation_history = deque(maxlen=100)
        
    def add_asset_return(self, asset: str, return_value: float):
        """Add asset return."""
        self.asset_returns[asset].append(return_value)
        
        if len(self.asset_returns) >= 2:
            self._update_correlations()
            
    def _update_correlations(self):
        """Update correlation matrix."""
        assets = list(self.asset_returns.keys())
        
        if len(assets) < 2:
            return
            
        # Get minimum common length
        min_length = min(len(returns) for returns in self.asset_returns.values())
        
        if min_length < 20:
            return
            
        # Create correlation matrix
        correlation_matrix = {}
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i < j:
                    returns1 = np.array(list(self.asset_returns[asset1])[-min_length:])
                    returns2 = np.array(list(self.asset_returns[asset2])[-min_length:])
                    
                    correlation = np.corrcoef(returns1, returns2)[0, 1]
                    correlation_matrix[f"{asset1}_{asset2}"] = correlation
                    
        self.correlation_history.append(correlation_matrix)
        
    def detect_correlation_breakdown(self) -> List[Tuple[str, float]]:
        """Detect correlation breakdown."""
        if len(self.correlation_history) < 10:
            return []
            
        recent_corr = self.correlation_history[-1]
        historical_corr = list(self.correlation_history)[:-5]
        
        breakdowns = []
        
        for asset_pair, recent_value in recent_corr.items():
            historical_values = [corr.get(asset_pair, 0.0) for corr in historical_corr]
            historical_mean = np.mean(historical_values)
            
            # Check for significant decrease in correlation
            if abs(recent_value - historical_mean) > 0.3:
                breakdowns.append((asset_pair, recent_value - historical_mean))
                
        return breakdowns

class MarketRegimeDetector:
    """Advanced market regime detection system."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.volatility_analyzer = VolatilityAnalyzer()
        self.trend_analyzer = TrendAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        
        # Current state
        self.current_regime = MarketRegime.SIDEWAYS
        self.current_confidence = 0.5
        self.regime_history = deque(maxlen=100)
        self.transition_history = deque(maxlen=50)
        
        # ML model (simplified)
        self.ml_scaler = StandardScaler()
        self.ml_model = None
        self.feature_history = deque(maxlen=1000)
        
    async def update_market_data(self, price: float, volume: float, 
                               asset_returns: Dict[str, float]):
        """Update market data for regime detection."""
        # Update analyzers
        if len(asset_returns) > 0:
            main_return = list(asset_returns.values())[0]
            self.volatility_analyzer.add_return(main_return)
            
        self.trend_analyzer.add_price(price)
        
        for asset, return_value in asset_returns.items():
            self.correlation_analyzer.add_asset_return(asset, return_value)
            
        # Perform regime detection
        await self._detect_regime()
        
    async def _detect_regime(self):
        """Perform comprehensive regime detection."""
        # Collect signals from different methods
        signals = []
        
        # Volatility-based detection
        vol_signal = self._detect_volatility_regime()
        if vol_signal:
            signals.append(vol_signal)
            
        # Trend-based detection
        trend_signal = self._detect_trend_regime()
        if trend_signal:
            signals.append(trend_signal)
            
        # Correlation-based detection
        corr_signal = self._detect_correlation_regime()
        if corr_signal:
            signals.append(corr_signal)
            
        # Composite detection
        if signals:
            composite_signal = self._create_composite_signal(signals)
            
            # Check for regime change
            if composite_signal.regime != self.current_regime:
                await self._handle_regime_transition(composite_signal)
            else:
                # Update confidence
                self.current_confidence = composite_signal.confidence
                
        # Update metrics
        REGIME_CONFIDENCE_SCORE.labels(
            regime_type=self.current_regime.value,
            detection_method="composite"
        ).set(self.current_confidence)
        
    def _detect_volatility_regime(self) -> Optional[MarketRegimeSignal]:
        """Detect regime based on volatility."""
        if len(self.volatility_analyzer.volatility_regimes) < 10:
            return None
            
        current_vol_regime = self.volatility_analyzer.volatility_regimes[-1]
        vol_persistence = self.volatility_analyzer.get_volatility_persistence()
        vol_breakout = self.volatility_analyzer.detect_volatility_breakout()
        
        confidence = 0.5
        regime = MarketRegime.SIDEWAYS
        
        if current_vol_regime == VolatilityRegime.EXTREME:
            regime = MarketRegime.CRISIS
            confidence = 0.8
        elif current_vol_regime == VolatilityRegime.HIGH:
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = 0.7
        elif current_vol_regime == VolatilityRegime.LOW:
            regime = MarketRegime.LOW_VOLATILITY
            confidence = 0.6
            
        # Adjust confidence based on persistence
        confidence *= (0.5 + vol_persistence * 0.5)
        
        return MarketRegimeSignal(
            regime=regime,
            confidence=confidence,
            method=RegimeDetectionMethod.VOLATILITY_CLUSTERING,
            timestamp=datetime.utcnow(),
            supporting_evidence={
                'volatility_regime': current_vol_regime.value,
                'volatility_persistence': vol_persistence,
                'volatility_breakout': vol_breakout
            },
            volatility_regime=current_vol_regime,
            trend_strength=0.0
        )
        
    def _detect_trend_regime(self) -> Optional[MarketRegimeSignal]:
        """Detect regime based on trend analysis."""
        if len(self.trend_analyzer.trend_history) < 10:
            return None
            
        current_trend = self.trend_analyzer.trend_history[-1]
        trend_reversal = self.trend_analyzer.detect_trend_reversal()
        
        confidence = 0.5
        regime = MarketRegime.SIDEWAYS
        
        if current_trend > 0.02:
            regime = MarketRegime.BULL
            confidence = min(abs(current_trend) * 25, 0.9)
        elif current_trend < -0.02:
            regime = MarketRegime.BEAR
            confidence = min(abs(current_trend) * 25, 0.9)
            
        # Reduce confidence if trend reversal detected
        if trend_reversal:
            confidence *= 0.7
            
        return MarketRegimeSignal(
            regime=regime,
            confidence=confidence,
            method=RegimeDetectionMethod.TREND_ANALYSIS,
            timestamp=datetime.utcnow(),
            supporting_evidence={
                'trend_strength': current_trend,
                'trend_reversal': trend_reversal
            },
            volatility_regime=VolatilityRegime.NORMAL,
            trend_strength=current_trend
        )
        
    def _detect_correlation_regime(self) -> Optional[MarketRegimeSignal]:
        """Detect regime based on correlation analysis."""
        correlation_breakdowns = self.correlation_analyzer.detect_correlation_breakdown()
        
        if not correlation_breakdowns:
            return None
            
        # Strong correlation breakdown suggests crisis
        max_breakdown = max(abs(breakdown[1]) for breakdown in correlation_breakdowns)
        
        confidence = min(max_breakdown * 2, 0.8)
        regime = MarketRegime.CRISIS if max_breakdown > 0.4 else MarketRegime.HIGH_VOLATILITY
        
        return MarketRegimeSignal(
            regime=regime,
            confidence=confidence,
            method=RegimeDetectionMethod.CORRELATION_ANALYSIS,
            timestamp=datetime.utcnow(),
            supporting_evidence={
                'correlation_breakdowns': correlation_breakdowns,
                'max_breakdown': max_breakdown
            },
            volatility_regime=VolatilityRegime.HIGH,
            trend_strength=0.0,
            correlation_breakdown=True
        )
        
    def _create_composite_signal(self, signals: List[MarketRegimeSignal]) -> MarketRegimeSignal:
        """Create composite signal from multiple detection methods."""
        if not signals:
            return MarketRegimeSignal(
                regime=self.current_regime,
                confidence=self.current_confidence,
                method=RegimeDetectionMethod.COMPOSITE,
                timestamp=datetime.utcnow(),
                supporting_evidence={},
                volatility_regime=VolatilityRegime.NORMAL,
                trend_strength=0.0
            )
            
        # Weight signals by confidence
        regime_votes = defaultdict(float)
        total_confidence = 0.0
        
        for signal in signals:
            regime_votes[signal.regime] += signal.confidence
            total_confidence += signal.confidence
            
        # Find regime with highest weighted vote
        best_regime = max(regime_votes.items(), key=lambda x: x[1])[0]
        composite_confidence = regime_votes[best_regime] / total_confidence
        
        # Aggregate supporting evidence
        composite_evidence = {}
        for signal in signals:
            composite_evidence[signal.method.value] = signal.supporting_evidence
            
        # Get volatility and trend from relevant signals
        volatility_regime = VolatilityRegime.NORMAL
        trend_strength = 0.0
        correlation_breakdown = False
        
        for signal in signals:
            if signal.method == RegimeDetectionMethod.VOLATILITY_CLUSTERING:
                volatility_regime = signal.volatility_regime
            elif signal.method == RegimeDetectionMethod.TREND_ANALYSIS:
                trend_strength = signal.trend_strength
            elif signal.method == RegimeDetectionMethod.CORRELATION_ANALYSIS:
                correlation_breakdown = signal.correlation_breakdown
                
        return MarketRegimeSignal(
            regime=best_regime,
            confidence=composite_confidence,
            method=RegimeDetectionMethod.COMPOSITE,
            timestamp=datetime.utcnow(),
            supporting_evidence=composite_evidence,
            volatility_regime=volatility_regime,
            trend_strength=trend_strength,
            correlation_breakdown=correlation_breakdown
        )
        
    async def _handle_regime_transition(self, new_signal: MarketRegimeSignal):
        """Handle regime transition."""
        old_regime = self.current_regime
        new_regime = new_signal.regime
        
        # Calculate time in previous regime
        duration_in_previous = timedelta(seconds=0)
        if self.regime_history:
            last_regime_time = self.regime_history[-1]['timestamp']
            duration_in_previous = datetime.utcnow() - last_regime_time
            
        # Create transition record
        transition = RegimeTransition(
            from_regime=old_regime,
            to_regime=new_regime,
            timestamp=datetime.utcnow(),
            confidence=new_signal.confidence,
            triggers=[new_signal.method.value],
            duration_in_previous=duration_in_previous
        )
        
        # Update state
        self.current_regime = new_regime
        self.current_confidence = new_signal.confidence
        
        # Record transition
        self.regime_history.append({
            'regime': new_regime.value,
            'confidence': new_signal.confidence,
            'timestamp': datetime.utcnow()
        })
        self.transition_history.append(transition)
        
        # Update metrics
        REGIME_TRANSITIONS.labels(
            from_regime=old_regime.value,
            to_regime=new_regime.value
        ).inc()
        
        # Store in Redis
        await self.redis_client.setex(
            'market_regime:current',
            3600,  # 1 hour TTL
            json.dumps({
                'regime': new_regime.value,
                'confidence': new_signal.confidence,
                'timestamp': datetime.utcnow().isoformat(),
                'transition_from': old_regime.value,
                'supporting_evidence': new_signal.supporting_evidence
            })
        )
        
        logger.warning(f"Market regime transition: {old_regime.value} -> {new_regime.value} (confidence: {new_signal.confidence:.2f})")

class BearMarketDefenseSystem:
    """Enhanced bear market defense system."""
    
    def __init__(self, alerting_system: EnhancedAlertingSystem):
        self.alerting_system = alerting_system
        self.current_mode = DefenseMode.NORMAL
        self.defensive_params = self._get_defensive_parameters()
        self.action_history = deque(maxlen=100)
        
    def _get_defensive_parameters(self) -> Dict[DefenseMode, DefensiveParameters]:
        """Get defensive parameters for each mode."""
        return {
            DefenseMode.NORMAL: DefensiveParameters(
                max_position_size=1.0,
                cash_buffer_target=0.05,
                stop_loss_tightness=1.0,
                diversification_threshold=0.8,
                hedging_ratio=0.0,
                trading_pause_duration=timedelta(minutes=0)
            ),
            DefenseMode.DEFENSIVE: DefensiveParameters(
                max_position_size=0.7,
                cash_buffer_target=0.15,
                stop_loss_tightness=0.8,
                diversification_threshold=0.6,
                hedging_ratio=0.2,
                trading_pause_duration=timedelta(minutes=5)
            ),
            DefenseMode.ULTRA_DEFENSIVE: DefensiveParameters(
                max_position_size=0.4,
                cash_buffer_target=0.30,
                stop_loss_tightness=0.6,
                diversification_threshold=0.4,
                hedging_ratio=0.5,
                trading_pause_duration=timedelta(minutes=15)
            ),
            DefenseMode.SHUTDOWN: DefensiveParameters(
                max_position_size=0.0,
                cash_buffer_target=0.50,
                stop_loss_tightness=0.0,
                diversification_threshold=0.0,
                hedging_ratio=0.8,
                trading_pause_duration=timedelta(hours=1)
            )
        }
        
    async def evaluate_defense_mode(self, regime_signal: MarketRegimeSignal, 
                                  current_drawdown: float, portfolio_metrics: Dict[str, float]) -> DefenseMode:
        """Evaluate and potentially change defense mode."""
        new_mode = self._determine_defense_mode(regime_signal, current_drawdown, portfolio_metrics)
        
        if new_mode != self.current_mode:
            await self._activate_defense_mode(new_mode, regime_signal, current_drawdown, portfolio_metrics)
            
        return self.current_mode
        
    def _determine_defense_mode(self, regime_signal: MarketRegimeSignal, 
                              current_drawdown: float, portfolio_metrics: Dict[str, float]) -> DefenseMode:
        """Determine appropriate defense mode."""
        # Crisis conditions
        if (regime_signal.regime == MarketRegime.CRISIS or 
            current_drawdown > 0.15 or 
            portfolio_metrics.get('volatility', 0) > 0.05):
            return DefenseMode.SHUTDOWN
            
        # Ultra-defensive conditions
        elif (regime_signal.regime == MarketRegime.BEAR and regime_signal.confidence > 0.7 or
              current_drawdown > 0.08 or
              portfolio_metrics.get('volatility', 0) > 0.03):
            return DefenseMode.ULTRA_DEFENSIVE
            
        # Defensive conditions
        elif (regime_signal.regime == MarketRegime.BEAR and regime_signal.confidence > 0.5 or
              current_drawdown > 0.05 or
              portfolio_metrics.get('volatility', 0) > 0.02 or
              regime_signal.volatility_regime == VolatilityRegime.HIGH):
            return DefenseMode.DEFENSIVE
            
        # Normal conditions
        else:
            return DefenseMode.NORMAL
            
    async def _activate_defense_mode(self, new_mode: DefenseMode, regime_signal: MarketRegimeSignal,
                                   current_drawdown: float, portfolio_metrics: Dict[str, float]):
        """Activate new defense mode."""
        old_mode = self.current_mode
        self.current_mode = new_mode
        
        # Record action
        action_record = {
            'timestamp': datetime.utcnow(),
            'old_mode': old_mode.value,
            'new_mode': new_mode.value,
            'trigger_regime': regime_signal.regime.value,
            'confidence': regime_signal.confidence,
            'drawdown': current_drawdown,
            'portfolio_metrics': portfolio_metrics
        }
        
        self.action_history.append(action_record)
        
        # Execute defensive actions
        await self._execute_defensive_actions(new_mode, regime_signal, current_drawdown)
        
        # Generate alerts
        await self._generate_defense_alerts(new_mode, old_mode, regime_signal, current_drawdown)
        
        logger.warning(f"Defense mode activated: {old_mode.value} -> {new_mode.value}")
        
    async def _execute_defensive_actions(self, mode: DefenseMode, regime_signal: MarketRegimeSignal,
                                       current_drawdown: float):
        """Execute defensive actions for the new mode."""
        params = self.defensive_params[mode]
        
        # Trigger appropriate defensive actions
        if mode == DefenseMode.SHUTDOWN:
            await self._trigger_action(DefensiveAction.EMERGENCY_LIQUIDATION, params)
            await self._trigger_action(DefensiveAction.PAUSE_TRADING, params)
            
        elif mode == DefenseMode.ULTRA_DEFENSIVE:
            await self._trigger_action(DefensiveAction.REDUCE_POSITION_SIZE, params)
            await self._trigger_action(DefensiveAction.INCREASE_CASH_BUFFER, params)
            await self._trigger_action(DefensiveAction.ACTIVATE_HEDGING, params)
            await self._trigger_action(DefensiveAction.TIGHTEN_STOP_LOSSES, params)
            
        elif mode == DefenseMode.DEFENSIVE:
            await self._trigger_action(DefensiveAction.REDUCE_POSITION_SIZE, params)
            await self._trigger_action(DefensiveAction.INCREASE_CASH_BUFFER, params)
            await self._trigger_action(DefensiveAction.TIGHTEN_STOP_LOSSES, params)
            
    async def _trigger_action(self, action: DefensiveAction, params: DefensiveParameters):
        """Trigger a specific defensive action."""
        DEFENSIVE_ACTIONS_TRIGGERED.labels(
            action_type=action.value,
            trigger_reason=self.current_mode.value
        ).inc()
        
        logger.info(f"Triggered defensive action: {action.value}")
        
    async def _generate_defense_alerts(self, new_mode: DefenseMode, old_mode: DefenseMode,
                                     regime_signal: MarketRegimeSignal, current_drawdown: float):
        """Generate defense mode alerts."""
        priority = AlertPriority.CRITICAL if new_mode == DefenseMode.SHUTDOWN else AlertPriority.HIGH
        
        alert = EnhancedAlert(
            id=f"defense_mode_{new_mode.value}_{int(time.time())}",
            timestamp=datetime.utcnow(),
            priority=priority,
            status=AlertStatus.ACTIVE,
            source="bear_market_defense",
            alert_type="defense_mode_change",
            title=f"Defense Mode Activated: {new_mode.value.upper()}",
            message=f"Bear market defense mode changed from {old_mode.value} to {new_mode.value}. "
                   f"Market regime: {regime_signal.regime.value} (confidence: {regime_signal.confidence:.2f}). "
                   f"Current drawdown: {current_drawdown:.2%}",
            metrics={
                'old_mode': old_mode.value,
                'new_mode': new_mode.value,
                'market_regime': regime_signal.regime.value,
                'regime_confidence': regime_signal.confidence,
                'drawdown': current_drawdown,
                'max_position_size': self.defensive_params[new_mode].max_position_size,
                'cash_buffer_target': self.defensive_params[new_mode].cash_buffer_target
            },
            tags={f"defense_mode:{new_mode.value}", f"regime:{regime_signal.regime.value}"}
        )
        
        await self.alerting_system.process_alert(alert)

class MarketRegimeMonitor:
    """Main market regime monitoring system."""
    
    def __init__(self, redis_client: redis.Redis, alerting_system: EnhancedAlertingSystem):
        self.redis_client = redis_client
        self.alerting_system = alerting_system
        self.regime_detector = MarketRegimeDetector(redis_client)
        self.defense_system = BearMarketDefenseSystem(alerting_system)
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task = None
        
    async def start_monitoring(self):
        """Start market regime monitoring."""
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Market regime monitoring started")
        
    async def stop_monitoring(self):
        """Stop market regime monitoring."""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
        logger.info("Market regime monitoring stopped")
        
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Get market data
                market_data = await self._get_market_data()
                
                if market_data:
                    # Update regime detection
                    await self.regime_detector.update_market_data(
                        price=market_data['price'],
                        volume=market_data['volume'],
                        asset_returns=market_data['asset_returns']
                    )
                    
                    # Get current regime signal
                    current_signal = MarketRegimeSignal(
                        regime=self.regime_detector.current_regime,
                        confidence=self.regime_detector.current_confidence,
                        method=RegimeDetectionMethod.COMPOSITE,
                        timestamp=datetime.utcnow(),
                        supporting_evidence={},
                        volatility_regime=VolatilityRegime.NORMAL,
                        trend_strength=0.0
                    )
                    
                    # Evaluate defense mode
                    current_drawdown = market_data.get('drawdown', 0.0)
                    portfolio_metrics = market_data.get('portfolio_metrics', {})
                    
                    await self.defense_system.evaluate_defense_mode(
                        current_signal, current_drawdown, portfolio_metrics
                    )
                    
                await asyncio.sleep(60)  # 1 minute intervals
                
            except Exception as e:
                logger.error(f"Error in market regime monitoring: {e}")
                await asyncio.sleep(120)
                
    async def _get_market_data(self) -> Optional[Dict[str, Any]]:
        """Get current market data from Redis."""
        try:
            # Get market price
            price_data = await self.redis_client.get('market:price')
            volume_data = await self.redis_client.get('market:volume')
            
            # Get asset returns
            asset_returns = {}
            for asset in ['BTC', 'ETH', 'SPY']:  # Example assets
                return_data = await self.redis_client.get(f'market:returns:{asset}')
                if return_data:
                    asset_returns[asset] = float(return_data)
                    
            # Get portfolio metrics
            drawdown_data = await self.redis_client.get('portfolio:drawdown')
            volatility_data = await self.redis_client.get('portfolio:volatility')
            
            if price_data and volume_data:
                return {
                    'price': float(price_data),
                    'volume': float(volume_data),
                    'asset_returns': asset_returns,
                    'drawdown': float(drawdown_data) if drawdown_data else 0.0,
                    'portfolio_metrics': {
                        'volatility': float(volatility_data) if volatility_data else 0.0
                    }
                }
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None
            
    async def get_regime_status(self) -> Dict[str, Any]:
        """Get current regime monitoring status."""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'current_regime': self.regime_detector.current_regime.value,
            'confidence': self.regime_detector.current_confidence,
            'defense_mode': self.defense_system.current_mode.value,
            'regime_history': [
                {
                    'regime': record['regime'],
                    'confidence': record['confidence'],
                    'timestamp': record['timestamp'].isoformat()
                }
                for record in list(self.regime_detector.regime_history)[-10:]
            ],
            'transition_history': [
                {
                    'from_regime': transition.from_regime.value,
                    'to_regime': transition.to_regime.value,
                    'timestamp': transition.timestamp.isoformat(),
                    'confidence': transition.confidence
                }
                for transition in list(self.regime_detector.transition_history)[-5:]
            ],
            'defensive_actions': [
                {
                    'timestamp': action['timestamp'].isoformat(),
                    'old_mode': action['old_mode'],
                    'new_mode': action['new_mode'],
                    'trigger_regime': action['trigger_regime'],
                    'confidence': action['confidence']
                }
                for action in list(self.defense_system.action_history)[-5:]
            ]
        }

# Factory function
def create_market_regime_monitor(redis_client: redis.Redis, 
                               alerting_system: EnhancedAlertingSystem) -> MarketRegimeMonitor:
    """Create market regime monitor instance."""
    return MarketRegimeMonitor(redis_client, alerting_system)

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Setup
        redis_client = redis.Redis(host='localhost', port=6379, db=0)
        alerting_system = EnhancedAlertingSystem(redis_client)
        
        # Create monitor
        monitor = create_market_regime_monitor(redis_client, alerting_system)
        
        # Start monitoring
        await monitor.start_monitoring()
        
    asyncio.run(main())