"""
Graceful degradation mechanisms for different failure scenarios.

Implements sophisticated degradation strategies that allow the trading system
to continue operating with reduced functionality when components fail,
maintaining core business continuity while preserving system stability.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
from contextlib import contextmanager

from .base_exceptions import (
    BaseGrandModelError, ErrorSeverity, ErrorCategory, ErrorContext
)
from .agent_error_decorators import AgentType
from .dependency_circuit_breakers import DependencyType

logger = logging.getLogger(__name__)


class DegradationLevel(Enum):
    """Levels of system degradation"""
    NORMAL = "normal"           # Full functionality
    MINOR = "minor"             # Minor feature limitations
    MODERATE = "moderate"       # Significant feature limitations
    MAJOR = "major"             # Core functionality only
    CRITICAL = "critical"       # Emergency operations only
    SHUTDOWN = "shutdown"       # System shutdown required


class DegradationTrigger(Enum):
    """Triggers for degradation"""
    ERROR_RATE_THRESHOLD = "error_rate_threshold"
    DEPENDENCY_FAILURE = "dependency_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MANUAL_TRIGGER = "manual_trigger"
    SECURITY_INCIDENT = "security_incident"
    CASCADING_FAILURE = "cascading_failure"


@dataclass
class DegradationConfig:
    """Configuration for degradation behavior"""
    agent_type: AgentType
    degradation_triggers: Dict[DegradationTrigger, float] = field(default_factory=dict)
    recovery_thresholds: Dict[DegradationLevel, float] = field(default_factory=dict)
    feature_priorities: Dict[str, int] = field(default_factory=dict)
    essential_features: List[str] = field(default_factory=list)
    fallback_values: Dict[str, Any] = field(default_factory=dict)
    monitoring_interval: float = 10.0
    auto_recovery: bool = True
    max_degradation_time: float = 1800.0  # 30 minutes


@dataclass
class DegradationState:
    """Current degradation state"""
    level: DegradationLevel = DegradationLevel.NORMAL
    trigger: Optional[DegradationTrigger] = None
    start_time: Optional[datetime] = None
    disabled_features: List[str] = field(default_factory=list)
    active_fallbacks: Dict[str, Any] = field(default_factory=dict)
    degradation_reason: Optional[str] = None
    recovery_in_progress: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'level': self.level.value,
            'trigger': self.trigger.value if self.trigger else None,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'disabled_features': self.disabled_features,
            'active_fallbacks': self.active_fallbacks,
            'degradation_reason': self.degradation_reason,
            'recovery_in_progress': self.recovery_in_progress
        }


class BaseGracefulDegradation(ABC):
    """Base class for graceful degradation strategies"""
    
    def __init__(self, config: DegradationConfig):
        self.config = config
        self.current_state = DegradationState()
        self.degradation_history: List[DegradationState] = []
        self.feature_registry: Dict[str, Callable] = {}
        self.fallback_registry: Dict[str, Callable] = {}
        self.lock = threading.RLock()
        self.monitoring_thread = None
        self.monitoring_running = False
        
        # Initialize default configurations
        self._initialize_default_config()
        
        # Start monitoring
        self._start_monitoring()
    
    def _initialize_default_config(self):
        """Initialize default configuration based on agent type"""
        if not self.config.degradation_triggers:
            self.config.degradation_triggers = {
                DegradationTrigger.ERROR_RATE_THRESHOLD: 0.1,  # 10% error rate
                DegradationTrigger.DEPENDENCY_FAILURE: 1.0,
                DegradationTrigger.RESOURCE_EXHAUSTION: 0.8,    # 80% resource usage
                DegradationTrigger.PERFORMANCE_DEGRADATION: 2.0  # 2x normal response time
            }
        
        if not self.config.recovery_thresholds:
            self.config.recovery_thresholds = {
                DegradationLevel.CRITICAL: 0.95,
                DegradationLevel.MAJOR: 0.8,
                DegradationLevel.MODERATE: 0.6,
                DegradationLevel.MINOR: 0.4
            }
    
    def _start_monitoring(self):
        """Start degradation monitoring"""
        self.monitoring_running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_running:
            try:
                time.sleep(self.config.monitoring_interval)
                self._check_degradation_triggers()
                self._check_recovery_conditions()
            except Exception as e:
                logger.error(f"Error in degradation monitoring loop: {e}")
    
    @abstractmethod
    def _check_degradation_triggers(self):
        """Check if degradation should be triggered"""
        pass
    
    @abstractmethod
    def _check_recovery_conditions(self):
        """Check if recovery should be attempted"""
        pass
    
    @abstractmethod
    def _get_system_health_score(self) -> float:
        """Get current system health score (0.0 to 1.0)"""
        pass
    
    def register_feature(self, name: str, feature_func: Callable, priority: int = 0):
        """Register a feature that can be degraded"""
        with self.lock:
            self.feature_registry[name] = feature_func
            self.config.feature_priorities[name] = priority
        
        logger.info(f"Feature '{name}' registered with priority {priority}")
    
    def register_fallback(self, feature_name: str, fallback_func: Callable):
        """Register a fallback for a feature"""
        with self.lock:
            self.fallback_registry[feature_name] = fallback_func
        
        logger.info(f"Fallback registered for feature '{feature_name}'")
    
    def trigger_degradation(self, level: DegradationLevel, trigger: DegradationTrigger,
                          reason: Optional[str] = None):
        """Trigger degradation to specified level"""
        with self.lock:
            if level.value >= self.current_state.level.value:
                return  # Already at this level or higher
            
            logger.warning(f"Triggering degradation to {level.value} due to {trigger.value}")
            
            # Record current state in history
            if self.current_state.level != DegradationLevel.NORMAL:
                self.degradation_history.append(self.current_state)
            
            # Update state
            self.current_state = DegradationState(
                level=level,
                trigger=trigger,
                start_time=datetime.utcnow(),
                degradation_reason=reason
            )
            
            # Apply degradation
            self._apply_degradation(level)
    
    def _apply_degradation(self, level: DegradationLevel):
        """Apply degradation at specified level"""
        features_to_disable = self._get_features_to_disable(level)
        
        for feature_name in features_to_disable:
            self._disable_feature(feature_name)
        
        logger.info(f"Degradation applied: {len(features_to_disable)} features disabled")
    
    def _get_features_to_disable(self, level: DegradationLevel) -> List[str]:
        """Get features to disable at degradation level"""
        features_to_disable = []
        
        # Sort features by priority (lower priority disabled first)
        sorted_features = sorted(
            self.config.feature_priorities.items(),
            key=lambda x: x[1]
        )
        
        # Determine how many features to disable based on level
        if level == DegradationLevel.MINOR:
            features_to_disable = [f for f, p in sorted_features if p < 3]
        elif level == DegradationLevel.MODERATE:
            features_to_disable = [f for f, p in sorted_features if p < 6]
        elif level == DegradationLevel.MAJOR:
            features_to_disable = [f for f, p in sorted_features if p < 9]
        elif level == DegradationLevel.CRITICAL:
            # Keep only essential features
            features_to_disable = [f for f in sorted_features if f not in self.config.essential_features]
        
        return features_to_disable
    
    def _disable_feature(self, feature_name: str):
        """Disable a specific feature"""
        if feature_name in self.current_state.disabled_features:
            return
        
        self.current_state.disabled_features.append(feature_name)
        
        # Activate fallback if available
        if feature_name in self.fallback_registry:
            self.current_state.active_fallbacks[feature_name] = self.fallback_registry[feature_name]
        
        logger.info(f"Feature '{feature_name}' disabled")
    
    def _enable_feature(self, feature_name: str):
        """Enable a previously disabled feature"""
        if feature_name not in self.current_state.disabled_features:
            return
        
        self.current_state.disabled_features.remove(feature_name)
        
        # Deactivate fallback
        if feature_name in self.current_state.active_fallbacks:
            del self.current_state.active_fallbacks[feature_name]
        
        logger.info(f"Feature '{feature_name}' enabled")
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled"""
        return feature_name not in self.current_state.disabled_features
    
    def execute_feature(self, feature_name: str, *args, **kwargs) -> Any:
        """Execute a feature with fallback support"""
        with self.lock:
            if self.is_feature_enabled(feature_name):
                # Execute normal feature
                if feature_name in self.feature_registry:
                    return self.feature_registry[feature_name](*args, **kwargs)
                else:
                    raise ValueError(f"Feature '{feature_name}' not registered")
            else:
                # Execute fallback if available
                if feature_name in self.current_state.active_fallbacks:
                    logger.debug(f"Executing fallback for feature '{feature_name}'")
                    return self.current_state.active_fallbacks[feature_name](*args, **kwargs)
                else:
                    # Return fallback value if configured
                    if feature_name in self.config.fallback_values:
                        return self.config.fallback_values[feature_name]
                    else:
                        raise RuntimeError(f"Feature '{feature_name}' is disabled and no fallback available")
    
    def attempt_recovery(self):
        """Attempt to recover from degradation"""
        with self.lock:
            if self.current_state.level == DegradationLevel.NORMAL:
                return
            
            if self.current_state.recovery_in_progress:
                return
            
            self.current_state.recovery_in_progress = True
            
            try:
                logger.info("Attempting recovery from degradation")
                
                # Check system health
                health_score = self._get_system_health_score()
                
                # Determine if recovery is possible
                if health_score >= self.config.recovery_thresholds.get(self.current_state.level, 0.8):
                    # Gradual recovery
                    self._gradual_recovery(health_score)
                else:
                    logger.info(f"Recovery not possible yet, health score: {health_score}")
                
            finally:
                self.current_state.recovery_in_progress = False
    
    def _gradual_recovery(self, health_score: float):
        """Perform gradual recovery based on health score"""
        # Determine target level based on health score
        target_level = DegradationLevel.NORMAL
        
        for level, threshold in self.config.recovery_thresholds.items():
            if health_score >= threshold:
                target_level = level
                break
        
        if target_level.value < self.current_state.level.value:
            # Recover to better level
            self._recover_to_level(target_level)
    
    def _recover_to_level(self, target_level: DegradationLevel):
        """Recover to specified level"""
        logger.info(f"Recovering from {self.current_state.level.value} to {target_level.value}")
        
        # Re-enable features based on new level
        features_to_enable = self._get_features_to_enable(target_level)
        
        for feature_name in features_to_enable:
            self._enable_feature(feature_name)
        
        # Update state
        if target_level == DegradationLevel.NORMAL:
            self.current_state = DegradationState()
        else:
            self.current_state.level = target_level
            self.current_state.trigger = None
            self.current_state.degradation_reason = None
        
        logger.info(f"Recovery completed: {len(features_to_enable)} features enabled")
    
    def _get_features_to_enable(self, target_level: DegradationLevel) -> List[str]:
        """Get features to enable when recovering to target level"""
        if target_level == DegradationLevel.NORMAL:
            return self.current_state.disabled_features.copy()
        
        current_disabled = set(self.current_state.disabled_features)
        target_disabled = set(self._get_features_to_disable(target_level))
        
        return list(current_disabled - target_disabled)
    
    def force_recovery(self):
        """Force recovery to normal state"""
        with self.lock:
            logger.info("Forcing recovery to normal state")
            self._recover_to_level(DegradationLevel.NORMAL)
    
    def get_degradation_status(self) -> Dict[str, Any]:
        """Get current degradation status"""
        with self.lock:
            return {
                'current_state': self.current_state.to_dict(),
                'health_score': self._get_system_health_score(),
                'feature_count': len(self.feature_registry),
                'disabled_features': len(self.current_state.disabled_features),
                'active_fallbacks': len(self.current_state.active_fallbacks),
                'degradation_history_count': len(self.degradation_history)
            }
    
    def shutdown(self):
        """Shutdown degradation system"""
        self.monitoring_running = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)
        
        logger.info("Graceful degradation system shut down")


class StrategicAgentDegradation(BaseGracefulDegradation):
    """Graceful degradation for strategic agents"""
    
    def __init__(self, config: Optional[DegradationConfig] = None):
        if config is None:
            config = DegradationConfig(
                agent_type=AgentType.STRATEGIC,
                essential_features=['risk_check', 'position_sizing'],
                fallback_values={
                    'market_regime': 'unknown',
                    'trend_direction': 'neutral',
                    'volatility_estimate': 0.2
                }
            )
        
        super().__init__(config)
        self._register_strategic_features()
    
    def _register_strategic_features(self):
        """Register strategic agent features"""
        self.register_feature('market_regime_analysis', self._market_regime_analysis, priority=8)
        self.register_feature('trend_detection', self._trend_detection, priority=7)
        self.register_feature('volatility_estimation', self._volatility_estimation, priority=6)
        self.register_feature('support_resistance', self._support_resistance, priority=5)
        self.register_feature('sentiment_analysis', self._sentiment_analysis, priority=4)
        self.register_feature('macro_analysis', self._macro_analysis, priority=3)
        self.register_feature('correlation_analysis', self._correlation_analysis, priority=2)
        self.register_feature('advanced_indicators', self._advanced_indicators, priority=1)
        
        # Register fallbacks
        self.register_fallback('market_regime_analysis', lambda: {'regime': 'unknown', 'confidence': 0.0})
        self.register_fallback('trend_detection', lambda: {'direction': 'neutral', 'strength': 0.0})
        self.register_fallback('volatility_estimation', lambda: {'volatility': 0.2, 'confidence': 0.0})
    
    def _market_regime_analysis(self):
        """Placeholder for market regime analysis"""
        return {'regime': 'bull', 'confidence': 0.8}
    
    def _trend_detection(self):
        """Placeholder for trend detection"""
        return {'direction': 'up', 'strength': 0.7}
    
    def _volatility_estimation(self):
        """Placeholder for volatility estimation"""
        return {'volatility': 0.15, 'confidence': 0.9}
    
    def _support_resistance(self):
        """Placeholder for support/resistance analysis"""
        return {'support': 100, 'resistance': 110}
    
    def _sentiment_analysis(self):
        """Placeholder for sentiment analysis"""
        return {'sentiment': 'positive', 'score': 0.6}
    
    def _macro_analysis(self):
        """Placeholder for macro analysis"""
        return {'outlook': 'positive', 'factors': []}
    
    def _correlation_analysis(self):
        """Placeholder for correlation analysis"""
        return {'correlations': {}}
    
    def _advanced_indicators(self):
        """Placeholder for advanced indicators"""
        return {'indicators': {}}
    
    def _check_degradation_triggers(self):
        """Check strategic agent degradation triggers"""
        # Placeholder implementation
        pass
    
    def _check_recovery_conditions(self):
        """Check strategic agent recovery conditions"""
        if self.current_state.level != DegradationLevel.NORMAL and self.config.auto_recovery:
            health_score = self._get_system_health_score()
            if health_score > 0.8:
                self.attempt_recovery()
    
    def _get_system_health_score(self) -> float:
        """Get strategic agent health score"""
        # Placeholder implementation
        return 0.85


class TacticalAgentDegradation(BaseGracefulDegradation):
    """Graceful degradation for tactical agents"""
    
    def __init__(self, config: Optional[DegradationConfig] = None):
        if config is None:
            config = DegradationConfig(
                agent_type=AgentType.TACTICAL,
                essential_features=['signal_generation', 'position_sizing'],
                fallback_values={
                    'signal_strength': 0.0,
                    'entry_price': None,
                    'exit_price': None
                }
            )
        
        super().__init__(config)
        self._register_tactical_features()
    
    def _register_tactical_features(self):
        """Register tactical agent features"""
        self.register_feature('signal_generation', self._signal_generation, priority=9)
        self.register_feature('entry_timing', self._entry_timing, priority=8)
        self.register_feature('exit_timing', self._exit_timing, priority=7)
        self.register_feature('position_sizing', self._position_sizing, priority=6)
        self.register_feature('stop_loss_calc', self._stop_loss_calc, priority=5)
        self.register_feature('take_profit_calc', self._take_profit_calc, priority=4)
        self.register_feature('momentum_analysis', self._momentum_analysis, priority=3)
        self.register_feature('mean_reversion', self._mean_reversion, priority=2)
        self.register_feature('arbitrage_detection', self._arbitrage_detection, priority=1)
        
        # Register fallbacks
        self.register_fallback('signal_generation', lambda: {'signal': 0.0, 'confidence': 0.0})
        self.register_fallback('position_sizing', lambda: {'size': 0.01, 'max_risk': 0.01})
    
    def _signal_generation(self):
        """Placeholder for signal generation"""
        return {'signal': 0.5, 'confidence': 0.8}
    
    def _entry_timing(self):
        """Placeholder for entry timing"""
        return {'timing_score': 0.7}
    
    def _exit_timing(self):
        """Placeholder for exit timing"""
        return {'timing_score': 0.6}
    
    def _position_sizing(self):
        """Placeholder for position sizing"""
        return {'size': 0.02, 'max_risk': 0.02}
    
    def _stop_loss_calc(self):
        """Placeholder for stop loss calculation"""
        return {'stop_loss': 0.98}
    
    def _take_profit_calc(self):
        """Placeholder for take profit calculation"""
        return {'take_profit': 1.05}
    
    def _momentum_analysis(self):
        """Placeholder for momentum analysis"""
        return {'momentum': 0.3}
    
    def _mean_reversion(self):
        """Placeholder for mean reversion"""
        return {'reversion_signal': -0.2}
    
    def _arbitrage_detection(self):
        """Placeholder for arbitrage detection"""
        return {'opportunities': []}
    
    def _check_degradation_triggers(self):
        """Check tactical agent degradation triggers"""
        # Placeholder implementation
        pass
    
    def _check_recovery_conditions(self):
        """Check tactical agent recovery conditions"""
        if self.current_state.level != DegradationLevel.NORMAL and self.config.auto_recovery:
            health_score = self._get_system_health_score()
            if health_score > 0.8:
                self.attempt_recovery()
    
    def _get_system_health_score(self) -> float:
        """Get tactical agent health score"""
        # Placeholder implementation
        return 0.90


class RiskAgentDegradation(BaseGracefulDegradation):
    """Graceful degradation for risk agents"""
    
    def __init__(self, config: Optional[DegradationConfig] = None):
        if config is None:
            config = DegradationConfig(
                agent_type=AgentType.RISK,
                essential_features=['position_risk_check', 'portfolio_risk_check'],
                fallback_values={
                    'risk_score': 1.0,  # Maximum risk in fallback
                    'position_limit': 0.01,  # Very conservative limit
                    'portfolio_var': 0.05
                }
            )
        
        super().__init__(config)
        self._register_risk_features()
    
    def _register_risk_features(self):
        """Register risk agent features"""
        self.register_feature('position_risk_check', self._position_risk_check, priority=10)
        self.register_feature('portfolio_risk_check', self._portfolio_risk_check, priority=9)
        self.register_feature('var_calculation', self._var_calculation, priority=8)
        self.register_feature('stress_testing', self._stress_testing, priority=7)
        self.register_feature('correlation_monitoring', self._correlation_monitoring, priority=6)
        self.register_feature('concentration_analysis', self._concentration_analysis, priority=5)
        self.register_feature('scenario_analysis', self._scenario_analysis, priority=4)
        self.register_feature('backtesting', self._backtesting, priority=3)
        self.register_feature('risk_reporting', self._risk_reporting, priority=2)
        self.register_feature('compliance_check', self._compliance_check, priority=1)
        
        # Register conservative fallbacks
        self.register_fallback('position_risk_check', lambda: {'approved': False, 'reason': 'Risk system degraded'})
        self.register_fallback('portfolio_risk_check', lambda: {'approved': False, 'reason': 'Risk system degraded'})
        self.register_fallback('var_calculation', lambda: {'var': 0.10, 'confidence': 0.0})
    
    def _position_risk_check(self):
        """Placeholder for position risk check"""
        return {'approved': True, 'risk_score': 0.3}
    
    def _portfolio_risk_check(self):
        """Placeholder for portfolio risk check"""
        return {'approved': True, 'risk_score': 0.4}
    
    def _var_calculation(self):
        """Placeholder for VaR calculation"""
        return {'var': 0.03, 'confidence': 0.95}
    
    def _stress_testing(self):
        """Placeholder for stress testing"""
        return {'stress_results': {}}
    
    def _correlation_monitoring(self):
        """Placeholder for correlation monitoring"""
        return {'correlations': {}}
    
    def _concentration_analysis(self):
        """Placeholder for concentration analysis"""
        return {'concentration_risk': 0.2}
    
    def _scenario_analysis(self):
        """Placeholder for scenario analysis"""
        return {'scenarios': {}}
    
    def _backtesting(self):
        """Placeholder for backtesting"""
        return {'backtest_results': {}}
    
    def _risk_reporting(self):
        """Placeholder for risk reporting"""
        return {'report': {}}
    
    def _compliance_check(self):
        """Placeholder for compliance check"""
        return {'compliant': True}
    
    def _check_degradation_triggers(self):
        """Check risk agent degradation triggers"""
        # Risk agents should be very conservative about degradation
        pass
    
    def _check_recovery_conditions(self):
        """Check risk agent recovery conditions"""
        if self.current_state.level != DegradationLevel.NORMAL and self.config.auto_recovery:
            health_score = self._get_system_health_score()
            if health_score > 0.95:  # Very high threshold for risk agents
                self.attempt_recovery()
    
    def _get_system_health_score(self) -> float:
        """Get risk agent health score"""
        # Placeholder implementation
        return 0.95


# Factory functions
def create_strategic_degradation(config: Optional[DegradationConfig] = None) -> StrategicAgentDegradation:
    """Create degradation system for strategic agents"""
    return StrategicAgentDegradation(config)


def create_tactical_degradation(config: Optional[DegradationConfig] = None) -> TacticalAgentDegradation:
    """Create degradation system for tactical agents"""
    return TacticalAgentDegradation(config)


def create_risk_degradation(config: Optional[DegradationConfig] = None) -> RiskAgentDegradation:
    """Create degradation system for risk agents"""
    return RiskAgentDegradation(config)


@contextmanager
def degradation_context(degradation_system: BaseGracefulDegradation, feature_name: str):
    """Context manager for executing features with degradation support"""
    try:
        yield lambda *args, **kwargs: degradation_system.execute_feature(feature_name, *args, **kwargs)
    except Exception as e:
        logger.error(f"Error in degradation context for feature '{feature_name}': {e}")
        raise