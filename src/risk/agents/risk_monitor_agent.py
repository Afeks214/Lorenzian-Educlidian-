"""
Risk Monitor Agent (π₃) for Risk Management MARL System

Specialized risk agent for continuous risk monitoring and emergency response.
Acts as the primary risk oversight agent with authority to trigger emergency stops.
Enhanced with comprehensive error handling, recovery, and degradation mechanisms.

Action Space: Discrete(4) - [no_action, alert, reduce_risk, emergency_stop]
Focuses on: Global risk assessment, emergency detection, and system protection
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, Union
import structlog
from datetime import datetime, timedelta
from collections import deque

from src.risk.agents.base_risk_agent import BaseRiskAgent, RiskState, RiskAction, RiskMetrics
from src.core.events import EventBus, Event, EventType
from src.core.errors.agent_error_decorators import (
    risk_agent_decorator, AgentType, AgentErrorConfig
)
from src.core.errors.agent_recovery_strategies import (
    create_risk_recovery_manager, RecoveryConfig
)
from src.core.errors.graceful_degradation import (
    create_risk_degradation, DegradationConfig
)
from src.core.errors.error_monitoring import record_error
from src.core.errors.base_exceptions import (
    ValidationError, DataError, SystemError, ErrorContext, CriticalError
)

logger = structlog.get_logger()


class RiskMonitorAction:
    """Risk monitor actions"""
    NO_ACTION = 0      # Continue monitoring
    ALERT = 1          # Generate risk alert
    REDUCE_RISK = 2    # Recommend risk reduction
    EMERGENCY_STOP = 3 # Execute emergency stop


class RiskMonitorAgent(BaseRiskAgent):
    """
    Risk Monitor Agent (π₃)
    
    Primary risk oversight agent responsible for:
    - Continuous risk monitoring across all dimensions
    - Early warning system for risk escalation
    - Emergency stop authority and execution
    - Risk event detection and classification
    - System-wide risk coordination
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: Optional[EventBus] = None):
        """Initialize Risk Monitor Agent with enhanced error handling"""
        config['name'] = config.get('name', 'risk_monitor_agent')
        config['action_dim'] = 4  # Discrete action space
        
        super().__init__(config, event_bus)
        
        # Risk thresholds
        self.alert_threshold = config.get('alert_threshold', 0.6)
        self.reduce_threshold = config.get('reduce_threshold', 0.75)
        self.emergency_threshold = config.get('emergency_threshold', 0.9)
        
        # Multi-dimensional risk weights
        self.risk_weights = config.get('risk_weights', {
            'var_risk': 0.25,
            'leverage_risk': 0.20,
            'correlation_risk': 0.15,
            'drawdown_risk': 0.15,
            'liquidity_risk': 0.10,
            'market_stress': 0.10,
            'volatility_risk': 0.05
        })
        
        # Performance tracking
        self.alerts_generated = 0
        self.risk_reductions_triggered = 0
        self.emergency_stops_executed = 0
        self.false_alarms = 0
        self.correct_alerts = 0
        
        # Risk monitoring state
        self.risk_history = deque(maxlen=100)
        self.alert_history = deque(maxlen=50)
        self.last_emergency_time = None
        
        # Enhanced error handling setup
        self._setup_error_handling()
        self._setup_recovery_system()
        self._setup_degradation_system()
        
        logger.info("Risk Monitor Agent initialized with enhanced error handling",
                   alert_threshold=self.alert_threshold,
                   emergency_threshold=self.emergency_threshold)
    
    def _setup_error_handling(self):
        """Setup risk agent specific error handling"""
        error_config = AgentErrorConfig(
            agent_type=AgentType.RISK,
            max_retries=self.config.get('max_retries', 2),  # Risk agents should be conservative
            retry_delay=self.config.get('retry_delay', 0.5),
            circuit_breaker_threshold=self.config.get('circuit_breaker_threshold', 3),
            graceful_degradation=self.config.get('graceful_degradation', False)  # Risk agents should fail fast
        )
        
        self.error_decorator = risk_agent_decorator(error_config)
    
    def _setup_recovery_system(self):
        """Setup recovery management system for risk agent"""
        recovery_config = RecoveryConfig(
            max_recovery_attempts=self.config.get('max_recovery_attempts', 2),
            recovery_timeout=self.config.get('recovery_timeout', 10.0),  # Quick recovery for risk
            gradual_recovery_steps=self.config.get('gradual_recovery_steps', 2)
        )
        
        self.recovery_manager = create_risk_recovery_manager(recovery_config)
    
    def _setup_degradation_system(self):
        """Setup graceful degradation system for risk agent"""
        degradation_config = DegradationConfig(
            agent_type=AgentType.RISK,
            essential_features=['risk_assessment', 'emergency_detection'],
            fallback_values={
                'risk_score': 1.0,  # Maximum risk in fallback
                'recommendation': 'emergency_stop',
                'confidence': 0.0
            }
        )
        
        self.degradation_system = create_risk_degradation(degradation_config)
    
    def _create_error_context(self, operation: str, **kwargs) -> ErrorContext:
        """Create error context for risk monitoring"""
        return ErrorContext(
            service_name=self.config['name'],
            additional_data={
                'operation': operation,
                'agent_type': 'risk_monitor',
                'alerts_generated': self.alerts_generated,
                'emergency_stops': self.emergency_stops_executed,
                **kwargs
            }
        )
    
    def calculate_risk_action(self, risk_state: RiskState) -> Tuple[int, float]:
        """Calculate risk monitoring action with comprehensive error handling"""
        context = self._create_error_context('calculate_risk_action', risk_state=risk_state)
        
        try:
            # Validate risk state
            if not self._validate_risk_state(risk_state):
                raise ValidationError("Invalid risk state", field="risk_state")
            
            # Calculate comprehensive risk score with error handling
            risk_score = self._calculate_risk_score_with_error_handling(risk_state)
            
            # Detect specific risk patterns with error handling
            risk_patterns = self._detect_risk_patterns_with_error_handling(risk_state)
            
            # Determine action based on risk level and patterns
            action = self._determine_monitor_action_with_error_handling(risk_score, risk_patterns, risk_state)
            
            # Calculate confidence with error handling
            confidence = self._calculate_confidence_with_error_handling(risk_score, risk_patterns, action)
            
            # Update monitoring state
            self._update_monitoring_state(risk_score, action, risk_state)
            
            # Log decision
            self._log_monitoring_decision(action, risk_score, risk_patterns, confidence)
            
            return action, confidence
            
        except Exception as e:
            # Record error for monitoring
            record_error(e, context, AgentType.RISK, self.config['name'])
            
            # For risk agents, any error is critical
            if isinstance(e, CriticalError):
                logger.critical("Critical error in risk monitoring", error=str(e))
                return RiskMonitorAction.EMERGENCY_STOP, 1.0
            
            # Attempt recovery
            recovery_result = self.recovery_manager.recover(e, context, self.__dict__)
            
            if recovery_result.get('success', False):
                # Retry with recovered state
                try:
                    return self.calculate_risk_action(risk_state)
                except Exception as retry_error:
                    logger.error("Retry failed after recovery", error=str(retry_error))
            
            # Use conservative fallback - emergency stop
            return self._get_emergency_fallback_action(e, context)
    
    def _validate_risk_state(self, risk_state: RiskState) -> bool:
        """Validate risk state input"""
        if risk_state is None:
            return False
        
        # Check for required fields
        required_fields = ['current_drawdown_pct', 'var_estimate_5pct', 'correlation_risk']
        for field in required_fields:
            if not hasattr(risk_state, field):
                return False
            
            value = getattr(risk_state, field)
            if value is None or (isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value))):
                return False
        
        return True
    
    def _calculate_risk_score_with_error_handling(self, risk_state: RiskState) -> float:
        """Calculate risk score with error handling"""
        try:
            return self.degradation_system.execute_feature(
                'risk_score_calculation',
                risk_state
            )
        except Exception as e:
            logger.warning("Risk score calculation failed, using fallback", error=str(e))
            # Conservative fallback - assume high risk
            return 0.9
    
    def _detect_risk_patterns_with_error_handling(self, risk_state: RiskState) -> Dict[str, bool]:
        """Detect risk patterns with error handling"""
        try:
            return self.degradation_system.execute_feature(
                'risk_pattern_detection',
                risk_state
            )
        except Exception as e:
            logger.warning("Risk pattern detection failed, using fallback", error=str(e))
            # Conservative fallback - assume multiple risks
            return {
                'rapid_drawdown': True,
                'correlation_spike': True,
                'multiple_risk_convergence': True
            }
    
    def _determine_monitor_action_with_error_handling(self, risk_score: float, 
                                                    risk_patterns: Dict[str, bool], 
                                                    risk_state: RiskState) -> int:
        """Determine monitoring action with error handling"""
        try:
            return self.degradation_system.execute_feature(
                'action_determination',
                risk_score, risk_patterns, risk_state
            )
        except Exception as e:
            logger.warning("Action determination failed, using emergency fallback", error=str(e))
            # Conservative fallback - emergency stop
            return RiskMonitorAction.EMERGENCY_STOP
    
    def _calculate_confidence_with_error_handling(self, risk_score: float, 
                                                risk_patterns: Dict[str, bool], 
                                                action: int) -> float:
        """Calculate confidence with error handling"""
        try:
            return self.degradation_system.execute_feature(
                'confidence_calculation',
                risk_score, risk_patterns, action
            )
        except Exception as e:
            logger.warning("Confidence calculation failed, using fallback", error=str(e))
            # Conservative fallback - high confidence in emergency actions
            if action == RiskMonitorAction.EMERGENCY_STOP:
                return 1.0
            else:
                return 0.5
    
    def _get_emergency_fallback_action(self, error: Exception, context: ErrorContext) -> Tuple[int, float]:
        """Get emergency fallback action when all else fails"""
        logger.critical("Using emergency fallback due to error", error=str(error))
        
        # Ultimate fallback - emergency stop with full confidence
        self.emergency_stops_executed += 1
        self.last_emergency_time = datetime.now()
        
        return RiskMonitorAction.EMERGENCY_STOP, 1.0
    
    def _calculate_comprehensive_risk_score(self, risk_state: RiskState) -> float:
        """Calculate comprehensive risk score across all dimensions"""
        
        risk_components = {
            'var_risk': min(1.0, risk_state.var_estimate_5pct / 0.05),  # 5% VaR threshold
            'leverage_risk': min(1.0, risk_state.margin_usage_pct / 0.8),  # 80% margin usage
            'correlation_risk': min(1.0, risk_state.correlation_risk / 0.8),  # 80% correlation
            'drawdown_risk': min(1.0, risk_state.current_drawdown_pct / 0.15),  # 15% drawdown
            'liquidity_risk': 1.0 - risk_state.liquidity_conditions,
            'market_stress': risk_state.market_stress_level,
            'volatility_risk': risk_state.volatility_regime
        }
        
        # Calculate weighted risk score
        risk_score = sum(
            risk_components[component] * self.risk_weights.get(component, 0.0)
            for component in risk_components
        )
        
        # Add penalty for multiple simultaneous risks
        high_risk_count = sum(1 for risk in risk_components.values() if risk > 0.7)
        if high_risk_count > 2:
            risk_score *= (1.0 + 0.1 * (high_risk_count - 2))  # 10% penalty per additional high risk
        
        return min(1.0, risk_score)
    
    def _detect_risk_patterns(self, risk_state: RiskState) -> Dict[str, bool]:
        """Detect specific risk patterns and scenarios"""
        
        patterns = {
            'rapid_drawdown': False,
            'correlation_spike': False,
            'liquidity_crisis': False,
            'leverage_buildup': False,
            'volatility_explosion': False,
            'market_stress_escalation': False,
            'multiple_risk_convergence': False
        }
        
        # Store current state in history
        self.risk_history.append({
            'timestamp': datetime.now(),
            'drawdown': risk_state.current_drawdown_pct,
            'correlation': risk_state.correlation_risk,
            'volatility': risk_state.volatility_regime,
            'stress': risk_state.market_stress_level,
            'leverage': risk_state.margin_usage_pct
        })
        
        if len(self.risk_history) >= 5:
            recent_history = list(self.risk_history)[-5:]
            
            # Rapid drawdown detection
            drawdown_change = recent_history[-1]['drawdown'] - recent_history[0]['drawdown']
            if drawdown_change > 0.05:  # 5% drawdown in recent history
                patterns['rapid_drawdown'] = True
            
            # Correlation spike detection
            if risk_state.correlation_risk > 0.85:
                patterns['correlation_spike'] = True
            
            # Liquidity crisis detection
            if risk_state.liquidity_conditions < 0.3:
                patterns['liquidity_crisis'] = True
            
            # Leverage buildup detection
            leverage_trend = recent_history[-1]['leverage'] - recent_history[0]['leverage']
            if leverage_trend > 0.2:  # 20% increase in leverage usage
                patterns['leverage_buildup'] = True
            
            # Volatility explosion detection
            volatility_change = recent_history[-1]['volatility'] - recent_history[0]['volatility']
            if volatility_change > 0.3:  # 30% volatility increase
                patterns['volatility_explosion'] = True
            
            # Market stress escalation
            stress_trend = recent_history[-1]['stress'] - recent_history[0]['stress']
            if stress_trend > 0.25:  # 25% stress increase
                patterns['market_stress_escalation'] = True
            
            # Multiple risk convergence
            high_risks = sum([
                risk_state.current_drawdown_pct > 0.1,
                risk_state.correlation_risk > 0.8,
                risk_state.volatility_regime > 0.8,
                risk_state.market_stress_level > 0.7,
                risk_state.margin_usage_pct > 0.8
            ])
            if high_risks >= 3:
                patterns['multiple_risk_convergence'] = True
        
        return patterns
    
    def _determine_monitor_action(self, 
                                risk_score: float, 
                                risk_patterns: Dict[str, bool],
                                risk_state: RiskState) -> int:
        """Determine monitoring action based on risk assessment"""
        
        # Emergency conditions - immediate stop
        emergency_patterns = [
            'multiple_risk_convergence',
            'rapid_drawdown'
        ]
        
        if (risk_score >= self.emergency_threshold or 
            any(risk_patterns.get(pattern, False) for pattern in emergency_patterns) or
            risk_state.current_drawdown_pct > 0.2):  # 20% absolute drawdown limit
            
            self.emergency_stops_executed += 1
            self.last_emergency_time = datetime.now()
            return RiskMonitorAction.EMERGENCY_STOP
        
        # Risk reduction conditions
        reduction_patterns = [
            'correlation_spike',
            'volatility_explosion', 
            'leverage_buildup',
            'market_stress_escalation'
        ]
        
        if (risk_score >= self.reduce_threshold or
            any(risk_patterns.get(pattern, False) for pattern in reduction_patterns)):
            
            self.risk_reductions_triggered += 1
            return RiskMonitorAction.REDUCE_RISK
        
        # Alert conditions
        alert_patterns = [
            'liquidity_crisis'
        ]
        
        if (risk_score >= self.alert_threshold or
            any(risk_patterns.get(pattern, False) for pattern in alert_patterns)):
            
            self.alerts_generated += 1
            return RiskMonitorAction.ALERT
        
        # Normal conditions
        return RiskMonitorAction.NO_ACTION
    
    def _calculate_confidence(self, 
                            risk_score: float,
                            risk_patterns: Dict[str, bool],
                            action: int) -> float:
        """Calculate confidence in monitoring decision"""
        
        # Base confidence from risk score clarity
        if risk_score > 0.8 or risk_score < 0.2:
            score_confidence = 0.9  # Clear signal
        elif risk_score > 0.6 or risk_score < 0.4:
            score_confidence = 0.7  # Moderate signal
        else:
            score_confidence = 0.4  # Ambiguous signal
        
        # Pattern confidence
        pattern_count = sum(risk_patterns.values())
        pattern_confidence = min(1.0, 0.5 + 0.2 * pattern_count)
        
        # Action appropriateness confidence
        if action == RiskMonitorAction.EMERGENCY_STOP:
            action_confidence = 0.95 if risk_score > 0.85 else 0.7
        elif action == RiskMonitorAction.REDUCE_RISK:
            action_confidence = 0.8 if risk_score > 0.7 else 0.6
        elif action == RiskMonitorAction.ALERT:
            action_confidence = 0.7 if risk_score > 0.6 else 0.5
        else:  # NO_ACTION
            action_confidence = 0.8 if risk_score < 0.4 else 0.4
        
        # Historical accuracy factor
        if self.alerts_generated > 0:
            accuracy_rate = self.correct_alerts / self.alerts_generated
            historical_confidence = max(0.3, min(0.9, accuracy_rate))
        else:
            historical_confidence = 0.6
        
        # Combined confidence
        confidence = (score_confidence * 0.3 +
                     pattern_confidence * 0.3 +
                     action_confidence * 0.25 +
                     historical_confidence * 0.15)
        
        return max(0.1, min(1.0, confidence))
    
    def _update_monitoring_state(self, risk_score: float, action: int, risk_state: RiskState):
        """Update internal monitoring state"""
        
        # Record alert
        if action in [RiskMonitorAction.ALERT, RiskMonitorAction.REDUCE_RISK, RiskMonitorAction.EMERGENCY_STOP]:
            self.alert_history.append({
                'timestamp': datetime.now(),
                'action': action,
                'risk_score': risk_score,
                'drawdown': risk_state.current_drawdown_pct
            })
    
    def _log_monitoring_decision(self, 
                               action: int,
                               risk_score: float,
                               risk_patterns: Dict[str, bool],
                               confidence: float):
        """Log monitoring decision for transparency"""
        
        action_names = {
            0: "NO_ACTION",
            1: "ALERT",
            2: "REDUCE_RISK", 
            3: "EMERGENCY_STOP"
        }
        
        active_patterns = [pattern for pattern, active in risk_patterns.items() if active]
        
        logger.info("Risk monitoring decision",
                   action=action_names.get(action, f"UNKNOWN({action})"),
                   confidence=f"{confidence:.3f}",
                   risk_score=f"{risk_score:.3f}",
                   active_patterns=active_patterns,
                   alert_threshold=self.alert_threshold,
                   emergency_threshold=self.emergency_threshold)
    
    def validate_risk_constraints(self, risk_state: RiskState) -> bool:
        """Validate risk constraints from monitor perspective"""
        
        # Monitor has stricter constraints as the oversight agent
        violations = []
        
        # Absolute risk limits
        if risk_state.current_drawdown_pct > 0.25:  # 25% absolute limit
            violations.append(f"critical_drawdown: {risk_state.current_drawdown_pct:.3f} > 0.25")
        
        if risk_state.var_estimate_5pct > 0.1:  # 10% VaR limit
            violations.append(f"critical_var: {risk_state.var_estimate_5pct:.3f} > 0.10")
        
        if risk_state.correlation_risk > 0.95:  # 95% correlation limit
            violations.append(f"critical_correlation: {risk_state.correlation_risk:.3f} > 0.95")
        
        if risk_state.margin_usage_pct > 0.9:  # 90% margin usage limit
            violations.append(f"critical_margin: {risk_state.margin_usage_pct:.3f} > 0.90")
        
        if violations:
            logger.critical("CRITICAL RISK CONSTRAINT VIOLATIONS", violations=violations)
            return False
        
        return True
    
    def _get_safe_action(self) -> int:
        """Get safe default action for error cases"""
        return RiskMonitorAction.EMERGENCY_STOP  # Fail-safe to emergency stop
    
    def record_alert_outcome(self, was_correct: bool):
        """Record the outcome of a risk alert for learning"""
        if was_correct:
            self.correct_alerts += 1
        else:
            self.false_alarms += 1
        
        logger.info("Alert outcome recorded", 
                   was_correct=was_correct,
                   accuracy_rate=self.correct_alerts / max(1, self.alerts_generated))
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get risk monitor agent metrics"""
        base_metrics = super().get_risk_metrics()
        
        total_actions = (self.alerts_generated + self.risk_reductions_triggered + 
                        self.emergency_stops_executed)
        
        alert_accuracy = self.correct_alerts / max(1, self.alerts_generated)
        false_alarm_rate = self.false_alarms / max(1, self.alerts_generated)
        
        return RiskMetrics(
            total_risk_decisions=total_actions,
            risk_events_detected=self.alerts_generated,
            false_positive_rate=false_alarm_rate,
            avg_response_time_ms=base_metrics.avg_response_time_ms,
            risk_adjusted_return=alert_accuracy,  # Proxy using alert accuracy
            max_drawdown=0.0,  # Would track from monitoring history
            sharpe_ratio=0.0,  # Not directly applicable
            var_accuracy=alert_accuracy,  # Proxy using alert accuracy
            correlation_prediction_accuracy=alert_accuracy  # Proxy using alert accuracy
        )
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get risk monitoring summary"""
        return {
            'alerts_generated': self.alerts_generated,
            'risk_reductions_triggered': self.risk_reductions_triggered,
            'emergency_stops_executed': self.emergency_stops_executed,
            'correct_alerts': self.correct_alerts,
            'false_alarms': self.false_alarms,
            'alert_accuracy': self.correct_alerts / max(1, self.alerts_generated),
            'false_alarm_rate': self.false_alarms / max(1, self.alerts_generated),
            'last_emergency': self.last_emergency_time.isoformat() if self.last_emergency_time else None
        }
    
    def reset(self):
        """Reset agent state"""
        super().reset()
        self.alerts_generated = 0
        self.risk_reductions_triggered = 0
        self.emergency_stops_executed = 0
        self.false_alarms = 0
        self.correct_alerts = 0
        self.last_emergency_time = None
        self.risk_history.clear()
        self.alert_history.clear()
        logger.info("Risk Monitor Agent reset")