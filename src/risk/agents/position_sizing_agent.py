"""
Position Sizing Agent (π₁) for Risk Management MARL System

Specialized risk agent for dynamic position sizing based on portfolio
risk metrics and Kelly Criterion optimization.

Action Space: Discrete(5) - [reduce_large, reduce_small, hold, increase_small, increase_large]
Focuses on: Account equity, leverage, VaR, and correlation risk
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, Union
import structlog
from datetime import datetime

from src.risk.agents.base_risk_agent import BaseRiskAgent, RiskState, RiskAction, RiskMetrics
from src.risk.core.kelly_calculator import KellyCalculator, KellyOutput
from src.core.events import EventBus, Event, EventType

logger = structlog.get_logger()


class PositionSizingAction:
    """Position sizing actions"""
    REDUCE_LARGE = 0    # Reduce positions by 20%
    REDUCE_SMALL = 1    # Reduce positions by 10%
    HOLD = 2           # No change
    INCREASE_SMALL = 3  # Increase positions by 10%
    INCREASE_LARGE = 4  # Increase positions by 20%


class PositionSizingAgent(BaseRiskAgent):
    """
    Position Sizing Agent (π₁)
    
    Manages position sizes based on:
    - Kelly Criterion optimization
    - Risk-adjusted returns
    - Portfolio leverage constraints
    - VaR-based position limits
    - Correlation-based diversification
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: Optional[EventBus] = None):
        """
        Initialize Position Sizing Agent
        
        Args:
            config: Agent configuration
            event_bus: Event bus for communication
        """
        # Set agent-specific configuration
        config['name'] = config.get('name', 'position_sizing_agent')
        config['action_dim'] = 5  # Discrete action space
        
        super().__init__(config, event_bus)
        
        # Position sizing parameters
        self.max_leverage = config.get('max_leverage', 3.0)
        self.max_position_size = config.get('max_position_size', 0.1)  # 10% per position
        self.var_limit = config.get('var_limit', 0.02)  # 2% VaR limit
        self.correlation_threshold = config.get('correlation_threshold', 0.7)
        
        # Kelly Criterion calculator
        kelly_config = config.get('kelly_config', {})
        self.kelly_calculator = KellyCriterionCalculator(kelly_config)
        
        # Position sizing state
        self.current_positions = {}
        self.position_performance = {}
        self.kelly_fractions = {}
        
        # Performance tracking
        self.sizing_decisions = 0
        self.profitable_decisions = 0
        self.risk_reductions = 0
        self.leverage_violations = 0
        
        logger.info("Position Sizing Agent initialized",
                   max_leverage=self.max_leverage,
                   var_limit=self.var_limit)
    
    def calculate_risk_action(self, risk_state: RiskState) -> Tuple[int, float]:
        """
        Calculate position sizing action based on risk state
        
        Args:
            risk_state: Current risk state
            
        Returns:
            Tuple of (action, confidence)
        """
        try:
            # Analyze current risk levels
            risk_assessment = self._assess_risk_levels(risk_state)
            
            # Calculate optimal position sizes using Kelly Criterion
            kelly_recommendation = self._calculate_kelly_optimal(risk_state)
            
            # Determine action based on risk and Kelly analysis
            action = self._determine_sizing_action(risk_assessment, kelly_recommendation, risk_state)
            
            # Calculate confidence based on signal strength
            confidence = self._calculate_confidence(risk_assessment, kelly_recommendation)
            
            # Track decision
            self.sizing_decisions += 1
            
            # Log decision reasoning
            self._log_decision_reasoning(action, risk_assessment, kelly_recommendation, confidence)
            
            return action, confidence
            
        except Exception as e:
            logger.error("Error in position sizing calculation", error=str(e))
            return PositionSizingAction.HOLD, 0.1
    
    def _assess_risk_levels(self, risk_state: RiskState) -> Dict[str, float]:
        """Assess current risk levels across multiple dimensions"""
        
        assessment = {
            'leverage_risk': 0.0,
            'var_risk': 0.0,
            'correlation_risk': 0.0,
            'drawdown_risk': 0.0,
            'equity_risk': 0.0,
            'overall_risk': 0.0
        }
        
        # Leverage risk assessment
        if self.max_leverage > 0:
            implied_leverage = risk_state.margin_usage_pct * 4.0  # Approximate leverage
            assessment['leverage_risk'] = min(1.0, implied_leverage / self.max_leverage)
        
        # VaR risk assessment
        if self.var_limit > 0:
            assessment['var_risk'] = min(1.0, risk_state.var_estimate_5pct / self.var_limit)
        
        # Correlation risk assessment
        assessment['correlation_risk'] = min(1.0, risk_state.correlation_risk / self.correlation_threshold)
        
        # Drawdown risk assessment
        assessment['drawdown_risk'] = min(1.0, risk_state.current_drawdown_pct / 0.15)  # 15% drawdown threshold
        
        # Equity risk assessment (deviation from baseline)
        equity_deviation = abs(risk_state.account_equity_normalized - 1.0)
        assessment['equity_risk'] = min(1.0, equity_deviation / 0.2)  # 20% deviation threshold
        
        # Overall risk (weighted average)
        weights = {
            'leverage_risk': 0.3,
            'var_risk': 0.25,
            'correlation_risk': 0.2,
            'drawdown_risk': 0.15,
            'equity_risk': 0.1
        }
        
        assessment['overall_risk'] = sum(
            assessment[risk_type] * weight 
            for risk_type, weight in weights.items()
        )
        
        return assessment
    
    def _calculate_kelly_optimal(self, risk_state: RiskState) -> Dict[str, float]:
        """Calculate Kelly optimal position sizes"""
        
        try:
            # Simplified Kelly calculation based on risk state
            # In practice, this would use historical return data
            
            # Estimate win probability based on market conditions
            market_favorability = 1.0 - risk_state.market_stress_level
            base_win_prob = 0.5 + 0.1 * market_favorability  # 50-60% range
            
            # Estimate average win/loss ratios
            volatility_factor = risk_state.volatility_regime
            avg_win = 0.02 * (1 + volatility_factor)  # 2-4% average win
            avg_loss = 0.015 * (1 + volatility_factor)  # 1.5-3% average loss
            
            # Kelly fraction calculation
            if avg_loss > 0:
                kelly_fraction = (base_win_prob * avg_win - (1 - base_win_prob) * avg_loss) / avg_win
            else:
                kelly_fraction = 0.0
            
            # Apply safety factor and constraints
            kelly_fraction = max(0.0, min(0.25, kelly_fraction))  # Cap at 25%
            
            # Adjust for correlation (reduce if high correlation)
            correlation_adjustment = 1.0 - 0.5 * risk_state.correlation_risk
            adjusted_kelly = kelly_fraction * correlation_adjustment
            
            return {
                'kelly_fraction': adjusted_kelly,
                'win_probability': base_win_prob,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'correlation_adjustment': correlation_adjustment
            }
            
        except Exception as e:
            logger.error("Error calculating Kelly optimal", error=str(e))
            return {'kelly_fraction': 0.0}
    
    def _determine_sizing_action(self, 
                                risk_assessment: Dict[str, float],
                                kelly_recommendation: Dict[str, float],
                                risk_state: RiskState) -> int:
        """Determine position sizing action based on analysis"""
        
        overall_risk = risk_assessment.get('overall_risk', 0.0)
        kelly_fraction = kelly_recommendation.get('kelly_fraction', 0.0)
        current_leverage = risk_state.margin_usage_pct * 4.0
        
        # Emergency risk reduction
        if overall_risk > 0.8 or current_leverage > self.max_leverage:
            self.risk_reductions += 1
            if overall_risk > 0.9 or current_leverage > self.max_leverage * 1.2:
                return PositionSizingAction.REDUCE_LARGE
            else:
                return PositionSizingAction.REDUCE_SMALL
        
        # High risk - reduce positions
        elif overall_risk > 0.6:
            self.risk_reductions += 1
            return PositionSizingAction.REDUCE_SMALL
        
        # Low risk and good Kelly signal - increase positions
        elif overall_risk < 0.3 and kelly_fraction > 0.15:
            if kelly_fraction > 0.2 and risk_state.account_equity_normalized > 1.0:
                return PositionSizingAction.INCREASE_LARGE
            else:
                return PositionSizingAction.INCREASE_SMALL
        
        # Moderate conditions - small adjustments based on Kelly
        elif kelly_fraction > 0.1:
            return PositionSizingAction.INCREASE_SMALL
        elif kelly_fraction < 0.05:
            return PositionSizingAction.REDUCE_SMALL
        
        # Default - hold current positions
        else:
            return PositionSizingAction.HOLD
    
    def _calculate_confidence(self, 
                             risk_assessment: Dict[str, float],
                             kelly_recommendation: Dict[str, float]) -> float:
        """Calculate confidence in the sizing decision"""
        
        # Base confidence from signal strength
        kelly_fraction = kelly_recommendation.get('kelly_fraction', 0.0)
        kelly_confidence = min(1.0, kelly_fraction * 4.0)  # Scale to [0,1]
        
        # Risk clarity (how clear the risk signal is)
        overall_risk = risk_assessment.get('overall_risk', 0.0)
        if overall_risk > 0.7:  # High risk - high confidence in reduction
            risk_confidence = 0.8 + 0.2 * (overall_risk - 0.7) / 0.3
        elif overall_risk < 0.3:  # Low risk - moderate confidence in increase
            risk_confidence = 0.5 + 0.3 * (0.3 - overall_risk) / 0.3
        else:  # Moderate risk - lower confidence
            risk_confidence = 0.3 + 0.4 * (1.0 - abs(overall_risk - 0.5) / 0.5)
        
        # Win probability confidence
        win_prob = kelly_recommendation.get('win_probability', 0.5)
        prob_confidence = abs(win_prob - 0.5) * 2.0  # Distance from 50%
        
        # Combined confidence
        confidence = (kelly_confidence * 0.4 + risk_confidence * 0.4 + prob_confidence * 0.2)
        
        return max(0.1, min(1.0, confidence))
    
    def _log_decision_reasoning(self, 
                               action: int,
                               risk_assessment: Dict[str, float],
                               kelly_recommendation: Dict[str, float],
                               confidence: float):
        """Log decision reasoning for transparency"""
        
        action_names = {
            0: "REDUCE_LARGE",
            1: "REDUCE_SMALL", 
            2: "HOLD",
            3: "INCREASE_SMALL",
            4: "INCREASE_LARGE"
        }
        
        logger.info("Position sizing decision",
                   action=action_names.get(action, f"UNKNOWN({action})"),
                   confidence=f"{confidence:.3f}",
                   overall_risk=f"{risk_assessment.get('overall_risk', 0):.3f}",
                   kelly_fraction=f"{kelly_recommendation.get('kelly_fraction', 0):.3f}",
                   leverage_risk=f"{risk_assessment.get('leverage_risk', 0):.3f}",
                   var_risk=f"{risk_assessment.get('var_risk', 0):.3f}",
                   correlation_risk=f"{risk_assessment.get('correlation_risk', 0):.3f}")
    
    def validate_risk_constraints(self, risk_state: RiskState) -> bool:
        """
        Validate current risk state against position sizing constraints
        
        Args:
            risk_state: Current risk state
            
        Returns:
            True if constraints are satisfied
        """
        violations = []
        
        # Check leverage constraint
        implied_leverage = risk_state.margin_usage_pct * 4.0
        if implied_leverage > self.max_leverage:
            violations.append(f"leverage_violation: {implied_leverage:.2f} > {self.max_leverage}")
            self.leverage_violations += 1
        
        # Check VaR constraint
        if risk_state.var_estimate_5pct > self.var_limit:
            violations.append(f"var_violation: {risk_state.var_estimate_5pct:.3f} > {self.var_limit}")
        
        # Check correlation constraint
        if risk_state.correlation_risk > self.correlation_threshold:
            violations.append(f"correlation_violation: {risk_state.correlation_risk:.3f} > {self.correlation_threshold}")
        
        # Check drawdown constraint
        if risk_state.current_drawdown_pct > 0.2:  # 20% max drawdown
            violations.append(f"drawdown_violation: {risk_state.current_drawdown_pct:.3f} > 0.20")
        
        if violations:
            logger.warning("Position sizing constraint violations", violations=violations)
            return False
        
        return True
    
    def _get_safe_action(self) -> int:
        """Get safe default action for error cases"""
        return PositionSizingAction.REDUCE_SMALL  # Conservative reduction
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get position sizing agent metrics"""
        base_metrics = super().get_risk_metrics()
        
        # Calculate position sizing specific metrics
        decision_success_rate = (self.profitable_decisions / max(1, self.sizing_decisions))
        risk_reduction_rate = (self.risk_reductions / max(1, self.sizing_decisions))
        
        # Update base metrics with position sizing specifics
        return RiskMetrics(
            total_risk_decisions=self.sizing_decisions,
            risk_events_detected=self.risk_reductions,
            false_positive_rate=base_metrics.false_positive_rate,
            avg_response_time_ms=base_metrics.avg_response_time_ms,
            risk_adjusted_return=decision_success_rate,  # Proxy for risk-adjusted return
            max_drawdown=0.0,  # Would be calculated from position history
            sharpe_ratio=0.0,  # Would be calculated from position history
            var_accuracy=1.0 - base_metrics.false_positive_rate,  # Proxy for VaR accuracy
            correlation_prediction_accuracy=0.0  # Not directly applicable
        )
    
    def update_position_performance(self, symbol: str, pnl: float, position_size: float):
        """Update position performance tracking"""
        if symbol not in self.position_performance:
            self.position_performance[symbol] = {'total_pnl': 0.0, 'trade_count': 0}
        
        self.position_performance[symbol]['total_pnl'] += pnl
        self.position_performance[symbol]['trade_count'] += 1
        
        if pnl > 0:
            self.profitable_decisions += 1
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get current position summary"""
        return {
            'sizing_decisions': self.sizing_decisions,
            'profitable_decisions': self.profitable_decisions,
            'risk_reductions': self.risk_reductions,
            'leverage_violations': self.leverage_violations,
            'success_rate': self.profitable_decisions / max(1, self.sizing_decisions),
            'risk_reduction_rate': self.risk_reductions / max(1, self.sizing_decisions),
            'position_performance': dict(self.position_performance)
        }
    
    def reset(self):
        """Reset agent state"""
        super().reset()
        self.sizing_decisions = 0
        self.profitable_decisions = 0
        self.risk_reductions = 0
        self.leverage_violations = 0
        self.position_performance.clear()
        self.kelly_fractions.clear()
        logger.info("Position Sizing Agent reset")