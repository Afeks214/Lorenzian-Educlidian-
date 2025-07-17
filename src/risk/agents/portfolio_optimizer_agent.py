"""
Portfolio Optimizer Agent (π₄) for Risk Management MARL System

Specialized risk agent for dynamic portfolio optimization and asset allocation
based on risk-adjusted returns and correlation analysis.

Action Space: Box(0.0, 1.0, (5,)) - [equity_weight, fixed_income, commodities, cash, alternatives]
Focuses on: Portfolio diversification, correlation management, and risk-adjusted allocation
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, Union
import structlog
from datetime import datetime, timedelta
from collections import deque

from src.risk.agents.base_risk_agent import BaseRiskAgent, RiskState, RiskAction, RiskMetrics
from src.core.events import EventBus, Event, EventType

logger = structlog.get_logger()


class PortfolioOptimizerAgent(BaseRiskAgent):
    """
    Portfolio Optimizer Agent (π₄)
    
    Manages portfolio allocation across asset classes based on:
    - Risk-adjusted return optimization
    - Correlation diversification
    - Dynamic rebalancing triggers
    - Market regime adaptation
    - Liquidity management
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: Optional[EventBus] = None):
        """Initialize Portfolio Optimizer Agent"""
        config['name'] = config.get('name', 'portfolio_optimizer_agent')
        config['action_dim'] = 5  # Continuous action space for 5 asset classes
        
        super().__init__(config, event_bus)
        
        # Asset class parameters
        self.asset_classes = ['equity', 'fixed_income', 'commodities', 'cash', 'alternatives']
        self.min_weight = config.get('min_weight', 0.0)
        self.max_weight = config.get('max_weight', 1.0)
        self.rebalance_threshold = config.get('rebalance_threshold', 0.05)  # 5% deviation
        
        # Risk parameters
        self.target_volatility = config.get('target_volatility', 0.12)  # 12% annual
        self.max_correlation = config.get('max_correlation', 0.8)
        self.liquidity_buffer = config.get('liquidity_buffer', 0.1)  # 10% cash minimum
        
        # Market regime parameters
        self.regime_weights = config.get('regime_weights', {
            'normal': {'equity': 0.6, 'fixed_income': 0.25, 'commodities': 0.05, 'cash': 0.05, 'alternatives': 0.05},
            'stress': {'equity': 0.3, 'fixed_income': 0.4, 'commodities': 0.1, 'cash': 0.15, 'alternatives': 0.05},
            'crisis': {'equity': 0.1, 'fixed_income': 0.5, 'commodities': 0.05, 'cash': 0.3, 'alternatives': 0.05}
        })
        
        # Current state
        self.current_weights = np.array([0.6, 0.25, 0.05, 0.05, 0.05])  # Initial allocation
        self.target_weights = self.current_weights.copy()
        self.last_rebalance_time = datetime.now()
        
        # Performance tracking
        self.rebalances_executed = 0
        self.correlation_adjustments = 0
        self.regime_changes = 0
        self.weight_violations = 0
        
        # Historical tracking
        self.allocation_history = deque(maxlen=100)
        self.performance_history = deque(maxlen=100)
        self.correlation_history = deque(maxlen=50)
        
        logger.info("Portfolio Optimizer Agent initialized",
                   target_volatility=self.target_volatility,
                   rebalance_threshold=self.rebalance_threshold)
    
    def calculate_risk_action(self, risk_state: RiskState) -> Tuple[np.ndarray, float]:
        """Calculate portfolio optimization action"""
        try:
            # Analyze market conditions and regime
            market_analysis = self._analyze_market_regime(risk_state)
            
            # Calculate optimal weights based on current conditions
            optimal_weights = self._calculate_optimal_weights(risk_state, market_analysis)
            
            # Apply risk and liquidity constraints
            constrained_weights = self._apply_constraints(optimal_weights, risk_state)
            
            # Normalize weights to sum to 1
            normalized_weights = self._normalize_weights(constrained_weights)
            
            # Calculate confidence in allocation
            confidence = self._calculate_confidence(normalized_weights, risk_state, market_analysis)
            
            # Update internal state
            self._update_optimizer_state(normalized_weights, risk_state)
            
            # Log decision
            self._log_optimization_decision(normalized_weights, market_analysis, confidence)
            
            return normalized_weights.astype(np.float32), confidence
            
        except Exception as e:
            logger.error("Error in portfolio optimization", error=str(e))
            # Return safe default allocation
            return np.array([0.2, 0.4, 0.05, 0.3, 0.05], dtype=np.float32), 0.1
    
    def _analyze_market_regime(self, risk_state: RiskState) -> Dict[str, Any]:
        """Analyze current market regime for optimization"""
        
        analysis = {
            'regime': 'normal',
            'stress_level': risk_state.market_stress_level,
            'volatility_level': risk_state.volatility_regime,
            'correlation_level': risk_state.correlation_risk,
            'liquidity_level': risk_state.liquidity_conditions,
            'regime_confidence': 0.5
        }
        
        # Determine market regime
        if risk_state.market_stress_level > 0.8 or risk_state.current_drawdown_pct > 0.15:
            analysis['regime'] = 'crisis'
            analysis['regime_confidence'] = 0.9
        elif (risk_state.market_stress_level > 0.6 or 
              risk_state.volatility_regime > 0.7 or 
              risk_state.correlation_risk > 0.8):
            analysis['regime'] = 'stress'
            analysis['regime_confidence'] = 0.8
        else:
            analysis['regime'] = 'normal'
            analysis['regime_confidence'] = 0.7
        
        # Calculate diversification opportunity
        analysis['diversification_opportunity'] = 1.0 - risk_state.correlation_risk
        
        # Calculate rebalancing urgency
        weight_deviation = np.sum(np.abs(self.current_weights - self.target_weights))
        analysis['rebalancing_urgency'] = min(1.0, weight_deviation / 0.2)  # Scale to [0,1]
        
        return analysis
    
    def _calculate_optimal_weights(self, risk_state: RiskState, market_analysis: Dict[str, Any]) -> np.ndarray:
        """Calculate optimal portfolio weights"""
        
        # Start with regime-based baseline
        regime = market_analysis['regime']
        baseline_weights = np.array([
            self.regime_weights[regime]['equity'],
            self.regime_weights[regime]['fixed_income'], 
            self.regime_weights[regime]['commodities'],
            self.regime_weights[regime]['cash'],
            self.regime_weights[regime]['alternatives']
        ])
        
        # Adjust for correlation risk
        if risk_state.correlation_risk > self.max_correlation:
            # Increase diversification - reduce equity, increase alternatives
            correlation_adjustment = (risk_state.correlation_risk - self.max_correlation) * 2.0
            baseline_weights[0] *= (1.0 - correlation_adjustment * 0.3)  # Reduce equity
            baseline_weights[4] *= (1.0 + correlation_adjustment * 0.5)  # Increase alternatives
            baseline_weights[3] *= (1.0 + correlation_adjustment * 0.2)  # Increase cash
            self.correlation_adjustments += 1
        
        # Adjust for volatility
        if risk_state.volatility_regime > 0.8:
            # High volatility - increase defensive assets
            vol_adjustment = (risk_state.volatility_regime - 0.8) * 5.0
            baseline_weights[0] *= (1.0 - vol_adjustment * 0.2)  # Reduce equity
            baseline_weights[1] *= (1.0 + vol_adjustment * 0.3)  # Increase fixed income
            baseline_weights[3] *= (1.0 + vol_adjustment * 0.4)  # Increase cash
        
        # Adjust for drawdown
        if risk_state.current_drawdown_pct > 0.05:
            # In drawdown - defensive positioning
            drawdown_severity = min(1.0, risk_state.current_drawdown_pct / 0.2)
            baseline_weights[0] *= (1.0 - drawdown_severity * 0.4)  # Reduce equity
            baseline_weights[3] *= (1.0 + drawdown_severity * 0.6)  # Increase cash
        
        # Adjust for liquidity conditions
        if risk_state.liquidity_conditions < 0.5:
            # Poor liquidity - increase liquid assets
            liquidity_stress = (0.5 - risk_state.liquidity_conditions) * 2.0
            baseline_weights[3] *= (1.0 + liquidity_stress * 0.5)  # Increase cash
            baseline_weights[4] *= (1.0 - liquidity_stress * 0.3)  # Reduce alternatives
        
        return baseline_weights
    
    def _apply_constraints(self, weights: np.ndarray, risk_state: RiskState) -> np.ndarray:
        """Apply portfolio constraints"""
        
        constrained_weights = weights.copy()
        
        # Apply min/max weight constraints
        constrained_weights = np.clip(constrained_weights, self.min_weight, self.max_weight)
        
        # Ensure minimum cash buffer during stress
        if risk_state.market_stress_level > 0.7:
            min_cash = max(self.liquidity_buffer, 0.15)  # 15% minimum in stress
            if constrained_weights[3] < min_cash:
                deficit = min_cash - constrained_weights[3]
                constrained_weights[3] = min_cash
                # Reduce other weights proportionally
                other_weights = constrained_weights[:3].sum() + constrained_weights[4]
                if other_weights > 0:
                    reduction_factor = (1.0 - min_cash) / other_weights
                    constrained_weights[0] *= reduction_factor
                    constrained_weights[1] *= reduction_factor 
                    constrained_weights[2] *= reduction_factor
                    constrained_weights[4] *= reduction_factor
        
        # Maximum equity exposure during high correlation
        if risk_state.correlation_risk > 0.8:
            max_equity = 0.4  # 40% maximum when correlation is high
            if constrained_weights[0] > max_equity:
                excess = constrained_weights[0] - max_equity
                constrained_weights[0] = max_equity
                # Distribute excess to fixed income and cash
                constrained_weights[1] += excess * 0.6
                constrained_weights[3] += excess * 0.4
        
        return constrained_weights
    
    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Normalize weights to sum to 1.0"""
        
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            normalized = weights / weight_sum
        else:
            # Fallback to equal weights
            normalized = np.ones(5) / 5.0
            self.weight_violations += 1
        
        # Final validation
        if not np.isclose(np.sum(normalized), 1.0, atol=1e-6):
            logger.warning("Weight normalization failed", 
                          weights=normalized.tolist(),
                          sum=np.sum(normalized))
            normalized = np.array([0.2, 0.4, 0.1, 0.25, 0.05])  # Safe fallback
        
        return normalized
    
    def _calculate_confidence(self, 
                            weights: np.ndarray,
                            risk_state: RiskState,
                            market_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in portfolio allocation"""
        
        # Regime confidence
        regime_confidence = market_analysis.get('regime_confidence', 0.5)
        
        # Diversification confidence - higher when well diversified
        weight_entropy = -np.sum(weights * np.log(weights + 1e-8))
        max_entropy = np.log(len(weights))
        diversification_confidence = weight_entropy / max_entropy
        
        # Constraint satisfaction confidence
        constraint_violations = 0
        if np.any(weights < self.min_weight) or np.any(weights > self.max_weight):
            constraint_violations += 1
        if not np.isclose(np.sum(weights), 1.0, atol=1e-3):
            constraint_violations += 1
        
        constraint_confidence = max(0.0, 1.0 - constraint_violations * 0.3)
        
        # Historical performance confidence (simplified)
        if len(self.performance_history) > 10:
            recent_performance = list(self.performance_history)[-10:]
            avg_performance = np.mean(recent_performance)
            performance_confidence = max(0.3, min(0.9, 0.5 + avg_performance))
        else:
            performance_confidence = 0.6
        
        # Market condition clarity
        if risk_state.market_stress_level > 0.8 or risk_state.market_stress_level < 0.2:
            market_confidence = 0.8  # Clear market signal
        else:
            market_confidence = 0.5  # Ambiguous conditions
        
        # Combined confidence
        confidence = (regime_confidence * 0.25 +
                     diversification_confidence * 0.25 +
                     constraint_confidence * 0.2 +
                     performance_confidence * 0.15 +
                     market_confidence * 0.15)
        
        return max(0.1, min(1.0, confidence))
    
    def _update_optimizer_state(self, weights: np.ndarray, risk_state: RiskState):
        """Update internal optimizer state"""
        
        # Check if significant rebalancing occurred
        weight_change = np.sum(np.abs(weights - self.current_weights))
        if weight_change > self.rebalance_threshold:
            self.rebalances_executed += 1
            self.last_rebalance_time = datetime.now()
        
        # Update current weights
        self.current_weights = weights.copy()
        self.target_weights = weights.copy()
        
        # Record allocation history
        self.allocation_history.append({
            'timestamp': datetime.now(),
            'weights': weights.copy(),
            'regime': 'normal',  # Would be determined from market analysis
            'stress_level': risk_state.market_stress_level
        })
        
        # Record performance (simplified - would need actual returns)
        portfolio_return = np.random.normal(0.0, 0.01)  # Placeholder
        self.performance_history.append(portfolio_return)
    
    def _log_optimization_decision(self, 
                                 weights: np.ndarray,
                                 market_analysis: Dict[str, Any],
                                 confidence: float):
        """Log optimization decision for transparency"""
        
        weight_dict = {
            asset: f"{weight:.3f}" 
            for asset, weight in zip(self.asset_classes, weights)
        }
        
        logger.info("Portfolio optimization decision",
                   weights=weight_dict,
                   confidence=f"{confidence:.3f}",
                   regime=market_analysis.get('regime', 'unknown'),
                   stress_level=f"{market_analysis.get('stress_level', 0):.3f}",
                   correlation_level=f"{market_analysis.get('correlation_level', 0):.3f}",
                   rebalances=self.rebalances_executed)
    
    def validate_risk_constraints(self, risk_state: RiskState) -> bool:
        """Validate portfolio constraints"""
        
        violations = []
        
        # Check weight constraints
        if np.any(self.current_weights < self.min_weight):
            violations.append(f"min_weight_violation: {self.current_weights.min():.3f} < {self.min_weight}")
        
        if np.any(self.current_weights > self.max_weight):
            violations.append(f"max_weight_violation: {self.current_weights.max():.3f} > {self.max_weight}")
        
        # Check weight sum
        weight_sum = np.sum(self.current_weights)
        if not np.isclose(weight_sum, 1.0, atol=1e-2):
            violations.append(f"weight_sum_violation: {weight_sum:.3f} != 1.0")
        
        # Check liquidity buffer
        cash_weight = self.current_weights[3]  # Cash is index 3
        if risk_state.market_stress_level > 0.7 and cash_weight < self.liquidity_buffer:
            violations.append(f"liquidity_buffer_violation: {cash_weight:.3f} < {self.liquidity_buffer}")
        
        if violations:
            logger.warning("Portfolio constraint violations", violations=violations)
            return False
        
        return True
    
    def _get_safe_action(self) -> np.ndarray:
        """Get safe default action for error cases"""
        # Conservative allocation: higher cash, lower risk assets
        return np.array([0.2, 0.4, 0.05, 0.3, 0.05], dtype=np.float32)
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get portfolio optimizer agent metrics"""
        base_metrics = super().get_risk_metrics()
        
        # Calculate optimizer-specific metrics
        avg_performance = np.mean(self.performance_history) if self.performance_history else 0.0
        performance_volatility = np.std(self.performance_history) if len(self.performance_history) > 1 else 0.0
        
        sharpe_ratio = avg_performance / max(performance_volatility, 0.001) if performance_volatility > 0 else 0.0
        
        return RiskMetrics(
            total_risk_decisions=self.rebalances_executed,
            risk_events_detected=self.correlation_adjustments,
            false_positive_rate=base_metrics.false_positive_rate,
            avg_response_time_ms=base_metrics.avg_response_time_ms,
            risk_adjusted_return=avg_performance,
            max_drawdown=min(self.performance_history) if self.performance_history else 0.0,
            sharpe_ratio=sharpe_ratio,
            var_accuracy=0.0,  # Not directly applicable
            correlation_prediction_accuracy=0.0  # Not directly applicable
        )
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary"""
        return {
            'current_weights': {
                asset: float(weight) 
                for asset, weight in zip(self.asset_classes, self.current_weights)
            },
            'rebalances_executed': self.rebalances_executed,
            'correlation_adjustments': self.correlation_adjustments,
            'regime_changes': self.regime_changes,
            'weight_violations': self.weight_violations,
            'last_rebalance': self.last_rebalance_time.isoformat(),
            'avg_performance': np.mean(self.performance_history) if self.performance_history else 0.0,
            'portfolio_volatility': np.std(self.performance_history) if len(self.performance_history) > 1 else 0.0
        }
    
    def reset(self):
        """Reset agent state"""
        super().reset()
        self.current_weights = np.array([0.6, 0.25, 0.05, 0.05, 0.05])
        self.target_weights = self.current_weights.copy()
        self.rebalances_executed = 0
        self.correlation_adjustments = 0
        self.regime_changes = 0
        self.weight_violations = 0
        self.last_rebalance_time = datetime.now()
        self.allocation_history.clear()
        self.performance_history.clear()
        self.correlation_history.clear()
        logger.info("Portfolio Optimizer Agent reset")