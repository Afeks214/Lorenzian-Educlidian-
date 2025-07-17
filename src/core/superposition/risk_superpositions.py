"""
Risk Management Superposition Classes for MARL Risk Agents.

This module provides specialized superposition implementations for risk management agents
focusing on position sizing, stop/target management, risk monitoring, and portfolio optimization.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
import structlog

from .base_superposition import UniversalSuperposition, SuperpositionState

logger = structlog.get_logger()


class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    EXTREME = "extreme"


class PositionAction(Enum):
    """Position management actions"""
    HOLD = "hold"
    REDUCE = "reduce"
    INCREASE = "increase"
    CLOSE = "close"
    HEDGE = "hedge"


class RiskAlert(Enum):
    """Risk alert types"""
    DRAWDOWN_WARNING = "drawdown_warning"
    VOLATILITY_SPIKE = "volatility_spike"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    MARGIN_CALL = "margin_call"


class PositionSizingSuperposition(UniversalSuperposition):
    """
    Specialized superposition for Position Sizing agents.
    
    Focuses on optimal position sizing, Kelly criterion, risk-based sizing,
    and dynamic position adjustment with enhanced attention mechanisms.
    """
    
    def get_agent_type(self) -> str:
        return "PositionSizing"
    
    def get_state_dimension(self) -> int:
        return 18  # Position sizing state dimension
    
    def _initialize_domain_features(self) -> None:
        """Initialize Position Sizing-specific domain features"""
        self.domain_features = {
            # Position Size Calculations
            'optimal_position_size': 0.0,
            'kelly_position_size': 0.0,
            'risk_parity_size': 0.0,
            'volatility_adjusted_size': 0.0,
            'correlation_adjusted_size': 0.0,
            
            # Risk Metrics
            'portfolio_heat': 0.0,
            'var_contribution': 0.0,
            'marginal_var': 0.0,
            'component_var': 0.0,
            'risk_budget_utilization': 0.0,
            
            # Kelly Criterion
            'kelly_fraction': 0.0,
            'win_probability': 0.0,
            'win_loss_ratio': 0.0,
            'expected_value': 0.0,
            'kelly_confidence': 0.0,
            
            # Position Adjustments
            'size_adjustment_factor': 1.0,
            'drawdown_adjustment': 1.0,
            'volatility_adjustment': 1.0,
            'correlation_adjustment': 1.0,
            
            # Performance Metrics
            'sizing_accuracy': 0.0,
            'risk_adjusted_return': 0.0,
            'position_efficiency': 0.0,
            'sizing_consistency': 0.0
        }
        
        # Position sizing-specific attention weights
        self.attention_weights = {
            'size_calculation': 0.35,
            'risk_assessment': 0.3,
            'kelly_analysis': 0.2,
            'adjustment_factors': 0.15
        }
        
        self.update_reasoning_chain("Position Sizing superposition initialized")
    
    def calculate_optimal_position_size(self, 
                                      trade_setup: Dict[str, Any],
                                      portfolio_state: Dict[str, Any],
                                      risk_parameters: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate optimal position size using multiple methods
        
        Args:
            trade_setup: Trade setup information
            portfolio_state: Current portfolio state
            risk_parameters: Risk management parameters
            
        Returns:
            Position sizing results
        """
        self.update_reasoning_chain("Calculating optimal position size")
        
        # Extract key parameters
        entry_price = trade_setup.get('entry_price', 0.0)
        stop_price = trade_setup.get('stop_price', 0.0)
        target_price = trade_setup.get('target_price', 0.0)
        account_equity = portfolio_state.get('account_equity', 100000.0)
        max_risk_per_trade = risk_parameters.get('max_risk_per_trade', 0.02)
        
        # Calculate basic risk metrics
        if entry_price > 0 and stop_price > 0:
            risk_per_unit = abs(entry_price - stop_price)
            reward_per_unit = abs(target_price - entry_price) if target_price > 0 else risk_per_unit
            risk_reward_ratio = reward_per_unit / risk_per_unit if risk_per_unit > 0 else 0.0
        else:
            risk_per_unit = 0.0
            reward_per_unit = 0.0
            risk_reward_ratio = 0.0
        
        # Method 1: Fixed Risk Position Sizing
        if risk_per_unit > 0:
            max_risk_amount = account_equity * max_risk_per_trade
            fixed_risk_size = max_risk_amount / risk_per_unit
        else:
            fixed_risk_size = 0.0
        
        # Method 2: Kelly Criterion
        kelly_size = self._calculate_kelly_position_size(
            trade_setup, portfolio_state, risk_parameters
        )
        
        # Method 3: Volatility-Adjusted Sizing
        volatility_adjusted_size = self._calculate_volatility_adjusted_size(
            fixed_risk_size, portfolio_state, risk_parameters
        )
        
        # Method 4: Correlation-Adjusted Sizing
        correlation_adjusted_size = self._calculate_correlation_adjusted_size(
            fixed_risk_size, portfolio_state, risk_parameters
        )
        
        # Method 5: Risk Parity Sizing
        risk_parity_size = self._calculate_risk_parity_size(
            trade_setup, portfolio_state, risk_parameters
        )
        
        # Combine methods with weights
        method_weights = {
            'fixed_risk': 0.3,
            'kelly': 0.25,
            'volatility_adjusted': 0.2,
            'correlation_adjusted': 0.15,
            'risk_parity': 0.1
        }
        
        optimal_size = (
            fixed_risk_size * method_weights['fixed_risk'] +
            kelly_size * method_weights['kelly'] +
            volatility_adjusted_size * method_weights['volatility_adjusted'] +
            correlation_adjusted_size * method_weights['correlation_adjusted'] +
            risk_parity_size * method_weights['risk_parity']
        )
        
        # Apply adjustment factors
        optimal_size *= self._calculate_adjustment_factors(portfolio_state, risk_parameters)
        
        # Update domain features
        self.domain_features['optimal_position_size'] = optimal_size
        self.domain_features['kelly_position_size'] = kelly_size
        self.domain_features['risk_parity_size'] = risk_parity_size
        self.domain_features['volatility_adjusted_size'] = volatility_adjusted_size
        self.domain_features['correlation_adjusted_size'] = correlation_adjusted_size
        
        self.add_attention_weight('size_calculation', min(optimal_size / 1000, 1.0))
        
        return {
            'optimal_size': optimal_size,
            'fixed_risk_size': fixed_risk_size,
            'kelly_size': kelly_size,
            'volatility_adjusted_size': volatility_adjusted_size,
            'correlation_adjusted_size': correlation_adjusted_size,
            'risk_parity_size': risk_parity_size,
            'risk_reward_ratio': risk_reward_ratio
        }
    
    def calculate_kelly_criterion(self, 
                                trade_history: List[Dict[str, Any]],
                                current_setup: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate Kelly criterion for position sizing
        
        Args:
            trade_history: Historical trade data
            current_setup: Current trade setup
            
        Returns:
            Kelly criterion results
        """
        self.update_reasoning_chain("Calculating Kelly criterion")
        
        if not trade_history:
            return {'kelly_fraction': 0.0, 'win_probability': 0.0, 'win_loss_ratio': 0.0}
        
        # Calculate win probability
        winning_trades = [trade for trade in trade_history if trade.get('pnl', 0) > 0]
        losing_trades = [trade for trade in trade_history if trade.get('pnl', 0) < 0]
        
        win_probability = len(winning_trades) / len(trade_history)
        loss_probability = 1.0 - win_probability
        
        # Calculate average win/loss
        if winning_trades:
            avg_win = np.mean([trade['pnl'] for trade in winning_trades])
        else:
            avg_win = 0.0
        
        if losing_trades:
            avg_loss = abs(np.mean([trade['pnl'] for trade in losing_trades]))
        else:
            avg_loss = 1.0  # Avoid division by zero
        
        # Calculate win/loss ratio
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
        
        # Calculate Kelly fraction
        if loss_probability > 0:
            kelly_fraction = (win_probability * win_loss_ratio - loss_probability) / win_loss_ratio
        else:
            kelly_fraction = 0.0
        
        # Apply safety factors
        kelly_fraction = max(0.0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        # Calculate expected value
        expected_value = win_probability * avg_win - loss_probability * avg_loss
        
        # Calculate Kelly confidence based on sample size
        kelly_confidence = min(len(trade_history) / 100.0, 1.0)
        
        # Update domain features
        self.domain_features['kelly_fraction'] = kelly_fraction
        self.domain_features['win_probability'] = win_probability
        self.domain_features['win_loss_ratio'] = win_loss_ratio
        self.domain_features['expected_value'] = expected_value
        self.domain_features['kelly_confidence'] = kelly_confidence
        
        self.add_attention_weight('kelly_analysis', kelly_confidence)
        
        return {
            'kelly_fraction': kelly_fraction,
            'win_probability': win_probability,
            'win_loss_ratio': win_loss_ratio,
            'expected_value': expected_value,
            'kelly_confidence': kelly_confidence
        }
    
    def _calculate_kelly_position_size(self, 
                                     trade_setup: Dict[str, Any],
                                     portfolio_state: Dict[str, Any],
                                     risk_parameters: Dict[str, Any]) -> float:
        """Calculate Kelly-based position size"""
        kelly_fraction = self.domain_features.get('kelly_fraction', 0.0)
        account_equity = portfolio_state.get('account_equity', 100000.0)
        
        # Conservative Kelly (fractional Kelly)
        fractional_kelly = kelly_fraction * 0.5  # Use half-Kelly for safety
        
        kelly_size = account_equity * fractional_kelly
        
        return max(0.0, kelly_size)
    
    def _calculate_volatility_adjusted_size(self, 
                                          base_size: float,
                                          portfolio_state: Dict[str, Any],
                                          risk_parameters: Dict[str, Any]) -> float:
        """Calculate volatility-adjusted position size"""
        current_volatility = portfolio_state.get('portfolio_volatility', 0.01)
        target_volatility = risk_parameters.get('target_volatility', 0.01)
        
        if current_volatility > 0:
            volatility_adjustment = target_volatility / current_volatility
        else:
            volatility_adjustment = 1.0
        
        # Apply limits
        volatility_adjustment = max(0.5, min(volatility_adjustment, 2.0))
        
        self.domain_features['volatility_adjustment'] = volatility_adjustment
        
        return base_size * volatility_adjustment
    
    def _calculate_correlation_adjusted_size(self, 
                                           base_size: float,
                                           portfolio_state: Dict[str, Any],
                                           risk_parameters: Dict[str, Any]) -> float:
        """Calculate correlation-adjusted position size"""
        portfolio_correlation = portfolio_state.get('avg_correlation', 0.0)
        max_correlation = risk_parameters.get('max_correlation', 0.7)
        
        if portfolio_correlation > max_correlation:
            correlation_adjustment = max_correlation / portfolio_correlation
        else:
            correlation_adjustment = 1.0
        
        # Apply limits
        correlation_adjustment = max(0.3, min(correlation_adjustment, 1.0))
        
        self.domain_features['correlation_adjustment'] = correlation_adjustment
        
        return base_size * correlation_adjustment
    
    def _calculate_risk_parity_size(self, 
                                   trade_setup: Dict[str, Any],
                                   portfolio_state: Dict[str, Any],
                                   risk_parameters: Dict[str, Any]) -> float:
        """Calculate risk parity position size"""
        target_risk_contribution = risk_parameters.get('target_risk_contribution', 0.1)
        portfolio_var = portfolio_state.get('portfolio_var', 0.0)
        
        if portfolio_var > 0:
            risk_parity_size = target_risk_contribution / portfolio_var
        else:
            risk_parity_size = 0.0
        
        return max(0.0, risk_parity_size)
    
    def _calculate_adjustment_factors(self, 
                                    portfolio_state: Dict[str, Any],
                                    risk_parameters: Dict[str, Any]) -> float:
        """Calculate overall adjustment factors"""
        adjustment_factor = 1.0
        
        # Drawdown adjustment
        current_drawdown = portfolio_state.get('current_drawdown', 0.0)
        max_drawdown = risk_parameters.get('max_drawdown', 0.1)
        
        if current_drawdown > max_drawdown * 0.5:
            drawdown_adjustment = 1.0 - (current_drawdown / max_drawdown) * 0.5
            adjustment_factor *= max(0.2, drawdown_adjustment)
        
        # Market regime adjustment
        market_regime = portfolio_state.get('market_regime', 'normal')
        if market_regime == 'crisis':
            adjustment_factor *= 0.5
        elif market_regime == 'volatile':
            adjustment_factor *= 0.7
        
        self.domain_features['size_adjustment_factor'] = adjustment_factor
        
        return adjustment_factor


class StopTargetSuperposition(UniversalSuperposition):
    """
    Specialized superposition for Stop/Target management agents.
    
    Focuses on dynamic stop loss and take profit management, trailing stops,
    and risk-reward optimization with enhanced attention mechanisms.
    """
    
    def get_agent_type(self) -> str:
        return "StopTarget"
    
    def get_state_dimension(self) -> int:
        return 16  # Stop/target state dimension
    
    def _initialize_domain_features(self) -> None:
        """Initialize Stop/Target-specific domain features"""
        self.domain_features = {
            # Stop Loss Management
            'initial_stop_price': 0.0,
            'current_stop_price': 0.0,
            'stop_distance': 0.0,
            'stop_adjustment_count': 0,
            'trailing_stop_active': False,
            
            # Take Profit Management
            'initial_target_price': 0.0,
            'current_target_price': 0.0,
            'target_distance': 0.0,
            'target_adjustment_count': 0,
            'partial_profit_taken': 0.0,
            
            # Risk-Reward Metrics
            'current_risk_reward': 0.0,
            'initial_risk_reward': 0.0,
            'risk_reward_improvement': 0.0,
            'profit_potential': 0.0,
            'risk_exposure': 0.0,
            
            # Dynamic Adjustments
            'volatility_adjustment': 1.0,
            'trend_adjustment': 1.0,
            'time_adjustment': 1.0,
            'support_resistance_adjustment': 1.0,
            
            # Performance Metrics
            'stop_hit_rate': 0.0,
            'target_hit_rate': 0.0,
            'avg_risk_reward_achieved': 0.0,
            'stop_target_efficiency': 0.0
        }
        
        # Stop/target-specific attention weights
        self.attention_weights = {
            'stop_management': 0.3,
            'target_management': 0.3,
            'risk_reward_optimization': 0.25,
            'dynamic_adjustment': 0.15
        }
        
        self.update_reasoning_chain("Stop/Target superposition initialized")
    
    def initialize_stop_target_levels(self, 
                                    trade_setup: Dict[str, Any],
                                    market_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Initialize stop loss and take profit levels
        
        Args:
            trade_setup: Trade setup information
            market_data: Current market data
            
        Returns:
            Initial stop/target levels
        """
        self.update_reasoning_chain("Initializing stop/target levels")
        
        entry_price = trade_setup.get('entry_price', 0.0)
        direction = trade_setup.get('direction', 1)  # 1 for long, -1 for short
        volatility = market_data.get('volatility', 0.01)
        atr = market_data.get('atr', volatility * entry_price if entry_price > 0 else 0.01)
        
        # Calculate initial stop distance based on ATR
        stop_multiplier = trade_setup.get('stop_multiplier', 2.0)
        stop_distance = atr * stop_multiplier
        
        # Calculate initial stop price
        if direction > 0:  # Long position
            initial_stop_price = entry_price - stop_distance
        else:  # Short position
            initial_stop_price = entry_price + stop_distance
        
        # Calculate initial target price (2:1 risk-reward)
        target_multiplier = trade_setup.get('target_multiplier', 2.0)
        target_distance = stop_distance * target_multiplier
        
        if direction > 0:  # Long position
            initial_target_price = entry_price + target_distance
        else:  # Short position
            initial_target_price = entry_price - target_distance
        
        # Calculate initial risk-reward ratio
        risk = abs(entry_price - initial_stop_price)
        reward = abs(initial_target_price - entry_price)
        initial_risk_reward = reward / risk if risk > 0 else 0.0
        
        # Update domain features
        self.domain_features['initial_stop_price'] = initial_stop_price
        self.domain_features['current_stop_price'] = initial_stop_price
        self.domain_features['stop_distance'] = stop_distance
        self.domain_features['initial_target_price'] = initial_target_price
        self.domain_features['current_target_price'] = initial_target_price
        self.domain_features['target_distance'] = target_distance
        self.domain_features['initial_risk_reward'] = initial_risk_reward
        self.domain_features['current_risk_reward'] = initial_risk_reward
        
        self.add_attention_weight('stop_management', 0.3)
        self.add_attention_weight('target_management', 0.3)
        
        return {
            'initial_stop_price': initial_stop_price,
            'initial_target_price': initial_target_price,
            'stop_distance': stop_distance,
            'target_distance': target_distance,
            'initial_risk_reward': initial_risk_reward
        }
    
    def update_trailing_stop(self, 
                           current_price: float,
                           position_data: Dict[str, Any],
                           market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update trailing stop loss
        
        Args:
            current_price: Current market price
            position_data: Position information
            market_data: Current market data
            
        Returns:
            Updated stop information
        """
        self.update_reasoning_chain("Updating trailing stop")
        
        direction = position_data.get('direction', 1)
        entry_price = position_data.get('entry_price', 0.0)
        current_stop = self.domain_features.get('current_stop_price', 0.0)
        
        # Calculate trailing stop parameters
        atr = market_data.get('atr', 0.01)
        trailing_multiplier = position_data.get('trailing_multiplier', 2.0)
        trailing_distance = atr * trailing_multiplier
        
        # Calculate new trailing stop
        if direction > 0:  # Long position
            new_stop = current_price - trailing_distance
            # Only move stop up, never down
            updated_stop = max(current_stop, new_stop)
        else:  # Short position
            new_stop = current_price + trailing_distance
            # Only move stop down, never up
            updated_stop = min(current_stop, new_stop)
        
        # Check if stop should be activated
        if not self.domain_features.get('trailing_stop_active', False):
            # Activate trailing stop when position is profitable
            if direction > 0 and current_price > entry_price:
                self.domain_features['trailing_stop_active'] = True
            elif direction < 0 and current_price < entry_price:
                self.domain_features['trailing_stop_active'] = True
        
        # Update stop if trailing is active
        if self.domain_features.get('trailing_stop_active', False):
            if updated_stop != current_stop:
                self.domain_features['current_stop_price'] = updated_stop
                self.domain_features['stop_adjustment_count'] += 1
                
                # Update risk-reward ratio
                new_risk = abs(current_price - updated_stop)
                current_target = self.domain_features.get('current_target_price', 0.0)
                new_reward = abs(current_target - current_price)
                new_risk_reward = new_reward / new_risk if new_risk > 0 else 0.0
                
                self.domain_features['current_risk_reward'] = new_risk_reward
                self.domain_features['risk_reward_improvement'] = new_risk_reward - self.domain_features.get('initial_risk_reward', 0.0)
        
        self.add_attention_weight('stop_management', 0.4)
        
        return {
            'updated_stop': updated_stop,
            'stop_moved': updated_stop != current_stop,
            'trailing_active': self.domain_features.get('trailing_stop_active', False),
            'new_risk_reward': self.domain_features.get('current_risk_reward', 0.0)
        }
    
    def manage_partial_profits(self, 
                             current_price: float,
                             position_data: Dict[str, Any],
                             profit_rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Manage partial profit taking
        
        Args:
            current_price: Current market price
            position_data: Position information
            profit_rules: Profit taking rules
            
        Returns:
            Partial profit management results
        """
        self.update_reasoning_chain("Managing partial profits")
        
        entry_price = position_data.get('entry_price', 0.0)
        direction = position_data.get('direction', 1)
        position_size = position_data.get('position_size', 0.0)
        
        # Calculate current profit
        if direction > 0:  # Long position
            current_profit = (current_price - entry_price) / entry_price
        else:  # Short position
            current_profit = (entry_price - current_price) / entry_price
        
        # Profit taking levels
        profit_levels = profit_rules.get('profit_levels', [0.01, 0.02, 0.03])
        profit_percentages = profit_rules.get('profit_percentages', [0.25, 0.25, 0.25])
        
        total_profit_taken = self.domain_features.get('partial_profit_taken', 0.0)
        profit_actions = []
        
        for i, (level, percentage) in enumerate(zip(profit_levels, profit_percentages)):
            if current_profit >= level and total_profit_taken < sum(profit_percentages[:i+1]):
                # Take partial profit
                profit_to_take = percentage * position_size
                profit_actions.append({
                    'action': 'take_profit',
                    'level': level,
                    'amount': profit_to_take,
                    'price': current_price
                })
                
                total_profit_taken += percentage
        
        # Update domain features
        self.domain_features['partial_profit_taken'] = total_profit_taken
        
        # Adjust target price based on partial profits taken
        if profit_actions:
            remaining_position = 1.0 - total_profit_taken
            if remaining_position > 0:
                # Move target further for remaining position
                current_target = self.domain_features.get('current_target_price', 0.0)
                target_extension = profit_rules.get('target_extension', 1.5)
                
                if direction > 0:
                    new_target = current_price + (current_target - current_price) * target_extension
                else:
                    new_target = current_price - (current_price - current_target) * target_extension
                
                self.domain_features['current_target_price'] = new_target
                self.domain_features['target_adjustment_count'] += 1
        
        self.add_attention_weight('target_management', min(total_profit_taken, 1.0))
        
        return {
            'profit_actions': profit_actions,
            'total_profit_taken': total_profit_taken,
            'current_profit': current_profit,
            'remaining_position': 1.0 - total_profit_taken
        }


class RiskMonitorSuperposition(UniversalSuperposition):
    """
    Specialized superposition for Risk Monitoring agents.
    
    Focuses on real-time risk monitoring, risk alerts, and risk limit enforcement
    with enhanced attention mechanisms for risk event detection.
    """
    
    def get_agent_type(self) -> str:
        return "RiskMonitor"
    
    def get_state_dimension(self) -> int:
        return 20  # Risk monitoring state dimension
    
    def _initialize_domain_features(self) -> None:
        """Initialize Risk Monitoring-specific domain features"""
        self.domain_features = {
            # Risk Metrics
            'current_risk_level': RiskLevel.LOW,
            'portfolio_var': 0.0,
            'portfolio_var_limit': 0.0,
            'current_drawdown': 0.0,
            'max_drawdown_limit': 0.0,
            'risk_utilization': 0.0,
            
            # Risk Alerts
            'active_alerts': [],
            'alert_count': 0,
            'alert_severity': 0.0,
            'alert_history': [],
            
            # Risk Limits
            'position_limit_usage': 0.0,
            'concentration_limit_usage': 0.0,
            'leverage_limit_usage': 0.0,
            'correlation_limit_usage': 0.0,
            
            # Risk Monitoring
            'risk_trend': 0.0,
            'risk_velocity': 0.0,
            'risk_acceleration': 0.0,
            'risk_stability': 0.0,
            
            # Risk Events
            'risk_event_probability': 0.0,
            'stress_test_result': 0.0,
            'liquidity_risk_score': 0.0,
            'operational_risk_score': 0.0,
            'market_risk_score': 0.0
        }
        
        # Risk monitoring-specific attention weights
        self.attention_weights = {
            'risk_measurement': 0.3,
            'alert_management': 0.25,
            'limit_monitoring': 0.25,
            'risk_forecasting': 0.2
        }
        
        self.update_reasoning_chain("Risk Monitor superposition initialized")
    
    def monitor_risk_metrics(self, 
                           portfolio_data: Dict[str, Any],
                           market_data: Dict[str, Any],
                           risk_limits: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor key risk metrics
        
        Args:
            portfolio_data: Portfolio information
            market_data: Market data
            risk_limits: Risk limits configuration
            
        Returns:
            Risk monitoring results
        """
        self.update_reasoning_chain("Monitoring risk metrics")
        
        # Calculate portfolio VaR
        portfolio_var = self._calculate_portfolio_var(portfolio_data, market_data)
        var_limit = risk_limits.get('var_limit', 0.05)
        var_utilization = portfolio_var / var_limit if var_limit > 0 else 0.0
        
        # Calculate current drawdown
        current_drawdown = portfolio_data.get('current_drawdown', 0.0)
        max_drawdown_limit = risk_limits.get('max_drawdown', 0.1)
        drawdown_utilization = current_drawdown / max_drawdown_limit if max_drawdown_limit > 0 else 0.0
        
        # Calculate risk utilization
        risk_utilization = max(var_utilization, drawdown_utilization)
        
        # Determine risk level
        if risk_utilization > 0.9:
            risk_level = RiskLevel.EXTREME
        elif risk_utilization > 0.7:
            risk_level = RiskLevel.CRITICAL
        elif risk_utilization > 0.5:
            risk_level = RiskLevel.HIGH
        elif risk_utilization > 0.3:
            risk_level = RiskLevel.MODERATE
        else:
            risk_level = RiskLevel.LOW
        
        # Update domain features
        self.domain_features['current_risk_level'] = risk_level
        self.domain_features['portfolio_var'] = portfolio_var
        self.domain_features['portfolio_var_limit'] = var_limit
        self.domain_features['current_drawdown'] = current_drawdown
        self.domain_features['max_drawdown_limit'] = max_drawdown_limit
        self.domain_features['risk_utilization'] = risk_utilization
        
        self.add_attention_weight('risk_measurement', risk_utilization)
        
        return {
            'risk_level': risk_level,
            'portfolio_var': portfolio_var,
            'risk_utilization': risk_utilization,
            'var_utilization': var_utilization,
            'drawdown_utilization': drawdown_utilization
        }
    
    def generate_risk_alerts(self, 
                           risk_metrics: Dict[str, Any],
                           portfolio_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate risk alerts based on current conditions
        
        Args:
            risk_metrics: Risk metrics
            portfolio_data: Portfolio data
            
        Returns:
            List of risk alerts
        """
        self.update_reasoning_chain("Generating risk alerts")
        
        alerts = []
        
        # VaR alert
        var_utilization = risk_metrics.get('var_utilization', 0.0)
        if var_utilization > 0.8:
            alerts.append({
                'type': 'VAR_BREACH',
                'severity': 'HIGH' if var_utilization > 0.9 else 'MODERATE',
                'message': f'Portfolio VaR utilization at {var_utilization:.1%}',
                'timestamp': datetime.now(),
                'recommended_action': 'REDUCE_POSITIONS'
            })
        
        # Drawdown alert
        drawdown_utilization = risk_metrics.get('drawdown_utilization', 0.0)
        if drawdown_utilization > 0.7:
            alerts.append({
                'type': RiskAlert.DRAWDOWN_WARNING,
                'severity': 'HIGH' if drawdown_utilization > 0.9 else 'MODERATE',
                'message': f'Drawdown at {drawdown_utilization:.1%} of limit',
                'timestamp': datetime.now(),
                'recommended_action': 'REDUCE_RISK'
            })
        
        # Volatility alert
        current_volatility = portfolio_data.get('volatility', 0.0)
        historical_volatility = portfolio_data.get('historical_volatility', 0.01)
        if current_volatility > historical_volatility * 2:
            alerts.append({
                'type': RiskAlert.VOLATILITY_SPIKE,
                'severity': 'MODERATE',
                'message': f'Volatility spike detected: {current_volatility:.2%}',
                'timestamp': datetime.now(),
                'recommended_action': 'MONITOR_CLOSELY'
            })
        
        # Correlation alert
        avg_correlation = portfolio_data.get('avg_correlation', 0.0)
        if avg_correlation > 0.8:
            alerts.append({
                'type': RiskAlert.CORRELATION_BREAKDOWN,
                'severity': 'HIGH',
                'message': f'High correlation detected: {avg_correlation:.2f}',
                'timestamp': datetime.now(),
                'recommended_action': 'DIVERSIFY'
            })
        
        # Liquidity alert
        liquidity_score = portfolio_data.get('liquidity_score', 1.0)
        if liquidity_score < 0.3:
            alerts.append({
                'type': RiskAlert.LIQUIDITY_CRISIS,
                'severity': 'CRITICAL',
                'message': f'Low liquidity detected: {liquidity_score:.2f}',
                'timestamp': datetime.now(),
                'recommended_action': 'REDUCE_POSITIONS'
            })
        
        # Update domain features
        self.domain_features['active_alerts'] = alerts
        self.domain_features['alert_count'] = len(alerts)
        self.domain_features['alert_severity'] = max([self._get_severity_score(alert['severity']) for alert in alerts], default=0.0)
        
        self.add_attention_weight('alert_management', min(len(alerts) * 0.2, 1.0))
        
        return alerts
    
    def _calculate_portfolio_var(self, 
                               portfolio_data: Dict[str, Any],
                               market_data: Dict[str, Any],
                               confidence_level: float = 0.95) -> float:
        """Calculate portfolio Value at Risk"""
        # Simplified VaR calculation
        portfolio_value = portfolio_data.get('total_value', 0.0)
        portfolio_volatility = portfolio_data.get('volatility', 0.01)
        
        # Normal VaR calculation
        from scipy.stats import norm
        var_multiplier = norm.ppf(confidence_level)
        portfolio_var = portfolio_value * portfolio_volatility * var_multiplier
        
        return portfolio_var
    
    def _get_severity_score(self, severity: str) -> float:
        """Convert severity string to numeric score"""
        severity_map = {
            'LOW': 0.2,
            'MODERATE': 0.5,
            'HIGH': 0.8,
            'CRITICAL': 1.0
        }
        return severity_map.get(severity, 0.0)


class PortfolioOptimizerSuperposition(UniversalSuperposition):
    """
    Specialized superposition for Portfolio Optimizer agents.
    
    Focuses on portfolio optimization, asset allocation, and rebalancing
    with enhanced attention mechanisms for optimization efficiency.
    """
    
    def get_agent_type(self) -> str:
        return "PortfolioOptimizer"
    
    def get_state_dimension(self) -> int:
        return 22  # Portfolio optimization state dimension
    
    def _initialize_domain_features(self) -> None:
        """Initialize Portfolio Optimizer-specific domain features"""
        self.domain_features = {
            # Portfolio Weights
            'current_weights': np.zeros(10),
            'target_weights': np.zeros(10),
            'optimal_weights': np.zeros(10),
            'weight_deviations': np.zeros(10),
            
            # Optimization Results
            'expected_return': 0.0,
            'expected_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'optimization_score': 0.0,
            
            # Risk Metrics
            'portfolio_beta': 0.0,
            'tracking_error': 0.0,
            'information_ratio': 0.0,
            'maximum_drawdown': 0.0,
            
            # Rebalancing
            'rebalancing_frequency': 0,
            'rebalancing_threshold': 0.0,
            'rebalancing_cost': 0.0,
            'last_rebalance_date': None,
            
            # Constraints
            'position_limits': np.zeros(10),
            'sector_limits': np.zeros(5),
            'leverage_limit': 0.0,
            'turnover_limit': 0.0,
            
            # Performance
            'optimization_accuracy': 0.0,
            'rebalancing_efficiency': 0.0,
            'cost_efficiency': 0.0,
            'risk_adjusted_performance': 0.0
        }
        
        # Portfolio optimization-specific attention weights
        self.attention_weights = {
            'weight_optimization': 0.3,
            'risk_management': 0.25,
            'rebalancing': 0.25,
            'performance_monitoring': 0.2
        }
        
        self.update_reasoning_chain("Portfolio Optimizer superposition initialized")
    
    def optimize_portfolio_weights(self, 
                                 expected_returns: np.ndarray,
                                 covariance_matrix: np.ndarray,
                                 constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize portfolio weights using modern portfolio theory
        
        Args:
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            constraints: Optimization constraints
            
        Returns:
            Optimization results
        """
        self.update_reasoning_chain("Optimizing portfolio weights")
        
        n_assets = len(expected_returns)
        
        # Mean-variance optimization (simplified)
        risk_aversion = constraints.get('risk_aversion', 1.0)
        min_weight = constraints.get('min_weight', 0.0)
        max_weight = constraints.get('max_weight', 1.0)
        
        # Objective function: maximize return - risk_aversion * variance
        try:
            # Inverse of covariance matrix
            inv_cov = np.linalg.inv(covariance_matrix + np.eye(n_assets) * 1e-8)
            
            # Optimal weights (analytical solution for unconstrained case)
            ones = np.ones(n_assets)
            optimal_weights = inv_cov @ (expected_returns - risk_aversion * ones)
            
            # Normalize to sum to 1
            optimal_weights = optimal_weights / np.sum(optimal_weights)
            
            # Apply constraints
            optimal_weights = np.clip(optimal_weights, min_weight, max_weight)
            optimal_weights = optimal_weights / np.sum(optimal_weights)  # Renormalize
            
        except np.linalg.LinAlgError:
            # Fallback to equal weights if optimization fails
            optimal_weights = np.ones(n_assets) / n_assets
        
        # Calculate portfolio metrics
        expected_return = np.dot(optimal_weights, expected_returns)
        expected_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(covariance_matrix, optimal_weights)))
        sharpe_ratio = expected_return / expected_volatility if expected_volatility > 0 else 0.0
        
        # Calculate optimization score
        optimization_score = self._calculate_optimization_score(
            optimal_weights, expected_return, expected_volatility, sharpe_ratio
        )
        
        # Update domain features
        self.domain_features['optimal_weights'] = optimal_weights
        self.domain_features['expected_return'] = expected_return
        self.domain_features['expected_volatility'] = expected_volatility
        self.domain_features['sharpe_ratio'] = sharpe_ratio
        self.domain_features['optimization_score'] = optimization_score
        
        self.add_attention_weight('weight_optimization', optimization_score)
        
        return {
            'optimal_weights': optimal_weights,
            'expected_return': expected_return,
            'expected_volatility': expected_volatility,
            'sharpe_ratio': sharpe_ratio,
            'optimization_score': optimization_score
        }
    
    def calculate_rebalancing_needs(self, 
                                  current_weights: np.ndarray,
                                  target_weights: np.ndarray,
                                  transaction_costs: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate rebalancing needs and costs
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            transaction_costs: Transaction cost parameters
            
        Returns:
            Rebalancing analysis
        """
        self.update_reasoning_chain("Calculating rebalancing needs")
        
        # Calculate weight deviations
        weight_deviations = target_weights - current_weights
        total_deviation = np.sum(np.abs(weight_deviations))
        
        # Calculate turnover
        turnover = total_deviation / 2  # Half of total absolute deviation
        
        # Calculate transaction costs
        cost_per_transaction = transaction_costs.get('cost_per_transaction', 0.001)
        total_cost = turnover * cost_per_transaction
        
        # Determine if rebalancing is needed
        rebalancing_threshold = self.domain_features.get('rebalancing_threshold', 0.05)
        rebalancing_needed = total_deviation > rebalancing_threshold
        
        # Calculate rebalancing efficiency
        if rebalancing_needed:
            benefit_estimate = self._estimate_rebalancing_benefit(weight_deviations)
            efficiency = benefit_estimate / total_cost if total_cost > 0 else 0.0
        else:
            efficiency = 0.0
        
        # Update domain features
        self.domain_features['weight_deviations'] = weight_deviations
        self.domain_features['rebalancing_cost'] = total_cost
        self.domain_features['rebalancing_efficiency'] = efficiency
        
        self.add_attention_weight('rebalancing', min(total_deviation * 5, 1.0))
        
        return {
            'rebalancing_needed': rebalancing_needed,
            'weight_deviations': weight_deviations,
            'total_deviation': total_deviation,
            'turnover': turnover,
            'total_cost': total_cost,
            'efficiency': efficiency
        }
    
    def _calculate_optimization_score(self, 
                                    weights: np.ndarray,
                                    expected_return: float,
                                    expected_volatility: float,
                                    sharpe_ratio: float) -> float:
        """Calculate optimization quality score"""
        score = 0.0
        
        # Sharpe ratio component (40%)
        normalized_sharpe = min(sharpe_ratio / 2.0, 1.0) if sharpe_ratio > 0 else 0.0
        score += normalized_sharpe * 0.4
        
        # Diversification component (30%)
        # Higher entropy = better diversification
        entropy = -np.sum(weights * np.log(weights + 1e-10))
        max_entropy = np.log(len(weights))
        diversification_score = entropy / max_entropy if max_entropy > 0 else 0.0
        score += diversification_score * 0.3
        
        # Return component (20%)
        normalized_return = min(expected_return / 0.1, 1.0) if expected_return > 0 else 0.0
        score += normalized_return * 0.2
        
        # Risk component (10%) - lower risk is better
        risk_score = max(0, 1.0 - expected_volatility / 0.2)
        score += risk_score * 0.1
        
        return score
    
    def _estimate_rebalancing_benefit(self, weight_deviations: np.ndarray) -> float:
        """Estimate benefit of rebalancing"""
        # Simplified benefit estimation
        # Assume benefit is proportional to squared deviations
        squared_deviations = np.sum(weight_deviations ** 2)
        estimated_benefit = squared_deviations * 0.1  # Arbitrary scaling
        
        return estimated_benefit