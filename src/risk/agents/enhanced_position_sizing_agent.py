"""
Enhanced Position Sizing Agent for Risk Management MARL System

Implements comprehensive position sizing with:
- Volatility-based position sizing
- 2% maximum risk per trade
- Kelly Criterion optimization
- Dynamic sizing based on market conditions
- Correlation adjustments
- Portfolio heat calculations
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
import structlog
from datetime import datetime, timedelta
import yaml

from src.risk.agents.base_risk_agent import BaseRiskAgent, RiskState, RiskAction, RiskMetrics
from src.risk.core.kelly_calculator import KellyCalculator
from src.core.events import EventBus, Event, EventType

logger = structlog.get_logger()


class PositionSizingStrategy:
    """Position sizing strategy enumeration"""
    VOLATILITY_BASED = "volatility_based"
    KELLY_CRITERION = "kelly_criterion"
    FIXED_PERCENT = "fixed_percent"
    RISK_PARITY = "risk_parity"
    CORRELATION_ADJUSTED = "correlation_adjusted"


class EnhancedPositionSizingAgent(BaseRiskAgent):
    """
    Enhanced Position Sizing Agent with comprehensive risk management
    
    Features:
    - Volatility-based position sizing with dynamic adjustments
    - Kelly Criterion optimization with safety constraints
    - Maximum 2% risk per trade enforcement
    - Portfolio heat calculation and monitoring
    - Correlation-based position adjustments
    - Market condition adaptive sizing
    """
    
    def __init__(self, config_path: str = "/home/QuantNova/GrandModel/config/risk_management_config.yaml", 
                 event_bus: Optional[EventBus] = None):
        """
        Initialize Enhanced Position Sizing Agent
        
        Args:
            config_path: Path to risk management configuration
            event_bus: Event bus for communication
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize base agent
        agent_config = {
            'name': 'enhanced_position_sizing_agent',
            'action_dim': 5,
            'max_leverage': self.config['portfolio_management']['leverage_controls']['max_leverage'],
            'var_limit': self.config['risk_controls']['drawdown_protection']['max_drawdown_percent']
        }
        
        super().__init__(agent_config, event_bus)
        
        # Position sizing configuration
        self.position_config = self.config['position_sizing']
        self.risk_config = self.config['risk_controls']
        self.portfolio_config = self.config['portfolio_management']
        
        # Core parameters
        self.max_risk_per_trade = self.position_config['max_risk_per_trade']
        self.base_position_size = self.position_config['base_position_size']
        self.max_position_size = self.position_config['max_position_size']
        self.min_position_size = self.position_config['min_position_size']
        self.volatility_lookback = self.position_config['volatility_lookback']
        
        # Kelly Criterion parameters
        self.kelly_config = self.position_config['kelly_criterion']
        self.kelly_calculator = KellyCalculator(self.kelly_config) if self.kelly_config['enabled'] else None
        
        # Portfolio tracking
        self.positions = {}  # Current positions
        self.historical_returns = {}  # Historical returns by symbol
        self.volatility_cache = {}  # Cached volatility calculations
        self.correlation_matrix = None  # Asset correlation matrix
        self.portfolio_heat = 0.0  # Current portfolio heat
        
        # Performance tracking
        self.sizing_decisions = 0
        self.risk_adjusted_returns = []
        self.portfolio_values = []
        self.drawdown_history = []
        
        logger.info("Enhanced Position Sizing Agent initialized",
                   max_risk_per_trade=self.max_risk_per_trade,
                   base_position_size=self.base_position_size,
                   kelly_enabled=self.kelly_config['enabled'])
    
    def calculate_position_size(self, symbol: str, signal_strength: float, 
                              current_price: float, volatility: float = None,
                              market_condition: str = "normal") -> Dict[str, Any]:
        """
        Calculate optimal position size based on multiple factors
        
        Args:
            symbol: Trading symbol
            signal_strength: Signal confidence (0-1)
            current_price: Current market price
            volatility: Asset volatility (optional)
            market_condition: Market condition (trending, ranging, volatile)
            
        Returns:
            Dict containing position sizing recommendation
        """
        try:
            # Calculate volatility if not provided
            if volatility is None:
                volatility = self._calculate_volatility(symbol)
            
            # Get base position size
            base_size = self._calculate_base_position_size(symbol, volatility)
            
            # Apply signal strength adjustment
            signal_adjusted_size = base_size * signal_strength
            
            # Apply volatility adjustment
            volatility_adjusted_size = self._apply_volatility_adjustment(
                signal_adjusted_size, volatility)
            
            # Apply Kelly Criterion if enabled
            if self.kelly_calculator:
                kelly_size = self._calculate_kelly_position_size(
                    symbol, volatility_adjusted_size, current_price)
            else:
                kelly_size = volatility_adjusted_size
            
            # Apply correlation adjustment
            correlation_adjusted_size = self._apply_correlation_adjustment(
                symbol, kelly_size)
            
            # Apply market condition adjustment
            market_adjusted_size = self._apply_market_condition_adjustment(
                correlation_adjusted_size, market_condition)
            
            # Apply portfolio heat constraint
            heat_adjusted_size = self._apply_portfolio_heat_constraint(
                symbol, market_adjusted_size, current_price)
            
            # Final risk constraint check
            final_size = self._apply_final_risk_constraints(
                heat_adjusted_size, current_price, volatility)
            
            # Create position sizing recommendation
            recommendation = {
                'symbol': symbol,
                'recommended_size': final_size,
                'risk_amount': final_size * current_price * volatility,
                'risk_percent': (final_size * current_price * volatility) / self._get_portfolio_value(),
                'volatility': volatility,
                'signal_strength': signal_strength,
                'market_condition': market_condition,
                'sizing_components': {
                    'base_size': base_size,
                    'signal_adjusted': signal_adjusted_size,
                    'volatility_adjusted': volatility_adjusted_size,
                    'kelly_adjusted': kelly_size,
                    'correlation_adjusted': correlation_adjusted_size,
                    'market_adjusted': market_adjusted_size,
                    'heat_adjusted': heat_adjusted_size,
                    'final_size': final_size
                }
            }
            
            logger.info("Position size calculated",
                       symbol=symbol,
                       final_size=final_size,
                       risk_percent=recommendation['risk_percent'],
                       volatility=volatility)
            
            return recommendation
            
        except Exception as e:
            logger.error("Error calculating position size", error=str(e), symbol=symbol)
            return self._get_safe_position_size(symbol, current_price)
    
    def _calculate_volatility(self, symbol: str) -> float:
        """Calculate asset volatility using historical returns"""
        if symbol in self.volatility_cache:
            return self.volatility_cache[symbol]
        
        if symbol not in self.historical_returns or len(self.historical_returns[symbol]) < 10:
            # Default volatility if insufficient data
            volatility = 0.02  # 2% default volatility
        else:
            returns = np.array(self.historical_returns[symbol][-self.volatility_lookback:])
            volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
        
        self.volatility_cache[symbol] = volatility
        return volatility
    
    def _calculate_base_position_size(self, symbol: str, volatility: float) -> float:
        """Calculate base position size based on volatility"""
        # Volatility-based sizing: higher volatility = smaller position
        volatility_factor = 1.0 / (1.0 + volatility * self.position_config['volatility_multiplier'])
        base_size = self.base_position_size * volatility_factor
        
        # Ensure within bounds
        return max(self.min_position_size, min(self.max_position_size, base_size))
    
    def _apply_volatility_adjustment(self, base_size: float, volatility: float) -> float:
        """Apply volatility adjustment to position size"""
        if not self.position_config['volatility_adjustment']:
            return base_size
        
        # Inverse relationship: higher volatility = smaller position
        volatility_multiplier = 1.0 / (1.0 + volatility)
        adjusted_size = base_size * volatility_multiplier
        
        return max(self.min_position_size, min(self.max_position_size, adjusted_size))
    
    def _calculate_kelly_position_size(self, symbol: str, current_size: float, 
                                     current_price: float) -> float:
        """Calculate Kelly Criterion optimal position size"""
        if not self.kelly_calculator:
            return current_size
        
        try:
            # Get historical performance for Kelly calculation
            if symbol not in self.historical_returns or len(self.historical_returns[symbol]) < 20:
                return current_size
            
            returns = np.array(self.historical_returns[symbol][-100:])  # Last 100 trades
            
            # Calculate Kelly parameters
            win_rate = np.mean(returns > 0)
            avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0.01
            avg_loss = np.mean(np.abs(returns[returns < 0])) if np.any(returns < 0) else 0.01
            
            # Kelly fraction
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            
            # Apply safety constraints
            kelly_fraction = max(0.0, min(self.kelly_config['max_kelly_fraction'], kelly_fraction))
            kelly_fraction *= self.kelly_config['safety_factor']
            
            # Convert to position size
            portfolio_value = self._get_portfolio_value()
            kelly_position_value = portfolio_value * kelly_fraction
            kelly_size = kelly_position_value / current_price
            
            # Use minimum of current size and Kelly size
            return min(current_size, kelly_size)
            
        except Exception as e:
            logger.error("Error in Kelly calculation", error=str(e), symbol=symbol)
            return current_size
    
    def _apply_correlation_adjustment(self, symbol: str, current_size: float) -> float:
        """Apply correlation adjustment to reduce correlated positions"""
        if not self.position_config['correlation_adjustment'] or not self.correlation_matrix:
            return current_size
        
        try:
            # Calculate correlation penalty
            correlation_penalty = 0.0
            
            for existing_symbol, position in self.positions.items():
                if existing_symbol != symbol and position['size'] > 0:
                    correlation = self._get_correlation(symbol, existing_symbol)
                    if correlation > self.risk_config['correlation_limits']['max_correlation']:
                        correlation_penalty += correlation * position['size']
            
            # Apply penalty
            adjustment_factor = 1.0 / (1.0 + correlation_penalty)
            adjusted_size = current_size * adjustment_factor
            
            return max(self.min_position_size, adjusted_size)
            
        except Exception as e:
            logger.error("Error in correlation adjustment", error=str(e), symbol=symbol)
            return current_size
    
    def _apply_market_condition_adjustment(self, current_size: float, 
                                         market_condition: str) -> float:
        """Apply market condition adjustment to position size"""
        adjustments = self.position_config['market_condition_adjustments']
        
        if market_condition == "trending":
            multiplier = adjustments['trending_market_multiplier']
        elif market_condition == "ranging":
            multiplier = adjustments['ranging_market_multiplier']
        elif market_condition == "high_volatility":
            multiplier = adjustments['high_volatility_multiplier']
        elif market_condition == "low_volatility":
            multiplier = adjustments['low_volatility_multiplier']
        else:
            multiplier = 1.0
        
        adjusted_size = current_size * multiplier
        return max(self.min_position_size, min(self.max_position_size, adjusted_size))
    
    def _apply_portfolio_heat_constraint(self, symbol: str, current_size: float, 
                                       current_price: float) -> float:
        """Apply portfolio heat constraint to prevent overexposure"""
        heat_config = self.portfolio_config['portfolio_heat']
        if not heat_config['enabled']:
            return current_size
        
        # Calculate position heat
        position_value = current_size * current_price
        portfolio_value = self._get_portfolio_value()
        position_heat = position_value / portfolio_value
        
        # Calculate total portfolio heat with this position
        total_heat = self.portfolio_heat + position_heat
        
        # Check if exceeds threshold
        if total_heat > heat_config['max_heat_threshold']:
            # Reduce position size to stay within threshold
            max_allowed_heat = heat_config['max_heat_threshold'] - self.portfolio_heat
            max_position_value = max_allowed_heat * portfolio_value
            max_size = max_position_value / current_price
            
            adjusted_size = min(current_size, max_size)
            logger.warning("Position size reduced due to portfolio heat",
                         symbol=symbol,
                         original_size=current_size,
                         adjusted_size=adjusted_size,
                         total_heat=total_heat,
                         max_heat=heat_config['max_heat_threshold'])
            
            return max(self.min_position_size, adjusted_size)
        
        return current_size
    
    def _apply_final_risk_constraints(self, current_size: float, current_price: float, 
                                    volatility: float) -> float:
        """Apply final risk constraints to ensure 2% max risk per trade"""
        # Calculate risk amount
        position_value = current_size * current_price
        risk_amount = position_value * volatility  # Simplified risk calculation
        
        # Calculate portfolio value
        portfolio_value = self._get_portfolio_value()
        
        # Check max risk per trade
        risk_percent = risk_amount / portfolio_value
        
        if risk_percent > self.max_risk_per_trade:
            # Reduce position size to meet risk limit
            max_risk_amount = portfolio_value * self.max_risk_per_trade
            max_position_value = max_risk_amount / volatility
            max_size = max_position_value / current_price
            
            adjusted_size = min(current_size, max_size)
            logger.warning("Position size reduced due to risk limit",
                         risk_percent=risk_percent,
                         max_risk=self.max_risk_per_trade,
                         original_size=current_size,
                         adjusted_size=adjusted_size)
            
            return max(self.min_position_size, adjusted_size)
        
        return current_size
    
    def calculate_portfolio_heat(self) -> float:
        """Calculate current portfolio heat (total risk exposure)"""
        total_heat = 0.0
        portfolio_value = self._get_portfolio_value()
        
        for symbol, position in self.positions.items():
            if position['size'] > 0:
                position_value = position['size'] * position['current_price']
                volatility = self._calculate_volatility(symbol)
                position_risk = position_value * volatility
                position_heat = position_risk / portfolio_value
                total_heat += position_heat
        
        self.portfolio_heat = total_heat
        return total_heat
    
    def update_historical_returns(self, symbol: str, return_value: float):
        """Update historical returns for volatility calculation"""
        if symbol not in self.historical_returns:
            self.historical_returns[symbol] = []
        
        self.historical_returns[symbol].append(return_value)
        
        # Keep only recent returns
        max_history = max(self.volatility_lookback * 2, 100)
        if len(self.historical_returns[symbol]) > max_history:
            self.historical_returns[symbol] = self.historical_returns[symbol][-max_history:]
        
        # Invalidate volatility cache
        if symbol in self.volatility_cache:
            del self.volatility_cache[symbol]
    
    def update_position(self, symbol: str, size: float, current_price: float):
        """Update position information"""
        self.positions[symbol] = {
            'size': size,
            'current_price': current_price,
            'timestamp': datetime.now()
        }
    
    def _get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        # Simplified portfolio value calculation
        # In practice, this would come from account information
        return 1000000.0  # $1M default portfolio value
    
    def _get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols"""
        if not self.correlation_matrix:
            return 0.0
        
        # Simplified correlation lookup
        # In practice, this would use the correlation matrix
        return 0.3  # Default correlation
    
    def _get_safe_position_size(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """Get safe default position size for error cases"""
        safe_size = self.min_position_size
        return {
            'symbol': symbol,
            'recommended_size': safe_size,
            'risk_amount': safe_size * current_price * 0.02,
            'risk_percent': 0.01,
            'volatility': 0.02,
            'signal_strength': 0.5,
            'market_condition': 'normal',
            'sizing_components': {
                'base_size': safe_size,
                'final_size': safe_size
            }
        }
    
    def get_position_sizing_metrics(self) -> Dict[str, Any]:
        """Get position sizing performance metrics"""
        return {
            'current_portfolio_heat': self.portfolio_heat,
            'active_positions': len([p for p in self.positions.values() if p['size'] > 0]),
            'total_exposure': sum(p['size'] * p['current_price'] for p in self.positions.values()),
            'average_position_size': np.mean([p['size'] for p in self.positions.values()]) if self.positions else 0,
            'sizing_decisions': self.sizing_decisions,
            'risk_adjusted_returns': self.risk_adjusted_returns[-100:] if self.risk_adjusted_returns else []
        }
    
    def validate_position_sizing(self) -> Dict[str, Any]:
        """Validate current position sizing against constraints"""
        violations = []
        warnings = []
        
        # Check portfolio heat
        if self.portfolio_heat > self.portfolio_config['portfolio_heat']['max_heat_threshold']:
            violations.append(f"Portfolio heat exceeded: {self.portfolio_heat:.3f}")
        
        # Check individual position sizes
        for symbol, position in self.positions.items():
            position_percent = (position['size'] * position['current_price']) / self._get_portfolio_value()
            if position_percent > self.max_position_size:
                violations.append(f"Position {symbol} exceeds max size: {position_percent:.3f}")
        
        # Check risk per trade
        for symbol, position in self.positions.items():
            volatility = self._calculate_volatility(symbol)
            risk_amount = position['size'] * position['current_price'] * volatility
            risk_percent = risk_amount / self._get_portfolio_value()
            if risk_percent > self.max_risk_per_trade:
                violations.append(f"Position {symbol} exceeds max risk: {risk_percent:.3f}")
        
        return {
            'is_valid': len(violations) == 0,
            'violations': violations,
            'warnings': warnings,
            'portfolio_heat': self.portfolio_heat,
            'total_positions': len(self.positions)
        }
    
    def reset(self):
        """Reset agent state"""
        super().reset()
        self.positions.clear()
        self.volatility_cache.clear()
        self.portfolio_heat = 0.0
        self.sizing_decisions = 0
        logger.info("Enhanced Position Sizing Agent reset")