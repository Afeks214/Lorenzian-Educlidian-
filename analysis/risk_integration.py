"""
Advanced Risk Integration Module

This module integrates sophisticated risk management into the backtesting framework,
including Kelly criterion position sizing, correlation-based adjustments, and
dynamic drawdown controls.

Features:
- Kelly criterion position sizing with security validation
- Correlation-based risk adjustments using existing VaR framework
- Dynamic drawdown controls with automated risk reduction
- Real-time risk monitoring during backtests
- Integration with existing risk systems
- Portfolio-level risk aggregation
"""

import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import warnings
from numba import jit
from collections import deque
from scipy import stats
import asyncio

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import existing risk systems
try:
    from src.risk.core.kelly_calculator import KellyCalculator, KellyInputs, KellyOutput
    from src.risk.core.var_calculator import VaRCalculator, VaRResult, PositionData
    from src.risk.core.correlation_tracker import CorrelationTracker, CorrelationRegime
    from src.risk.utils.performance_monitor import PerformanceMonitor
    from src.core.events import EventBus, Event, EventType
except ImportError as e:
    logging.warning(f"Could not import existing risk systems: {e}")
    # Fallback implementations will be provided

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Risk metrics for portfolio"""
    timestamp: datetime
    portfolio_value: float
    var_95: float
    var_99: float
    expected_shortfall: float
    max_drawdown: float
    current_drawdown: float
    correlation_regime: str
    kelly_fraction: float
    position_size_multiplier: float
    risk_budget_used: float
    

@dataclass
class DrawdownControl:
    """Drawdown control configuration"""
    max_drawdown_limit: float = 0.20  # 20% max drawdown
    high_drawdown_threshold: float = 0.15  # 15% high drawdown
    critical_drawdown_threshold: float = 0.18  # 18% critical drawdown
    reduction_factor_high: float = 0.5  # 50% reduction
    reduction_factor_critical: float = 0.25  # 75% reduction
    recovery_threshold: float = 0.05  # 5% recovery needed
    
    
@dataclass
class PositionSizing:
    """Position sizing configuration"""
    base_position_size: float = 0.1  # 10% base position
    kelly_multiplier: float = 0.25  # 25% of Kelly fraction
    max_position_size: float = 0.3  # 30% max position
    min_position_size: float = 0.01  # 1% min position
    correlation_adjustment: bool = True
    volatility_adjustment: bool = True
    

class RiskIntegrator:
    """
    Advanced risk integration system for backtesting
    
    Integrates Kelly criterion position sizing, correlation-based adjustments,
    and dynamic drawdown controls with existing risk systems.
    """
    
    def __init__(
        self,
        symbols: List[str],
        initial_capital: float,
        drawdown_control: Optional[DrawdownControl] = None,
        position_sizing: Optional[PositionSizing] = None,
        use_existing_systems: bool = True
    ):
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Configuration
        self.drawdown_control = drawdown_control or DrawdownControl()
        self.position_sizing = position_sizing or PositionSizing()
        
        # Risk state
        self.positions: Dict[str, float] = {symbol: 0.0 for symbol in symbols}
        self.equity_curve: List[float] = [initial_capital]
        self.drawdown_history: List[float] = [0.0]
        self.risk_metrics_history: List[RiskMetrics] = []
        
        # Peak tracking for drawdown calculation
        self.peak_value = initial_capital
        self.peak_timestamp = datetime.now()
        
        # Risk reduction state
        self.risk_reduction_active = False
        self.risk_reduction_level = 1.0  # 1.0 = no reduction
        self.last_risk_check = datetime.now()
        
        # Performance tracking
        self.returns_history: deque = deque(maxlen=252)  # 1 year of returns
        self.volatility_history: deque = deque(maxlen=252)
        self.correlation_history: deque = deque(maxlen=100)
        
        # Initialize risk systems
        self.kelly_calculator = None
        self.var_calculator = None
        self.correlation_tracker = None
        self.performance_monitor = None
        self.event_bus = None
        
        if use_existing_systems:
            self._initialize_risk_systems()
    
    def _initialize_risk_systems(self):
        """Initialize existing risk systems"""
        
        try:
            # Initialize event bus
            self.event_bus = EventBus()
            
            # Initialize Kelly calculator
            self.kelly_calculator = KellyCalculator(
                max_kelly_fraction=0.25,
                security_mode=True,
                validation_window_days=30
            )
            
            # Initialize correlation tracker
            self.correlation_tracker = CorrelationTracker(
                symbols=self.symbols,
                event_bus=self.event_bus
            )
            
            # Initialize VaR calculator
            self.var_calculator = VaRCalculator(
                correlation_tracker=self.correlation_tracker,
                event_bus=self.event_bus
            )
            
            # Initialize performance monitor
            self.performance_monitor = PerformanceMonitor(
                performance_target_ms=5.0,
                event_bus=self.event_bus
            )
            
            logger.info("Risk systems initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize risk systems: {e}")
            self._initialize_fallback_systems()
    
    def _initialize_fallback_systems(self):
        """Initialize fallback risk systems"""
        
        # Simple fallback implementations
        self.kelly_calculator = FallbackKellyCalculator()
        self.var_calculator = FallbackVaRCalculator()
        self.correlation_tracker = FallbackCorrelationTracker()
        
        logger.info("Fallback risk systems initialized")
    
    def update_market_data(
        self,
        symbol: str,
        price: float,
        volume: float,
        timestamp: datetime
    ):
        """Update market data for risk calculations"""
        
        # Update correlation tracker
        if self.correlation_tracker:
            self.correlation_tracker.update_price(symbol, price, timestamp)
        
        # Update returns history
        if len(self.returns_history) > 0:
            last_price = self.returns_history[-1].get(symbol, price)
            return_val = (price - last_price) / last_price
            
            # Update volatility
            if len(self.volatility_history) > 0:
                self.volatility_history.append(return_val)
    
    def calculate_position_size(
        self,
        symbol: str,
        signal_strength: float,
        win_probability: float,
        payout_ratio: float,
        current_price: float,
        volatility: float
    ) -> float:
        """
        Calculate optimal position size using Kelly criterion and risk adjustments
        
        Args:
            symbol: Symbol to trade
            signal_strength: Signal strength (-1 to 1)
            win_probability: Probability of winning trade
            payout_ratio: Expected payout ratio
            current_price: Current price
            volatility: Current volatility
            
        Returns:
            Position size as fraction of capital
        """
        
        # Base Kelly calculation
        kelly_fraction = self._calculate_kelly_fraction(
            win_probability, payout_ratio, symbol
        )
        
        # Apply Kelly multiplier
        kelly_position = kelly_fraction * self.position_sizing.kelly_multiplier
        
        # Apply signal strength
        signal_adjusted_position = kelly_position * abs(signal_strength)
        
        # Apply risk adjustments
        risk_adjusted_position = self._apply_risk_adjustments(
            signal_adjusted_position, symbol, volatility
        )
        
        # Apply drawdown controls
        final_position = self._apply_drawdown_controls(risk_adjusted_position)
        
        # Ensure position limits
        final_position = np.clip(
            final_position,
            self.position_sizing.min_position_size,
            self.position_sizing.max_position_size
        )
        
        # Apply direction
        if signal_strength < 0:
            final_position = -final_position
        
        return final_position
    
    def _calculate_kelly_fraction(
        self,
        win_probability: float,
        payout_ratio: float,
        symbol: str
    ) -> float:
        """Calculate Kelly fraction using existing system or fallback"""
        
        if self.kelly_calculator:
            try:
                # Use existing Kelly calculator
                if hasattr(self.kelly_calculator, 'calculate_kelly_fraction'):
                    result = self.kelly_calculator.calculate_kelly_fraction(
                        win_probability, payout_ratio
                    )
                    return result.kelly_fraction if hasattr(result, 'kelly_fraction') else result
                else:
                    # Fallback Kelly calculation
                    return self._fallback_kelly_calculation(win_probability, payout_ratio)
            except Exception as e:
                logger.warning(f"Kelly calculation failed: {e}")
                return self._fallback_kelly_calculation(win_probability, payout_ratio)
        
        return self._fallback_kelly_calculation(win_probability, payout_ratio)
    
    def _fallback_kelly_calculation(self, win_prob: float, payout_ratio: float) -> float:
        """Fallback Kelly calculation"""
        
        # Validate inputs
        win_prob = np.clip(win_prob, 0.01, 0.99)
        payout_ratio = max(0.01, payout_ratio)
        
        # Kelly formula: f = (bp - q) / b
        # where b = payout ratio, p = win probability, q = loss probability
        b = payout_ratio
        p = win_prob
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Safety limits
        kelly_fraction = np.clip(kelly_fraction, 0.0, 0.25)
        
        return kelly_fraction
    
    def _apply_risk_adjustments(
        self,
        base_position: float,
        symbol: str,
        volatility: float
    ) -> float:
        """Apply correlation and volatility adjustments"""
        
        adjusted_position = base_position
        
        # Volatility adjustment
        if self.position_sizing.volatility_adjustment:
            # Reduce position size for high volatility
            vol_adjustment = 1.0 / (1.0 + volatility * 10)  # Scale factor
            adjusted_position *= vol_adjustment
        
        # Correlation adjustment
        if self.position_sizing.correlation_adjustment:
            correlation_adj = self._calculate_correlation_adjustment(symbol)
            adjusted_position *= correlation_adj
        
        return adjusted_position
    
    def _calculate_correlation_adjustment(self, symbol: str) -> float:
        """Calculate correlation-based position adjustment"""
        
        if not self.correlation_tracker:
            return 1.0
        
        try:
            # Get current correlation regime
            if hasattr(self.correlation_tracker, 'get_current_regime'):
                regime = self.correlation_tracker.get_current_regime()
                
                # Adjust position based on correlation regime
                if regime == CorrelationRegime.NORMAL:
                    return 1.0
                elif regime == CorrelationRegime.ELEVATED:
                    return 0.8
                elif regime == CorrelationRegime.CRISIS:
                    return 0.6
                elif regime == CorrelationRegime.SHOCK:
                    return 0.4
                else:
                    return 1.0
            else:
                # Fallback: use average correlation
                return 0.8  # Conservative default
                
        except Exception as e:
            logger.warning(f"Correlation adjustment failed: {e}")
            return 1.0
    
    def _apply_drawdown_controls(self, position: float) -> float:
        """Apply drawdown-based position adjustments"""
        
        current_drawdown = self.get_current_drawdown()
        
        # No adjustment if drawdown is acceptable
        if current_drawdown < self.drawdown_control.high_drawdown_threshold:
            return position
        
        # High drawdown - reduce position
        if current_drawdown < self.drawdown_control.critical_drawdown_threshold:
            return position * self.drawdown_control.reduction_factor_high
        
        # Critical drawdown - severely reduce position
        return position * self.drawdown_control.reduction_factor_critical
    
    def update_portfolio_value(self, new_value: float, timestamp: datetime):
        """Update portfolio value and risk metrics"""
        
        self.current_capital = new_value
        self.equity_curve.append(new_value)
        
        # Update peak tracking
        if new_value > self.peak_value:
            self.peak_value = new_value
            self.peak_timestamp = timestamp
        
        # Calculate current drawdown
        current_drawdown = (self.peak_value - new_value) / self.peak_value
        self.drawdown_history.append(current_drawdown)
        
        # Update risk metrics
        self._update_risk_metrics(timestamp)
        
        # Check for risk reduction triggers
        self._check_risk_reduction_triggers(current_drawdown)
    
    def _update_risk_metrics(self, timestamp: datetime):
        """Update comprehensive risk metrics"""
        
        # Calculate VaR if available
        var_95 = 0.0
        var_99 = 0.0
        expected_shortfall = 0.0
        
        if self.var_calculator and len(self.equity_curve) > 10:
            try:
                # Convert positions to PositionData
                position_data = {}
                for symbol, position in self.positions.items():
                    if position != 0:
                        position_data[symbol] = PositionData(
                            symbol=symbol,
                            quantity=position,
                            market_value=abs(position * self.current_capital),
                            price=1.0,  # Placeholder
                            volatility=0.02  # Placeholder
                        )
                
                # Calculate VaR
                if hasattr(self.var_calculator, 'calculate_var'):
                    var_result = self.var_calculator.calculate_var(
                        positions=position_data,
                        confidence_levels=[0.95, 0.99]
                    )
                    var_95 = var_result.portfolio_var if hasattr(var_result, 'portfolio_var') else 0.0
                    var_99 = var_95 * 1.3  # Approximate scaling
                    
            except Exception as e:
                logger.warning(f"VaR calculation failed: {e}")
        
        # Calculate expected shortfall (ES)
        if len(self.equity_curve) > 10:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            var_95_return = np.percentile(returns, 5)
            tail_returns = returns[returns <= var_95_return]
            expected_shortfall = np.mean(tail_returns) if len(tail_returns) > 0 else 0.0
        
        # Get correlation regime
        correlation_regime = "NORMAL"
        if self.correlation_tracker:
            try:
                if hasattr(self.correlation_tracker, 'get_current_regime'):
                    regime = self.correlation_tracker.get_current_regime()
                    correlation_regime = regime.name if hasattr(regime, 'name') else str(regime)
            except Exception:
                pass
        
        # Calculate Kelly fraction (using recent performance)
        kelly_fraction = 0.0
        if len(self.equity_curve) > 10:
            recent_returns = np.diff(self.equity_curve[-10:]) / self.equity_curve[-11:-1]
            win_rate = np.sum(recent_returns > 0) / len(recent_returns)
            avg_win = np.mean(recent_returns[recent_returns > 0]) if np.any(recent_returns > 0) else 0.0
            avg_loss = np.mean(recent_returns[recent_returns < 0]) if np.any(recent_returns < 0) else 0.0
            
            if avg_loss != 0:
                payout_ratio = abs(avg_win / avg_loss)
                kelly_fraction = self._fallback_kelly_calculation(win_rate, payout_ratio)
        
        # Create risk metrics
        risk_metrics = RiskMetrics(
            timestamp=timestamp,
            portfolio_value=self.current_capital,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
            max_drawdown=self.get_max_drawdown(),
            current_drawdown=self.get_current_drawdown(),
            correlation_regime=correlation_regime,
            kelly_fraction=kelly_fraction,
            position_size_multiplier=self.risk_reduction_level,
            risk_budget_used=self._calculate_risk_budget_used()
        )
        
        self.risk_metrics_history.append(risk_metrics)
    
    def _calculate_risk_budget_used(self) -> float:
        """Calculate percentage of risk budget used"""
        
        # Simple risk budget calculation
        current_drawdown = self.get_current_drawdown()
        max_allowed_drawdown = self.drawdown_control.max_drawdown_limit
        
        return current_drawdown / max_allowed_drawdown if max_allowed_drawdown > 0 else 0.0
    
    def _check_risk_reduction_triggers(self, current_drawdown: float):
        """Check if risk reduction should be triggered"""
        
        # Check drawdown triggers
        if current_drawdown >= self.drawdown_control.critical_drawdown_threshold:
            self._trigger_risk_reduction("CRITICAL_DRAWDOWN", 0.25)
        elif current_drawdown >= self.drawdown_control.high_drawdown_threshold:
            self._trigger_risk_reduction("HIGH_DRAWDOWN", 0.5)
        elif self.risk_reduction_active and current_drawdown < self.drawdown_control.recovery_threshold:
            self._reset_risk_reduction()
    
    def _trigger_risk_reduction(self, reason: str, reduction_level: float):
        """Trigger risk reduction protocols"""
        
        if not self.risk_reduction_active or self.risk_reduction_level > reduction_level:
            self.risk_reduction_active = True
            self.risk_reduction_level = reduction_level
            
            logger.warning(f"Risk reduction triggered: {reason}, level: {reduction_level}")
            
            # Emit event if event bus available
            if self.event_bus:
                event = Event(
                    type=EventType.RISK_ALERT,
                    data={
                        'reason': reason,
                        'reduction_level': reduction_level,
                        'timestamp': datetime.now(),
                        'portfolio_value': self.current_capital,
                        'drawdown': self.get_current_drawdown()
                    }
                )
                self.event_bus.emit(event)
    
    def _reset_risk_reduction(self):
        """Reset risk reduction to normal levels"""
        
        self.risk_reduction_active = False
        self.risk_reduction_level = 1.0
        
        logger.info("Risk reduction reset - returning to normal position sizing")
        
        # Emit event if event bus available
        if self.event_bus:
            event = Event(
                type=EventType.RISK_UPDATE,
                data={
                    'action': 'RESET',
                    'timestamp': datetime.now(),
                    'portfolio_value': self.current_capital,
                    'drawdown': self.get_current_drawdown()
                }
            )
            self.event_bus.emit(event)
    
    def get_current_drawdown(self) -> float:
        """Get current drawdown"""
        if len(self.drawdown_history) > 0:
            return self.drawdown_history[-1]
        return 0.0
    
    def get_max_drawdown(self) -> float:
        """Get maximum drawdown"""
        if len(self.drawdown_history) > 0:
            return max(self.drawdown_history)
        return 0.0
    
    def get_current_risk_metrics(self) -> Optional[RiskMetrics]:
        """Get current risk metrics"""
        if len(self.risk_metrics_history) > 0:
            return self.risk_metrics_history[-1]
        return None
    
    def get_position_size_multiplier(self) -> float:
        """Get current position size multiplier"""
        return self.risk_reduction_level
    
    def is_risk_reduction_active(self) -> bool:
        """Check if risk reduction is active"""
        return self.risk_reduction_active
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        
        current_metrics = self.get_current_risk_metrics()
        
        if not current_metrics:
            return {}
        
        return {
            'portfolio_value': self.current_capital,
            'initial_capital': self.initial_capital,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'current_drawdown': self.get_current_drawdown(),
            'max_drawdown': self.get_max_drawdown(),
            'var_95': current_metrics.var_95,
            'var_99': current_metrics.var_99,
            'expected_shortfall': current_metrics.expected_shortfall,
            'correlation_regime': current_metrics.correlation_regime,
            'kelly_fraction': current_metrics.kelly_fraction,
            'position_size_multiplier': current_metrics.position_size_multiplier,
            'risk_budget_used': current_metrics.risk_budget_used,
            'risk_reduction_active': self.risk_reduction_active,
            'peak_value': self.peak_value,
            'peak_timestamp': self.peak_timestamp
        }
    
    def update_position(self, symbol: str, new_position: float):
        """Update position for a symbol"""
        self.positions[symbol] = new_position
    
    def get_position(self, symbol: str) -> float:
        """Get current position for a symbol"""
        return self.positions.get(symbol, 0.0)
    
    def get_total_exposure(self) -> float:
        """Get total portfolio exposure"""
        return sum(abs(pos) for pos in self.positions.values())
    
    def validate_trade(
        self,
        symbol: str,
        proposed_position: float,
        current_price: float
    ) -> Tuple[bool, str]:
        """
        Validate if a trade is acceptable under current risk constraints
        
        Returns:
            (is_valid, reason)
        """
        
        # Check if risk reduction is active
        if self.risk_reduction_active:
            adjusted_position = proposed_position * self.risk_reduction_level
            if abs(adjusted_position) < abs(proposed_position):
                return False, f"Position reduced due to risk controls: {self.risk_reduction_level:.2f}"
        
        # Check position limits
        if abs(proposed_position) > self.position_sizing.max_position_size:
            return False, f"Position exceeds maximum limit: {self.position_sizing.max_position_size:.2f}"
        
        # Check total exposure
        new_exposure = self.get_total_exposure() + abs(proposed_position) - abs(self.positions.get(symbol, 0.0))
        if new_exposure > 1.0:  # 100% exposure limit
            return False, f"Total exposure would exceed 100%: {new_exposure:.2f}"
        
        # Check drawdown limits
        current_drawdown = self.get_current_drawdown()
        if current_drawdown >= self.drawdown_control.max_drawdown_limit:
            return False, f"Maximum drawdown exceeded: {current_drawdown:.2%}"
        
        return True, "Trade validated"


# Fallback implementations for when existing systems are not available

class FallbackKellyCalculator:
    """Fallback Kelly calculator"""
    
    def calculate_kelly_fraction(self, win_prob: float, payout_ratio: float) -> float:
        """Simple Kelly calculation"""
        win_prob = np.clip(win_prob, 0.01, 0.99)
        payout_ratio = max(0.01, payout_ratio)
        
        kelly_fraction = (payout_ratio * win_prob - (1 - win_prob)) / payout_ratio
        return np.clip(kelly_fraction, 0.0, 0.25)


class FallbackVaRCalculator:
    """Fallback VaR calculator"""
    
    def calculate_var(self, positions: Dict, confidence_levels: List[float]) -> Any:
        """Simple VaR calculation"""
        
        # Mock VaR result
        class MockVaRResult:
            def __init__(self):
                self.portfolio_var = 0.02  # 2% VaR
        
        return MockVaRResult()


class FallbackCorrelationTracker:
    """Fallback correlation tracker"""
    
    def __init__(self):
        self.current_regime = "NORMAL"
    
    def update_price(self, symbol: str, price: float, timestamp: datetime):
        """Update price data"""
        pass
    
    def get_current_regime(self) -> str:
        """Get current correlation regime"""
        return self.current_regime


@jit(nopython=True)
def calculate_portfolio_volatility(
    positions: np.ndarray,
    correlation_matrix: np.ndarray,
    volatilities: np.ndarray
) -> float:
    """Calculate portfolio volatility using correlation matrix"""
    
    # Weighted volatilities
    weighted_vols = positions * volatilities
    
    # Portfolio variance
    portfolio_variance = np.dot(weighted_vols, np.dot(correlation_matrix, weighted_vols))
    
    # Portfolio volatility
    return np.sqrt(portfolio_variance)


@jit(nopython=True)
def calculate_marginal_var(
    positions: np.ndarray,
    correlation_matrix: np.ndarray,
    volatilities: np.ndarray,
    confidence_level: float = 0.95
) -> np.ndarray:
    """Calculate marginal VaR for each position"""
    
    portfolio_vol = calculate_portfolio_volatility(positions, correlation_matrix, volatilities)
    
    # Z-score for confidence level
    z_score = stats.norm.ppf(1 - confidence_level)
    
    # Marginal VaR
    marginal_vars = np.zeros(len(positions))
    
    for i in range(len(positions)):
        # Contribution to portfolio variance
        variance_contribution = 0.0
        for j in range(len(positions)):
            variance_contribution += positions[j] * correlation_matrix[i, j] * volatilities[i] * volatilities[j]
        
        # Marginal VaR
        marginal_vars[i] = z_score * variance_contribution / portfolio_vol if portfolio_vol > 0 else 0.0
    
    return marginal_vars


def create_risk_integrator(
    symbols: List[str],
    initial_capital: float,
    risk_config: Optional[Dict[str, Any]] = None
) -> RiskIntegrator:
    """
    Factory function to create RiskIntegrator with configuration
    
    Args:
        symbols: List of symbols to track
        initial_capital: Initial capital amount
        risk_config: Optional risk configuration
        
    Returns:
        Configured RiskIntegrator instance
    """
    
    config = risk_config or {}
    
    # Create drawdown control
    drawdown_control = DrawdownControl(
        max_drawdown_limit=config.get('max_drawdown_limit', 0.20),
        high_drawdown_threshold=config.get('high_drawdown_threshold', 0.15),
        critical_drawdown_threshold=config.get('critical_drawdown_threshold', 0.18),
        reduction_factor_high=config.get('reduction_factor_high', 0.5),
        reduction_factor_critical=config.get('reduction_factor_critical', 0.25)
    )
    
    # Create position sizing
    position_sizing = PositionSizing(
        base_position_size=config.get('base_position_size', 0.1),
        kelly_multiplier=config.get('kelly_multiplier', 0.25),
        max_position_size=config.get('max_position_size', 0.3),
        min_position_size=config.get('min_position_size', 0.01),
        correlation_adjustment=config.get('correlation_adjustment', True),
        volatility_adjustment=config.get('volatility_adjustment', True)
    )
    
    return RiskIntegrator(
        symbols=symbols,
        initial_capital=initial_capital,
        drawdown_control=drawdown_control,
        position_sizing=position_sizing,
        use_existing_systems=config.get('use_existing_systems', True)
    )