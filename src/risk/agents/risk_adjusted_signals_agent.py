"""
Risk-Adjusted Signals Output System

Combines all risk management components to produce final risk-adjusted signals:
- Position size recommendations
- Risk-adjusted signal strength
- Portfolio risk metrics
- Execution parameters
- Risk management validation
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import structlog
from datetime import datetime
from dataclasses import dataclass, field
import yaml

from src.risk.agents.enhanced_position_sizing_agent import EnhancedPositionSizingAgent
from src.risk.agents.portfolio_heat_calculator import PortfolioHeatCalculator
from src.risk.agents.dynamic_stop_target_agent import DynamicStopTargetAgent
from src.risk.agents.drawdown_protection_agent import DrawdownProtectionAgent
from src.risk.agents.correlation_limits_agent import CorrelationLimitsAgent
from src.risk.agents.execution_rules_agent import ExecutionRulesAgent

logger = structlog.get_logger()


@dataclass
class RiskAdjustedSignal:
    """Risk-adjusted signal output"""
    symbol: str
    original_signal_strength: float
    risk_adjusted_strength: float
    position_size: float
    risk_amount: float
    risk_percent: float
    execution_allowed: bool
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    execution_parameters: Dict[str, Any] = field(default_factory=dict)
    risk_metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'original_signal_strength': self.original_signal_strength,
            'risk_adjusted_strength': self.risk_adjusted_strength,
            'position_size': self.position_size,
            'risk_amount': self.risk_amount,
            'risk_percent': self.risk_percent,
            'execution_allowed': self.execution_allowed,
            'stop_loss_price': self.stop_loss_price,
            'take_profit_price': self.take_profit_price,
            'execution_parameters': self.execution_parameters,
            'risk_metrics': self.risk_metrics,
            'warnings': self.warnings,
            'timestamp': datetime.now().isoformat()
        }


@dataclass
class PortfolioRiskMetrics:
    """Portfolio-level risk metrics"""
    total_portfolio_heat: float
    max_drawdown: float
    current_drawdown: float
    var_95: float
    expected_shortfall: float
    portfolio_correlation: float
    diversification_score: float
    leverage_ratio: float
    risk_budget_utilization: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_portfolio_heat': self.total_portfolio_heat,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'var_95': self.var_95,
            'expected_shortfall': self.expected_shortfall,
            'portfolio_correlation': self.portfolio_correlation,
            'diversification_score': self.diversification_score,
            'leverage_ratio': self.leverage_ratio,
            'risk_budget_utilization': self.risk_budget_utilization,
            'timestamp': datetime.now().isoformat()
        }


class RiskAdjustedSignalsAgent:
    """
    Risk-Adjusted Signals Agent
    
    Integrates all risk management components to produce final trading signals
    with comprehensive risk management parameters.
    """
    
    def __init__(self, config_path: str = "/home/QuantNova/GrandModel/config/risk_management_config.yaml"):
        """Initialize Risk-Adjusted Signals Agent"""
        self.config_path = config_path
        
        # Initialize all risk management components
        self.position_sizer = EnhancedPositionSizingAgent(config_path)
        self.heat_calculator = PortfolioHeatCalculator(config_path)
        self.stop_target_agent = DynamicStopTargetAgent(config_path)
        self.drawdown_protection = DrawdownProtectionAgent(config_path)
        self.correlation_limits = CorrelationLimitsAgent(config_path)
        self.execution_rules = ExecutionRulesAgent(config_path)
        
        # Portfolio state
        self.current_portfolio_value = 1000000.0  # $1M default
        self.current_positions = {}
        self.signals_history = []
        
        # Risk metrics
        self.risk_metrics_history = []
        
        logger.info("Risk-Adjusted Signals Agent initialized")
    
    def process_signal(self, symbol: str, signal_strength: float, 
                      direction: str, market_data: Dict[str, Any],
                      portfolio_data: Dict[str, Any] = None) -> RiskAdjustedSignal:
        """
        Process raw signal and produce risk-adjusted signal
        
        Args:
            symbol: Trading symbol
            signal_strength: Raw signal strength (0-1)
            direction: Signal direction ("long" or "short")
            market_data: Current market data
            portfolio_data: Current portfolio data
            
        Returns:
            RiskAdjustedSignal with comprehensive risk management
        """
        try:
            # Extract market data
            current_price = market_data.get('price', 0)
            volatility = market_data.get('volatility', 0.02)
            volume = market_data.get('volume', 0)
            
            # Update portfolio data if provided
            if portfolio_data:
                self.current_portfolio_value = portfolio_data.get('value', self.current_portfolio_value)
                self.current_positions = portfolio_data.get('positions', self.current_positions)
            
            # Step 1: Calculate position size
            position_recommendation = self.position_sizer.calculate_position_size(
                symbol=symbol,
                signal_strength=signal_strength,
                current_price=current_price,
                volatility=volatility,
                market_condition=market_data.get('market_condition', 'normal')
            )
            
            # Step 2: Check portfolio heat constraints
            heat_metrics = self.heat_calculator.calculate_portfolio_heat(
                self.current_positions, self.current_portfolio_value)
            
            # Step 3: Apply correlation limits
            correlation_check = self.correlation_limits.check_position_correlation(
                symbol, position_recommendation['recommended_size'])
            
            # Step 4: Check drawdown protection
            drawdown_status = self.drawdown_protection.get_protection_status()
            
            # Step 5: Evaluate execution rules
            execution_evaluation = self.execution_rules.evaluate_entry_execution(
                symbol, position_recommendation['recommended_size'], 
                signal_strength, market_data)
            
            # Step 6: Create stop-loss and take-profit levels
            if execution_evaluation.get('execution_allowed', True):
                position_profile = self.stop_target_agent.create_position_levels(
                    symbol=symbol,
                    entry_price=current_price,
                    position_size=position_recommendation['recommended_size'],
                    direction=direction,
                    volatility=volatility
                )
                
                stop_loss_price = position_profile.stop_levels[0].price if position_profile.stop_levels else None
                take_profit_price = position_profile.target_levels[0].price if position_profile.target_levels else None
            else:
                stop_loss_price = None
                take_profit_price = None
            
            # Step 7: Calculate final risk-adjusted signal
            risk_adjusted_signal = self._calculate_risk_adjusted_signal(
                symbol=symbol,
                original_strength=signal_strength,
                position_recommendation=position_recommendation,
                heat_metrics=heat_metrics,
                correlation_check=correlation_check,
                drawdown_status=drawdown_status,
                execution_evaluation=execution_evaluation,
                current_price=current_price,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price
            )
            
            # Store in history
            self.signals_history.append(risk_adjusted_signal)
            if len(self.signals_history) > 10000:
                self.signals_history = self.signals_history[-10000:]
            
            return risk_adjusted_signal
            
        except Exception as e:
            logger.error("Error processing signal", error=str(e), symbol=symbol)
            return self._create_default_signal(symbol, signal_strength, current_price)
    
    def _calculate_risk_adjusted_signal(self, symbol: str, original_strength: float,
                                      position_recommendation: Dict[str, Any],
                                      heat_metrics: Any, correlation_check: Dict[str, Any],
                                      drawdown_status: Dict[str, Any],
                                      execution_evaluation: Dict[str, Any],
                                      current_price: float,
                                      stop_loss_price: Optional[float],
                                      take_profit_price: Optional[float]) -> RiskAdjustedSignal:
        """Calculate final risk-adjusted signal"""
        
        # Start with original signal strength
        adjusted_strength = original_strength
        warnings = []
        
        # Apply heat adjustment
        if heat_metrics.total_heat > 0.12:  # 12% heat threshold
            heat_adjustment = max(0.5, 1.0 - (heat_metrics.total_heat - 0.12) / 0.08)
            adjusted_strength *= heat_adjustment
            warnings.append(f"Signal reduced due to portfolio heat: {heat_metrics.total_heat:.3f}")
        
        # Apply correlation adjustment
        if correlation_check.get('action_required', False):
            correlation_adjustment = max(0.3, 1.0 - len(correlation_check['correlation_violations']) * 0.2)
            adjusted_strength *= correlation_adjustment
            warnings.append(f"Signal reduced due to correlation violations: {len(correlation_check['correlation_violations'])}")
        
        # Apply drawdown protection
        if drawdown_status.get('protection_active', False):
            drawdown_adjustment = max(0.1, 1.0 - drawdown_status['current_drawdown'] / 0.2)
            adjusted_strength *= drawdown_adjustment
            warnings.append(f"Signal reduced due to drawdown protection: {drawdown_status['current_drawdown']:.3f}")
        
        # Determine execution allowance
        execution_allowed = (
            execution_evaluation.get('execution_allowed', True) and
            not drawdown_status.get('trading_halted', False) and
            adjusted_strength > 0.3  # Minimum adjusted strength
        )
        
        # Get final position size
        final_position_size = correlation_check.get('max_allowed_size', 
                                                   position_recommendation['recommended_size'])
        
        # Calculate risk metrics
        risk_amount = final_position_size * current_price * position_recommendation.get('volatility', 0.02)
        risk_percent = risk_amount / self.current_portfolio_value
        
        # Create execution parameters
        execution_parameters = {
            'order_type': 'limit',
            'time_in_force': 'day',
            'execution_strategy': execution_evaluation.get('execution_strategy', 'standard'),
            'estimated_costs': execution_evaluation.get('estimated_costs', {}),
            'fill_probability': 0.8,  # Default estimate
            'market_impact': execution_evaluation.get('market_impact', 0.0)
        }
        
        # Create risk metrics
        risk_metrics = {
            'portfolio_heat': heat_metrics.total_heat,
            'position_heat': position_recommendation.get('risk_percent', 0),
            'correlation_risk': len(correlation_check.get('correlation_violations', [])),
            'drawdown_risk': drawdown_status.get('current_drawdown', 0),
            'execution_quality': execution_evaluation.get('execution_quality', 'average'),
            'diversification_score': heat_metrics.diversification_ratio
        }
        
        # Create final signal
        return RiskAdjustedSignal(
            symbol=symbol,
            original_signal_strength=original_strength,
            risk_adjusted_strength=adjusted_strength,
            position_size=final_position_size,
            risk_amount=risk_amount,
            risk_percent=risk_percent,
            execution_allowed=execution_allowed,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            execution_parameters=execution_parameters,
            risk_metrics=risk_metrics,
            warnings=warnings
        )
    
    def generate_portfolio_risk_metrics(self) -> PortfolioRiskMetrics:
        """Generate comprehensive portfolio risk metrics"""
        try:
            # Calculate portfolio heat
            heat_metrics = self.heat_calculator.calculate_portfolio_heat(
                self.current_positions, self.current_portfolio_value)
            
            # Get drawdown metrics
            drawdown_status = self.drawdown_protection.get_protection_status()
            
            # Get correlation metrics
            correlation_status = self.correlation_limits.get_correlation_status()
            
            # Calculate VaR (simplified)
            var_95 = self._calculate_var(0.95)
            expected_shortfall = self._calculate_expected_shortfall(0.95)
            
            # Calculate leverage ratio
            leverage_ratio = self._calculate_leverage_ratio()
            
            # Calculate risk budget utilization
            risk_budget_utilization = min(1.0, heat_metrics.total_heat / 0.15)  # 15% max heat
            
            portfolio_metrics = PortfolioRiskMetrics(
                total_portfolio_heat=heat_metrics.total_heat,
                max_drawdown=drawdown_status.get('max_drawdown_threshold', 0.2),
                current_drawdown=drawdown_status.get('current_drawdown', 0.0),
                var_95=var_95,
                expected_shortfall=expected_shortfall,
                portfolio_correlation=correlation_status.get('avg_portfolio_correlation', 0.0),
                diversification_score=correlation_status.get('diversification_score', 1.0),
                leverage_ratio=leverage_ratio,
                risk_budget_utilization=risk_budget_utilization
            )
            
            # Store in history
            self.risk_metrics_history.append(portfolio_metrics)
            if len(self.risk_metrics_history) > 1000:
                self.risk_metrics_history = self.risk_metrics_history[-1000:]
            
            return portfolio_metrics
            
        except Exception as e:
            logger.error("Error generating portfolio risk metrics", error=str(e))
            return self._create_default_portfolio_metrics()
    
    def _calculate_var(self, confidence_level: float) -> float:
        """Calculate Value at Risk (simplified)"""
        try:
            if not self.current_positions:
                return 0.0
            
            # Simplified VaR calculation
            total_exposure = sum(abs(pos.get('size', 0) * pos.get('price', 0)) 
                               for pos in self.current_positions.values())
            
            # Assume 2% daily volatility
            daily_volatility = 0.02
            
            # Normal distribution inverse for confidence level
            if confidence_level == 0.95:
                z_score = 1.645
            elif confidence_level == 0.99:
                z_score = 2.326
            else:
                z_score = 1.645
            
            var = total_exposure * daily_volatility * z_score
            return var / self.current_portfolio_value  # Return as percentage
            
        except Exception as e:
            logger.error("Error calculating VaR", error=str(e))
            return 0.0
    
    def _calculate_expected_shortfall(self, confidence_level: float) -> float:
        """Calculate Expected Shortfall (simplified)"""
        try:
            var = self._calculate_var(confidence_level)
            # ES is typically 1.3-1.5x VaR for normal distribution
            return var * 1.4
            
        except Exception as e:
            logger.error("Error calculating Expected Shortfall", error=str(e))
            return 0.0
    
    def _calculate_leverage_ratio(self) -> float:
        """Calculate current leverage ratio"""
        try:
            if not self.current_positions:
                return 0.0
            
            gross_exposure = sum(abs(pos.get('size', 0) * pos.get('price', 0)) 
                               for pos in self.current_positions.values())
            
            return gross_exposure / self.current_portfolio_value
            
        except Exception as e:
            logger.error("Error calculating leverage ratio", error=str(e))
            return 0.0
    
    def _create_default_signal(self, symbol: str, signal_strength: float, 
                             current_price: float) -> RiskAdjustedSignal:
        """Create default signal for error cases"""
        return RiskAdjustedSignal(
            symbol=symbol,
            original_signal_strength=signal_strength,
            risk_adjusted_strength=signal_strength * 0.5,  # Conservative adjustment
            position_size=0.01,  # 1% position size
            risk_amount=0.01 * current_price * 0.02,  # 2% risk
            risk_percent=0.0002,  # 0.02% of portfolio
            execution_allowed=False,
            warnings=['Error in signal processing - using conservative defaults']
        )
    
    def _create_default_portfolio_metrics(self) -> PortfolioRiskMetrics:
        """Create default portfolio metrics for error cases"""
        return PortfolioRiskMetrics(
            total_portfolio_heat=0.0,
            max_drawdown=0.2,
            current_drawdown=0.0,
            var_95=0.0,
            expected_shortfall=0.0,
            portfolio_correlation=0.0,
            diversification_score=1.0,
            leverage_ratio=0.0,
            risk_budget_utilization=0.0
        )
    
    def get_signals_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get signals summary for specified hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            recent_signals = [
                signal for signal in self.signals_history
                if datetime.fromisoformat(signal.to_dict()['timestamp']) >= cutoff_time
            ]
            
            if not recent_signals:
                return {'status': 'no_signals', 'period_hours': hours}
            
            # Calculate summary statistics
            total_signals = len(recent_signals)
            executed_signals = len([s for s in recent_signals if s.execution_allowed])
            avg_original_strength = np.mean([s.original_signal_strength for s in recent_signals])
            avg_adjusted_strength = np.mean([s.risk_adjusted_strength for s in recent_signals])
            avg_risk_percent = np.mean([s.risk_percent for s in recent_signals])
            
            return {
                'period_hours': hours,
                'total_signals': total_signals,
                'executed_signals': executed_signals,
                'execution_rate': executed_signals / total_signals,
                'avg_original_strength': avg_original_strength,
                'avg_adjusted_strength': avg_adjusted_strength,
                'avg_risk_adjustment': (avg_original_strength - avg_adjusted_strength) / avg_original_strength,
                'avg_risk_percent': avg_risk_percent,
                'symbols_traded': list(set(s.symbol for s in recent_signals)),
                'total_warnings': sum(len(s.warnings) for s in recent_signals)
            }
            
        except Exception as e:
            logger.error("Error getting signals summary", error=str(e))
            return {'error': str(e)}
    
    def validate_risk_management_system(self) -> Dict[str, Any]:
        """Validate the entire risk management system"""
        try:
            validation_results = {
                'system_status': 'healthy',
                'component_status': {},
                'violations': [],
                'warnings': [],
                'recommendations': []
            }
            
            # Validate position sizing
            position_sizing_validation = self.position_sizer.validate_position_sizing()
            validation_results['component_status']['position_sizing'] = position_sizing_validation
            
            # Validate correlation limits
            correlation_validation = self.correlation_limits.validate_correlation_limits()
            validation_results['component_status']['correlation_limits'] = correlation_validation
            
            # Validate drawdown protection
            drawdown_status = self.drawdown_protection.get_protection_status()
            validation_results['component_status']['drawdown_protection'] = drawdown_status
            
            # Check overall system health
            total_violations = sum(
                len(status.get('violations', [])) 
                for status in validation_results['component_status'].values()
            )
            
            if total_violations > 0:
                validation_results['system_status'] = 'warnings'
            
            if total_violations > 5:
                validation_results['system_status'] = 'critical'
            
            return validation_results
            
        except Exception as e:
            logger.error("Error validating risk management system", error=str(e))
            return {'error': str(e)}
    
    def update_portfolio_data(self, portfolio_value: float, positions: Dict[str, Any]):
        """Update portfolio data"""
        self.current_portfolio_value = portfolio_value
        self.current_positions = positions
        
        # Update individual components
        for symbol, position in positions.items():
            if 'price' in position:
                self.correlation_limits.update_price_data(symbol, position['price'])
        
        # Update drawdown protection
        self.drawdown_protection.update_equity(portfolio_value, positions)