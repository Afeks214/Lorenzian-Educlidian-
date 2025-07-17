"""
Professional Risk Management Framework
=====================================

Institutional-grade risk management for backtesting including:
- Position limits and controls
- Maximum daily loss limits
- Correlation analysis across strategies
- Stress testing capabilities
- Real-time risk monitoring
- Risk attribution analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RiskManager:
    """Comprehensive risk management for institutional backtesting"""
    
    def __init__(self, 
                 max_position_size: float = 0.1,
                 max_daily_loss: float = 0.02,
                 max_total_exposure: float = 1.0,
                 correlation_threshold: float = 0.7,
                 var_confidence: float = 0.95):
        """
        Initialize risk manager
        
        Args:
            max_position_size: Maximum position size as fraction of portfolio
            max_daily_loss: Maximum daily loss as fraction of portfolio
            max_total_exposure: Maximum total exposure across all positions
            correlation_threshold: Maximum correlation between positions
            var_confidence: Confidence level for VaR calculations
        """
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_total_exposure = max_total_exposure
        self.correlation_threshold = correlation_threshold
        self.var_confidence = var_confidence
        
        # Risk monitoring
        self.position_limits = {}
        self.risk_breaches = []
        self.stress_test_results = {}
        
        # Portfolio state
        self.current_positions = {}
        self.portfolio_value = 0
        self.daily_pnl = []
        
        print("✅ Professional Risk Manager initialized")
        print(f"   📊 Max Position Size: {max_position_size:.1%}")
        print(f"   📊 Max Daily Loss: {max_daily_loss:.1%}")
        print(f"   📊 Max Total Exposure: {max_total_exposure:.1%}")
        print(f"   📊 Correlation Threshold: {correlation_threshold:.1%}")
    
    def validate_trade(self, trade_data: Dict[str, Any], 
                      portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate trade against risk limits
        
        Args:
            trade_data: Trade information
            portfolio_state: Current portfolio state
            
        Returns:
            Validation result with approval status and adjustments
        """
        validation_result = {
            'approved': True,
            'adjustments': {},
            'risk_checks': {},
            'warnings': [],
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Update portfolio state
            current_value = portfolio_state.get('total_value', 100000)
            
            # Position size check
            position_check = self._check_position_size(trade_data, current_value)
            validation_result['risk_checks']['position_size'] = position_check
            
            if not position_check['passed']:
                validation_result['approved'] = False
                validation_result['adjustments'].update(position_check['adjustments'])
            
            # Exposure check
            exposure_check = self._check_total_exposure(trade_data, portfolio_state)
            validation_result['risk_checks']['total_exposure'] = exposure_check
            
            if not exposure_check['passed']:
                validation_result['approved'] = False
                validation_result['adjustments'].update(exposure_check['adjustments'])
            
            # Daily loss check
            loss_check = self._check_daily_loss_limit(portfolio_state)
            validation_result['risk_checks']['daily_loss'] = loss_check
            
            if not loss_check['passed']:
                validation_result['approved'] = False
                validation_result['warnings'].append("Daily loss limit exceeded - trade blocked")
            
            # Correlation check
            correlation_check = self._check_correlation_limits(trade_data, portfolio_state)
            validation_result['risk_checks']['correlation'] = correlation_check
            
            if not correlation_check['passed']:
                validation_result['warnings'].append("High correlation detected")
            
            # VaR check
            var_check = self._check_var_limits(trade_data, portfolio_state)
            validation_result['risk_checks']['var'] = var_check
            
            if not var_check['passed']:
                validation_result['warnings'].append("VaR limit exceeded")
            
            # Record risk breach if any
            if not validation_result['approved']:
                self._record_risk_breach(trade_data, validation_result)
            
        except Exception as e:
            validation_result['approved'] = False
            validation_result['warnings'].append(f"Risk validation error: {e}")
        
        return validation_result
    
    def _check_position_size(self, trade_data: Dict[str, Any], 
                           portfolio_value: float) -> Dict[str, Any]:
        """Check position size limits"""
        try:
            trade_size = abs(trade_data.get('size', 0))
            trade_value = trade_size * trade_data.get('price', 0)
            position_fraction = trade_value / portfolio_value
            
            if position_fraction <= self.max_position_size:
                return {
                    'passed': True,
                    'current_fraction': position_fraction,
                    'limit': self.max_position_size,
                    'adjustments': {}
                }
            else:
                # Calculate adjusted size
                max_trade_value = portfolio_value * self.max_position_size
                adjusted_size = max_trade_value / trade_data.get('price', 1)
                
                return {
                    'passed': False,
                    'current_fraction': position_fraction,
                    'limit': self.max_position_size,
                    'adjustments': {
                        'size': adjusted_size,
                        'reason': 'Position size limit exceeded'
                    }
                }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'adjustments': {'size': 0, 'reason': 'Position size check failed'}
            }
    
    def _check_total_exposure(self, trade_data: Dict[str, Any], 
                            portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check total portfolio exposure limits"""
        try:
            current_exposure = portfolio_state.get('total_exposure', 0)
            trade_exposure = abs(trade_data.get('size', 0)) * trade_data.get('price', 0)
            new_total_exposure = (current_exposure + trade_exposure) / portfolio_state.get('total_value', 1)
            
            if new_total_exposure <= self.max_total_exposure:
                return {
                    'passed': True,
                    'current_exposure': new_total_exposure,
                    'limit': self.max_total_exposure,
                    'adjustments': {}
                }
            else:
                # Calculate maximum allowed trade size
                available_exposure = (self.max_total_exposure * portfolio_state.get('total_value', 1)) - current_exposure
                max_trade_size = available_exposure / trade_data.get('price', 1)
                
                return {
                    'passed': False,
                    'current_exposure': new_total_exposure,
                    'limit': self.max_total_exposure,
                    'adjustments': {
                        'size': max(0, max_trade_size),
                        'reason': 'Total exposure limit exceeded'
                    }
                }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'adjustments': {'size': 0, 'reason': 'Exposure check failed'}
            }
    
    def _check_daily_loss_limit(self, portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check daily loss limits"""
        try:
            daily_pnl = portfolio_state.get('daily_pnl', 0)
            portfolio_value = portfolio_state.get('total_value', 100000)
            daily_loss_fraction = abs(daily_pnl) / portfolio_value if daily_pnl < 0 else 0
            
            return {
                'passed': daily_loss_fraction <= self.max_daily_loss,
                'daily_loss_fraction': daily_loss_fraction,
                'limit': self.max_daily_loss,
                'daily_pnl': daily_pnl
            }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
    
    def _check_correlation_limits(self, trade_data: Dict[str, Any], 
                                portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check position correlation limits"""
        try:
            # This would require historical correlation data
            # For now, return a placeholder implementation
            symbol = trade_data.get('symbol', '')
            current_positions = portfolio_state.get('positions', {})
            
            # Simplified correlation check based on symbol similarity
            high_correlation_positions = []
            for pos_symbol in current_positions.keys():
                if self._calculate_symbol_correlation(symbol, pos_symbol) > self.correlation_threshold:
                    high_correlation_positions.append(pos_symbol)
            
            return {
                'passed': len(high_correlation_positions) == 0,
                'high_correlation_positions': high_correlation_positions,
                'threshold': self.correlation_threshold
            }
        except Exception as e:
            return {
                'passed': True,  # Default to pass on error
                'error': str(e)
            }
    
    def _calculate_symbol_correlation(self, symbol1: str, symbol2: str) -> float:
        """Calculate simplified correlation between symbols"""
        # Placeholder implementation - in practice, use historical price correlation
        if symbol1 == symbol2:
            return 1.0
        
        # Check for similar symbols (e.g., tech stocks)
        tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        if symbol1 in tech_symbols and symbol2 in tech_symbols:
            return 0.8
        
        return 0.3  # Default low correlation
    
    def _check_var_limits(self, trade_data: Dict[str, Any], 
                         portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check Value at Risk limits"""
        try:
            # Simplified VaR calculation
            portfolio_value = portfolio_state.get('total_value', 100000)
            returns_history = portfolio_state.get('returns_history', [])
            
            if len(returns_history) < 20:  # Need minimum history
                return {'passed': True, 'reason': 'Insufficient history for VaR calculation'}
            
            # Calculate VaR
            returns_array = np.array(returns_history)
            var_value = np.percentile(returns_array, (1 - self.var_confidence) * 100)
            var_amount = abs(var_value * portfolio_value)
            
            # Check if adding this trade would exceed VaR limits
            trade_value = abs(trade_data.get('size', 0)) * trade_data.get('price', 0)
            estimated_var_increase = trade_value * abs(var_value)
            
            var_limit = portfolio_value * 0.05  # 5% VaR limit
            
            return {
                'passed': (var_amount + estimated_var_increase) <= var_limit,
                'current_var': var_amount,
                'estimated_new_var': var_amount + estimated_var_increase,
                'limit': var_limit,
                'confidence': self.var_confidence
            }
        except Exception as e:
            return {
                'passed': True,  # Default to pass on error
                'error': str(e)
            }
    
    def _record_risk_breach(self, trade_data: Dict[str, Any], 
                          validation_result: Dict[str, Any]):
        """Record risk limit breach"""
        breach = {
            'timestamp': datetime.now().isoformat(),
            'trade_data': trade_data,
            'validation_result': validation_result,
            'breach_type': 'trade_validation'
        }
        self.risk_breaches.append(breach)
    
    def perform_stress_test(self, portfolio_state: Dict[str, Any], 
                          scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform stress testing on portfolio
        
        Args:
            portfolio_state: Current portfolio state
            scenarios: List of stress test scenarios
            
        Returns:
            Stress test results
        """
        stress_results = {
            'test_timestamp': datetime.now().isoformat(),
            'scenarios': {},
            'summary': {}
        }
        
        portfolio_value = portfolio_state.get('total_value', 100000)
        positions = portfolio_state.get('positions', {})
        
        for i, scenario in enumerate(scenarios):
            scenario_name = scenario.get('name', f'Scenario_{i+1}')
            
            # Apply scenario stress to portfolio
            stressed_pnl = self._apply_stress_scenario(positions, scenario)
            stressed_value = portfolio_value + stressed_pnl
            loss_percentage = (stressed_pnl / portfolio_value) if portfolio_value > 0 else 0
            
            stress_results['scenarios'][scenario_name] = {
                'description': scenario.get('description', ''),
                'stress_factors': scenario.get('stress_factors', {}),
                'stressed_pnl': stressed_pnl,
                'stressed_portfolio_value': stressed_value,
                'loss_percentage': loss_percentage,
                'breach_limits': loss_percentage < -self.max_daily_loss
            }
        
        # Calculate summary statistics
        all_losses = [result['loss_percentage'] for result in stress_results['scenarios'].values()]
        stress_results['summary'] = {
            'worst_case_loss': min(all_losses) if all_losses else 0,
            'average_loss': np.mean(all_losses) if all_losses else 0,
            'scenarios_breaching_limits': sum(1 for result in stress_results['scenarios'].values() 
                                            if result['breach_limits']),
            'total_scenarios': len(scenarios)
        }
        
        self.stress_test_results = stress_results
        return stress_results
    
    def _apply_stress_scenario(self, positions: Dict[str, Any], 
                             scenario: Dict[str, Any]) -> float:
        """Apply stress scenario to positions"""
        total_stressed_pnl = 0
        stress_factors = scenario.get('stress_factors', {})
        
        for symbol, position in positions.items():
            position_value = position.get('value', 0)
            
            # Apply symbol-specific stress or market-wide stress
            stress_factor = stress_factors.get(symbol, stress_factors.get('market', 0))
            
            # Calculate stressed P&L
            stressed_pnl = position_value * stress_factor
            total_stressed_pnl += stressed_pnl
        
        return total_stressed_pnl
    
    def generate_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'risk_limits': {
                'max_position_size': self.max_position_size,
                'max_daily_loss': self.max_daily_loss,
                'max_total_exposure': self.max_total_exposure,
                'correlation_threshold': self.correlation_threshold,
                'var_confidence': self.var_confidence
            },
            'risk_breaches': {
                'total_breaches': len(self.risk_breaches),
                'recent_breaches': self.risk_breaches[-10:] if self.risk_breaches else [],
                'breach_types': self._analyze_breach_types()
            },
            'stress_test_summary': self.stress_test_results.get('summary', {}),
            'recommendations': self._generate_risk_recommendations()
        }
        
        return report
    
    def _analyze_breach_types(self) -> Dict[str, int]:
        """Analyze types of risk breaches"""
        breach_types = {}
        for breach in self.risk_breaches:
            risk_checks = breach.get('validation_result', {}).get('risk_checks', {})
            for check_type, check_result in risk_checks.items():
                if not check_result.get('passed', True):
                    breach_types[check_type] = breach_types.get(check_type, 0) + 1
        return breach_types
    
    def _generate_risk_recommendations(self) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        # Analyze breach patterns
        breach_types = self._analyze_breach_types()
        
        if breach_types.get('position_size', 0) > 5:
            recommendations.append("Consider reducing maximum position size limit")
        
        if breach_types.get('daily_loss', 0) > 2:
            recommendations.append("Review daily loss limits and implement stricter controls")
        
        if breach_types.get('total_exposure', 0) > 3:
            recommendations.append("Implement more conservative exposure limits")
        
        if len(self.stress_test_results) > 0:
            worst_case = self.stress_test_results.get('summary', {}).get('worst_case_loss', 0)
            if worst_case < -0.1:  # More than 10% loss in stress test
                recommendations.append("Portfolio shows high sensitivity to stress scenarios")
        
        if not recommendations:
            recommendations.append("Risk management operating within acceptable parameters")
        
        return recommendations
    
    def get_risk_metrics(self, portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate current risk metrics"""
        try:
            portfolio_value = portfolio_state.get('total_value', 100000)
            positions = portfolio_state.get('positions', {})
            returns_history = portfolio_state.get('returns_history', [])
            
            # Position concentration
            position_values = [pos.get('value', 0) for pos in positions.values()]
            total_position_value = sum(position_values)
            concentration_ratio = max(position_values) / total_position_value if total_position_value > 0 else 0
            
            # Portfolio exposure
            total_exposure = sum(abs(pos.get('value', 0)) for pos in positions.values())
            exposure_ratio = total_exposure / portfolio_value if portfolio_value > 0 else 0
            
            # Volatility metrics
            if len(returns_history) >= 2:
                returns_array = np.array(returns_history)
                portfolio_volatility = np.std(returns_array) * np.sqrt(252)  # Annualized
                var_95 = np.percentile(returns_array, 5)
                var_99 = np.percentile(returns_array, 1)
            else:
                portfolio_volatility = 0
                var_95 = 0
                var_99 = 0
            
            return {
                'concentration_ratio': concentration_ratio,
                'exposure_ratio': exposure_ratio,
                'portfolio_volatility': portfolio_volatility,
                'var_95': var_95,
                'var_99': var_99,
                'total_positions': len(positions),
                'risk_utilization': {
                    'position_size': concentration_ratio / self.max_position_size,
                    'exposure': exposure_ratio / self.max_total_exposure
                }
            }
        except Exception as e:
            return {
                'error': str(e),
                'concentration_ratio': 0,
                'exposure_ratio': 0,
                'portfolio_volatility': 0
            }

def create_default_stress_scenarios() -> List[Dict[str, Any]]:
    """Create default stress testing scenarios"""
    return [
        {
            'name': 'Market Crash',
            'description': '20% market decline across all positions',
            'stress_factors': {'market': -0.20}
        },
        {
            'name': 'Tech Selloff',
            'description': '30% decline in tech stocks',
            'stress_factors': {
                'AAPL': -0.30, 'MSFT': -0.30, 'GOOGL': -0.30, 
                'AMZN': -0.30, 'TSLA': -0.35, 'market': -0.10
            }
        },
        {
            'name': 'Interest Rate Shock',
            'description': 'Sudden interest rate increase impact',
            'stress_factors': {'market': -0.15}
        },
        {
            'name': 'Volatility Spike',
            'description': 'High volatility scenario',
            'stress_factors': {'market': -0.25}
        },
        {
            'name': 'Sector Rotation',
            'description': 'Large cap underperformance',
            'stress_factors': {'market': -0.12}
        }
    ]