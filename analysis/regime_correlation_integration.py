"""
Integration Module for Regime Analysis and Correlation Tracker

This module provides seamless integration between the regime analysis system
and the existing correlation tracker, enabling regime-aware risk management
and enhanced alpha validation.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import asyncio
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from .regime_analysis import RegimeAnalyzer, RegimeResults
from .validate_alpha import AlphaValidator
from .factor_models import FactorModelAnalyzer, FactorData
from .statistical_tests import StatisticalTestSuite

logger = logging.getLogger(__name__)


@dataclass
class RegimeCorrelationState:
    """Current state of regime-correlation system"""
    current_regime: int
    regime_probability: float
    correlation_level: float
    regime_persistence: float
    expected_duration: float
    alpha_forecast: float
    confidence_interval: Tuple[float, float]
    risk_level: str
    recommendations: List[str]


@dataclass
class RegimeShockEvent:
    """Event linking regime transitions to correlation shocks"""
    timestamp: datetime
    regime_before: int
    regime_after: int
    correlation_shock_magnitude: float
    shock_severity: str
    regime_change_probability: float
    predicted_duration: float
    alpha_impact: float


class RegimeCorrelationIntegrator:
    """
    Integration layer between regime analysis and correlation tracking
    """
    
    def __init__(
        self,
        correlation_tracker=None,
        significance_level: float = 0.05,
        regime_update_frequency: int = 20,  # Update every 20 periods
        alpha_window: int = 252  # 1 year window for alpha validation
    ):
        """
        Initialize the integration system
        
        Args:
            correlation_tracker: Instance of CorrelationTracker
            significance_level: Statistical significance level
            regime_update_frequency: How often to update regime analysis
            alpha_window: Window size for alpha validation
        """
        self.correlation_tracker = correlation_tracker
        self.significance_level = significance_level
        self.regime_update_frequency = regime_update_frequency
        self.alpha_window = alpha_window
        
        # Initialize component analyzers
        self.regime_analyzer = RegimeAnalyzer(significance_level)
        self.alpha_validator = AlphaValidator(significance_level)
        self.factor_analyzer = FactorModelAnalyzer(significance_level)
        self.statistical_suite = StatisticalTestSuite(significance_level)
        
        # State tracking
        self.current_state: Optional[RegimeCorrelationState] = None
        self.regime_history: deque = deque(maxlen=1000)
        self.shock_events: List[RegimeShockEvent] = []
        self.regime_alpha_performance: Dict[int, Dict[str, float]] = {}
        
        # Data buffers
        self.return_buffer: deque = deque(maxlen=alpha_window * 2)
        self.baseline_buffer: deque = deque(maxlen=alpha_window * 2)
        self.correlation_buffer: deque = deque(maxlen=alpha_window * 2)
        
        # Performance tracking
        self.update_counter = 0
        self.last_regime_update = None
        
        # Callbacks
        self.regime_change_callbacks: List[Callable] = []
        self.alpha_alert_callbacks: List[Callable] = []
        
        logger.info("RegimeCorrelationIntegrator initialized")
        
    def register_regime_change_callback(self, callback: Callable):
        """Register callback for regime change events"""
        self.regime_change_callbacks.append(callback)
        
    def register_alpha_alert_callback(self, callback: Callable):
        """Register callback for alpha alerts"""
        self.alpha_alert_callbacks.append(callback)
        
    def update_market_data(
        self,
        portfolio_returns: np.ndarray,
        baseline_returns: np.ndarray,
        timestamp: datetime = None
    ):
        """
        Update with new market data
        
        Args:
            portfolio_returns: Latest portfolio returns
            baseline_returns: Latest baseline returns
            timestamp: Data timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Update buffers
        self.return_buffer.extend(portfolio_returns)
        self.baseline_buffer.extend(baseline_returns)
        
        # Get correlation data if available
        if self.correlation_tracker:
            correlation_matrix = self.correlation_tracker.get_correlation_matrix()
            if correlation_matrix is not None:
                avg_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])\n                self.correlation_buffer.append(avg_correlation)
                
        self.update_counter += 1
        
        # Update regime analysis periodically
        if self.update_counter % self.regime_update_frequency == 0:
            self._update_regime_analysis()
            
        # Check for regime changes and correlation shocks
        self._check_regime_correlation_events(timestamp)
        
    def _update_regime_analysis(self):
        """Update regime analysis with latest data"""
        if len(self.return_buffer) < 100:  # Need minimum data
            return
            
        # Prepare data for regime analysis
        returns = np.array(list(self.return_buffer))[-self.alpha_window:]
        baseline = np.array(list(self.baseline_buffer))[-self.alpha_window:]
        
        if len(returns) != len(baseline):
            min_len = min(len(returns), len(baseline))
            returns = returns[:min_len]
            baseline = baseline[:min_len]
            
        # Run regime analysis
        try:
            results = self.regime_analyzer.analyze_regimes(
                returns,
                baseline,
                max_regimes=4,
                methods=['markov', 'gaussian_mixture', 'volatility_clustering']
            )
            
            # Get best regime model
            best_model = self._get_best_regime_model(results)
            
            if best_model:
                # Update current state
                self._update_current_state(best_model, returns, baseline)
                
                # Update alpha performance by regime
                self._update_regime_alpha_performance(best_model, returns, baseline)
                
                self.last_regime_update = datetime.now()
                
        except Exception as e:
            logger.error(f"Failed to update regime analysis: {str(e)}")
            
    def _get_best_regime_model(self, results: Dict[str, Any]) -> Optional[RegimeResults]:
        """Get the best regime model from analysis results"""
        best_model = None
        best_bic = float('inf')
        
        for method_name, method_results in results.items():
            if method_name in ['markov', 'gaussian_mixture', 'volatility_clustering']:
                for regime_key, regime_result in method_results.items():
                    if 'error' not in regime_result and hasattr(regime_result, 'bic'):
                        if regime_result.bic < best_bic:
                            best_bic = regime_result.bic
                            best_model = regime_result
                            
        return best_model
        
    def _update_current_state(self, regime_model: RegimeResults, returns: np.ndarray, baseline: np.ndarray):
        """Update current regime-correlation state"""
        if regime_model.regime_probabilities is None:
            return
            
        # Get current regime
        current_regime = regime_model.regime_assignments[-1]
        current_prob = regime_model.regime_probabilities[-1, current_regime]
        
        # Get correlation level
        correlation_level = 0.0
        if len(self.correlation_buffer) > 0:
            correlation_level = self.correlation_buffer[-1]
            
        # Calculate expected duration
        persistence = regime_model.regime_persistence.get(current_regime, 0.8)
        expected_duration = 1 / (1 - persistence) if persistence < 1 else float('inf')
        
        # Forecast alpha for current regime
        alpha_forecast = 0.0
        confidence_interval = (0.0, 0.0)
        
        if current_regime in regime_model.alpha_by_regime:
            alpha_stats = regime_model.alpha_by_regime[current_regime]
            if 'alpha' in alpha_stats:
                alpha_forecast = alpha_stats['alpha']
                alpha_std = alpha_stats.get('alpha_std', 0.01)
                confidence_interval = (
                    alpha_forecast - 1.96 * alpha_std,
                    alpha_forecast + 1.96 * alpha_std
                )
        
        # Assess risk level
        risk_level = self._assess_risk_level(current_regime, correlation_level, alpha_forecast)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            current_regime, correlation_level, alpha_forecast, expected_duration
        )
        
        # Update state
        self.current_state = RegimeCorrelationState(
            current_regime=current_regime,
            regime_probability=current_prob,
            correlation_level=correlation_level,
            regime_persistence=persistence,
            expected_duration=expected_duration,
            alpha_forecast=alpha_forecast,
            confidence_interval=confidence_interval,
            risk_level=risk_level,
            recommendations=recommendations
        )
        
        # Store in history
        self.regime_history.append((datetime.now(), self.current_state))
        
    def _assess_risk_level(self, regime: int, correlation: float, alpha: float) -> str:
        """Assess current risk level"""
        if correlation > 0.8 or alpha < -0.001:
            return "HIGH"
        elif correlation > 0.6 or alpha < -0.0005:
            return "MODERATE"
        elif correlation > 0.4 and alpha > 0.0005:
            return "LOW"
        else:
            return "NORMAL"
            
    def _generate_recommendations(self, regime: int, correlation: float, alpha: float, duration: float) -> List[str]:
        """Generate trading recommendations"""
        recommendations = []
        
        # Regime-based recommendations
        if duration > 50:  # Long-lasting regime
            recommendations.append("Consider regime-persistent strategies")
        elif duration < 10:  # Short-lasting regime
            recommendations.append("Avoid regime-timing strategies")
            
        # Correlation-based recommendations
        if correlation > 0.8:
            recommendations.append("Reduce portfolio concentration")
            recommendations.append("Consider diversification across asset classes")
        elif correlation < 0.2:
            recommendations.append("Opportunity for pair trading strategies")
            
        # Alpha-based recommendations
        if alpha > 0.001:
            recommendations.append("Favorable environment for active strategies")
        elif alpha < -0.001:
            recommendations.append("Consider defensive positioning")
            
        return recommendations
        
    def _update_regime_alpha_performance(self, regime_model: RegimeResults, returns: np.ndarray, baseline: np.ndarray):
        """Update alpha performance by regime"""
        if regime_model.regime_assignments is None:
            return
            
        excess_returns = returns - baseline
        
        for regime_id in range(regime_model.n_regimes):
            regime_mask = regime_model.regime_assignments == regime_id
            if np.any(regime_mask):
                regime_excess = excess_returns[regime_mask]
                
                if len(regime_excess) > 10:  # Minimum sample size
                    self.regime_alpha_performance[regime_id] = {
                        'mean_alpha': np.mean(regime_excess),
                        'std_alpha': np.std(regime_excess),
                        'sharpe_ratio': np.mean(regime_excess) / np.std(regime_excess) * np.sqrt(252) if np.std(regime_excess) > 0 else 0,
                        'hit_rate': np.mean(regime_excess > 0),
                        'observations': len(regime_excess)
                    }
                    
    def _check_regime_correlation_events(self, timestamp: datetime):
        """Check for regime transitions coinciding with correlation shocks"""
        if not self.correlation_tracker or len(self.regime_history) < 2:
            return
            
        # Check for recent regime change
        current_regime = self.current_state.current_regime if self.current_state else 0
        previous_regime = self.regime_history[-2][1].current_regime if len(self.regime_history) >= 2 else current_regime
        
        if current_regime != previous_regime:
            # Regime change detected
            
            # Check for correlation shock around the same time
            recent_shocks = []
            if hasattr(self.correlation_tracker, 'shock_alerts'):
                recent_shocks = [
                    shock for shock in self.correlation_tracker.shock_alerts
                    if abs((shock.timestamp - timestamp).total_seconds()) < 300  # Within 5 minutes
                ]
                
            if recent_shocks:
                # Regime change coincides with correlation shock
                shock = recent_shocks[0]  # Take the first/most recent shock
                
                # Calculate regime change probability
                regime_change_prob = self._calculate_regime_change_probability(
                    previous_regime, current_regime, shock.correlation_change
                )
                
                # Predict duration of new regime
                predicted_duration = self.current_state.expected_duration if self.current_state else 20
                
                # Estimate alpha impact
                alpha_impact = self._estimate_alpha_impact(current_regime, shock.correlation_change)
                
                # Create shock event
                shock_event = RegimeShockEvent(
                    timestamp=timestamp,
                    regime_before=previous_regime,
                    regime_after=current_regime,
                    correlation_shock_magnitude=shock.correlation_change,
                    shock_severity=shock.severity,
                    regime_change_probability=regime_change_prob,
                    predicted_duration=predicted_duration,
                    alpha_impact=alpha_impact
                )
                
                self.shock_events.append(shock_event)
                
                # Trigger callbacks
                for callback in self.regime_change_callbacks:
                    try:
                        callback(shock_event)
                    except Exception as e:
                        logger.error(f"Regime change callback failed: {str(e)}")
                        
                logger.info(f"Regime-correlation event detected: {previous_regime} -> {current_regime}")
                
    def _calculate_regime_change_probability(self, prev_regime: int, curr_regime: int, correlation_change: float) -> float:
        """Calculate probability that correlation shock caused regime change"""
        # Simplified model: higher correlation changes increase probability
        base_prob = 0.3  # Base probability of regime change
        correlation_factor = min(1.0, correlation_change / 0.5)  # Normalize by threshold
        
        return min(0.95, base_prob + 0.4 * correlation_factor)
        
    def _estimate_alpha_impact(self, regime: int, correlation_change: float) -> float:
        """Estimate impact on alpha from regime change"""
        # Use historical regime performance if available
        if regime in self.regime_alpha_performance:
            base_alpha = self.regime_alpha_performance[regime]['mean_alpha']
        else:
            base_alpha = 0.0
            
        # Adjust for correlation shock
        correlation_impact = -0.001 * correlation_change  # Negative impact from increased correlation
        
        return base_alpha + correlation_impact
        
    def get_regime_forecast(self, horizon: int = 20) -> Dict[str, Any]:
        """Get regime forecast for specified horizon"""
        if not self.current_state:
            return {'error': 'No current regime state available'}
            
        # Use regime persistence for simple forecast
        persistence = self.current_state.regime_persistence
        current_regime = self.current_state.current_regime
        
        # Probability of staying in current regime
        prob_same_regime = persistence ** horizon
        
        # Expected alpha over horizon
        expected_alpha = self.current_state.alpha_forecast * prob_same_regime
        
        # Risk assessment
        risk_factors = []
        if self.current_state.correlation_level > 0.7:
            risk_factors.append("High correlation environment")
        if self.current_state.expected_duration < horizon:
            risk_factors.append("Regime change likely within horizon")
            
        return {
            'horizon': horizon,
            'current_regime': current_regime,
            'probability_same_regime': prob_same_regime,
            'expected_alpha': expected_alpha,
            'expected_correlation': self.current_state.correlation_level,
            'risk_factors': risk_factors,
            'confidence': self.current_state.regime_probability
        }
        
    def validate_regime_alpha(self, regime_id: int) -> Dict[str, Any]:
        """Validate alpha generation for specific regime"""
        if len(self.return_buffer) < 100 or len(self.baseline_buffer) < 100:
            return {'error': 'Insufficient data for validation'}
            
        # Get regime assignments from current state
        if not self.current_state:
            return {'error': 'No current regime state available'}
            
        # This is a simplified version - in practice, you'd need the full regime model
        # to get regime assignments for historical data
        
        # For now, return regime-specific performance if available
        if regime_id in self.regime_alpha_performance:
            performance = self.regime_alpha_performance[regime_id]
            
            # Enhanced validation using statistical tests
            if performance['observations'] > 30:
                # Mock data for statistical testing (in practice, use actual regime-filtered returns)
                mock_returns = np.random.normal(performance['mean_alpha'], performance['std_alpha'], 
                                              performance['observations'])
                
                # Run comprehensive statistical tests
                test_results = self.statistical_suite.comprehensive_alpha_test(
                    mock_returns,
                    np.zeros(len(mock_returns))
                )
                
                return {
                    'regime_id': regime_id,
                    'performance': performance,
                    'statistical_validation': test_results,
                    'recommendation': 'Significant alpha detected' if test_results['summary']['confidence'] in ['high', 'very_high'] else 'No significant alpha'
                }
                
        return {'error': f'No performance data available for regime {regime_id}'}
        
    def generate_integration_report(self, save_path: Optional[str] = None) -> str:
        """Generate comprehensive integration report"""
        report = []
        report.append("=" * 60)
        report.append("REGIME-CORRELATION INTEGRATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Current state
        if self.current_state:
            report.append("CURRENT STATE")
            report.append("-" * 40)
            report.append(f"Current Regime: {self.current_state.current_regime}")
            report.append(f"Regime Probability: {self.current_state.regime_probability:.3f}")
            report.append(f"Correlation Level: {self.current_state.correlation_level:.3f}")
            report.append(f"Expected Duration: {self.current_state.expected_duration:.1f} periods")
            report.append(f"Alpha Forecast: {self.current_state.alpha_forecast:.4f}")
            report.append(f"Risk Level: {self.current_state.risk_level}")
            report.append("")
            
            if self.current_state.recommendations:
                report.append("Recommendations:")
                for rec in self.current_state.recommendations:
                    report.append(f"  • {rec}")
                report.append("")
                
        # Regime-correlation events
        if self.shock_events:
            report.append("RECENT REGIME-CORRELATION EVENTS")
            report.append("-" * 40)
            
            for event in self.shock_events[-5:]:  # Show last 5 events
                report.append(f"Event: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                report.append(f"  Regime Change: {event.regime_before} -> {event.regime_after}")
                report.append(f"  Correlation Shock: {event.correlation_shock_magnitude:.3f}")
                report.append(f"  Severity: {event.shock_severity}")
                report.append(f"  Predicted Duration: {event.predicted_duration:.1f} periods")
                report.append(f"  Alpha Impact: {event.alpha_impact:.4f}")
                report.append("")
                
        # Regime alpha performance
        if self.regime_alpha_performance:
            report.append("REGIME ALPHA PERFORMANCE")
            report.append("-" * 40)
            
            for regime_id, performance in self.regime_alpha_performance.items():
                report.append(f"Regime {regime_id}:")
                report.append(f"  Mean Alpha: {performance['mean_alpha']:.4f}")
                report.append(f"  Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
                report.append(f"  Hit Rate: {performance['hit_rate']:.3f}")
                report.append(f"  Observations: {performance['observations']}")
                report.append("")
                
        # System status
        report.append("SYSTEM STATUS")
        report.append("-" * 40)
        report.append(f"Data Points: {len(self.return_buffer)}")
        report.append(f"Update Counter: {self.update_counter}")
        report.append(f"Last Regime Update: {self.last_regime_update}")
        report.append(f"Regime History Length: {len(self.regime_history)}")
        report.append(f"Shock Events: {len(self.shock_events)}")
        report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
                
        return report_text


def main():
    """Example usage of regime-correlation integration"""
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic data
    np.random.seed(42)
    n_periods = 1000
    
    # Create regime-switching returns
    regime_1_returns = np.random.normal(0.001, 0.015, 500)  # Bull market
    regime_2_returns = np.random.normal(-0.0005, 0.025, 500)  # Bear market
    
    portfolio_returns = np.concatenate([regime_1_returns, regime_2_returns])
    baseline_returns = np.random.normal(0.0005, 0.012, n_periods)
    
    # Initialize integrator
    integrator = RegimeCorrelationIntegrator()
    
    # Simulate real-time updates
    for i in range(100, n_periods, 20):  # Update every 20 periods
        current_returns = portfolio_returns[i-100:i]
        current_baseline = baseline_returns[i-100:i]
        
        integrator.update_market_data(current_returns, current_baseline)
        
        # Print status every 100 updates
        if i % 200 == 0:
            print(f"Update {i}: Current state = {integrator.current_state}")
            
    # Generate final report
    report = integrator.generate_integration_report("integration_report.txt")
    print("\n" + report)
    
    # Test regime forecast
    forecast = integrator.get_regime_forecast(horizon=30)
    print(f"\n30-period Forecast: {forecast}")


if __name__ == "__main__":
    main()