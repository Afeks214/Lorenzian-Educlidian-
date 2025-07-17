"""
Advanced Portfolio Correlation Manager

This module implements sophisticated correlation management for the Portfolio Optimizer Agent,
providing real-time correlation analysis, correlation risk scoring, and diversification
optimization with regime-aware adjustments.

Key Features:
- Real-time correlation monitoring and regime detection
- Dynamic correlation risk scoring and alerts
- Diversification optimization algorithms
- Correlation breakdown detection and response
- Advanced correlation forecasting models
- Risk-adjusted correlation weighting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import structlog
from scipy import linalg, stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.core.events import EventBus, Event, EventType
from src.risk.core.correlation_tracker import CorrelationTracker, CorrelationRegime

logger = structlog.get_logger()


class CorrelationRiskLevel(Enum):
    """Correlation risk classification levels"""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class CorrelationRiskMetrics:
    """Comprehensive correlation risk metrics"""
    timestamp: datetime
    average_correlation: float
    max_correlation: float
    correlation_spread: float
    eigenvalue_concentration: float  # First eigenvalue / total eigenvalues
    effective_bets: float  # Diversification measure
    correlation_risk_score: float  # Composite risk score (0-1)
    risk_level: CorrelationRiskLevel
    regime: CorrelationRegime


@dataclass
class DiversificationMetrics:
    """Portfolio diversification metrics"""
    diversification_ratio: float  # Weighted avg vol / portfolio vol
    effective_strategies: float  # Number of effective strategies
    concentration_index: float  # Herfindahl index for weights
    correlation_adjusted_concentration: float  # Risk-adjusted concentration
    max_component_risk: float  # Maximum single strategy risk contribution


@dataclass
class CorrelationForecast:
    """Correlation forecast and confidence intervals"""
    horizon_days: int
    forecasted_correlation: np.ndarray
    confidence_lower: np.ndarray
    confidence_upper: np.ndarray
    forecast_confidence: float
    regime_probability: Dict[CorrelationRegime, float]


class PortfolioCorrelationManager:
    """
    Advanced correlation management system for portfolio optimization.
    
    Provides real-time correlation analysis, risk scoring, and diversification
    optimization with sophisticated regime detection and forecasting capabilities.
    """
    
    def __init__(
        self,
        correlation_tracker: CorrelationTracker,
        event_bus: EventBus,
        n_strategies: int = 5,
        correlation_window: int = 252,  # 1 year of daily observations
        forecast_horizon: int = 21  # 3 weeks forecast
    ):
        """
        Initialize Portfolio Correlation Manager
        
        Args:
            correlation_tracker: Base correlation tracking system
            event_bus: Event bus for communication
            n_strategies: Number of strategies in portfolio
            correlation_window: Rolling window for correlation calculation
            forecast_horizon: Days ahead for correlation forecasting
        """
        self.correlation_tracker = correlation_tracker
        self.event_bus = event_bus
        self.n_strategies = n_strategies
        self.correlation_window = correlation_window
        self.forecast_horizon = forecast_horizon
        
        # Correlation analysis parameters
        self.risk_thresholds = {
            CorrelationRiskLevel.LOW: 0.3,
            CorrelationRiskLevel.MODERATE: 0.5,
            CorrelationRiskLevel.HIGH: 0.7,
            CorrelationRiskLevel.CRITICAL: 0.8
        }
        
        # Historical correlation data
        self.correlation_history: List[Tuple[datetime, np.ndarray]] = []
        self.risk_metrics_history: List[CorrelationRiskMetrics] = []
        
        # Diversification tracking
        self.diversification_target = 0.8  # Target diversification ratio
        self.max_concentration = 0.4  # Maximum single strategy concentration
        
        # Principal Component Analysis for regime detection
        self.pca_model = PCA(n_components=min(5, n_strategies))
        self.scaler = StandardScaler()
        
        # Correlation forecasting
        self.forecast_cache: Dict[int, CorrelationForecast] = {}
        self.forecast_cache_ttl = timedelta(hours=1)
        self.last_forecast_time: Optional[datetime] = None
        
        # Performance tracking
        self.calculation_times: List[float] = []
        self.forecast_accuracy_history: List[float] = []
        
        # Event subscriptions
        self._setup_event_subscriptions()
        
        logger.info("Portfolio Correlation Manager initialized",
                   n_strategies=n_strategies,
                   correlation_window=correlation_window,
                   forecast_horizon=forecast_horizon)
    
    def _setup_event_subscriptions(self):
        """Setup event subscriptions for real-time updates"""
        self.event_bus.subscribe(EventType.VAR_UPDATE, self._handle_correlation_update)
        self.event_bus.subscribe(EventType.RISK_UPDATE, self._handle_risk_update)
    
    def _handle_correlation_update(self, event: Event):
        """Handle correlation matrix updates"""
        correlation_data = event.payload
        if isinstance(correlation_data, dict) and 'correlation_matrix' in correlation_data:
            correlation_matrix = correlation_data['correlation_matrix']
            self._update_correlation_analysis(correlation_matrix)
    
    def _handle_risk_update(self, event: Event):
        """Handle risk update events"""
        risk_data = event.payload
        if isinstance(risk_data, dict) and risk_data.get('type') == 'CORRELATION_SHOCK':
            self._handle_correlation_shock(risk_data)
    
    def analyze_correlation_risk(
        self,
        correlation_matrix: Optional[np.ndarray] = None
    ) -> CorrelationRiskMetrics:
        """
        Comprehensive correlation risk analysis
        
        Args:
            correlation_matrix: Correlation matrix to analyze (uses current if None)
            
        Returns:
            Comprehensive correlation risk metrics
        """
        start_time = datetime.now()
        
        if correlation_matrix is None:
            correlation_matrix = self.correlation_tracker.get_correlation_matrix()
        
        if correlation_matrix is None or correlation_matrix.shape[0] < 2:
            # Return default metrics for insufficient data
            return CorrelationRiskMetrics(
                timestamp=datetime.now(),
                average_correlation=0.0,
                max_correlation=0.0,
                correlation_spread=0.0,
                eigenvalue_concentration=1.0 / self.n_strategies,
                effective_bets=self.n_strategies,
                correlation_risk_score=0.0,
                risk_level=CorrelationRiskLevel.LOW,
                regime=CorrelationRegime.NORMAL
            )
        
        try:
            # Basic correlation statistics
            upper_tri_mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
            off_diagonal_correlations = correlation_matrix[upper_tri_mask]
            
            avg_correlation = np.mean(off_diagonal_correlations)
            max_correlation = np.max(off_diagonal_correlations)
            correlation_spread = np.std(off_diagonal_correlations)
            
            # Eigenvalue analysis for concentration risk
            eigenvalues = np.linalg.eigvals(correlation_matrix)
            eigenvalues = np.real(eigenvalues[eigenvalues > 1e-10])  # Remove numerical zeros
            eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
            
            total_eigenvalues = np.sum(eigenvalues)
            eigenvalue_concentration = eigenvalues[0] / total_eigenvalues if total_eigenvalues > 0 else 1.0
            
            # Effective number of bets (diversification measure)
            effective_bets = total_eigenvalues**2 / np.sum(eigenvalues**2) if np.sum(eigenvalues**2) > 0 else 1.0
            
            # Composite correlation risk score
            risk_score = self._calculate_correlation_risk_score(
                avg_correlation, max_correlation, eigenvalue_concentration, effective_bets
            )
            
            # Risk level classification
            risk_level = self._classify_correlation_risk(risk_score)
            
            # Current regime from correlation tracker
            current_regime = self.correlation_tracker.current_regime
            
            metrics = CorrelationRiskMetrics(
                timestamp=datetime.now(),
                average_correlation=avg_correlation,
                max_correlation=max_correlation,
                correlation_spread=correlation_spread,
                eigenvalue_concentration=eigenvalue_concentration,
                effective_bets=effective_bets,
                correlation_risk_score=risk_score,
                risk_level=risk_level,
                regime=current_regime
            )
            
            # Store metrics history
            self.risk_metrics_history.append(metrics)
            if len(self.risk_metrics_history) > 1000:  # Keep recent history only
                self.risk_metrics_history = self.risk_metrics_history[-1000:]
            
            # Track calculation time
            calc_time = (datetime.now() - start_time).total_seconds() * 1000
            self.calculation_times.append(calc_time)
            
            # Publish risk metrics if significant change
            self._publish_correlation_risk_update(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error("Correlation risk analysis failed", error=str(e))
            # Return safe default
            return CorrelationRiskMetrics(
                timestamp=datetime.now(),
                average_correlation=0.5,
                max_correlation=0.7,
                correlation_spread=0.2,
                eigenvalue_concentration=0.5,
                effective_bets=2.0,
                correlation_risk_score=0.5,
                risk_level=CorrelationRiskLevel.MODERATE,
                regime=CorrelationRegime.ELEVATED
            )
    
    def calculate_diversification_metrics(
        self,
        weights: np.ndarray,
        volatilities: np.ndarray,
        correlation_matrix: Optional[np.ndarray] = None
    ) -> DiversificationMetrics:
        """
        Calculate comprehensive diversification metrics
        
        Args:
            weights: Portfolio weights
            volatilities: Strategy volatilities
            correlation_matrix: Correlation matrix
            
        Returns:
            Diversification metrics
        """
        if correlation_matrix is None:
            correlation_matrix = self.correlation_tracker.get_correlation_matrix()
            if correlation_matrix is None:
                correlation_matrix = np.eye(len(weights))
        
        # Ensure dimensions match
        if correlation_matrix.shape[0] != len(weights):
            correlation_matrix = np.eye(len(weights))
        
        try:
            # 1. Diversification ratio
            weighted_avg_vol = np.dot(weights, volatilities)
            volatility_matrix = np.outer(volatilities, volatilities) * correlation_matrix
            portfolio_variance = np.dot(weights, np.dot(volatility_matrix, weights))
            portfolio_vol = np.sqrt(max(0, portfolio_variance))
            
            diversification_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0
            
            # 2. Effective number of strategies (inverse Herfindahl)
            concentration_index = np.sum(weights**2)
            effective_strategies = 1.0 / concentration_index if concentration_index > 0 else 1.0
            
            # 3. Correlation-adjusted concentration
            # Weight each strategy by its correlation with others
            correlation_adjusted_weights = np.zeros_like(weights)
            for i in range(len(weights)):
                avg_correlation_with_others = np.mean([
                    correlation_matrix[i, j] for j in range(len(weights)) if j != i
                ])
                correlation_adjusted_weights[i] = weights[i] * (1 + avg_correlation_with_others)
            
            # Normalize
            correlation_adjusted_weights /= np.sum(correlation_adjusted_weights)
            correlation_adjusted_concentration = np.sum(correlation_adjusted_weights**2)
            
            # 4. Maximum component risk contribution
            if portfolio_variance > 0:
                marginal_contributions = np.dot(volatility_matrix, weights) / np.sqrt(portfolio_variance)
                component_contributions = weights * marginal_contributions
                component_contributions /= np.sum(component_contributions)  # Normalize
                max_component_risk = np.max(component_contributions)
            else:
                max_component_risk = np.max(weights)
            
            return DiversificationMetrics(
                diversification_ratio=diversification_ratio,
                effective_strategies=effective_strategies,
                concentration_index=concentration_index,
                correlation_adjusted_concentration=correlation_adjusted_concentration,
                max_component_risk=max_component_risk
            )
            
        except Exception as e:
            logger.error("Diversification metrics calculation failed", error=str(e))
            # Return conservative estimates
            return DiversificationMetrics(
                diversification_ratio=1.0,
                effective_strategies=len(weights),
                concentration_index=1.0 / len(weights),
                correlation_adjusted_concentration=1.0 / len(weights),
                max_component_risk=np.max(weights)
            )
    
    def optimize_for_diversification(
        self,
        expected_returns: np.ndarray,
        volatilities: np.ndarray,
        correlation_matrix: Optional[np.ndarray] = None,
        min_weight: float = 0.05,
        max_weight: float = 0.5
    ) -> Tuple[np.ndarray, DiversificationMetrics]:
        """
        Optimize portfolio for maximum diversification
        
        Args:
            expected_returns: Expected strategy returns
            volatilities: Strategy volatilities
            correlation_matrix: Correlation matrix
            min_weight: Minimum weight constraint
            max_weight: Maximum weight constraint
            
        Returns:
            Tuple of (optimal_weights, diversification_metrics)
        """
        if correlation_matrix is None:
            correlation_matrix = self.correlation_tracker.get_correlation_matrix()
            if correlation_matrix is None:
                correlation_matrix = np.eye(len(expected_returns))
        
        try:
            from scipy.optimize import minimize
            
            def diversification_objective(weights):
                """Negative diversification ratio (to minimize)"""
                weighted_avg_vol = np.dot(weights, volatilities)
                volatility_matrix = np.outer(volatilities, volatilities) * correlation_matrix
                portfolio_variance = np.dot(weights, np.dot(volatility_matrix, weights))
                portfolio_vol = np.sqrt(max(portfolio_variance, 1e-10))
                
                diversification_ratio = weighted_avg_vol / portfolio_vol
                return -diversification_ratio  # Minimize negative = maximize positive
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
            ]
            
            # Bounds
            bounds = [(min_weight, max_weight) for _ in range(len(expected_returns))]
            
            # Initial guess (equal weights)
            x0 = np.array([1.0 / len(expected_returns)] * len(expected_returns))
            
            # Optimization
            result = minimize(
                diversification_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-9}
            )
            
            if result.success:
                optimal_weights = result.x
            else:
                logger.warning("Diversification optimization failed, using equal weights")
                optimal_weights = x0
            
            # Calculate diversification metrics for optimal weights
            div_metrics = self.calculate_diversification_metrics(
                optimal_weights, volatilities, correlation_matrix
            )
            
            return optimal_weights, div_metrics
            
        except Exception as e:
            logger.error("Diversification optimization failed", error=str(e))
            # Return equal weights as fallback
            equal_weights = np.array([1.0 / len(expected_returns)] * len(expected_returns))
            div_metrics = self.calculate_diversification_metrics(
                equal_weights, volatilities, correlation_matrix
            )
            return equal_weights, div_metrics
    
    def forecast_correlation_regime(
        self,
        horizon_days: int = None
    ) -> CorrelationForecast:
        """
        Forecast correlation regime and matrix evolution
        
        Args:
            horizon_days: Forecast horizon (uses default if None)
            
        Returns:
            Correlation forecast with confidence intervals
        """
        if horizon_days is None:
            horizon_days = self.forecast_horizon
        
        # Check cache
        if (horizon_days in self.forecast_cache and 
            self.last_forecast_time and
            datetime.now() - self.last_forecast_time < self.forecast_cache_ttl):
            return self.forecast_cache[horizon_days]
        
        try:
            # Get recent correlation history
            if len(self.correlation_history) < 30:  # Need minimum history
                return self._default_correlation_forecast(horizon_days)
            
            # Extract correlation time series
            correlation_series = []
            for timestamp, corr_matrix in self.correlation_history[-252:]:  # Last year
                # Use average correlation as single metric
                upper_tri_mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                avg_corr = np.mean(corr_matrix[upper_tri_mask])
                correlation_series.append(avg_corr)
            
            correlation_series = np.array(correlation_series)
            
            # Simple autoregressive forecast
            if len(correlation_series) >= 10:
                # AR(1) model: corr(t+1) = alpha + beta * corr(t) + error
                X = correlation_series[:-1].reshape(-1, 1)
                y = correlation_series[1:]
                
                # Simple linear regression
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X, y)
                
                # Forecast
                current_corr = correlation_series[-1]
                forecasted_correlations = []
                forecast_point = current_corr
                
                for _ in range(horizon_days):
                    forecast_point = model.predict([[forecast_point]])[0]
                    forecasted_correlations.append(forecast_point)
                
                forecasted_correlations = np.array(forecasted_correlations)
                
                # Simple confidence intervals (±2 std of residuals)
                residuals = y - model.predict(X)
                forecast_std = np.std(residuals)
                confidence_lower = forecasted_correlations - 2 * forecast_std
                confidence_upper = forecasted_correlations + 2 * forecast_std
                
                # Clip to valid correlation range
                forecasted_correlations = np.clip(forecasted_correlations, -1, 1)
                confidence_lower = np.clip(confidence_lower, -1, 1)
                confidence_upper = np.clip(confidence_upper, -1, 1)
                
                # Forecast confidence based on model R²
                forecast_confidence = max(0.1, model.score(X, y))
                
            else:
                # Insufficient data for forecasting
                return self._default_correlation_forecast(horizon_days)
            
            # Regime probability forecast (simplified)
            current_avg_corr = correlation_series[-1]
            regime_probabilities = {
                CorrelationRegime.NORMAL: max(0.1, 0.8 - current_avg_corr),
                CorrelationRegime.ELEVATED: 0.6 * current_avg_corr if current_avg_corr < 0.7 else 0.3,
                CorrelationRegime.CRISIS: max(0.0, current_avg_corr - 0.6) if current_avg_corr > 0.6 else 0.1,
                CorrelationRegime.SHOCK: 0.05
            }
            
            # Normalize probabilities
            total_prob = sum(regime_probabilities.values())
            regime_probabilities = {k: v/total_prob for k, v in regime_probabilities.items()}
            
            # Create forecast matrix (simplified - use average correlation)
            current_matrix = self.correlation_tracker.get_correlation_matrix()
            if current_matrix is None:
                current_matrix = np.eye(self.n_strategies)
            
            forecasted_matrix = current_matrix.copy()
            # Adjust all off-diagonal elements by forecast change
            if len(forecasted_correlations) > 0:
                avg_change = forecasted_correlations[-1] - current_avg_corr
                for i in range(forecasted_matrix.shape[0]):
                    for j in range(i+1, forecasted_matrix.shape[1]):
                        new_corr = forecasted_matrix[i, j] + avg_change
                        new_corr = np.clip(new_corr, -0.99, 0.99)  # Valid correlation range
                        forecasted_matrix[i, j] = new_corr
                        forecasted_matrix[j, i] = new_corr
            
            forecast = CorrelationForecast(
                horizon_days=horizon_days,
                forecasted_correlation=forecasted_matrix,
                confidence_lower=np.full_like(forecasted_matrix, np.mean(confidence_lower)),
                confidence_upper=np.full_like(forecasted_matrix, np.mean(confidence_upper)),
                forecast_confidence=forecast_confidence,
                regime_probability=regime_probabilities
            )
            
            # Cache result
            self.forecast_cache[horizon_days] = forecast
            self.last_forecast_time = datetime.now()
            
            return forecast
            
        except Exception as e:
            logger.error("Correlation forecasting failed", error=str(e))
            return self._default_correlation_forecast(horizon_days)
    
    def _default_correlation_forecast(self, horizon_days: int) -> CorrelationForecast:
        """Create default correlation forecast for insufficient data"""
        current_matrix = self.correlation_tracker.get_correlation_matrix()
        if current_matrix is None:
            current_matrix = np.eye(self.n_strategies)
        
        return CorrelationForecast(
            horizon_days=horizon_days,
            forecasted_correlation=current_matrix,
            confidence_lower=current_matrix * 0.8,
            confidence_upper=current_matrix * 1.2,
            forecast_confidence=0.3,
            regime_probability={
                CorrelationRegime.NORMAL: 0.7,
                CorrelationRegime.ELEVATED: 0.2,
                CorrelationRegime.CRISIS: 0.08,
                CorrelationRegime.SHOCK: 0.02
            }
        )
    
    def _calculate_correlation_risk_score(
        self,
        avg_correlation: float,
        max_correlation: float,
        eigenvalue_concentration: float,
        effective_bets: float
    ) -> float:
        """Calculate composite correlation risk score (0-1)"""
        # Normalize components to 0-1 scale
        corr_component = min(1.0, avg_correlation / 0.8)  # Risk increases with correlation
        max_corr_component = min(1.0, max_correlation / 0.9)
        concentration_component = min(1.0, eigenvalue_concentration / 0.8)
        diversity_component = 1.0 - min(1.0, effective_bets / self.n_strategies)
        
        # Weighted combination
        risk_score = (
            0.3 * corr_component +
            0.25 * max_corr_component +
            0.25 * concentration_component +
            0.2 * diversity_component
        )
        
        return np.clip(risk_score, 0.0, 1.0)
    
    def _classify_correlation_risk(self, risk_score: float) -> CorrelationRiskLevel:
        """Classify correlation risk level based on score"""
        if risk_score >= 0.8:
            return CorrelationRiskLevel.CRITICAL
        elif risk_score >= 0.6:
            return CorrelationRiskLevel.HIGH
        elif risk_score >= 0.4:
            return CorrelationRiskLevel.MODERATE
        else:
            return CorrelationRiskLevel.LOW
    
    def _update_correlation_analysis(self, correlation_matrix: np.ndarray):
        """Update correlation analysis with new matrix"""
        # Store correlation history
        self.correlation_history.append((datetime.now(), correlation_matrix.copy()))
        
        # Keep only recent history for memory efficiency
        if len(self.correlation_history) > self.correlation_window * 2:
            self.correlation_history = self.correlation_history[-self.correlation_window:]
        
        # Perform analysis
        risk_metrics = self.analyze_correlation_risk(correlation_matrix)
        
        # Check for risk level changes
        if len(self.risk_metrics_history) >= 2:
            previous_level = self.risk_metrics_history[-2].risk_level
            current_level = risk_metrics.risk_level
            
            if current_level.value != previous_level.value:
                self._handle_risk_level_change(previous_level, current_level, risk_metrics)
    
    def _handle_correlation_shock(self, shock_data: Dict):
        """Handle correlation shock events"""
        severity = shock_data.get('severity', 'MODERATE')
        
        logger.warning("Correlation shock detected by manager",
                      severity=severity,
                      timestamp=datetime.now())
        
        # Update internal state and trigger additional analysis if needed
        # This could trigger more sophisticated risk management responses
    
    def _handle_risk_level_change(
        self,
        previous_level: CorrelationRiskLevel,
        current_level: CorrelationRiskLevel,
        metrics: CorrelationRiskMetrics
    ):
        """Handle changes in correlation risk level"""
        logger.info("Correlation risk level changed",
                   from_level=previous_level.value,
                   to_level=current_level.value,
                   risk_score=metrics.correlation_risk_score)
        
        # Publish risk level change event
        if self.event_bus:
            self.event_bus.publish(
                self.event_bus.create_event(
                    EventType.RISK_UPDATE,
                    {
                        'type': 'CORRELATION_RISK_LEVEL_CHANGE',
                        'previous_level': previous_level.value,
                        'current_level': current_level.value,
                        'risk_metrics': {
                            'risk_score': metrics.correlation_risk_score,
                            'average_correlation': metrics.average_correlation,
                            'effective_bets': metrics.effective_bets
                        }
                    },
                    'PortfolioCorrelationManager'
                )
            )
    
    def _publish_correlation_risk_update(self, metrics: CorrelationRiskMetrics):
        """Publish correlation risk metrics update"""
        # Only publish significant changes to avoid noise
        if (not self.risk_metrics_history or 
            len(self.risk_metrics_history) < 2 or
            abs(metrics.correlation_risk_score - self.risk_metrics_history[-2].correlation_risk_score) > 0.1):
            
            if self.event_bus:
                self.event_bus.publish(
                    self.event_bus.create_event(
                        EventType.RISK_UPDATE,
                        {
                            'type': 'CORRELATION_RISK_UPDATE',
                            'risk_metrics': metrics,
                            'timestamp': metrics.timestamp
                        },
                        'PortfolioCorrelationManager'
                    )
                )
    
    def get_correlation_summary(self) -> Dict:
        """Get comprehensive correlation analysis summary"""
        if not self.risk_metrics_history:
            return {"status": "No correlation analysis available"}
        
        latest_metrics = self.risk_metrics_history[-1]
        
        return {
            "current_risk_level": latest_metrics.risk_level.value,
            "risk_score": latest_metrics.correlation_risk_score,
            "average_correlation": latest_metrics.average_correlation,
            "max_correlation": latest_metrics.max_correlation,
            "effective_bets": latest_metrics.effective_bets,
            "eigenvalue_concentration": latest_metrics.eigenvalue_concentration,
            "correlation_regime": latest_metrics.regime.value,
            "calculation_count": len(self.calculation_times),
            "avg_calculation_time_ms": np.mean(self.calculation_times) if self.calculation_times else 0,
            "forecast_available": len(self.forecast_cache) > 0
        }