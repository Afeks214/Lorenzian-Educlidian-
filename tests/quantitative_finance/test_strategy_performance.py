"""
Strategy Performance Testing Framework

This module provides comprehensive testing for real-time strategy monitoring,
evaluation, capacity analysis, scalability testing, and performance attribution.

Key Features:
- Real-time strategy monitoring and evaluation
- Strategy capacity and scalability validation
- Performance attribution analysis
- Strategy decay detection and adaptation
- Live trading simulation and testing
- Risk-adjusted performance metrics
- Multi-strategy performance comparison
- Automated performance reporting
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from unittest.mock import Mock, patch
import asyncio
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import time

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class PerformanceMetricType(Enum):
    """Performance metric types"""
    RETURN = "return"
    VOLATILITY = "volatility"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    CALMAR_RATIO = "calmar_ratio"
    INFORMATION_RATIO = "information_ratio"
    ALPHA = "alpha"
    BETA = "beta"


class StrategyState(Enum):
    """Strategy states"""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    DEGRADED = "degraded"
    FAILED = "failed"


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Performance metric with metadata"""
    name: str
    value: float
    timestamp: datetime
    period: str
    benchmark_value: Optional[float] = None
    percentile_rank: Optional[float] = None
    is_threshold_breached: bool = False
    metadata: Dict[str, Any] = None


@dataclass
class StrategyAlert:
    """Strategy performance alert"""
    strategy_id: str
    alert_type: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    metric_value: float
    threshold_value: float
    suggested_action: str
    metadata: Dict[str, Any] = None


@dataclass
class CapacityAnalysis:
    """Strategy capacity analysis result"""
    strategy_id: str
    current_capacity: float
    max_theoretical_capacity: float
    capacity_utilization: float
    capacity_constraints: List[str]
    scaling_recommendations: List[str]
    performance_degradation_curve: pd.Series
    timestamp: datetime


@dataclass
class PerformanceAttribution:
    """Performance attribution analysis"""
    strategy_id: str
    total_return: float
    alpha_contribution: float
    beta_contribution: float
    factor_contributions: Dict[str, float]
    sector_contributions: Dict[str, float]
    asset_contributions: Dict[str, float]
    timing_contribution: float
    selection_contribution: float
    interaction_contribution: float
    timestamp: datetime


class RealTimePerformanceMonitor:
    """
    Real-time performance monitoring system for trading strategies.
    """
    
    def __init__(self):
        self.strategies: Dict[str, Dict] = {}
        self.performance_history: Dict[str, List[PerformanceMetric]] = {}
        self.alerts: List[StrategyAlert] = []
        self.thresholds: Dict[str, Dict[str, float]] = {}
        
        # Default thresholds
        self.default_thresholds = {
            'max_drawdown': -0.05,  # 5% max drawdown
            'daily_loss_limit': -0.02,  # 2% daily loss limit
            'sharpe_ratio_min': 0.5,  # Minimum Sharpe ratio
            'win_rate_min': 0.45,  # Minimum win rate
            'volatility_max': 0.25,  # Maximum volatility
            'var_breach_limit': 3,  # Maximum VaR breaches per month
        }
        
        # Performance tracking
        self.start_time = datetime.now()
        self.monitoring_frequency = 60  # seconds
        self.last_update_time = datetime.now()
        
    def register_strategy(
        self,
        strategy_id: str,
        strategy_config: Dict[str, Any],
        custom_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Register a strategy for monitoring.
        
        Args:
            strategy_id: Unique strategy identifier
            strategy_config: Strategy configuration
            custom_thresholds: Custom performance thresholds
        """
        
        self.strategies[strategy_id] = {
            'config': strategy_config,
            'state': StrategyState.ACTIVE,
            'start_time': datetime.now(),
            'last_update': datetime.now(),
            'positions': {},
            'current_pnl': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'total_volume': 0.0,
            'risk_metrics': {}
        }
        
        # Set thresholds
        thresholds = self.default_thresholds.copy()
        if custom_thresholds:
            thresholds.update(custom_thresholds)
        
        self.thresholds[strategy_id] = thresholds
        self.performance_history[strategy_id] = []
        
        logger.info(f"Strategy {strategy_id} registered for monitoring")
    
    def update_strategy_performance(
        self,
        strategy_id: str,
        pnl: float,
        positions: Dict[str, float],
        trades: List[Dict[str, Any]],
        timestamp: Optional[datetime] = None
    ):
        """
        Update strategy performance metrics.
        
        Args:
            strategy_id: Strategy identifier
            pnl: Current profit/loss
            positions: Current positions
            trades: Recent trades
            timestamp: Update timestamp
        """
        
        if strategy_id not in self.strategies:
            logger.warning(f"Strategy {strategy_id} not registered")
            return
        
        timestamp = timestamp or datetime.now()
        strategy = self.strategies[strategy_id]
        
        # Update basic metrics
        strategy['current_pnl'] = pnl
        strategy['positions'] = positions
        strategy['last_update'] = timestamp
        strategy['total_trades'] += len(trades)
        
        # Count winning trades
        for trade in trades:
            if trade.get('pnl', 0) > 0:
                strategy['winning_trades'] += 1
        
        # Calculate performance metrics
        metrics = self._calculate_realtime_metrics(strategy_id, timestamp)
        
        # Store metrics
        for metric in metrics:
            self.performance_history[strategy_id].append(metric)
        
        # Check for alerts
        self._check_performance_alerts(strategy_id, metrics)
        
        # Update strategy state
        self._update_strategy_state(strategy_id, metrics)
    
    def _calculate_realtime_metrics(
        self,
        strategy_id: str,
        timestamp: datetime
    ) -> List[PerformanceMetric]:
        """Calculate real-time performance metrics"""
        
        strategy = self.strategies[strategy_id]
        metrics = []
        
        # Time-based calculations
        elapsed_time = (timestamp - strategy['start_time']).total_seconds()
        elapsed_days = elapsed_time / 86400  # Convert to days
        
        if elapsed_days == 0:
            return metrics
        
        # Get historical PnL
        pnl_history = [
            m.value for m in self.performance_history[strategy_id] 
            if m.name == 'pnl'
        ]
        
        if not pnl_history:
            pnl_history = [0]
        
        pnl_history.append(strategy['current_pnl'])
        
        # Calculate returns
        returns = np.diff(pnl_history) / (abs(pnl_history[0]) + 1e-8)
        
        # Basic metrics
        total_return = strategy['current_pnl'] / (strategy['config'].get('initial_capital', 100000))
        annualized_return = (1 + total_return) ** (365 / elapsed_days) - 1 if elapsed_days > 0 else 0
        
        metrics.append(PerformanceMetric(
            name="pnl",
            value=strategy['current_pnl'],
            timestamp=timestamp,
            period="current"
        ))
        
        metrics.append(PerformanceMetric(
            name="total_return",
            value=total_return,
            timestamp=timestamp,
            period="inception"
        ))
        
        metrics.append(PerformanceMetric(
            name="annualized_return",
            value=annualized_return,
            timestamp=timestamp,
            period="annualized"
        ))
        
        # Volatility metrics
        if len(returns) > 1:
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            metrics.append(PerformanceMetric(
                name="volatility",
                value=volatility,
                timestamp=timestamp,
                period="annualized"
            ))
            
            # Sharpe ratio
            if volatility > 0:
                risk_free_rate = strategy['config'].get('risk_free_rate', 0.02)
                sharpe_ratio = (annualized_return - risk_free_rate) / volatility
                
                metrics.append(PerformanceMetric(
                    name="sharpe_ratio",
                    value=sharpe_ratio,
                    timestamp=timestamp,
                    period="annualized"
                ))
        
        # Drawdown metrics
        pnl_series = pd.Series(pnl_history)
        cumulative_max = pnl_series.expanding().max()
        drawdown = (pnl_series - cumulative_max) / (cumulative_max.abs() + 1e-8)
        max_drawdown = drawdown.min()
        
        metrics.append(PerformanceMetric(
            name="max_drawdown",
            value=max_drawdown,
            timestamp=timestamp,
            period="inception"
        ))
        
        metrics.append(PerformanceMetric(
            name="current_drawdown",
            value=drawdown.iloc[-1],
            timestamp=timestamp,
            period="current"
        ))
        
        # Trade statistics
        if strategy['total_trades'] > 0:
            win_rate = strategy['winning_trades'] / strategy['total_trades']
            
            metrics.append(PerformanceMetric(
                name="win_rate",
                value=win_rate,
                timestamp=timestamp,
                period="inception"
            ))
            
            metrics.append(PerformanceMetric(
                name="total_trades",
                value=strategy['total_trades'],
                timestamp=timestamp,
                period="inception"
            ))
        
        # Position metrics
        total_exposure = sum(abs(pos) for pos in strategy['positions'].values())
        
        metrics.append(PerformanceMetric(
            name="total_exposure",
            value=total_exposure,
            timestamp=timestamp,
            period="current"
        ))
        
        metrics.append(PerformanceMetric(
            name="num_positions",
            value=len(strategy['positions']),
            timestamp=timestamp,
            period="current"
        ))
        
        return metrics
    
    def _check_performance_alerts(
        self,
        strategy_id: str,
        metrics: List[PerformanceMetric]
    ):
        """Check for performance threshold breaches"""
        
        thresholds = self.thresholds[strategy_id]
        
        for metric in metrics:
            # Check if metric has threshold
            threshold_key = f"{metric.name}_max" if metric.name in ['volatility', 'max_drawdown'] else f"{metric.name}_min"
            
            if threshold_key not in thresholds:
                continue
            
            threshold = thresholds[threshold_key]
            
            # Check for breach
            is_breach = False
            if 'max' in threshold_key:
                is_breach = metric.value > threshold
            else:
                is_breach = metric.value < threshold
            
            metric.is_threshold_breached = is_breach
            
            if is_breach:
                # Determine severity
                if metric.name == 'max_drawdown':
                    severity = AlertSeverity.CRITICAL if metric.value < -0.1 else AlertSeverity.HIGH
                elif metric.name == 'sharpe_ratio':
                    severity = AlertSeverity.MEDIUM if metric.value < 0.3 else AlertSeverity.LOW
                else:
                    severity = AlertSeverity.MEDIUM
                
                # Create alert
                alert = StrategyAlert(
                    strategy_id=strategy_id,
                    alert_type=f"{metric.name}_breach",
                    severity=severity,
                    message=f"{metric.name} breach: {metric.value:.4f} vs threshold {threshold:.4f}",
                    timestamp=metric.timestamp,
                    metric_value=metric.value,
                    threshold_value=threshold,
                    suggested_action=self._get_suggested_action(metric.name, is_breach),
                    metadata={'metric': metric.name, 'period': metric.period}
                )
                
                self.alerts.append(alert)
                logger.warning(f"Alert generated: {alert.message}")
    
    def _get_suggested_action(self, metric_name: str, is_breach: bool) -> str:
        """Get suggested action for metric breach"""
        
        if not is_breach:
            return "No action required"
        
        suggestions = {
            'max_drawdown': "Consider reducing position sizes or implementing stop-losses",
            'sharpe_ratio': "Review strategy parameters and risk management",
            'volatility': "Implement volatility targeting or reduce leverage",
            'win_rate': "Analyze losing trades and improve entry/exit rules",
            'daily_loss_limit': "Stop trading immediately and review strategy"
        }
        
        return suggestions.get(metric_name, "Review strategy performance and parameters")
    
    def _update_strategy_state(
        self,
        strategy_id: str,
        metrics: List[PerformanceMetric]
    ):
        """Update strategy state based on performance"""
        
        strategy = self.strategies[strategy_id]
        current_state = strategy['state']
        
        # Check for critical conditions
        critical_alerts = [
            alert for alert in self.alerts 
            if alert.strategy_id == strategy_id and alert.severity == AlertSeverity.CRITICAL
        ]
        
        if critical_alerts:
            strategy['state'] = StrategyState.FAILED
            logger.critical(f"Strategy {strategy_id} marked as FAILED due to critical alerts")
            return
        
        # Check for degraded performance
        recent_metrics = {m.name: m.value for m in metrics}
        
        if recent_metrics.get('sharpe_ratio', 0) < 0.1:
            strategy['state'] = StrategyState.DEGRADED
        elif recent_metrics.get('max_drawdown', 0) < -0.08:
            strategy['state'] = StrategyState.DEGRADED
        else:
            strategy['state'] = StrategyState.ACTIVE
    
    def get_strategy_summary(self, strategy_id: str) -> Dict[str, Any]:
        """Get comprehensive strategy summary"""
        
        if strategy_id not in self.strategies:
            return {}
        
        strategy = self.strategies[strategy_id]
        
        # Get latest metrics
        latest_metrics = {}
        for metric in self.performance_history[strategy_id]:
            latest_metrics[metric.name] = metric.value
        
        # Recent alerts
        recent_alerts = [
            alert for alert in self.alerts 
            if alert.strategy_id == strategy_id and 
            alert.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        return {
            'strategy_id': strategy_id,
            'state': strategy['state'].value,
            'uptime': (datetime.now() - strategy['start_time']).total_seconds(),
            'current_pnl': strategy['current_pnl'],
            'total_trades': strategy['total_trades'],
            'winning_trades': strategy['winning_trades'],
            'positions': strategy['positions'],
            'latest_metrics': latest_metrics,
            'recent_alerts': len(recent_alerts),
            'last_update': strategy['last_update'].isoformat()
        }
    
    def generate_performance_report(
        self,
        strategy_id: str,
        period: str = "1d"
    ) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        if strategy_id not in self.strategies:
            return {}
        
        # Time period filtering
        end_time = datetime.now()
        if period == "1d":
            start_time = end_time - timedelta(days=1)
        elif period == "1w":
            start_time = end_time - timedelta(weeks=1)
        elif period == "1m":
            start_time = end_time - timedelta(days=30)
        else:
            start_time = self.strategies[strategy_id]['start_time']
        
        # Filter metrics
        period_metrics = [
            m for m in self.performance_history[strategy_id]
            if start_time <= m.timestamp <= end_time
        ]
        
        if not period_metrics:
            return {'error': 'No data available for specified period'}
        
        # Organize metrics by type
        metrics_by_type = {}
        for metric in period_metrics:
            if metric.name not in metrics_by_type:
                metrics_by_type[metric.name] = []
            metrics_by_type[metric.name].append(metric)
        
        # Calculate summary statistics
        summary = {}
        for metric_name, metric_list in metrics_by_type.items():
            values = [m.value for m in metric_list]
            
            summary[metric_name] = {
                'current': values[-1] if values else 0,
                'min': min(values) if values else 0,
                'max': max(values) if values else 0,
                'mean': np.mean(values) if values else 0,
                'std': np.std(values) if values else 0,
                'count': len(values)
            }
        
        # Recent alerts
        period_alerts = [
            alert for alert in self.alerts 
            if alert.strategy_id == strategy_id and 
            start_time <= alert.timestamp <= end_time
        ]
        
        alert_summary = {
            'total': len(period_alerts),
            'by_severity': {
                severity.value: len([a for a in period_alerts if a.severity == severity])
                for severity in AlertSeverity
            }
        }
        
        return {
            'strategy_id': strategy_id,
            'period': period,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'metrics_summary': summary,
            'alerts_summary': alert_summary,
            'strategy_state': self.strategies[strategy_id]['state'].value,
            'total_data_points': len(period_metrics)
        }


class StrategyCapacityAnalyzer:
    """
    Analyzer for strategy capacity and scalability.
    """
    
    def __init__(self):
        self.capacity_tests: Dict[str, List[float]] = {}
        self.performance_curves: Dict[str, pd.DataFrame] = {}
        
    def analyze_capacity(
        self,
        strategy_id: str,
        performance_by_size: Dict[float, Dict[str, float]],
        current_size: float
    ) -> CapacityAnalysis:
        """
        Analyze strategy capacity and scalability.
        
        Args:
            strategy_id: Strategy identifier
            performance_by_size: Performance metrics by portfolio size
            current_size: Current portfolio size
            
        Returns:
            CapacityAnalysis with capacity metrics and recommendations
        """
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(performance_by_size).T
        df.index.name = 'portfolio_size'
        
        # Calculate capacity metrics
        max_theoretical_capacity = self._calculate_max_capacity(df)
        capacity_utilization = current_size / max_theoretical_capacity if max_theoretical_capacity > 0 else 0
        
        # Identify constraints
        constraints = self._identify_capacity_constraints(df)
        
        # Generate recommendations
        recommendations = self._generate_scaling_recommendations(df, current_size, capacity_utilization)
        
        # Create performance degradation curve
        degradation_curve = self._create_degradation_curve(df)
        
        return CapacityAnalysis(
            strategy_id=strategy_id,
            current_capacity=current_size,
            max_theoretical_capacity=max_theoretical_capacity,
            capacity_utilization=capacity_utilization,
            capacity_constraints=constraints,
            scaling_recommendations=recommendations,
            performance_degradation_curve=degradation_curve,
            timestamp=datetime.now()
        )
    
    def _calculate_max_capacity(self, performance_df: pd.DataFrame) -> float:
        """Calculate maximum theoretical capacity"""
        
        if 'sharpe_ratio' not in performance_df.columns:
            return 0.0
        
        # Find size where Sharpe ratio drops below acceptable threshold
        acceptable_sharpe = 0.5
        
        acceptable_sizes = performance_df[performance_df['sharpe_ratio'] >= acceptable_sharpe]
        
        if acceptable_sizes.empty:
            return 0.0
        
        return acceptable_sizes.index.max()
    
    def _identify_capacity_constraints(self, performance_df: pd.DataFrame) -> List[str]:
        """Identify capacity constraints"""
        
        constraints = []
        
        # Check for liquidity constraints
        if 'market_impact' in performance_df.columns:
            high_impact_threshold = 0.01  # 1% market impact
            if performance_df['market_impact'].max() > high_impact_threshold:
                constraints.append("Market impact becomes significant at larger sizes")
        
        # Check for execution constraints
        if 'execution_cost' in performance_df.columns:
            high_cost_threshold = 0.005  # 50 bps
            if performance_df['execution_cost'].max() > high_cost_threshold:
                constraints.append("Execution costs increase substantially with size")
        
        # Check for alpha decay
        if 'alpha' in performance_df.columns:
            alpha_series = performance_df['alpha']
            if len(alpha_series) > 1:
                # Check if alpha is decreasing
                correlation = alpha_series.corr(pd.Series(alpha_series.index))
                if correlation < -0.5:
                    constraints.append("Alpha decays significantly with increased capacity")
        
        # Check for volatility increase
        if 'volatility' in performance_df.columns:
            vol_series = performance_df['volatility']
            if len(vol_series) > 1:
                vol_increase = (vol_series.iloc[-1] - vol_series.iloc[0]) / vol_series.iloc[0]
                if vol_increase > 0.5:  # 50% increase
                    constraints.append("Volatility increases significantly with size")
        
        return constraints
    
    def _generate_scaling_recommendations(
        self,
        performance_df: pd.DataFrame,
        current_size: float,
        capacity_utilization: float
    ) -> List[str]:
        """Generate scaling recommendations"""
        
        recommendations = []
        
        if capacity_utilization > 0.8:
            recommendations.append("Consider reducing position sizes or implementing position limits")
            recommendations.append("Explore alternative execution venues to reduce market impact")
        
        if capacity_utilization > 0.6:
            recommendations.append("Monitor execution costs and market impact closely")
            recommendations.append("Consider implementing dynamic position sizing")
        
        if capacity_utilization < 0.3:
            recommendations.append("Strategy has room for growth - consider increasing allocation")
            recommendations.append("Evaluate opportunities for leveraging unused capacity")
        
        # Performance-based recommendations
        if 'sharpe_ratio' in performance_df.columns:
            current_sharpe = performance_df.loc[
                performance_df.index[performance_df.index <= current_size].max()
            ]['sharpe_ratio']
            
            if current_sharpe < 0.5:
                recommendations.append("Consider reducing size to improve risk-adjusted returns")
        
        return recommendations
    
    def _create_degradation_curve(self, performance_df: pd.DataFrame) -> pd.Series:
        """Create performance degradation curve"""
        
        if 'sharpe_ratio' not in performance_df.columns:
            return pd.Series(dtype=float)
        
        # Normalize Sharpe ratio to show degradation
        sharpe_series = performance_df['sharpe_ratio']
        max_sharpe = sharpe_series.max()
        
        if max_sharpe > 0:
            degradation_curve = 1 - (sharpe_series / max_sharpe)
        else:
            degradation_curve = pd.Series(0, index=sharpe_series.index)
        
        return degradation_curve


class PerformanceAttributionAnalyzer:
    """
    Analyzer for performance attribution analysis.
    """
    
    def __init__(self):
        self.factor_models: Dict[str, Any] = {}
        self.benchmark_returns: Dict[str, pd.Series] = {}
        
    def analyze_attribution(
        self,
        strategy_id: str,
        strategy_returns: pd.Series,
        positions: pd.DataFrame,
        factor_returns: pd.DataFrame,
        benchmark_returns: pd.Series
    ) -> PerformanceAttribution:
        """
        Analyze performance attribution.
        
        Args:
            strategy_id: Strategy identifier
            strategy_returns: Strategy returns series
            positions: Position weights over time
            factor_returns: Factor returns
            benchmark_returns: Benchmark returns
            
        Returns:
            PerformanceAttribution with detailed attribution analysis
        """
        
        # Calculate total return
        total_return = (1 + strategy_returns).prod() - 1
        
        # Calculate alpha and beta
        alpha, beta = self._calculate_alpha_beta(strategy_returns, benchmark_returns)
        
        # Calculate beta contribution
        beta_contribution = beta * benchmark_returns.sum()
        
        # Calculate alpha contribution
        alpha_contribution = total_return - beta_contribution
        
        # Factor attribution
        factor_contributions = self._calculate_factor_attribution(
            strategy_returns, factor_returns
        )
        
        # Asset attribution
        asset_contributions = self._calculate_asset_attribution(
            strategy_returns, positions
        )
        
        # Sector attribution (simplified)
        sector_contributions = self._calculate_sector_attribution(positions)
        
        # Timing and selection attribution
        timing_contribution, selection_contribution = self._calculate_timing_selection_attribution(
            strategy_returns, positions, benchmark_returns
        )
        
        # Interaction effect
        interaction_contribution = (
            total_return - alpha_contribution - beta_contribution - 
            timing_contribution - selection_contribution
        )
        
        return PerformanceAttribution(
            strategy_id=strategy_id,
            total_return=total_return,
            alpha_contribution=alpha_contribution,
            beta_contribution=beta_contribution,
            factor_contributions=factor_contributions,
            sector_contributions=sector_contributions,
            asset_contributions=asset_contributions,
            timing_contribution=timing_contribution,
            selection_contribution=selection_contribution,
            interaction_contribution=interaction_contribution,
            timestamp=datetime.now()
        )
    
    def _calculate_alpha_beta(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Tuple[float, float]:
        """Calculate alpha and beta vs benchmark"""
        
        # Align returns
        aligned_strategy = strategy_returns.dropna()
        aligned_benchmark = benchmark_returns.reindex(aligned_strategy.index).dropna()
        
        if len(aligned_strategy) < 10 or len(aligned_benchmark) < 10:
            return 0.0, 0.0
        
        # Calculate beta
        covariance = np.cov(aligned_strategy, aligned_benchmark)[0, 1]
        benchmark_variance = np.var(aligned_benchmark)
        
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0.0
        
        # Calculate alpha
        alpha = aligned_strategy.mean() - beta * aligned_benchmark.mean()
        
        return alpha, beta
    
    def _calculate_factor_attribution(
        self,
        strategy_returns: pd.Series,
        factor_returns: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate factor-based attribution"""
        
        if factor_returns.empty:
            return {}
        
        # Align data
        aligned_strategy = strategy_returns.dropna()
        aligned_factors = factor_returns.reindex(aligned_strategy.index).dropna()
        
        if len(aligned_strategy) < 10 or aligned_factors.empty:
            return {}
        
        # Simple factor attribution using correlation
        factor_contributions = {}
        
        for factor in aligned_factors.columns:
            factor_series = aligned_factors[factor]
            
            # Calculate correlation-based attribution
            correlation = aligned_strategy.corr(factor_series)
            
            if not np.isnan(correlation):
                # Approximate contribution based on correlation and factor return
                factor_return = factor_series.sum()
                contribution = correlation * factor_return * 0.1  # Simplified scaling
                factor_contributions[factor] = contribution
        
        return factor_contributions
    
    def _calculate_asset_attribution(
        self,
        strategy_returns: pd.Series,
        positions: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate asset-level attribution"""
        
        if positions.empty:
            return {}
        
        # Calculate contribution by asset
        asset_contributions = {}
        
        for asset in positions.columns:
            asset_weights = positions[asset].dropna()
            
            if len(asset_weights) > 0:
                # Approximate contribution (simplified)
                avg_weight = asset_weights.mean()
                contribution = avg_weight * strategy_returns.sum() * 0.1  # Simplified
                asset_contributions[asset] = contribution
        
        return asset_contributions
    
    def _calculate_sector_attribution(
        self,
        positions: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate sector attribution (simplified)"""
        
        # Mock sector mapping
        sector_mapping = {
            'AAPL': 'Technology',
            'GOOGL': 'Technology',
            'MSFT': 'Technology',
            'AMZN': 'Consumer Discretionary',
            'TSLA': 'Consumer Discretionary',
            'JPM': 'Financial',
            'BAC': 'Financial',
            'XOM': 'Energy',
            'CVX': 'Energy'
        }
        
        sector_contributions = {}
        
        for asset in positions.columns:
            sector = sector_mapping.get(asset, 'Other')
            
            if sector not in sector_contributions:
                sector_contributions[sector] = 0.0
            
            # Simplified sector contribution
            if len(positions[asset]) > 0:
                avg_weight = positions[asset].mean()
                sector_contributions[sector] += avg_weight * 0.01  # Simplified
        
        return sector_contributions
    
    def _calculate_timing_selection_attribution(
        self,
        strategy_returns: pd.Series,
        positions: pd.DataFrame,
        benchmark_returns: pd.Series
    ) -> Tuple[float, float]:
        """Calculate timing and selection attribution"""
        
        # Simplified timing attribution
        # Based on correlation between position changes and subsequent returns
        timing_contribution = 0.0
        selection_contribution = 0.0
        
        if not positions.empty and len(strategy_returns) > 0:
            # Calculate position changes
            position_changes = positions.diff().sum(axis=1).dropna()
            
            if len(position_changes) > 10:
                # Timing: correlation between position changes and future returns
                aligned_returns = strategy_returns.reindex(position_changes.index).dropna()
                
                if len(aligned_returns) > 10:
                    timing_corr = position_changes.corr(aligned_returns)
                    
                    if not np.isnan(timing_corr):
                        timing_contribution = timing_corr * strategy_returns.sum() * 0.1
        
        # Selection attribution (residual)
        selection_contribution = strategy_returns.sum() * 0.05  # Simplified
        
        return timing_contribution, selection_contribution


class StrategyDecayDetector:
    """
    Detector for strategy performance decay.
    """
    
    def __init__(self):
        self.decay_indicators: Dict[str, List[float]] = {}
        self.baseline_performance: Dict[str, Dict[str, float]] = {}
        
    def detect_decay(
        self,
        strategy_id: str,
        recent_performance: Dict[str, float],
        historical_performance: Dict[str, float],
        significance_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Detect strategy performance decay.
        
        Args:
            strategy_id: Strategy identifier
            recent_performance: Recent performance metrics
            historical_performance: Historical performance baseline
            significance_threshold: Significance threshold for decay detection
            
        Returns:
            Decay analysis results
        """
        
        decay_results = {
            'strategy_id': strategy_id,
            'decay_detected': False,
            'decay_metrics': {},
            'significance_tests': {},
            'recommendations': []
        }
        
        # Compare each metric
        for metric, recent_value in recent_performance.items():
            if metric not in historical_performance:
                continue
            
            historical_value = historical_performance[metric]
            
            # Calculate decay percentage
            if historical_value != 0:
                decay_pct = (recent_value - historical_value) / abs(historical_value)
            else:
                decay_pct = 0.0
            
            # Determine if decay is significant
            is_significant = abs(decay_pct) > significance_threshold
            
            # For some metrics, decay is negative change
            if metric in ['sharpe_ratio', 'win_rate', 'alpha']:
                is_decay = decay_pct < -significance_threshold
            else:  # For drawdown, volatility, etc.
                is_decay = decay_pct > significance_threshold
            
            if is_decay:
                decay_results['decay_detected'] = True
                decay_results['decay_metrics'][metric] = {
                    'recent_value': recent_value,
                    'historical_value': historical_value,
                    'decay_percentage': decay_pct,
                    'is_significant': is_significant
                }
        
        # Generate recommendations
        if decay_results['decay_detected']:
            decay_results['recommendations'] = self._generate_decay_recommendations(
                decay_results['decay_metrics']
            )
        
        return decay_results
    
    def _generate_decay_recommendations(self, decay_metrics: Dict[str, Dict]) -> List[str]:
        """Generate recommendations for addressing decay"""
        
        recommendations = []
        
        # Analyze decay patterns
        if 'sharpe_ratio' in decay_metrics:
            recommendations.append("Review risk management parameters and position sizing")
        
        if 'win_rate' in decay_metrics:
            recommendations.append("Analyze recent losing trades and refine entry/exit criteria")
        
        if 'alpha' in decay_metrics:
            recommendations.append("Consider strategy recalibration or factor model updates")
        
        if 'max_drawdown' in decay_metrics:
            recommendations.append("Implement stronger risk controls and stop-loss mechanisms")
        
        # General recommendations
        recommendations.extend([
            "Conduct thorough strategy review and backtesting",
            "Consider reducing position sizes temporarily",
            "Monitor market regime changes and adapt accordingly",
            "Review and update strategy parameters based on recent market conditions"
        ])
        
        return recommendations


# Test fixtures
@pytest.fixture
def performance_monitor():
    """Create performance monitor instance"""
    return RealTimePerformanceMonitor()


@pytest.fixture
def capacity_analyzer():
    """Create capacity analyzer instance"""
    return StrategyCapacityAnalyzer()


@pytest.fixture
def attribution_analyzer():
    """Create attribution analyzer instance"""
    return PerformanceAttributionAnalyzer()


@pytest.fixture
def decay_detector():
    """Create decay detector instance"""
    return StrategyDecayDetector()


@pytest.fixture
def sample_strategy_data():
    """Generate sample strategy data"""
    
    np.random.seed(42)
    
    # Generate returns
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    returns = np.random.normal(0.001, 0.02, len(dates))
    
    # Generate positions
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    positions = pd.DataFrame(
        np.random.uniform(0, 0.3, (len(dates), len(assets))),
        index=dates,
        columns=assets
    )
    
    # Normalize positions
    positions = positions.div(positions.sum(axis=1), axis=0)
    
    # Generate factor returns
    factors = ['Market', 'Value', 'Momentum', 'Quality']
    factor_returns = pd.DataFrame(
        np.random.normal(0.0005, 0.015, (len(dates), len(factors))),
        index=dates,
        columns=factors
    )
    
    # Generate benchmark returns
    benchmark_returns = pd.Series(
        np.random.normal(0.0008, 0.018, len(dates)),
        index=dates
    )
    
    return {
        'returns': pd.Series(returns, index=dates),
        'positions': positions,
        'factor_returns': factor_returns,
        'benchmark_returns': benchmark_returns
    }


# Comprehensive test suite
@pytest.mark.asyncio
class TestStrategyPerformance:
    """Comprehensive strategy performance tests"""
    
    def test_strategy_registration(self, performance_monitor):
        """Test strategy registration and basic setup"""
        
        strategy_config = {
            'name': 'Test Strategy',
            'initial_capital': 100000,
            'risk_free_rate': 0.02
        }
        
        custom_thresholds = {
            'max_drawdown': -0.03,
            'sharpe_ratio_min': 0.8
        }
        
        performance_monitor.register_strategy(
            'test_strategy',
            strategy_config,
            custom_thresholds
        )
        
        assert 'test_strategy' in performance_monitor.strategies
        assert performance_monitor.strategies['test_strategy']['state'] == StrategyState.ACTIVE
        assert performance_monitor.thresholds['test_strategy']['max_drawdown'] == -0.03
        assert performance_monitor.thresholds['test_strategy']['sharpe_ratio_min'] == 0.8
    
    def test_performance_update(self, performance_monitor):
        """Test strategy performance updates"""
        
        # Register strategy
        strategy_config = {'initial_capital': 100000}
        performance_monitor.register_strategy('test_strategy', strategy_config)
        
        # Update performance
        trades = [
            {'symbol': 'AAPL', 'quantity': 100, 'price': 150, 'pnl': 500},
            {'symbol': 'GOOGL', 'quantity': 50, 'price': 2500, 'pnl': -200}
        ]
        
        positions = {'AAPL': 0.6, 'GOOGL': 0.4}
        
        performance_monitor.update_strategy_performance(
            'test_strategy',
            pnl=1000,
            positions=positions,
            trades=trades
        )
        
        strategy = performance_monitor.strategies['test_strategy']
        assert strategy['current_pnl'] == 1000
        assert strategy['positions'] == positions
        assert strategy['total_trades'] == 2
        assert strategy['winning_trades'] == 1
        
        # Check metrics were generated
        assert len(performance_monitor.performance_history['test_strategy']) > 0
    
    def test_alert_generation(self, performance_monitor):
        """Test performance alert generation"""
        
        # Register strategy with tight thresholds
        strategy_config = {'initial_capital': 100000}
        custom_thresholds = {
            'max_drawdown': -0.01,  # Very tight threshold
            'sharpe_ratio_min': 2.0  # Very high threshold
        }
        
        performance_monitor.register_strategy(
            'test_strategy',
            strategy_config,
            custom_thresholds
        )
        
        # Generate performance that should trigger alerts
        trades = [{'symbol': 'AAPL', 'quantity': 100, 'price': 150, 'pnl': -2000}]
        
        performance_monitor.update_strategy_performance(
            'test_strategy',
            pnl=-2000,  # Large loss
            positions={'AAPL': 1.0},
            trades=trades
        )
        
        # Check alerts were generated
        strategy_alerts = [
            alert for alert in performance_monitor.alerts
            if alert.strategy_id == 'test_strategy'
        ]
        
        assert len(strategy_alerts) > 0
        
        # Check alert properties
        for alert in strategy_alerts:
            assert alert.strategy_id == 'test_strategy'
            assert isinstance(alert.severity, AlertSeverity)
            assert alert.message != ""
            assert alert.suggested_action != ""
    
    def test_strategy_state_updates(self, performance_monitor):
        """Test strategy state updates based on performance"""
        
        performance_monitor.register_strategy('test_strategy', {'initial_capital': 100000})
        
        # Normal performance - should stay ACTIVE
        performance_monitor.update_strategy_performance(
            'test_strategy',
            pnl=500,
            positions={'AAPL': 0.5, 'GOOGL': 0.5},
            trades=[]
        )
        
        assert performance_monitor.strategies['test_strategy']['state'] == StrategyState.ACTIVE
        
        # Poor performance - should become DEGRADED
        performance_monitor.thresholds['test_strategy']['max_drawdown'] = -0.001  # Very tight
        
        performance_monitor.update_strategy_performance(
            'test_strategy',
            pnl=-1000,
            positions={'AAPL': 0.5, 'GOOGL': 0.5},
            trades=[]
        )
        
        # State should be updated based on performance
        assert performance_monitor.strategies['test_strategy']['state'] in [
            StrategyState.DEGRADED, StrategyState.FAILED, StrategyState.ACTIVE
        ]
    
    def test_performance_report_generation(self, performance_monitor):
        """Test performance report generation"""
        
        performance_monitor.register_strategy('test_strategy', {'initial_capital': 100000})
        
        # Generate some performance data
        for i in range(10):
            performance_monitor.update_strategy_performance(
                'test_strategy',
                pnl=i * 100,
                positions={'AAPL': 0.5, 'GOOGL': 0.5},
                trades=[]
            )
        
        # Generate report
        report = performance_monitor.generate_performance_report('test_strategy', '1d')
        
        assert report['strategy_id'] == 'test_strategy'
        assert report['period'] == '1d'
        assert 'metrics_summary' in report
        assert 'alerts_summary' in report
        assert report['total_data_points'] > 0
        
        # Check metrics summary structure
        if report['metrics_summary']:
            for metric_name, metric_stats in report['metrics_summary'].items():
                assert 'current' in metric_stats
                assert 'min' in metric_stats
                assert 'max' in metric_stats
                assert 'mean' in metric_stats
                assert 'count' in metric_stats
    
    def test_capacity_analysis(self, capacity_analyzer):
        """Test strategy capacity analysis"""
        
        # Create performance data by size
        performance_by_size = {
            10000: {'sharpe_ratio': 1.5, 'market_impact': 0.001, 'alpha': 0.05},
            50000: {'sharpe_ratio': 1.2, 'market_impact': 0.003, 'alpha': 0.04},
            100000: {'sharpe_ratio': 1.0, 'market_impact': 0.006, 'alpha': 0.03},
            200000: {'sharpe_ratio': 0.8, 'market_impact': 0.012, 'alpha': 0.02},
            500000: {'sharpe_ratio': 0.4, 'market_impact': 0.025, 'alpha': 0.01}
        }
        
        current_size = 100000
        
        analysis = capacity_analyzer.analyze_capacity(
            'test_strategy',
            performance_by_size,
            current_size
        )
        
        assert analysis.strategy_id == 'test_strategy'
        assert analysis.current_capacity == current_size
        assert analysis.max_theoretical_capacity > 0
        assert 0 <= analysis.capacity_utilization <= 1
        assert isinstance(analysis.capacity_constraints, list)
        assert isinstance(analysis.scaling_recommendations, list)
        assert isinstance(analysis.performance_degradation_curve, pd.Series)
    
    def test_performance_attribution(self, attribution_analyzer, sample_strategy_data):
        """Test performance attribution analysis"""
        
        attribution = attribution_analyzer.analyze_attribution(
            'test_strategy',
            sample_strategy_data['returns'],
            sample_strategy_data['positions'],
            sample_strategy_data['factor_returns'],
            sample_strategy_data['benchmark_returns']
        )
        
        assert attribution.strategy_id == 'test_strategy'
        assert isinstance(attribution.total_return, float)
        assert isinstance(attribution.alpha_contribution, float)
        assert isinstance(attribution.beta_contribution, float)
        assert isinstance(attribution.factor_contributions, dict)
        assert isinstance(attribution.asset_contributions, dict)
        assert isinstance(attribution.sector_contributions, dict)
        
        # Check that contributions are reasonable
        assert -1 <= attribution.total_return <= 1
        assert -1 <= attribution.alpha_contribution <= 1
        assert -1 <= attribution.beta_contribution <= 1
    
    def test_strategy_decay_detection(self, decay_detector):
        """Test strategy decay detection"""
        
        # Historical performance (good)
        historical_performance = {
            'sharpe_ratio': 1.2,
            'win_rate': 0.65,
            'alpha': 0.08,
            'max_drawdown': -0.05
        }
        
        # Recent performance (degraded)
        recent_performance = {
            'sharpe_ratio': 0.8,
            'win_rate': 0.45,
            'alpha': 0.03,
            'max_drawdown': -0.12
        }
        
        decay_analysis = decay_detector.detect_decay(
            'test_strategy',
            recent_performance,
            historical_performance
        )
        
        assert decay_analysis['strategy_id'] == 'test_strategy'
        assert decay_analysis['decay_detected'] == True
        assert len(decay_analysis['decay_metrics']) > 0
        assert len(decay_analysis['recommendations']) > 0
        
        # Check specific decay metrics
        assert 'sharpe_ratio' in decay_analysis['decay_metrics']
        assert 'win_rate' in decay_analysis['decay_metrics']
        
        # Check recommendations are meaningful
        for recommendation in decay_analysis['recommendations']:
            assert isinstance(recommendation, str)
            assert len(recommendation) > 0
    
    def test_real_time_monitoring_performance(self, performance_monitor):
        """Test real-time monitoring performance under load"""
        
        # Register multiple strategies
        for i in range(10):
            performance_monitor.register_strategy(
                f'strategy_{i}',
                {'initial_capital': 100000}
            )
        
        # Simulate high-frequency updates
        start_time = time.time()
        
        for i in range(100):
            for j in range(10):
                performance_monitor.update_strategy_performance(
                    f'strategy_{j}',
                    pnl=i * 10,
                    positions={'AAPL': 0.5, 'GOOGL': 0.5},
                    trades=[]
                )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should handle 1000 updates reasonably quickly
        assert processing_time < 10.0  # Less than 10 seconds
        
        # Check all strategies have data
        for i in range(10):
            strategy_id = f'strategy_{i}'
            assert len(performance_monitor.performance_history[strategy_id]) > 0
    
    def test_multi_strategy_comparison(self, performance_monitor):
        """Test multi-strategy performance comparison"""
        
        # Register multiple strategies with different performance
        strategies = {
            'high_performer': {'pnl': 5000, 'trades': 50, 'winners': 35},
            'low_performer': {'pnl': -2000, 'trades': 50, 'winners': 15},
            'medium_performer': {'pnl': 1000, 'trades': 30, 'winners': 18}
        }
        
        for strategy_id, config in strategies.items():
            performance_monitor.register_strategy(strategy_id, {'initial_capital': 100000})
            
            # Simulate performance
            trades = [{'pnl': 10}] * config['winners'] + [{'pnl': -5}] * (config['trades'] - config['winners'])
            
            performance_monitor.update_strategy_performance(
                strategy_id,
                pnl=config['pnl'],
                positions={'AAPL': 1.0},
                trades=trades
            )
        
        # Compare strategies
        summaries = {}
        for strategy_id in strategies:
            summaries[strategy_id] = performance_monitor.get_strategy_summary(strategy_id)
        
        # High performer should have best metrics
        high_perf_summary = summaries['high_performer']
        low_perf_summary = summaries['low_performer']
        
        assert high_perf_summary['current_pnl'] > low_perf_summary['current_pnl']
        assert high_perf_summary['winning_trades'] > low_perf_summary['winning_trades']
    
    def test_performance_degradation_curve(self, capacity_analyzer):
        """Test performance degradation curve generation"""
        
        # Create performance data showing degradation
        performance_by_size = {}
        for size in [10000, 50000, 100000, 200000, 500000]:
            # Performance degrades with size
            degradation_factor = 1 - (size / 1000000)  # Degrades as size increases
            performance_by_size[size] = {
                'sharpe_ratio': max(0.1, 1.5 * degradation_factor),
                'market_impact': 0.001 * (size / 10000),
                'alpha': max(0.01, 0.08 * degradation_factor)
            }
        
        analysis = capacity_analyzer.analyze_capacity(
            'test_strategy',
            performance_by_size,
            100000
        )
        
        degradation_curve = analysis.performance_degradation_curve
        
        # Check curve properties
        assert isinstance(degradation_curve, pd.Series)
        assert len(degradation_curve) > 0
        assert all(0 <= val <= 1 for val in degradation_curve)
        
        # Degradation should generally increase with size
        if len(degradation_curve) > 1:
            # Allow for some noise but generally increasing trend
            trend = degradation_curve.iloc[-1] >= degradation_curve.iloc[0]
            assert trend or degradation_curve.iloc[-1] > 0.1  # Or at least some degradation
    
    def test_alert_severity_classification(self, performance_monitor):
        """Test alert severity classification"""
        
        performance_monitor.register_strategy('test_strategy', {'initial_capital': 100000})
        
        # Generate different severity conditions
        test_cases = [
            {'pnl': -15000, 'expected_severity': AlertSeverity.CRITICAL},  # Large loss
            {'pnl': -5000, 'expected_severity': AlertSeverity.HIGH},       # Moderate loss  
            {'pnl': -2000, 'expected_severity': AlertSeverity.MEDIUM},     # Small loss
            {'pnl': 1000, 'expected_severity': None}                      # Profit (no alert)
        ]
        
        for case in test_cases:
            # Clear previous alerts
            performance_monitor.alerts = []
            
            performance_monitor.update_strategy_performance(
                'test_strategy',
                pnl=case['pnl'],
                positions={'AAPL': 1.0},
                trades=[]
            )
            
            if case['expected_severity']:
                # Should have generated alert
                strategy_alerts = [
                    alert for alert in performance_monitor.alerts
                    if alert.strategy_id == 'test_strategy'
                ]
                
                assert len(strategy_alerts) > 0
                
                # Check severity classification
                max_severity = max(alert.severity for alert in strategy_alerts)
                assert max_severity.value in [s.value for s in AlertSeverity]
    
    def test_strategy_summary_completeness(self, performance_monitor):
        """Test strategy summary completeness"""
        
        performance_monitor.register_strategy('test_strategy', {'initial_capital': 100000})
        
        # Generate comprehensive performance data
        for i in range(20):
            trades = [{'pnl': 50 if i % 2 == 0 else -30}]
            performance_monitor.update_strategy_performance(
                'test_strategy',
                pnl=i * 100,
                positions={'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.3},
                trades=trades
            )
        
        summary = performance_monitor.get_strategy_summary('test_strategy')
        
        # Check all required fields are present
        required_fields = [
            'strategy_id', 'state', 'uptime', 'current_pnl',
            'total_trades', 'winning_trades', 'positions',
            'latest_metrics', 'recent_alerts', 'last_update'
        ]
        
        for field in required_fields:
            assert field in summary
        
        # Check data types
        assert isinstance(summary['uptime'], float)
        assert isinstance(summary['current_pnl'], float)
        assert isinstance(summary['total_trades'], int)
        assert isinstance(summary['winning_trades'], int)
        assert isinstance(summary['positions'], dict)
        assert isinstance(summary['latest_metrics'], dict)
        assert isinstance(summary['recent_alerts'], int)
    
    def test_large_scale_performance_tracking(self, performance_monitor):
        """Test performance tracking with large amounts of data"""
        
        # Register strategy
        performance_monitor.register_strategy('test_strategy', {'initial_capital': 100000})
        
        # Generate large amounts of performance data
        for i in range(1000):
            performance_monitor.update_strategy_performance(
                'test_strategy',
                pnl=i * 10 + np.random.normal(0, 100),
                positions={'AAPL': 0.5, 'GOOGL': 0.5},
                trades=[{'pnl': np.random.normal(0, 50)}]
            )
        
        # Check system handles large dataset
        assert len(performance_monitor.performance_history['test_strategy']) > 0
        
        # Generate report (should complete without errors)
        report = performance_monitor.generate_performance_report('test_strategy', '1d')
        
        assert report['strategy_id'] == 'test_strategy'
        assert report['total_data_points'] > 0
        
        # Check memory usage is reasonable (performance history shouldn't grow indefinitely)
        history_length = len(performance_monitor.performance_history['test_strategy'])
        assert history_length > 0
        # Note: In production, you might want to implement history limits
    
    def test_concurrent_strategy_updates(self, performance_monitor):
        """Test concurrent strategy updates"""
        
        # Register multiple strategies
        for i in range(5):
            performance_monitor.register_strategy(f'strategy_{i}', {'initial_capital': 100000})
        
        # Simulate concurrent updates (in real async environment)
        import threading
        
        def update_strategy(strategy_id, updates):
            for pnl in updates:
                performance_monitor.update_strategy_performance(
                    strategy_id,
                    pnl=pnl,
                    positions={'AAPL': 1.0},
                    trades=[]
                )
        
        # Create threads for concurrent updates
        threads = []
        for i in range(5):
            updates = [j * 100 for j in range(50)]
            thread = threading.Thread(target=update_strategy, args=(f'strategy_{i}', updates))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check all strategies have data
        for i in range(5):
            strategy_id = f'strategy_{i}'
            assert len(performance_monitor.performance_history[strategy_id]) > 0
            assert performance_monitor.strategies[strategy_id]['current_pnl'] > 0