"""
Portfolio Heat and Correlation-Based Risk Controls
==================================================

This module implements sophisticated portfolio heat monitoring and correlation-based
risk controls including:

- Portfolio heat calculation and monitoring
- Dynamic correlation tracking
- Risk concentration controls
- Correlation clustering and regime detection
- Real-time risk allocation monitoring
- Automated risk limit enforcement

Author: Risk Management System
Date: 2025-07-17
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import logging
from numba import jit, njit, prange
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class PortfolioHeatConfig:
    """Configuration for portfolio heat monitoring"""
    # Heat calculation parameters
    max_portfolio_heat: float = 0.15
    heat_warning_threshold: float = 0.12
    heat_rebalance_threshold: float = 0.10
    
    # Correlation parameters
    max_correlation: float = 0.7
    correlation_warning_threshold: float = 0.6
    correlation_lookback: int = 60
    correlation_decay_factor: float = 0.95
    
    # Concentration parameters
    max_concentration: float = 0.15
    max_sector_concentration: float = 0.25
    concentration_warning_threshold: float = 0.12
    
    # Clustering parameters
    n_clusters: int = 5
    cluster_method: str = "kmeans"
    cluster_update_frequency: int = 10
    
    # Risk allocation parameters
    risk_allocation_method: str = "equal_risk"
    target_risk_allocation: Dict[str, float] = field(default_factory=dict)
    rebalance_frequency: int = 5
    
    # Monitoring parameters
    monitoring_frequency: int = 1  # minutes
    alert_cooldown: int = 300  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'max_portfolio_heat': self.max_portfolio_heat,
            'heat_warning_threshold': self.heat_warning_threshold,
            'heat_rebalance_threshold': self.heat_rebalance_threshold,
            'max_correlation': self.max_correlation,
            'correlation_warning_threshold': self.correlation_warning_threshold,
            'correlation_lookback': self.correlation_lookback,
            'correlation_decay_factor': self.correlation_decay_factor,
            'max_concentration': self.max_concentration,
            'max_sector_concentration': self.max_sector_concentration,
            'concentration_warning_threshold': self.concentration_warning_threshold,
            'n_clusters': self.n_clusters,
            'cluster_method': self.cluster_method,
            'cluster_update_frequency': self.cluster_update_frequency,
            'risk_allocation_method': self.risk_allocation_method,
            'target_risk_allocation': self.target_risk_allocation,
            'rebalance_frequency': self.rebalance_frequency,
            'monitoring_frequency': self.monitoring_frequency,
            'alert_cooldown': self.alert_cooldown
        }


@dataclass
class PortfolioHeat:
    """Portfolio heat calculation result"""
    total_heat: float
    component_heat: Dict[str, float]
    correlation_heat: float
    concentration_heat: float
    sector_heat: Dict[str, float]
    risk_contribution: Dict[str, float]
    heat_breakdown: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_heat': self.total_heat,
            'component_heat': self.component_heat,
            'correlation_heat': self.correlation_heat,
            'concentration_heat': self.concentration_heat,
            'sector_heat': self.sector_heat,
            'risk_contribution': self.risk_contribution,
            'heat_breakdown': self.heat_breakdown
        }


@dataclass
class CorrelationAnalysis:
    """Correlation analysis result"""
    correlation_matrix: np.ndarray
    average_correlation: float
    max_correlation: float
    correlation_clusters: Dict[str, List[str]]
    correlation_regime: str
    regime_transition_probability: float
    rolling_correlations: Dict[str, np.ndarray]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'correlation_matrix': self.correlation_matrix.tolist(),
            'average_correlation': self.average_correlation,
            'max_correlation': self.max_correlation,
            'correlation_clusters': self.correlation_clusters,
            'correlation_regime': self.correlation_regime,
            'regime_transition_probability': self.regime_transition_probability,
            'rolling_correlations': {k: v.tolist() for k, v in self.rolling_correlations.items()}
        }


@dataclass
class RiskAllocation:
    """Risk allocation result"""
    current_allocation: Dict[str, float]
    target_allocation: Dict[str, float]
    allocation_drift: Dict[str, float]
    rebalance_needed: bool
    rebalance_trades: Dict[str, float]
    risk_budget_utilization: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'current_allocation': self.current_allocation,
            'target_allocation': self.target_allocation,
            'allocation_drift': self.allocation_drift,
            'rebalance_needed': self.rebalance_needed,
            'rebalance_trades': self.rebalance_trades,
            'risk_budget_utilization': self.risk_budget_utilization
        }


# Numba JIT optimized functions
@njit(cache=True, fastmath=True)
def calculate_portfolio_heat_jit(
    weights: np.ndarray,
    volatilities: np.ndarray,
    correlation_matrix: np.ndarray
) -> float:
    """
    Calculate portfolio heat - JIT optimized
    
    Args:
        weights: Portfolio weights
        volatilities: Asset volatilities
        correlation_matrix: Correlation matrix
    
    Returns:
        Portfolio heat (volatility)
    """
    n_assets = len(weights)
    
    # Calculate portfolio variance
    portfolio_variance = 0.0
    
    for i in range(n_assets):
        for j in range(n_assets):
            portfolio_variance += (weights[i] * weights[j] * 
                                 volatilities[i] * volatilities[j] * 
                                 correlation_matrix[i, j])
    
    # Portfolio heat is the square root (volatility)
    portfolio_heat = np.sqrt(max(0.0, portfolio_variance))
    
    return portfolio_heat


@njit(cache=True, fastmath=True)
def calculate_component_heat_jit(
    weights: np.ndarray,
    volatilities: np.ndarray,
    correlation_matrix: np.ndarray
) -> np.ndarray:
    """
    Calculate component heat contributions - JIT optimized
    
    Args:
        weights: Portfolio weights
        volatilities: Asset volatilities
        correlation_matrix: Correlation matrix
    
    Returns:
        Component heat contributions
    """
    n_assets = len(weights)
    component_heat = np.zeros(n_assets)
    
    # Calculate portfolio variance
    portfolio_variance = 0.0
    for i in range(n_assets):
        for j in range(n_assets):
            portfolio_variance += (weights[i] * weights[j] * 
                                 volatilities[i] * volatilities[j] * 
                                 correlation_matrix[i, j])
    
    portfolio_volatility = np.sqrt(max(0.0, portfolio_variance))
    
    if portfolio_volatility > 0:
        # Calculate marginal contributions
        for i in range(n_assets):
            marginal_contribution = 0.0
            for j in range(n_assets):
                marginal_contribution += (weights[j] * volatilities[i] * 
                                        volatilities[j] * correlation_matrix[i, j])
            
            # Component heat = weight * marginal contribution / portfolio volatility
            component_heat[i] = weights[i] * marginal_contribution / portfolio_volatility
    
    return component_heat


@njit(cache=True, fastmath=True)
def calculate_correlation_heat_jit(
    weights: np.ndarray,
    correlation_matrix: np.ndarray
) -> float:
    """
    Calculate correlation heat - JIT optimized
    
    Args:
        weights: Portfolio weights
        correlation_matrix: Correlation matrix
    
    Returns:
        Correlation heat measure
    """
    n_assets = len(weights)
    correlation_heat = 0.0
    
    # Calculate weighted average correlation
    total_weight = 0.0
    
    for i in range(n_assets):
        for j in range(n_assets):
            if i != j:
                weight = weights[i] * weights[j]
                correlation_heat += weight * correlation_matrix[i, j]
                total_weight += weight
    
    if total_weight > 0:
        correlation_heat = correlation_heat / total_weight
    
    return correlation_heat


@njit(cache=True, fastmath=True)
def calculate_concentration_heat_jit(weights: np.ndarray) -> float:
    """
    Calculate concentration heat using Herfindahl index - JIT optimized
    
    Args:
        weights: Portfolio weights
    
    Returns:
        Concentration heat measure
    """
    if len(weights) == 0:
        return 0.0
    
    # Herfindahl index
    herfindahl_index = 0.0
    for i in range(len(weights)):
        herfindahl_index += weights[i] ** 2
    
    return herfindahl_index


@njit(cache=True, fastmath=True)
def calculate_risk_contribution_jit(
    weights: np.ndarray,
    volatilities: np.ndarray,
    correlation_matrix: np.ndarray
) -> np.ndarray:
    """
    Calculate risk contribution for each asset - JIT optimized
    
    Args:
        weights: Portfolio weights
        volatilities: Asset volatilities
        correlation_matrix: Correlation matrix
    
    Returns:
        Risk contribution for each asset
    """
    n_assets = len(weights)
    risk_contribution = np.zeros(n_assets)
    
    # Calculate portfolio variance
    portfolio_variance = 0.0
    for i in range(n_assets):
        for j in range(n_assets):
            portfolio_variance += (weights[i] * weights[j] * 
                                 volatilities[i] * volatilities[j] * 
                                 correlation_matrix[i, j])
    
    if portfolio_variance > 0:
        # Calculate marginal contributions
        for i in range(n_assets):
            marginal_contribution = 0.0
            for j in range(n_assets):
                marginal_contribution += (weights[j] * volatilities[i] * 
                                        volatilities[j] * correlation_matrix[i, j])
            
            # Risk contribution = weight * marginal contribution / portfolio variance
            risk_contribution[i] = weights[i] * marginal_contribution / portfolio_variance
    
    return risk_contribution


@njit(cache=True, fastmath=True)
def calculate_exponential_correlation_jit(
    returns_x: np.ndarray,
    returns_y: np.ndarray,
    decay_factor: float
) -> float:
    """
    Calculate exponentially weighted correlation - JIT optimized
    
    Args:
        returns_x: First return series
        returns_y: Second return series
        decay_factor: Decay factor for exponential weighting
    
    Returns:
        Exponentially weighted correlation
    """
    n = len(returns_x)
    
    if n == 0 or len(returns_y) != n:
        return 0.0
    
    # Calculate exponentially weighted means
    sum_weights = 0.0
    sum_x = 0.0
    sum_y = 0.0
    
    for i in range(n):
        weight = decay_factor ** (n - 1 - i)
        sum_weights += weight
        sum_x += weight * returns_x[i]
        sum_y += weight * returns_y[i]
    
    if sum_weights == 0:
        return 0.0
    
    mean_x = sum_x / sum_weights
    mean_y = sum_y / sum_weights
    
    # Calculate exponentially weighted covariance and variances
    covariance = 0.0
    variance_x = 0.0
    variance_y = 0.0
    
    for i in range(n):
        weight = decay_factor ** (n - 1 - i)
        dx = returns_x[i] - mean_x
        dy = returns_y[i] - mean_y
        
        covariance += weight * dx * dy
        variance_x += weight * dx * dx
        variance_y += weight * dy * dy
    
    if sum_weights == 0:
        return 0.0
    
    covariance /= sum_weights
    variance_x /= sum_weights
    variance_y /= sum_weights
    
    # Calculate correlation
    denominator = np.sqrt(variance_x * variance_y)
    
    if denominator == 0:
        return 0.0
    
    correlation = covariance / denominator
    
    # Clamp to [-1, 1]
    return max(-1.0, min(1.0, correlation))


@njit(parallel=True, cache=True, fastmath=True)
def calculate_rolling_correlation_matrix_jit(
    returns: np.ndarray,
    window: int,
    decay_factor: float
) -> np.ndarray:
    """
    Calculate rolling correlation matrix - JIT optimized with parallel processing
    
    Args:
        returns: Return matrix (time x assets)
        window: Rolling window size
        decay_factor: Decay factor for exponential weighting
    
    Returns:
        Latest correlation matrix
    """
    n_time, n_assets = returns.shape
    correlation_matrix = np.eye(n_assets)
    
    if n_time < window:
        return correlation_matrix
    
    # Get latest window
    latest_returns = returns[-window:, :]
    
    # Calculate correlations in parallel
    for i in prange(n_assets):
        for j in range(i + 1, n_assets):
            corr = calculate_exponential_correlation_jit(
                latest_returns[:, i], latest_returns[:, j], decay_factor
            )
            correlation_matrix[i, j] = corr
            correlation_matrix[j, i] = corr
    
    return correlation_matrix


class PortfolioHeatController:
    """
    Portfolio Heat and Correlation-Based Risk Controller
    
    This class implements sophisticated portfolio heat monitoring and correlation-based
    risk controls with real-time monitoring and automated enforcement.
    """
    
    def __init__(self, config: PortfolioHeatConfig):
        """
        Initialize the portfolio heat controller
        
        Args:
            config: Portfolio heat configuration
        """
        self.config = config
        
        # State tracking
        self.current_heat: Optional[PortfolioHeat] = None
        self.correlation_analysis: Optional[CorrelationAnalysis] = None
        self.risk_allocation: Optional[RiskAllocation] = None
        
        # Historical data
        self.heat_history: List[PortfolioHeat] = []
        self.correlation_history: List[CorrelationAnalysis] = []
        
        # Clustering state
        self.asset_clusters: Dict[str, List[str]] = {}
        self.cluster_update_counter = 0
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.last_alert_time: Dict[str, datetime] = {}
        
        # Performance tracking
        self.calculation_times: Dict[str, List[float]] = {}
        
        logger.info("PortfolioHeatController initialized",
                   extra={'config': config.to_dict()})
    
    async def calculate_portfolio_heat(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        symbols: List[str],
        sectors: Optional[Dict[str, str]] = None
    ) -> PortfolioHeat:
        """
        Calculate comprehensive portfolio heat
        
        Args:
            weights: Portfolio weights
            returns: Return matrix (time x assets)
            symbols: Asset symbols
            sectors: Optional sector mapping
        
        Returns:
            PortfolioHeat object
        """
        start_time = datetime.now()
        
        try:
            # Calculate volatilities
            volatilities = np.std(returns, axis=0)
            
            # Calculate correlation matrix
            correlation_matrix = calculate_rolling_correlation_matrix_jit(
                returns, self.config.correlation_lookback, self.config.correlation_decay_factor
            )
            
            # Calculate total portfolio heat
            total_heat = calculate_portfolio_heat_jit(weights, volatilities, correlation_matrix)
            
            # Calculate component heat
            component_heat_array = calculate_component_heat_jit(weights, volatilities, correlation_matrix)
            component_heat = dict(zip(symbols, component_heat_array))
            
            # Calculate correlation heat
            correlation_heat = calculate_correlation_heat_jit(weights, correlation_matrix)
            
            # Calculate concentration heat
            concentration_heat = calculate_concentration_heat_jit(weights)
            
            # Calculate sector heat
            sector_heat = {}
            if sectors:
                sector_weights = {}
                for symbol, sector in sectors.items():
                    if symbol in symbols:
                        idx = symbols.index(symbol)
                        sector_weights[sector] = sector_weights.get(sector, 0) + weights[idx]
                
                for sector, weight in sector_weights.items():
                    sector_heat[sector] = weight ** 2  # Simplified sector heat
            
            # Calculate risk contribution
            risk_contribution_array = calculate_risk_contribution_jit(
                weights, volatilities, correlation_matrix
            )
            risk_contribution = dict(zip(symbols, risk_contribution_array))
            
            # Calculate heat breakdown
            heat_breakdown = {
                'idiosyncratic': total_heat * (1 - correlation_heat),
                'correlation': total_heat * correlation_heat,
                'concentration': concentration_heat * 0.1,  # Scaled
                'sector': sum(sector_heat.values()) * 0.1  # Scaled
            }
            
            # Create portfolio heat object
            portfolio_heat = PortfolioHeat(
                total_heat=total_heat,
                component_heat=component_heat,
                correlation_heat=correlation_heat,
                concentration_heat=concentration_heat,
                sector_heat=sector_heat,
                risk_contribution=risk_contribution,
                heat_breakdown=heat_breakdown
            )
            
            # Store in history
            self.heat_history.append(portfolio_heat)
            self.current_heat = portfolio_heat
            
            # Keep only recent history
            if len(self.heat_history) > 1000:
                self.heat_history = self.heat_history[-1000:]
            
            # Check heat limits
            await self._check_heat_limits(portfolio_heat)
            
            return portfolio_heat
            
        except Exception as e:
            logger.error(f"Portfolio heat calculation failed: {e}")
            raise
        
        finally:
            # Track performance
            calc_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if 'portfolio_heat' not in self.calculation_times:
                self.calculation_times['portfolio_heat'] = []
            
            self.calculation_times['portfolio_heat'].append(calc_time)
    
    async def analyze_correlations(
        self,
        returns: np.ndarray,
        symbols: List[str],
        update_clusters: bool = True
    ) -> CorrelationAnalysis:
        """
        Analyze portfolio correlations
        
        Args:
            returns: Return matrix (time x assets)
            symbols: Asset symbols
            update_clusters: Whether to update clusters
        
        Returns:
            CorrelationAnalysis object
        """
        start_time = datetime.now()
        
        try:
            # Calculate correlation matrix
            correlation_matrix = calculate_rolling_correlation_matrix_jit(
                returns, self.config.correlation_lookback, self.config.correlation_decay_factor
            )
            
            # Calculate statistics
            n_assets = len(symbols)
            correlations = []
            
            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    correlations.append(correlation_matrix[i, j])
            
            average_correlation = np.mean(correlations) if correlations else 0.0
            max_correlation = np.max(correlations) if correlations else 0.0
            
            # Update clusters if needed
            if update_clusters:
                self.cluster_update_counter += 1
                if self.cluster_update_counter >= self.config.cluster_update_frequency:
                    self.asset_clusters = await self._update_correlation_clusters(
                        correlation_matrix, symbols
                    )
                    self.cluster_update_counter = 0
            
            # Determine correlation regime
            correlation_regime = self._determine_correlation_regime(average_correlation)
            
            # Calculate regime transition probability
            regime_transition_prob = self._calculate_regime_transition_probability(
                average_correlation
            )
            
            # Calculate rolling correlations for key pairs
            rolling_correlations = {}
            if len(symbols) >= 2:
                for i in range(min(5, len(symbols))):
                    for j in range(i + 1, min(5, len(symbols))):
                        key = f"{symbols[i]}_{symbols[j]}"
                        rolling_correlations[key] = self._calculate_rolling_correlation(
                            returns[:, i], returns[:, j]
                        )
            
            # Create correlation analysis object
            correlation_analysis = CorrelationAnalysis(
                correlation_matrix=correlation_matrix,
                average_correlation=average_correlation,
                max_correlation=max_correlation,
                correlation_clusters=self.asset_clusters,
                correlation_regime=correlation_regime,
                regime_transition_probability=regime_transition_prob,
                rolling_correlations=rolling_correlations
            )
            
            # Store in history
            self.correlation_history.append(correlation_analysis)
            self.correlation_analysis = correlation_analysis
            
            # Keep only recent history
            if len(self.correlation_history) > 1000:
                self.correlation_history = self.correlation_history[-1000:]
            
            # Check correlation limits
            await self._check_correlation_limits(correlation_analysis)
            
            return correlation_analysis
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            raise
        
        finally:
            # Track performance
            calc_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if 'correlation_analysis' not in self.calculation_times:
                self.calculation_times['correlation_analysis'] = []
            
            self.calculation_times['correlation_analysis'].append(calc_time)
    
    async def _update_correlation_clusters(
        self,
        correlation_matrix: np.ndarray,
        symbols: List[str]
    ) -> Dict[str, List[str]]:
        """Update correlation clusters"""
        
        try:
            # Convert correlation to distance
            distance_matrix = 1 - np.abs(correlation_matrix)
            
            # Perform clustering
            if self.config.cluster_method == "hierarchical":
                # Hierarchical clustering
                condensed_distances = pdist(distance_matrix)
                linkage_matrix = linkage(condensed_distances, method='ward')
                cluster_labels = fcluster(linkage_matrix, self.config.n_clusters, criterion='maxclust')
            
            else:  # KMeans
                # K-means clustering
                kmeans = KMeans(n_clusters=self.config.n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(distance_matrix)
            
            # Create cluster mapping
            clusters = {}
            for i, symbol in enumerate(symbols):
                cluster_id = f"cluster_{cluster_labels[i]}"
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(symbol)
            
            return clusters
            
        except Exception as e:
            logger.warning(f"Cluster update failed: {e}")
            return self.asset_clusters
    
    def _determine_correlation_regime(self, average_correlation: float) -> str:
        """Determine correlation regime"""
        
        if average_correlation > 0.8:
            return "crisis"
        elif average_correlation > 0.6:
            return "elevated"
        elif average_correlation > 0.4:
            return "normal"
        else:
            return "low"
    
    def _calculate_regime_transition_probability(self, average_correlation: float) -> float:
        """Calculate regime transition probability"""
        
        if not self.correlation_history:
            return 0.0
        
        # Calculate correlation change
        recent_correlations = [analysis.average_correlation for analysis in self.correlation_history[-10:]]
        
        if len(recent_correlations) < 2:
            return 0.0
        
        # Simple momentum-based probability
        correlation_change = recent_correlations[-1] - recent_correlations[0]
        
        # Normalize to probability
        transition_prob = np.tanh(abs(correlation_change) * 10)
        
        return min(1.0, max(0.0, transition_prob))
    
    def _calculate_rolling_correlation(
        self,
        returns_x: np.ndarray,
        returns_y: np.ndarray,
        window: int = 30
    ) -> np.ndarray:
        """Calculate rolling correlation"""
        
        rolling_corr = np.zeros(len(returns_x))
        
        for i in range(window, len(returns_x)):
            window_x = returns_x[i-window:i]
            window_y = returns_y[i-window:i]
            
            # Calculate correlation
            if len(window_x) > 1 and len(window_y) > 1:
                corr_matrix = np.corrcoef(window_x, window_y)
                rolling_corr[i] = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
        
        return rolling_corr
    
    async def calculate_risk_allocation(
        self,
        weights: np.ndarray,
        symbols: List[str],
        target_allocation: Optional[Dict[str, float]] = None
    ) -> RiskAllocation:
        """
        Calculate risk allocation analysis
        
        Args:
            weights: Current portfolio weights
            symbols: Asset symbols
            target_allocation: Target risk allocation
        
        Returns:
            RiskAllocation object
        """
        
        # Current risk allocation
        current_allocation = dict(zip(symbols, weights))
        
        # Use configured target or equal allocation
        if target_allocation is None:
            target_allocation = self.config.target_risk_allocation
        
        if not target_allocation:
            # Equal risk allocation
            n_assets = len(symbols)
            target_allocation = {symbol: 1.0 / n_assets for symbol in symbols}
        
        # Calculate allocation drift
        allocation_drift = {}
        for symbol in symbols:
            current = current_allocation.get(symbol, 0.0)
            target = target_allocation.get(symbol, 0.0)
            allocation_drift[symbol] = current - target
        
        # Check if rebalancing is needed
        max_drift = max(abs(drift) for drift in allocation_drift.values())
        rebalance_needed = max_drift > 0.05  # 5% threshold
        
        # Calculate rebalance trades
        rebalance_trades = {}
        if rebalance_needed:
            for symbol, drift in allocation_drift.items():
                if abs(drift) > 0.01:  # 1% threshold for trades
                    rebalance_trades[symbol] = -drift  # Opposite of drift
        
        # Calculate risk budget utilization
        total_risk = sum(abs(weight) for weight in weights)
        risk_budget_utilization = total_risk
        
        # Create risk allocation object
        risk_allocation = RiskAllocation(
            current_allocation=current_allocation,
            target_allocation=target_allocation,
            allocation_drift=allocation_drift,
            rebalance_needed=rebalance_needed,
            rebalance_trades=rebalance_trades,
            risk_budget_utilization=risk_budget_utilization
        )
        
        self.risk_allocation = risk_allocation
        
        return risk_allocation
    
    async def _check_heat_limits(self, portfolio_heat: PortfolioHeat) -> None:
        """Check portfolio heat limits"""
        
        # Check total heat
        if portfolio_heat.total_heat > self.config.max_portfolio_heat:
            await self._generate_alert(
                "HEAT_LIMIT_BREACH",
                f"Portfolio heat {portfolio_heat.total_heat:.2%} exceeds limit {self.config.max_portfolio_heat:.2%}",
                {"current_heat": portfolio_heat.total_heat, "limit": self.config.max_portfolio_heat}
            )
        
        # Check heat warning
        elif portfolio_heat.total_heat > self.config.heat_warning_threshold:
            await self._generate_alert(
                "HEAT_WARNING",
                f"Portfolio heat {portfolio_heat.total_heat:.2%} exceeds warning threshold {self.config.heat_warning_threshold:.2%}",
                {"current_heat": portfolio_heat.total_heat, "threshold": self.config.heat_warning_threshold}
            )
        
        # Check concentration heat
        if portfolio_heat.concentration_heat > self.config.max_concentration:
            await self._generate_alert(
                "CONCENTRATION_LIMIT_BREACH",
                f"Portfolio concentration {portfolio_heat.concentration_heat:.2%} exceeds limit {self.config.max_concentration:.2%}",
                {"current_concentration": portfolio_heat.concentration_heat, "limit": self.config.max_concentration}
            )
    
    async def _check_correlation_limits(self, correlation_analysis: CorrelationAnalysis) -> None:
        """Check correlation limits"""
        
        # Check maximum correlation
        if correlation_analysis.max_correlation > self.config.max_correlation:
            await self._generate_alert(
                "CORRELATION_LIMIT_BREACH",
                f"Maximum correlation {correlation_analysis.max_correlation:.2%} exceeds limit {self.config.max_correlation:.2%}",
                {"max_correlation": correlation_analysis.max_correlation, "limit": self.config.max_correlation}
            )
        
        # Check average correlation warning
        elif correlation_analysis.average_correlation > self.config.correlation_warning_threshold:
            await self._generate_alert(
                "CORRELATION_WARNING",
                f"Average correlation {correlation_analysis.average_correlation:.2%} exceeds warning threshold {self.config.correlation_warning_threshold:.2%}",
                {"avg_correlation": correlation_analysis.average_correlation, "threshold": self.config.correlation_warning_threshold}
            )
    
    async def _generate_alert(
        self,
        alert_type: str,
        message: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Generate risk alert"""
        
        # Check cooldown
        last_alert = self.last_alert_time.get(alert_type)
        if last_alert and (datetime.now() - last_alert).seconds < self.config.alert_cooldown:
            return
        
        # Log alert
        logger.warning(f"Portfolio heat alert: {message}",
                      extra={'alert_type': alert_type, 'metadata': metadata})
        
        # Update last alert time
        self.last_alert_time[alert_type] = datetime.now()
    
    async def start_monitoring(self) -> None:
        """Start real-time monitoring"""
        
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Portfolio heat monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop real-time monitoring"""
        
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Portfolio heat monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        
        while self.monitoring_active:
            try:
                # This would be called by external system with real data
                await asyncio.sleep(self.config.monitoring_frequency * 60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.config.monitoring_frequency * 60)
    
    def get_heat_summary(self) -> Dict[str, Any]:
        """Get comprehensive heat summary"""
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'current_heat': self.current_heat.to_dict() if self.current_heat else {},
            'correlation_analysis': self.correlation_analysis.to_dict() if self.correlation_analysis else {},
            'risk_allocation': self.risk_allocation.to_dict() if self.risk_allocation else {},
            'monitoring_active': self.monitoring_active,
            'heat_history_count': len(self.heat_history),
            'correlation_history_count': len(self.correlation_history),
            'asset_clusters': self.asset_clusters,
            'performance_metrics': {
                method: {
                    'avg_time_ms': np.mean(times),
                    'calculation_count': len(times)
                }
                for method, times in self.calculation_times.items()
            }
        }
        
        return summary


# Factory function
def create_portfolio_heat_controller(config_dict: Optional[Dict[str, Any]] = None) -> PortfolioHeatController:
    """
    Create a portfolio heat controller with configuration
    
    Args:
        config_dict: Configuration dictionary
    
    Returns:
        PortfolioHeatController instance
    """
    
    if config_dict is None:
        config = PortfolioHeatConfig()
    else:
        config = PortfolioHeatConfig(**config_dict)
    
    return PortfolioHeatController(config)