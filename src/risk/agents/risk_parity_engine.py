"""
Advanced Risk Parity Engine

This module implements sophisticated risk parity allocation methods for portfolio optimization,
ensuring equal risk contribution across strategies with dynamic rebalancing and regime-aware
adjustments.

Key Features:
- Equal Risk Contribution (ERC) portfolio optimization
- Hierarchical Risk Parity (HRP) using machine learning clustering
- Risk budgeting with custom risk allocation targets
- Dynamic risk parity with correlation regime adjustments
- Volatility-adjusted risk parity scaling
- Multi-timeframe risk contribution analysis
- Advanced risk decomposition and attribution
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import structlog
from scipy import optimize, cluster
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings('ignore')

from src.core.events import EventBus, Event, EventType
from src.risk.core.correlation_tracker import CorrelationRegime

logger = structlog.get_logger()


class RiskParityMethod(Enum):
    """Risk parity allocation methods"""
    EQUAL_RISK_CONTRIBUTION = "erc"
    HIERARCHICAL_RISK_PARITY = "hrp"
    RISK_BUDGETING = "risk_budgeting"
    INVERSE_VOLATILITY = "inverse_volatility"
    EQUAL_MARGINAL_RISK = "equal_marginal_risk"


@dataclass
class RiskBudget:
    """Risk budget allocation specification"""
    strategy_id: str
    target_risk_contribution: float  # Target % of total portfolio risk
    min_weight: float = 0.01
    max_weight: float = 0.99
    priority: int = 1  # Higher priority strategies get preference


@dataclass
class RiskContribution:
    """Risk contribution analysis for a strategy"""
    strategy_id: str
    weight: float
    marginal_risk: float  # dRisk/dWeight
    component_risk: float  # Weight * Marginal Risk
    risk_contribution_pct: float  # % of total portfolio risk
    volatility: float
    correlation_with_portfolio: float


@dataclass
class RiskParityResult:
    """Risk parity optimization result"""
    timestamp: datetime
    method: RiskParityMethod
    weights: np.ndarray
    risk_contributions: List[RiskContribution]
    total_portfolio_risk: float
    risk_concentration_index: float  # Herfindahl index of risk contributions
    diversification_ratio: float
    effective_strategies: float  # Effective number of strategies by risk
    convergence_iterations: int
    convergence_error: float
    optimization_time_ms: float


@dataclass
class HierarchicalCluster:
    """Hierarchical clustering result for HRP"""
    cluster_id: int
    strategy_indices: List[int]
    intra_cluster_weights: np.ndarray
    cluster_risk: float
    cluster_weight: float


class RiskParityEngine:
    """
    Advanced Risk Parity Engine implementing multiple risk parity methods
    with sophisticated risk decomposition and dynamic regime adjustments.
    """
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        n_strategies: int = 5,
        convergence_tolerance: float = 1e-8,
        max_iterations: int = 1000
    ):
        """
        Initialize Risk Parity Engine
        
        Args:
            event_bus: Event bus for communication
            n_strategies: Number of strategies
            convergence_tolerance: Convergence tolerance for optimization
            max_iterations: Maximum optimization iterations
        """
        self.event_bus = event_bus
        self.n_strategies = n_strategies
        self.convergence_tolerance = convergence_tolerance
        self.max_iterations = max_iterations
        
        # Risk parity parameters
        self.risk_aversion = 1.0  # Risk aversion parameter for utility
        self.shrinkage_intensity = 0.1  # Covariance shrinkage
        
        # Hierarchical clustering parameters
        self.linkage_method = 'ward'  # Clustering linkage method
        self.distance_metric = 'correlation'  # Distance metric for clustering
        
        # Performance tracking
        self.optimization_history: List[RiskParityResult] = []
        self.calculation_times: List[float] = []
        
        logger.info("Risk Parity Engine initialized",
                   n_strategies=n_strategies,
                   convergence_tolerance=convergence_tolerance,
                   max_iterations=max_iterations)
    
    def optimize_equal_risk_contribution(
        self,
        covariance_matrix: np.ndarray,
        initial_weights: Optional[np.ndarray] = None,
        regime: CorrelationRegime = CorrelationRegime.NORMAL
    ) -> RiskParityResult:
        """
        Optimize for Equal Risk Contribution (ERC) portfolio
        
        Args:
            covariance_matrix: Strategy covariance matrix
            initial_weights: Initial weight guess
            regime: Current market regime
            
        Returns:
            Risk parity optimization result
        """
        start_time = datetime.now()
        
        if initial_weights is None:
            initial_weights = np.array([1.0 / self.n_strategies] * self.n_strategies)
        
        # Adjust covariance matrix for regime
        adjusted_cov = self._adjust_covariance_for_regime(covariance_matrix, regime)
        
        try:
            # ERC optimization: minimize sum of squared differences in risk contributions
            def erc_objective(weights):
                """Objective function for ERC optimization"""
                try:
                    # Calculate portfolio variance
                    portfolio_variance = np.dot(weights, np.dot(adjusted_cov, weights))
                    portfolio_vol = np.sqrt(max(portfolio_variance, 1e-12))
                    
                    # Calculate marginal risk contributions
                    marginal_contributions = np.dot(adjusted_cov, weights) / portfolio_vol
                    
                    # Risk contributions
                    risk_contributions = weights * marginal_contributions
                    
                    # Target equal risk contribution
                    target_contribution = 1.0 / self.n_strategies
                    
                    # Sum of squared deviations from equal risk contribution
                    risk_contrib_pct = risk_contributions / np.sum(risk_contributions)
                    deviations = risk_contrib_pct - target_contribution
                    
                    return np.sum(deviations**2)
                    
                except Exception as e:
                    logger.warning("ERC objective evaluation failed", error=str(e))
                    return 1e6  # Large penalty
            
            # Constraints: weights sum to 1, all weights positive
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
            ]
            
            # Bounds: small positive weights
            bounds = [(0.01, 0.99) for _ in range(self.n_strategies)]
            
            # Optimization
            result = optimize.minimize(
                erc_objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self.max_iterations, 'ftol': self.convergence_tolerance}
            )
            
            if result.success:
                optimal_weights = result.x
                convergence_error = result.fun
                convergence_iterations = result.nit
            else:
                logger.warning("ERC optimization failed, using inverse volatility weights")
                optimal_weights = self._inverse_volatility_weights(adjusted_cov)
                convergence_error = 1.0
                convergence_iterations = 0
            
            # Calculate final risk contributions
            risk_contributions = self._calculate_risk_contributions(optimal_weights, adjusted_cov)
            
            # Portfolio metrics
            portfolio_variance = np.dot(optimal_weights, np.dot(adjusted_cov, optimal_weights))
            total_portfolio_risk = np.sqrt(max(portfolio_variance, 1e-12))
            
            # Risk concentration (Herfindahl index)
            risk_contrib_pcts = [rc.risk_contribution_pct for rc in risk_contributions]
            risk_concentration = np.sum(np.array(risk_contrib_pcts)**2)
            
            # Diversification ratio
            individual_vols = np.sqrt(np.diag(adjusted_cov))
            weighted_avg_vol = np.dot(optimal_weights, individual_vols)
            diversification_ratio = weighted_avg_vol / total_portfolio_risk if total_portfolio_risk > 0 else 1.0
            
            # Effective number of strategies
            effective_strategies = 1.0 / risk_concentration if risk_concentration > 0 else self.n_strategies
            
            optimization_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = RiskParityResult(
                timestamp=datetime.now(),
                method=RiskParityMethod.EQUAL_RISK_CONTRIBUTION,
                weights=optimal_weights,
                risk_contributions=risk_contributions,
                total_portfolio_risk=total_portfolio_risk,
                risk_concentration_index=risk_concentration,
                diversification_ratio=diversification_ratio,
                effective_strategies=effective_strategies,
                convergence_iterations=convergence_iterations,
                convergence_error=convergence_error,
                optimization_time_ms=optimization_time
            )
            
            # Store result
            self.optimization_history.append(result)
            self.calculation_times.append(optimization_time)
            
            logger.info("ERC optimization completed",
                       convergence_iterations=convergence_iterations,
                       convergence_error=convergence_error,
                       effective_strategies=effective_strategies,
                       optimization_time_ms=optimization_time)
            
            return result
            
        except Exception as e:
            logger.error("ERC optimization failed", error=str(e))
            return self._fallback_equal_weights_result(covariance_matrix, start_time)
    
    def optimize_hierarchical_risk_parity(
        self,
        covariance_matrix: np.ndarray,
        returns_data: Optional[np.ndarray] = None,
        regime: CorrelationRegime = CorrelationRegime.NORMAL
    ) -> RiskParityResult:
        """
        Optimize using Hierarchical Risk Parity (HRP) method
        
        Args:
            covariance_matrix: Strategy covariance matrix
            returns_data: Historical returns data for clustering (optional)
            regime: Current market regime
            
        Returns:
            HRP optimization result
        """
        start_time = datetime.now()
        
        # Adjust covariance matrix for regime
        adjusted_cov = self._adjust_covariance_for_regime(covariance_matrix, regime)
        
        try:
            # Step 1: Tree clustering based on correlation matrix
            correlation_matrix = self._cov_to_corr(adjusted_cov)
            distance_matrix = np.sqrt(0.5 * (1 - correlation_matrix))
            
            # Hierarchical clustering
            from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
            
            # Convert to condensed distance matrix for clustering
            condensed_distances = squareform(distance_matrix, checks=False)
            
            # Perform clustering
            linkage_matrix = linkage(condensed_distances, method=self.linkage_method)
            
            # Step 2: Quasi-diagonalization
            # Get optimal leaf order from dendrogram
            def get_quasi_diag_order(linkage_matrix):
                """Get quasi-diagonal order from hierarchical clustering"""
                from scipy.cluster.hierarchy import dendrogram
                
                # Create dendrogram to get leaf order
                dend = dendrogram(linkage_matrix, no_plot=True)
                return dend['leaves']
            
            quasi_diag_order = get_quasi_diag_order(linkage_matrix)
            
            # Reorder covariance matrix
            ordered_cov = adjusted_cov[np.ix_(quasi_diag_order, quasi_diag_order)]
            
            # Step 3: Recursive bisection for weight allocation
            hrp_weights = self._recursive_bisection(ordered_cov, 0, len(quasi_diag_order))
            
            # Reorder weights back to original order
            original_weights = np.zeros(self.n_strategies)
            for i, orig_idx in enumerate(quasi_diag_order):
                original_weights[orig_idx] = hrp_weights[i]
            
            # Calculate risk contributions
            risk_contributions = self._calculate_risk_contributions(original_weights, adjusted_cov)
            
            # Portfolio metrics
            portfolio_variance = np.dot(original_weights, np.dot(adjusted_cov, original_weights))
            total_portfolio_risk = np.sqrt(max(portfolio_variance, 1e-12))
            
            # Risk concentration
            risk_contrib_pcts = [rc.risk_contribution_pct for rc in risk_contributions]
            risk_concentration = np.sum(np.array(risk_contrib_pcts)**2)
            
            # Diversification ratio
            individual_vols = np.sqrt(np.diag(adjusted_cov))
            weighted_avg_vol = np.dot(original_weights, individual_vols)
            diversification_ratio = weighted_avg_vol / total_portfolio_risk if total_portfolio_risk > 0 else 1.0
            
            # Effective strategies
            effective_strategies = 1.0 / risk_concentration if risk_concentration > 0 else self.n_strategies
            
            optimization_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = RiskParityResult(
                timestamp=datetime.now(),
                method=RiskParityMethod.HIERARCHICAL_RISK_PARITY,
                weights=original_weights,
                risk_contributions=risk_contributions,
                total_portfolio_risk=total_portfolio_risk,
                risk_concentration_index=risk_concentration,
                diversification_ratio=diversification_ratio,
                effective_strategies=effective_strategies,
                convergence_iterations=1,  # HRP doesn't iterate
                convergence_error=0.0,
                optimization_time_ms=optimization_time
            )
            
            self.optimization_history.append(result)
            self.calculation_times.append(optimization_time)
            
            logger.info("HRP optimization completed",
                       effective_strategies=effective_strategies,
                       diversification_ratio=diversification_ratio,
                       optimization_time_ms=optimization_time)
            
            return result
            
        except Exception as e:
            logger.error("HRP optimization failed", error=str(e))
            return self._fallback_equal_weights_result(covariance_matrix, start_time)
    
    def optimize_risk_budgeting(
        self,
        covariance_matrix: np.ndarray,
        risk_budgets: List[RiskBudget],
        regime: CorrelationRegime = CorrelationRegime.NORMAL
    ) -> RiskParityResult:
        """
        Optimize portfolio with custom risk budget allocations
        
        Args:
            covariance_matrix: Strategy covariance matrix
            risk_budgets: Target risk budget allocations
            regime: Current market regime
            
        Returns:
            Risk budgeting optimization result
        """
        start_time = datetime.now()
        
        # Validate risk budgets
        total_budget = sum(rb.target_risk_contribution for rb in risk_budgets)
        if abs(total_budget - 1.0) > 1e-6:
            # Normalize risk budgets
            for rb in risk_budgets:
                rb.target_risk_contribution /= total_budget
        
        # Adjust covariance matrix for regime
        adjusted_cov = self._adjust_covariance_for_regime(covariance_matrix, regime)
        
        try:
            # Create target risk contribution vector
            target_risk_contributions = np.zeros(self.n_strategies)
            weight_bounds = []
            
            for i in range(self.n_strategies):
                # Find corresponding risk budget
                budget = next((rb for rb in risk_budgets if rb.strategy_id == str(i)), None)
                if budget:
                    target_risk_contributions[i] = budget.target_risk_contribution
                    weight_bounds.append((budget.min_weight, budget.max_weight))
                else:
                    # Default equal allocation for unspecified strategies
                    remaining_strategies = self.n_strategies - len(risk_budgets)
                    remaining_budget = 1.0 - sum(rb.target_risk_contribution for rb in risk_budgets)
                    target_risk_contributions[i] = remaining_budget / max(1, remaining_strategies)
                    weight_bounds.append((0.01, 0.99))
            
            def risk_budgeting_objective(weights):
                """Risk budgeting objective function"""
                try:
                    # Portfolio variance and volatility
                    portfolio_variance = np.dot(weights, np.dot(adjusted_cov, weights))
                    portfolio_vol = np.sqrt(max(portfolio_variance, 1e-12))
                    
                    # Marginal risk contributions
                    marginal_contributions = np.dot(adjusted_cov, weights) / portfolio_vol
                    
                    # Actual risk contributions
                    risk_contributions = weights * marginal_contributions
                    total_risk = np.sum(risk_contributions)
                    
                    if total_risk > 0:
                        risk_contrib_pct = risk_contributions / total_risk
                    else:
                        risk_contrib_pct = weights / np.sum(weights)
                    
                    # Squared deviations from target risk contributions
                    deviations = risk_contrib_pct - target_risk_contributions
                    return np.sum(deviations**2)
                    
                except Exception as e:
                    logger.warning("Risk budgeting objective failed", error=str(e))
                    return 1e6
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
            ]
            
            # Initial guess
            initial_weights = np.array([1.0 / self.n_strategies] * self.n_strategies)
            
            # Optimization
            result = optimize.minimize(
                risk_budgeting_objective,
                initial_weights,
                method='SLSQP',
                bounds=weight_bounds,
                constraints=constraints,
                options={'maxiter': self.max_iterations, 'ftol': self.convergence_tolerance}
            )
            
            if result.success:
                optimal_weights = result.x
                convergence_error = result.fun
                convergence_iterations = result.nit
            else:
                logger.warning("Risk budgeting optimization failed")
                optimal_weights = self._risk_budget_fallback_weights(risk_budgets, adjusted_cov)
                convergence_error = 1.0
                convergence_iterations = 0
            
            # Calculate risk contributions
            risk_contributions = self._calculate_risk_contributions(optimal_weights, adjusted_cov)
            
            # Portfolio metrics
            portfolio_variance = np.dot(optimal_weights, np.dot(adjusted_cov, optimal_weights))
            total_portfolio_risk = np.sqrt(max(portfolio_variance, 1e-12))
            
            # Risk concentration
            risk_contrib_pcts = [rc.risk_contribution_pct for rc in risk_contributions]
            risk_concentration = np.sum(np.array(risk_contrib_pcts)**2)
            
            # Diversification ratio
            individual_vols = np.sqrt(np.diag(adjusted_cov))
            weighted_avg_vol = np.dot(optimal_weights, individual_vols)
            diversification_ratio = weighted_avg_vol / total_portfolio_risk if total_portfolio_risk > 0 else 1.0
            
            # Effective strategies
            effective_strategies = 1.0 / risk_concentration if risk_concentration > 0 else self.n_strategies
            
            optimization_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = RiskParityResult(
                timestamp=datetime.now(),
                method=RiskParityMethod.RISK_BUDGETING,
                weights=optimal_weights,
                risk_contributions=risk_contributions,
                total_portfolio_risk=total_portfolio_risk,
                risk_concentration_index=risk_concentration,
                diversification_ratio=diversification_ratio,
                effective_strategies=effective_strategies,
                convergence_iterations=convergence_iterations,
                convergence_error=convergence_error,
                optimization_time_ms=optimization_time
            )
            
            self.optimization_history.append(result)
            self.calculation_times.append(optimization_time)
            
            logger.info("Risk budgeting optimization completed",
                       convergence_iterations=convergence_iterations,
                       effective_strategies=effective_strategies,
                       optimization_time_ms=optimization_time)
            
            return result
            
        except Exception as e:
            logger.error("Risk budgeting optimization failed", error=str(e))
            return self._fallback_equal_weights_result(covariance_matrix, start_time)
    
    def optimize_inverse_volatility(
        self,
        covariance_matrix: np.ndarray,
        regime: CorrelationRegime = CorrelationRegime.NORMAL
    ) -> RiskParityResult:
        """
        Simple inverse volatility weighting
        
        Args:
            covariance_matrix: Strategy covariance matrix
            regime: Current market regime
            
        Returns:
            Inverse volatility result
        """
        start_time = datetime.now()
        
        try:
            # Adjust covariance matrix for regime
            adjusted_cov = self._adjust_covariance_for_regime(covariance_matrix, regime)
            
            # Extract volatilities
            volatilities = np.sqrt(np.diag(adjusted_cov))
            
            # Inverse volatility weights
            inv_vol_weights = 1.0 / volatilities
            weights = inv_vol_weights / np.sum(inv_vol_weights)
            
            # Calculate risk contributions
            risk_contributions = self._calculate_risk_contributions(weights, adjusted_cov)
            
            # Portfolio metrics
            portfolio_variance = np.dot(weights, np.dot(adjusted_cov, weights))
            total_portfolio_risk = np.sqrt(max(portfolio_variance, 1e-12))
            
            # Risk concentration
            risk_contrib_pcts = [rc.risk_contribution_pct for rc in risk_contributions]
            risk_concentration = np.sum(np.array(risk_contrib_pcts)**2)
            
            # Diversification ratio
            weighted_avg_vol = np.dot(weights, volatilities)
            diversification_ratio = weighted_avg_vol / total_portfolio_risk if total_portfolio_risk > 0 else 1.0
            
            # Effective strategies
            effective_strategies = 1.0 / risk_concentration if risk_concentration > 0 else self.n_strategies
            
            optimization_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = RiskParityResult(
                timestamp=datetime.now(),
                method=RiskParityMethod.INVERSE_VOLATILITY,
                weights=weights,
                risk_contributions=risk_contributions,
                total_portfolio_risk=total_portfolio_risk,
                risk_concentration_index=risk_concentration,
                diversification_ratio=diversification_ratio,
                effective_strategies=effective_strategies,
                convergence_iterations=1,
                convergence_error=0.0,
                optimization_time_ms=optimization_time
            )
            
            self.optimization_history.append(result)
            self.calculation_times.append(optimization_time)
            
            return result
            
        except Exception as e:
            logger.error("Inverse volatility optimization failed", error=str(e))
            return self._fallback_equal_weights_result(covariance_matrix, start_time)
    
    def _calculate_risk_contributions(
        self,
        weights: np.ndarray,
        covariance_matrix: np.ndarray
    ) -> List[RiskContribution]:
        """Calculate detailed risk contributions for each strategy"""
        
        risk_contributions = []
        
        try:
            # Portfolio variance and volatility
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            portfolio_vol = np.sqrt(max(portfolio_variance, 1e-12))
            
            # Marginal risk contributions
            marginal_contributions = np.dot(covariance_matrix, weights) / portfolio_vol
            
            # Component risk contributions
            component_risks = weights * marginal_contributions
            total_component_risk = np.sum(component_risks)
            
            # Individual volatilities
            individual_vols = np.sqrt(np.diag(covariance_matrix))
            
            for i in range(len(weights)):
                # Correlation with portfolio
                portfolio_cov = np.dot(covariance_matrix[i, :], weights)
                correlation_with_portfolio = portfolio_cov / (individual_vols[i] * portfolio_vol)
                
                risk_contribution = RiskContribution(
                    strategy_id=str(i),
                    weight=weights[i],
                    marginal_risk=marginal_contributions[i],
                    component_risk=component_risks[i],
                    risk_contribution_pct=component_risks[i] / total_component_risk if total_component_risk > 0 else 0,
                    volatility=individual_vols[i],
                    correlation_with_portfolio=correlation_with_portfolio
                )
                
                risk_contributions.append(risk_contribution)
        
        except Exception as e:
            logger.error("Risk contribution calculation failed", error=str(e))
            # Return default contributions
            for i in range(len(weights)):
                risk_contributions.append(
                    RiskContribution(
                        strategy_id=str(i),
                        weight=weights[i],
                        marginal_risk=0.0,
                        component_risk=0.0,
                        risk_contribution_pct=weights[i],
                        volatility=0.15,  # Default
                        correlation_with_portfolio=0.5  # Default
                    )
                )
        
        return risk_contributions
    
    def _recursive_bisection(self, cov_matrix: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
        """Recursive bisection for HRP weight allocation"""
        
        size = end_idx - start_idx
        if size == 1:
            return np.array([1.0])
        
        # Split into two clusters
        mid_idx = start_idx + size // 2
        
        # Get left and right cluster covariance matrices
        left_cov = cov_matrix[start_idx:mid_idx, start_idx:mid_idx]
        right_cov = cov_matrix[mid_idx:end_idx, mid_idx:end_idx]
        
        # Calculate cluster variances (using inverse volatility)
        left_vol = self._cluster_variance(left_cov)
        right_vol = self._cluster_variance(right_cov)
        
        # Allocate weight between clusters (inverse volatility)
        total_inv_vol = 1.0 / left_vol + 1.0 / right_vol
        left_cluster_weight = (1.0 / left_vol) / total_inv_vol
        right_cluster_weight = (1.0 / right_vol) / total_inv_vol
        
        # Recursive allocation within clusters
        left_weights = self._recursive_bisection(cov_matrix, start_idx, mid_idx) * left_cluster_weight
        right_weights = self._recursive_bisection(cov_matrix, mid_idx, end_idx) * right_cluster_weight
        
        return np.concatenate([left_weights, right_weights])
    
    def _cluster_variance(self, cov_matrix: np.ndarray) -> float:
        """Calculate cluster variance for HRP"""
        # Use inverse volatility weights within cluster
        volatilities = np.sqrt(np.diag(cov_matrix))
        inv_vol_weights = 1.0 / volatilities
        weights = inv_vol_weights / np.sum(inv_vol_weights)
        
        # Cluster variance
        cluster_variance = np.dot(weights, np.dot(cov_matrix, weights))
        return np.sqrt(max(cluster_variance, 1e-12))
    
    def _cov_to_corr(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Convert covariance matrix to correlation matrix"""
        volatilities = np.sqrt(np.diag(cov_matrix))
        volatilities_outer = np.outer(volatilities, volatilities)
        correlation_matrix = cov_matrix / volatilities_outer
        np.fill_diagonal(correlation_matrix, 1.0)
        return correlation_matrix
    
    def _adjust_covariance_for_regime(
        self,
        cov_matrix: np.ndarray,
        regime: CorrelationRegime
    ) -> np.ndarray:
        """Adjust covariance matrix based on market regime"""
        
        adjusted_cov = cov_matrix.copy()
        
        if regime == CorrelationRegime.CRISIS or regime == CorrelationRegime.SHOCK:
            # Increase volatilities and correlations in crisis
            volatilities = np.sqrt(np.diag(cov_matrix))
            correlation_matrix = self._cov_to_corr(cov_matrix)
            
            # Scale volatilities up
            scaled_volatilities = volatilities * 1.5
            
            # Increase correlations
            scaled_correlations = np.minimum(correlation_matrix * 1.3, 0.95)
            np.fill_diagonal(scaled_correlations, 1.0)
            
            # Reconstruct covariance matrix
            vol_outer = np.outer(scaled_volatilities, scaled_volatilities)
            adjusted_cov = vol_outer * scaled_correlations
            
        elif regime == CorrelationRegime.ELEVATED:
            # Moderate adjustments
            volatilities = np.sqrt(np.diag(cov_matrix))
            correlation_matrix = self._cov_to_corr(cov_matrix)
            
            scaled_volatilities = volatilities * 1.2
            scaled_correlations = np.minimum(correlation_matrix * 1.1, 0.9)
            np.fill_diagonal(scaled_correlations, 1.0)
            
            vol_outer = np.outer(scaled_volatilities, scaled_volatilities)
            adjusted_cov = vol_outer * scaled_correlations
        
        return adjusted_cov
    
    def _inverse_volatility_weights(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Calculate inverse volatility weights"""
        volatilities = np.sqrt(np.diag(cov_matrix))
        inv_vol = 1.0 / volatilities
        return inv_vol / np.sum(inv_vol)
    
    def _risk_budget_fallback_weights(
        self,
        risk_budgets: List[RiskBudget],
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """Fallback weights for risk budgeting optimization failure"""
        # Use target risk contributions as proxy weights
        weights = np.zeros(self.n_strategies)
        
        for i in range(self.n_strategies):
            budget = next((rb for rb in risk_budgets if rb.strategy_id == str(i)), None)
            if budget:
                weights[i] = budget.target_risk_contribution
            else:
                weights[i] = 1.0 / self.n_strategies
        
        # Normalize
        weights /= np.sum(weights)
        return weights
    
    def _fallback_equal_weights_result(
        self,
        cov_matrix: np.ndarray,
        start_time: datetime
    ) -> RiskParityResult:
        """Create fallback result with equal weights"""
        
        equal_weights = np.array([1.0 / self.n_strategies] * self.n_strategies)
        risk_contributions = self._calculate_risk_contributions(equal_weights, cov_matrix)
        
        portfolio_variance = np.dot(equal_weights, np.dot(cov_matrix, equal_weights))
        total_portfolio_risk = np.sqrt(max(portfolio_variance, 1e-12))
        
        optimization_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return RiskParityResult(
            timestamp=datetime.now(),
            method=RiskParityMethod.EQUAL_RISK_CONTRIBUTION,
            weights=equal_weights,
            risk_contributions=risk_contributions,
            total_portfolio_risk=total_portfolio_risk,
            risk_concentration_index=1.0 / self.n_strategies,
            diversification_ratio=1.0,
            effective_strategies=self.n_strategies,
            convergence_iterations=0,
            convergence_error=1.0,
            optimization_time_ms=optimization_time
        )
    
    def get_risk_parity_summary(self) -> Dict:
        """Get risk parity engine performance summary"""
        if not self.optimization_history:
            return {"status": "No risk parity optimizations performed"}
        
        latest_result = self.optimization_history[-1]
        
        return {
            "total_optimizations": len(self.optimization_history),
            "avg_optimization_time_ms": np.mean(self.calculation_times) if self.calculation_times else 0,
            "latest_result": {
                "method": latest_result.method.value,
                "effective_strategies": latest_result.effective_strategies,
                "risk_concentration_index": latest_result.risk_concentration_index,
                "diversification_ratio": latest_result.diversification_ratio,
                "convergence_iterations": latest_result.convergence_iterations,
                "convergence_error": latest_result.convergence_error,
                "optimization_time_ms": latest_result.optimization_time_ms
            },
            "risk_contributions": [
                {
                    "strategy_id": rc.strategy_id,
                    "weight": rc.weight,
                    "risk_contribution_pct": rc.risk_contribution_pct
                }
                for rc in latest_result.risk_contributions
            ]
        }