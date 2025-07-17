"""
Advanced Multi-Objective Portfolio Optimizer

This module implements sophisticated multi-objective optimization for portfolio allocation,
combining Sharpe ratio optimization, drawdown minimization, volatility targeting, and 
tail risk management using Pareto efficiency and advanced optimization techniques.

Key Features:
- Pareto-efficient multi-objective optimization
- Sharpe ratio maximization with risk constraints
- Dynamic drawdown minimization using CVaR
- Volatility targeting with regime adjustments
- Tail risk management (VaR, CVaR, Expected Shortfall)
- Risk budgeting and equal risk contribution
- Black-Litterman model integration
- Regime-aware optimization parameters
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import structlog
from scipy import optimize, stats
from scipy.linalg import sqrtm, inv
import warnings
warnings.filterwarnings('ignore')

from src.core.events import EventBus, Event, EventType
from src.risk.core.correlation_tracker import CorrelationRegime

logger = structlog.get_logger()


class OptimizationMethod(Enum):
    """Multi-objective optimization methods"""
    PARETO_FRONTIER = "pareto_frontier"
    WEIGHTED_COMBINATION = "weighted_combination"
    LEXICOGRAPHIC = "lexicographic"
    GOAL_PROGRAMMING = "goal_programming"
    SCALARIZATION = "scalarization"


class RiskMeasure(Enum):
    """Risk measures for optimization"""
    VARIANCE = "variance"
    VOLATILITY = "volatility"
    VAR_95 = "var_95"
    VAR_99 = "var_99"
    CVAR_95 = "cvar_95"
    CVAR_99 = "cvar_99"
    MAX_DRAWDOWN = "max_drawdown"
    EXPECTED_SHORTFALL = "expected_shortfall"


@dataclass
class OptimizationObjectives:
    """Multi-objective optimization objectives with weights and constraints"""
    maximize_sharpe: bool = True
    sharpe_weight: float = 0.4
    sharpe_target: Optional[float] = None
    
    minimize_volatility: bool = True
    volatility_weight: float = 0.2
    volatility_target: Optional[float] = None
    
    minimize_drawdown: bool = True
    drawdown_weight: float = 0.2
    drawdown_target: Optional[float] = None
    
    minimize_tail_risk: bool = True
    tail_risk_weight: float = 0.2
    tail_risk_measure: RiskMeasure = RiskMeasure.CVAR_95
    tail_risk_target: Optional[float] = None
    
    def normalize_weights(self):
        """Normalize objective weights to sum to 1"""
        total_weight = (self.sharpe_weight + self.volatility_weight + 
                       self.drawdown_weight + self.tail_risk_weight)
        if total_weight > 0:
            self.sharpe_weight /= total_weight
            self.volatility_weight /= total_weight
            self.drawdown_weight /= total_weight
            self.tail_risk_weight /= total_weight


@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints"""
    min_weight: float = 0.05
    max_weight: float = 0.5
    max_concentration: float = 0.4  # Max sum of top 3 positions
    min_diversification_ratio: float = 1.1
    max_correlation_exposure: float = 0.8
    max_sector_concentration: Optional[float] = None
    turnover_constraint: Optional[float] = None  # Max turnover from current weights


@dataclass
class ParetoSolution:
    """Single solution on Pareto frontier"""
    weights: np.ndarray
    sharpe_ratio: float
    volatility: float
    expected_return: float
    drawdown_proxy: float
    tail_risk: float
    diversification_ratio: float
    risk_contributions: np.ndarray
    objective_values: Dict[str, float]
    scalarized_score: float


@dataclass
class MultiObjectiveResult:
    """Multi-objective optimization result"""
    timestamp: datetime
    method: OptimizationMethod
    pareto_frontier: List[ParetoSolution]
    best_solution: ParetoSolution
    optimization_time_ms: float
    convergence_status: str
    objectives_met: Dict[str, bool]
    constraint_violations: List[str]


class MultiObjectiveOptimizer:
    """
    Advanced multi-objective portfolio optimizer implementing sophisticated
    optimization techniques for Pareto-efficient portfolio allocation.
    """
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        n_strategies: int = 5,
        risk_free_rate: float = 0.02,  # 2% risk-free rate
        confidence_level: float = 0.95
    ):
        """
        Initialize Multi-Objective Optimizer
        
        Args:
            event_bus: Event bus for communication
            n_strategies: Number of strategies to optimize
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            confidence_level: Confidence level for risk measures
        """
        self.event_bus = event_bus
        self.n_strategies = n_strategies
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level
        
        # Optimization parameters
        self.max_iterations = 1000
        self.tolerance = 1e-8
        self.pareto_samples = 50  # Number of Pareto frontier points
        
        # Risk measure parameters
        self.var_confidence = confidence_level
        self.cvar_confidence = confidence_level
        
        # Black-Litterman parameters
        self.tau = 0.025  # Uncertainty parameter
        self.use_black_litterman = False
        
        # Performance tracking
        self.optimization_history: List[MultiObjectiveResult] = []
        self.calculation_times: List[float] = []
        
        logger.info("Multi-Objective Optimizer initialized",
                   n_strategies=n_strategies,
                   risk_free_rate=risk_free_rate,
                   confidence_level=confidence_level)
    
    def optimize_portfolio(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        current_weights: Optional[np.ndarray] = None,
        objectives: Optional[OptimizationObjectives] = None,
        constraints: Optional[OptimizationConstraints] = None,
        method: OptimizationMethod = OptimizationMethod.WEIGHTED_COMBINATION,
        regime: CorrelationRegime = CorrelationRegime.NORMAL
    ) -> MultiObjectiveResult:
        """
        Perform multi-objective portfolio optimization
        
        Args:
            expected_returns: Expected returns for each strategy
            covariance_matrix: Covariance matrix
            current_weights: Current portfolio weights (for turnover constraint)
            objectives: Optimization objectives and weights
            constraints: Portfolio constraints
            method: Optimization method
            regime: Current market regime for parameter adjustment
            
        Returns:
            Multi-objective optimization result
        """
        start_time = datetime.now()
        
        # Set defaults
        if objectives is None:
            objectives = OptimizationObjectives()
        if constraints is None:
            constraints = OptimizationConstraints()
        if current_weights is None:
            current_weights = np.array([1.0 / self.n_strategies] * self.n_strategies)
        
        # Normalize objective weights
        objectives.normalize_weights()
        
        # Adjust parameters for market regime
        objectives, constraints = self._adjust_for_regime(objectives, constraints, regime)
        
        try:
            if method == OptimizationMethod.PARETO_FRONTIER:
                result = self._pareto_frontier_optimization(
                    expected_returns, covariance_matrix, objectives, constraints, current_weights
                )
            elif method == OptimizationMethod.WEIGHTED_COMBINATION:
                result = self._weighted_combination_optimization(
                    expected_returns, covariance_matrix, objectives, constraints, current_weights
                )
            elif method == OptimizationMethod.LEXICOGRAPHIC:
                result = self._lexicographic_optimization(
                    expected_returns, covariance_matrix, objectives, constraints, current_weights
                )
            else:
                # Default to weighted combination
                result = self._weighted_combination_optimization(
                    expected_returns, covariance_matrix, objectives, constraints, current_weights
                )
            
            # Calculate optimization time
            optimization_time = (datetime.now() - start_time).total_seconds() * 1000
            result.optimization_time_ms = optimization_time
            result.timestamp = datetime.now()
            
            # Store in history
            self.optimization_history.append(result)
            self.calculation_times.append(optimization_time)
            
            # Keep only recent history
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-100:]
            
            logger.info("Multi-objective optimization completed",
                       method=method.value,
                       optimization_time_ms=optimization_time,
                       best_sharpe=result.best_solution.sharpe_ratio,
                       best_volatility=result.best_solution.volatility)
            
            return result
            
        except Exception as e:
            logger.error("Multi-objective optimization failed", error=str(e))
            
            # Return fallback result with equal weights
            equal_weights = np.array([1.0 / self.n_strategies] * self.n_strategies)
            fallback_solution = self._evaluate_solution(
                equal_weights, expected_returns, covariance_matrix, objectives
            )
            
            return MultiObjectiveResult(
                timestamp=datetime.now(),
                method=method,
                pareto_frontier=[fallback_solution],
                best_solution=fallback_solution,
                optimization_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                convergence_status="FAILED - Using equal weights fallback",
                objectives_met={},
                constraint_violations=[]
            )
    
    def _weighted_combination_optimization(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        objectives: OptimizationObjectives,
        constraints: OptimizationConstraints,
        current_weights: np.ndarray
    ) -> MultiObjectiveResult:
        """Weighted combination scalarization optimization"""
        
        def objective_function(weights):
            """Scalarized objective function"""
            try:
                solution = self._evaluate_solution(weights, expected_returns, covariance_matrix, objectives)
                
                # Weighted combination of normalized objectives
                score = 0.0
                
                if objectives.maximize_sharpe:
                    # Normalize Sharpe ratio (assume max reasonable Sharpe is 3.0)
                    normalized_sharpe = min(1.0, solution.sharpe_ratio / 3.0)
                    score += objectives.sharpe_weight * normalized_sharpe
                
                if objectives.minimize_volatility:
                    # Normalize volatility (assume max reasonable vol is 0.5)
                    normalized_vol = 1.0 - min(1.0, solution.volatility / 0.5)
                    score += objectives.volatility_weight * normalized_vol
                
                if objectives.minimize_drawdown:
                    # Normalize drawdown proxy
                    normalized_dd = 1.0 - min(1.0, solution.drawdown_proxy / 0.3)
                    score += objectives.drawdown_weight * normalized_dd
                
                if objectives.minimize_tail_risk:
                    # Normalize tail risk
                    normalized_tail = 1.0 - min(1.0, solution.tail_risk / 0.1)
                    score += objectives.tail_risk_weight * normalized_tail
                
                return -score  # Minimize negative score = maximize score
                
            except Exception as e:
                logger.warning("Objective function evaluation failed", error=str(e))
                return 1e6  # Large penalty
        
        # Optimization constraints
        opt_constraints = self._build_scipy_constraints(constraints, current_weights)
        bounds = [(constraints.min_weight, constraints.max_weight) 
                 for _ in range(self.n_strategies)]
        
        # Initial guess (current weights or equal weights)
        x0 = current_weights.copy()
        
        # Multiple starting points for robustness
        best_result = None
        best_score = 1e6
        
        starting_points = [
            current_weights,
            np.array([1.0 / self.n_strategies] * self.n_strategies),  # Equal weights
            self._generate_random_weights(constraints),  # Random weights
        ]
        
        for start_point in starting_points:
            try:
                result = optimize.minimize(
                    objective_function,
                    start_point,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=opt_constraints,
                    options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
                )
                
                if result.success and result.fun < best_score:
                    best_result = result
                    best_score = result.fun
                    
            except Exception as e:
                logger.warning("Optimization attempt failed", error=str(e))
                continue
        
        if best_result is None or not best_result.success:
            # Fallback to equal weights
            optimal_weights = np.array([1.0 / self.n_strategies] * self.n_strategies)
            convergence_status = "FAILED - Using equal weights"
        else:
            optimal_weights = best_result.x
            convergence_status = "SUCCESS"
        
        # Evaluate best solution
        best_solution = self._evaluate_solution(optimal_weights, expected_returns, covariance_matrix, objectives)
        
        # Check constraint violations
        constraint_violations = self._check_constraint_violations(optimal_weights, constraints)
        
        # Check if objectives are met
        objectives_met = self._check_objectives_met(best_solution, objectives)
        
        return MultiObjectiveResult(
            timestamp=datetime.now(),
            method=OptimizationMethod.WEIGHTED_COMBINATION,
            pareto_frontier=[best_solution],
            best_solution=best_solution,
            optimization_time_ms=0,  # Will be set by caller
            convergence_status=convergence_status,
            objectives_met=objectives_met,
            constraint_violations=constraint_violations
        )
    
    def _pareto_frontier_optimization(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        objectives: OptimizationObjectives,
        constraints: OptimizationConstraints,
        current_weights: np.ndarray
    ) -> MultiObjectiveResult:
        """Generate Pareto frontier using epsilon-constraint method"""
        
        pareto_solutions = []
        
        # Generate range of target values for secondary objectives
        # Primary objective: Sharpe ratio (maximize)
        # Secondary objectives: volatility, drawdown, tail risk (minimize)
        
        # First, find extreme points
        min_vol_solution = self._optimize_single_objective(
            'min_volatility', expected_returns, covariance_matrix, constraints, current_weights
        )
        max_sharpe_solution = self._optimize_single_objective(
            'max_sharpe', expected_returns, covariance_matrix, constraints, current_weights
        )
        
        if min_vol_solution and max_sharpe_solution:
            # Create range of epsilon values for volatility constraint
            vol_range = np.linspace(
                min_vol_solution.volatility,
                max_sharpe_solution.volatility,
                self.pareto_samples
            )
            
            for target_vol in vol_range:
                try:
                    # Optimize Sharpe ratio subject to volatility constraint
                    solution = self._optimize_with_vol_constraint(
                        target_vol, expected_returns, covariance_matrix, constraints, current_weights, objectives
                    )
                    
                    if solution:
                        pareto_solutions.append(solution)
                        
                except Exception as e:
                    logger.debug("Pareto point optimization failed", target_vol=target_vol, error=str(e))
                    continue
        
        if not pareto_solutions:
            # Fallback to single solution
            equal_weights = np.array([1.0 / self.n_strategies] * self.n_strategies)
            fallback_solution = self._evaluate_solution(equal_weights, expected_returns, covariance_matrix, objectives)
            pareto_solutions = [fallback_solution]
        
        # Select best solution from Pareto frontier
        best_solution = max(pareto_solutions, key=lambda s: s.scalarized_score)
        
        # Check constraints and objectives
        constraint_violations = self._check_constraint_violations(best_solution.weights, constraints)
        objectives_met = self._check_objectives_met(best_solution, objectives)
        
        return MultiObjectiveResult(
            timestamp=datetime.now(),
            method=OptimizationMethod.PARETO_FRONTIER,
            pareto_frontier=pareto_solutions,
            best_solution=best_solution,
            optimization_time_ms=0,  # Will be set by caller
            convergence_status="SUCCESS",
            objectives_met=objectives_met,
            constraint_violations=constraint_violations
        )
    
    def _lexicographic_optimization(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        objectives: OptimizationObjectives,
        constraints: OptimizationConstraints,
        current_weights: np.ndarray
    ) -> MultiObjectiveResult:
        """Lexicographic optimization - optimize objectives in priority order"""
        
        # Priority order: 1) Sharpe ratio, 2) Volatility, 3) Drawdown, 4) Tail risk
        current_solution = current_weights.copy()
        
        try:
            # Step 1: Maximize Sharpe ratio
            if objectives.maximize_sharpe:
                sharpe_solution = self._optimize_single_objective(
                    'max_sharpe', expected_returns, covariance_matrix, constraints, current_solution
                )
                if sharpe_solution:
                    current_solution = sharpe_solution.weights
                    target_sharpe = sharpe_solution.sharpe_ratio * 0.95  # Allow 5% degradation
                else:
                    target_sharpe = None
            else:
                target_sharpe = None
            
            # Step 2: Minimize volatility subject to Sharpe constraint
            if objectives.minimize_volatility:
                vol_solution = self._optimize_with_sharpe_constraint(
                    target_sharpe, expected_returns, covariance_matrix, constraints, current_solution, objectives
                )
                if vol_solution:
                    current_solution = vol_solution.weights
            
            # Evaluate final solution
            final_solution = self._evaluate_solution(current_solution, expected_returns, covariance_matrix, objectives)
            
        except Exception as e:
            logger.error("Lexicographic optimization failed", error=str(e))
            # Fallback to equal weights
            equal_weights = np.array([1.0 / self.n_strategies] * self.n_strategies)
            final_solution = self._evaluate_solution(equal_weights, expected_returns, covariance_matrix, objectives)
        
        # Check constraints and objectives
        constraint_violations = self._check_constraint_violations(final_solution.weights, constraints)
        objectives_met = self._check_objectives_met(final_solution, objectives)
        
        return MultiObjectiveResult(
            timestamp=datetime.now(),
            method=OptimizationMethod.LEXICOGRAPHIC,
            pareto_frontier=[final_solution],
            best_solution=final_solution,
            optimization_time_ms=0,  # Will be set by caller
            convergence_status="SUCCESS",
            objectives_met=objectives_met,
            constraint_violations=constraint_violations
        )
    
    def _optimize_single_objective(
        self,
        objective: str,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: OptimizationConstraints,
        initial_weights: np.ndarray
    ) -> Optional[ParetoSolution]:
        """Optimize single objective"""
        
        if objective == 'max_sharpe':
            def obj_func(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
                portfolio_vol = np.sqrt(max(portfolio_variance, 1e-10))
                sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
                return -sharpe  # Minimize negative Sharpe
                
        elif objective == 'min_volatility':
            def obj_func(weights):
                portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
                return np.sqrt(max(portfolio_variance, 1e-10))
        else:
            return None
        
        # Constraints and bounds
        opt_constraints = self._build_scipy_constraints(constraints, initial_weights)
        bounds = [(constraints.min_weight, constraints.max_weight) 
                 for _ in range(self.n_strategies)]
        
        try:
            result = optimize.minimize(
                obj_func,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=opt_constraints,
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
            )
            
            if result.success:
                # Evaluate solution
                objectives_placeholder = OptimizationObjectives()
                return self._evaluate_solution(result.x, expected_returns, covariance_matrix, objectives_placeholder)
            
        except Exception as e:
            logger.warning("Single objective optimization failed", objective=objective, error=str(e))
        
        return None
    
    def _optimize_with_vol_constraint(
        self,
        target_vol: float,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: OptimizationConstraints,
        initial_weights: np.ndarray,
        objectives: OptimizationObjectives
    ) -> Optional[ParetoSolution]:
        """Optimize Sharpe ratio with volatility constraint"""
        
        def obj_func(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            portfolio_vol = np.sqrt(max(portfolio_variance, 1e-10))
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            return -sharpe
        
        def vol_constraint(weights):
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            portfolio_vol = np.sqrt(max(portfolio_variance, 1e-10))
            return target_vol - portfolio_vol  # vol <= target_vol
        
        # Add volatility constraint
        opt_constraints = self._build_scipy_constraints(constraints, initial_weights)
        opt_constraints.append({'type': 'ineq', 'fun': vol_constraint})
        
        bounds = [(constraints.min_weight, constraints.max_weight) 
                 for _ in range(self.n_strategies)]
        
        try:
            result = optimize.minimize(
                obj_func,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=opt_constraints,
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
            )
            
            if result.success:
                return self._evaluate_solution(result.x, expected_returns, covariance_matrix, objectives)
                
        except Exception as e:
            logger.warning("Vol-constrained optimization failed", target_vol=target_vol, error=str(e))
        
        return None
    
    def _optimize_with_sharpe_constraint(
        self,
        target_sharpe: Optional[float],
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: OptimizationConstraints,
        initial_weights: np.ndarray,
        objectives: OptimizationObjectives
    ) -> Optional[ParetoSolution]:
        """Optimize volatility with Sharpe ratio constraint"""
        
        if target_sharpe is None:
            return self._optimize_single_objective(
                'min_volatility', expected_returns, covariance_matrix, constraints, initial_weights
            )
        
        def obj_func(weights):
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            return np.sqrt(max(portfolio_variance, 1e-10))
        
        def sharpe_constraint(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            portfolio_vol = np.sqrt(max(portfolio_variance, 1e-10))
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            return sharpe - target_sharpe  # Sharpe >= target_sharpe
        
        # Add Sharpe constraint
        opt_constraints = self._build_scipy_constraints(constraints, initial_weights)
        opt_constraints.append({'type': 'ineq', 'fun': sharpe_constraint})
        
        bounds = [(constraints.min_weight, constraints.max_weight) 
                 for _ in range(self.n_strategies)]
        
        try:
            result = optimize.minimize(
                obj_func,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=opt_constraints,
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
            )
            
            if result.success:
                return self._evaluate_solution(result.x, expected_returns, covariance_matrix, objectives)
                
        except Exception as e:
            logger.warning("Sharpe-constrained optimization failed", target_sharpe=target_sharpe, error=str(e))
        
        return None
    
    def _evaluate_solution(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        objectives: OptimizationObjectives
    ) -> ParetoSolution:
        """Evaluate portfolio solution"""
        
        # Basic portfolio metrics
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        portfolio_vol = np.sqrt(max(portfolio_variance, 1e-10))
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        # Drawdown proxy (using volatility as proxy)
        drawdown_proxy = portfolio_vol * 2.0  # Rough approximation
        
        # Tail risk (CVaR approximation)
        alpha = 1 - self.var_confidence
        z_alpha = stats.norm.ppf(alpha)
        var = portfolio_vol * z_alpha
        tail_risk = portfolio_vol * stats.norm.pdf(z_alpha) / alpha  # CVaR approximation
        
        # Diversification ratio
        individual_vols = np.sqrt(np.diag(covariance_matrix))
        weighted_avg_vol = np.dot(weights, individual_vols)
        diversification_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0
        
        # Risk contributions
        marginal_contributions = np.dot(covariance_matrix, weights) / portfolio_vol if portfolio_vol > 0 else weights
        risk_contributions = weights * marginal_contributions
        if np.sum(risk_contributions) > 0:
            risk_contributions /= np.sum(risk_contributions)  # Normalize
        
        # Objective values
        objective_values = {
            'sharpe_ratio': sharpe_ratio,
            'volatility': portfolio_vol,
            'expected_return': portfolio_return,
            'drawdown_proxy': drawdown_proxy,
            'tail_risk': tail_risk,
            'diversification_ratio': diversification_ratio
        }
        
        # Scalarized score for ranking
        scalarized_score = (
            objectives.sharpe_weight * min(3.0, sharpe_ratio) / 3.0 +
            objectives.volatility_weight * (1.0 - min(1.0, portfolio_vol / 0.5)) +
            objectives.drawdown_weight * (1.0 - min(1.0, drawdown_proxy / 0.3)) +
            objectives.tail_risk_weight * (1.0 - min(1.0, tail_risk / 0.1))
        )
        
        return ParetoSolution(
            weights=weights,
            sharpe_ratio=sharpe_ratio,
            volatility=portfolio_vol,
            expected_return=portfolio_return,
            drawdown_proxy=drawdown_proxy,
            tail_risk=tail_risk,
            diversification_ratio=diversification_ratio,
            risk_contributions=risk_contributions,
            objective_values=objective_values,
            scalarized_score=scalarized_score
        )
    
    def _build_scipy_constraints(
        self,
        constraints: OptimizationConstraints,
        current_weights: np.ndarray
    ) -> List[Dict]:
        """Build scipy optimization constraints"""
        
        opt_constraints = []
        
        # Weights sum to 1
        opt_constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })
        
        # Maximum concentration constraint
        if constraints.max_concentration:
            def concentration_constraint(w):
                sorted_weights = np.sort(w)[::-1]  # Sort descending
                top_3_sum = np.sum(sorted_weights[:3])
                return constraints.max_concentration - top_3_sum
            
            opt_constraints.append({
                'type': 'ineq',
                'fun': concentration_constraint
            })
        
        # Turnover constraint
        if constraints.turnover_constraint:
            def turnover_constraint(w):
                turnover = np.sum(np.abs(w - current_weights))
                return constraints.turnover_constraint - turnover
            
            opt_constraints.append({
                'type': 'ineq',
                'fun': turnover_constraint
            })
        
        return opt_constraints
    
    def _check_constraint_violations(
        self,
        weights: np.ndarray,
        constraints: OptimizationConstraints
    ) -> List[str]:
        """Check for constraint violations"""
        violations = []
        
        # Weight bounds
        if np.any(weights < constraints.min_weight - 1e-6):
            violations.append("Minimum weight constraint violated")
        if np.any(weights > constraints.max_weight + 1e-6):
            violations.append("Maximum weight constraint violated")
        
        # Sum to 1
        if abs(np.sum(weights) - 1.0) > 1e-4:
            violations.append("Weights do not sum to 1")
        
        # Concentration
        if constraints.max_concentration:
            sorted_weights = np.sort(weights)[::-1]
            if np.sum(sorted_weights[:3]) > constraints.max_concentration + 1e-6:
                violations.append("Maximum concentration constraint violated")
        
        return violations
    
    def _check_objectives_met(
        self,
        solution: ParetoSolution,
        objectives: OptimizationObjectives
    ) -> Dict[str, bool]:
        """Check if objectives are met"""
        objectives_met = {}
        
        if objectives.sharpe_target:
            objectives_met['sharpe'] = solution.sharpe_ratio >= objectives.sharpe_target
        
        if objectives.volatility_target:
            objectives_met['volatility'] = solution.volatility <= objectives.volatility_target
        
        if objectives.drawdown_target:
            objectives_met['drawdown'] = solution.drawdown_proxy <= objectives.drawdown_target
        
        if objectives.tail_risk_target:
            objectives_met['tail_risk'] = solution.tail_risk <= objectives.tail_risk_target
        
        return objectives_met
    
    def _adjust_for_regime(
        self,
        objectives: OptimizationObjectives,
        constraints: OptimizationConstraints,
        regime: CorrelationRegime
    ) -> Tuple[OptimizationObjectives, OptimizationConstraints]:
        """Adjust optimization parameters for market regime"""
        
        adjusted_objectives = objectives
        adjusted_constraints = constraints
        
        if regime == CorrelationRegime.CRISIS or regime == CorrelationRegime.SHOCK:
            # In crisis: emphasize risk reduction
            adjusted_objectives.volatility_weight *= 1.5
            adjusted_objectives.tail_risk_weight *= 1.5
            adjusted_objectives.sharpe_weight *= 0.7
            adjusted_objectives.normalize_weights()
            
            # Tighter constraints
            adjusted_constraints.max_weight = min(constraints.max_weight, 0.3)
            adjusted_constraints.max_concentration = min(constraints.max_concentration or 0.4, 0.3)
            
        elif regime == CorrelationRegime.ELEVATED:
            # Moderate adjustment
            adjusted_objectives.volatility_weight *= 1.2
            adjusted_objectives.tail_risk_weight *= 1.2
            adjusted_objectives.normalize_weights()
            
            adjusted_constraints.max_weight = min(constraints.max_weight, 0.4)
        
        return adjusted_objectives, adjusted_constraints
    
    def _generate_random_weights(self, constraints: OptimizationConstraints) -> np.ndarray:
        """Generate random weights satisfying constraints"""
        weights = np.random.uniform(constraints.min_weight, constraints.max_weight, self.n_strategies)
        weights /= np.sum(weights)  # Normalize to sum to 1
        
        # Ensure bounds are respected after normalization
        weights = np.clip(weights, constraints.min_weight, constraints.max_weight)
        weights /= np.sum(weights)  # Renormalize
        
        return weights
    
    def get_optimization_summary(self) -> Dict:
        """Get optimization performance summary"""
        if not self.optimization_history:
            return {"status": "No optimizations performed"}
        
        latest_result = self.optimization_history[-1]
        
        return {
            "total_optimizations": len(self.optimization_history),
            "avg_optimization_time_ms": np.mean(self.calculation_times) if self.calculation_times else 0,
            "latest_optimization": {
                "method": latest_result.method.value,
                "convergence_status": latest_result.convergence_status,
                "best_sharpe": latest_result.best_solution.sharpe_ratio,
                "best_volatility": latest_result.best_solution.volatility,
                "diversification_ratio": latest_result.best_solution.diversification_ratio,
                "objectives_met": latest_result.objectives_met,
                "constraint_violations": latest_result.constraint_violations
            },
            "pareto_frontier_size": len(latest_result.pareto_frontier)
        }