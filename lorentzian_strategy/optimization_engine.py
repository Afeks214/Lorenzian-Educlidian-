"""
Advanced Parameter Optimization Engine
=====================================

Multi-parameter optimization with:
- Walk-forward analysis
- Out-of-sample validation
- Robust optimization techniques
- Genetic algorithm optimization
- Bayesian optimization
- Performance target validation

Author: Claude Code
Date: 2025-07-20
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import logging
from datetime import datetime, timedelta
import json
import pickle
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Scientific computing
from scipy.optimize import minimize, differential_evolution
from scipy.stats import rankdata
import itertools
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Machine learning
try:
    from sklearn.model_selection import ParameterGrid
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    ADVANCED_OPTIMIZATION = True
except ImportError:
    ADVANCED_OPTIMIZATION = False
    logging.warning("Advanced optimization libraries not available. Using basic optimization only.")

# Add project paths
project_root = Path("/home/QuantNova/GrandModel")
sys.path.append(str(project_root))

from lorentzian_strategy.backtesting.vectorbt_framework import VectorBTFramework, BacktestConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for optimization parameters"""
    # Optimization method
    method: str = "genetic"  # "grid", "random", "genetic", "bayesian"
    
    # Search constraints
    max_iterations: int = 1000
    population_size: int = 50
    convergence_tolerance: float = 1e-6
    
    # Objective function
    primary_metric: str = "sharpe_ratio"
    secondary_metrics: List[str] = field(default_factory=lambda: ["max_drawdown", "win_rate"])
    
    # Multi-objective weights
    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        "sharpe_ratio": 0.4,
        "max_drawdown": 0.3,
        "win_rate": 0.2,
        "profit_factor": 0.1
    })
    
    # Walk-forward settings
    use_walk_forward: bool = True
    training_window_months: int = 12
    validation_window_months: int = 3
    step_months: int = 1
    
    # Out-of-sample validation
    out_of_sample_pct: float = 0.2  # 20% for out-of-sample testing
    
    # Robustness testing
    noise_levels: List[float] = field(default_factory=lambda: [0.01, 0.02, 0.05])
    monte_carlo_iterations: int = 100
    
    # Parallel processing
    use_parallel: bool = True
    max_workers: int = None  # Will use available CPUs

@dataclass
class ParameterSpace:
    """Define parameter search space"""
    # Trading parameters
    fast_period: Tuple[int, int] = (5, 30)
    slow_period: Tuple[int, int] = (15, 100)
    rsi_period: Tuple[int, int] = (10, 30)
    rsi_overbought: Tuple[float, float] = (60.0, 85.0)
    rsi_oversold: Tuple[float, float] = (15.0, 40.0)
    
    # Risk management
    stop_loss_pct: Tuple[float, float] = (0.005, 0.05)
    take_profit_pct: Tuple[float, float] = (0.01, 0.10)
    max_position_size: Tuple[float, float] = (0.10, 0.50)
    
    # Lorentzian parameters
    lookback_periods: Tuple[int, int] = (50, 200)
    neighbors: Tuple[int, int] = (5, 20)
    sigma: Tuple[float, float] = (0.5, 2.0)
    
    def to_dict(self) -> Dict[str, Tuple]:
        """Convert to dictionary format"""
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'rsi_period': self.rsi_period,
            'rsi_overbought': self.rsi_overbought,
            'rsi_oversold': self.rsi_oversold,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'max_position_size': self.max_position_size,
            'lookback_periods': self.lookback_periods,
            'neighbors': self.neighbors,
            'sigma': self.sigma
        }

class OptimizationObjective:
    """Objective function for optimization"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.evaluation_count = 0
        self.best_score = -np.inf
        self.evaluation_history = []
    
    def evaluate(self, parameters: Dict[str, Any], 
                framework: VectorBTFramework) -> float:
        """Evaluate parameter set and return fitness score"""
        try:
            self.evaluation_count += 1
            
            # Update framework configuration
            for key, value in parameters.items():
                if hasattr(framework.config, key):
                    setattr(framework.config, key, value)
            
            # Run backtest
            framework.generate_signals()
            portfolio = framework.run_backtest()
            metrics = framework.calculate_performance_metrics(portfolio)
            
            # Calculate composite score
            score = self._calculate_composite_score(metrics)
            
            # Store evaluation
            evaluation = {
                'parameters': parameters.copy(),
                'metrics': metrics.copy(),
                'score': score,
                'evaluation_id': self.evaluation_count
            }
            self.evaluation_history.append(evaluation)
            
            # Update best score
            if score > self.best_score:
                self.best_score = score
                logger.info(f"New best score: {score:.4f} (evaluation {self.evaluation_count})")
            
            return score
            
        except Exception as e:
            logger.warning(f"Evaluation failed for parameters {parameters}: {e}")
            return -np.inf
    
    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """Calculate composite score from multiple metrics"""
        score = 0.0
        
        for metric, weight in self.config.metric_weights.items():
            if metric in metrics:
                value = metrics[metric]
                
                # Normalize metrics (higher is better)
                if metric == "max_drawdown":
                    # Convert drawdown to positive score (less drawdown = higher score)
                    normalized_value = max(0, 1 + value)  # value is negative
                elif metric in ["sharpe_ratio", "win_rate", "profit_factor"]:
                    # These metrics are already positive-oriented
                    normalized_value = max(0, value)
                else:
                    normalized_value = max(0, value)
                
                score += weight * normalized_value
        
        # Apply penalties for extreme values
        score = self._apply_penalties(score, metrics)
        
        return score
    
    def _apply_penalties(self, score: float, metrics: Dict[str, float]) -> float:
        """Apply penalties for undesirable characteristics"""
        # Penalty for excessive drawdown
        if abs(metrics.get('max_drawdown', 0)) > 0.25:  # >25% drawdown
            score *= 0.5
        
        # Penalty for negative Sharpe ratio
        if metrics.get('sharpe_ratio', 0) < 0:
            score *= 0.1
        
        # Penalty for very low win rate
        if metrics.get('win_rate', 0) < 0.3:  # <30% win rate
            score *= 0.7
        
        # Penalty for insufficient trades
        if metrics.get('total_trades', 0) < 10:
            score *= 0.8
        
        return score

class BaseOptimizer(ABC):
    """Base class for optimization algorithms"""
    
    def __init__(self, config: OptimizationConfig, parameter_space: ParameterSpace):
        self.config = config
        self.parameter_space = parameter_space
        self.objective = OptimizationObjective(config)
        self.results = {}
    
    @abstractmethod
    def optimize(self, framework: VectorBTFramework) -> Dict[str, Any]:
        """Run optimization algorithm"""
        pass
    
    def _generate_random_parameters(self) -> Dict[str, Any]:
        """Generate random parameters within bounds"""
        params = {}
        param_dict = self.parameter_space.to_dict()
        
        for param, (min_val, max_val) in param_dict.items():
            if isinstance(min_val, int):
                params[param] = random.randint(min_val, max_val)
            else:
                params[param] = random.uniform(min_val, max_val)
        
        return params

class GridSearchOptimizer(BaseOptimizer):
    """Grid search optimization"""
    
    def optimize(self, framework: VectorBTFramework) -> Dict[str, Any]:
        logger.info("Starting grid search optimization...")
        
        # Create parameter grid
        param_grid = self._create_parameter_grid()
        
        best_score = -np.inf
        best_params = None
        
        for i, params in enumerate(param_grid):
            if i >= self.config.max_iterations:
                break
                
            score = self.objective.evaluate(params, framework)
            
            if score > best_score:
                best_score = score
                best_params = params
            
            if (i + 1) % 50 == 0:
                logger.info(f"Grid search progress: {i+1}/{min(len(param_grid), self.config.max_iterations)}")
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'evaluation_history': self.objective.evaluation_history,
            'total_evaluations': self.objective.evaluation_count
        }
    
    def _create_parameter_grid(self) -> List[Dict[str, Any]]:
        """Create parameter grid for grid search"""
        param_dict = self.parameter_space.to_dict()
        
        # Create discrete values for each parameter
        grid_params = {}
        for param, (min_val, max_val) in param_dict.items():
            if isinstance(min_val, int):
                # Create 5 integer values
                grid_params[param] = np.linspace(min_val, max_val, 5, dtype=int).tolist()
            else:
                # Create 5 float values
                grid_params[param] = np.linspace(min_val, max_val, 5).tolist()
        
        # Generate all combinations
        param_grid = list(ParameterGrid(grid_params))
        
        # Shuffle for random order
        random.shuffle(param_grid)
        
        return param_grid

class RandomSearchOptimizer(BaseOptimizer):
    """Random search optimization"""
    
    def optimize(self, framework: VectorBTFramework) -> Dict[str, Any]:
        logger.info("Starting random search optimization...")
        
        best_score = -np.inf
        best_params = None
        
        for i in range(self.config.max_iterations):
            params = self._generate_random_parameters()
            score = self.objective.evaluate(params, framework)
            
            if score > best_score:
                best_score = score
                best_params = params
            
            if (i + 1) % 100 == 0:
                logger.info(f"Random search progress: {i+1}/{self.config.max_iterations}")
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'evaluation_history': self.objective.evaluation_history,
            'total_evaluations': self.objective.evaluation_count
        }

class GeneticOptimizer(BaseOptimizer):
    """Genetic algorithm optimization"""
    
    def optimize(self, framework: VectorBTFramework) -> Dict[str, Any]:
        logger.info("Starting genetic algorithm optimization...")
        
        param_dict = self.parameter_space.to_dict()
        
        # Create bounds for scipy differential evolution
        bounds = []
        param_names = []
        
        for param, (min_val, max_val) in param_dict.items():
            bounds.append((min_val, max_val))
            param_names.append(param)
        
        def objective_function(x):
            params = {param_names[i]: x[i] for i in range(len(x))}
            # Convert integer parameters
            for param in ['fast_period', 'slow_period', 'rsi_period', 'lookback_periods', 'neighbors']:
                if param in params:
                    params[param] = int(params[param])
            
            return -self.objective.evaluate(params, framework)  # Minimize (negative)
        
        # Run differential evolution
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=self.config.max_iterations // self.config.population_size,
            popsize=self.config.population_size,
            tol=self.config.convergence_tolerance,
            seed=42
        )
        
        # Convert result back to parameters
        best_params = {param_names[i]: result.x[i] for i in range(len(result.x))}
        for param in ['fast_period', 'slow_period', 'rsi_period', 'lookback_periods', 'neighbors']:
            if param in best_params:
                best_params[param] = int(best_params[param])
        
        return {
            'best_parameters': best_params,
            'best_score': -result.fun,
            'evaluation_history': self.objective.evaluation_history,
            'total_evaluations': self.objective.evaluation_count,
            'convergence_info': {
                'success': result.success,
                'message': result.message,
                'iterations': result.nit
            }
        }

class BayesianOptimizer(BaseOptimizer):
    """Bayesian optimization using Gaussian processes"""
    
    def optimize(self, framework: VectorBTFramework) -> Dict[str, Any]:
        if not ADVANCED_OPTIMIZATION:
            logger.warning("Bayesian optimization not available. Falling back to genetic algorithm.")
            return GeneticOptimizer(self.config, self.parameter_space).optimize(framework)
        
        logger.info("Starting Bayesian optimization...")
        
        param_dict = self.parameter_space.to_dict()
        
        # Create search space for skopt
        dimensions = []
        param_names = []
        
        for param, (min_val, max_val) in param_dict.items():
            param_names.append(param)
            if isinstance(min_val, int):
                dimensions.append(Integer(min_val, max_val, name=param))
            else:
                dimensions.append(Real(min_val, max_val, name=param))
        
        @use_named_args(dimensions)
        def objective_function(**params):
            return -self.objective.evaluate(params, framework)  # Minimize (negative)
        
        # Run Bayesian optimization
        result = gp_minimize(
            objective_function,
            dimensions,
            n_calls=self.config.max_iterations,
            n_initial_points=min(20, self.config.max_iterations // 5),
            random_state=42
        )
        
        # Convert result back to parameters
        best_params = {param_names[i]: result.x[i] for i in range(len(result.x))}
        
        return {
            'best_parameters': best_params,
            'best_score': -result.fun,
            'evaluation_history': self.objective.evaluation_history,
            'total_evaluations': self.objective.evaluation_count,
            'convergence_info': {
                'convergence': result.func_vals,
                'acquisition_values': getattr(result, 'acquisition_values', None)
            }
        }

class WalkForwardAnalyzer:
    """Walk-forward analysis system"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
    
    def analyze(self, framework: VectorBTFramework, 
               best_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform walk-forward analysis"""
        logger.info("Starting walk-forward analysis...")
        
        # Load data
        data = framework.load_data()
        
        # Calculate window parameters
        training_days = self.config.training_window_months * 30
        validation_days = self.config.validation_window_months * 30
        step_days = self.config.step_months * 30
        
        start_date = data.index[0]
        end_date = data.index[-1]
        
        periods = []
        current_start = start_date
        
        while current_start + timedelta(days=training_days + validation_days) <= end_date:
            training_end = current_start + timedelta(days=training_days)
            validation_start = training_end
            validation_end = validation_start + timedelta(days=validation_days)
            
            periods.append({
                'training_start': current_start,
                'training_end': training_end,
                'validation_start': validation_start,
                'validation_end': validation_end
            })
            
            current_start += timedelta(days=step_days)
        
        logger.info(f"Generated {len(periods)} walk-forward periods")
        
        # Analyze each period
        period_results = []
        
        for i, period in enumerate(periods):
            try:
                result = self._analyze_period(framework, best_parameters, period, i)
                period_results.append(result)
                
                if (i + 1) % 5 == 0:
                    logger.info(f"Walk-forward progress: {i+1}/{len(periods)}")
                    
            except Exception as e:
                logger.warning(f"Walk-forward period {i} failed: {e}")
                continue
        
        # Calculate aggregate statistics
        aggregate_stats = self._calculate_aggregate_statistics(period_results)
        
        return {
            'periods': period_results,
            'aggregate_statistics': aggregate_stats,
            'total_periods': len(periods),
            'successful_periods': len(period_results)
        }
    
    def _analyze_period(self, framework: VectorBTFramework, 
                       parameters: Dict[str, Any], 
                       period: Dict, period_id: int) -> Dict[str, Any]:
        """Analyze single walk-forward period"""
        
        # Filter data for training period
        training_data = framework.data[period['training_start']:period['training_end']]
        validation_data = framework.data[period['validation_start']:period['validation_end']]
        
        if len(training_data) < 100 or len(validation_data) < 20:
            raise ValueError(f"Insufficient data in period {period_id}")
        
        # Update framework with parameters
        for key, value in parameters.items():
            if hasattr(framework.config, key):
                setattr(framework.config, key, value)
        
        # Train on training data
        framework.data = training_data
        framework.generate_signals()
        
        # Validate on validation data
        framework.data = validation_data
        portfolio = framework.run_backtest()
        metrics = framework.calculate_performance_metrics(portfolio)
        
        return {
            'period_id': period_id,
            'period_info': period,
            'training_samples': len(training_data),
            'validation_samples': len(validation_data),
            'metrics': metrics,
            'parameters_used': parameters.copy()
        }
    
    def _calculate_aggregate_statistics(self, period_results: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate statistics across all periods"""
        if not period_results:
            return {}
        
        # Extract metrics
        metrics_by_period = []
        for result in period_results:
            metrics_by_period.append(result['metrics'])
        
        # Calculate statistics for each metric
        aggregate_stats = {}
        
        if metrics_by_period:
            sample_metrics = metrics_by_period[0]
            
            for metric_name in sample_metrics.keys():
                if isinstance(sample_metrics[metric_name], (int, float)):
                    values = [m[metric_name] for m in metrics_by_period if not np.isnan(m[metric_name])]
                    
                    if values:
                        aggregate_stats[metric_name] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values),
                            'median': np.median(values),
                            'positive_periods': sum(1 for v in values if v > 0) / len(values)
                        }
        
        return aggregate_stats

class RobustnessAnalyzer:
    """Analyze parameter robustness"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
    
    def analyze(self, framework: VectorBTFramework, 
               best_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze parameter robustness"""
        logger.info("Starting robustness analysis...")
        
        results = {
            'noise_sensitivity': self._analyze_noise_sensitivity(framework, best_parameters),
            'parameter_sensitivity': self._analyze_parameter_sensitivity(framework, best_parameters),
            'monte_carlo_stability': self._analyze_monte_carlo_stability(framework, best_parameters)
        }
        
        return results
    
    def _analyze_noise_sensitivity(self, framework: VectorBTFramework, 
                                 parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Test sensitivity to data noise"""
        logger.info("Analyzing noise sensitivity...")
        
        noise_results = {}
        
        for noise_level in self.config.noise_levels:
            # Add noise to data
            original_data = framework.data.copy()
            noisy_data = original_data.copy()
            
            # Add random noise to prices
            for col in ['Open', 'High', 'Low', 'Close']:
                noise = np.random.normal(0, noise_level, len(noisy_data))
                noisy_data[col] *= (1 + noise)
            
            # Run backtest with noisy data
            framework.data = noisy_data
            
            try:
                # Update parameters
                for key, value in parameters.items():
                    if hasattr(framework.config, key):
                        setattr(framework.config, key, value)
                
                framework.generate_signals()
                portfolio = framework.run_backtest()
                metrics = framework.calculate_performance_metrics(portfolio)
                
                noise_results[f'noise_{noise_level}'] = metrics
                
            except Exception as e:
                logger.warning(f"Noise sensitivity test failed for level {noise_level}: {e}")
                noise_results[f'noise_{noise_level}'] = None
            
            # Restore original data
            framework.data = original_data
        
        return noise_results
    
    def _analyze_parameter_sensitivity(self, framework: VectorBTFramework, 
                                     parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Test sensitivity to parameter changes"""
        logger.info("Analyzing parameter sensitivity...")
        
        sensitivity_results = {}
        
        for param_name, base_value in parameters.items():
            if isinstance(base_value, (int, float)):
                # Test parameter variations
                variations = [-0.2, -0.1, 0.1, 0.2]  # ±20%, ±10%
                param_results = {}
                
                for variation in variations:
                    new_value = base_value * (1 + variation)
                    
                    # Ensure parameter bounds
                    if isinstance(base_value, int):
                        new_value = max(1, int(new_value))
                    
                    # Update parameter
                    test_params = parameters.copy()
                    test_params[param_name] = new_value
                    
                    try:
                        # Update framework
                        for key, value in test_params.items():
                            if hasattr(framework.config, key):
                                setattr(framework.config, key, value)
                        
                        framework.generate_signals()
                        portfolio = framework.run_backtest()
                        metrics = framework.calculate_performance_metrics(portfolio)
                        
                        param_results[f'variation_{variation}'] = metrics
                        
                    except Exception as e:
                        logger.warning(f"Parameter sensitivity test failed for {param_name} variation {variation}: {e}")
                        param_results[f'variation_{variation}'] = None
                
                sensitivity_results[param_name] = param_results
        
        return sensitivity_results
    
    def _analyze_monte_carlo_stability(self, framework: VectorBTFramework, 
                                     parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Monte Carlo stability analysis"""
        logger.info("Analyzing Monte Carlo stability...")
        
        results = []
        
        for i in range(self.config.monte_carlo_iterations):
            try:
                # Randomly bootstrap data
                n_samples = len(framework.data)
                bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
                bootstrap_data = framework.data.iloc[bootstrap_indices].copy()
                bootstrap_data.index = framework.data.index  # Keep original timestamps
                
                # Update framework
                framework.data = bootstrap_data
                
                for key, value in parameters.items():
                    if hasattr(framework.config, key):
                        setattr(framework.config, key, value)
                
                framework.generate_signals()
                portfolio = framework.run_backtest()
                metrics = framework.calculate_performance_metrics(portfolio)
                
                results.append(metrics)
                
            except Exception as e:
                logger.warning(f"Monte Carlo iteration {i} failed: {e}")
                continue
        
        # Calculate stability statistics
        if results:
            stability_stats = {}
            sample_metrics = results[0]
            
            for metric_name in sample_metrics.keys():
                if isinstance(sample_metrics[metric_name], (int, float)):
                    values = [r[metric_name] for r in results if not np.isnan(r[metric_name])]
                    
                    if values:
                        stability_stats[metric_name] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'coefficient_of_variation': np.std(values) / np.mean(values) if np.mean(values) != 0 else np.inf,
                            'percentile_5': np.percentile(values, 5),
                            'percentile_95': np.percentile(values, 95)
                        }
            
            return {
                'stability_statistics': stability_stats,
                'total_iterations': len(results),
                'success_rate': len(results) / self.config.monte_carlo_iterations
            }
        
        return {}

class OptimizationEngine:
    """Main optimization engine"""
    
    def __init__(self, config: OptimizationConfig = None, 
                 parameter_space: ParameterSpace = None):
        
        self.config = config or OptimizationConfig()
        self.parameter_space = parameter_space or ParameterSpace()
        
        # Initialize components
        self.optimizer = self._create_optimizer()
        self.walk_forward_analyzer = WalkForwardAnalyzer(self.config)
        self.robustness_analyzer = RobustnessAnalyzer(self.config)
        
        # Results storage
        self.optimization_results = {}
        self.validation_results = {}
        
        logger.info(f"Optimization engine initialized with {self.config.method} optimizer")
    
    def _create_optimizer(self) -> BaseOptimizer:
        """Create optimizer based on configuration"""
        optimizers = {
            'grid': GridSearchOptimizer,
            'random': RandomSearchOptimizer,
            'genetic': GeneticOptimizer,
            'bayesian': BayesianOptimizer
        }
        
        optimizer_class = optimizers.get(self.config.method, GeneticOptimizer)
        return optimizer_class(self.config, self.parameter_space)
    
    def run_optimization(self, framework: VectorBTFramework) -> Dict[str, Any]:
        """Run complete optimization process"""
        logger.info("Starting complete optimization process...")
        
        start_time = datetime.now()
        
        try:
            # 1. Parameter optimization
            logger.info("Step 1: Parameter optimization...")
            optimization_results = self.optimizer.optimize(framework)
            self.optimization_results = optimization_results
            
            best_parameters = optimization_results['best_parameters']
            
            # 2. Walk-forward analysis (if enabled)
            if self.config.use_walk_forward:
                logger.info("Step 2: Walk-forward analysis...")
                wf_results = self.walk_forward_analyzer.analyze(framework, best_parameters)
                self.validation_results['walk_forward'] = wf_results
            
            # 3. Robustness analysis
            logger.info("Step 3: Robustness analysis...")
            robustness_results = self.robustness_analyzer.analyze(framework, best_parameters)
            self.validation_results['robustness'] = robustness_results
            
            # 4. Out-of-sample validation
            logger.info("Step 4: Out-of-sample validation...")
            oos_results = self._out_of_sample_validation(framework, best_parameters)
            self.validation_results['out_of_sample'] = oos_results
            
            # Calculate total time
            total_time = datetime.now() - start_time
            
            # Compile final results
            final_results = {
                'optimization': optimization_results,
                'validation': self.validation_results,
                'summary': self._generate_summary(),
                'metadata': {
                    'optimization_time': str(total_time),
                    'method': self.config.method,
                    'total_evaluations': optimization_results.get('total_evaluations', 0)
                }
            }
            
            logger.info(f"Optimization process completed in {total_time}")
            logger.info(f"Best score: {optimization_results['best_score']:.4f}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Optimization process failed: {e}")
            raise
    
    def _out_of_sample_validation(self, framework: VectorBTFramework, 
                                 parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform out-of-sample validation"""
        
        # Split data
        data = framework.data
        split_point = int(len(data) * (1 - self.config.out_of_sample_pct))
        
        in_sample_data = data.iloc[:split_point]
        out_of_sample_data = data.iloc[split_point:]
        
        # Validate on out-of-sample data
        framework.data = out_of_sample_data
        
        # Update parameters
        for key, value in parameters.items():
            if hasattr(framework.config, key):
                setattr(framework.config, key, value)
        
        framework.generate_signals()
        portfolio = framework.run_backtest()
        metrics = framework.calculate_performance_metrics(portfolio)
        
        # Restore original data
        framework.data = data
        
        return {
            'in_sample_periods': len(in_sample_data),
            'out_of_sample_periods': len(out_of_sample_data),
            'out_of_sample_metrics': metrics,
            'data_split_date': out_of_sample_data.index[0].isoformat()
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate optimization summary"""
        summary = {
            'optimization_successful': bool(self.optimization_results),
            'validation_completed': bool(self.validation_results)
        }
        
        if self.optimization_results:
            summary['best_score'] = self.optimization_results['best_score']
            summary['total_evaluations'] = self.optimization_results.get('total_evaluations', 0)
        
        if 'walk_forward' in self.validation_results:
            wf_results = self.validation_results['walk_forward']
            summary['walk_forward_periods'] = wf_results.get('total_periods', 0)
            summary['walk_forward_success_rate'] = (
                wf_results.get('successful_periods', 0) / wf_results.get('total_periods', 1)
            )
        
        if 'robustness' in self.validation_results:
            robustness = self.validation_results['robustness']
            summary['robustness_tested'] = bool(robustness)
        
        if 'out_of_sample' in self.validation_results:
            oos = self.validation_results['out_of_sample']
            summary['out_of_sample_sharpe'] = oos['out_of_sample_metrics'].get('sharpe_ratio', 0)
        
        return summary
    
    def save_results(self, output_dir: str) -> str:
        """Save optimization results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save optimization results
        if self.optimization_results:
            opt_path = output_path / f"optimization_results_{timestamp}.json"
            with open(opt_path, 'w') as f:
                json.dump(self.optimization_results, f, indent=2, default=str)
        
        # Save validation results
        if self.validation_results:
            val_path = output_path / f"validation_results_{timestamp}.json"
            with open(val_path, 'w') as f:
                json.dump(self.validation_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")
        return str(output_path)

def create_optimization_engine(config: OptimizationConfig = None,
                             parameter_space: ParameterSpace = None) -> OptimizationEngine:
    """Factory function to create optimization engine"""
    return OptimizationEngine(config, parameter_space)

# Example usage
if __name__ == "__main__":
    # Create configuration
    config = OptimizationConfig(
        method="genetic",
        max_iterations=500,
        use_walk_forward=True,
        training_window_months=12,
        validation_window_months=3
    )
    
    # Create parameter space
    parameter_space = ParameterSpace()
    
    # Create optimization engine
    engine = create_optimization_engine(config, parameter_space)
    
    print("Advanced Parameter Optimization Engine initialized successfully!")
    print(f"Method: {config.method}")
    print(f"Max iterations: {config.max_iterations}")
    print(f"Walk-forward enabled: {config.use_walk_forward}")
    print(f"Parameter space: {len(parameter_space.to_dict())} parameters")