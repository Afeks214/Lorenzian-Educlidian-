"""
Automated Hyperparameter Tuning for MARL Training
Implements advanced hyperparameter optimization using multiple strategies
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import pickle
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import optuna
from hyperopt import hp, fmin, tpe, Trials
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler, PopulationBasedTraining
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.hyperopt import HyperOptSearch

logger = logging.getLogger(__name__)

@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter optimization"""
    # Optimization strategy
    strategy: str = "bayesian"  # "random", "grid", "bayesian", "optuna", "hyperopt", "pbt"
    
    # Search space
    search_space: Dict[str, Any] = None
    
    # Optimization settings
    max_trials: int = 100
    max_concurrent_trials: int = 4
    timeout_seconds: int = 3600  # 1 hour
    
    # Early stopping
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    
    # Multi-objective optimization
    objectives: List[str] = None  # ["reward", "loss", "stability"]
    objective_weights: List[float] = None
    
    # Pruning
    enable_pruning: bool = True
    pruning_warmup_steps: int = 100
    
    # Population-based training
    pbt_population_size: int = 8
    pbt_mutation_rate: float = 0.2
    
    # Gaussian Process settings
    gp_kernel: str = "matern"  # "rbf", "matern"
    gp_acquisition: str = "ei"  # "ei", "ucb", "pi"
    
    # Results
    results_dir: str = "hyperopt_results"
    save_intermediate_results: bool = True
    
    # Distributed settings
    use_distributed: bool = False
    num_workers: int = 4
    
    def __post_init__(self):
        if self.search_space is None:
            self.search_space = self._get_default_search_space()
        
        if self.objectives is None:
            self.objectives = ["reward"]
        
        if self.objective_weights is None:
            self.objective_weights = [1.0] * len(self.objectives)
    
    def _get_default_search_space(self) -> Dict[str, Any]:
        """Get default search space for MARL hyperparameters"""
        return {
            # Learning parameters
            'learning_rate': {'type': 'log_uniform', 'low': 1e-5, 'high': 1e-2},
            'batch_size': {'type': 'choice', 'choices': [16, 32, 64, 128, 256]},
            'gamma': {'type': 'uniform', 'low': 0.9, 'high': 0.999},
            'tau': {'type': 'uniform', 'low': 0.001, 'high': 0.1},
            
            # Network architecture
            'hidden_dim': {'type': 'choice', 'choices': [64, 128, 256, 512]},
            'num_layers': {'type': 'int', 'low': 2, 'high': 6},
            'dropout_rate': {'type': 'uniform', 'low': 0.0, 'high': 0.5},
            
            # Training parameters
            'entropy_coef': {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-1},
            'value_loss_coef': {'type': 'uniform', 'low': 0.1, 'high': 1.0},
            'max_grad_norm': {'type': 'uniform', 'low': 0.1, 'high': 10.0},
            
            # Optimizer parameters
            'optimizer': {'type': 'choice', 'choices': ['adam', 'adamw', 'sgd', 'rmsprop']},
            'weight_decay': {'type': 'log_uniform', 'low': 1e-6, 'high': 1e-2},
            'momentum': {'type': 'uniform', 'low': 0.8, 'high': 0.99},
            
            # Regularization
            'l1_reg': {'type': 'log_uniform', 'low': 1e-6, 'high': 1e-2},
            'l2_reg': {'type': 'log_uniform', 'low': 1e-6, 'high': 1e-2},
            
            # Experience replay
            'buffer_size': {'type': 'choice', 'choices': [10000, 50000, 100000, 500000]},
            'replay_ratio': {'type': 'uniform', 'low': 0.1, 'high': 4.0},
            
            # Exploration
            'exploration_noise': {'type': 'uniform', 'low': 0.01, 'high': 0.3},
            'exploration_decay': {'type': 'uniform', 'low': 0.99, 'high': 0.9999}
        }


class HyperparameterTrial:
    """Represents a single hyperparameter trial"""
    
    def __init__(self, trial_id: str, parameters: Dict[str, Any]):
        self.trial_id = trial_id
        self.parameters = parameters
        self.start_time = time.time()
        self.end_time = None
        self.status = "running"
        self.results = {}
        self.intermediate_results = []
        self.error = None
        
    def update_result(self, step: int, metrics: Dict[str, float]):
        """Update intermediate results"""
        self.intermediate_results.append({
            'step': step,
            'metrics': metrics,
            'timestamp': time.time()
        })
    
    def complete(self, final_results: Dict[str, float], error: str = None):
        """Complete the trial"""
        self.end_time = time.time()
        self.results = final_results
        self.error = error
        self.status = "completed" if error is None else "failed"
    
    def get_duration(self) -> float:
        """Get trial duration"""
        end_time = self.end_time or time.time()
        return end_time - self.start_time
    
    def get_best_metric(self, metric_name: str) -> float:
        """Get best value for a metric"""
        if metric_name in self.results:
            return self.results[metric_name]
        
        # Look in intermediate results
        values = [r['metrics'].get(metric_name, float('-inf')) for r in self.intermediate_results]
        return max(values) if values else float('-inf')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'trial_id': self.trial_id,
            'parameters': self.parameters,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'status': self.status,
            'results': self.results,
            'intermediate_results': self.intermediate_results,
            'error': self.error,
            'duration': self.get_duration()
        }


class BayesianOptimizer:
    """Bayesian optimization using Gaussian Process"""
    
    def __init__(self, search_space: Dict[str, Any], config: HyperparameterConfig):
        self.search_space = search_space
        self.config = config
        self.trials = []
        self.gp_model = None
        self.scaler = StandardScaler()
        self.parameter_bounds = self._get_parameter_bounds()
        
    def _get_parameter_bounds(self) -> List[Tuple[float, float]]:
        """Get parameter bounds for optimization"""
        bounds = []
        for param_name, param_config in self.search_space.items():
            if param_config['type'] == 'uniform':
                bounds.append((param_config['low'], param_config['high']))
            elif param_config['type'] == 'log_uniform':
                bounds.append((np.log(param_config['low']), np.log(param_config['high'])))
            elif param_config['type'] == 'int':
                bounds.append((param_config['low'], param_config['high']))
            elif param_config['type'] == 'choice':
                bounds.append((0, len(param_config['choices']) - 1))
        return bounds
    
    def _encode_parameters(self, parameters: Dict[str, Any]) -> np.ndarray:
        """Encode parameters for GP"""
        encoded = []
        for param_name, param_config in self.search_space.items():
            value = parameters[param_name]
            
            if param_config['type'] == 'uniform':
                encoded.append(value)
            elif param_config['type'] == 'log_uniform':
                encoded.append(np.log(value))
            elif param_config['type'] == 'int':
                encoded.append(value)
            elif param_config['type'] == 'choice':
                encoded.append(param_config['choices'].index(value))
        
        return np.array(encoded)
    
    def _decode_parameters(self, encoded: np.ndarray) -> Dict[str, Any]:
        """Decode parameters from GP"""
        parameters = {}
        for i, (param_name, param_config) in enumerate(self.search_space.items()):
            value = encoded[i]
            
            if param_config['type'] == 'uniform':
                parameters[param_name] = value
            elif param_config['type'] == 'log_uniform':
                parameters[param_name] = np.exp(value)
            elif param_config['type'] == 'int':
                parameters[param_name] = int(round(value))
            elif param_config['type'] == 'choice':
                idx = int(round(np.clip(value, 0, len(param_config['choices']) - 1)))
                parameters[param_name] = param_config['choices'][idx]
        
        return parameters
    
    def _fit_gp_model(self):
        """Fit Gaussian Process model"""
        if len(self.trials) < 2:
            return
        
        # Prepare training data
        X = np.array([self._encode_parameters(trial.parameters) for trial in self.trials])
        y = np.array([trial.get_best_metric('reward') for trial in self.trials])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create and fit GP model
        if self.config.gp_kernel == 'rbf':
            kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        else:  # matern
            kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )
        
        self.gp_model.fit(X_scaled, y)
    
    def _acquisition_function(self, x: np.ndarray) -> float:
        """Acquisition function for Bayesian optimization"""
        if self.gp_model is None:
            return np.random.random()
        
        x_scaled = self.scaler.transform(x.reshape(1, -1))
        mean, std = self.gp_model.predict(x_scaled, return_std=True)
        
        if self.config.gp_acquisition == 'ei':
            # Expected Improvement
            best_value = max(trial.get_best_metric('reward') for trial in self.trials)
            z = (mean - best_value) / std
            ei = (mean - best_value) * norm.cdf(z) + std * norm.pdf(z)
            return ei[0]
        elif self.config.gp_acquisition == 'ucb':
            # Upper Confidence Bound
            beta = 2.0
            return mean[0] + beta * std[0]
        else:  # pi
            # Probability of Improvement
            best_value = max(trial.get_best_metric('reward') for trial in self.trials)
            z = (mean - best_value) / std
            return norm.cdf(z)[0]
    
    def suggest_parameters(self) -> Dict[str, Any]:
        """Suggest next set of parameters"""
        if len(self.trials) < 5:
            # Random search for initial trials
            return self._random_parameters()
        
        # Fit GP model
        self._fit_gp_model()
        
        # Optimize acquisition function
        best_x = None
        best_value = float('-inf')
        
        for _ in range(100):  # Multiple random starts
            x0 = np.random.uniform(
                low=[b[0] for b in self.parameter_bounds],
                high=[b[1] for b in self.parameter_bounds]
            )
            
            result = minimize(
                lambda x: -self._acquisition_function(x),
                x0,
                bounds=self.parameter_bounds,
                method='L-BFGS-B'
            )
            
            if result.success and -result.fun > best_value:
                best_value = -result.fun
                best_x = result.x
        
        if best_x is not None:
            return self._decode_parameters(best_x)
        else:
            return self._random_parameters()
    
    def _random_parameters(self) -> Dict[str, Any]:
        """Generate random parameters"""
        parameters = {}
        for param_name, param_config in self.search_space.items():
            if param_config['type'] == 'uniform':
                parameters[param_name] = np.random.uniform(param_config['low'], param_config['high'])
            elif param_config['type'] == 'log_uniform':
                parameters[param_name] = np.exp(np.random.uniform(
                    np.log(param_config['low']), 
                    np.log(param_config['high'])
                ))
            elif param_config['type'] == 'int':
                parameters[param_name] = np.random.randint(param_config['low'], param_config['high'] + 1)
            elif param_config['type'] == 'choice':
                parameters[param_name] = np.random.choice(param_config['choices'])
        
        return parameters
    
    def add_trial(self, trial: HyperparameterTrial):
        """Add completed trial"""
        self.trials.append(trial)


class OptunaTuner:
    """Hyperparameter tuning using Optuna"""
    
    def __init__(self, search_space: Dict[str, Any], config: HyperparameterConfig):
        self.search_space = search_space
        self.config = config
        self.study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=config.pruning_warmup_steps,
                n_warmup_steps=config.pruning_warmup_steps
            ) if config.enable_pruning else None
        )
    
    def create_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Create parameters for Optuna trial"""
        parameters = {}
        
        for param_name, param_config in self.search_space.items():
            if param_config['type'] == 'uniform':
                parameters[param_name] = trial.suggest_float(
                    param_name, 
                    param_config['low'], 
                    param_config['high']
                )
            elif param_config['type'] == 'log_uniform':
                parameters[param_name] = trial.suggest_float(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    log=True
                )
            elif param_config['type'] == 'int':
                parameters[param_name] = trial.suggest_int(
                    param_name,
                    param_config['low'],
                    param_config['high']
                )
            elif param_config['type'] == 'choice':
                parameters[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )
        
        return parameters
    
    def optimize(self, objective_function: Callable) -> Dict[str, Any]:
        """Run Optuna optimization"""
        self.study.optimize(
            objective_function,
            n_trials=self.config.max_trials,
            n_jobs=self.config.max_concurrent_trials,
            timeout=self.config.timeout_seconds
        )
        
        return {
            'best_params': self.study.best_params,
            'best_value': self.study.best_value,
            'best_trial': self.study.best_trial,
            'trials': self.study.trials
        }


class HyperparameterOptimizer:
    """Main hyperparameter optimization coordinator"""
    
    def __init__(self, config: HyperparameterConfig):
        self.config = config
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Trial management
        self.trials = []
        self.active_trials = {}
        self.completed_trials = []
        self.failed_trials = []
        
        # Optimization components
        self.optimizer = None
        self._initialize_optimizer()
        
        # Results tracking
        self.optimization_history = []
        self.best_trial = None
        self.best_score = float('-inf')
        
        # Threading
        self.trial_executor = ThreadPoolExecutor(max_workers=config.max_concurrent_trials)
        self.optimization_start_time = None
        
        logger.info(f"Hyperparameter optimizer initialized with strategy: {config.strategy}")
    
    def _initialize_optimizer(self):
        """Initialize optimization strategy"""
        if self.config.strategy == "bayesian":
            self.optimizer = BayesianOptimizer(self.config.search_space, self.config)
        elif self.config.strategy == "optuna":
            self.optimizer = OptunaTuner(self.config.search_space, self.config)
        elif self.config.strategy == "random":
            self.optimizer = None  # Will use random sampling
        else:
            raise ValueError(f"Unsupported optimization strategy: {self.config.strategy}")
    
    def optimize(self, 
                 training_function: Callable[[Dict[str, Any]], float],
                 validation_function: Optional[Callable[[Dict[str, Any]], Dict[str, float]]] = None) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        
        logger.info(f"Starting hyperparameter optimization with {self.config.max_trials} trials")
        self.optimization_start_time = time.time()
        
        try:
            if self.config.strategy == "optuna":
                return self._optimize_with_optuna(training_function, validation_function)
            elif self.config.strategy == "ray":
                return self._optimize_with_ray(training_function, validation_function)
            else:
                return self._optimize_sequential(training_function, validation_function)
        
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
        finally:
            # Clean up
            self.trial_executor.shutdown(wait=True)
    
    def _optimize_sequential(self, 
                           training_function: Callable,
                           validation_function: Optional[Callable] = None) -> Dict[str, Any]:
        """Sequential optimization using configured strategy"""
        
        for trial_idx in range(self.config.max_trials):
            # Check timeout
            if self._check_timeout():
                logger.info("Optimization timeout reached")
                break
            
            # Generate parameters
            if self.config.strategy == "random":
                parameters = self._generate_random_parameters()
            elif self.config.strategy == "bayesian":
                parameters = self.optimizer.suggest_parameters()
            else:
                raise ValueError(f"Unsupported strategy: {self.config.strategy}")
            
            # Create trial
            trial_id = f"trial_{trial_idx:04d}"
            trial = HyperparameterTrial(trial_id, parameters)
            self.trials.append(trial)
            
            # Run trial
            logger.info(f"Starting trial {trial_id} with parameters: {parameters}")
            
            try:
                # Train model
                score = training_function(parameters)
                
                # Validate if validation function provided
                validation_metrics = {}
                if validation_function:
                    validation_metrics = validation_function(parameters)
                
                # Complete trial
                results = {'score': score, **validation_metrics}
                trial.complete(results)
                self.completed_trials.append(trial)
                
                # Update best
                if score > self.best_score:
                    self.best_score = score
                    self.best_trial = trial
                    logger.info(f"New best trial: {trial_id} with score {score:.4f}")
                
                # Add to optimizer
                if self.optimizer:
                    self.optimizer.add_trial(trial)
                
                # Save intermediate results
                if self.config.save_intermediate_results:
                    self._save_intermediate_results()
                
                # Check early stopping
                if self._should_stop_early():
                    logger.info("Early stopping triggered")
                    break
                
            except Exception as e:
                logger.error(f"Trial {trial_id} failed: {e}")
                trial.complete({}, str(e))
                self.failed_trials.append(trial)
        
        return self._compile_results()
    
    def _optimize_with_optuna(self, 
                            training_function: Callable,
                            validation_function: Optional[Callable] = None) -> Dict[str, Any]:
        """Optimize using Optuna"""
        
        def objective(trial):
            parameters = self.optimizer.create_trial_params(trial)
            
            try:
                score = training_function(parameters)
                
                # Report intermediate values if available
                if hasattr(training_function, 'intermediate_values'):
                    for step, value in training_function.intermediate_values.items():
                        trial.report(value, step)
                        
                        # Pruning
                        if trial.should_prune():
                            raise optuna.exceptions.TrialPruned()
                
                return score
                
            except Exception as e:
                logger.error(f"Optuna trial failed: {e}")
                return float('-inf')
        
        # Run optimization
        results = self.optimizer.optimize(objective)
        
        # Convert to standard format
        return {
            'best_parameters': results['best_params'],
            'best_score': results['best_value'],
            'optimization_history': [
                {
                    'trial_id': f"trial_{i:04d}",
                    'parameters': trial.params,
                    'score': trial.value,
                    'state': trial.state.name
                }
                for i, trial in enumerate(results['trials'])
            ],
            'total_trials': len(results['trials']),
            'successful_trials': len([t for t in results['trials'] if t.state == optuna.trial.TrialState.COMPLETE]),
            'failed_trials': len([t for t in results['trials'] if t.state == optuna.trial.TrialState.FAIL])
        }
    
    def _optimize_with_ray(self, 
                          training_function: Callable,
                          validation_function: Optional[Callable] = None) -> Dict[str, Any]:
        """Optimize using Ray Tune"""
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init()
        
        # Convert search space to Ray format
        ray_search_space = self._convert_search_space_to_ray()
        
        # Create scheduler
        if self.config.strategy == "pbt":
            scheduler = PopulationBasedTraining(
                time_attr="training_iteration",
                metric="score",
                mode="max",
                perturbation_interval=10,
                hyperparam_mutations=ray_search_space
            )
        else:
            scheduler = AsyncHyperBandScheduler(
                time_attr="training_iteration",
                metric="score",
                mode="max",
                max_t=100,
                grace_period=10
            )
        
        # Run optimization
        analysis = tune.run(
            training_function,
            config=ray_search_space,
            num_samples=self.config.max_trials,
            scheduler=scheduler,
            resources_per_trial={"cpu": 1, "gpu": 0.1},
            stop={"training_iteration": 100}
        )
        
        # Get results
        best_trial = analysis.get_best_trial("score", "max")
        
        return {
            'best_parameters': best_trial.config,
            'best_score': best_trial.last_result["score"],
            'optimization_history': [
                {
                    'trial_id': trial.trial_id,
                    'parameters': trial.config,
                    'score': trial.last_result.get("score", float('-inf')),
                    'state': trial.status
                }
                for trial in analysis.trials
            ],
            'total_trials': len(analysis.trials),
            'successful_trials': len([t for t in analysis.trials if t.status == "TERMINATED"]),
            'failed_trials': len([t for t in analysis.trials if t.status == "ERROR"])
        }
    
    def _convert_search_space_to_ray(self) -> Dict[str, Any]:
        """Convert search space to Ray Tune format"""
        ray_space = {}
        
        for param_name, param_config in self.config.search_space.items():
            if param_config['type'] == 'uniform':
                ray_space[param_name] = tune.uniform(param_config['low'], param_config['high'])
            elif param_config['type'] == 'log_uniform':
                ray_space[param_name] = tune.loguniform(param_config['low'], param_config['high'])
            elif param_config['type'] == 'int':
                ray_space[param_name] = tune.randint(param_config['low'], param_config['high'] + 1)
            elif param_config['type'] == 'choice':
                ray_space[param_name] = tune.choice(param_config['choices'])
        
        return ray_space
    
    def _generate_random_parameters(self) -> Dict[str, Any]:
        """Generate random parameters"""
        parameters = {}
        for param_name, param_config in self.config.search_space.items():
            if param_config['type'] == 'uniform':
                parameters[param_name] = np.random.uniform(param_config['low'], param_config['high'])
            elif param_config['type'] == 'log_uniform':
                parameters[param_name] = np.exp(np.random.uniform(
                    np.log(param_config['low']), 
                    np.log(param_config['high'])
                ))
            elif param_config['type'] == 'int':
                parameters[param_name] = np.random.randint(param_config['low'], param_config['high'] + 1)
            elif param_config['type'] == 'choice':
                parameters[param_name] = np.random.choice(param_config['choices'])
        
        return parameters
    
    def _check_timeout(self) -> bool:
        """Check if optimization timeout reached"""
        if self.optimization_start_time is None:
            return False
        
        elapsed = time.time() - self.optimization_start_time
        return elapsed >= self.config.timeout_seconds
    
    def _should_stop_early(self) -> bool:
        """Check early stopping conditions"""
        if not self.config.enable_early_stopping:
            return False
        
        if len(self.completed_trials) < self.config.early_stopping_patience:
            return False
        
        # Check if recent trials show improvement
        recent_scores = [
            trial.get_best_metric('score') 
            for trial in self.completed_trials[-self.config.early_stopping_patience:]
        ]
        
        # Check if improvement is less than threshold
        max_recent = max(recent_scores)
        if len(self.completed_trials) >= self.config.early_stopping_patience * 2:
            older_scores = [
                trial.get_best_metric('score') 
                for trial in self.completed_trials[-self.config.early_stopping_patience * 2:-self.config.early_stopping_patience]
            ]
            max_older = max(older_scores)
            
            improvement = max_recent - max_older
            return improvement < self.config.early_stopping_min_delta
        
        return False
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compile optimization results"""
        optimization_time = time.time() - self.optimization_start_time
        
        results = {
            'best_parameters': self.best_trial.parameters if self.best_trial else {},
            'best_score': self.best_score,
            'optimization_history': [trial.to_dict() for trial in self.trials],
            'total_trials': len(self.trials),
            'successful_trials': len(self.completed_trials),
            'failed_trials': len(self.failed_trials),
            'optimization_time': optimization_time,
            'config': asdict(self.config)
        }
        
        # Save final results
        self._save_results(results)
        
        return results
    
    def _save_intermediate_results(self):
        """Save intermediate results"""
        results = {
            'trials': [trial.to_dict() for trial in self.trials],
            'best_score': self.best_score,
            'best_parameters': self.best_trial.parameters if self.best_trial else {},
            'timestamp': time.time()
        }
        
        results_file = self.results_dir / "intermediate_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def _save_results(self, results: Dict[str, Any]):
        """Save final results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"optimization_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Optimization results saved to {results_file}")
    
    def visualize_results(self):
        """Create visualization of optimization results"""
        if not self.completed_trials:
            logger.warning("No completed trials to visualize")
            return
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Optimization history
        trial_numbers = range(len(self.completed_trials))
        scores = [trial.get_best_metric('score') for trial in self.completed_trials]
        
        axes[0, 0].plot(trial_numbers, scores, 'b-', alpha=0.7)
        axes[0, 0].scatter(trial_numbers, scores, alpha=0.7)
        axes[0, 0].set_xlabel('Trial Number')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Optimization History')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Best score progression
        best_scores = []
        current_best = float('-inf')
        for score in scores:
            if score > current_best:
                current_best = score
            best_scores.append(current_best)
        
        axes[0, 1].plot(trial_numbers, best_scores, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Trial Number')
        axes[0, 1].set_ylabel('Best Score')
        axes[0, 1].set_title('Best Score Progression')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Parameter importance (for numeric parameters)
        numeric_params = {}
        for trial in self.completed_trials:
            for param_name, param_value in trial.parameters.items():
                if isinstance(param_value, (int, float)):
                    if param_name not in numeric_params:
                        numeric_params[param_name] = []
                    numeric_params[param_name].append((param_value, trial.get_best_metric('score')))
        
        if numeric_params:
            param_name = list(numeric_params.keys())[0]  # Show first numeric parameter
            values, scores = zip(*numeric_params[param_name])
            axes[1, 0].scatter(values, scores, alpha=0.7)
            axes[1, 0].set_xlabel(param_name)
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_title(f'Parameter vs Score: {param_name}')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Trial duration distribution
        durations = [trial.get_duration() for trial in self.completed_trials]
        axes[1, 1].hist(durations, bins=20, alpha=0.7, color='orange')
        axes[1, 1].set_xlabel('Trial Duration (seconds)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Trial Duration Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / f"optimization_visualization_{int(time.time())}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {plot_file}")


def create_hyperparameter_config(**kwargs) -> HyperparameterConfig:
    """Create hyperparameter configuration"""
    return HyperparameterConfig(**kwargs)


def run_hyperparameter_optimization_example():
    """Example of hyperparameter optimization"""
    
    # Create configuration
    config = create_hyperparameter_config(
        strategy="bayesian",
        max_trials=20,
        max_concurrent_trials=2,
        enable_early_stopping=True,
        results_dir="example_hyperopt"
    )
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(config)
    
    # Define training function
    def training_function(parameters: Dict[str, Any]) -> float:
        """Mock training function"""
        # Simulate training with these parameters
        lr = parameters['learning_rate']
        batch_size = parameters['batch_size']
        hidden_dim = parameters['hidden_dim']
        
        # Mock objective function (quadratic with noise)
        score = -(lr - 0.001)**2 * 1000 - (batch_size - 64)**2 * 0.001 - (hidden_dim - 256)**2 * 0.00001
        score += np.random.normal(0, 0.1)  # Add noise
        
        # Simulate training time
        time.sleep(0.1)
        
        return score
    
    # Run optimization
    results = optimizer.optimize(training_function)
    
    print("Optimization completed!")
    print(f"Best parameters: {results['best_parameters']}")
    print(f"Best score: {results['best_score']:.4f}")
    print(f"Total trials: {results['total_trials']}")
    print(f"Successful trials: {results['successful_trials']}")
    
    # Create visualization
    optimizer.visualize_results()
    
    return results


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run example
    results = run_hyperparameter_optimization_example()
    
    print("Example completed successfully!")