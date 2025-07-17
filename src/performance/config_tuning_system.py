#!/usr/bin/env python3
"""
Configuration Tuning System for Maximum Throughput
Implements adaptive parameter tuning, performance monitoring, and configuration optimization
"""

import torch
import torch.nn as nn
import numpy as np
import json
import logging
import time
import threading
import psutil
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
from functools import wraps
import yaml
import sqlite3
import pickle
from enum import Enum
from abc import ABC, abstractmethod
import optuna
from concurrent.futures import ThreadPoolExecutor
import asyncio
from contextlib import contextmanager
import copy

logger = logging.getLogger(__name__)

class ConfigType(Enum):
    """Types of configuration parameters"""
    PERFORMANCE = "performance"
    MEMORY = "memory"
    NETWORK = "network"
    TRAINING = "training"
    INFERENCE = "inference"
    SYSTEM = "system"

class OptimizationGoal(Enum):
    """Optimization goals"""
    MINIMIZE_LATENCY = "minimize_latency"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    MINIMIZE_MEMORY = "minimize_memory"
    MAXIMIZE_ACCURACY = "maximize_accuracy"
    BALANCE_ALL = "balance_all"

@dataclass
class ConfigParameter:
    """Configuration parameter definition"""
    name: str
    type: type
    default: Any
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    choices: Optional[List[Any]] = None
    description: str = ""
    config_type: ConfigType = ConfigType.PERFORMANCE
    impact_weight: float = 1.0
    
    def validate(self, value: Any) -> bool:
        """Validate parameter value"""
        if not isinstance(value, self.type):
            return False
        
        if self.choices and value not in self.choices:
            return False
        
        if self.min_value is not None and value < self.min_value:
            return False
        
        if self.max_value is not None and value > self.max_value:
            return False
        
        return True
    
    def suggest_value(self, trial: optuna.Trial) -> Any:
        """Suggest value for Optuna trial"""
        if self.choices:
            return trial.suggest_categorical(self.name, self.choices)
        
        if self.type == int:
            return trial.suggest_int(
                self.name, 
                self.min_value or 1, 
                self.max_value or 100
            )
        elif self.type == float:
            return trial.suggest_float(
                self.name,
                self.min_value or 0.0,
                self.max_value or 1.0
            )
        elif self.type == bool:
            return trial.suggest_categorical(self.name, [True, False])
        else:
            return self.default

@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization"""
    latency_ms: float = 0.0
    throughput_qps: float = 0.0
    memory_mb: float = 0.0
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    accuracy: float = 0.0
    stability_score: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def composite_score(self, goal: OptimizationGoal) -> float:
        """Calculate composite score based on optimization goal"""
        if goal == OptimizationGoal.MINIMIZE_LATENCY:
            return 1.0 / (1.0 + self.latency_ms)
        elif goal == OptimizationGoal.MAXIMIZE_THROUGHPUT:
            return self.throughput_qps
        elif goal == OptimizationGoal.MINIMIZE_MEMORY:
            return 1.0 / (1.0 + self.memory_mb)
        elif goal == OptimizationGoal.MAXIMIZE_ACCURACY:
            return self.accuracy
        else:  # BALANCE_ALL
            # Normalized composite score
            latency_score = 1.0 / (1.0 + self.latency_ms / 10.0)
            throughput_score = min(self.throughput_qps / 1000.0, 1.0)
            memory_score = 1.0 / (1.0 + self.memory_mb / 1000.0)
            accuracy_score = self.accuracy
            stability_score = self.stability_score
            
            return (latency_score + throughput_score + memory_score + 
                   accuracy_score + stability_score) / 5.0

class ConfigurationRegistry:
    """Registry for configuration parameters"""
    
    def __init__(self):
        self.parameters: Dict[str, ConfigParameter] = {}
        self.groups: Dict[str, List[str]] = defaultdict(list)
        self.dependencies: Dict[str, List[str]] = defaultdict(list)
        
        # Register default parameters
        self._register_default_parameters()
    
    def _register_default_parameters(self):
        """Register default configuration parameters"""
        
        # Performance parameters
        self.register(ConfigParameter(
            name="batch_size",
            type=int,
            default=32,
            min_value=1,
            max_value=256,
            description="Batch size for inference",
            config_type=ConfigType.PERFORMANCE,
            impact_weight=2.0
        ))
        
        self.register(ConfigParameter(
            name="num_workers",
            type=int,
            default=4,
            min_value=1,
            max_value=16,
            description="Number of worker threads",
            config_type=ConfigType.PERFORMANCE,
            impact_weight=1.5
        ))
        
        self.register(ConfigParameter(
            name="prefetch_size",
            type=int,
            default=10,
            min_value=1,
            max_value=100,
            description="Prefetch buffer size",
            config_type=ConfigType.PERFORMANCE,
            impact_weight=1.0
        ))
        
        self.register(ConfigParameter(
            name="enable_jit",
            type=bool,
            default=True,
            description="Enable JIT compilation",
            config_type=ConfigType.PERFORMANCE,
            impact_weight=2.0
        ))
        
        self.register(ConfigParameter(
            name="enable_quantization",
            type=bool,
            default=False,
            description="Enable model quantization",
            config_type=ConfigType.PERFORMANCE,
            impact_weight=1.5
        ))
        
        self.register(ConfigParameter(
            name="optimization_level",
            type=str,
            default="O2",
            choices=["O0", "O1", "O2", "O3"],
            description="Optimization level",
            config_type=ConfigType.PERFORMANCE,
            impact_weight=1.5
        ))
        
        # Memory parameters
        self.register(ConfigParameter(
            name="cache_size",
            type=int,
            default=1000,
            min_value=100,
            max_value=10000,
            description="Cache size",
            config_type=ConfigType.MEMORY,
            impact_weight=1.0
        ))
        
        self.register(ConfigParameter(
            name="memory_pool_size",
            type=int,
            default=500,
            min_value=100,
            max_value=5000,
            description="Memory pool size",
            config_type=ConfigType.MEMORY,
            impact_weight=1.0
        ))
        
        self.register(ConfigParameter(
            name="gc_threshold",
            type=float,
            default=0.85,
            min_value=0.5,
            max_value=0.95,
            description="Garbage collection threshold",
            config_type=ConfigType.MEMORY,
            impact_weight=0.5
        ))
        
        # Network parameters
        self.register(ConfigParameter(
            name="connection_timeout",
            type=float,
            default=5.0,
            min_value=1.0,
            max_value=30.0,
            description="Connection timeout in seconds",
            config_type=ConfigType.NETWORK,
            impact_weight=0.5
        ))
        
        self.register(ConfigParameter(
            name="max_connections",
            type=int,
            default=100,
            min_value=10,
            max_value=1000,
            description="Maximum connections",
            config_type=ConfigType.NETWORK,
            impact_weight=0.5
        ))
        
        # Training parameters
        self.register(ConfigParameter(
            name="learning_rate",
            type=float,
            default=0.001,
            min_value=1e-6,
            max_value=1.0,
            description="Learning rate",
            config_type=ConfigType.TRAINING,
            impact_weight=2.0
        ))
        
        self.register(ConfigParameter(
            name="dropout_rate",
            type=float,
            default=0.2,
            min_value=0.0,
            max_value=0.8,
            description="Dropout rate",
            config_type=ConfigType.TRAINING,
            impact_weight=1.0
        ))
        
        # Inference parameters
        self.register(ConfigParameter(
            name="inference_batch_timeout",
            type=float,
            default=0.01,
            min_value=0.001,
            max_value=0.1,
            description="Batch timeout for inference",
            config_type=ConfigType.INFERENCE,
            impact_weight=1.5
        ))
        
        self.register(ConfigParameter(
            name="max_inference_batch_size",
            type=int,
            default=32,
            min_value=1,
            max_value=128,
            description="Maximum batch size for inference",
            config_type=ConfigType.INFERENCE,
            impact_weight=1.5
        ))
        
        # System parameters
        self.register(ConfigParameter(
            name="thread_pool_size",
            type=int,
            default=8,
            min_value=2,
            max_value=32,
            description="Thread pool size",
            config_type=ConfigType.SYSTEM,
            impact_weight=1.0
        ))
        
        self.register(ConfigParameter(
            name="async_queue_size",
            type=int,
            default=1000,
            min_value=100,
            max_value=10000,
            description="Async queue size",
            config_type=ConfigType.SYSTEM,
            impact_weight=0.5
        ))
    
    def register(self, parameter: ConfigParameter):
        """Register a configuration parameter"""
        self.parameters[parameter.name] = parameter
        self.groups[parameter.config_type.value].append(parameter.name)
    
    def get_parameter(self, name: str) -> Optional[ConfigParameter]:
        """Get parameter by name"""
        return self.parameters.get(name)
    
    def get_group_parameters(self, group: str) -> List[ConfigParameter]:
        """Get all parameters in a group"""
        return [self.parameters[name] for name in self.groups[group]]
    
    def add_dependency(self, param: str, depends_on: str):
        """Add parameter dependency"""
        self.dependencies[param].append(depends_on)
    
    def get_dependencies(self, param: str) -> List[str]:
        """Get parameter dependencies"""
        return self.dependencies.get(param, [])

class ConfigurationOptimizer:
    """Optimizes configuration parameters using Optuna"""
    
    def __init__(self, registry: ConfigurationRegistry, 
                 goal: OptimizationGoal = OptimizationGoal.BALANCE_ALL):
        self.registry = registry
        self.goal = goal
        self.study = None
        self.optimization_history = []
        self.best_config = None
        self.best_score = float('-inf')
        
        # Performance evaluator
        self.evaluator = None
        
        # Optimization settings
        self.n_trials = 100
        self.timeout = 3600  # 1 hour
        self.n_jobs = 1
        
        # Statistics
        self.stats = {
            'trials_completed': 0,
            'best_score': float('-inf'),
            'optimization_time': 0.0,
            'convergence_trials': 0
        }
    
    def set_evaluator(self, evaluator: Callable[[Dict[str, Any]], PerformanceMetrics]):
        """Set performance evaluator function"""
        self.evaluator = evaluator
    
    def create_study(self, study_name: str = "config_optimization"):
        """Create Optuna study"""
        if self.goal == OptimizationGoal.MINIMIZE_LATENCY:
            direction = "minimize"
        elif self.goal == OptimizationGoal.MINIMIZE_MEMORY:
            direction = "minimize"
        else:
            direction = "maximize"
        
        self.study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for optimization"""
        # Generate configuration
        config = {}
        for param in self.registry.parameters.values():
            config[param.name] = param.suggest_value(trial)
        
        # Validate dependencies
        if not self._validate_dependencies(config):
            raise optuna.exceptions.TrialPruned()
        
        # Evaluate configuration
        if self.evaluator is None:
            raise ValueError("No evaluator set")
        
        try:
            metrics = self.evaluator(config)
            score = metrics.composite_score(self.goal)
            
            # Update best configuration
            if score > self.best_score:
                self.best_score = score
                self.best_config = config.copy()
            
            # Store in history
            self.optimization_history.append({
                'trial': trial.number,
                'config': config,
                'metrics': metrics,
                'score': score,
                'timestamp': time.time()
            })
            
            return score
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            raise optuna.exceptions.TrialPruned()
    
    def _validate_dependencies(self, config: Dict[str, Any]) -> bool:
        """Validate configuration dependencies"""
        for param, deps in self.registry.dependencies.items():
            if param in config:
                for dep in deps:
                    if dep not in config:
                        return False
                    # Add specific dependency validation logic here
        return True
    
    def optimize(self, n_trials: int = 100, timeout: float = 3600) -> Dict[str, Any]:
        """Run optimization"""
        if self.study is None:
            self.create_study()
        
        start_time = time.time()
        
        try:
            self.study.optimize(
                self.objective,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=self.n_jobs
            )
            
            # Update statistics
            self.stats['trials_completed'] = len(self.study.trials)
            self.stats['best_score'] = self.best_score
            self.stats['optimization_time'] = time.time() - start_time
            
            # Find convergence point
            self._analyze_convergence()
            
            return self.best_config
            
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")
            return self.best_config
    
    def _analyze_convergence(self):
        """Analyze convergence characteristics"""
        if len(self.optimization_history) < 10:
            return
        
        # Find when best score plateaued
        scores = [h['score'] for h in self.optimization_history]
        best_scores = np.maximum.accumulate(scores)
        
        # Find convergence point (when improvement < 1% for 10 consecutive trials)
        convergence_threshold = 0.01
        consecutive_trials = 10
        
        for i in range(consecutive_trials, len(best_scores)):
            recent_scores = best_scores[i-consecutive_trials:i]
            if len(set(recent_scores)) == 1:  # No improvement
                improvement = (best_scores[i] - recent_scores[0]) / recent_scores[0]
                if improvement < convergence_threshold:
                    self.stats['convergence_trials'] = i
                    break
    
    def get_best_config(self) -> Dict[str, Any]:
        """Get best configuration found"""
        return self.best_config
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history"""
        return self.optimization_history
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return self.stats

class AdaptiveConfigManager:
    """Manages adaptive configuration changes based on runtime performance"""
    
    def __init__(self, registry: ConfigurationRegistry):
        self.registry = registry
        self.current_config = {}
        self.performance_history = deque(maxlen=100)
        self.adaptation_rules = []
        
        # Adaptation settings
        self.adaptation_interval = 60  # seconds
        self.performance_threshold = 0.1  # 10% change threshold
        self.adaptation_step_size = 0.1  # 10% step size
        
        # Monitoring
        self.monitor_thread = None
        self.monitoring_active = False
        
        # Statistics
        self.stats = {
            'adaptations_made': 0,
            'performance_improvements': 0,
            'performance_degradations': 0,
            'total_monitoring_time': 0.0
        }
        
        # Initialize with default configuration
        self._initialize_default_config()
    
    def _initialize_default_config(self):
        """Initialize with default configuration"""
        for param in self.registry.parameters.values():
            self.current_config[param.name] = param.default
    
    def add_adaptation_rule(self, condition: Callable[[PerformanceMetrics], bool],
                           action: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """Add adaptation rule"""
        self.adaptation_rules.append({
            'condition': condition,
            'action': action
        })
    
    def start_monitoring(self, evaluator: Callable[[Dict[str, Any]], PerformanceMetrics]):
        """Start adaptive monitoring"""
        self.evaluator = evaluator
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_worker, daemon=True)
        self.monitor_thread.start()
        logger.info("Adaptive monitoring started")
    
    def stop_monitoring(self):
        """Stop adaptive monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Adaptive monitoring stopped")
    
    def _monitor_worker(self):
        """Background monitoring worker"""
        start_time = time.time()
        
        while self.monitoring_active:
            try:
                # Evaluate current performance
                metrics = self.evaluator(self.current_config)
                self.performance_history.append(metrics)
                
                # Check adaptation rules
                for rule in self.adaptation_rules:
                    if rule['condition'](metrics):
                        new_config = rule['action'](self.current_config.copy())
                        if self._validate_config(new_config):
                            self._apply_config_change(new_config)
                            self.stats['adaptations_made'] += 1
                
                # Check for performance trends
                if len(self.performance_history) >= 2:
                    self._analyze_performance_trend()
                
                time.sleep(self.adaptation_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(10)
        
        self.stats['total_monitoring_time'] = time.time() - start_time
    
    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration"""
        for name, value in config.items():
            param = self.registry.get_parameter(name)
            if param and not param.validate(value):
                return False
        return True
    
    def _apply_config_change(self, new_config: Dict[str, Any]):
        """Apply configuration change"""
        old_config = self.current_config.copy()
        self.current_config.update(new_config)
        
        logger.info(f"Config adapted: {old_config} -> {new_config}")
        
        # Trigger configuration update callback if registered
        if hasattr(self, 'config_update_callback'):
            self.config_update_callback(self.current_config)
    
    def _analyze_performance_trend(self):
        """Analyze performance trend and adapt if needed"""
        if len(self.performance_history) < 10:
            return
        
        # Get recent performance
        recent_metrics = list(self.performance_history)[-10:]
        older_metrics = list(self.performance_history)[-20:-10]
        
        # Calculate average performance
        recent_avg = np.mean([m.composite_score(OptimizationGoal.BALANCE_ALL) 
                            for m in recent_metrics])
        older_avg = np.mean([m.composite_score(OptimizationGoal.BALANCE_ALL) 
                           for m in older_metrics])
        
        # Check for significant change
        if abs(recent_avg - older_avg) > self.performance_threshold:
            if recent_avg > older_avg:
                self.stats['performance_improvements'] += 1
            else:
                self.stats['performance_degradations'] += 1
                # Consider reverting recent changes or making adjustments
                self._handle_performance_degradation()
    
    def _handle_performance_degradation(self):
        """Handle performance degradation"""
        # Simple strategy: reduce batch size if performance is degrading
        if self.current_config.get('batch_size', 32) > 1:
            new_batch_size = max(1, int(self.current_config['batch_size'] * 0.9))
            self.current_config['batch_size'] = new_batch_size
            logger.info(f"Reduced batch size to {new_batch_size} due to performance degradation")
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.current_config.copy()
    
    def set_config(self, config: Dict[str, Any]):
        """Set configuration"""
        if self._validate_config(config):
            self.current_config.update(config)
        else:
            raise ValueError("Invalid configuration")
    
    def get_performance_history(self) -> List[PerformanceMetrics]:
        """Get performance history"""
        return list(self.performance_history)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adaptation statistics"""
        return self.stats

class ConfigurationProfiler:
    """Profiles configuration performance across different scenarios"""
    
    def __init__(self, registry: ConfigurationRegistry):
        self.registry = registry
        self.profiles = {}
        self.current_profile = None
        
        # Profiling scenarios
        self.scenarios = {
            'high_load': {'description': 'High load scenario'},
            'low_latency': {'description': 'Low latency scenario'},
            'memory_constrained': {'description': 'Memory constrained scenario'},
            'batch_processing': {'description': 'Batch processing scenario'}
        }
    
    def create_profile(self, name: str, config: Dict[str, Any], 
                      scenario: str = 'default'):
        """Create configuration profile"""
        self.profiles[name] = {
            'config': config,
            'scenario': scenario,
            'performance_history': [],
            'created_at': time.time()
        }
    
    def profile_config(self, config: Dict[str, Any], 
                      evaluator: Callable[[Dict[str, Any]], PerformanceMetrics],
                      n_runs: int = 10) -> Dict[str, Any]:
        """Profile configuration performance"""
        results = []
        
        for run in range(n_runs):
            start_time = time.time()
            metrics = evaluator(config)
            duration = time.time() - start_time
            
            results.append({
                'run': run,
                'metrics': metrics,
                'duration': duration
            })
        
        # Calculate statistics
        latencies = [r['metrics'].latency_ms for r in results]
        throughputs = [r['metrics'].throughput_qps for r in results]
        memory_usage = [r['metrics'].memory_mb for r in results]
        
        profile_stats = {
            'config': config,
            'n_runs': n_runs,
            'avg_latency': np.mean(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99),
            'avg_throughput': np.mean(throughputs),
            'std_throughput': np.std(throughputs),
            'avg_memory': np.mean(memory_usage),
            'max_memory': np.max(memory_usage),
            'stability_score': 1.0 - (np.std(latencies) / np.mean(latencies))
        }
        
        return profile_stats
    
    def compare_profiles(self, profile1: str, profile2: str) -> Dict[str, Any]:
        """Compare two configuration profiles"""
        if profile1 not in self.profiles or profile2 not in self.profiles:
            raise ValueError("Profile not found")
        
        p1 = self.profiles[profile1]
        p2 = self.profiles[profile2]
        
        # Compare configurations
        config_diff = {}
        for key in set(p1['config'].keys()) | set(p2['config'].keys()):
            if p1['config'].get(key) != p2['config'].get(key):
                config_diff[key] = {
                    'profile1': p1['config'].get(key),
                    'profile2': p2['config'].get(key)
                }
        
        return {
            'config_differences': config_diff,
            'profile1_scenario': p1['scenario'],
            'profile2_scenario': p2['scenario']
        }
    
    def get_best_profile(self, goal: OptimizationGoal) -> Optional[str]:
        """Get best profile for given goal"""
        if not self.profiles:
            return None
        
        best_profile = None
        best_score = float('-inf')
        
        for name, profile in self.profiles.items():
            if profile['performance_history']:
                latest_metrics = profile['performance_history'][-1]
                score = latest_metrics.composite_score(goal)
                if score > best_score:
                    best_score = score
                    best_profile = name
        
        return best_profile
    
    def export_profiles(self, filepath: Path):
        """Export profiles to file"""
        with open(filepath, 'w') as f:
            json.dump(self.profiles, f, indent=2, default=str)
    
    def import_profiles(self, filepath: Path):
        """Import profiles from file"""
        with open(filepath, 'r') as f:
            self.profiles = json.load(f)

class ConfigurationManager:
    """Main configuration management system"""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.registry = ConfigurationRegistry()
        self.optimizer = ConfigurationOptimizer(self.registry)
        self.adaptive_manager = AdaptiveConfigManager(self.registry)
        self.profiler = ConfigurationProfiler(self.registry)
        
        # Configuration storage
        self.config_file = config_file or Path("config.yaml")
        self.config_history = []
        
        # Load configuration if file exists
        if self.config_file.exists():
            self.load_config()
    
    def optimize_configuration(self, evaluator: Callable[[Dict[str, Any]], PerformanceMetrics],
                             goal: OptimizationGoal = OptimizationGoal.BALANCE_ALL,
                             n_trials: int = 100) -> Dict[str, Any]:
        """Optimize configuration for given goal"""
        self.optimizer.goal = goal
        self.optimizer.set_evaluator(evaluator)
        
        best_config = self.optimizer.optimize(n_trials=n_trials)
        
        # Save optimized configuration
        self.save_config(best_config)
        
        return best_config
    
    def start_adaptive_tuning(self, evaluator: Callable[[Dict[str, Any]], PerformanceMetrics]):
        """Start adaptive configuration tuning"""
        self.adaptive_manager.start_monitoring(evaluator)
    
    def stop_adaptive_tuning(self):
        """Stop adaptive configuration tuning"""
        self.adaptive_manager.stop_monitoring()
    
    def profile_configuration(self, config: Dict[str, Any],
                            evaluator: Callable[[Dict[str, Any]], PerformanceMetrics],
                            profile_name: str = None) -> Dict[str, Any]:
        """Profile configuration performance"""
        profile_stats = self.profiler.profile_config(config, evaluator)
        
        if profile_name:
            self.profiler.create_profile(profile_name, config)
        
        return profile_stats
    
    def get_recommended_config(self, scenario: str = 'default') -> Dict[str, Any]:
        """Get recommended configuration for scenario"""
        # Define scenario-specific recommendations
        scenarios = {
            'high_throughput': {
                'batch_size': 64,
                'num_workers': 8,
                'enable_jit': True,
                'optimization_level': 'O3'
            },
            'low_latency': {
                'batch_size': 1,
                'num_workers': 4,
                'enable_jit': True,
                'inference_batch_timeout': 0.001
            },
            'memory_efficient': {
                'batch_size': 16,
                'cache_size': 500,
                'memory_pool_size': 200,
                'enable_quantization': True
            }
        }
        
        base_config = {param.name: param.default 
                      for param in self.registry.parameters.values()}
        
        if scenario in scenarios:
            base_config.update(scenarios[scenario])
        
        return base_config
    
    def save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        config_data = {
            'config': config,
            'timestamp': time.time(),
            'optimization_stats': self.optimizer.get_stats(),
            'adaptive_stats': self.adaptive_manager.get_stats()
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        
        self.config_history.append(config_data)
        logger.info(f"Configuration saved to {self.config_file}")
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        with open(self.config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        if isinstance(config_data, dict) and 'config' in config_data:
            config = config_data['config']
        else:
            config = config_data
        
        # Validate configuration
        if self._validate_config(config):
            self.adaptive_manager.set_config(config)
            logger.info(f"Configuration loaded from {self.config_file}")
            return config
        else:
            logger.error("Invalid configuration in file")
            return self.get_recommended_config()
    
    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration"""
        for name, value in config.items():
            param = self.registry.get_parameter(name)
            if param and not param.validate(value):
                logger.error(f"Invalid value for {name}: {value}")
                return False
        return True
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            'registry_stats': {
                'total_parameters': len(self.registry.parameters),
                'parameter_groups': dict(self.registry.groups),
                'dependencies': dict(self.registry.dependencies)
            },
            'optimization_stats': self.optimizer.get_stats(),
            'adaptive_stats': self.adaptive_manager.get_stats(),
            'current_config': self.adaptive_manager.get_current_config(),
            'config_history_length': len(self.config_history)
        }

# Context manager for configuration
@contextmanager
def config_context(config_manager: ConfigurationManager, config: Dict[str, Any]):
    """Context manager for temporary configuration"""
    original_config = config_manager.adaptive_manager.get_current_config()
    try:
        config_manager.adaptive_manager.set_config(config)
        yield config
    finally:
        config_manager.adaptive_manager.set_config(original_config)

# Example usage and testing
if __name__ == "__main__":
    # Example evaluator function
    def example_evaluator(config: Dict[str, Any]) -> PerformanceMetrics:
        """Example performance evaluator"""
        # Simulate performance evaluation
        latency = 10.0 / config.get('batch_size', 32)
        throughput = config.get('batch_size', 32) * config.get('num_workers', 4)
        memory = config.get('cache_size', 1000) * 0.1
        
        return PerformanceMetrics(
            latency_ms=latency,
            throughput_qps=throughput,
            memory_mb=memory,
            cpu_usage=50.0,
            accuracy=0.95,
            stability_score=0.9
        )
    
    # Create configuration manager
    config_manager = ConfigurationManager()
    
    # Optimize configuration
    best_config = config_manager.optimize_configuration(
        evaluator=example_evaluator,
        goal=OptimizationGoal.MAXIMIZE_THROUGHPUT,
        n_trials=50
    )
    
    print(f"Best configuration: {best_config}")
    
    # Profile configuration
    profile_stats = config_manager.profile_configuration(
        config=best_config,
        evaluator=example_evaluator,
        profile_name="optimized_config"
    )
    
    print(f"Profile stats: {profile_stats}")
    
    # Get comprehensive stats
    stats = config_manager.get_comprehensive_stats()
    print(f"Comprehensive stats: {json.dumps(stats, indent=2, default=str)}")
    
    logger.info("Configuration tuning system test completed")