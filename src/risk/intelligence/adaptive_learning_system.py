"""
Adaptive Learning Framework - Intelligence Integration Layer

Continuous optimization and self-improvement system for intelligence coordination.
Implements meta-learning, performance optimization, and adaptive parameter tuning.

Features:
- Real-time performance monitoring and optimization
- Automatic parameter tuning based on market conditions
- A/B testing framework for intelligence component improvements
- Continuous learning from human decisions and outcomes
- Self-optimization with performance feedback loops
- Multi-objective optimization for competing goals

Architecture:
- Performance Monitor: Real-time metric tracking and analysis
- Parameter Optimizer: Adaptive tuning of system parameters
- A/B Testing Engine: Systematic improvement validation
- Learning Algorithms: Online learning and meta-learning
- Feedback Loop: Continuous improvement based on outcomes
"""

import asyncio
import threading
import time
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import structlog
from concurrent.futures import ThreadPoolExecutor
import queue
from collections import defaultdict, deque
import json
import math
from abc import ABC, abstractmethod
from scipy import optimize
from scipy.stats import ttest_ind, mannwhitneyu
import random

logger = structlog.get_logger()


class OptimizationMetric(Enum):
    """Metrics to optimize"""
    COORDINATION_LATENCY = "coordination_latency"
    FUSION_ACCURACY = "fusion_accuracy"
    CONFLICT_RESOLUTION_RATE = "conflict_resolution_rate"
    AGENT_CREDIBILITY_ACCURACY = "agent_credibility_accuracy"
    EMERGENCY_RESPONSE_TIME = "emergency_response_time"
    HUMAN_SATISFACTION = "human_satisfaction"
    OVERALL_PERFORMANCE = "overall_performance"


class LearningAlgorithm(Enum):
    """Learning algorithms for optimization"""
    GRADIENT_DESCENT = "gradient_descent"
    GENETIC_ALGORITHM = "genetic_algorithm"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    THOMPSON_SAMPLING = "thompson_sampling"
    CONTEXTUAL_BANDIT = "contextual_bandit"
    EVOLUTIONARY_STRATEGY = "evolutionary_strategy"


class AdaptationStrategy(Enum):
    """Strategies for parameter adaptation"""
    AGGRESSIVE = "aggressive"     # Fast adaptation, higher learning rates
    CONSERVATIVE = "conservative" # Slow adaptation, lower learning rates
    BALANCED = "balanced"         # Moderate adaptation
    CONTEXT_AWARE = "context_aware" # Adaptation based on market conditions


@dataclass
class PerformanceMetric:
    """Performance metric with metadata"""
    metric_name: str
    value: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    confidence_interval: Optional[Tuple[float, float]] = None
    target_value: Optional[float] = None
    improvement_rate: Optional[float] = None


@dataclass
class OptimizationParameter:
    """Parameter to optimize"""
    name: str
    current_value: float
    min_value: float
    max_value: float
    step_size: float
    learning_rate: float
    momentum: float = 0.0
    last_gradient: float = 0.0
    best_value: float = None
    best_performance: float = None
    update_count: int = 0


@dataclass
class ABTestConfiguration:
    """A/B test configuration"""
    test_id: str
    name: str
    description: str
    parameter_name: str
    control_value: float
    treatment_value: float
    allocation_ratio: float = 0.5  # 50% treatment
    min_sample_size: int = 100
    significance_level: float = 0.05
    test_duration_hours: int = 24
    start_time: datetime = field(default_factory=datetime.now)
    status: str = "active"  # active, completed, stopped


@dataclass
class ABTestResult:
    """A/B test result"""
    test_id: str
    control_performance: List[float]
    treatment_performance: List[float]
    statistical_significance: bool
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    recommendation: str
    completion_time: datetime


@dataclass
class LearningContext:
    """Context for learning decisions"""
    market_volatility: float
    trading_volume: float
    time_of_day: str
    market_phase: str  # opening, trading, closing
    system_load: float
    recent_performance: float
    error_rate: float


class PerformanceMonitor:
    """Real-time performance monitoring system"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.targets: Dict[str, float] = {}
        self.alerts: List[str] = []
        self.last_alert_time: Dict[str, datetime] = {}
        
    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric"""
        self.metrics[metric.metric_name].append(metric)
        
        # Check for alerts
        if metric.target_value is not None:
            self._check_performance_alert(metric)
    
    def get_recent_performance(self, 
                             metric_name: str, 
                             lookback_minutes: int = 10) -> List[float]:
        """Get recent performance values"""
        cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)
        
        recent_values = []
        for metric in self.metrics[metric_name]:
            if metric.timestamp >= cutoff_time:
                recent_values.append(metric.value)
        
        return recent_values
    
    def get_performance_trend(self, metric_name: str) -> Tuple[float, str]:
        """Get performance trend (slope and direction)"""
        values = [m.value for m in list(self.metrics[metric_name])[-50:]]  # Last 50 values
        
        if len(values) < 10:
            return 0.0, "insufficient_data"
        
        # Calculate linear trend
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        if slope > 0.01:
            direction = "improving"
        elif slope < -0.01:
            direction = "degrading"
        else:
            direction = "stable"
        
        return slope, direction
    
    def _check_performance_alert(self, metric: PerformanceMetric):
        """Check if metric triggers performance alert"""
        if metric.target_value is None:
            return
        
        # Avoid alert spam
        last_alert = self.last_alert_time.get(metric.metric_name)
        if last_alert and (datetime.now() - last_alert).total_seconds() < 300:  # 5 minutes
            return
        
        deviation = abs(metric.value - metric.target_value) / metric.target_value
        
        if deviation > 0.2:  # 20% deviation threshold
            alert_msg = f"Performance alert: {metric.metric_name} = {metric.value:.3f} " \
                       f"(target: {metric.target_value:.3f}, deviation: {deviation:.1%})"
            self.alerts.append(alert_msg)
            self.last_alert_time[metric.metric_name] = datetime.now()
            logger.warning(alert_msg)


class ParameterOptimizer:
    """Adaptive parameter optimization system"""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.parameters: Dict[str, OptimizationParameter] = {}
        self.optimization_history: deque = deque(maxlen=1000)
        self.objective_function: Optional[Callable] = None
        
    def register_parameter(self, 
                         name: str,
                         initial_value: float,
                         min_value: float,
                         max_value: float,
                         step_size: float = 0.01):
        """Register parameter for optimization"""
        param = OptimizationParameter(
            name=name,
            current_value=initial_value,
            min_value=min_value,
            max_value=max_value,
            step_size=step_size,
            learning_rate=self.learning_rate,
            best_value=initial_value,
            best_performance=float('-inf')
        )
        
        self.parameters[name] = param
        logger.debug("Parameter registered for optimization", 
                    name=name, initial_value=initial_value)
    
    def set_objective_function(self, objective_fn: Callable[[Dict[str, float]], float]):
        """Set objective function to optimize"""
        self.objective_function = objective_fn
    
    def optimize_step(self, current_performance: float) -> Dict[str, float]:
        """Perform one optimization step"""
        if not self.objective_function:
            return {name: param.current_value for name, param in self.parameters.items()}
        
        # Update best performance tracking
        current_params = {name: param.current_value for name, param in self.parameters.items()}
        
        for param in self.parameters.values():
            if current_performance > param.best_performance:
                param.best_performance = current_performance
                param.best_value = param.current_value
        
        # Gradient-based optimization step
        gradients = self._estimate_gradients(current_params, current_performance)
        
        # Update parameters using gradients with momentum
        updated_params = {}
        for name, param in self.parameters.items():
            gradient = gradients.get(name, 0.0)
            
            # Momentum update
            param.last_gradient = 0.9 * param.last_gradient + 0.1 * gradient
            
            # Parameter update
            update = param.learning_rate * param.last_gradient
            new_value = param.current_value + update
            
            # Clamp to bounds
            new_value = np.clip(new_value, param.min_value, param.max_value)
            
            param.current_value = new_value
            param.update_count += 1
            updated_params[name] = new_value
        
        # Record optimization step
        self.optimization_history.append({
            'timestamp': datetime.now(),
            'parameters': dict(updated_params),
            'performance': current_performance,
            'gradients': dict(gradients)
        })
        
        return updated_params
    
    def _estimate_gradients(self, 
                          current_params: Dict[str, float], 
                          current_performance: float) -> Dict[str, float]:
        """Estimate gradients using finite differences"""
        gradients = {}
        
        for name, param in self.parameters.items():
            # Forward difference
            test_params = current_params.copy()
            test_params[name] += param.step_size
            
            try:
                forward_performance = self.objective_function(test_params)
                gradient = (forward_performance - current_performance) / param.step_size
                gradients[name] = gradient
            except Exception as e:
                logger.warning("Error estimating gradient", parameter=name, error=str(e))
                gradients[name] = 0.0
        
        return gradients
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get optimization status and statistics"""
        param_status = {}
        for name, param in self.parameters.items():
            param_status[name] = {
                'current_value': param.current_value,
                'best_value': param.best_value,
                'best_performance': param.best_performance,
                'update_count': param.update_count,
                'last_gradient': param.last_gradient
            }
        
        return {
            'parameters': param_status,
            'optimization_steps': len(self.optimization_history),
            'learning_rate': self.learning_rate
        }


class ABTestingEngine:
    """A/B testing framework for systematic improvements"""
    
    def __init__(self):
        self.active_tests: Dict[str, ABTestConfiguration] = {}
        self.completed_tests: Dict[str, ABTestResult] = {}
        self.test_assignments: Dict[str, str] = {}  # user/session -> test_id
        self.test_data: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {'control': [], 'treatment': []})
        
    def create_ab_test(self, config: ABTestConfiguration) -> bool:
        """Create new A/B test"""
        if config.test_id in self.active_tests:
            logger.warning("A/B test already exists", test_id=config.test_id)
            return False
        
        self.active_tests[config.test_id] = config
        logger.info("A/B test created", 
                   test_id=config.test_id,
                   parameter=config.parameter_name,
                   control=config.control_value,
                   treatment=config.treatment_value)
        return True
    
    def assign_test_group(self, session_id: str, test_id: str) -> str:
        """Assign session to test group"""
        if test_id not in self.active_tests:
            return "control"
        
        config = self.active_tests[test_id]
        
        # Check if already assigned
        if session_id in self.test_assignments:
            existing_assignment = self.test_assignments[session_id]
            if existing_assignment.startswith(test_id):
                return existing_assignment.split('_')[1]
        
        # Random assignment based on allocation ratio
        if random.random() < config.allocation_ratio:
            group = "treatment"
        else:
            group = "control"
        
        self.test_assignments[session_id] = f"{test_id}_{group}"
        return group
    
    def record_test_outcome(self, test_id: str, group: str, outcome: float):
        """Record outcome for A/B test"""
        if test_id not in self.active_tests:
            return
        
        self.test_data[test_id][group].append(outcome)
    
    def analyze_test(self, test_id: str) -> Optional[ABTestResult]:
        """Analyze A/B test results"""
        if test_id not in self.active_tests:
            return None
        
        config = self.active_tests[test_id]
        control_data = self.test_data[test_id]['control']
        treatment_data = self.test_data[test_id]['treatment']
        
        # Check minimum sample size
        if len(control_data) < config.min_sample_size or len(treatment_data) < config.min_sample_size:
            return None
        
        # Statistical test
        try:
            # Use t-test for normally distributed data, Mann-Whitney U for non-normal
            if len(control_data) > 30 and len(treatment_data) > 30:
                statistic, p_value = ttest_ind(control_data, treatment_data)
            else:
                statistic, p_value = mannwhitneyu(control_data, treatment_data, alternative='two-sided')
                p_value *= 2  # Two-sided test
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(control_data) - 1) * np.var(control_data) + 
                                (len(treatment_data) - 1) * np.var(treatment_data)) / 
                               (len(control_data) + len(treatment_data) - 2))
            
            if pooled_std > 0:
                effect_size = (np.mean(treatment_data) - np.mean(control_data)) / pooled_std
            else:
                effect_size = 0.0
            
            # Confidence interval for effect size
            se_effect = np.sqrt((len(control_data) + len(treatment_data)) / 
                              (len(control_data) * len(treatment_data)) + 
                              effect_size**2 / (2 * (len(control_data) + len(treatment_data))))
            
            ci_lower = effect_size - 1.96 * se_effect
            ci_upper = effect_size + 1.96 * se_effect
            
            # Statistical significance
            significant = p_value < config.significance_level
            
            # Recommendation
            if significant and effect_size > 0.2:  # Meaningful effect
                if np.mean(treatment_data) > np.mean(control_data):
                    recommendation = f"Adopt treatment value ({config.treatment_value})"
                else:
                    recommendation = f"Keep control value ({config.control_value})"
            else:
                recommendation = "No significant difference detected"
            
            result = ABTestResult(
                test_id=test_id,
                control_performance=control_data,
                treatment_performance=treatment_data,
                statistical_significance=significant,
                p_value=p_value,
                effect_size=effect_size,
                confidence_interval=(ci_lower, ci_upper),
                recommendation=recommendation,
                completion_time=datetime.now()
            )
            
            return result
            
        except Exception as e:
            logger.error("Error analyzing A/B test", test_id=test_id, error=str(e))
            return None
    
    def complete_test(self, test_id: str) -> Optional[ABTestResult]:
        """Complete A/B test and move to completed"""
        result = self.analyze_test(test_id)
        
        if result and test_id in self.active_tests:
            self.completed_tests[test_id] = result
            del self.active_tests[test_id]
            
            # Clean up assignments
            assignments_to_remove = [k for k, v in self.test_assignments.items() 
                                   if v.startswith(test_id)]
            for assignment in assignments_to_remove:
                del self.test_assignments[assignment]
            
            logger.info("A/B test completed",
                       test_id=test_id,
                       significant=result.statistical_significance,
                       recommendation=result.recommendation)
        
        return result


class AdaptiveLearningSystem:
    """
    Adaptive Learning Framework for Intelligence Coordination
    
    Provides continuous optimization, A/B testing, and meta-learning for
    improving intelligence coordination performance over time.
    """
    
    def __init__(self, adaptation_strategy: AdaptationStrategy = AdaptationStrategy.BALANCED):
        self.adaptation_strategy = adaptation_strategy
        
        # Core components
        self.performance_monitor = PerformanceMonitor()
        self.parameter_optimizer = ParameterOptimizer()
        self.ab_testing_engine = ABTestingEngine()
        
        # Learning state
        self.learning_enabled = True
        self.learning_history: deque = deque(maxlen=1000)
        self.adaptation_count = 0
        self.improvement_rate = 0.0
        
        # Learning context
        self.current_context = LearningContext(
            market_volatility=0.0,
            trading_volume=0.0,
            time_of_day="trading",
            market_phase="trading",
            system_load=0.0,
            recent_performance=0.0,
            error_rate=0.0
        )
        
        # Threading for background learning
        self.learning_thread = None
        self.running = False
        
        # Performance targets
        self.performance_targets = {
            OptimizationMetric.COORDINATION_LATENCY: 5.0,  # 5ms target
            OptimizationMetric.FUSION_ACCURACY: 0.85,     # 85% accuracy
            OptimizationMetric.CONFLICT_RESOLUTION_RATE: 0.90,  # 90% resolution
            OptimizationMetric.EMERGENCY_RESPONSE_TIME: 1.0,    # 1ms emergency response
        }
        
        self._initialize_parameters()
        
        logger.info("Adaptive learning system initialized", 
                   strategy=adaptation_strategy.value)
    
    def _initialize_parameters(self):
        """Initialize optimization parameters"""
        # Coordination parameters
        self.parameter_optimizer.register_parameter(
            "coordination_timeout_ms", 5.0, 1.0, 20.0, 0.5
        )
        self.parameter_optimizer.register_parameter(
            "fusion_confidence_threshold", 0.6, 0.3, 0.95, 0.05
        )
        self.parameter_optimizer.register_parameter(
            "conflict_resolution_threshold", 0.5, 0.1, 0.9, 0.05
        )
        self.parameter_optimizer.register_parameter(
            "emergency_priority_weight", 2.0, 1.0, 5.0, 0.1
        )
        
        # Set objective function for multi-objective optimization
        self.parameter_optimizer.set_objective_function(self._compute_objective_score)
    
    def start_learning(self):
        """Start background learning process"""
        if self.learning_thread and self.learning_thread.is_alive():
            return
        
        self.running = True
        self.learning_thread = threading.Thread(
            target=self._learning_loop,
            name="adaptive_learning"
        )
        self.learning_thread.start()
        
        logger.info("Adaptive learning started")
    
    def stop_learning(self):
        """Stop background learning process"""
        self.running = False
        
        if self.learning_thread and self.learning_thread.is_alive():
            self.learning_thread.join(timeout=2.0)
        
        logger.info("Adaptive learning stopped")
    
    def _learning_loop(self):
        """Main learning loop"""
        while self.running:
            try:
                # Update learning context
                self._update_learning_context()
                
                # Check for completed A/B tests
                self._check_ab_tests()
                
                # Perform parameter optimization
                if self.learning_enabled:
                    self._optimize_parameters()
                
                # Record learning step
                self._record_learning_step()
                
                # Sleep based on adaptation strategy
                sleep_time = self._get_learning_interval()
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error("Error in learning loop", error=str(e))
                time.sleep(5.0)  # Longer sleep on error
    
    def record_performance(self, 
                         metric_type: OptimizationMetric,
                         value: float,
                         context: Optional[Dict[str, Any]] = None):
        """Record performance metric for learning"""
        metric = PerformanceMetric(
            metric_name=metric_type.value,
            value=value,
            timestamp=datetime.now(),
            context=context or {},
            target_value=self.performance_targets.get(metric_type)
        )
        
        self.performance_monitor.record_metric(metric)
        
        # Update learning context
        if metric_type == OptimizationMetric.OVERALL_PERFORMANCE:
            self.current_context.recent_performance = value
    
    def adapt_to_context(self, context: LearningContext):
        """Adapt parameters based on current context"""
        if not self.learning_enabled:
            return
        
        self.current_context = context
        
        # Context-aware adaptation
        if self.adaptation_strategy == AdaptationStrategy.CONTEXT_AWARE:
            self._context_aware_adaptation()
    
    def _context_aware_adaptation(self):
        """Perform context-aware parameter adaptation"""
        # Adjust learning rates based on market conditions
        base_learning_rate = 0.01
        
        # Higher volatility -> more conservative learning
        volatility_factor = 1.0 - min(0.5, self.current_context.market_volatility)
        
        # Higher system load -> slower adaptation
        load_factor = 1.0 - min(0.3, self.current_context.system_load)
        
        # Recent poor performance -> more aggressive learning
        performance_factor = 1.0 + max(0.0, 0.7 - self.current_context.recent_performance)
        
        adjusted_learning_rate = base_learning_rate * volatility_factor * load_factor * performance_factor
        
        # Update optimizer learning rate
        self.parameter_optimizer.learning_rate = adjusted_learning_rate
        
        logger.debug("Learning rate adapted",
                    base=base_learning_rate,
                    adjusted=adjusted_learning_rate,
                    volatility_factor=volatility_factor,
                    load_factor=load_factor,
                    performance_factor=performance_factor)
    
    def _optimize_parameters(self):
        """Optimize system parameters"""
        # Get current performance score
        current_score = self._compute_current_performance()
        
        # Optimize parameters
        optimized_params = self.parameter_optimizer.optimize_step(current_score)
        
        self.adaptation_count += 1
        
        # Log significant parameter changes
        for name, value in optimized_params.items():
            param = self.parameter_optimizer.parameters[name]
            if abs(value - param.best_value) > param.step_size * 2:
                logger.debug("Parameter significantly updated",
                           parameter=name,
                           old_value=param.best_value,
                           new_value=value)
    
    def _compute_current_performance(self) -> float:
        """Compute current overall performance score"""
        scores = []
        weights = []
        
        # Latency performance (lower is better)
        latency_metrics = self.performance_monitor.get_recent_performance(
            OptimizationMetric.COORDINATION_LATENCY.value, 5
        )
        if latency_metrics:
            avg_latency = np.mean(latency_metrics)
            target_latency = self.performance_targets[OptimizationMetric.COORDINATION_LATENCY]
            latency_score = max(0.0, 1.0 - avg_latency / target_latency)
            scores.append(latency_score)
            weights.append(0.3)
        
        # Accuracy performance (higher is better)
        accuracy_metrics = self.performance_monitor.get_recent_performance(
            OptimizationMetric.FUSION_ACCURACY.value, 5
        )
        if accuracy_metrics:
            avg_accuracy = np.mean(accuracy_metrics)
            scores.append(avg_accuracy)
            weights.append(0.4)
        
        # Conflict resolution performance (higher is better)
        conflict_metrics = self.performance_monitor.get_recent_performance(
            OptimizationMetric.CONFLICT_RESOLUTION_RATE.value, 5
        )
        if conflict_metrics:
            avg_conflict_resolution = np.mean(conflict_metrics)
            scores.append(avg_conflict_resolution)
            weights.append(0.3)
        
        # Weighted average performance
        if scores and weights:
            performance = np.average(scores, weights=weights)
        else:
            performance = 0.5  # Default neutral performance
        
        return performance
    
    def _compute_objective_score(self, parameters: Dict[str, float]) -> float:
        """Compute objective score for parameter optimization"""
        # This would be called with test parameters to evaluate performance
        # For now, return current performance (would be replaced with actual evaluation)
        return self._compute_current_performance()
    
    def _update_learning_context(self):
        """Update learning context from system state"""
        # This would be updated with real system metrics
        # For now, use placeholder values
        pass
    
    def _check_ab_tests(self):
        """Check and complete finished A/B tests"""
        tests_to_complete = []
        
        for test_id, config in self.ab_testing_engine.active_tests.items():
            # Check if test duration exceeded
            elapsed_hours = (datetime.now() - config.start_time).total_seconds() / 3600
            
            if elapsed_hours >= config.test_duration_hours:
                tests_to_complete.append(test_id)
        
        # Complete finished tests
        for test_id in tests_to_complete:
            result = self.ab_testing_engine.complete_test(test_id)
            if result:
                self._apply_ab_test_result(result)
    
    def _apply_ab_test_result(self, result: ABTestResult):
        """Apply A/B test result to system parameters"""
        if result.statistical_significance and result.effect_size > 0.2:
            # Apply significant improvements
            if "treatment" in result.recommendation.lower():
                logger.info("Applying A/B test improvement",
                           test_id=result.test_id,
                           effect_size=result.effect_size,
                           p_value=result.p_value)
                # Parameter update would be applied here
    
    def _record_learning_step(self):
        """Record learning step for history"""
        step_record = {
            'timestamp': datetime.now(),
            'adaptation_count': self.adaptation_count,
            'current_performance': self._compute_current_performance(),
            'learning_context': {
                'market_volatility': self.current_context.market_volatility,
                'system_load': self.current_context.system_load,
                'recent_performance': self.current_context.recent_performance
            },
            'parameter_status': self.parameter_optimizer.get_optimization_status()
        }
        
        self.learning_history.append(step_record)
    
    def _get_learning_interval(self) -> float:
        """Get learning interval based on adaptation strategy"""
        base_interval = 10.0  # 10 seconds
        
        if self.adaptation_strategy == AdaptationStrategy.AGGRESSIVE:
            return base_interval * 0.5
        elif self.adaptation_strategy == AdaptationStrategy.CONSERVATIVE:
            return base_interval * 2.0
        elif self.adaptation_strategy == AdaptationStrategy.CONTEXT_AWARE:
            # Adapt based on context
            if self.current_context.recent_performance < 0.5:
                return base_interval * 0.7  # Faster learning when performing poorly
            else:
                return base_interval * 1.2  # Slower when performing well
        else:  # BALANCED
            return base_interval
    
    def create_ab_test(self, 
                      name: str,
                      parameter_name: str,
                      control_value: float,
                      treatment_value: float,
                      duration_hours: int = 24) -> str:
        """Create new A/B test"""
        test_id = f"test_{int(time.time())}_{parameter_name}"
        
        config = ABTestConfiguration(
            test_id=test_id,
            name=name,
            description=f"Testing {parameter_name}: {control_value} vs {treatment_value}",
            parameter_name=parameter_name,
            control_value=control_value,
            treatment_value=treatment_value,
            test_duration_hours=duration_hours
        )
        
        if self.ab_testing_engine.create_ab_test(config):
            return test_id
        else:
            return ""
    
    def get_learning_metrics(self) -> Dict[str, Any]:
        """Get comprehensive learning metrics"""
        # Performance trends
        trends = {}
        for metric in OptimizationMetric:
            slope, direction = self.performance_monitor.get_performance_trend(metric.value)
            trends[metric.value] = {'slope': slope, 'direction': direction}
        
        # Optimization status
        optimization_status = self.parameter_optimizer.get_optimization_status()
        
        # A/B testing status
        ab_test_status = {
            'active_tests': len(self.ab_testing_engine.active_tests),
            'completed_tests': len(self.ab_testing_engine.completed_tests),
            'total_assignments': len(self.ab_testing_engine.test_assignments)
        }
        
        return {
            'learning_enabled': self.learning_enabled,
            'adaptation_strategy': self.adaptation_strategy.value,
            'adaptation_count': self.adaptation_count,
            'current_performance': self._compute_current_performance(),
            'improvement_rate': self.improvement_rate,
            'performance_trends': trends,
            'optimization_status': optimization_status,
            'ab_test_status': ab_test_status,
            'learning_context': {
                'market_volatility': self.current_context.market_volatility,
                'trading_volume': self.current_context.trading_volume,
                'system_load': self.current_context.system_load,
                'recent_performance': self.current_context.recent_performance
            },
            'alerts': self.performance_monitor.alerts[-10:]  # Last 10 alerts
        }
    
    def set_performance_target(self, metric: OptimizationMetric, target_value: float):
        """Set performance target for metric"""
        self.performance_targets[metric] = target_value
        logger.debug("Performance target updated", 
                    metric=metric.value, target=target_value)
    
    def enable_learning(self):
        """Enable adaptive learning"""
        self.learning_enabled = True
        logger.info("Adaptive learning enabled")
    
    def disable_learning(self):
        """Disable adaptive learning"""
        self.learning_enabled = False
        logger.info("Adaptive learning disabled")
    
    def reset_learning_state(self):
        """Reset all learning state"""
        self.learning_history.clear()
        self.adaptation_count = 0
        self.improvement_rate = 0.0
        
        # Reset optimizer
        for param in self.parameter_optimizer.parameters.values():
            param.update_count = 0
            param.last_gradient = 0.0
            param.best_performance = float('-inf')
        
        # Clear performance monitor
        self.performance_monitor.metrics.clear()
        self.performance_monitor.alerts.clear()
        
        logger.info("Learning state reset")