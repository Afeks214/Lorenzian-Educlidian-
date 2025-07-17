"""
Advanced Distribution Features with Load Balancing
Agent 2 Mission: Intelligent Load Balancing and Distribution

This module implements sophisticated load balancing algorithms, dynamic test 
redistribution, adaptive scheduling, and intelligent worker assignment for 
optimal parallel test execution performance.
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import heapq
import statistics
import json
import math
import random
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DistributionStrategy(Enum):
    """Distribution strategy enumeration"""
    ROUND_ROBIN = "round_robin"
    LOAD_BASED = "load_based"
    PERFORMANCE_BASED = "performance_based"
    ADAPTIVE = "adaptive"
    WORK_STEALING = "work_stealing"
    GENETIC_ALGORITHM = "genetic_algorithm"


@dataclass
class TestTask:
    """Represents a test task to be executed"""
    test_id: str
    test_name: str
    estimated_duration: float
    priority: int
    dependencies: List[str]
    resource_requirements: Dict[str, Any]
    assigned_worker: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    actual_duration: Optional[float] = None
    success: bool = False
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.resource_requirements is None:
            self.resource_requirements = {}
    
    @property
    def is_ready(self) -> bool:
        """Check if task is ready for execution"""
        return len(self.dependencies) == 0 and self.assigned_worker is None
    
    @property
    def is_running(self) -> bool:
        """Check if task is currently running"""
        return self.assigned_worker is not None and self.started_at is not None and self.completed_at is None
    
    @property
    def is_completed(self) -> bool:
        """Check if task is completed"""
        return self.completed_at is not None


@dataclass
class WorkerCapacity:
    """Represents worker capacity and current load"""
    worker_id: str
    max_concurrent_tests: int
    current_load: int
    cpu_capacity: float
    memory_capacity: float
    current_cpu_usage: float
    current_memory_usage: float
    performance_score: float
    specialty_tags: Set[str]
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def available_slots(self) -> int:
        """Get available test slots"""
        return max(0, self.max_concurrent_tests - self.current_load)
    
    @property
    def utilization(self) -> float:
        """Get current utilization (0-1)"""
        return self.current_load / self.max_concurrent_tests if self.max_concurrent_tests > 0 else 0
    
    @property
    def cpu_utilization(self) -> float:
        """Get CPU utilization (0-1)"""
        return self.current_cpu_usage / self.cpu_capacity if self.cpu_capacity > 0 else 0
    
    @property
    def memory_utilization(self) -> float:
        """Get memory utilization (0-1)"""
        return self.current_memory_usage / self.memory_capacity if self.memory_capacity > 0 else 0
    
    @property
    def overall_utilization(self) -> float:
        """Get overall utilization score"""
        return max(self.utilization, self.cpu_utilization, self.memory_utilization)
    
    def can_accept_task(self, task: TestTask) -> bool:
        """Check if worker can accept a new task"""
        if self.available_slots <= 0:
            return False
        
        # Check resource requirements
        required_cpu = task.resource_requirements.get('cpu', 0)
        required_memory = task.resource_requirements.get('memory', 0)
        
        if (self.current_cpu_usage + required_cpu > self.cpu_capacity or
            self.current_memory_usage + required_memory > self.memory_capacity):
            return False
        
        # Check specialty requirements
        required_tags = task.resource_requirements.get('specialty_tags', set())
        if required_tags and not required_tags.issubset(self.specialty_tags):
            return False
        
        return True


class LoadBalancingAlgorithm(ABC):
    """Abstract base class for load balancing algorithms"""
    
    @abstractmethod
    def assign_task(self, task: TestTask, available_workers: List[WorkerCapacity]) -> Optional[str]:
        """Assign a task to the best available worker"""
        pass
    
    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Get the name of this algorithm"""
        pass


class RoundRobinBalancer(LoadBalancingAlgorithm):
    """Simple round-robin load balancer"""
    
    def __init__(self):
        self.current_index = 0
    
    def assign_task(self, task: TestTask, available_workers: List[WorkerCapacity]) -> Optional[str]:
        """Assign task using round-robin strategy"""
        if not available_workers:
            return None
        
        # Filter workers that can accept the task
        capable_workers = [w for w in available_workers if w.can_accept_task(task)]
        
        if not capable_workers:
            return None
        
        # Select worker using round-robin
        worker = capable_workers[self.current_index % len(capable_workers)]
        self.current_index += 1
        
        return worker.worker_id
    
    def get_algorithm_name(self) -> str:
        return "Round Robin"


class LoadBasedBalancer(LoadBalancingAlgorithm):
    """Load-based balancer that assigns to least loaded worker"""
    
    def assign_task(self, task: TestTask, available_workers: List[WorkerCapacity]) -> Optional[str]:
        """Assign task to least loaded worker"""
        capable_workers = [w for w in available_workers if w.can_accept_task(task)]
        
        if not capable_workers:
            return None
        
        # Sort by utilization (ascending)
        capable_workers.sort(key=lambda w: w.overall_utilization)
        
        return capable_workers[0].worker_id
    
    def get_algorithm_name(self) -> str:
        return "Load Based"


class PerformanceBasedBalancer(LoadBalancingAlgorithm):
    """Performance-based balancer that considers worker performance scores"""
    
    def assign_task(self, task: TestTask, available_workers: List[WorkerCapacity]) -> Optional[str]:
        """Assign task to best performing available worker"""
        capable_workers = [w for w in available_workers if w.can_accept_task(task)]
        
        if not capable_workers:
            return None
        
        # Calculate score: performance_score / (utilization + 0.1)
        # Higher score is better
        def calculate_score(worker: WorkerCapacity) -> float:
            return worker.performance_score / (worker.overall_utilization + 0.1)
        
        best_worker = max(capable_workers, key=calculate_score)
        return best_worker.worker_id
    
    def get_algorithm_name(self) -> str:
        return "Performance Based"


class AdaptiveBalancer(LoadBalancingAlgorithm):
    """Adaptive balancer that learns from execution history"""
    
    def __init__(self):
        self.execution_history: Dict[str, List[float]] = defaultdict(list)
        self.worker_affinity: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.learning_rate = 0.1
    
    def assign_task(self, task: TestTask, available_workers: List[WorkerCapacity]) -> Optional[str]:
        """Assign task using adaptive strategy"""
        capable_workers = [w for w in available_workers if w.can_accept_task(task)]
        
        if not capable_workers:
            return None
        
        # Calculate adaptive score for each worker
        scores = {}
        for worker in capable_workers:
            score = self._calculate_adaptive_score(task, worker)
            scores[worker.worker_id] = score
        
        # Select worker with highest score
        best_worker_id = max(scores, key=scores.get)
        return best_worker_id
    
    def _calculate_adaptive_score(self, task: TestTask, worker: WorkerCapacity) -> float:
        """Calculate adaptive score for worker-task combination"""
        # Base score from performance and utilization
        base_score = worker.performance_score / (worker.overall_utilization + 0.1)
        
        # Affinity score from historical performance
        affinity_score = self.worker_affinity.get(worker.worker_id, {}).get(task.test_name, 0.5)
        
        # Combine scores
        total_score = base_score * 0.7 + affinity_score * 0.3
        
        return total_score
    
    def record_execution(self, task: TestTask, worker_id: str, duration: float, success: bool):
        """Record execution result for learning"""
        test_key = task.test_name
        
        # Update execution history
        self.execution_history[test_key].append(duration)
        if len(self.execution_history[test_key]) > 100:
            self.execution_history[test_key].pop(0)
        
        # Update worker affinity
        current_affinity = self.worker_affinity[worker_id].get(test_key, 0.5)
        
        # Calculate new affinity based on performance
        expected_duration = task.estimated_duration
        performance_ratio = expected_duration / duration if duration > 0 else 1.0
        success_factor = 1.0 if success else 0.5
        
        new_affinity = current_affinity + self.learning_rate * (performance_ratio * success_factor - current_affinity)
        self.worker_affinity[worker_id][test_key] = max(0.0, min(1.0, new_affinity))
    
    def get_algorithm_name(self) -> str:
        return "Adaptive"


class WorkStealingBalancer(LoadBalancingAlgorithm):
    """Work-stealing balancer for dynamic load redistribution"""
    
    def __init__(self):
        self.worker_queues: Dict[str, deque] = defaultdict(deque)
        self.steal_threshold = 0.7  # Utilization threshold for stealing
        self.queue_lock = threading.Lock()
    
    def assign_task(self, task: TestTask, available_workers: List[WorkerCapacity]) -> Optional[str]:
        """Assign task with work-stealing capability"""
        capable_workers = [w for w in available_workers if w.can_accept_task(task)]
        
        if not capable_workers:
            return None
        
        # Find least loaded worker
        least_loaded = min(capable_workers, key=lambda w: w.overall_utilization)
        
        with self.queue_lock:
            # Add task to worker's queue
            self.worker_queues[least_loaded.worker_id].append(task)
        
        return least_loaded.worker_id
    
    def steal_work(self, idle_worker_id: str, all_workers: List[WorkerCapacity]) -> Optional[TestTask]:
        """Steal work from overloaded workers"""
        with self.queue_lock:
            # Find overloaded workers
            overloaded_workers = [
                w for w in all_workers 
                if w.overall_utilization > self.steal_threshold and 
                len(self.worker_queues[w.worker_id]) > 1
            ]
            
            if not overloaded_workers:
                return None
            
            # Sort by utilization (descending)
            overloaded_workers.sort(key=lambda w: w.overall_utilization, reverse=True)
            
            # Try to steal from most overloaded worker
            for worker in overloaded_workers:
                worker_queue = self.worker_queues[worker.worker_id]
                if worker_queue:
                    # Steal from the end of the queue (least priority)
                    stolen_task = worker_queue.pop()
                    stolen_task.assigned_worker = idle_worker_id
                    return stolen_task
        
        return None
    
    def get_algorithm_name(self) -> str:
        return "Work Stealing"


class GeneticAlgorithmBalancer(LoadBalancingAlgorithm):
    """Genetic algorithm-based load balancer for optimal task distribution"""
    
    def __init__(self, population_size: int = 50, generations: int = 10):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
    
    def assign_task(self, task: TestTask, available_workers: List[WorkerCapacity]) -> Optional[str]:
        """Assign task using genetic algorithm optimization"""
        capable_workers = [w for w in available_workers if w.can_accept_task(task)]
        
        if not capable_workers:
            return None
        
        if len(capable_workers) == 1:
            return capable_workers[0].worker_id
        
        # For single task assignment, use simplified approach
        # Full genetic algorithm would be used for batch assignment
        return self._simple_genetic_selection(task, capable_workers)
    
    def _simple_genetic_selection(self, task: TestTask, workers: List[WorkerCapacity]) -> str:
        """Simplified genetic selection for single task"""
        # Calculate fitness for each worker
        fitness_scores = []
        for worker in workers:
            fitness = self._calculate_fitness(task, worker)
            fitness_scores.append((worker.worker_id, fitness))
        
        # Sort by fitness (higher is better)
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select based on fitness with some randomness
        total_fitness = sum(score[1] for score in fitness_scores)
        
        if total_fitness == 0:
            return random.choice(workers).worker_id
        
        # Roulette wheel selection
        r = random.random() * total_fitness
        cumulative = 0
        
        for worker_id, fitness in fitness_scores:
            cumulative += fitness
            if cumulative >= r:
                return worker_id
        
        return fitness_scores[0][0]  # Fallback to best
    
    def _calculate_fitness(self, task: TestTask, worker: WorkerCapacity) -> float:
        """Calculate fitness score for worker-task combination"""
        # Multiple factors contribute to fitness
        performance_factor = worker.performance_score / 100.0
        utilization_factor = 1.0 - worker.overall_utilization
        capacity_factor = worker.available_slots / worker.max_concurrent_tests
        
        # Specialty match factor
        required_tags = task.resource_requirements.get('specialty_tags', set())
        specialty_factor = 1.0
        if required_tags:
            matched_tags = required_tags.intersection(worker.specialty_tags)
            specialty_factor = len(matched_tags) / len(required_tags)
        
        # Combine factors
        fitness = (
            performance_factor * 0.3 +
            utilization_factor * 0.3 +
            capacity_factor * 0.2 +
            specialty_factor * 0.2
        )
        
        return max(0.0, fitness)
    
    def optimize_batch_assignment(self, tasks: List[TestTask], 
                                workers: List[WorkerCapacity]) -> Dict[str, str]:
        """Optimize assignment of multiple tasks using genetic algorithm"""
        if not tasks or not workers:
            return {}
        
        # Create initial population
        population = []
        for _ in range(self.population_size):
            assignment = self._create_random_assignment(tasks, workers)
            population.append(assignment)
        
        # Evolve population
        for generation in range(self.generations):
            # Calculate fitness for each assignment
            fitness_scores = []
            for assignment in population:
                fitness = self._calculate_assignment_fitness(assignment, tasks, workers)
                fitness_scores.append(fitness)
            
            # Create new population
            new_population = []
            
            # Elitism: keep best assignments
            elite_count = self.population_size // 10
            elite_indices = sorted(range(len(fitness_scores)), 
                                 key=lambda i: fitness_scores[i], reverse=True)[:elite_count]
            
            for i in elite_indices:
                new_population.append(population[i].copy())
            
            # Generate rest through crossover and mutation
            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate:
                    parent1, parent2 = self._select_parents(population, fitness_scores)
                    child = self._crossover(parent1, parent2)
                else:
                    child = random.choice(population).copy()
                
                if random.random() < self.mutation_rate:
                    child = self._mutate(child, tasks, workers)
                
                new_population.append(child)
            
            population = new_population
        
        # Return best assignment
        final_fitness = [self._calculate_assignment_fitness(assignment, tasks, workers) 
                        for assignment in population]
        best_index = final_fitness.index(max(final_fitness))
        
        return population[best_index]
    
    def _create_random_assignment(self, tasks: List[TestTask], 
                                workers: List[WorkerCapacity]) -> Dict[str, str]:
        """Create random task-worker assignment"""
        assignment = {}
        
        for task in tasks:
            capable_workers = [w for w in workers if w.can_accept_task(task)]
            if capable_workers:
                assignment[task.test_id] = random.choice(capable_workers).worker_id
        
        return assignment
    
    def _calculate_assignment_fitness(self, assignment: Dict[str, str], 
                                    tasks: List[TestTask], 
                                    workers: List[WorkerCapacity]) -> float:
        """Calculate fitness for entire assignment"""
        total_fitness = 0
        worker_loads = {w.worker_id: 0 for w in workers}
        
        for task in tasks:
            worker_id = assignment.get(task.test_id)
            if worker_id:
                worker = next((w for w in workers if w.worker_id == worker_id), None)
                if worker:
                    # Individual task fitness
                    task_fitness = self._calculate_fitness(task, worker)
                    total_fitness += task_fitness
                    
                    # Update worker load for load balancing
                    worker_loads[worker_id] += 1
        
        # Penalty for load imbalance
        if worker_loads:
            load_values = list(worker_loads.values())
            load_std = statistics.stdev(load_values) if len(load_values) > 1 else 0
            load_penalty = load_std * 0.1
            total_fitness -= load_penalty
        
        return total_fitness
    
    def _select_parents(self, population: List[Dict[str, str]], 
                       fitness_scores: List[float]) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Select parents for crossover"""
        # Tournament selection
        tournament_size = 3
        
        def tournament_select() -> Dict[str, str]:
            tournament_indices = random.sample(range(len(population)), tournament_size)
            best_index = max(tournament_indices, key=lambda i: fitness_scores[i])
            return population[best_index]
        
        return tournament_select(), tournament_select()
    
    def _crossover(self, parent1: Dict[str, str], parent2: Dict[str, str]) -> Dict[str, str]:
        """Perform crossover between two parents"""
        child = {}
        
        for task_id in parent1:
            if random.random() < 0.5:
                child[task_id] = parent1[task_id]
            else:
                child[task_id] = parent2.get(task_id, parent1[task_id])
        
        return child
    
    def _mutate(self, assignment: Dict[str, str], tasks: List[TestTask], 
               workers: List[WorkerCapacity]) -> Dict[str, str]:
        """Mutate assignment"""
        mutated = assignment.copy()
        
        # Randomly reassign some tasks
        for task_id in list(mutated.keys()):
            if random.random() < 0.1:  # 10% mutation rate per task
                task = next((t for t in tasks if t.test_id == task_id), None)
                if task:
                    capable_workers = [w for w in workers if w.can_accept_task(task)]
                    if capable_workers:
                        mutated[task_id] = random.choice(capable_workers).worker_id
        
        return mutated
    
    def get_algorithm_name(self) -> str:
        return "Genetic Algorithm"


class AdvancedLoadBalancer:
    """Advanced load balancer with multiple algorithms and adaptive selection"""
    
    def __init__(self):
        self.algorithms = {
            DistributionStrategy.ROUND_ROBIN: RoundRobinBalancer(),
            DistributionStrategy.LOAD_BASED: LoadBasedBalancer(),
            DistributionStrategy.PERFORMANCE_BASED: PerformanceBasedBalancer(),
            DistributionStrategy.ADAPTIVE: AdaptiveBalancer(),
            DistributionStrategy.WORK_STEALING: WorkStealingBalancer(),
            DistributionStrategy.GENETIC_ALGORITHM: GeneticAlgorithmBalancer()
        }
        
        self.current_strategy = DistributionStrategy.ADAPTIVE
        self.workers: Dict[str, WorkerCapacity] = {}
        self.task_queue: deque = deque()
        self.completed_tasks: List[TestTask] = []
        self.assignment_history: List[Dict[str, Any]] = []
        
        self.balancer_lock = threading.Lock()
        self.performance_metrics: Dict[DistributionStrategy, List[float]] = defaultdict(list)
        
        # Auto-selection parameters
        self.evaluation_window = 100  # Number of tasks to evaluate
        self.strategy_evaluation_interval = 50  # Evaluate strategies every N tasks
        
    def register_worker(self, worker_id: str, capacity: WorkerCapacity):
        """Register a worker with the load balancer"""
        with self.balancer_lock:
            self.workers[worker_id] = capacity
            logger.info(f"Registered worker {worker_id} with capacity {capacity.max_concurrent_tests}")
    
    def unregister_worker(self, worker_id: str):
        """Unregister a worker"""
        with self.balancer_lock:
            if worker_id in self.workers:
                del self.workers[worker_id]
                logger.info(f"Unregistered worker {worker_id}")
    
    def update_worker_capacity(self, worker_id: str, capacity_updates: Dict[str, Any]):
        """Update worker capacity information"""
        with self.balancer_lock:
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                for key, value in capacity_updates.items():
                    if hasattr(worker, key):
                        setattr(worker, key, value)
                worker.last_updated = datetime.now()
    
    def assign_task(self, task: TestTask, strategy: Optional[DistributionStrategy] = None) -> Optional[str]:
        """Assign a task to a worker using specified or current strategy"""
        if strategy is None:
            strategy = self.current_strategy
        
        with self.balancer_lock:
            available_workers = [w for w in self.workers.values() if w.available_slots > 0]
            
            if not available_workers:
                return None
            
            algorithm = self.algorithms[strategy]
            assigned_worker_id = algorithm.assign_task(task, available_workers)
            
            if assigned_worker_id:
                # Update worker capacity
                self.workers[assigned_worker_id].current_load += 1
                task.assigned_worker = assigned_worker_id
                task.started_at = datetime.now()
                
                # Record assignment
                self.assignment_history.append({
                    'task_id': task.test_id,
                    'worker_id': assigned_worker_id,
                    'strategy': strategy.value,
                    'timestamp': datetime.now(),
                    'worker_utilization': self.workers[assigned_worker_id].overall_utilization
                })
                
                logger.debug(f"Assigned task {task.test_id} to worker {assigned_worker_id} using {strategy.value}")
            
            return assigned_worker_id
    
    def complete_task(self, task: TestTask, success: bool):
        """Mark task as completed and update metrics"""
        with self.balancer_lock:
            if task.assigned_worker:
                # Update worker capacity
                if task.assigned_worker in self.workers:
                    self.workers[task.assigned_worker].current_load -= 1
                
                # Update task
                task.completed_at = datetime.now()
                task.success = success
                
                if task.started_at:
                    task.actual_duration = (task.completed_at - task.started_at).total_seconds()
                
                # Record completion
                self.completed_tasks.append(task)
                
                # Update algorithm-specific metrics
                if isinstance(self.algorithms[self.current_strategy], AdaptiveBalancer):
                    self.algorithms[self.current_strategy].record_execution(
                        task, task.assigned_worker, task.actual_duration or 0, success
                    )
                
                # Evaluate strategy performance
                if len(self.completed_tasks) % self.strategy_evaluation_interval == 0:
                    self._evaluate_strategies()
    
    def _evaluate_strategies(self):
        """Evaluate and potentially switch strategies"""
        if len(self.completed_tasks) < self.evaluation_window:
            return
        
        recent_tasks = self.completed_tasks[-self.evaluation_window:]
        
        # Calculate performance metrics for current strategy
        current_performance = self._calculate_strategy_performance(recent_tasks)
        self.performance_metrics[self.current_strategy].append(current_performance)
        
        # Evaluate if strategy switch is needed
        if len(self.performance_metrics[self.current_strategy]) > 5:
            recent_performance = self.performance_metrics[self.current_strategy][-5:]
            avg_performance = sum(recent_performance) / len(recent_performance)
            
            # Consider switching if performance is declining
            if avg_performance < 0.7:  # 70% threshold
                self._select_best_strategy()
    
    def _calculate_strategy_performance(self, tasks: List[TestTask]) -> float:
        """Calculate performance score for a strategy"""
        if not tasks:
            return 0.0
        
        # Factors: success rate, duration accuracy, load balance
        success_rate = sum(1 for t in tasks if t.success) / len(tasks)
        
        # Duration accuracy (how close actual vs estimated)
        duration_accuracy = 0.0
        duration_count = 0
        for task in tasks:
            if task.actual_duration and task.estimated_duration > 0:
                accuracy = 1.0 - abs(task.actual_duration - task.estimated_duration) / task.estimated_duration
                duration_accuracy += max(0, accuracy)
                duration_count += 1
        
        if duration_count > 0:
            duration_accuracy /= duration_count
        
        # Load balance (standard deviation of worker utilizations)
        if self.workers:
            utilizations = [w.overall_utilization for w in self.workers.values()]
            if len(utilizations) > 1:
                load_balance = 1.0 - (statistics.stdev(utilizations) / statistics.mean(utilizations))
                load_balance = max(0, load_balance)
            else:
                load_balance = 1.0
        else:
            load_balance = 0.0
        
        # Combined score
        performance = success_rate * 0.4 + duration_accuracy * 0.3 + load_balance * 0.3
        
        return performance
    
    def _select_best_strategy(self):
        """Select the best performing strategy"""
        if not self.performance_metrics:
            return
        
        strategy_scores = {}
        for strategy, metrics in self.performance_metrics.items():
            if metrics:
                strategy_scores[strategy] = sum(metrics[-3:]) / len(metrics[-3:])  # Last 3 measurements
        
        if strategy_scores:
            best_strategy = max(strategy_scores, key=strategy_scores.get)
            if best_strategy != self.current_strategy:
                logger.info(f"Switching load balancing strategy from {self.current_strategy.value} to {best_strategy.value}")
                self.current_strategy = best_strategy
    
    def get_load_balance_report(self) -> Dict[str, Any]:
        """Get comprehensive load balancing report"""
        with self.balancer_lock:
            worker_stats = {}
            for worker_id, worker in self.workers.items():
                worker_stats[worker_id] = {
                    'utilization': worker.overall_utilization,
                    'current_load': worker.current_load,
                    'max_capacity': worker.max_concurrent_tests,
                    'performance_score': worker.performance_score,
                    'cpu_utilization': worker.cpu_utilization,
                    'memory_utilization': worker.memory_utilization
                }
            
            # Strategy performance
            strategy_performance = {}
            for strategy, metrics in self.performance_metrics.items():
                if metrics:
                    strategy_performance[strategy.value] = {
                        'avg_performance': sum(metrics) / len(metrics),
                        'recent_performance': sum(metrics[-5:]) / len(metrics[-5:]) if len(metrics) >= 5 else 0,
                        'measurement_count': len(metrics)
                    }
            
            # Recent assignments
            recent_assignments = self.assignment_history[-50:] if len(self.assignment_history) > 50 else self.assignment_history
            
            return {
                'timestamp': datetime.now().isoformat(),
                'current_strategy': self.current_strategy.value,
                'total_workers': len(self.workers),
                'active_workers': len([w for w in self.workers.values() if w.current_load > 0]),
                'worker_statistics': worker_stats,
                'strategy_performance': strategy_performance,
                'recent_assignments': [
                    {
                        'task_id': a['task_id'],
                        'worker_id': a['worker_id'],
                        'strategy': a['strategy'],
                        'timestamp': a['timestamp'].isoformat(),
                        'worker_utilization': a['worker_utilization']
                    }
                    for a in recent_assignments
                ],
                'completed_tasks': len(self.completed_tasks),
                'load_balance_score': self._calculate_current_load_balance()
            }
    
    def _calculate_current_load_balance(self) -> float:
        """Calculate current load balance score"""
        if not self.workers:
            return 0.0
        
        utilizations = [w.overall_utilization for w in self.workers.values()]
        
        if len(utilizations) <= 1:
            return 1.0
        
        mean_util = statistics.mean(utilizations)
        if mean_util == 0:
            return 1.0
        
        std_util = statistics.stdev(utilizations)
        balance_score = 1.0 - (std_util / mean_util)
        
        return max(0.0, balance_score)
    
    def optimize_current_distribution(self):
        """Optimize current task distribution using genetic algorithm"""
        if self.current_strategy != DistributionStrategy.GENETIC_ALGORITHM:
            return
        
        # Get current active tasks
        active_tasks = [t for t in self.task_queue if t.assigned_worker is None]
        
        if not active_tasks or not self.workers:
            return
        
        # Use genetic algorithm for optimization
        ga_balancer = self.algorithms[DistributionStrategy.GENETIC_ALGORITHM]
        optimized_assignment = ga_balancer.optimize_batch_assignment(
            active_tasks, list(self.workers.values())
        )
        
        # Apply optimized assignment
        for task_id, worker_id in optimized_assignment.items():
            task = next((t for t in active_tasks if t.test_id == task_id), None)
            if task:
                self.assign_task(task, DistributionStrategy.GENETIC_ALGORITHM)
    
    def get_worker_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for worker optimization"""
        recommendations = []
        
        with self.balancer_lock:
            if not self.workers:
                return recommendations
            
            # Analyze worker utilization
            utilizations = [w.overall_utilization for w in self.workers.values()]
            avg_utilization = statistics.mean(utilizations)
            
            # Find overloaded workers
            overloaded_workers = [
                w for w in self.workers.values() 
                if w.overall_utilization > 0.9
            ]
            
            for worker in overloaded_workers:
                recommendations.append({
                    'type': 'overloaded_worker',
                    'worker_id': worker.worker_id,
                    'utilization': worker.overall_utilization,
                    'recommendation': 'Consider reducing load or increasing capacity',
                    'priority': 'high'
                })
            
            # Find underutilized workers
            underutilized_workers = [
                w for w in self.workers.values()
                if w.overall_utilization < 0.3 and avg_utilization > 0.5
            ]
            
            for worker in underutilized_workers:
                recommendations.append({
                    'type': 'underutilized_worker',
                    'worker_id': worker.worker_id,
                    'utilization': worker.overall_utilization,
                    'recommendation': 'Consider increasing load or reducing capacity',
                    'priority': 'medium'
                })
            
            # Load balance recommendations
            if len(utilizations) > 1:
                std_util = statistics.stdev(utilizations)
                if std_util > 0.3:
                    recommendations.append({
                        'type': 'load_imbalance',
                        'std_deviation': std_util,
                        'recommendation': 'Consider rebalancing tasks across workers',
                        'priority': 'medium'
                    })
        
        return recommendations


if __name__ == "__main__":
    # Demo usage
    balancer = AdvancedLoadBalancer()
    
    # Register workers
    for i in range(4):
        capacity = WorkerCapacity(
            worker_id=f"worker_{i}",
            max_concurrent_tests=3,
            current_load=0,
            cpu_capacity=100.0,
            memory_capacity=2048.0,
            current_cpu_usage=0.0,
            current_memory_usage=0.0,
            performance_score=85.0 + random.uniform(-10, 10),
            specialty_tags=set(['unit', 'integration']) if i % 2 == 0 else set(['performance', 'security'])
        )
        balancer.register_worker(f"worker_{i}", capacity)
    
    # Create and assign tasks
    for i in range(20):
        task = TestTask(
            test_id=f"test_{i}",
            test_name=f"test_example_{i}",
            estimated_duration=random.uniform(0.5, 5.0),
            priority=random.randint(1, 5),
            dependencies=[],
            resource_requirements={
                'cpu': random.uniform(10, 50),
                'memory': random.uniform(100, 500),
                'specialty_tags': set(['unit']) if i % 3 == 0 else set()
            }
        )
        
        assigned_worker = balancer.assign_task(task)
        if assigned_worker:
            print(f"Assigned {task.test_id} to {assigned_worker}")
            
            # Simulate task completion
            import time
            time.sleep(0.1)
            balancer.complete_task(task, success=random.random() > 0.1)
    
    # Get load balance report
    report = balancer.get_load_balance_report()
    print(f"\nLoad Balance Report:")
    print(f"Current Strategy: {report['current_strategy']}")
    print(f"Active Workers: {report['active_workers']}/{report['total_workers']}")
    print(f"Load Balance Score: {report['load_balance_score']:.2f}")
    
    # Get recommendations
    recommendations = balancer.get_worker_recommendations()
    print(f"\nRecommendations:")
    for rec in recommendations:
        print(f"- {rec['type']}: {rec['recommendation']} (Priority: {rec['priority']})")