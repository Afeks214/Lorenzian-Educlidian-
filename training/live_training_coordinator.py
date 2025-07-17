"""
LiveTrainingCoordinator: Orchestrates real-time learning across all agents during live trading.

This system provides:
1. Concurrent training during live execution
2. Multi-agent learning coordination
3. Performance degradation detection and response
4. Automated model retraining triggers
5. Cross-agent learning synchronization
6. Real-time model performance monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import threading
import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Set, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
import queue
import psutil
import weakref
import pickle
import copy
from contextlib import contextmanager
import gc
from abc import ABC, abstractmethod

# Import existing training systems
from .training_progress_monitor import TrainingProgressMonitor, MonitoringConfig, MetricType
from .distributed_training_coordinator import DistributedTrainingCoordinator, DistributedConfig
from .unified_training_system import UnifiedTrainingSystem, UnifiedTrainingConfig
from .advanced_checkpoint_manager import AdvancedCheckpointManager, CheckpointConfig
from .memory_optimized_trainer import MemoryOptimizedTrainer, MemoryOptimizationConfig
from .performance_analysis_framework import PerformanceAnalysisFramework

logger = logging.getLogger(__name__)


class TrainingPhase(Enum):
    """Live training phases"""
    INITIALIZATION = "initialization"
    LIVE_EXECUTION = "live_execution"
    PERFORMANCE_MONITORING = "performance_monitoring"
    DEGRADATION_DETECTION = "degradation_detection"
    RETRAINING = "retraining"
    SYNCHRONIZATION = "synchronization"
    VALIDATION = "validation"
    RECOVERY = "recovery"


class AgentPerformanceState(Enum):
    """Agent performance states"""
    OPTIMAL = "optimal"
    GOOD = "good"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"
    RETRAINING = "retraining"


class LearningMode(Enum):
    """Learning modes for live training"""
    CONTINUOUS = "continuous"
    EPISODIC = "episodic"
    TRIGGERED = "triggered"
    SUPERVISED = "supervised"
    REINFORCEMENT = "reinforcement"


@dataclass
class AgentMetrics:
    """Metrics for individual agent performance"""
    agent_id: str
    agent_name: str
    performance_score: float
    loss_trend: List[float]
    accuracy_trend: List[float]
    prediction_confidence: float
    learning_rate: float
    gradient_norm: float
    memory_usage: float
    throughput: float
    latency: float
    error_rate: float
    last_update: datetime
    training_iterations: int
    convergence_score: float
    stability_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class LiveTrainingConfig:
    """Configuration for live training coordinator"""
    # Training strategy
    learning_mode: LearningMode = LearningMode.CONTINUOUS
    concurrent_training: bool = True
    max_concurrent_agents: int = 4
    
    # Performance monitoring
    monitoring_frequency: float = 1.0  # seconds
    performance_window: int = 100
    degradation_threshold: float = 0.1
    critical_threshold: float = 0.2
    
    # Retraining triggers
    enable_auto_retraining: bool = True
    retraining_cooldown: float = 300.0  # 5 minutes
    performance_trigger_threshold: float = 0.15
    accuracy_trigger_threshold: float = 0.05
    max_retraining_attempts: int = 3
    
    # Resource management
    max_memory_usage: float = 0.8  # 80% of available memory
    max_cpu_usage: float = 0.7  # 70% of available CPU
    training_batch_size: int = 32
    validation_batch_size: int = 64
    
    # Synchronization
    sync_frequency: float = 60.0  # seconds
    enable_cross_agent_learning: bool = True
    knowledge_sharing_threshold: float = 0.05
    
    # Persistence
    checkpoint_frequency: float = 300.0  # 5 minutes
    backup_frequency: float = 900.0  # 15 minutes
    model_history_size: int = 10
    
    # Directories
    checkpoints_dir: str = "live_training_checkpoints"
    models_dir: str = "live_training_models"
    logs_dir: str = "live_training_logs"
    
    # Advanced settings
    enable_distributed_training: bool = False
    enable_mixed_precision: bool = True
    enable_gradient_clipping: bool = True
    gradient_clip_norm: float = 1.0
    
    # Validation
    validation_frequency: float = 120.0  # 2 minutes
    validation_samples: int = 100
    early_stopping_patience: int = 10
    
    # Safety mechanisms
    emergency_stop_threshold: float = 0.5
    rollback_threshold: float = 0.3
    enable_safety_checks: bool = True


class AgentTrainingManager:
    """Manages training for individual agents"""
    
    def __init__(self, 
                 agent_id: str,
                 agent_name: str,
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 loss_fn: Callable,
                 device: torch.device,
                 config: LiveTrainingConfig):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.config = config
        
        # Training state
        self.training_active = False
        self.training_thread = None
        self.training_queue = queue.Queue()
        self.metrics_queue = queue.Queue()
        
        # Performance tracking
        self.metrics = AgentMetrics(
            agent_id=agent_id,
            agent_name=agent_name,
            performance_score=1.0,
            loss_trend=[],
            accuracy_trend=[],
            prediction_confidence=0.0,
            learning_rate=0.0,
            gradient_norm=0.0,
            memory_usage=0.0,
            throughput=0.0,
            latency=0.0,
            error_rate=0.0,
            last_update=datetime.now(),
            training_iterations=0,
            convergence_score=1.0,
            stability_score=1.0
        )
        
        # Model management
        self.model_history = deque(maxlen=config.model_history_size)
        self.best_model_state = None
        self.best_performance = float('-inf')
        
        # Training data buffer
        self.training_buffer = deque(maxlen=1000)
        self.validation_buffer = deque(maxlen=500)
        
        # Synchronization
        self.sync_lock = threading.Lock()
        self.last_sync = datetime.now()
        
        # Memory optimization
        self.memory_optimizer = None
        if config.max_memory_usage < 1.0:
            self._setup_memory_optimizer()
        
        logger.info(f"AgentTrainingManager initialized for {agent_name}")
    
    def _setup_memory_optimizer(self):
        """Setup memory optimization for training"""
        try:
            memory_config = MemoryOptimizationConfig(
                max_memory_usage_gb=psutil.virtual_memory().total * self.config.max_memory_usage / (1024**3),
                gradient_accumulation_steps=2,
                mixed_precision=self.config.enable_mixed_precision,
                memory_efficient_attention=True
            )
            
            self.memory_optimizer = MemoryOptimizedTrainer(
                memory_config,
                lambda: self.model,
                lambda params: self.optimizer,
                self.device
            )
            
        except Exception as e:
            logger.warning(f"Memory optimizer setup failed for {self.agent_name}: {e}")
    
    def start_training(self):
        """Start background training thread"""
        if self.training_active:
            return
        
        self.training_active = True
        self.training_thread = threading.Thread(target=self._training_loop)
        self.training_thread.daemon = True
        self.training_thread.start()
        
        logger.info(f"Training started for agent {self.agent_name}")
    
    def stop_training(self):
        """Stop background training"""
        self.training_active = False
        if self.training_thread:
            self.training_thread.join(timeout=5.0)
        
        logger.info(f"Training stopped for agent {self.agent_name}")
    
    def _training_loop(self):
        """Main training loop"""
        while self.training_active:
            try:
                # Check for new training data
                if not self.training_queue.empty():
                    batch_data = []
                    
                    # Collect batch
                    while not self.training_queue.empty() and len(batch_data) < self.config.training_batch_size:
                        try:
                            data = self.training_queue.get_nowait()
                            batch_data.append(data)
                        except queue.Empty:
                            break
                    
                    if batch_data:
                        self._train_batch(batch_data)
                
                # Update metrics
                self._update_metrics()
                
                # Sleep briefly to prevent busy waiting
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Training loop error for {self.agent_name}: {e}")
                time.sleep(0.1)
    
    def _train_batch(self, batch_data: List[Any]):
        """Train on a batch of data"""
        try:
            start_time = time.time()
            
            # Convert batch to tensors
            inputs, targets = self._prepare_batch(batch_data)
            
            # Forward pass
            self.model.train()
            
            if self.config.enable_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.config.enable_mixed_precision:
                scaler = torch.cuda.amp.GradScaler()
                scaler.scale(loss).backward()
                
                if self.config.enable_gradient_clipping:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                
                if self.config.enable_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                
                self.optimizer.step()
            
            # Update metrics
            training_time = time.time() - start_time
            self._update_training_metrics(loss.item(), training_time, len(batch_data))
            
            # Save best model
            if loss.item() < self.best_performance:
                self.best_performance = loss.item()
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            
            # Add to model history
            self.model_history.append({
                'state_dict': copy.deepcopy(self.model.state_dict()),
                'loss': loss.item(),
                'timestamp': datetime.now(),
                'iteration': self.metrics.training_iterations
            })
            
            self.metrics.training_iterations += 1
            
        except Exception as e:
            logger.error(f"Batch training failed for {self.agent_name}: {e}")
            self.metrics.error_rate += 1
    
    def _prepare_batch(self, batch_data: List[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare batch data for training"""
        # This is a generic implementation - should be customized for specific data types
        try:
            inputs = []
            targets = []
            
            for data in batch_data:
                if isinstance(data, dict):
                    inputs.append(data.get('input', data.get('features', [])))
                    targets.append(data.get('target', data.get('label', 0)))
                else:
                    # Assume tuple/list format
                    inputs.append(data[0] if len(data) > 0 else [])
                    targets.append(data[1] if len(data) > 1 else 0)
            
            # Convert to tensors
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device=self.device)
            targets_tensor = torch.tensor(targets, dtype=torch.float32, device=self.device)
            
            return inputs_tensor, targets_tensor
            
        except Exception as e:
            logger.error(f"Batch preparation failed for {self.agent_name}: {e}")
            # Return dummy tensors
            return torch.zeros(1, 1, device=self.device), torch.zeros(1, device=self.device)
    
    def _update_training_metrics(self, loss: float, training_time: float, batch_size: int):
        """Update training metrics"""
        # Update loss trend
        self.metrics.loss_trend.append(loss)
        if len(self.metrics.loss_trend) > self.config.performance_window:
            self.metrics.loss_trend.pop(0)
        
        # Update performance score
        if len(self.metrics.loss_trend) > 1:
            recent_loss = np.mean(self.metrics.loss_trend[-10:])
            older_loss = np.mean(self.metrics.loss_trend[:-10]) if len(self.metrics.loss_trend) > 10 else recent_loss
            
            if older_loss > 0:
                improvement = (older_loss - recent_loss) / older_loss
                self.metrics.performance_score = max(0.0, min(1.0, improvement + 0.5))
        
        # Update throughput
        self.metrics.throughput = batch_size / training_time if training_time > 0 else 0
        
        # Update latency
        self.metrics.latency = training_time
        
        # Update gradient norm
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        self.metrics.gradient_norm = total_norm ** (1. / 2)
        
        # Update learning rate
        self.metrics.learning_rate = self.optimizer.param_groups[0]['lr']
        
        # Update convergence score
        if len(self.metrics.loss_trend) > 10:
            recent_std = np.std(self.metrics.loss_trend[-10:])
            self.metrics.convergence_score = max(0.0, 1.0 - recent_std)
        
        # Update stability score
        if len(self.metrics.loss_trend) > 20:
            half_point = len(self.metrics.loss_trend) // 2
            first_half_std = np.std(self.metrics.loss_trend[:half_point])
            second_half_std = np.std(self.metrics.loss_trend[half_point:])
            
            if first_half_std > 0:
                stability_ratio = second_half_std / first_half_std
                self.metrics.stability_score = max(0.0, 1.0 - abs(stability_ratio - 1.0))
        
        self.metrics.last_update = datetime.now()
    
    def _update_metrics(self):
        """Update system metrics"""
        try:
            # Update memory usage
            process = psutil.Process()
            self.metrics.memory_usage = process.memory_info().rss / (1024**3)  # GB
            
            # Update error rate (decay over time)
            self.metrics.error_rate *= 0.99
            
        except Exception as e:
            logger.debug(f"Metrics update failed for {self.agent_name}: {e}")
    
    def add_training_data(self, data: Any):
        """Add data to training queue"""
        try:
            self.training_queue.put(data, block=False)
            self.training_buffer.append(data)
        except queue.Full:
            logger.warning(f"Training queue full for {self.agent_name}")
    
    def add_validation_data(self, data: Any):
        """Add data to validation buffer"""
        self.validation_buffer.append(data)
    
    def get_metrics(self) -> AgentMetrics:
        """Get current metrics"""
        with self.sync_lock:
            return copy.deepcopy(self.metrics)
    
    def get_performance_state(self) -> AgentPerformanceState:
        """Determine current performance state"""
        if self.metrics.performance_score > 0.8:
            return AgentPerformanceState.OPTIMAL
        elif self.metrics.performance_score > 0.6:
            return AgentPerformanceState.GOOD
        elif self.metrics.performance_score > 0.4:
            return AgentPerformanceState.DEGRADED
        elif self.metrics.performance_score > 0.2:
            return AgentPerformanceState.CRITICAL
        else:
            return AgentPerformanceState.FAILED
    
    def should_retrain(self) -> bool:
        """Check if agent should be retrained"""
        state = self.get_performance_state()
        return state in [AgentPerformanceState.CRITICAL, AgentPerformanceState.FAILED]
    
    def rollback_model(self, steps: int = 1):
        """Rollback model to previous state"""
        if len(self.model_history) >= steps:
            target_state = self.model_history[-steps]
            self.model.load_state_dict(target_state['state_dict'])
            logger.info(f"Model rolled back {steps} steps for {self.agent_name}")
        else:
            logger.warning(f"Cannot rollback {steps} steps for {self.agent_name} - insufficient history")
    
    def restore_best_model(self):
        """Restore best performing model"""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Best model restored for {self.agent_name}")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_training()
        
        # Clear buffers
        self.training_buffer.clear()
        self.validation_buffer.clear()
        self.model_history.clear()
        
        # Clear queues
        while not self.training_queue.empty():
            try:
                self.training_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info(f"AgentTrainingManager cleaned up for {self.agent_name}")


class PerformanceDegradationDetector:
    """Detects performance degradation in agents"""
    
    def __init__(self, config: LiveTrainingConfig):
        self.config = config
        self.agent_baselines = {}
        self.degradation_history = defaultdict(list)
        self.alerts = []
        
    def update_baseline(self, agent_id: str, metrics: AgentMetrics):
        """Update baseline performance for agent"""
        if agent_id not in self.agent_baselines:
            self.agent_baselines[agent_id] = {
                'performance_score': [],
                'accuracy_trend': [],
                'loss_trend': [],
                'last_update': datetime.now()
            }
        
        baseline = self.agent_baselines[agent_id]
        baseline['performance_score'].append(metrics.performance_score)
        baseline['accuracy_trend'].extend(metrics.accuracy_trend[-10:])
        baseline['loss_trend'].extend(metrics.loss_trend[-10:])
        baseline['last_update'] = datetime.now()
        
        # Keep only recent baseline data
        max_baseline_size = 1000
        for key in ['performance_score', 'accuracy_trend', 'loss_trend']:
            if len(baseline[key]) > max_baseline_size:
                baseline[key] = baseline[key][-max_baseline_size:]
    
    def detect_degradation(self, agent_id: str, metrics: AgentMetrics) -> Dict[str, Any]:
        """Detect performance degradation"""
        if agent_id not in self.agent_baselines:
            return {'degraded': False, 'reason': 'No baseline available'}
        
        baseline = self.agent_baselines[agent_id]
        degradation_info = {
            'degraded': False,
            'severity': 'none',
            'reasons': [],
            'metrics': {},
            'recommendations': []
        }
        
        # Check performance score degradation
        if baseline['performance_score']:
            avg_baseline_perf = np.mean(baseline['performance_score'][-100:])
            current_perf = metrics.performance_score
            
            if avg_baseline_perf - current_perf > self.config.degradation_threshold:
                degradation_info['degraded'] = True
                degradation_info['reasons'].append('Performance score degradation')
                degradation_info['metrics']['performance_degradation'] = avg_baseline_perf - current_perf
                
                if avg_baseline_perf - current_perf > self.config.critical_threshold:
                    degradation_info['severity'] = 'critical'
                else:
                    degradation_info['severity'] = 'moderate'
        
        # Check accuracy degradation
        if baseline['accuracy_trend'] and metrics.accuracy_trend:
            avg_baseline_acc = np.mean(baseline['accuracy_trend'][-100:])
            current_acc = np.mean(metrics.accuracy_trend[-10:]) if metrics.accuracy_trend else 0
            
            if avg_baseline_acc - current_acc > self.config.accuracy_trigger_threshold:
                degradation_info['degraded'] = True
                degradation_info['reasons'].append('Accuracy degradation')
                degradation_info['metrics']['accuracy_degradation'] = avg_baseline_acc - current_acc
        
        # Check loss trend
        if baseline['loss_trend'] and metrics.loss_trend:
            avg_baseline_loss = np.mean(baseline['loss_trend'][-100:])
            current_loss = np.mean(metrics.loss_trend[-10:]) if metrics.loss_trend else float('inf')
            
            if current_loss - avg_baseline_loss > self.config.performance_trigger_threshold:
                degradation_info['degraded'] = True
                degradation_info['reasons'].append('Loss increase')
                degradation_info['metrics']['loss_increase'] = current_loss - avg_baseline_loss
        
        # Check convergence issues
        if metrics.convergence_score < 0.5:
            degradation_info['degraded'] = True
            degradation_info['reasons'].append('Convergence issues')
            degradation_info['metrics']['convergence_score'] = metrics.convergence_score
        
        # Check stability issues
        if metrics.stability_score < 0.5:
            degradation_info['degraded'] = True
            degradation_info['reasons'].append('Stability issues')
            degradation_info['metrics']['stability_score'] = metrics.stability_score
        
        # Generate recommendations
        if degradation_info['degraded']:
            degradation_info['recommendations'] = self._generate_recommendations(degradation_info)
        
        # Record degradation history
        self.degradation_history[agent_id].append({
            'timestamp': datetime.now(),
            'degradation_info': degradation_info,
            'metrics': metrics.to_dict()
        })
        
        return degradation_info
    
    def _generate_recommendations(self, degradation_info: Dict[str, Any]) -> List[str]:
        """Generate recommendations for addressing degradation"""
        recommendations = []
        
        if 'Performance score degradation' in degradation_info['reasons']:
            recommendations.append('Consider retraining with fresh data')
            recommendations.append('Adjust learning rate')
        
        if 'Accuracy degradation' in degradation_info['reasons']:
            recommendations.append('Increase training data quality')
            recommendations.append('Review model architecture')
        
        if 'Loss increase' in degradation_info['reasons']:
            recommendations.append('Check for data distribution shift')
            recommendations.append('Consider model regularization')
        
        if 'Convergence issues' in degradation_info['reasons']:
            recommendations.append('Adjust optimization parameters')
            recommendations.append('Consider different optimizer')
        
        if 'Stability issues' in degradation_info['reasons']:
            recommendations.append('Implement gradient clipping')
            recommendations.append('Reduce learning rate')
        
        return recommendations
    
    def get_degradation_history(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get degradation history for agent"""
        return self.degradation_history.get(agent_id, [])
    
    def clear_history(self, agent_id: str):
        """Clear degradation history for agent"""
        if agent_id in self.degradation_history:
            self.degradation_history[agent_id].clear()


class CrossAgentLearningManager:
    """Manages cross-agent learning and knowledge sharing"""
    
    def __init__(self, config: LiveTrainingConfig):
        self.config = config
        self.knowledge_base = {}
        self.similarity_matrix = {}
        self.sharing_history = defaultdict(list)
        
    def analyze_agent_similarity(self, 
                                agent_metrics: Dict[str, AgentMetrics]) -> Dict[str, Dict[str, float]]:
        """Analyze similarity between agents"""
        similarities = {}
        
        agent_ids = list(agent_metrics.keys())
        for i, agent_id1 in enumerate(agent_ids):
            similarities[agent_id1] = {}
            
            for j, agent_id2 in enumerate(agent_ids):
                if i == j:
                    similarities[agent_id1][agent_id2] = 1.0
                else:
                    sim = self._calculate_similarity(agent_metrics[agent_id1], agent_metrics[agent_id2])
                    similarities[agent_id1][agent_id2] = sim
        
        self.similarity_matrix = similarities
        return similarities
    
    def _calculate_similarity(self, metrics1: AgentMetrics, metrics2: AgentMetrics) -> float:
        """Calculate similarity between two agents"""
        # Simple similarity based on performance patterns
        similarity = 0.0
        
        # Performance score similarity
        perf_diff = abs(metrics1.performance_score - metrics2.performance_score)
        similarity += max(0, 1 - perf_diff)
        
        # Loss trend similarity
        if metrics1.loss_trend and metrics2.loss_trend:
            min_len = min(len(metrics1.loss_trend), len(metrics2.loss_trend))
            if min_len > 0:
                trend1 = metrics1.loss_trend[-min_len:]
                trend2 = metrics2.loss_trend[-min_len:]
                
                correlation = np.corrcoef(trend1, trend2)[0, 1]
                if not np.isnan(correlation):
                    similarity += max(0, correlation)
        
        # Convergence similarity
        conv_diff = abs(metrics1.convergence_score - metrics2.convergence_score)
        similarity += max(0, 1 - conv_diff)
        
        # Stability similarity
        stab_diff = abs(metrics1.stability_score - metrics2.stability_score)
        similarity += max(0, 1 - stab_diff)
        
        return similarity / 4.0  # Normalize by number of factors
    
    def identify_knowledge_sharing_opportunities(self, 
                                               agent_metrics: Dict[str, AgentMetrics]) -> List[Dict[str, Any]]:
        """Identify opportunities for knowledge sharing"""
        opportunities = []
        
        # Update similarity matrix
        self.analyze_agent_similarity(agent_metrics)
        
        for source_id, metrics in agent_metrics.items():
            # Find agents that could benefit from this agent's knowledge
            if metrics.performance_score > 0.8:  # High-performing agent
                for target_id, target_metrics in agent_metrics.items():
                    if source_id != target_id and target_metrics.performance_score < 0.6:
                        # Check if they're similar enough to benefit
                        similarity = self.similarity_matrix[source_id][target_id]
                        
                        if similarity > self.config.knowledge_sharing_threshold:
                            opportunities.append({
                                'source_agent': source_id,
                                'target_agent': target_id,
                                'similarity': similarity,
                                'potential_benefit': 0.8 - target_metrics.performance_score,
                                'sharing_type': 'model_weights'
                            })
        
        return opportunities
    
    def execute_knowledge_sharing(self, 
                                opportunity: Dict[str, Any],
                                agent_managers: Dict[str, AgentTrainingManager]) -> bool:
        """Execute knowledge sharing between agents"""
        try:
            source_id = opportunity['source_agent']
            target_id = opportunity['target_agent']
            
            if source_id not in agent_managers or target_id not in agent_managers:
                return False
            
            source_manager = agent_managers[source_id]
            target_manager = agent_managers[target_id]
            
            # Share model weights (simple knowledge transfer)
            if opportunity['sharing_type'] == 'model_weights':
                source_state = source_manager.model.state_dict()
                target_state = target_manager.model.state_dict()
                
                # Weighted average of model parameters
                alpha = 0.3  # Transfer strength
                
                for key in source_state:
                    if key in target_state:
                        target_state[key] = (1 - alpha) * target_state[key] + alpha * source_state[key]
                
                target_manager.model.load_state_dict(target_state)
                
                # Record sharing
                self.sharing_history[target_id].append({
                    'timestamp': datetime.now(),
                    'source_agent': source_id,
                    'sharing_type': 'model_weights',
                    'transfer_strength': alpha
                })
                
                logger.info(f"Knowledge shared from {source_id} to {target_id}")
                return True
            
        except Exception as e:
            logger.error(f"Knowledge sharing failed: {e}")
            return False
        
        return False
    
    def get_sharing_history(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get knowledge sharing history for agent"""
        return self.sharing_history.get(agent_id, [])


class LiveTrainingCoordinator:
    """
    Main coordinator for live training across all agents
    """
    
    def __init__(self, config: LiveTrainingConfig):
        self.config = config
        self.agent_managers = {}
        self.training_active = False
        self.current_phase = TrainingPhase.INITIALIZATION
        
        # Initialize components
        self.performance_monitor = self._create_performance_monitor()
        self.degradation_detector = PerformanceDegradationDetector(config)
        self.cross_agent_manager = CrossAgentLearningManager(config)
        self.checkpoint_manager = self._create_checkpoint_manager()
        
        # Threading
        self.coordination_thread = None
        self.monitoring_thread = None
        self.sync_thread = None
        
        # Synchronization
        self.coordination_lock = threading.Lock()
        self.metrics_lock = threading.Lock()
        
        # State tracking
        self.training_history = []
        self.retraining_cooldowns = {}
        self.emergency_stops = set()
        
        # Performance tracking
        self.system_metrics = {
            'total_training_iterations': 0,
            'successful_retrainings': 0,
            'failed_retrainings': 0,
            'knowledge_sharing_events': 0,
            'degradation_detections': 0,
            'emergency_stops': 0
        }
        
        # Create directories
        for dir_path in [config.checkpoints_dir, config.models_dir, config.logs_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        logger.info("LiveTrainingCoordinator initialized")
    
    def _create_performance_monitor(self) -> TrainingProgressMonitor:
        """Create performance monitoring system"""
        monitoring_config = MonitoringConfig(
            update_frequency=self.config.monitoring_frequency,
            enable_live_plots=False,  # Disable for production
            enable_alerts=True,
            save_to_disk=True,
            track_system_metrics=True,
            track_gpu_metrics=True,
            memory_threshold_gb=self.config.max_memory_usage * psutil.virtual_memory().total / (1024**3)
        )
        
        return TrainingProgressMonitor(monitoring_config)
    
    def _create_checkpoint_manager(self) -> AdvancedCheckpointManager:
        """Create checkpoint management system"""
        checkpoint_config = CheckpointConfig(
            checkpoint_dir=self.config.checkpoints_dir,
            save_frequency=self.config.checkpoint_frequency,
            max_checkpoints=20,
            compression=True,
            verification=True
        )
        
        return AdvancedCheckpointManager(checkpoint_config)
    
    def register_agent(self, 
                      agent_id: str,
                      agent_name: str,
                      model: nn.Module,
                      optimizer: optim.Optimizer,
                      loss_fn: Callable,
                      device: torch.device = None) -> bool:
        """Register an agent for live training"""
        try:
            if agent_id in self.agent_managers:
                logger.warning(f"Agent {agent_id} already registered")
                return False
            
            device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Create agent training manager
            agent_manager = AgentTrainingManager(
                agent_id=agent_id,
                agent_name=agent_name,
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                config=self.config
            )
            
            with self.coordination_lock:
                self.agent_managers[agent_id] = agent_manager
            
            logger.info(f"Agent {agent_name} registered for live training")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent_name}: {e}")
            return False
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from live training"""
        try:
            with self.coordination_lock:
                if agent_id in self.agent_managers:
                    agent_manager = self.agent_managers.pop(agent_id)
                    agent_manager.cleanup()
                    logger.info(f"Agent {agent_id} unregistered")
                    return True
                else:
                    logger.warning(f"Agent {agent_id} not found for unregistration")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False
    
    def start_live_training(self):
        """Start live training coordination"""
        if self.training_active:
            logger.warning("Live training already active")
            return
        
        self.training_active = True
        self.current_phase = TrainingPhase.LIVE_EXECUTION
        
        # Start agent training
        for agent_manager in self.agent_managers.values():
            agent_manager.start_training()
        
        # Start monitoring
        self.performance_monitor.start_monitoring()
        
        # Start coordination threads
        self.coordination_thread = threading.Thread(target=self._coordination_loop)
        self.coordination_thread.daemon = True
        self.coordination_thread.start()
        
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.sync_thread = threading.Thread(target=self._synchronization_loop)
        self.sync_thread.daemon = True
        self.sync_thread.start()
        
        logger.info("Live training started")
    
    def stop_live_training(self):
        """Stop live training coordination"""
        if not self.training_active:
            return
        
        self.training_active = False
        
        # Stop agent training
        for agent_manager in self.agent_managers.values():
            agent_manager.stop_training()
        
        # Stop monitoring
        self.performance_monitor.stop_monitoring()
        
        # Wait for threads to finish
        for thread in [self.coordination_thread, self.monitoring_thread, self.sync_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5.0)
        
        # Save final checkpoint
        self._save_system_checkpoint()
        
        logger.info("Live training stopped")
    
    def _coordination_loop(self):
        """Main coordination loop"""
        while self.training_active:
            try:
                # Check system health
                self._check_system_health()
                
                # Check for degradation
                self._check_performance_degradation()
                
                # Handle retraining triggers
                self._handle_retraining_triggers()
                
                # Manage resources
                self._manage_resources()
                
                # Sleep
                time.sleep(self.config.monitoring_frequency)
                
            except Exception as e:
                logger.error(f"Coordination loop error: {e}")
                time.sleep(1.0)
    
    def _monitoring_loop(self):
        """Performance monitoring loop"""
        while self.training_active:
            try:
                self.current_phase = TrainingPhase.PERFORMANCE_MONITORING
                
                # Collect metrics from all agents
                all_metrics = {}
                
                with self.metrics_lock:
                    for agent_id, agent_manager in self.agent_managers.items():
                        metrics = agent_manager.get_metrics()
                        all_metrics[agent_id] = metrics
                        
                        # Log metrics to monitor
                        self.performance_monitor.log_multiple_metrics({
                            f"{agent_id}_performance_score": metrics.performance_score,
                            f"{agent_id}_loss": np.mean(metrics.loss_trend) if metrics.loss_trend else 0,
                            f"{agent_id}_accuracy": np.mean(metrics.accuracy_trend) if metrics.accuracy_trend else 0,
                            f"{agent_id}_gradient_norm": metrics.gradient_norm,
                            f"{agent_id}_memory_usage": metrics.memory_usage,
                            f"{agent_id}_throughput": metrics.throughput,
                            f"{agent_id}_latency": metrics.latency
                        }, epoch=0, step=self.system_metrics['total_training_iterations'])
                
                # Update degradation detector baselines
                for agent_id, metrics in all_metrics.items():
                    self.degradation_detector.update_baseline(agent_id, metrics)
                
                # Sleep
                time.sleep(self.config.monitoring_frequency)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(1.0)
    
    def _synchronization_loop(self):
        """Cross-agent synchronization loop"""
        while self.training_active:
            try:
                self.current_phase = TrainingPhase.SYNCHRONIZATION
                
                if self.config.enable_cross_agent_learning:
                    # Get current metrics
                    agent_metrics = {}
                    for agent_id, agent_manager in self.agent_managers.items():
                        agent_metrics[agent_id] = agent_manager.get_metrics()
                    
                    # Identify sharing opportunities
                    opportunities = self.cross_agent_manager.identify_knowledge_sharing_opportunities(agent_metrics)
                    
                    # Execute knowledge sharing
                    for opportunity in opportunities:
                        success = self.cross_agent_manager.execute_knowledge_sharing(
                            opportunity, self.agent_managers
                        )
                        
                        if success:
                            self.system_metrics['knowledge_sharing_events'] += 1
                
                # Sleep
                time.sleep(self.config.sync_frequency)
                
            except Exception as e:
                logger.error(f"Synchronization loop error: {e}")
                time.sleep(1.0)
    
    def _check_system_health(self):
        """Check overall system health"""
        try:
            # Check memory usage
            memory_info = psutil.virtual_memory()
            memory_usage = memory_info.percent / 100.0
            
            if memory_usage > self.config.max_memory_usage:
                logger.warning(f"High memory usage: {memory_usage:.1%}")
                self._handle_resource_pressure()
            
            # Check CPU usage
            cpu_usage = psutil.cpu_percent(interval=1) / 100.0
            
            if cpu_usage > self.config.max_cpu_usage:
                logger.warning(f"High CPU usage: {cpu_usage:.1%}")
                self._handle_resource_pressure()
            
            # Check agent health
            unhealthy_agents = []
            for agent_id, agent_manager in self.agent_managers.items():
                state = agent_manager.get_performance_state()
                if state == AgentPerformanceState.FAILED:
                    unhealthy_agents.append(agent_id)
            
            if unhealthy_agents:
                logger.warning(f"Unhealthy agents detected: {unhealthy_agents}")
                self._handle_agent_failures(unhealthy_agents)
                
        except Exception as e:
            logger.error(f"System health check failed: {e}")
    
    def _check_performance_degradation(self):
        """Check for performance degradation"""
        try:
            self.current_phase = TrainingPhase.DEGRADATION_DETECTION
            
            degraded_agents = []
            
            for agent_id, agent_manager in self.agent_managers.items():
                metrics = agent_manager.get_metrics()
                degradation_info = self.degradation_detector.detect_degradation(agent_id, metrics)
                
                if degradation_info['degraded']:
                    degraded_agents.append({
                        'agent_id': agent_id,
                        'degradation_info': degradation_info
                    })
                    
                    self.system_metrics['degradation_detections'] += 1
                    
                    logger.warning(f"Performance degradation detected for {agent_id}: {degradation_info['reasons']}")
            
            if degraded_agents:
                self._handle_performance_degradation(degraded_agents)
                
        except Exception as e:
            logger.error(f"Performance degradation check failed: {e}")
    
    def _handle_retraining_triggers(self):
        """Handle automatic retraining triggers"""
        try:
            current_time = datetime.now()
            
            for agent_id, agent_manager in self.agent_managers.items():
                # Check cooldown
                if agent_id in self.retraining_cooldowns:
                    time_since_last = (current_time - self.retraining_cooldowns[agent_id]).total_seconds()
                    if time_since_last < self.config.retraining_cooldown:
                        continue
                
                # Check if retraining is needed
                if agent_manager.should_retrain():
                    self._trigger_agent_retraining(agent_id)
                    
        except Exception as e:
            logger.error(f"Retraining trigger handling failed: {e}")
    
    def _trigger_agent_retraining(self, agent_id: str):
        """Trigger retraining for specific agent"""
        try:
            self.current_phase = TrainingPhase.RETRAINING
            
            agent_manager = self.agent_managers[agent_id]
            logger.info(f"Triggering retraining for agent {agent_id}")
            
            # Save current state
            checkpoint_id = self.checkpoint_manager.save_checkpoint(
                model=agent_manager.model,
                optimizer=agent_manager.optimizer,
                step=self.system_metrics['total_training_iterations'],
                metrics=agent_manager.get_metrics().to_dict(),
                metadata={'retraining_trigger': True}
            )
            
            # Reset training state
            agent_manager.optimizer.param_groups[0]['lr'] *= 0.5  # Reduce learning rate
            
            # Clear problematic data
            agent_manager.training_buffer.clear()
            
            # Record retraining
            self.retraining_cooldowns[agent_id] = datetime.now()
            self.system_metrics['successful_retrainings'] += 1
            
            logger.info(f"Retraining triggered for agent {agent_id}")
            
        except Exception as e:
            logger.error(f"Retraining trigger failed for agent {agent_id}: {e}")
            self.system_metrics['failed_retrainings'] += 1
    
    def _handle_resource_pressure(self):
        """Handle resource pressure"""
        try:
            # Reduce batch sizes
            for agent_manager in self.agent_managers.values():
                if hasattr(agent_manager, 'config'):
                    agent_manager.config.training_batch_size = max(8, agent_manager.config.training_batch_size // 2)
            
            # Force garbage collection
            gc.collect()
            
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Resource pressure handling failed: {e}")
    
    def _handle_agent_failures(self, failed_agents: List[str]):
        """Handle failed agents"""
        try:
            for agent_id in failed_agents:
                if agent_id in self.agent_managers:
                    agent_manager = self.agent_managers[agent_id]
                    
                    # Try to restore best model
                    agent_manager.restore_best_model()
                    
                    # Reset training state
                    agent_manager.metrics.error_rate = 0.0
                    agent_manager.metrics.performance_score = 0.5
                    
                    logger.info(f"Attempted recovery for failed agent {agent_id}")
                    
        except Exception as e:
            logger.error(f"Agent failure handling failed: {e}")
    
    def _handle_performance_degradation(self, degraded_agents: List[Dict[str, Any]]):
        """Handle performance degradation"""
        try:
            for agent_info in degraded_agents:
                agent_id = agent_info['agent_id']
                degradation_info = agent_info['degradation_info']
                
                if agent_id in self.agent_managers:
                    agent_manager = self.agent_managers[agent_id]
                    
                    # Apply recommendations
                    if 'Adjust learning rate' in degradation_info['recommendations']:
                        agent_manager.optimizer.param_groups[0]['lr'] *= 0.8
                    
                    if 'Consider retraining with fresh data' in degradation_info['recommendations']:
                        self._trigger_agent_retraining(agent_id)
                    
                    if 'Implement gradient clipping' in degradation_info['recommendations']:
                        # This would be handled in the training loop
                        pass
                    
                    logger.info(f"Applied degradation handling for agent {agent_id}")
                    
        except Exception as e:
            logger.error(f"Performance degradation handling failed: {e}")
    
    def _manage_resources(self):
        """Manage system resources"""
        try:
            # Monitor active agents
            active_agents = sum(1 for am in self.agent_managers.values() if am.training_active)
            
            if active_agents > self.config.max_concurrent_agents:
                # Temporarily pause some agents
                sorted_agents = sorted(
                    self.agent_managers.items(),
                    key=lambda x: x[1].get_metrics().performance_score,
                    reverse=True
                )
                
                for agent_id, agent_manager in sorted_agents[self.config.max_concurrent_agents:]:
                    if agent_manager.training_active:
                        agent_manager.stop_training()
                        logger.info(f"Temporarily paused training for agent {agent_id}")
                        
        except Exception as e:
            logger.error(f"Resource management failed: {e}")
    
    def _save_system_checkpoint(self):
        """Save system-wide checkpoint"""
        try:
            checkpoint_data = {
                'system_metrics': self.system_metrics,
                'config': asdict(self.config),
                'timestamp': datetime.now().isoformat(),
                'agent_states': {}
            }
            
            for agent_id, agent_manager in self.agent_managers.items():
                checkpoint_data['agent_states'][agent_id] = {
                    'model_state': agent_manager.model.state_dict(),
                    'optimizer_state': agent_manager.optimizer.state_dict(),
                    'metrics': agent_manager.get_metrics().to_dict()
                }
            
            checkpoint_path = Path(self.config.checkpoints_dir) / f"system_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            torch.save(checkpoint_data, checkpoint_path)
            
            logger.info(f"System checkpoint saved to {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"System checkpoint failed: {e}")
    
    def add_training_data(self, agent_id: str, data: Any):
        """Add training data for specific agent"""
        if agent_id in self.agent_managers:
            self.agent_managers[agent_id].add_training_data(data)
        else:
            logger.warning(f"Agent {agent_id} not found for training data")
    
    def add_validation_data(self, agent_id: str, data: Any):
        """Add validation data for specific agent"""
        if agent_id in self.agent_managers:
            self.agent_managers[agent_id].add_validation_data(data)
        else:
            logger.warning(f"Agent {agent_id} not found for validation data")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        agent_metrics = {}
        
        for agent_id, agent_manager in self.agent_managers.items():
            agent_metrics[agent_id] = agent_manager.get_metrics().to_dict()
        
        return {
            'system_metrics': self.system_metrics,
            'agent_metrics': agent_metrics,
            'current_phase': self.current_phase.value,
            'training_active': self.training_active,
            'total_agents': len(self.agent_managers),
            'active_agents': sum(1 for am in self.agent_managers.values() if am.training_active),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        report = {
            'summary': {
                'total_agents': len(self.agent_managers),
                'training_active': self.training_active,
                'current_phase': self.current_phase.value,
                'system_metrics': self.system_metrics
            },
            'agent_performance': {},
            'degradation_analysis': {},
            'cross_agent_analysis': {}
        }
        
        # Agent performance analysis
        for agent_id, agent_manager in self.agent_managers.items():
            metrics = agent_manager.get_metrics()
            state = agent_manager.get_performance_state()
            
            report['agent_performance'][agent_id] = {
                'metrics': metrics.to_dict(),
                'state': state.value,
                'should_retrain': agent_manager.should_retrain()
            }
            
            # Degradation analysis
            degradation_info = self.degradation_detector.detect_degradation(agent_id, metrics)
            report['degradation_analysis'][agent_id] = degradation_info
        
        # Cross-agent analysis
        if len(self.agent_managers) > 1:
            agent_metrics = {aid: am.get_metrics() for aid, am in self.agent_managers.items()}
            similarities = self.cross_agent_manager.analyze_agent_similarity(agent_metrics)
            opportunities = self.cross_agent_manager.identify_knowledge_sharing_opportunities(agent_metrics)
            
            report['cross_agent_analysis'] = {
                'similarities': similarities,
                'sharing_opportunities': opportunities
            }
        
        return report
    
    def emergency_stop(self, reason: str = "Manual emergency stop"):
        """Emergency stop all training"""
        logger.critical(f"Emergency stop triggered: {reason}")
        
        self.emergency_stops.add(reason)
        self.system_metrics['emergency_stops'] += 1
        
        # Stop all training immediately
        for agent_manager in self.agent_managers.values():
            agent_manager.stop_training()
        
        # Stop coordination
        self.training_active = False
        
        # Save emergency checkpoint
        self._save_system_checkpoint()
        
        logger.critical("Emergency stop completed")
    
    def resume_training(self):
        """Resume training after emergency stop"""
        if self.emergency_stops:
            logger.info("Clearing emergency stops and resuming training")
            self.emergency_stops.clear()
            self.start_live_training()
        else:
            logger.info("No emergency stops to clear")
    
    def cleanup(self):
        """Clean up all resources"""
        logger.info("Starting LiveTrainingCoordinator cleanup")
        
        # Stop training
        self.stop_live_training()
        
        # Cleanup agent managers
        for agent_manager in self.agent_managers.values():
            agent_manager.cleanup()
        
        # Cleanup monitoring
        if self.performance_monitor:
            self.performance_monitor.cleanup()
        
        # Clear data structures
        self.agent_managers.clear()
        self.training_history.clear()
        self.retraining_cooldowns.clear()
        self.emergency_stops.clear()
        
        logger.info("LiveTrainingCoordinator cleanup completed")


def create_live_training_config(**kwargs) -> LiveTrainingConfig:
    """Create live training configuration with defaults"""
    return LiveTrainingConfig(**kwargs)


def create_live_training_coordinator(config: LiveTrainingConfig = None) -> LiveTrainingCoordinator:
    """Create live training coordinator with default configuration"""
    if config is None:
        config = create_live_training_config()
    
    return LiveTrainingCoordinator(config)


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = create_live_training_config(
        learning_mode=LearningMode.CONTINUOUS,
        concurrent_training=True,
        max_concurrent_agents=4,
        enable_auto_retraining=True,
        enable_cross_agent_learning=True
    )
    
    # Create coordinator
    coordinator = create_live_training_coordinator(config)
    
    # Example agent registration (would normally be done with real models)
    try:
        # Create dummy model and optimizer
        model = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        
        # Register agent
        coordinator.register_agent(
            agent_id="strategic_agent",
            agent_name="Strategic Agent",
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn
        )
        
        # Start training
        coordinator.start_live_training()
        
        # Add some dummy training data
        for i in range(100):
            dummy_data = {
                'input': np.random.randn(100).tolist(),
                'target': np.random.randn(1).tolist()
            }
            coordinator.add_training_data("strategic_agent", dummy_data)
            time.sleep(0.1)
        
        # Let it run for a bit
        time.sleep(30)
        
        # Get performance report
        report = coordinator.get_performance_report()
        print(f"Performance report: {json.dumps(report, indent=2, default=str)}")
        
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        coordinator.cleanup()
        print("Example completed")