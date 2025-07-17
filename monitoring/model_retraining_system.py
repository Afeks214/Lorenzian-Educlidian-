#!/usr/bin/env python3
"""
Automated Model Retraining Trigger System
Intelligent system that monitors model performance and automatically triggers retraining
when performance degradation is detected or specific conditions are met.
"""

import asyncio
import numpy as np
import pandas as pd
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle
import sqlite3
import redis
from pathlib import Path
import hashlib

# Machine Learning
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Statistical analysis
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Monitoring imports
from .performance_degradation_detector import PerformanceDegradationDetector, DegradationEvent, DegradationSeverity
from .alerting_system import AlertManager, Alert, AlertType, AlertSeverity, AlertStatus

# Metrics
from prometheus_client import Counter, Histogram, Gauge, Summary
MODEL_RETRAINING_TRIGGERS = Counter('model_retraining_triggers_total', 'Total retraining triggers', ['model_type', 'trigger_reason'])
MODEL_RETRAINING_DURATION = Histogram('model_retraining_duration_seconds', 'Model retraining duration', ['model_type'])
MODEL_PERFORMANCE_SCORE = Gauge('model_performance_score', 'Model performance score', ['model_type', 'metric'])
MODEL_RETRAINING_SUCCESS = Counter('model_retraining_success_total', 'Successful retraining attempts', ['model_type'])
MODEL_RETRAINING_FAILURES = Counter('model_retraining_failures_total', 'Failed retraining attempts', ['model_type', 'error_type'])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrainingTrigger(Enum):
    """Retraining trigger types."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SCHEDULED_RETRAINING = "scheduled_retraining"
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    MODEL_STALENESS = "model_staleness"
    ERROR_RATE_INCREASE = "error_rate_increase"
    ACCURACY_DROP = "accuracy_drop"
    MARKET_REGIME_CHANGE = "market_regime_change"
    MANUAL_TRIGGER = "manual_trigger"

class RetrainingPriority(Enum):
    """Retraining priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RetrainingStatus(Enum):
    """Retraining status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ModelPerformanceMetrics:
    """Model performance metrics."""
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mse: float
    mae: float
    r2_score: float
    timestamp: datetime
    sample_size: int
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_type': self.model_type,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'mse': self.mse,
            'mae': self.mae,
            'r2_score': self.r2_score,
            'timestamp': self.timestamp.isoformat(),
            'sample_size': self.sample_size,
            'metadata': self.metadata or {}
        }

@dataclass
class RetrainingRequest:
    """Model retraining request."""
    request_id: str
    model_type: str
    trigger: RetrainingTrigger
    priority: RetrainingPriority
    status: RetrainingStatus
    created_time: datetime
    trigger_data: Dict[str, Any]
    performance_baseline: ModelPerformanceMetrics
    configuration: Dict[str, Any]
    estimated_duration: int  # seconds
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'request_id': self.request_id,
            'model_type': self.model_type,
            'trigger': self.trigger.value,
            'priority': self.priority.value,
            'status': self.status.value,
            'created_time': self.created_time.isoformat(),
            'trigger_data': self.trigger_data,
            'performance_baseline': self.performance_baseline.to_dict() if self.performance_baseline else None,
            'configuration': self.configuration,
            'estimated_duration': self.estimated_duration,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries
        }

class ModelPerformanceMonitor:
    """Monitor model performance and detect degradation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.performance_history = defaultdict(deque)
        self.performance_baselines = {}
        self.min_samples = config.get('min_samples', 100)
        self.performance_window = config.get('performance_window', 1000)
        
        # Performance thresholds
        self.accuracy_threshold = config.get('accuracy_threshold', 0.05)  # 5% drop
        self.error_rate_threshold = config.get('error_rate_threshold', 0.1)  # 10% increase
        self.drift_threshold = config.get('drift_threshold', 0.1)
        
    def update_performance(self, metrics: ModelPerformanceMetrics):
        """Update model performance metrics."""
        model_type = metrics.model_type
        
        # Add to history
        self.performance_history[model_type].append(metrics)
        
        # Maintain window size
        if len(self.performance_history[model_type]) > self.performance_window:
            self.performance_history[model_type].popleft()
        
        # Update baseline if this is better performance
        if (model_type not in self.performance_baselines or 
            metrics.accuracy > self.performance_baselines[model_type].accuracy):
            self.performance_baselines[model_type] = metrics
        
        # Update Prometheus metrics
        MODEL_PERFORMANCE_SCORE.labels(model_type=model_type, metric='accuracy').set(metrics.accuracy)
        MODEL_PERFORMANCE_SCORE.labels(model_type=model_type, metric='f1_score').set(metrics.f1_score)
        MODEL_PERFORMANCE_SCORE.labels(model_type=model_type, metric='mse').set(metrics.mse)
    
    def detect_performance_degradation(self, model_type: str) -> Optional[Dict[str, Any]]:
        """Detect performance degradation for a model."""
        if model_type not in self.performance_history:
            return None
        
        history = list(self.performance_history[model_type])
        
        if len(history) < self.min_samples:
            return None
        
        try:
            # Get baseline performance
            baseline = self.performance_baselines.get(model_type)
            if not baseline:
                return None
            
            # Calculate recent performance
            recent_window = min(50, len(history) // 4)
            recent_metrics = history[-recent_window:]
            
            recent_accuracy = np.mean([m.accuracy for m in recent_metrics])
            recent_f1 = np.mean([m.f1_score for m in recent_metrics])
            recent_mse = np.mean([m.mse for m in recent_metrics])
            
            # Check for significant degradation
            accuracy_drop = baseline.accuracy - recent_accuracy
            f1_drop = baseline.f1_score - recent_f1
            mse_increase = recent_mse - baseline.mse
            
            # Determine if retraining is needed
            degradation_detected = False
            degradation_reasons = []
            
            if accuracy_drop > self.accuracy_threshold:
                degradation_detected = True
                degradation_reasons.append(f"Accuracy dropped by {accuracy_drop:.3f}")
            
            if f1_drop > self.accuracy_threshold:
                degradation_detected = True
                degradation_reasons.append(f"F1 score dropped by {f1_drop:.3f}")
            
            if mse_increase > baseline.mse * 0.2:  # 20% increase in error
                degradation_detected = True
                degradation_reasons.append(f"MSE increased by {mse_increase:.3f}")
            
            if degradation_detected:
                return {
                    'model_type': model_type,
                    'degradation_detected': True,
                    'degradation_reasons': degradation_reasons,
                    'baseline_accuracy': baseline.accuracy,
                    'recent_accuracy': recent_accuracy,
                    'accuracy_drop': accuracy_drop,
                    'baseline_f1': baseline.f1_score,
                    'recent_f1': recent_f1,
                    'f1_drop': f1_drop,
                    'baseline_mse': baseline.mse,
                    'recent_mse': recent_mse,
                    'mse_increase': mse_increase,
                    'sample_size': len(recent_metrics)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting performance degradation for {model_type}: {e}")
            return None
    
    def detect_data_drift(self, model_type: str, current_data: np.ndarray, 
                         reference_data: np.ndarray) -> Dict[str, Any]:
        """Detect data drift using statistical tests."""
        try:
            drift_results = {}
            
            # Kolmogorov-Smirnov test for distribution difference
            if len(current_data) > 10 and len(reference_data) > 10:
                ks_statistic, ks_p_value = stats.ks_2samp(reference_data, current_data)
                
                drift_results['ks_test'] = {
                    'statistic': ks_statistic,
                    'p_value': ks_p_value,
                    'drift_detected': ks_p_value < 0.05 and ks_statistic > self.drift_threshold
                }
            
            # Mean shift detection
            if len(current_data) > 10 and len(reference_data) > 10:
                t_statistic, t_p_value = stats.ttest_ind(reference_data, current_data)
                
                drift_results['mean_shift'] = {
                    'statistic': t_statistic,
                    'p_value': t_p_value,
                    'drift_detected': t_p_value < 0.05,
                    'reference_mean': np.mean(reference_data),
                    'current_mean': np.mean(current_data)
                }
            
            # Variance change detection
            if len(current_data) > 10 and len(reference_data) > 10:
                f_statistic = np.var(current_data) / np.var(reference_data)
                
                drift_results['variance_change'] = {
                    'f_statistic': f_statistic,
                    'drift_detected': f_statistic > 2.0 or f_statistic < 0.5,
                    'reference_var': np.var(reference_data),
                    'current_var': np.var(current_data)
                }
            
            # Overall drift assessment
            drift_detected = any(
                result.get('drift_detected', False) 
                for result in drift_results.values()
            )
            
            return {
                'model_type': model_type,
                'drift_detected': drift_detected,
                'drift_tests': drift_results,
                'drift_score': self._calculate_drift_score(drift_results)
            }
            
        except Exception as e:
            logger.error(f"Error detecting data drift for {model_type}: {e}")
            return {'model_type': model_type, 'drift_detected': False, 'error': str(e)}
    
    def _calculate_drift_score(self, drift_results: Dict[str, Any]) -> float:
        """Calculate overall drift score."""
        try:
            scores = []
            
            # KS test contribution
            if 'ks_test' in drift_results:
                ks_score = drift_results['ks_test']['statistic']
                scores.append(ks_score)
            
            # Mean shift contribution
            if 'mean_shift' in drift_results:
                t_score = min(1.0, abs(drift_results['mean_shift']['statistic']) / 3.0)
                scores.append(t_score)
            
            # Variance change contribution
            if 'variance_change' in drift_results:
                f_score = min(1.0, abs(np.log(drift_results['variance_change']['f_statistic'])))
                scores.append(f_score)
            
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating drift score: {e}")
            return 0.0

class ModelRetrainingManager:
    """Manage model retraining requests and execution."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.retraining_queue = asyncio.Queue()
        self.active_retraining = {}
        self.retraining_history = deque(maxlen=config.get('history_size', 100))
        
        # Retraining configuration
        self.max_concurrent_retraining = config.get('max_concurrent_retraining', 2)
        self.retraining_timeout = config.get('retraining_timeout', 3600)  # 1 hour
        self.retry_delay = config.get('retry_delay', 300)  # 5 minutes
        
        # Model configurations
        self.model_configs = config.get('model_configs', {})
        
        # External dependencies
        self.training_system = None
        self.model_storage = None
        self.alert_manager = None
        
    def set_dependencies(self, training_system=None, model_storage=None, alert_manager=None):
        """Set external dependencies."""
        self.training_system = training_system
        self.model_storage = model_storage
        self.alert_manager = alert_manager
    
    async def trigger_retraining(self, model_type: str, trigger: RetrainingTrigger, 
                               priority: RetrainingPriority, trigger_data: Dict[str, Any],
                               performance_baseline: ModelPerformanceMetrics = None) -> str:
        """Trigger model retraining."""
        request_id = f"retrain_{model_type}_{int(time.time())}"
        
        try:
            # Create retraining request
            request = RetrainingRequest(
                request_id=request_id,
                model_type=model_type,
                trigger=trigger,
                priority=priority,
                status=RetrainingStatus.PENDING,
                created_time=datetime.utcnow(),
                trigger_data=trigger_data,
                performance_baseline=performance_baseline,
                configuration=self.model_configs.get(model_type, {}),
                estimated_duration=self._estimate_retraining_duration(model_type)
            )
            
            # Add to queue
            await self.retraining_queue.put(request)
            
            # Update metrics
            MODEL_RETRAINING_TRIGGERS.labels(
                model_type=model_type,
                trigger_reason=trigger.value
            ).inc()
            
            # Send alert
            if self.alert_manager:
                await self._send_retraining_alert(request)
            
            logger.info(f"Retraining triggered for {model_type}: {request_id}")
            
            return request_id
            
        except Exception as e:
            logger.error(f"Error triggering retraining for {model_type}: {e}")
            raise
    
    async def start_retraining_manager(self):
        """Start the retraining manager."""
        logger.info("Starting model retraining manager")
        
        # Start worker tasks
        workers = [
            asyncio.create_task(self._retraining_worker(i))
            for i in range(self.max_concurrent_retraining)
        ]
        
        # Start monitoring task
        monitor_task = asyncio.create_task(self._monitor_retraining())
        
        # Wait for all tasks
        await asyncio.gather(*workers, monitor_task)
    
    async def _retraining_worker(self, worker_id: int):
        """Worker for processing retraining requests."""
        logger.info(f"Starting retraining worker {worker_id}")
        
        while True:
            try:
                # Get request from queue
                request = await self.retraining_queue.get()
                
                # Update status
                request.status = RetrainingStatus.QUEUED
                
                # Check if we can start retraining
                if len(self.active_retraining) >= self.max_concurrent_retraining:
                    await self.retraining_queue.put(request)  # Put back in queue
                    await asyncio.sleep(30)  # Wait and try again
                    continue
                
                # Start retraining
                await self._execute_retraining(request)
                
            except Exception as e:
                logger.error(f"Error in retraining worker {worker_id}: {e}")
                await asyncio.sleep(60)
    
    async def _execute_retraining(self, request: RetrainingRequest):
        """Execute model retraining."""
        start_time = time.time()
        
        try:
            # Update status
            request.status = RetrainingStatus.RUNNING
            self.active_retraining[request.request_id] = request
            
            logger.info(f"Starting retraining for {request.model_type}: {request.request_id}")
            
            # Execute retraining based on model type
            if request.model_type == 'tactical_model':
                success = await self._retrain_tactical_model(request)
            elif request.model_type == 'strategic_model':
                success = await self._retrain_strategic_model(request)
            elif request.model_type == 'risk_model':
                success = await self._retrain_risk_model(request)
            else:
                success = await self._retrain_default_model(request)
            
            # Update status based on success
            if success:
                request.status = RetrainingStatus.COMPLETED
                MODEL_RETRAINING_SUCCESS.labels(model_type=request.model_type).inc()
                logger.info(f"Retraining completed successfully: {request.request_id}")
            else:
                request.status = RetrainingStatus.FAILED
                MODEL_RETRAINING_FAILURES.labels(
                    model_type=request.model_type,
                    error_type='training_failed'
                ).inc()
                logger.error(f"Retraining failed: {request.request_id}")
                
                # Retry if possible
                if request.retry_count < request.max_retries:
                    request.retry_count += 1
                    request.status = RetrainingStatus.PENDING
                    await asyncio.sleep(self.retry_delay)
                    await self.retraining_queue.put(request)
            
            # Record duration
            duration = time.time() - start_time
            MODEL_RETRAINING_DURATION.labels(model_type=request.model_type).observe(duration)
            
            # Send completion alert
            if self.alert_manager:
                await self._send_completion_alert(request, duration)
            
        except Exception as e:
            logger.error(f"Error executing retraining for {request.request_id}: {e}")
            request.status = RetrainingStatus.FAILED
            MODEL_RETRAINING_FAILURES.labels(
                model_type=request.model_type,
                error_type='execution_error'
            ).inc()
            
        finally:
            # Remove from active retraining
            self.active_retraining.pop(request.request_id, None)
            
            # Add to history
            self.retraining_history.append(request)
    
    async def _retrain_tactical_model(self, request: RetrainingRequest) -> bool:
        """Retrain tactical model."""
        try:
            if not self.training_system:
                logger.error("Training system not available")
                return False
            
            # Get configuration
            config = request.configuration
            
            # Prepare training data
            training_data = await self._prepare_training_data(request.model_type, config)
            
            # Train model
            model = await self.training_system.train_tactical_model(
                training_data=training_data,
                config=config,
                baseline_performance=request.performance_baseline
            )
            
            # Validate model
            validation_metrics = await self._validate_model(model, request.model_type)
            
            # Save model if validation passes
            if validation_metrics and validation_metrics.accuracy > 0.7:  # Minimum threshold
                await self._save_model(model, request.model_type, validation_metrics)
                return True
            else:
                logger.warning(f"Model validation failed for {request.model_type}")
                return False
            
        except Exception as e:
            logger.error(f"Error retraining tactical model: {e}")
            return False
    
    async def _retrain_strategic_model(self, request: RetrainingRequest) -> bool:
        """Retrain strategic model."""
        try:
            if not self.training_system:
                logger.error("Training system not available")
                return False
            
            # Get configuration
            config = request.configuration
            
            # Prepare training data
            training_data = await self._prepare_training_data(request.model_type, config)
            
            # Train model
            model = await self.training_system.train_strategic_model(
                training_data=training_data,
                config=config,
                baseline_performance=request.performance_baseline
            )
            
            # Validate model
            validation_metrics = await self._validate_model(model, request.model_type)
            
            # Save model if validation passes
            if validation_metrics and validation_metrics.accuracy > 0.7:
                await self._save_model(model, request.model_type, validation_metrics)
                return True
            else:
                logger.warning(f"Model validation failed for {request.model_type}")
                return False
            
        except Exception as e:
            logger.error(f"Error retraining strategic model: {e}")
            return False
    
    async def _retrain_risk_model(self, request: RetrainingRequest) -> bool:
        """Retrain risk model."""
        try:
            if not self.training_system:
                logger.error("Training system not available")
                return False
            
            # Get configuration
            config = request.configuration
            
            # Prepare training data
            training_data = await self._prepare_training_data(request.model_type, config)
            
            # Train model
            model = await self.training_system.train_risk_model(
                training_data=training_data,
                config=config,
                baseline_performance=request.performance_baseline
            )
            
            # Validate model
            validation_metrics = await self._validate_model(model, request.model_type)
            
            # Save model if validation passes
            if validation_metrics and validation_metrics.accuracy > 0.7:
                await self._save_model(model, request.model_type, validation_metrics)
                return True
            else:
                logger.warning(f"Model validation failed for {request.model_type}")
                return False
            
        except Exception as e:
            logger.error(f"Error retraining risk model: {e}")
            return False
    
    async def _retrain_default_model(self, request: RetrainingRequest) -> bool:
        """Retrain default model."""
        try:
            # Placeholder for default model retraining
            logger.info(f"Retraining default model: {request.model_type}")
            
            # Simulate training time
            await asyncio.sleep(60)
            
            return True
            
        except Exception as e:
            logger.error(f"Error retraining default model: {e}")
            return False
    
    async def _prepare_training_data(self, model_type: str, config: Dict[str, Any]) -> Any:
        """Prepare training data for model."""
        try:
            # This is a placeholder for data preparation
            # In practice, this would fetch and prepare actual training data
            logger.info(f"Preparing training data for {model_type}")
            
            # Simulate data preparation
            await asyncio.sleep(30)
            
            return {"prepared_data": True, "model_type": model_type}
            
        except Exception as e:
            logger.error(f"Error preparing training data for {model_type}: {e}")
            return None
    
    async def _validate_model(self, model: Any, model_type: str) -> Optional[ModelPerformanceMetrics]:
        """Validate trained model."""
        try:
            # This is a placeholder for model validation
            # In practice, this would run validation on held-out data
            logger.info(f"Validating model: {model_type}")
            
            # Simulate validation
            await asyncio.sleep(60)
            
            # Return mock validation metrics
            return ModelPerformanceMetrics(
                model_type=model_type,
                accuracy=0.85,
                precision=0.82,
                recall=0.88,
                f1_score=0.85,
                mse=0.15,
                mae=0.12,
                r2_score=0.78,
                timestamp=datetime.utcnow(),
                sample_size=1000
            )
            
        except Exception as e:
            logger.error(f"Error validating model {model_type}: {e}")
            return None
    
    async def _save_model(self, model: Any, model_type: str, metrics: ModelPerformanceMetrics):
        """Save trained model."""
        try:
            if self.model_storage:
                await self.model_storage.save_model(
                    model=model,
                    model_type=model_type,
                    metrics=metrics,
                    timestamp=datetime.utcnow()
                )
                logger.info(f"Model saved: {model_type}")
            else:
                logger.warning("Model storage not available")
                
        except Exception as e:
            logger.error(f"Error saving model {model_type}: {e}")
    
    async def _monitor_retraining(self):
        """Monitor retraining progress."""
        while True:
            try:
                # Check for stuck retraining jobs
                current_time = datetime.utcnow()
                
                for request_id, request in list(self.active_retraining.items()):
                    if request.status == RetrainingStatus.RUNNING:
                        elapsed = (current_time - request.created_time).total_seconds()
                        
                        if elapsed > self.retraining_timeout:
                            logger.warning(f"Retraining timeout: {request_id}")
                            request.status = RetrainingStatus.FAILED
                            self.active_retraining.pop(request_id, None)
                            
                            MODEL_RETRAINING_FAILURES.labels(
                                model_type=request.model_type,
                                error_type='timeout'
                            ).inc()
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in retraining monitor: {e}")
                await asyncio.sleep(300)
    
    async def _send_retraining_alert(self, request: RetrainingRequest):
        """Send alert for retraining start."""
        try:
            alert = Alert(
                alert_id=f"retraining_start_{request.request_id}",
                alert_type=AlertType.SYSTEM_PERFORMANCE,
                severity=AlertSeverity.MEDIUM,
                title=f"Model Retraining Started: {request.model_type}",
                description=f"Retraining triggered for {request.model_type} due to {request.trigger.value}",
                timestamp=request.created_time,
                source='model_retraining_system',
                status=AlertStatus.ACTIVE,
                metadata={
                    'request_id': request.request_id,
                    'model_type': request.model_type,
                    'trigger': request.trigger.value,
                    'priority': request.priority.value,
                    'estimated_duration': request.estimated_duration
                },
                tags=[request.model_type, 'retraining', request.trigger.value]
            )
            
            await self.alert_manager.create_alert(alert)
            
        except Exception as e:
            logger.error(f"Error sending retraining alert: {e}")
    
    async def _send_completion_alert(self, request: RetrainingRequest, duration: float):
        """Send alert for retraining completion."""
        try:
            severity = AlertSeverity.LOW if request.status == RetrainingStatus.COMPLETED else AlertSeverity.HIGH
            
            alert = Alert(
                alert_id=f"retraining_complete_{request.request_id}",
                alert_type=AlertType.SYSTEM_PERFORMANCE,
                severity=severity,
                title=f"Model Retraining {'Completed' if request.status == RetrainingStatus.COMPLETED else 'Failed'}: {request.model_type}",
                description=f"Retraining {'completed successfully' if request.status == RetrainingStatus.COMPLETED else 'failed'} for {request.model_type}",
                timestamp=datetime.utcnow(),
                source='model_retraining_system',
                status=AlertStatus.ACTIVE,
                metadata={
                    'request_id': request.request_id,
                    'model_type': request.model_type,
                    'final_status': request.status.value,
                    'duration_seconds': duration,
                    'retry_count': request.retry_count
                },
                tags=[request.model_type, 'retraining', request.status.value]
            )
            
            await self.alert_manager.create_alert(alert)
            
        except Exception as e:
            logger.error(f"Error sending completion alert: {e}")
    
    def _estimate_retraining_duration(self, model_type: str) -> int:
        """Estimate retraining duration."""
        # Default durations in seconds
        durations = {
            'tactical_model': 3600,    # 1 hour
            'strategic_model': 7200,   # 2 hours
            'risk_model': 1800,        # 30 minutes
            'default_model': 1800      # 30 minutes
        }
        
        return durations.get(model_type, 1800)
    
    def get_retraining_status(self) -> Dict[str, Any]:
        """Get current retraining status."""
        return {
            'active_retraining': {
                req_id: req.to_dict() for req_id, req in self.active_retraining.items()
            },
            'queue_size': self.retraining_queue.qsize(),
            'recent_history': [req.to_dict() for req in list(self.retraining_history)[-10:]],
            'configuration': {
                'max_concurrent_retraining': self.max_concurrent_retraining,
                'retraining_timeout': self.retraining_timeout,
                'retry_delay': self.retry_delay
            }
        }

class AutomatedModelRetrainingSystem:
    """Complete automated model retraining system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.performance_monitor = ModelPerformanceMonitor(config.get('performance_monitor', {}))
        self.retraining_manager = ModelRetrainingManager(config.get('retraining_manager', {}))
        
        # Monitoring settings
        self.monitoring_active = False
        self.monitoring_interval = config.get('monitoring_interval', 300)  # 5 minutes
        
        # External dependencies
        self.degradation_detector = None
        self.alert_manager = None
        
    def set_dependencies(self, degradation_detector=None, alert_manager=None, 
                       training_system=None, model_storage=None):
        """Set external dependencies."""
        self.degradation_detector = degradation_detector
        self.alert_manager = alert_manager
        self.retraining_manager.set_dependencies(
            training_system=training_system,
            model_storage=model_storage,
            alert_manager=alert_manager
        )
    
    async def start_system(self):
        """Start the automated retraining system."""
        self.monitoring_active = True
        logger.info("Starting automated model retraining system")
        
        # Start components
        monitoring_task = asyncio.create_task(self._monitoring_loop())
        retraining_task = asyncio.create_task(self.retraining_manager.start_retraining_manager())
        
        await asyncio.gather(monitoring_task, retraining_task)
    
    async def stop_system(self):
        """Stop the system."""
        self.monitoring_active = False
        logger.info("Stopping automated model retraining system")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Check for performance degradation
                await self._check_performance_degradation()
                
                # Check for data drift
                await self._check_data_drift()
                
                # Check for scheduled retraining
                await self._check_scheduled_retraining()
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _check_performance_degradation(self):
        """Check for model performance degradation."""
        try:
            # Get all monitored models
            monitored_models = self.config.get('monitored_models', ['tactical_model', 'strategic_model', 'risk_model'])
            
            for model_type in monitored_models:
                degradation = self.performance_monitor.detect_performance_degradation(model_type)
                
                if degradation and degradation.get('degradation_detected'):
                    logger.warning(f"Performance degradation detected for {model_type}")
                    
                    # Determine priority based on degradation severity
                    accuracy_drop = degradation.get('accuracy_drop', 0)
                    if accuracy_drop > 0.2:  # 20% drop
                        priority = RetrainingPriority.CRITICAL
                    elif accuracy_drop > 0.1:  # 10% drop
                        priority = RetrainingPriority.HIGH
                    elif accuracy_drop > 0.05:  # 5% drop
                        priority = RetrainingPriority.MEDIUM
                    else:
                        priority = RetrainingPriority.LOW
                    
                    # Trigger retraining
                    await self.retraining_manager.trigger_retraining(
                        model_type=model_type,
                        trigger=RetrainingTrigger.PERFORMANCE_DEGRADATION,
                        priority=priority,
                        trigger_data=degradation
                    )
                    
        except Exception as e:
            logger.error(f"Error checking performance degradation: {e}")
    
    async def _check_data_drift(self):
        """Check for data drift."""
        try:
            # This is a placeholder for data drift detection
            # In practice, this would compare current data distributions
            # with reference distributions
            
            logger.debug("Checking for data drift")
            
        except Exception as e:
            logger.error(f"Error checking data drift: {e}")
    
    async def _check_scheduled_retraining(self):
        """Check for scheduled retraining."""
        try:
            # This is a placeholder for scheduled retraining logic
            # In practice, this would check schedules and trigger retraining
            
            logger.debug("Checking for scheduled retraining")
            
        except Exception as e:
            logger.error(f"Error checking scheduled retraining: {e}")
    
    def update_model_performance(self, metrics: ModelPerformanceMetrics):
        """Update model performance metrics."""
        self.performance_monitor.update_performance(metrics)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            'monitoring_active': self.monitoring_active,
            'monitoring_interval': self.monitoring_interval,
            'performance_monitor': {
                'models_monitored': len(self.performance_monitor.performance_history),
                'baselines_established': len(self.performance_monitor.performance_baselines)
            },
            'retraining_manager': self.retraining_manager.get_retraining_status(),
            'last_updated': datetime.utcnow().isoformat()
        }

# Factory function
def create_automated_retraining_system(config: Dict[str, Any]) -> AutomatedModelRetrainingSystem:
    """Create automated retraining system instance."""
    return AutomatedModelRetrainingSystem(config)

# Example configuration
EXAMPLE_CONFIG = {
    'monitoring_interval': 300,  # 5 minutes
    'monitored_models': ['tactical_model', 'strategic_model', 'risk_model'],
    'performance_monitor': {
        'min_samples': 100,
        'performance_window': 1000,
        'accuracy_threshold': 0.05,
        'error_rate_threshold': 0.1,
        'drift_threshold': 0.1
    },
    'retraining_manager': {
        'max_concurrent_retraining': 2,
        'retraining_timeout': 3600,
        'retry_delay': 300,
        'history_size': 100,
        'model_configs': {
            'tactical_model': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100
            },
            'strategic_model': {
                'learning_rate': 0.0001,
                'batch_size': 64,
                'epochs': 200
            },
            'risk_model': {
                'learning_rate': 0.01,
                'batch_size': 16,
                'epochs': 50
            }
        }
    }
}

# Example usage
async def main():
    """Example usage of automated retraining system."""
    config = EXAMPLE_CONFIG
    system = create_automated_retraining_system(config)
    
    # Start system
    await system.start_system()

if __name__ == "__main__":
    asyncio.run(main())