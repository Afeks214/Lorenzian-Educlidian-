"""
Live Learning Integration: Real-time Execution Learning System
============================================================

This module integrates the ExecutionBackpropagationBridge with the existing
UnifiedExecutionMARLSystem to enable continuous learning during live trading.
It provides the complete pipeline from execution outcomes to model updates
with <100ms latency requirements.

Key Features:
1. Seamless integration with existing execution system
2. Real-time execution monitoring and capture
3. Automatic gradient computation and model updates
4. Performance monitoring and optimization
5. Failsafe mechanisms for production stability

Architecture:
- ExecutionResultCapture: Captures execution outcomes in real-time
- LiveLearningOrchestrator: Coordinates the learning pipeline
- PerformanceMonitor: Tracks latency and learning effectiveness
- SafetyController: Ensures system stability during learning

Author: Claude - Live Learning Integration
Date: 2025-07-17
"""

import asyncio
import time
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import structlog
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from .execution_backpropagation_bridge import (
    ExecutionBackpropagationBridge,
    ExecutionResult,
    ExecutionOutcome,
    create_execution_backpropagation_bridge,
    DEFAULT_BRIDGE_CONFIG
)
from .unified_execution_marl_system import UnifiedExecutionMARLSystem, ExecutionDecision
from .agents.centralized_critic import MAPPOTrainer

logger = structlog.get_logger()


@dataclass
class LearningPerformanceMetrics:
    """Performance metrics for live learning system"""
    total_executions_processed: int = 0
    total_model_updates: int = 0
    avg_capture_to_update_latency_ms: float = 0.0
    learning_rate_adaptations: int = 0
    gradient_overflow_events: int = 0
    safety_interventions: int = 0
    
    # Learning effectiveness
    pre_learning_performance: Dict[str, float] = field(default_factory=dict)
    post_learning_performance: Dict[str, float] = field(default_factory=dict)
    performance_improvement: Dict[str, float] = field(default_factory=dict)
    
    # Latency breakdown
    execution_capture_latency_ms: float = 0.0
    gradient_computation_latency_ms: float = 0.0
    model_update_latency_ms: float = 0.0
    
    # Error tracking
    capture_errors: int = 0
    gradient_computation_errors: int = 0
    model_update_errors: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'total_executions_processed': self.total_executions_processed,
            'total_model_updates': self.total_model_updates,
            'avg_capture_to_update_latency_ms': self.avg_capture_to_update_latency_ms,
            'learning_rate_adaptations': self.learning_rate_adaptations,
            'gradient_overflow_events': self.gradient_overflow_events,
            'safety_interventions': self.safety_interventions,
            'pre_learning_performance': self.pre_learning_performance,
            'post_learning_performance': self.post_learning_performance,
            'performance_improvement': self.performance_improvement,
            'execution_capture_latency_ms': self.execution_capture_latency_ms,
            'gradient_computation_latency_ms': self.gradient_computation_latency_ms,
            'model_update_latency_ms': self.model_update_latency_ms,
            'capture_errors': self.capture_errors,
            'gradient_computation_errors': self.gradient_computation_errors,
            'model_update_errors': self.model_update_errors
        }


class ExecutionResultCapture:
    """Captures execution results in real-time for learning"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.capture_callbacks = []
        self.capture_metrics = deque(maxlen=1000)
        self.active = True
        
    def register_callback(self, callback: Callable[[ExecutionResult], None]):
        """Register callback for execution result capture"""
        self.capture_callbacks.append(callback)
    
    def capture_execution_result(self, 
                                execution_decision: ExecutionDecision,
                                actual_execution_data: Dict[str, Any]) -> ExecutionResult:
        """
        Capture execution result from decision and actual execution data
        
        Args:
            execution_decision: The original execution decision
            actual_execution_data: Actual execution data from broker
            
        Returns:
            ExecutionResult for learning
        """
        capture_start = time.perf_counter()
        
        try:
            # Extract execution details
            execution_result = ExecutionResult(
                timestamp=datetime.now(),
                symbol=actual_execution_data.get('symbol', ''),
                side=actual_execution_data.get('side', 'buy'),
                intended_quantity=execution_decision.final_position_size,
                filled_quantity=actual_execution_data.get('filled_quantity', 0),
                intended_price=actual_execution_data.get('intended_price', 0),
                fill_price=actual_execution_data.get('fill_price', 0),
                execution_time_ms=actual_execution_data.get('execution_time_ms', 0),
                slippage_bps=actual_execution_data.get('slippage_bps', 0),
                market_impact_bps=actual_execution_data.get('market_impact_bps', 0),
                fill_rate=actual_execution_data.get('fill_rate', execution_decision.fill_rate),
                latency_us=actual_execution_data.get('latency_us', execution_decision.total_latency_us),
                realized_pnl=actual_execution_data.get('realized_pnl', 0),
                unrealized_pnl=actual_execution_data.get('unrealized_pnl', 0),
                commission=actual_execution_data.get('commission', 0),
                fees=actual_execution_data.get('fees', 0),
                position_sizing_decision=self._extract_decision_dict(execution_decision.position_sizing),
                stop_target_decision=self._extract_decision_dict(execution_decision.stop_target),
                risk_monitor_decision=self._extract_decision_dict(execution_decision.risk_monitor),
                portfolio_optimizer_decision=self._extract_decision_dict(execution_decision.portfolio_optimizer),
                routing_decision=self._extract_decision_dict(execution_decision.routing),
                outcome=self._classify_execution_outcome(execution_decision, actual_execution_data),
                quality_score=self._compute_quality_score(execution_decision, actual_execution_data)
            )
            
            # Notify callbacks
            for callback in self.capture_callbacks:
                try:
                    callback(execution_result)
                except Exception as e:
                    logger.error("Error in capture callback", error=str(e))
            
            # Track capture metrics
            capture_time = (time.perf_counter() - capture_start) * 1000
            self.capture_metrics.append(capture_time)
            
            return execution_result
            
        except Exception as e:
            logger.error("Error capturing execution result", error=str(e))
            raise
    
    def _extract_decision_dict(self, decision: Any) -> Optional[Dict[str, Any]]:
        """Extract decision as dictionary"""
        if decision is None:
            return None
        
        if hasattr(decision, 'to_dict'):
            return decision.to_dict()
        elif hasattr(decision, '__dict__'):
            return decision.__dict__
        elif isinstance(decision, dict):
            return decision
        else:
            return {'value': str(decision)}
    
    def _classify_execution_outcome(self, 
                                   execution_decision: ExecutionDecision,
                                   actual_execution_data: Dict[str, Any]) -> ExecutionOutcome:
        """Classify execution outcome"""
        # Check for emergency stop
        if execution_decision.emergency_stop:
            return ExecutionOutcome.RISK_REJECTION
        
        # Check for routing failure
        if not execution_decision.selected_broker:
            return ExecutionOutcome.ROUTING_FAILURE
        
        # Check for timeout
        if actual_execution_data.get('execution_time_ms', 0) > 5000:  # 5 second timeout
            return ExecutionOutcome.TIMEOUT
        
        # Check for latency violation
        if actual_execution_data.get('latency_us', 0) > 1000:  # 1ms latency limit
            return ExecutionOutcome.LATENCY_VIOLATION
        
        # Check for excessive slippage
        if abs(actual_execution_data.get('slippage_bps', 0)) > 10:  # 10 bps threshold
            return ExecutionOutcome.SLIPPAGE_EXCESS
        
        # Check for partial fill
        intended_qty = execution_decision.final_position_size
        filled_qty = actual_execution_data.get('filled_quantity', 0)
        if intended_qty > 0 and filled_qty < intended_qty * 0.9:
            return ExecutionOutcome.PARTIAL_FILL
        
        # Check for significant market impact
        if abs(actual_execution_data.get('market_impact_bps', 0)) > 5:
            return ExecutionOutcome.MARKET_IMPACT
        
        return ExecutionOutcome.SUCCESS
    
    def _compute_quality_score(self, 
                              execution_decision: ExecutionDecision,
                              actual_execution_data: Dict[str, Any]) -> float:
        """Compute execution quality score (0.0 to 1.0)"""
        base_score = 1.0
        
        # Fill rate component
        fill_rate = actual_execution_data.get('fill_rate', execution_decision.fill_rate)
        fill_component = fill_rate
        
        # Slippage component
        slippage_bps = abs(actual_execution_data.get('slippage_bps', 0))
        slippage_component = max(0, 1.0 - slippage_bps / 20.0)  # Penalty for >20 bps
        
        # Latency component
        latency_us = actual_execution_data.get('latency_us', execution_decision.total_latency_us)
        latency_component = max(0, 1.0 - latency_us / 1000.0)  # Penalty for >1000μs
        
        # Market impact component
        market_impact_bps = abs(actual_execution_data.get('market_impact_bps', 0))
        impact_component = max(0, 1.0 - market_impact_bps / 10.0)  # Penalty for >10 bps
        
        # Weighted average
        quality_score = (
            0.3 * fill_component +
            0.3 * slippage_component +
            0.2 * latency_component +
            0.2 * impact_component
        )
        
        return max(0.0, min(1.0, quality_score))
    
    def get_capture_metrics(self) -> Dict[str, float]:
        """Get capture performance metrics"""
        if not self.capture_metrics:
            return {}
        
        return {
            'avg_capture_time_ms': np.mean(self.capture_metrics),
            'p95_capture_time_ms': np.percentile(self.capture_metrics, 95),
            'max_capture_time_ms': np.max(self.capture_metrics),
            'total_captures': len(self.capture_metrics)
        }


class SafetyController:
    """Safety controller for live learning system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Safety thresholds
        self.max_gradient_norm = self.config.get('max_gradient_norm', 10.0)
        self.max_learning_rate = self.config.get('max_learning_rate', 1e-2)
        self.max_consecutive_failures = self.config.get('max_consecutive_failures', 5)
        self.performance_degradation_threshold = self.config.get('performance_degradation_threshold', 0.1)
        
        # Safety state
        self.consecutive_failures = 0
        self.interventions = 0
        self.learning_enabled = True
        
        # Performance baseline
        self.baseline_metrics = {}
        self.current_metrics = {}
        
    def check_gradient_safety(self, gradients: Dict[str, torch.Tensor]) -> bool:
        """Check if gradients are safe to apply"""
        try:
            total_norm = 0.0
            for grad in gradients.values():
                total_norm += grad.norm().item() ** 2
            total_norm = total_norm ** 0.5
            
            if total_norm > self.max_gradient_norm:
                logger.warning("Gradient norm exceeded safety threshold",
                             gradient_norm=total_norm,
                             threshold=self.max_gradient_norm)
                self.interventions += 1
                return False
            
            # Check for NaN or infinite gradients
            for name, grad in gradients.items():
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    logger.warning("Invalid gradient detected",
                                 parameter=name,
                                 has_nan=torch.isnan(grad).any().item(),
                                 has_inf=torch.isinf(grad).any().item())
                    self.interventions += 1
                    return False
            
            return True
            
        except Exception as e:
            logger.error("Error checking gradient safety", error=str(e))
            self.interventions += 1
            return False
    
    def check_learning_rate_safety(self, learning_rate: float) -> bool:
        """Check if learning rate is safe"""
        if learning_rate > self.max_learning_rate:
            logger.warning("Learning rate exceeded safety threshold",
                         learning_rate=learning_rate,
                         threshold=self.max_learning_rate)
            self.interventions += 1
            return False
        
        return True
    
    def check_performance_degradation(self, current_metrics: Dict[str, float]) -> bool:
        """Check for performance degradation"""
        if not self.baseline_metrics:
            self.baseline_metrics = current_metrics.copy()
            return True
        
        # Check key performance metrics
        key_metrics = ['avg_latency_us', 'avg_slippage_bps', 'success_rate']
        
        for metric in key_metrics:
            if metric in current_metrics and metric in self.baseline_metrics:
                current = current_metrics[metric]
                baseline = self.baseline_metrics[metric]
                
                # Check for degradation (higher is worse for latency/slippage, lower is worse for success_rate)
                if metric == 'success_rate':
                    if current < baseline * (1 - self.performance_degradation_threshold):
                        logger.warning("Performance degradation detected",
                                     metric=metric,
                                     current=current,
                                     baseline=baseline)
                        self.interventions += 1
                        return False
                else:
                    if current > baseline * (1 + self.performance_degradation_threshold):
                        logger.warning("Performance degradation detected",
                                     metric=metric,
                                     current=current,
                                     baseline=baseline)
                        self.interventions += 1
                        return False
        
        return True
    
    def report_execution_failure(self):
        """Report execution failure for safety tracking"""
        self.consecutive_failures += 1
        
        if self.consecutive_failures >= self.max_consecutive_failures:
            logger.error("Too many consecutive failures, disabling learning",
                        consecutive_failures=self.consecutive_failures,
                        threshold=self.max_consecutive_failures)
            self.learning_enabled = False
            self.interventions += 1
    
    def report_execution_success(self):
        """Report execution success"""
        self.consecutive_failures = 0
    
    def is_learning_enabled(self) -> bool:
        """Check if learning is enabled"""
        return self.learning_enabled
    
    def enable_learning(self):
        """Enable learning (manual override)"""
        self.learning_enabled = True
        self.consecutive_failures = 0
    
    def get_safety_metrics(self) -> Dict[str, Any]:
        """Get safety metrics"""
        return {
            'consecutive_failures': self.consecutive_failures,
            'total_interventions': self.interventions,
            'learning_enabled': self.learning_enabled,
            'baseline_metrics': self.baseline_metrics,
            'current_metrics': self.current_metrics
        }


class LiveLearningOrchestrator:
    """Orchestrates the complete live learning pipeline"""
    
    def __init__(self,
                 unified_execution_system: UnifiedExecutionMARLSystem,
                 mappo_trainer: Optional[MAPPOTrainer] = None,
                 config: Dict[str, Any] = None):
        """
        Initialize live learning orchestrator
        
        Args:
            unified_execution_system: Unified execution MARL system
            mappo_trainer: MAPPO trainer (optional)
            config: Configuration dictionary
        """
        self.unified_execution_system = unified_execution_system
        self.mappo_trainer = mappo_trainer
        self.config = config or {}
        
        # Initialize components
        self.execution_capture = ExecutionResultCapture(self.config.get('capture', {}))
        self.safety_controller = SafetyController(self.config.get('safety', {}))
        
        # Initialize backpropagation bridge
        bridge_config = self.config.get('bridge', DEFAULT_BRIDGE_CONFIG)
        self.backprop_bridge = create_execution_backpropagation_bridge(
            mappo_trainer=self.mappo_trainer,
            unified_execution_system=self.unified_execution_system,
            config=bridge_config
        )
        
        # Performance metrics
        self.performance_metrics = LearningPerformanceMetrics()
        self.metrics_history = deque(maxlen=1000)
        
        # Register capture callback
        self.execution_capture.register_callback(self._handle_execution_result)
        
        # Background tasks
        self.monitoring_task = None
        self.active = False
        
        logger.info("LiveLearningOrchestrator initialized",
                   bridge_config=bridge_config,
                   safety_enabled=True)
    
    async def start(self):
        """Start the live learning system"""
        self.active = True
        
        # Start backpropagation bridge
        await self.backprop_bridge.start_processing()
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitor_performance())
        
        logger.info("Live learning system started")
    
    async def stop(self):
        """Stop the live learning system"""
        self.active = False
        
        # Stop monitoring
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Stop backpropagation bridge
        await self.backprop_bridge.stop_processing()
        
        logger.info("Live learning system stopped")
    
    def _handle_execution_result(self, execution_result: ExecutionResult):
        """Handle captured execution result"""
        try:
            # Check if learning is enabled
            if not self.safety_controller.is_learning_enabled():
                return
            
            # Update performance metrics
            self.performance_metrics.total_executions_processed += 1
            
            # Report to safety controller
            if execution_result.outcome == ExecutionOutcome.SUCCESS:
                self.safety_controller.report_execution_success()
            else:
                self.safety_controller.report_execution_failure()
            
            # Process through backpropagation bridge
            asyncio.create_task(self._process_execution_for_learning(execution_result))
            
        except Exception as e:
            logger.error("Error handling execution result", error=str(e))
            self.performance_metrics.capture_errors += 1
    
    async def _process_execution_for_learning(self, execution_result: ExecutionResult):
        """Process execution result for learning"""
        start_time = time.perf_counter()
        
        try:
            # Process through backpropagation bridge
            await self.backprop_bridge.process_execution_result(execution_result)
            
            # Update metrics
            self.performance_metrics.total_model_updates += 1
            
            # Track latency
            total_latency = (time.perf_counter() - start_time) * 1000
            self.performance_metrics.avg_capture_to_update_latency_ms = (
                self.performance_metrics.avg_capture_to_update_latency_ms * 0.9 +
                total_latency * 0.1
            )
            
        except Exception as e:
            logger.error("Error processing execution for learning", error=str(e))
            self.performance_metrics.model_update_errors += 1
    
    async def _monitor_performance(self):
        """Monitor system performance"""
        while self.active:
            try:
                # Get current metrics
                current_metrics = self.unified_execution_system.get_performance_report()
                
                # Check performance degradation
                if 'system_metrics' in current_metrics:
                    self.safety_controller.check_performance_degradation(
                        current_metrics['system_metrics']
                    )
                
                # Update performance metrics
                self._update_performance_metrics(current_metrics)
                
                # Sleep for monitoring interval
                await asyncio.sleep(self.config.get('monitoring_interval_seconds', 10))
                
            except Exception as e:
                logger.error("Error in performance monitoring", error=str(e))
                await asyncio.sleep(1)
    
    def _update_performance_metrics(self, current_metrics: Dict[str, Any]):
        """Update performance metrics"""
        try:
            # Extract relevant metrics
            system_metrics = current_metrics.get('system_metrics', {})
            
            # Update learning performance metrics
            self.performance_metrics.execution_capture_latency_ms = np.mean(
                list(self.execution_capture.capture_metrics) or [0]
            )
            
            # Get bridge metrics
            bridge_metrics = self.backprop_bridge.get_performance_metrics()
            if bridge_metrics:
                self.performance_metrics.gradient_computation_latency_ms = bridge_metrics.get(
                    'avg_gradient_computation_time_ms', 0
                )
                self.performance_metrics.model_update_latency_ms = bridge_metrics.get(
                    'avg_update_latency_ms', 0
                )
            
            # Store metrics history
            self.metrics_history.append({
                'timestamp': datetime.now().isoformat(),
                'performance_metrics': self.performance_metrics.to_dict(),
                'system_metrics': system_metrics,
                'bridge_metrics': bridge_metrics,
                'safety_metrics': self.safety_controller.get_safety_metrics()
            })
            
        except Exception as e:
            logger.error("Error updating performance metrics", error=str(e))
    
    def process_execution_decision(self, 
                                  execution_decision: ExecutionDecision,
                                  actual_execution_data: Dict[str, Any]):
        """
        Process execution decision and actual execution data for learning
        
        Args:
            execution_decision: The execution decision made by the system
            actual_execution_data: Actual execution data from broker
        """
        try:
            # Capture execution result
            execution_result = self.execution_capture.capture_execution_result(
                execution_decision, actual_execution_data
            )
            
            logger.debug("Execution result captured for learning",
                        symbol=execution_result.symbol,
                        outcome=execution_result.outcome.value,
                        quality_score=execution_result.quality_score)
            
        except Exception as e:
            logger.error("Error processing execution decision", error=str(e))
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for the live learning system"""
        return {
            'learning_metrics': self.performance_metrics.to_dict(),
            'capture_metrics': self.execution_capture.get_capture_metrics(),
            'bridge_metrics': self.backprop_bridge.get_performance_metrics(),
            'safety_metrics': self.safety_controller.get_safety_metrics(),
            'system_metrics': self.unified_execution_system.get_performance_report(),
            'is_active': self.active,
            'learning_enabled': self.safety_controller.is_learning_enabled()
        }
    
    def save_metrics(self, filepath: str):
        """Save comprehensive metrics to file"""
        try:
            metrics = self.get_comprehensive_metrics()
            metrics['metrics_history'] = list(self.metrics_history)
            
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info("Metrics saved", filepath=filepath)
            
        except Exception as e:
            logger.error("Error saving metrics", error=str(e))
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down LiveLearningOrchestrator")
        
        # Stop the system
        await self.stop()
        
        # Shutdown backpropagation bridge
        await self.backprop_bridge.shutdown()
        
        logger.info("LiveLearningOrchestrator shutdown complete")


# Factory function
def create_live_learning_orchestrator(
    unified_execution_system: UnifiedExecutionMARLSystem,
    mappo_trainer: Optional[MAPPOTrainer] = None,
    config: Dict[str, Any] = None
) -> LiveLearningOrchestrator:
    """Create and initialize live learning orchestrator"""
    return LiveLearningOrchestrator(
        unified_execution_system=unified_execution_system,
        mappo_trainer=mappo_trainer,
        config=config
    )


# Example configuration
DEFAULT_LIVE_LEARNING_CONFIG = {
    'monitoring_interval_seconds': 10,
    'capture': {
        'max_capture_time_ms': 10
    },
    'bridge': DEFAULT_BRIDGE_CONFIG,
    'safety': {
        'max_gradient_norm': 10.0,
        'max_learning_rate': 1e-2,
        'max_consecutive_failures': 5,
        'performance_degradation_threshold': 0.1
    }
}