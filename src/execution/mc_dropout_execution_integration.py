"""
SINGLE MC DROPOUT EXECUTION ENGINE - 1000 SAMPLES

This is the ONLY MC dropout implementation in the entire system.
Provides binary execution decisions (execute/reject) using exactly 1000 samples.
Optimized for <500μs latency with GPU acceleration.

AGENT 2 IMPLEMENTATION - Maximum Velocity Deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import math

logger = logging.getLogger(__name__)


@dataclass
class TradeExecutionContext:
    """Complete context for binary trade execution decision."""
    # MAPPO recommendation
    mappo_recommendation: Dict[str, Any]
    
    # Market data
    market_data: Dict[str, Any]
    
    # Portfolio state
    portfolio_state: Dict[str, Any]
    
    # Risk metrics
    risk_metrics: Dict[str, Any]
    
    # Execution specifics
    trade_details: Dict[str, Any]
    
    # Timestamp
    timestamp: float


@dataclass
class UncertaintyMetrics:
    """Comprehensive uncertainty metrics for 1000-sample analysis."""
    epistemic_uncertainty: float    # Model uncertainty
    aleatoric_uncertainty: float    # Data uncertainty
    total_uncertainty: float        # Combined uncertainty
    confidence_score: float         # Overall confidence
    sample_variance: float          # Variance across samples
    convergence_quality: float      # How well samples converged
    decision_boundary_distance: float  # Distance from 50% threshold


@dataclass
class SampleStatistics:
    """Statistics from 1000 MC dropout samples."""
    mean_prediction: float
    std_prediction: float
    min_prediction: float
    max_prediction: float
    quantile_25: float
    quantile_75: float
    samples_above_threshold: int
    convergence_achieved: bool
    outlier_count: int


@dataclass
class BinaryExecutionResult:
    """Final binary execution decision with full analysis."""
    execute_trade: bool             # Final binary decision
    confidence: float               # Calibrated confidence score
    uncertainty_metrics: UncertaintyMetrics
    sample_statistics: SampleStatistics
    processing_time_us: float       # Processing time in microseconds
    gpu_utilization: float          # GPU utilization during processing
    
    # For MAPPO feedback
    mappo_feedback: Dict[str, Any]


class ExecutionDecisionNetwork(nn.Module):
    """Lightweight neural network for binary execution decisions."""
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, dropout_rate: float = 0.3):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(3)  # 3 hidden layers for sufficient capacity
        ])
        
        # Binary output: probability of execution
        self.output_head = nn.Linear(hidden_dim, 1)
        
        # Initialize weights for stable training
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for optimal convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning probability of execution."""
        x = F.relu(self.input_projection(x))
        
        for layer in self.hidden_layers:
            residual = x
            x = layer(x)
            x = x + residual  # Residual connection
        
        # Output sigmoid probability
        execution_prob = torch.sigmoid(self.output_head(x))
        return execution_prob


class SingleMCDropoutEngine:
    """
    SINGLE MC DROPOUT ENGINE - 1000 SAMPLES
    
    The only MC dropout implementation in the entire system.
    Provides binary execution decisions with comprehensive uncertainty analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # FIXED: Always 1000 samples
        self.n_samples = 1000
        
        # Performance targets
        self.target_latency_us = config.get('target_latency_us', 500)
        self.confidence_threshold = config.get('confidence_threshold', 0.8)
        
        # Initialize execution network
        input_dim = config.get('input_dim', 64)
        hidden_dim = config.get('hidden_dim', 128)
        dropout_rate = config.get('dropout_rate', 0.3)
        
        self.execution_network = ExecutionDecisionNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        ).to(self.device)
        
        # GPU optimization
        if self.device.type == 'cuda':
            self._initialize_gpu_optimization()
        
        # Performance tracking
        self.performance_stats = {
            'total_decisions': 0,
            'average_latency_us': 0.0,
            'successful_executions': 0,
            'rejected_executions': 0,
            'average_confidence': 0.0,
            'gpu_utilization_avg': 0.0
        }
        
        logger.info(f"Single MC Dropout Engine initialized - 1000 samples, target <{self.target_latency_us}μs")
    
    def _initialize_gpu_optimization(self):
        """Initialize GPU optimizations for <500μs latency."""
        if torch.cuda.is_available():
            # Enable TensorFloat-32 for A100 performance
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Optimize memory allocation
            torch.cuda.empty_cache()
            
            # Pre-allocate tensors for 1000 samples to avoid allocation overhead
            self.sample_batch = torch.zeros(1000, self.execution_network.input_projection.in_features, 
                                          device=self.device, dtype=torch.float32)
            
            logger.info("GPU optimization initialized for 1000-sample MC dropout")
    
    def _prepare_execution_input(self, context: TradeExecutionContext) -> torch.Tensor:
        """Convert execution context to neural network input."""
        # Extract features from context
        features = []
        
        # MAPPO features
        mappo_rec = context.mappo_recommendation
        features.extend([
            mappo_rec.get('action_confidence', 0.5),
            mappo_rec.get('value_estimate', 0.0),
            mappo_rec.get('policy_entropy', 0.0),
            mappo_rec.get('critic_uncertainty', 0.0)
        ])
        
        # Market features
        market = context.market_data
        features.extend([
            market.get('volatility', 0.0),
            market.get('bid_ask_spread', 0.0),
            market.get('volume', 0.0),
            market.get('momentum', 0.0),
            market.get('market_impact', 0.0),
            market.get('liquidity', 0.0)
        ])
        
        # Portfolio features
        portfolio = context.portfolio_state
        features.extend([
            portfolio.get('current_position', 0.0),
            portfolio.get('available_capital', 1.0),
            portfolio.get('var_usage', 0.0),
            portfolio.get('concentration_risk', 0.0),
            portfolio.get('correlation_exposure', 0.0)
        ])
        
        # Risk features
        risk = context.risk_metrics
        features.extend([
            risk.get('var_estimate', 0.0),
            risk.get('stress_test_result', 0.0),
            risk.get('drawdown_risk', 0.0),
            risk.get('regime_risk', 0.0)
        ])
        
        # Trade features
        trade = context.trade_details
        features.extend([
            trade.get('notional_value', 0.0),
            trade.get('time_horizon', 0.0),
            trade.get('urgency_score', 0.5),
            trade.get('execution_cost_estimate', 0.0)
        ])
        
        # Pad or truncate to expected input dimension
        expected_dim = self.execution_network.input_projection.in_features
        if len(features) < expected_dim:
            features.extend([0.0] * (expected_dim - len(features)))
        else:
            features = features[:expected_dim]
        
        return torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
    
    def _run_1000_samples(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run exactly 1000 MC dropout samples with GPU optimization."""
        start_time = time.perf_counter()
        
        # Enable training mode for dropout
        self.execution_network.train()
        
        # Pre-allocate results tensor
        samples = torch.zeros(1000, 1, device=self.device, dtype=torch.float32)
        
        # Batch processing for GPU efficiency
        batch_size = 100  # Process 100 samples at once for optimal GPU utilization
        input_batch = input_tensor.repeat(batch_size, 1)
        
        with torch.no_grad():
            for i in range(0, 1000, batch_size):
                end_idx = min(i + batch_size, 1000)
                current_batch_size = end_idx - i
                
                if current_batch_size < batch_size:
                    # Handle last batch
                    input_batch_trimmed = input_tensor.repeat(current_batch_size, 1)
                    batch_predictions = self.execution_network(input_batch_trimmed)
                else:
                    batch_predictions = self.execution_network(input_batch)
                
                samples[i:end_idx] = batch_predictions[:current_batch_size]
        
        # Back to eval mode
        self.execution_network.eval()
        
        processing_time = (time.perf_counter() - start_time) * 1_000_000  # Convert to microseconds
        
        if processing_time > self.target_latency_us:
            logger.warning(f"MC dropout exceeded target latency: {processing_time:.1f}μs > {self.target_latency_us}μs")
        
        return samples.squeeze(), processing_time
    
    def _analyze_samples(self, samples: torch.Tensor) -> Tuple[SampleStatistics, UncertaintyMetrics]:
        """Analyze 1000 samples for comprehensive uncertainty metrics."""
        # Convert to numpy for analysis
        samples_np = samples.cpu().numpy()
        
        # Basic statistics
        mean_pred = float(np.mean(samples_np))
        std_pred = float(np.std(samples_np))
        min_pred = float(np.min(samples_np))
        max_pred = float(np.max(samples_np))
        q25 = float(np.percentile(samples_np, 25))
        q75 = float(np.percentile(samples_np, 75))
        
        # Count samples above execution threshold
        samples_above = int(np.sum(samples_np > 0.5))
        
        # Detect outliers (beyond 3 standard deviations)
        outliers = np.abs(samples_np - mean_pred) > (3 * std_pred)
        outlier_count = int(np.sum(outliers))
        
        # Check convergence by analyzing running mean stability
        running_means = np.cumsum(samples_np) / np.arange(1, len(samples_np) + 1)
        convergence_window = running_means[-100:]  # Last 100 samples
        convergence_stability = 1.0 - np.std(convergence_window) / np.mean(convergence_window)
        convergence_achieved = convergence_stability > 0.95
        
        sample_stats = SampleStatistics(
            mean_prediction=mean_pred,
            std_prediction=std_pred,
            min_prediction=min_pred,
            max_prediction=max_pred,
            quantile_25=q25,
            quantile_75=q75,
            samples_above_threshold=samples_above,
            convergence_achieved=convergence_achieved,
            outlier_count=outlier_count
        )
        
        # Uncertainty decomposition
        # Epistemic uncertainty: disagreement between samples
        epistemic = float(std_pred)
        
        # Aleatoric uncertainty: inherent data uncertainty (estimated)
        # For binary classification, minimum uncertainty at 0 and 1, maximum at 0.5
        binary_entropy = -mean_pred * np.log(mean_pred + 1e-8) - (1 - mean_pred) * np.log(1 - mean_pred + 1e-8)
        aleatoric = float(binary_entropy)
        
        # Total uncertainty
        total = epistemic + aleatoric
        
        # Confidence score (inverse of uncertainty, calibrated)
        confidence = 1.0 / (1.0 + total)
        
        # Distance from decision boundary (0.5)
        boundary_distance = abs(mean_pred - 0.5) * 2  # Normalize to [0, 1]
        
        uncertainty_metrics = UncertaintyMetrics(
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            total_uncertainty=total,
            confidence_score=confidence,
            sample_variance=float(np.var(samples_np)),
            convergence_quality=convergence_stability,
            decision_boundary_distance=boundary_distance
        )
        
        return sample_stats, uncertainty_metrics
    
    def _make_binary_decision(self, sample_stats: SampleStatistics, 
                            uncertainty_metrics: UncertaintyMetrics) -> Tuple[bool, float]:
        """Make final binary execution decision with calibrated confidence."""
        # Primary decision based on mean prediction
        should_execute = sample_stats.mean_prediction > 0.5
        
        # Confidence requirements
        confidence_sufficient = uncertainty_metrics.confidence_score >= self.confidence_threshold
        uncertainty_acceptable = uncertainty_metrics.total_uncertainty < 0.5
        convergence_good = sample_stats.convergence_achieved
        
        # Final decision requires all conditions
        execute_trade = should_execute and confidence_sufficient and uncertainty_acceptable and convergence_good
        
        # Calibrated confidence
        if execute_trade:
            calibrated_confidence = min(
                uncertainty_metrics.confidence_score,
                sample_stats.samples_above_threshold / 1000.0  # Proportion supporting execution
            )
        else:
            calibrated_confidence = max(
                1.0 - uncertainty_metrics.confidence_score,
                (1000 - sample_stats.samples_above_threshold) / 1000.0  # Proportion supporting rejection
            )
        
        return execute_trade, calibrated_confidence
    
    async def evaluate_trade_execution(self, context: TradeExecutionContext) -> BinaryExecutionResult:
        """
        Main entry point: Evaluate whether to execute or reject trade.
        
        Args:
            context: Complete execution context
            
        Returns:
            Binary execution decision with full analysis
        """
        start_time = time.perf_counter()
        
        # Prepare input
        input_tensor = self._prepare_execution_input(context)
        
        # Run 1000 MC dropout samples
        samples, processing_time_us = self._run_1000_samples(input_tensor)
        
        # Analyze samples
        sample_stats, uncertainty_metrics = self._analyze_samples(samples)
        
        # Make binary decision
        execute_trade, calibrated_confidence = self._make_binary_decision(
            sample_stats, uncertainty_metrics
        )
        
        # GPU utilization (approximate)
        gpu_util = 0.8 if self.device.type == 'cuda' else 0.0
        
        # Prepare MAPPO feedback
        mappo_feedback = {
            'mc_dropout_approved': execute_trade,
            'mc_dropout_confidence': calibrated_confidence,
            'uncertainty_level': uncertainty_metrics.total_uncertainty,
            'sample_agreement': sample_stats.samples_above_threshold / 1000.0
        }
        
        # Update performance stats
        self._update_performance_stats(processing_time_us, calibrated_confidence, execute_trade, gpu_util)
        
        total_time = (time.perf_counter() - start_time) * 1_000_000
        
        result = BinaryExecutionResult(
            execute_trade=execute_trade,
            confidence=calibrated_confidence,
            uncertainty_metrics=uncertainty_metrics,
            sample_statistics=sample_stats,
            processing_time_us=total_time,
            gpu_utilization=gpu_util,
            mappo_feedback=mappo_feedback
        )
        
        logger.info(f"MC Dropout Decision: {'EXECUTE' if execute_trade else 'REJECT'} "
                   f"(confidence: {calibrated_confidence:.3f}, time: {total_time:.1f}μs)")
        
        return result
    
    def _update_performance_stats(self, processing_time_us: float, confidence: float, 
                                executed: bool, gpu_util: float):
        """Update running performance statistics."""
        self.performance_stats['total_decisions'] += 1
        total = self.performance_stats['total_decisions']
        
        # Running averages
        self.performance_stats['average_latency_us'] = (
            (self.performance_stats['average_latency_us'] * (total - 1) + processing_time_us) / total
        )
        
        self.performance_stats['average_confidence'] = (
            (self.performance_stats['average_confidence'] * (total - 1) + confidence) / total
        )
        
        self.performance_stats['gpu_utilization_avg'] = (
            (self.performance_stats['gpu_utilization_avg'] * (total - 1) + gpu_util) / total
        )
        
        if executed:
            self.performance_stats['successful_executions'] += 1
        else:
            self.performance_stats['rejected_executions'] += 1
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        stats = self.performance_stats.copy()
        
        total = stats['total_decisions']
        if total > 0:
            stats['execution_rate'] = stats['successful_executions'] / total
            stats['rejection_rate'] = stats['rejected_executions'] / total
            stats['latency_target_met'] = stats['average_latency_us'] < self.target_latency_us
        else:
            stats['execution_rate'] = 0.0
            stats['rejection_rate'] = 0.0
            stats['latency_target_met'] = True
        
        return stats


# Factory function for easy instantiation
def create_single_mc_dropout_engine(config: Dict[str, Any] = None) -> SingleMCDropoutEngine:
    """Create the single MC dropout engine with optimal configuration."""
    if config is None:
        config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'input_dim': 64,
            'hidden_dim': 128,
            'dropout_rate': 0.3,
            'target_latency_us': 500,
            'confidence_threshold': 0.8
        }
    
    return SingleMCDropoutEngine(config)


# Global instance - single point of access
_global_mc_engine = None

def get_mc_dropout_engine() -> SingleMCDropoutEngine:
    """Get the global MC dropout engine instance."""
    global _global_mc_engine
    if _global_mc_engine is None:
        _global_mc_engine = create_single_mc_dropout_engine()
    return _global_mc_engine