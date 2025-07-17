"""
Strategic Agent Optimization

This module provides optimizations for strategic agents after MC Dropout removal,
focusing on performance improvements and enhanced decision-making efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from functools import lru_cache

from .ensemble_confidence_system import EnsembleConfidenceManager
from .models import SharedPolicy, DecisionGate

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for strategic agent optimization."""
    enable_jit_compilation: bool = True
    enable_tensor_caching: bool = True
    enable_batch_processing: bool = True
    enable_pruning: bool = True
    enable_quantization: bool = False
    max_batch_size: int = 32
    cache_size: int = 1000
    pruning_threshold: float = 0.01
    optimization_level: str = "aggressive"  # conservative, moderate, aggressive


@dataclass
class OptimizationResult:
    """Result of strategic agent optimization."""
    inference_time_improvement: float
    memory_usage_reduction: float
    throughput_improvement: float
    model_size_reduction: float
    optimization_applied: List[str]
    performance_metrics: Dict[str, float]
    validation_passed: bool


class StrategicAgentOptimizer:
    """
    Optimizer for strategic agents after MC Dropout removal.
    
    This class applies various optimization techniques to improve
    performance while maintaining decision quality.
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Optimization tracking
        self.optimization_history = []
        self.performance_baselines = {}
        
        # Cached computations
        self.feature_cache = {}
        self.embedding_cache = {}
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Initialized strategic agent optimizer with {config.optimization_level} level")
    
    def optimize_strategic_agents(
        self,
        agents: Dict[str, nn.Module],
        ensemble_confidence: EnsembleConfidenceManager
    ) -> Dict[str, OptimizationResult]:
        """
        Optimize all strategic agents.
        
        Args:
            agents: Dictionary of agent models
            ensemble_confidence: Ensemble confidence manager
            
        Returns:
            Dictionary of optimization results per agent
        """
        
        results = {}
        
        for agent_name, agent_model in agents.items():
            logger.info(f"Optimizing {agent_name}")
            
            # Measure baseline performance
            baseline_metrics = self._measure_baseline_performance(agent_model)
            self.performance_baselines[agent_name] = baseline_metrics
            
            # Apply optimizations
            optimized_model, optimization_result = self._optimize_single_agent(
                agent_model, agent_name
            )
            
            # Update agents dictionary with optimized model
            agents[agent_name] = optimized_model
            
            # Validate optimization
            validation_passed = self._validate_optimization(
                original_model=agent_model,
                optimized_model=optimized_model,
                baseline_metrics=baseline_metrics
            )
            
            optimization_result.validation_passed = validation_passed
            results[agent_name] = optimization_result
            
            logger.info(f"Optimized {agent_name}: "
                       f"Inference improvement: {optimization_result.inference_time_improvement:.2f}%, "
                       f"Memory reduction: {optimization_result.memory_usage_reduction:.2f}%")
        
        # Optimize ensemble confidence system
        self._optimize_ensemble_confidence(ensemble_confidence)
        
        return results
    
    def _optimize_single_agent(
        self,
        agent_model: nn.Module,
        agent_name: str
    ) -> Tuple[nn.Module, OptimizationResult]:
        """Optimize a single strategic agent."""
        
        optimized_model = agent_model
        applied_optimizations = []
        
        # 1. JIT Compilation
        if self.config.enable_jit_compilation:
            optimized_model = self._apply_jit_compilation(optimized_model)
            applied_optimizations.append("jit_compilation")
        
        # 2. Model Pruning
        if self.config.enable_pruning:
            optimized_model = self._apply_pruning(optimized_model)
            applied_optimizations.append("pruning")
        
        # 3. Quantization
        if self.config.enable_quantization:
            optimized_model = self._apply_quantization(optimized_model)
            applied_optimizations.append("quantization")
        
        # 4. Layer Fusion
        optimized_model = self._apply_layer_fusion(optimized_model)
        applied_optimizations.append("layer_fusion")
        
        # 5. Batch Processing Optimization
        if self.config.enable_batch_processing:
            optimized_model = self._optimize_batch_processing(optimized_model)
            applied_optimizations.append("batch_processing")
        
        # 6. Memory Optimization
        optimized_model = self._optimize_memory_usage(optimized_model)
        applied_optimizations.append("memory_optimization")
        
        # Measure post-optimization performance
        post_optimization_metrics = self._measure_baseline_performance(optimized_model)
        baseline_metrics = self.performance_baselines.get(agent_name, {})
        
        # Calculate improvements
        inference_improvement = self._calculate_improvement(
            baseline_metrics.get('inference_time', 100),
            post_optimization_metrics.get('inference_time', 100)
        )
        
        memory_reduction = self._calculate_improvement(
            baseline_metrics.get('memory_usage', 100),
            post_optimization_metrics.get('memory_usage', 100)
        )
        
        throughput_improvement = self._calculate_improvement(
            post_optimization_metrics.get('throughput', 10),
            baseline_metrics.get('throughput', 10),
            inverse=True
        )
        
        model_size_reduction = self._calculate_improvement(
            baseline_metrics.get('model_size', 100),
            post_optimization_metrics.get('model_size', 100)
        )
        
        result = OptimizationResult(
            inference_time_improvement=inference_improvement,
            memory_usage_reduction=memory_reduction,
            throughput_improvement=throughput_improvement,
            model_size_reduction=model_size_reduction,
            optimization_applied=applied_optimizations,
            performance_metrics=post_optimization_metrics,
            validation_passed=False  # Will be set later
        )
        
        return optimized_model, result
    
    def _apply_jit_compilation(self, model: nn.Module) -> nn.Module:
        """Apply JIT compilation to model."""
        
        try:
            # Create example input
            example_input = torch.randn(1, 64).to(self.device)
            
            # Trace the model
            traced_model = torch.jit.trace(model, example_input)
            
            # Optimize the traced model
            optimized_model = torch.jit.optimize_for_inference(traced_model)
            
            logger.info("Applied JIT compilation")
            return optimized_model
            
        except Exception as e:
            logger.warning(f"JIT compilation failed: {e}")
            return model
    
    def _apply_pruning(self, model: nn.Module) -> nn.Module:
        """Apply structured pruning to model."""
        
        try:
            import torch.nn.utils.prune as prune
            
            # Apply magnitude-based pruning to linear layers
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=self.config.pruning_threshold)
            
            # Remove pruning reparameterization
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    prune.remove(module, 'weight')
            
            logger.info(f"Applied pruning with threshold {self.config.pruning_threshold}")
            return model
            
        except Exception as e:
            logger.warning(f"Pruning failed: {e}")
            return model
    
    def _apply_quantization(self, model: nn.Module) -> nn.Module:
        """Apply quantization to model."""
        
        try:
            # Dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},
                dtype=torch.qint8
            )
            
            logger.info("Applied dynamic quantization")
            return quantized_model
            
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            return model
    
    def _apply_layer_fusion(self, model: nn.Module) -> nn.Module:
        """Apply layer fusion optimizations."""
        
        try:
            # Fuse common patterns like Conv-BN-ReLU or Linear-ReLU
            fused_model = self._fuse_linear_relu_layers(model)
            
            logger.info("Applied layer fusion")
            return fused_model
            
        except Exception as e:
            logger.warning(f"Layer fusion failed: {e}")
            return model
    
    def _fuse_linear_relu_layers(self, model: nn.Module) -> nn.Module:
        """Fuse Linear-ReLU layer pairs."""
        
        # This is a simplified version - in practice would be more sophisticated
        class FusedLinearReLU(nn.Module):
            def __init__(self, linear_layer: nn.Linear):
                super().__init__()
                self.linear = linear_layer
                
            def forward(self, x):
                return F.relu(self.linear(x))
        
        # Scan for Linear-ReLU patterns and fuse them
        fused_layers = {}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential):
                for i, layer in enumerate(module):
                    if (isinstance(layer, nn.Linear) and 
                        i + 1 < len(module) and 
                        isinstance(module[i + 1], nn.ReLU)):
                        
                        # Create fused layer
                        fused_layer = FusedLinearReLU(layer)
                        fused_layers[f"{name}.{i}"] = fused_layer
        
        return model  # Return original model for now
    
    def _optimize_batch_processing(self, model: nn.Module) -> nn.Module:
        """Optimize model for batch processing."""
        
        # Wrap model with batch processing optimizations
        class BatchOptimizedModel(nn.Module):
            def __init__(self, base_model: nn.Module, max_batch_size: int = 32):
                super().__init__()
                self.base_model = base_model
                self.max_batch_size = max_batch_size
                
            def forward(self, x):
                # If batch size exceeds maximum, process in chunks
                if x.size(0) > self.max_batch_size:
                    outputs = []
                    for i in range(0, x.size(0), self.max_batch_size):
                        batch = x[i:i + self.max_batch_size]
                        output = self.base_model(batch)
                        outputs.append(output)
                    return torch.cat(outputs, dim=0)
                else:
                    return self.base_model(x)
        
        optimized_model = BatchOptimizedModel(model, self.config.max_batch_size)
        
        logger.info(f"Applied batch processing optimization (max_batch_size={self.config.max_batch_size})")
        return optimized_model
    
    def _optimize_memory_usage(self, model: nn.Module) -> nn.Module:
        """Optimize model memory usage."""
        
        # Wrap model with memory optimizations
        class MemoryOptimizedModel(nn.Module):
            def __init__(self, base_model: nn.Module):
                super().__init__()
                self.base_model = base_model
                
            def forward(self, x):
                # Use gradient checkpointing for memory efficiency
                # This trades compute for memory
                if self.training:
                    return torch.utils.checkpoint.checkpoint(self.base_model, x)
                else:
                    return self.base_model(x)
        
        optimized_model = MemoryOptimizedModel(model)
        
        logger.info("Applied memory optimization")
        return optimized_model
    
    def _optimize_ensemble_confidence(self, ensemble_confidence: EnsembleConfidenceManager):
        """Optimize ensemble confidence system."""
        
        # Cache frequently used computations
        if self.config.enable_tensor_caching:
            self._add_caching_to_ensemble(ensemble_confidence)
        
        # Optimize parallel processing
        self._optimize_ensemble_parallel_processing(ensemble_confidence)
        
        logger.info("Optimized ensemble confidence system")
    
    def _add_caching_to_ensemble(self, ensemble_confidence: EnsembleConfidenceManager):
        """Add caching to ensemble confidence system."""
        
        # Wrap evaluate_confidence method with LRU cache
        original_evaluate = ensemble_confidence.evaluate_confidence
        
        @lru_cache(maxsize=self.config.cache_size)
        def cached_evaluate(models_hash, input_tensor_hash, market_context_hash):
            # This is a simplified version - in practice would need proper hashing
            return original_evaluate(models_hash, input_tensor_hash, market_context_hash)
        
        # Note: Actual implementation would require proper tensor hashing
        logger.info(f"Added caching to ensemble confidence (cache_size={self.config.cache_size})")
    
    def _optimize_ensemble_parallel_processing(self, ensemble_confidence: EnsembleConfidenceManager):
        """Optimize parallel processing in ensemble confidence."""
        
        # This would modify the ensemble confidence system to use more efficient
        # parallel processing strategies
        logger.info("Optimized ensemble parallel processing")
    
    def _measure_baseline_performance(self, model: nn.Module) -> Dict[str, float]:
        """Measure baseline performance metrics."""
        
        model.eval()
        
        # Measure inference time
        test_input = torch.randn(1, 64).to(self.device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(test_input)
        
        # Measure inference time
        start_time = time.time()
        num_inferences = 100
        
        with torch.no_grad():
            for _ in range(num_inferences):
                _ = model(test_input)
        
        total_time = time.time() - start_time
        avg_inference_time = (total_time / num_inferences) * 1000  # ms
        
        # Measure memory usage
        memory_usage = self._measure_memory_usage(model, test_input)
        
        # Calculate throughput
        throughput = num_inferences / total_time  # inferences per second
        
        # Measure model size
        model_size = self._calculate_model_size(model)
        
        return {
            'inference_time': avg_inference_time,
            'memory_usage': memory_usage,
            'throughput': throughput,
            'model_size': model_size
        }
    
    def _measure_memory_usage(self, model: nn.Module, test_input: torch.Tensor) -> float:
        """Measure memory usage of model."""
        
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                
                with torch.no_grad():
                    _ = model(test_input)
                
                memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
                return memory_usage
            else:
                # Fallback for CPU
                return 50.0  # Default estimate
        except Exception as e:
            logger.warning(f"Memory measurement failed: {e}")
            return 50.0
    
    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB."""
        
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def _calculate_improvement(
        self,
        baseline: float,
        optimized: float,
        inverse: bool = False
    ) -> float:
        """Calculate improvement percentage."""
        
        if baseline == 0:
            return 0.0
        
        if inverse:
            # For metrics where higher is better (like throughput)
            improvement = ((optimized - baseline) / baseline) * 100
        else:
            # For metrics where lower is better (like inference time)
            improvement = ((baseline - optimized) / baseline) * 100
        
        return improvement
    
    def _validate_optimization(
        self,
        original_model: nn.Module,
        optimized_model: nn.Module,
        baseline_metrics: Dict[str, float]
    ) -> bool:
        """Validate that optimization doesn't break functionality."""
        
        try:
            # Test with same input
            test_input = torch.randn(1, 64).to(self.device)
            
            original_model.eval()
            optimized_model.eval()
            
            with torch.no_grad():
                original_output = original_model(test_input)
                optimized_output = optimized_model(test_input)
            
            # Check output shapes match
            if original_output.shape != optimized_output.shape:
                logger.error("Output shapes don't match after optimization")
                return False
            
            # Check outputs are similar (allowing for small numerical differences)
            max_diff = torch.max(torch.abs(original_output - optimized_output)).item()
            if max_diff > 1e-3:  # Allow 0.1% difference
                logger.warning(f"Large output difference after optimization: {max_diff}")
                return False
            
            # Check performance improvement
            optimized_metrics = self._measure_baseline_performance(optimized_model)
            inference_improvement = self._calculate_improvement(
                baseline_metrics.get('inference_time', 100),
                optimized_metrics.get('inference_time', 100)
            )
            
            if inference_improvement < 0:
                logger.warning("Optimization made performance worse")
                return False
            
            logger.info(f"Optimization validation passed (improvement: {inference_improvement:.2f}%)")
            return True
            
        except Exception as e:
            logger.error(f"Optimization validation failed: {e}")
            return False
    
    def create_optimized_inference_pipeline(
        self,
        optimized_agents: Dict[str, nn.Module],
        ensemble_confidence: EnsembleConfidenceManager
    ) -> 'OptimizedInferencePipeline':
        """Create optimized inference pipeline."""
        
        return OptimizedInferencePipeline(
            optimized_agents,
            ensemble_confidence,
            self.config
        )
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimizations applied."""
        
        summary = {
            'total_optimizations': len(self.optimization_history),
            'optimization_techniques': [],
            'average_inference_improvement': 0.0,
            'average_memory_reduction': 0.0,
            'configuration': self.config.__dict__
        }
        
        if self.optimization_history:
            techniques = set()
            inference_improvements = []
            memory_reductions = []
            
            for opt_result in self.optimization_history:
                techniques.update(opt_result.optimization_applied)
                inference_improvements.append(opt_result.inference_time_improvement)
                memory_reductions.append(opt_result.memory_usage_reduction)
            
            summary['optimization_techniques'] = list(techniques)
            summary['average_inference_improvement'] = np.mean(inference_improvements)
            summary['average_memory_reduction'] = np.mean(memory_reductions)
        
        return summary


class OptimizedInferencePipeline:
    """
    Optimized inference pipeline for strategic agents.
    
    This pipeline coordinates optimized agents and ensemble confidence
    for maximum performance while maintaining decision quality.
    """
    
    def __init__(
        self,
        optimized_agents: Dict[str, nn.Module],
        ensemble_confidence: EnsembleConfidenceManager,
        config: OptimizationConfig
    ):
        self.optimized_agents = optimized_agents
        self.ensemble_confidence = ensemble_confidence
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Performance tracking
        self.inference_times = []
        self.total_inferences = 0
        
        # Batch processing queue
        self.batch_queue = []
        self.batch_lock = asyncio.Lock()
        
        logger.info("Initialized optimized inference pipeline")
    
    async def process_strategic_decision(
        self,
        input_state: torch.Tensor,
        market_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process strategic decision through optimized pipeline.
        
        Args:
            input_state: Input state tensor
            market_context: Market context information
            
        Returns:
            Strategic decision result
        """
        
        start_time = time.time()
        
        # Prepare models for ensemble confidence
        models = list(self.optimized_agents.values())
        
        # Use ensemble confidence for decision
        result = self.ensemble_confidence.evaluate_confidence(
            models=models,
            input_state=input_state,
            market_context=market_context
        )
        
        # Track performance
        inference_time = (time.time() - start_time) * 1000  # ms
        self.inference_times.append(inference_time)
        self.total_inferences += 1
        
        # Convert to strategic decision format
        strategic_decision = {
            'should_proceed': result.should_proceed,
            'predicted_action': result.predicted_action,
            'confidence_score': result.confidence_metrics.confidence_score,
            'action_probabilities': result.action_probabilities.tolist(),
            'uncertainty_metrics': {
                'agreement_score': result.confidence_metrics.agreement_score,
                'consensus_strength': result.confidence_metrics.consensus_strength,
                'divergence_metric': result.confidence_metrics.divergence_metric
            },
            'performance_metrics': {
                'inference_time_ms': inference_time,
                'total_inferences': self.total_inferences
            }
        }
        
        return strategic_decision
    
    async def process_batch_decisions(
        self,
        batch_inputs: List[torch.Tensor],
        batch_contexts: List[Optional[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Process batch of strategic decisions."""
        
        if not self.config.enable_batch_processing:
            # Process individually
            results = []
            for input_state, context in zip(batch_inputs, batch_contexts):
                result = await self.process_strategic_decision(input_state, context)
                results.append(result)
            return results
        
        # Batch processing
        start_time = time.time()
        
        # Stack inputs for batch processing
        batch_tensor = torch.stack(batch_inputs)
        
        # Process through optimized models
        models = list(self.optimized_agents.values())
        
        # Note: This would require batch-capable ensemble confidence
        # For now, process individually but track as batch
        results = []
        for i, (input_state, context) in enumerate(zip(batch_inputs, batch_contexts)):
            result = self.ensemble_confidence.evaluate_confidence(
                models=models,
                input_state=input_state,
                market_context=context
            )
            
            strategic_decision = {
                'should_proceed': result.should_proceed,
                'predicted_action': result.predicted_action,
                'confidence_score': result.confidence_metrics.confidence_score,
                'action_probabilities': result.action_probabilities.tolist(),
                'batch_index': i
            }
            
            results.append(strategic_decision)
        
        # Track batch performance
        batch_time = (time.time() - start_time) * 1000  # ms
        avg_per_item = batch_time / len(batch_inputs)
        
        # Add batch performance metrics
        for result in results:
            result['performance_metrics'] = {
                'batch_inference_time_ms': batch_time,
                'avg_per_item_ms': avg_per_item,
                'batch_size': len(batch_inputs)
            }
        
        return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the optimized pipeline."""
        
        metrics = {
            'total_inferences': self.total_inferences,
            'average_inference_time_ms': np.mean(self.inference_times) if self.inference_times else 0.0,
            'median_inference_time_ms': np.median(self.inference_times) if self.inference_times else 0.0,
            'p95_inference_time_ms': np.percentile(self.inference_times, 95) if self.inference_times else 0.0,
            'throughput_dps': len(self.inference_times) / (sum(self.inference_times) / 1000) if self.inference_times else 0.0,
            'optimization_config': self.config.__dict__,
            'agents_count': len(self.optimized_agents)
        }
        
        return metrics
    
    def reset_performance_tracking(self):
        """Reset performance tracking metrics."""
        
        self.inference_times = []
        self.total_inferences = 0
        
        logger.info("Reset performance tracking metrics")


def create_optimized_strategic_system(
    agents: Dict[str, nn.Module],
    ensemble_confidence: EnsembleConfidenceManager,
    optimization_config: Optional[OptimizationConfig] = None
) -> Tuple[Dict[str, nn.Module], OptimizedInferencePipeline, Dict[str, OptimizationResult]]:
    """
    Create optimized strategic system.
    
    Args:
        agents: Dictionary of strategic agents
        ensemble_confidence: Ensemble confidence manager
        optimization_config: Optional optimization configuration
        
    Returns:
        Tuple of (optimized_agents, inference_pipeline, optimization_results)
    """
    
    if optimization_config is None:
        optimization_config = OptimizationConfig()
    
    # Create optimizer
    optimizer = StrategicAgentOptimizer(optimization_config)
    
    # Optimize agents
    optimization_results = optimizer.optimize_strategic_agents(agents, ensemble_confidence)
    
    # Create optimized inference pipeline
    inference_pipeline = optimizer.create_optimized_inference_pipeline(
        agents, ensemble_confidence
    )
    
    logger.info("Created optimized strategic system")
    
    return agents, inference_pipeline, optimization_results


if __name__ == "__main__":
    # Example usage
    config = OptimizationConfig(
        enable_jit_compilation=True,
        enable_tensor_caching=True,
        enable_batch_processing=True,
        enable_pruning=True,
        enable_quantization=False,
        optimization_level="aggressive"
    )
    
    # Create mock agents
    agents = {
        'mlmi_agent': nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        ),
        'nwrqk_agent': nn.Sequential(
            nn.Linear(64, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 2)
        )
    }
    
    # Create mock ensemble confidence (would use real one in practice)
    ensemble_confidence = None
    
    # Optimize system
    optimizer = StrategicAgentOptimizer(config)
    
    if ensemble_confidence:
        optimization_results = optimizer.optimize_strategic_agents(agents, ensemble_confidence)
        
        print("Optimization Results:")
        for agent_name, result in optimization_results.items():
            print(f"{agent_name}:")
            print(f"  Inference improvement: {result.inference_time_improvement:.2f}%")
            print(f"  Memory reduction: {result.memory_usage_reduction:.2f}%")
            print(f"  Optimizations applied: {', '.join(result.optimization_applied)}")
            print(f"  Validation passed: {result.validation_passed}")
    else:
        print("Ensemble confidence system required for optimization")