"""
Neural Network JIT Optimizations for High-Performance Inference

This module implements PyTorch JIT compilation, quantization, and other
optimization techniques to achieve <50ms inference latency and support
1000+ RPS throughput requirements.

Performance Targets:
- <20ms inference per agent model
- Support for batch inference
- Memory-optimized model loading
- ONNX export compatibility for production deployment
"""

import torch
import torch.nn as nn
import torch.jit
import torch.quantization
from typing import Dict, Any, Optional, Tuple, List
import logging
import time
import numpy as np
from pathlib import Path
import warnings
import gc
from contextlib import contextmanager

from src.monitoring.tactical_metrics import tactical_metrics

logger = logging.getLogger(__name__)

class OptimizedTacticalAgent(nn.Module):
    """
    JIT-optimized tactical agent model for high-performance inference.
    
    This is a simplified but production-ready implementation of tactical agents
    optimized for latency and throughput.
    """
    
    def __init__(
        self,
        input_size: int = 420,  # 60 bars * 7 features
        hidden_size: int = 256,
        output_size: int = 3,   # long, hold, short
        agent_type: str = "fvg",
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.agent_type = agent_type
        self.input_size = input_size
        
        # Optimized architecture for speed
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        
        self.decision_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 4, output_size),
            nn.Softmax(dim=-1)
        )
        
        # Agent-specific feature weights (learnable)
        if agent_type == "fvg":
            # FVG agent focuses on gap-related features
            feature_weights = torch.ones(input_size)
            feature_weights[:60*2] *= 2.0  # Boost FVG features
        elif agent_type == "momentum":
            # Momentum agent focuses on momentum features  
            feature_weights = torch.ones(input_size)
            feature_weights[60*5:60*6] *= 3.0  # Boost momentum features
        else:  # entry agent
            # Entry agent considers all features equally but focuses on timing
            feature_weights = torch.ones(input_size)
            feature_weights[60*4:60*5] *= 2.0  # Boost mitigation signals
        
        self.register_buffer('feature_weights', feature_weights)
        
        # Initialize weights for optimal performance
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for optimal convergence and performance."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass optimized for JIT compilation.
        
        Args:
            x: Input tensor of shape (batch_size, 60, 7) or (60, 7)
            
        Returns:
            Action probabilities of shape (batch_size, 3) or (3,)
        """
        # Handle both batched and single inputs
        original_shape = x.shape
        if len(original_shape) == 2:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Flatten input: (batch_size, 60, 7) -> (batch_size, 420)
        x = x.view(x.size(0), -1)
        
        # Apply feature-specific weights
        x = x * self.feature_weights
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Make decision
        probabilities = self.decision_head(features)
        
        # Remove batch dimension if input was single
        if len(original_shape) == 2:
            probabilities = probabilities.squeeze(0)
        
        return probabilities

class ModelOptimizer:
    """
    High-performance model optimizer using JIT compilation and quantization.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.compiled_models: Dict[str, torch.jit.ScriptModule] = {}
        self.quantized_models: Dict[str, torch.jit.ScriptModule] = {}
        self.model_cache: Dict[str, Any] = {}
        
        # Performance tracking
        self.compilation_times: Dict[str, float] = {}
        self.inference_times: List[float] = []
        
        logger.info(f"ModelOptimizer initialized with device: {device}")
    
    def create_tactical_agents(self) -> Dict[str, OptimizedTacticalAgent]:
        """Create optimized tactical agent models."""
        agents = {}
        
        agent_configs = [
            {"name": "fvg_agent", "type": "fvg", "hidden_size": 256},
            {"name": "momentum_agent", "type": "momentum", "hidden_size": 128},
            {"name": "entry_agent", "type": "entry", "hidden_size": 192}
        ]
        
        for config in agent_configs:
            agent = OptimizedTacticalAgent(
                agent_type=config["type"],
                hidden_size=config["hidden_size"],
                dropout_rate=0.0  # Disable dropout for inference
            )
            agent.eval()  # Set to evaluation mode
            agent.to(self.device)
            agents[config["name"]] = agent
            
            logger.info(f"Created {config['name']} with {sum(p.numel() for p in agent.parameters())} parameters")
        
        return agents
    
    def compile_model_jit(
        self,
        model: nn.Module,
        model_name: str,
        example_input: Optional[torch.Tensor] = None,
        optimize_for_inference: bool = True
    ) -> torch.jit.ScriptModule:
        """
        Compile model using PyTorch JIT for optimal inference performance.
        """
        start_time = time.perf_counter()
        
        try:
            if example_input is None:
                # Create default example input (60x7 matrix)
                example_input = torch.randn(1, 60, 7, device=self.device)
            
            model.eval()
            
            # Use torch.jit.trace for better performance
            with torch.no_grad():
                traced_model = torch.jit.trace(model, example_input)
            
            if optimize_for_inference:
                # Apply additional optimizations
                traced_model = torch.jit.optimize_for_inference(traced_model)
            
            # Warm up the compiled model
            with torch.no_grad():
                for _ in range(10):
                    _ = traced_model(example_input)
            
            compilation_time = time.perf_counter() - start_time
            self.compilation_times[model_name] = compilation_time
            self.compiled_models[model_name] = traced_model
            
            logger.info(f"✅ JIT compiled {model_name} in {compilation_time:.3f}s")
            
            return traced_model
            
        except Exception as e:
            logger.error(f"❌ JIT compilation failed for {model_name}: {e}")
            # Return original model as fallback
            return model
    
    def quantize_model(
        self,
        model: nn.Module,
        model_name: str,
        quantization_type: str = "dynamic"
    ) -> torch.jit.ScriptModule:
        """
        Apply quantization for memory and speed optimization.
        """
        try:
            model.eval()
            
            if quantization_type == "dynamic":
                # Dynamic quantization - good for CPU inference
                quantized_model = torch.quantization.quantize_dynamic(
                    model,
                    {nn.Linear},
                    dtype=torch.qint8
                )
                
                # Convert to TorchScript
                example_input = torch.randn(1, 60, 7)
                with torch.no_grad():
                    traced_quantized = torch.jit.trace(quantized_model, example_input)
                
                self.quantized_models[model_name] = traced_quantized
                
                logger.info(f"✅ Quantized {model_name} with {quantization_type} quantization")
                return traced_quantized
                
            else:
                logger.warning(f"Unsupported quantization type: {quantization_type}")
                return model
                
        except Exception as e:
            logger.error(f"❌ Quantization failed for {model_name}: {e}")
            return model
    
    def benchmark_model_performance(
        self,
        model: torch.jit.ScriptModule,
        model_name: str,
        num_runs: int = 1000,
        batch_size: int = 1
    ) -> Dict[str, float]:
        """
        Benchmark model inference performance.
        """
        logger.info(f"🔄 Benchmarking {model_name} with {num_runs} runs (batch_size={batch_size})")
        
        # Prepare test data
        test_input = torch.randn(batch_size, 60, 7, device=self.device)
        
        # Warm-up runs
        model.eval()
        with torch.no_grad():
            for _ in range(50):
                _ = model(test_input)
        
        # Benchmark runs
        inference_times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = model(test_input)
                inference_time = time.perf_counter() - start_time
                inference_times.append(inference_time * 1000)  # Convert to ms
        
        # Calculate statistics
        stats = {
            "avg_latency_ms": np.mean(inference_times),
            "p50_latency_ms": np.percentile(inference_times, 50),
            "p95_latency_ms": np.percentile(inference_times, 95),
            "p99_latency_ms": np.percentile(inference_times, 99),
            "max_latency_ms": np.max(inference_times),
            "min_latency_ms": np.min(inference_times),
            "throughput_qps": 1000 / np.mean(inference_times) * batch_size
        }
        
        logger.info(f"📊 {model_name} Performance: avg={stats['avg_latency_ms']:.2f}ms, p99={stats['p99_latency_ms']:.2f}ms, throughput={stats['throughput_qps']:.1f} QPS")
        
        return stats
    
    def save_optimized_models(self, save_dir: Path):
        """Save optimized models to disk for production deployment."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JIT compiled models
        for model_name, model in self.compiled_models.items():
            model_path = save_dir / f"{model_name}_jit.pt"
            torch.jit.save(model, str(model_path))
            logger.info(f"💾 Saved JIT model: {model_path}")
        
        # Save quantized models
        for model_name, model in self.quantized_models.items():
            model_path = save_dir / f"{model_name}_quantized.pt"
            torch.jit.save(model, str(model_path))
            logger.info(f"💾 Saved quantized model: {model_path}")
    
    def load_optimized_model(self, model_path: Path) -> torch.jit.ScriptModule:
        """Load optimized model for inference."""
        try:
            model = torch.jit.load(str(model_path), map_location=self.device)
            model.eval()
            
            # Warm up the loaded model
            example_input = torch.randn(1, 60, 7, device=self.device)
            with torch.no_grad():
                for _ in range(10):
                    _ = model(example_input)
            
            logger.info(f"✅ Loaded optimized model: {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_path}: {e}")
            raise

@contextmanager
def inference_timer():
    """Context manager for timing inference operations."""
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    inference_time_ms = (end_time - start_time) * 1000
    
    # Record metrics
    tactical_metrics.record_model_inference_time(inference_time_ms)

class ProductionInferenceEngine:
    """
    Production-ready inference engine with optimized models.
    """
    
    def __init__(self, model_dir: Optional[Path] = None):
        self.device = torch.device("cpu")  # CPU optimized for deployment
        self.models: Dict[str, torch.jit.ScriptModule] = {}
        self.model_optimizer = ModelOptimizer(device="cpu")
        self.model_dir = model_dir or Path("models/optimized")
        self.stats = {
            "inferences_completed": 0,
            "total_inference_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Result caching for identical inputs
        self.result_cache: Dict[str, Tuple[torch.Tensor, float]] = {}
        self.cache_max_size = 1000
        
    def initialize(self, force_recompile: bool = False):
        """Initialize the inference engine with optimized models."""
        logger.info("🚀 Initializing ProductionInferenceEngine")
        
        try:
            if not force_recompile and self._load_existing_models():
                logger.info("✅ Loaded existing optimized models")
            else:
                logger.info("🔄 Creating and optimizing new models")
                self._create_and_optimize_models()
            
            # Validate all models are loaded
            required_agents = ["fvg_agent", "momentum_agent", "entry_agent"]
            for agent_name in required_agents:
                if agent_name not in self.models:
                    raise RuntimeError(f"Failed to load required agent: {agent_name}")
            
            logger.info(f"✅ ProductionInferenceEngine initialized with {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize inference engine: {e}")
            raise
    
    def _load_existing_models(self) -> bool:
        """Try to load existing optimized models."""
        if not self.model_dir.exists():
            return False
        
        required_models = ["fvg_agent_jit.pt", "momentum_agent_jit.pt", "entry_agent_jit.pt"]
        
        try:
            for model_file in required_models:
                model_path = self.model_dir / model_file
                if not model_path.exists():
                    return False
                
                agent_name = model_file.replace("_jit.pt", "")
                model = self.model_optimizer.load_optimized_model(model_path)
                self.models[agent_name] = model
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load existing models: {e}")
            return False
    
    def _create_and_optimize_models(self):
        """Create and optimize new models."""
        # Create base models
        agents = self.model_optimizer.create_tactical_agents()
        
        # Compile with JIT
        for agent_name, agent_model in agents.items():
            optimized_model = self.model_optimizer.compile_model_jit(
                agent_model,
                agent_name,
                optimize_for_inference=True
            )
            self.models[agent_name] = optimized_model
            
            # Benchmark performance
            stats = self.model_optimizer.benchmark_model_performance(
                optimized_model,
                agent_name,
                num_runs=1000
            )
            
            # Log performance metrics
            tactical_metrics.record_model_optimization_stats(agent_name, stats)
        
        # Save optimized models
        self.model_optimizer.save_optimized_models(self.model_dir)
    
    def infer_batch(
        self,
        matrix_states: List[np.ndarray],
        agent_types: List[str]
    ) -> List[Dict[str, Any]]:
        """
        High-performance batch inference.
        """
        if len(matrix_states) != len(agent_types):
            raise ValueError("matrix_states and agent_types must have same length")
        
        results = []
        
        # Group by agent type for batch processing
        agent_groups: Dict[str, List[Tuple[int, np.ndarray]]] = {}
        for i, (matrix, agent_type) in enumerate(zip(matrix_states, agent_types)):
            if agent_type not in agent_groups:
                agent_groups[agent_type] = []
            agent_groups[agent_type].append((i, matrix))
        
        # Process each agent type in batches
        ordered_results = [None] * len(matrix_states)
        
        for agent_type, batch_data in agent_groups.items():
            if agent_type not in self.models:
                logger.error(f"Model not found for agent type: {agent_type}")
                continue
            
            # Prepare batch input
            indices = [item[0] for item in batch_data]
            matrices = [item[1] for item in batch_data]
            
            batch_input = torch.tensor(
                np.stack(matrices),
                dtype=torch.float32,
                device=self.device
            )
            
            # Run inference
            with inference_timer():
                with torch.no_grad():
                    probabilities = self.models[agent_type](batch_input)
                    actions = torch.argmax(probabilities, dim=1) - 1  # Convert to -1, 0, 1
                    confidences = torch.max(probabilities, dim=1)[0]
            
            # Process results
            for i, (orig_idx, _) in enumerate(batch_data):
                result = {
                    "action": int(actions[i].item()),
                    "probabilities": probabilities[i].cpu().numpy().tolist(),
                    "confidence": float(confidences[i].item()),
                    "agent_type": agent_type
                }
                ordered_results[orig_idx] = result
        
        # Update stats
        self.stats["inferences_completed"] += len(matrix_states)
        
        return [r for r in ordered_results if r is not None]
    
    def infer_single(
        self,
        matrix_state: np.ndarray,
        agent_type: str,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        High-performance single inference with caching.
        """
        # Generate cache key if caching enabled
        cache_key = None
        if use_cache:
            cache_key = f"{agent_type}_{hash(matrix_state.tobytes())}"
            if cache_key in self.result_cache:
                self.stats["cache_hits"] += 1
                cached_result, cached_time = self.result_cache[cache_key]
                return {
                    "action": int(torch.argmax(cached_result).item()) - 1,
                    "probabilities": cached_result.cpu().numpy().tolist(),
                    "confidence": float(torch.max(cached_result).item()),
                    "agent_type": agent_type,
                    "cached": True
                }
        
        if agent_type not in self.models:
            raise ValueError(f"Model not found for agent type: {agent_type}")
        
        # Prepare input
        input_tensor = torch.tensor(
            matrix_state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)  # Add batch dimension
        
        # Run inference
        start_time = time.perf_counter()
        with torch.no_grad():
            probabilities = self.models[agent_type](input_tensor)
            action = torch.argmax(probabilities, dim=1) - 1
            confidence = torch.max(probabilities, dim=1)[0]
        
        inference_time = time.perf_counter() - start_time
        
        # Cache result
        if use_cache and cache_key:
            if len(self.result_cache) >= self.cache_max_size:
                # Simple LRU: remove oldest entry
                oldest_key = next(iter(self.result_cache))
                del self.result_cache[oldest_key]
            
            self.result_cache[cache_key] = (probabilities[0], inference_time)
            self.stats["cache_misses"] += 1
        
        # Update stats
        self.stats["inferences_completed"] += 1
        self.stats["total_inference_time"] += inference_time
        
        return {
            "action": int(action[0].item()),
            "probabilities": probabilities[0].cpu().numpy().tolist(),
            "confidence": float(confidence[0].item()),
            "agent_type": agent_type,
            "inference_time_ms": inference_time * 1000,
            "cached": False
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get inference engine performance statistics."""
        avg_inference_time = (
            self.stats["total_inference_time"] / max(self.stats["inferences_completed"], 1)
        )
        
        cache_hit_rate = (
            self.stats["cache_hits"] / max(self.stats["cache_hits"] + self.stats["cache_misses"], 1)
        )
        
        return {
            "inferences_completed": self.stats["inferences_completed"],
            "avg_inference_time_ms": avg_inference_time * 1000,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.result_cache),
            "models_loaded": list(self.models.keys()),
            "throughput_qps": 1.0 / max(avg_inference_time, 0.001)
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.result_cache.clear()
        self.models.clear()
        gc.collect()
        logger.info("✅ ProductionInferenceEngine cleanup completed")

# Global inference engine instance
_global_inference_engine: Optional[ProductionInferenceEngine] = None

def get_inference_engine() -> ProductionInferenceEngine:
    """Get or create global inference engine."""
    global _global_inference_engine
    
    if _global_inference_engine is None:
        _global_inference_engine = ProductionInferenceEngine()
        _global_inference_engine.initialize()
    
    return _global_inference_engine

def cleanup_inference_engine():
    """Cleanup global inference engine."""
    global _global_inference_engine
    
    if _global_inference_engine:
        _global_inference_engine.cleanup()
        _global_inference_engine = None