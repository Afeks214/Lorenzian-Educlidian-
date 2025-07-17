"""
GPU Optimization Utilities for Google Colab
Handles device management, memory optimization, and performance monitoring
"""

import torch
import gc
import psutil
import os
import time
import logging
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUOptimizer:
    """
    GPU and Memory Optimization for Colab Training
    """
    
    def __init__(self):
        self.device = self._setup_device()
        self.memory_history = deque(maxlen=1000)
        self.gpu_memory_history = deque(maxlen=1000)
        self.performance_metrics = {
            'training_speed': [],
            'memory_usage': [],
            'gpu_utilization': []
        }
        
        # Memory thresholds
        self.memory_warning_threshold = 0.85
        self.memory_critical_threshold = 0.95
        self.gpu_memory_warning_threshold = 0.90
        
        logger.info(f"GPU Optimizer initialized with device: {self.device}")
        self._log_system_info()
    
    def _setup_device(self) -> torch.device:
        """Setup optimal device configuration"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            
            # Optimize CUDA settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Set memory fraction if needed
            if hasattr(torch.cuda, 'set_memory_fraction'):
                torch.cuda.set_memory_fraction(0.9)  # Use 90% of GPU memory
            
            logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = torch.device('cpu')
            logger.warning("CUDA not available, using CPU")
            
        return device
    
    def _log_system_info(self):
        """Log system information"""
        # CPU info
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / 1e9
        
        logger.info(f"CPU cores: {cpu_count}")
        logger.info(f"System RAM: {memory_gb:.1f} GB")
        
        # GPU info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU: {gpu_name}")
            logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
    
    def optimize_batch_size(self, model: torch.nn.Module, 
                          input_shape: Tuple[int, ...], 
                          start_batch_size: int = 64,
                          max_batch_size: int = 512) -> int:
        """
        Find optimal batch size through binary search
        """
        logger.info("Finding optimal batch size...")
        
        model.to(self.device)
        model.train()
        
        def test_batch_size(batch_size: int) -> bool:
            """Test if batch size fits in memory"""
            try:
                # Clear cache
                self.clear_cache()
                
                # Create dummy batch
                dummy_input = torch.randn(batch_size, *input_shape).to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    output = model(dummy_input)
                
                # Simulate backward pass memory requirement
                loss = output.sum()
                loss.backward()
                
                return True
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    return False
                else:
                    raise e
            finally:
                # Cleanup
                del dummy_input
                if 'output' in locals():
                    del output
                if 'loss' in locals():
                    del loss
                self.clear_cache()
        
        # Binary search for optimal batch size
        min_batch = 1
        max_batch = max_batch_size
        optimal_batch = start_batch_size
        
        while min_batch <= max_batch:
            mid_batch = (min_batch + max_batch) // 2
            
            if test_batch_size(mid_batch):
                optimal_batch = mid_batch
                min_batch = mid_batch + 1
            else:
                max_batch = mid_batch - 1
        
        # Use 80% of max found batch size for safety
        safe_batch_size = int(optimal_batch * 0.8)
        
        logger.info(f"Optimal batch size found: {safe_batch_size}")
        return safe_batch_size
    
    def clear_cache(self):
        """Clear GPU and system cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def monitor_memory(self) -> Dict[str, float]:
        """Monitor memory usage"""
        # System memory
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # GPU memory
        gpu_memory_percent = 0.0
        gpu_memory_used = 0.0
        gpu_memory_total = 0.0
        
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / 1e9
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
        
        memory_info = {
            'system_memory_percent': memory_percent,
            'system_memory_used_gb': memory.used / 1e9,
            'system_memory_total_gb': memory.total / 1e9,
            'gpu_memory_percent': gpu_memory_percent,
            'gpu_memory_used_gb': gpu_memory_used,
            'gpu_memory_total_gb': gpu_memory_total
        }
        
        # Store in history
        self.memory_history.append(memory_percent)
        self.gpu_memory_history.append(gpu_memory_percent)
        
        # Check thresholds
        if memory_percent > self.memory_critical_threshold * 100:
            logger.critical(f"Critical memory usage: {memory_percent:.1f}%")
            self.clear_cache()
        elif memory_percent > self.memory_warning_threshold * 100:
            logger.warning(f"High memory usage: {memory_percent:.1f}%")
        
        if gpu_memory_percent > self.gpu_memory_warning_threshold * 100:
            logger.warning(f"High GPU memory usage: {gpu_memory_percent:.1f}%")
        
        return memory_info
    
    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply model optimizations"""
        logger.info("Applying model optimizations...")
        
        # Move to device
        model = model.to(self.device)
        
        # Apply optimizations
        if torch.cuda.is_available():
            # Use JIT tracing if possible (more stable than scripting)
            try:
                # For JIT tracing, we need example input - skip if not available
                # This is safer and more compatible than torch.jit.script
                logger.info("Skipping TorchScript optimization (use trace with example input instead)")
            except Exception as e:
                logger.warning(f"TorchScript optimization failed: {e}")
        
        # Compile model for PyTorch 2.0+
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
                logger.info("Applied torch.compile optimization")
            except Exception as e:
                logger.warning(f"torch.compile optimization failed: {e}")
        
        return model
    
    def setup_dataloader_optimization(self, num_workers: int = None, 
                                    pin_memory: bool = None) -> Dict[str, any]:
        """Setup optimal DataLoader configuration"""
        # Auto-detect optimal num_workers
        if num_workers is None:
            if torch.cuda.is_available():
                num_workers = min(4, psutil.cpu_count())
            else:
                num_workers = min(2, psutil.cpu_count())
        
        # Auto-detect pin_memory
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()
        
        dataloader_config = {
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'persistent_workers': num_workers > 0,
            'prefetch_factor': 2 if num_workers > 0 else 2,
        }
        
        logger.info(f"DataLoader optimization: {dataloader_config}")
        return dataloader_config
    
    def benchmark_training_speed(self, model: torch.nn.Module, 
                                input_shape: Tuple[int, ...], 
                                batch_size: int, 
                                num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark training speed"""
        logger.info(f"Benchmarking training speed for {num_iterations} iterations...")
        
        model.to(self.device)
        model.train()
        
        # Warmup
        dummy_input = torch.randn(batch_size, *input_shape).to(self.device)
        dummy_target = torch.randn(batch_size, 1).to(self.device)
        
        for _ in range(10):  # Warmup iterations
            output = model(dummy_input)
            loss = torch.nn.functional.mse_loss(output, dummy_target)
            loss.backward()
            model.zero_grad()
        
        self.clear_cache()
        
        # Actual benchmark
        start_time = time.time()
        
        for i in range(num_iterations):
            output = model(dummy_input)
            loss = torch.nn.functional.mse_loss(output, dummy_target)
            loss.backward()
            model.zero_grad()
            
            if i % 20 == 0:
                self.monitor_memory()
        
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        iterations_per_second = num_iterations / total_time
        samples_per_second = iterations_per_second * batch_size
        
        benchmark_results = {
            'total_time_seconds': total_time,
            'iterations_per_second': iterations_per_second,
            'samples_per_second': samples_per_second,
            'average_iteration_time_ms': (total_time / num_iterations) * 1000
        }
        
        logger.info(f"Benchmark results: {benchmark_results}")
        return benchmark_results
    
    def auto_mixed_precision_scaler(self):
        """Create GradScaler for automatic mixed precision"""
        if torch.cuda.is_available():
            return torch.cuda.amp.GradScaler()
        return None
    
    def profile_model(self, model: torch.nn.Module, 
                     input_shape: Tuple[int, ...], 
                     batch_size: int = 32):
        """Profile model performance"""
        logger.info("Profiling model...")
        
        model.to(self.device)
        dummy_input = torch.randn(batch_size, *input_shape).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate memory usage
        with torch.no_grad():
            output = model(dummy_input)
            model_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        profile_results = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': (total_params * 4) / 1e6,  # Assuming float32
            'estimated_memory_mb': model_memory / 1e6,
            'output_shape': output.shape
        }
        
        logger.info(f"Model profile: {profile_results}")
        return profile_results
    
    def plot_memory_usage(self, save_path: str = None):
        """Plot memory usage over time"""
        if not self.memory_history:
            logger.warning("No memory history to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # System memory
        ax1.plot(list(self.memory_history))
        ax1.axhline(y=self.memory_warning_threshold * 100, color='orange', 
                   linestyle='--', label='Warning threshold')
        ax1.axhline(y=self.memory_critical_threshold * 100, color='red', 
                   linestyle='--', label='Critical threshold')
        ax1.set_title('System Memory Usage')
        ax1.set_ylabel('Memory Usage (%)')
        ax1.legend()
        ax1.grid(True)
        
        # GPU memory
        if self.gpu_memory_history and torch.cuda.is_available():
            ax2.plot(list(self.gpu_memory_history))
            ax2.axhline(y=self.gpu_memory_warning_threshold * 100, color='orange', 
                       linestyle='--', label='Warning threshold')
            ax2.set_title('GPU Memory Usage')
            ax2.set_ylabel('GPU Memory Usage (%)')
            ax2.legend()
            ax2.grid(True)
        else:
            ax2.text(0.5, 0.5, 'GPU not available', ha='center', va='center', 
                    transform=ax2.transAxes)
            ax2.set_title('GPU Memory Usage - N/A')
        
        ax2.set_xlabel('Time Steps')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on current state"""
        recommendations = []
        
        # Check memory usage
        current_memory = psutil.virtual_memory().percent
        if current_memory > 80:
            recommendations.append("Consider reducing batch size due to high memory usage")
        
        # Check GPU availability
        if not torch.cuda.is_available():
            recommendations.append("Enable GPU runtime in Colab for better performance")
        
        # Check GPU memory
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated()
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
            
            if gpu_memory_percent > 90:
                recommendations.append("GPU memory usage high - consider gradient accumulation")
            elif gpu_memory_percent < 50:
                recommendations.append("GPU underutilized - consider increasing batch size")
        
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 90:
            recommendations.append("High CPU usage - consider reducing DataLoader workers")
        
        return recommendations

def setup_colab_environment():
    """Setup optimal Colab environment"""
    logger.info("Setting up Colab environment...")
    
    # Set environment variables for optimal performance
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    # Disable unnecessary warnings
    import warnings
    warnings.filterwarnings('ignore')
    
    # Set matplotlib backend for Colab
    try:
        import matplotlib
        matplotlib.use('Agg')
        plt.style.use('default')
    except (ImportError, ModuleNotFoundError) as e:
        logger.error(f'Error occurred: {e}')
    
    # Create GPU optimizer
    gpu_optimizer = GPUOptimizer()
    
    # Log recommendations
    recommendations = gpu_optimizer.get_optimization_recommendations()
    if recommendations:
        logger.info("Optimization recommendations:")
        for rec in recommendations:
            logger.info(f"  - {rec}")
    
    return gpu_optimizer

# Convenience functions for Colab notebooks
def quick_gpu_check():
    """Quick GPU availability check"""
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("❌ GPU not available")
        return False

def quick_memory_check():
    """Quick memory status check"""
    memory = psutil.virtual_memory()
    print(f"System Memory: {memory.percent:.1f}% used ({memory.used/1e9:.1f}/{memory.total/1e9:.1f} GB)")
    
    if torch.cuda.is_available():
        gpu_memory_used = torch.cuda.memory_allocated() / 1e9
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_percent = (gpu_memory_used / gpu_memory_total) * 100
        print(f"GPU Memory: {gpu_percent:.1f}% used ({gpu_memory_used:.1f}/{gpu_memory_total:.1f} GB)")

def clear_all_cache():
    """Clear all caches"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("✅ All caches cleared")