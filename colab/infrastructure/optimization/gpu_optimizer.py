#!/usr/bin/env python3
"""
GPU Optimization System for Training Infrastructure
Optimizes GPU usage, memory management, and performance
"""

import os
import gc
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import numpy as np

try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

@dataclass
class GPUConfig:
    """GPU configuration settings"""
    device_ids: List[int]
    memory_fraction: float = 0.9
    allow_growth: bool = True
    mixed_precision: bool = True
    compile_model: bool = True
    use_flash_attention: bool = True
    gradient_checkpointing: bool = False
    pin_memory: bool = True
    non_blocking: bool = True
    benchmark: bool = True
    deterministic: bool = False

class GPUOptimizer:
    """Comprehensive GPU optimization system"""
    
    def __init__(self, config: Optional[GPUConfig] = None):
        self.config = config or GPUConfig(device_ids=[0])
        self.logger = logging.getLogger(__name__)
        
        # Initialize NVML if available
        if NVML_AVAILABLE:
            try:
                nvml.nvmlInit()
                self.nvml_initialized = True
            except:
                self.nvml_initialized = False
                self.logger.warning("NVML initialization failed")
        else:
            self.nvml_initialized = False
        
        # GPU information
        self.gpu_info = self._get_gpu_info()
        self.optimized_models = {}
        
        # Setup basic optimizations
        self._setup_basic_optimizations()
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get comprehensive GPU information"""
        gpu_info = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'devices': []
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_info = {
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'capability': torch.cuda.get_device_capability(i),
                    'memory_total': torch.cuda.get_device_properties(i).total_memory / (1024**3),
                    'memory_reserved': torch.cuda.memory_reserved(i) / (1024**3),
                    'memory_allocated': torch.cuda.memory_allocated(i) / (1024**3)
                }
                
                # Additional NVML info if available
                if self.nvml_initialized:
                    try:
                        handle = nvml.nvmlDeviceGetHandleByIndex(i)
                        device_info.update({
                            'temperature': nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU),
                            'utilization': nvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                            'power_usage': nvml.nvmlDeviceGetPowerUsage(handle) / 1000,  # Convert to watts
                            'memory_info': nvml.nvmlDeviceGetMemoryInfo(handle)
                        })
                    except Exception as e:
                        self.logger.warning(f"Failed to get NVML info for GPU {i}: {e}")
                
                gpu_info['devices'].append(device_info)
        
        return gpu_info
    
    def _setup_basic_optimizations(self):
        """Setup basic PyTorch optimizations"""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available, skipping GPU optimizations")
            return
        
        # Enable cudNN optimizations
        torch.backends.cudnn.benchmark = self.config.benchmark
        torch.backends.cudnn.deterministic = self.config.deterministic
        
        # Memory management
        if hasattr(torch.cuda, 'memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(self.config.memory_fraction)
        
        # Enable TensorFloat-32 for A100/H100
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.logger.info("Enabled TensorFloat-32 for improved performance")
    
    def optimize_model(self, model: nn.Module, name: str = "model") -> nn.Module:
        """Optimize model for GPU training"""
        if not torch.cuda.is_available():
            return model
        
        try:
            # Move to GPU
            if len(self.config.device_ids) == 1:
                device = f"cuda:{self.config.device_ids[0]}"
                model = model.to(device)
            else:
                # Multi-GPU setup
                model = model.cuda()
                model = nn.DataParallel(model, device_ids=self.config.device_ids)
            
            # Gradient checkpointing for memory efficiency
            if self.config.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            
            # Compile model for PyTorch 2.0+
            if self.config.compile_model and hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model, mode='reduce-overhead')
                    self.logger.info(f"Model {name} compiled successfully")
                except Exception as e:
                    self.logger.warning(f"Model compilation failed: {e}")
            
            self.optimized_models[name] = model
            self.logger.info(f"Model {name} optimized for GPU training")
            
            return model
        
        except Exception as e:
            self.logger.error(f"Model optimization failed: {e}")
            return model
    
    def optimize_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """Optimize dataloader for GPU training"""
        if not torch.cuda.is_available():
            return dataloader
        
        # Update dataloader settings
        dataloader.pin_memory = self.config.pin_memory
        dataloader.num_workers = min(dataloader.num_workers, 8)  # Avoid too many workers
        
        return dataloader
    
    def create_scaler(self) -> Optional[torch.cuda.amp.GradScaler]:
        """Create gradient scaler for mixed precision training"""
        if not torch.cuda.is_available() or not self.config.mixed_precision:
            return None
        
        return torch.cuda.amp.GradScaler()
    
    def optimize_memory(self, aggressive: bool = False):
        """Optimize GPU memory usage"""
        if not torch.cuda.is_available():
            return
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        if aggressive:
            # More aggressive memory management
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_max_memory_cached()
        
        self.logger.info("GPU memory optimized")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get detailed memory statistics"""
        if not torch.cuda.is_available():
            return {}
        
        stats = {}
        for device_id in self.config.device_ids:
            device_stats = {
                'allocated': torch.cuda.memory_allocated(device_id) / (1024**3),
                'reserved': torch.cuda.memory_reserved(device_id) / (1024**3),
                'max_allocated': torch.cuda.max_memory_allocated(device_id) / (1024**3),
                'max_reserved': torch.cuda.max_memory_reserved(device_id) / (1024**3),
                'total': torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
            }
            
            device_stats['utilization'] = (device_stats['allocated'] / device_stats['total']) * 100
            stats[f'gpu_{device_id}'] = device_stats
        
        return stats
    
    def monitor_gpu_utilization(self, duration: float = 10.0, interval: float = 1.0) -> Dict[str, List[float]]:
        """Monitor GPU utilization over time"""
        if not self.nvml_initialized:
            self.logger.warning("NVML not available, cannot monitor GPU utilization")
            return {}
        
        utilization_data = {f'gpu_{i}': [] for i in self.config.device_ids}
        
        start_time = time.time()
        while time.time() - start_time < duration:
            for device_id in self.config.device_ids:
                try:
                    handle = nvml.nvmlDeviceGetHandleByIndex(device_id)
                    util = nvml.nvmlDeviceGetUtilizationRates(handle)
                    utilization_data[f'gpu_{device_id}'].append(util.gpu)
                except Exception as e:
                    self.logger.warning(f"Failed to get utilization for GPU {device_id}: {e}")
            
            time.sleep(interval)
        
        return utilization_data
    
    def benchmark_model(self, model: nn.Module, input_shape: Tuple[int, ...], 
                       num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark model performance"""
        if not torch.cuda.is_available():
            return {}
        
        device = f"cuda:{self.config.device_ids[0]}"
        model = model.to(device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape, device=device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(dummy_input)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        throughput = num_iterations / total_time
        
        return {
            'total_time': total_time,
            'avg_inference_time': avg_time,
            'throughput': throughput,
            'memory_usage': torch.cuda.max_memory_allocated(device) / (1024**3)
        }
    
    def optimize_for_inference(self, model: nn.Module) -> nn.Module:
        """Optimize model for inference"""
        if not torch.cuda.is_available():
            return model
        
        # Set to evaluation mode
        model.eval()
        
        # Optimize with TorchScript
        try:
            # Create example input
            device = f"cuda:{self.config.device_ids[0]}"
            example_input = torch.randn(1, 3, 224, 224, device=device)  # Adjust as needed
            
            # Trace the model
            traced_model = torch.jit.trace(model, example_input)
            traced_model = torch.jit.optimize_for_inference(traced_model)
            
            self.logger.info("Model optimized for inference with TorchScript")
            return traced_model
        
        except Exception as e:
            self.logger.warning(f"TorchScript optimization failed: {e}")
            return model
    
    def save_optimization_report(self, filepath: str):
        """Save optimization report"""
        report = {
            'gpu_info': self.gpu_info,
            'config': {
                'device_ids': self.config.device_ids,
                'memory_fraction': self.config.memory_fraction,
                'mixed_precision': self.config.mixed_precision,
                'compile_model': self.config.compile_model,
                'gradient_checkpointing': self.config.gradient_checkpointing
            },
            'memory_stats': self.get_memory_stats(),
            'optimized_models': list(self.optimized_models.keys()),
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Optimization report saved to {filepath}")
    
    def cleanup(self):
        """Cleanup GPU resources"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        self.optimized_models.clear()
        
        if self.nvml_initialized:
            try:
                nvml.nvmlShutdown()
            except:
                pass
        
        self.logger.info("GPU resources cleaned up")

# Example usage and factory functions
def create_gpu_optimizer(device_ids: List[int] = None, 
                        mixed_precision: bool = True,
                        compile_model: bool = True) -> GPUOptimizer:
    """Create GPU optimizer with default settings"""
    if device_ids is None:
        device_ids = [0] if torch.cuda.is_available() else []
    
    config = GPUConfig(
        device_ids=device_ids,
        mixed_precision=mixed_precision,
        compile_model=compile_model
    )
    
    return GPUOptimizer(config)

def optimize_training_setup(model: nn.Module, dataloader: DataLoader, 
                          device_ids: List[int] = None) -> Tuple[nn.Module, DataLoader, Optional[torch.cuda.amp.GradScaler]]:
    """Optimize complete training setup"""
    optimizer = create_gpu_optimizer(device_ids)
    
    # Optimize model
    optimized_model = optimizer.optimize_model(model, "training_model")
    
    # Optimize dataloader
    optimized_dataloader = optimizer.optimize_dataloader(dataloader)
    
    # Create scaler
    scaler = optimizer.create_scaler()
    
    return optimized_model, optimized_dataloader, scaler

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create optimizer
    optimizer = create_gpu_optimizer()
    
    # Print GPU info
    print("GPU Information:")
    print(json.dumps(optimizer.gpu_info, indent=2))
    
    # Example model
    model = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    # Optimize model
    optimized_model = optimizer.optimize_model(model, "test_model")
    
    # Benchmark
    if torch.cuda.is_available():
        benchmark_results = optimizer.benchmark_model(
            optimized_model, 
            (32, 1024),  # batch_size, input_size
            num_iterations=100
        )
        print(f"Benchmark Results: {benchmark_results}")
    
    # Save report
    optimizer.save_optimization_report("/tmp/gpu_optimization_report.json")
    
    # Cleanup
    optimizer.cleanup()