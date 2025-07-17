"""
Scalability Manager for NQ Data Pipeline

Provides multi-GPU training, distributed processing, and automatic scaling
capabilities for handling massive datasets across multiple nodes.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
import time
import os
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import psutil
import threading
from queue import Queue
import json
import pickle
import mmap
import subprocess

@dataclass
class ScalingConfiguration:
    """Configuration for scaling operations"""
    max_workers: int = 8
    use_multiprocessing: bool = True
    enable_gpu_acceleration: bool = True
    distributed_training: bool = False
    data_parallelism: bool = True
    memory_limit_gb: float = 16.0
    auto_scaling_enabled: bool = True
    scaling_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.scaling_thresholds is None:
            self.scaling_thresholds = {
                'cpu_usage': 0.8,
                'memory_usage': 0.8,
                'gpu_utilization': 0.8,
                'queue_depth': 100
            }

class GPUManager:
    """Manage GPU resources and multi-GPU operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.devices = [torch.device(f'cuda:{i}') for i in range(self.device_count)]
        self.device_properties = {}
        
        # Initialize device properties
        for i, device in enumerate(self.devices):
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(i)
                self.device_properties[i] = {
                    'name': props.name,
                    'memory_total': props.total_memory,
                    'memory_available': props.total_memory,
                    'compute_capability': (props.major, props.minor),
                    'multiprocessor_count': props.multi_processor_count
                }
    
    def get_available_gpus(self) -> List[int]:
        """Get list of available GPU devices"""
        available = []
        
        for i in range(self.device_count):
            try:
                torch.cuda.set_device(i)
                # Check if device is available
                available.append(i)
            except:
                pass
        
        return available
    
    def get_gpu_memory_usage(self, device_id: int) -> Dict[str, float]:
        """Get GPU memory usage statistics"""
        if not torch.cuda.is_available() or device_id >= self.device_count:
            return {'error': 'GPU not available'}
        
        torch.cuda.set_device(device_id)
        
        return {
            'allocated': torch.cuda.memory_allocated(device_id),
            'cached': torch.cuda.memory_reserved(device_id),
            'total': torch.cuda.get_device_properties(device_id).total_memory,
            'utilization': torch.cuda.memory_allocated(device_id) / torch.cuda.get_device_properties(device_id).total_memory
        }
    
    def optimize_memory_usage(self, device_id: int):
        """Optimize GPU memory usage"""
        if torch.cuda.is_available() and device_id < self.device_count:
            torch.cuda.set_device(device_id)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def distribute_data_to_gpus(self, data: torch.Tensor, devices: List[int]) -> List[torch.Tensor]:
        """Distribute data across multiple GPUs"""
        if not devices or not torch.cuda.is_available():
            return [data]
        
        # Split data evenly across devices
        chunk_size = len(data) // len(devices)
        chunks = []
        
        for i, device_id in enumerate(devices):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < len(devices) - 1 else len(data)
            
            chunk = data[start_idx:end_idx].to(f'cuda:{device_id}')
            chunks.append(chunk)
        
        return chunks
    
    def gather_results_from_gpus(self, results: List[torch.Tensor]) -> torch.Tensor:
        """Gather results from multiple GPUs"""
        if not results:
            return torch.tensor([])
        
        # Move all results to CPU and concatenate
        cpu_results = [result.cpu() for result in results]
        return torch.cat(cpu_results, dim=0)

class MultiGPUProcessor:
    """Multi-GPU data processing system"""
    
    def __init__(self, config: ScalingConfiguration):
        self.config = config
        self.gpu_manager = GPUManager()
        self.logger = logging.getLogger(__name__)
        
        # Available GPUs
        self.available_gpus = self.gpu_manager.get_available_gpus()
        self.logger.info(f"Available GPUs: {self.available_gpus}")
        
        # Processing statistics
        self.processing_stats = {
            'total_batches': 0,
            'successful_batches': 0,
            'failed_batches': 0,
            'total_processing_time': 0.0,
            'gpu_utilization': {}
        }
    
    def process_data_parallel(self, 
                            data: torch.Tensor,
                            processing_func: Callable[[torch.Tensor], torch.Tensor],
                            batch_size: int = 1000) -> torch.Tensor:
        """Process data in parallel across multiple GPUs"""
        if not self.available_gpus:
            self.logger.warning("No GPUs available, falling back to CPU processing")
            return processing_func(data)
        
        start_time = time.time()
        
        # Split data into batches
        batches = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batches.append(batch)
        
        # Distribute batches across GPUs
        gpu_batches = {}
        for i, batch in enumerate(batches):
            gpu_id = self.available_gpus[i % len(self.available_gpus)]
            
            if gpu_id not in gpu_batches:
                gpu_batches[gpu_id] = []
            
            gpu_batches[gpu_id].append(batch)
        
        # Process on each GPU
        results = []
        
        def process_on_gpu(gpu_id: int, batches: List[torch.Tensor]):
            """Process batches on a specific GPU"""
            torch.cuda.set_device(gpu_id)
            gpu_results = []
            
            for batch in batches:
                try:
                    batch_gpu = batch.to(f'cuda:{gpu_id}')
                    result = processing_func(batch_gpu)
                    gpu_results.append(result)
                    
                    self.processing_stats['successful_batches'] += 1
                    
                except Exception as e:
                    self.logger.error(f"GPU {gpu_id} processing error: {e}")
                    self.processing_stats['failed_batches'] += 1
            
            return gpu_results
        
        # Use ThreadPoolExecutor for GPU processing
        with ThreadPoolExecutor(max_workers=len(self.available_gpus)) as executor:
            futures = []
            
            for gpu_id, batches in gpu_batches.items():
                future = executor.submit(process_on_gpu, gpu_id, batches)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                gpu_results = future.result()
                results.extend(gpu_results)
        
        # Gather all results
        if results:
            final_result = self.gpu_manager.gather_results_from_gpus(results)
        else:
            final_result = torch.tensor([])
        
        # Update statistics
        processing_time = time.time() - start_time
        self.processing_stats['total_batches'] += len(batches)
        self.processing_stats['total_processing_time'] += processing_time
        
        self.logger.info(f"Processed {len(batches)} batches on {len(self.available_gpus)} GPUs in {processing_time:.2f}s")
        
        return final_result
    
    def setup_distributed_training(self, rank: int, world_size: int):
        """Setup distributed training environment"""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # Initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
        # Set device for this process
        torch.cuda.set_device(rank)
        
        self.logger.info(f"Distributed training initialized: rank {rank}, world_size {world_size}")
    
    def create_distributed_dataloader(self, 
                                    dataset: torch.utils.data.Dataset,
                                    batch_size: int,
                                    shuffle: bool = True) -> DataLoader:
        """Create distributed data loader"""
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
    
    def wrap_model_for_distributed(self, model: torch.nn.Module, device_id: int) -> DDP:
        """Wrap model for distributed training"""
        model = model.to(device_id)
        return DDP(model, device_ids=[device_id])
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get multi-GPU processing statistics"""
        stats = self.processing_stats.copy()
        
        # Add GPU utilization info
        for gpu_id in self.available_gpus:
            stats['gpu_utilization'][gpu_id] = self.gpu_manager.get_gpu_memory_usage(gpu_id)
        
        # Calculate success rate
        total_batches = stats['successful_batches'] + stats['failed_batches']
        stats['success_rate'] = stats['successful_batches'] / total_batches if total_batches > 0 else 0
        
        # Calculate average processing time
        stats['avg_processing_time'] = stats['total_processing_time'] / stats['total_batches'] if stats['total_batches'] > 0 else 0
        
        return stats

class DistributedProcessor:
    """Distributed processing across multiple nodes"""
    
    def __init__(self, config: ScalingConfiguration):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Node configuration
        self.node_id = os.environ.get('NODE_ID', 0)
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        self.master_port = os.environ.get('MASTER_PORT', '12355')
        
        # Processing queue
        self.task_queue = Queue()
        self.result_queue = Queue()
        
        # Worker processes
        self.worker_processes = []
        self.is_running = False
    
    def start_distributed_processing(self):
        """Start distributed processing system"""
        self.is_running = True
        
        # Start worker processes
        for i in range(self.config.max_workers):
            if self.config.use_multiprocessing:
                process = mp.Process(target=self._worker_process, args=(i,))
            else:
                process = threading.Thread(target=self._worker_process, args=(i,))
            
            process.start()
            self.worker_processes.append(process)
        
        self.logger.info(f"Started {len(self.worker_processes)} worker processes")
    
    def stop_distributed_processing(self):
        """Stop distributed processing system"""
        self.is_running = False
        
        # Wait for workers to finish
        for process in self.worker_processes:
            process.join()
        
        self.worker_processes.clear()
        self.logger.info("Stopped distributed processing")
    
    def _worker_process(self, worker_id: int):
        """Worker process for distributed processing"""
        while self.is_running:
            try:
                # Get task from queue
                task = self.task_queue.get(timeout=1)
                
                if task is None:  # Shutdown signal
                    break
                
                # Process task
                result = self._process_task(task, worker_id)
                
                # Put result in result queue
                self.result_queue.put(result)
                
            except:
                continue  # Timeout or error, continue
    
    def _process_task(self, task: Dict[str, Any], worker_id: int) -> Dict[str, Any]:
        """Process a single task"""
        try:
            task_type = task['type']
            data = task['data']
            processing_func = task['processing_func']
            
            # Set GPU if available
            if torch.cuda.is_available() and worker_id < torch.cuda.device_count():
                torch.cuda.set_device(worker_id)
            
            # Process data
            start_time = time.time()
            result = processing_func(data)
            processing_time = time.time() - start_time
            
            return {
                'task_id': task.get('task_id'),
                'result': result,
                'processing_time': processing_time,
                'worker_id': worker_id,
                'success': True
            }
            
        except Exception as e:
            return {
                'task_id': task.get('task_id'),
                'error': str(e),
                'worker_id': worker_id,
                'success': False
            }
    
    def submit_task(self, task: Dict[str, Any]):
        """Submit a task for processing"""
        self.task_queue.put(task)
    
    def get_results(self, timeout: float = 10.0) -> List[Dict[str, Any]]:
        """Get processing results"""
        results = []
        
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get(timeout=timeout)
                results.append(result)
            except:
                break
        
        return results

class AutoScaler:
    """Automatic scaling based on system metrics"""
    
    def __init__(self, config: ScalingConfiguration):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Current scaling state
        self.current_workers = config.max_workers
        self.scaling_history = []
        
        # Monitoring
        self.monitoring_thread = None
        self.monitoring_enabled = False
        
        # Scaling decisions
        self.scaling_cooldown = 30  # seconds
        self.last_scaling_time = 0
    
    def start_monitoring(self):
        """Start automatic scaling monitoring"""
        self.monitoring_enabled = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info("Auto-scaling monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_enabled = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        self.logger.info("Auto-scaling monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_enabled:
            try:
                # Check system metrics
                metrics = self._get_system_metrics()
                
                # Make scaling decision
                scaling_decision = self._make_scaling_decision(metrics)
                
                # Apply scaling if needed
                if scaling_decision != 'no_change':
                    self._apply_scaling(scaling_decision, metrics)
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(5)
    
    def _get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        # CPU metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # GPU metrics
        gpu_utilization = 0
        if torch.cuda.is_available():
            gpu_utilization = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
        
        # Queue metrics (placeholder)
        queue_depth = 0  # Would need to be implemented based on actual queue
        
        return {
            'cpu_usage': cpu_usage / 100,
            'memory_usage': memory_usage / 100,
            'gpu_utilization': gpu_utilization / 100,
            'queue_depth': queue_depth
        }
    
    def _make_scaling_decision(self, metrics: Dict[str, float]) -> str:
        """Make scaling decision based on metrics"""
        if not self.config.auto_scaling_enabled:
            return 'no_change'
        
        # Check cooldown period
        if time.time() - self.last_scaling_time < self.scaling_cooldown:
            return 'no_change'
        
        thresholds = self.config.scaling_thresholds
        
        # Check if we need to scale up
        scale_up_conditions = [
            metrics['cpu_usage'] > thresholds['cpu_usage'],
            metrics['memory_usage'] > thresholds['memory_usage'],
            metrics['gpu_utilization'] > thresholds['gpu_utilization'],
            metrics['queue_depth'] > thresholds['queue_depth']
        ]
        
        # Check if we need to scale down
        scale_down_conditions = [
            metrics['cpu_usage'] < thresholds['cpu_usage'] * 0.5,
            metrics['memory_usage'] < thresholds['memory_usage'] * 0.5,
            metrics['gpu_utilization'] < thresholds['gpu_utilization'] * 0.5,
            metrics['queue_depth'] < thresholds['queue_depth'] * 0.5
        ]
        
        if sum(scale_up_conditions) >= 2:  # At least 2 conditions met
            return 'scale_up'
        elif sum(scale_down_conditions) >= 3:  # At least 3 conditions met
            return 'scale_down'
        else:
            return 'no_change'
    
    def _apply_scaling(self, decision: str, metrics: Dict[str, float]):
        """Apply scaling decision"""
        if decision == 'scale_up':
            new_workers = min(self.current_workers + 2, self.config.max_workers)
        elif decision == 'scale_down':
            new_workers = max(self.current_workers - 1, 1)
        else:
            return
        
        if new_workers != self.current_workers:
            self.current_workers = new_workers
            self.last_scaling_time = time.time()
            
            # Record scaling event
            self.scaling_history.append({
                'timestamp': time.time(),
                'decision': decision,
                'old_workers': self.current_workers,
                'new_workers': new_workers,
                'metrics': metrics.copy()
            })
            
            self.logger.info(f"Scaled {decision}: {self.current_workers} -> {new_workers} workers")
    
    def get_scaling_statistics(self) -> Dict[str, Any]:
        """Get scaling statistics"""
        return {
            'current_workers': self.current_workers,
            'max_workers': self.config.max_workers,
            'scaling_events': len(self.scaling_history),
            'scaling_history': self.scaling_history[-10:],  # Last 10 events
            'auto_scaling_enabled': self.config.auto_scaling_enabled
        }

class ScalabilityManager:
    """Main scalability management system"""
    
    def __init__(self, config: Optional[ScalingConfiguration] = None):
        self.config = config or ScalingConfiguration()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.gpu_processor = MultiGPUProcessor(self.config)
        self.distributed_processor = DistributedProcessor(self.config)
        self.auto_scaler = AutoScaler(self.config)
        
        # System state
        self.is_initialized = False
        self.processing_mode = 'single'  # 'single', 'multi_gpu', 'distributed'
        
        self.logger.info("Scalability manager initialized")
    
    def initialize_system(self, mode: str = 'auto'):
        """Initialize scalability system"""
        if mode == 'auto':
            # Automatically determine best mode
            if torch.cuda.device_count() > 1:
                mode = 'multi_gpu'
            elif self.config.distributed_training:
                mode = 'distributed'
            else:
                mode = 'single'
        
        self.processing_mode = mode
        
        if mode == 'multi_gpu':
            self.logger.info("Initializing multi-GPU processing")
            
        elif mode == 'distributed':
            self.logger.info("Initializing distributed processing")
            self.distributed_processor.start_distributed_processing()
            
        # Start auto-scaling if enabled
        if self.config.auto_scaling_enabled:
            self.auto_scaler.start_monitoring()
        
        self.is_initialized = True
        self.logger.info(f"Scalability system initialized in {mode} mode")
    
    def process_large_dataset(self, 
                            data: torch.Tensor,
                            processing_func: Callable[[torch.Tensor], torch.Tensor],
                            batch_size: int = 1000) -> torch.Tensor:
        """Process large dataset with automatic scaling"""
        if not self.is_initialized:
            self.initialize_system()
        
        if self.processing_mode == 'multi_gpu':
            return self.gpu_processor.process_data_parallel(data, processing_func, batch_size)
        
        elif self.processing_mode == 'distributed':
            # Submit tasks to distributed processor
            results = []
            
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                task = {
                    'type': 'process_batch',
                    'data': batch,
                    'processing_func': processing_func,
                    'task_id': i // batch_size
                }
                
                self.distributed_processor.submit_task(task)
            
            # Wait for results
            num_batches = (len(data) + batch_size - 1) // batch_size
            collected_results = []
            
            while len(collected_results) < num_batches:
                batch_results = self.distributed_processor.get_results()
                collected_results.extend(batch_results)
                time.sleep(0.1)
            
            # Combine results
            successful_results = [r['result'] for r in collected_results if r['success']]
            
            if successful_results:
                return torch.cat(successful_results, dim=0)
            else:
                return torch.tensor([])
        
        else:  # single mode
            return processing_func(data)
    
    def get_system_capabilities(self) -> Dict[str, Any]:
        """Get system scalability capabilities"""
        return {
            'processing_mode': self.processing_mode,
            'gpu_count': torch.cuda.device_count(),
            'gpu_available': torch.cuda.is_available(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'distributed_enabled': self.config.distributed_training,
            'auto_scaling_enabled': self.config.auto_scaling_enabled,
            'max_workers': self.config.max_workers
        }
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = {
            'system_capabilities': self.get_system_capabilities(),
            'processing_mode': self.processing_mode,
            'is_initialized': self.is_initialized
        }
        
        if self.processing_mode == 'multi_gpu':
            stats['gpu_processing'] = self.gpu_processor.get_processing_statistics()
        
        if self.config.auto_scaling_enabled:
            stats['auto_scaling'] = self.auto_scaler.get_scaling_statistics()
        
        return stats
    
    def optimize_for_data_size(self, data_size_gb: float) -> Dict[str, Any]:
        """Optimize configuration for specific data size"""
        recommendations = {
            'recommended_batch_size': 1000,
            'recommended_workers': self.config.max_workers,
            'recommended_mode': self.processing_mode,
            'memory_optimization': []
        }
        
        # Batch size recommendations
        if data_size_gb < 1:
            recommendations['recommended_batch_size'] = 500
        elif data_size_gb < 10:
            recommendations['recommended_batch_size'] = 1000
        else:
            recommendations['recommended_batch_size'] = 2000
        
        # Worker count recommendations
        if data_size_gb > 50:
            recommendations['recommended_workers'] = min(self.config.max_workers, 16)
            recommendations['recommended_mode'] = 'distributed'
        elif data_size_gb > 10:
            recommendations['recommended_mode'] = 'multi_gpu'
        
        # Memory optimization recommendations
        if data_size_gb > psutil.virtual_memory().total / (1024**3) * 0.8:
            recommendations['memory_optimization'].append('Enable memory mapping')
            recommendations['memory_optimization'].append('Use smaller batch sizes')
            recommendations['memory_optimization'].append('Enable gradient checkpointing')
        
        return recommendations
    
    def cleanup(self):
        """Cleanup scalability resources"""
        self.auto_scaler.stop_monitoring()
        self.distributed_processor.stop_distributed_processing()
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Scalability manager cleaned up")
    
    def __del__(self):
        """Destructor"""
        self.cleanup()