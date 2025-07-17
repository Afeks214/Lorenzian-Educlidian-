#!/usr/bin/env python3
"""
AGENT 3 Distributed Inference System
High-performance distributed model serving with load balancing
"""

import torch
import torch.nn as nn
import torch.jit
import torch.quantization
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
import time
import numpy as np
from pathlib import Path
import gc
import threading
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from collections import defaultdict
import json
import psutil
import os
import asyncio
import aiohttp
from aiohttp import web
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import queue
import socket
import uvloop

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics structure"""
    avg_latency_ms: float
    p99_latency_ms: float
    throughput_qps: float
    memory_usage_mb: float
    cpu_usage_percent: float
    meets_target: bool
    model_size_mb: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'avg_latency_ms': float(self.avg_latency_ms),
            'p99_latency_ms': float(self.p99_latency_ms),
            'throughput_qps': float(self.throughput_qps),
            'memory_usage_mb': float(self.memory_usage_mb),
            'cpu_usage_percent': float(self.cpu_usage_percent),
            'meets_target': bool(self.meets_target),
            'model_size_mb': float(self.model_size_mb)
        }

class OptimizedModelRegistry:
    """Registry for optimized models"""
    
    def __init__(self):
        self.models = {}
        self.model_metadata = {}
        self.performance_cache = {}
        
    def register_model(self, name: str, model: torch.jit.ScriptModule, metadata: Dict[str, Any]):
        """Register an optimized model"""
        self.models[name] = model
        self.model_metadata[name] = metadata
        logger.info(f"Registered model: {name}")
    
    def get_model(self, name: str) -> Optional[torch.jit.ScriptModule]:
        """Get registered model"""
        return self.models.get(name)
    
    def get_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get model metadata"""
        return self.model_metadata.get(name)
    
    def list_models(self) -> List[str]:
        """List all registered models"""
        return list(self.models.keys())

class InferenceWorker:
    """Worker process for model inference"""
    
    def __init__(self, worker_id: int, model_registry: OptimizedModelRegistry):
        self.worker_id = worker_id
        self.model_registry = model_registry
        self.stats = {
            'requests_handled': 0,
            'total_inference_time': 0.0,
            'errors': 0
        }
        
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process inference request"""
        try:
            model_name = request['model_name']
            input_data = torch.tensor(request['input_data'])
            
            model = self.model_registry.get_model(model_name)
            if model is None:
                raise ValueError(f"Model {model_name} not found")
            
            start_time = time.perf_counter()
            
            with torch.no_grad():
                if isinstance(input_data, list):
                    result = model(*input_data)
                else:
                    result = model(input_data)
            
            end_time = time.perf_counter()
            
            inference_time = (end_time - start_time) * 1000  # ms
            
            self.stats['requests_handled'] += 1
            self.stats['total_inference_time'] += inference_time
            
            return {
                'success': True,
                'result': result.tolist() if isinstance(result, torch.Tensor) else [r.tolist() for r in result],
                'inference_time_ms': inference_time,
                'worker_id': self.worker_id
            }
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Worker {self.worker_id} error: {e}")
            return {
                'success': False,
                'error': str(e),
                'worker_id': self.worker_id
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        avg_time = self.stats['total_inference_time'] / max(self.stats['requests_handled'], 1)
        return {
            'worker_id': self.worker_id,
            'requests_handled': self.stats['requests_handled'],
            'avg_inference_time_ms': avg_time,
            'total_inference_time_ms': self.stats['total_inference_time'],
            'errors': self.stats['errors'],
            'error_rate': self.stats['errors'] / max(self.stats['requests_handled'], 1)
        }

class LoadBalancer:
    """Load balancer for distributing inference requests"""
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.workers = []
        self.current_worker = 0
        self.worker_loads = defaultdict(int)
        self.strategy = "round_robin"  # round_robin, least_loaded, weighted
        
    def add_worker(self, worker: InferenceWorker):
        """Add worker to load balancer"""
        self.workers.append(worker)
        
    def get_next_worker(self) -> InferenceWorker:
        """Get next worker based on load balancing strategy"""
        if self.strategy == "round_robin":
            worker = self.workers[self.current_worker]
            self.current_worker = (self.current_worker + 1) % len(self.workers)
            return worker
        
        elif self.strategy == "least_loaded":
            # Find worker with least load
            min_load = min(self.worker_loads.values())
            for worker in self.workers:
                if self.worker_loads[worker.worker_id] == min_load:
                    return worker
            return self.workers[0]  # fallback
        
        else:  # round_robin fallback
            return self.workers[self.current_worker % len(self.workers)]
    
    def update_worker_load(self, worker_id: int, load_delta: int):
        """Update worker load"""
        self.worker_loads[worker_id] += load_delta
    
    def get_load_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics"""
        return {
            'strategy': self.strategy,
            'num_workers': len(self.workers),
            'worker_loads': dict(self.worker_loads),
            'current_worker': self.current_worker
        }

class InferenceCache:
    """High-performance inference cache"""
    
    def __init__(self, max_size: int = 100000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        self.lock = threading.RLock()
    
    def _generate_key(self, model_name: str, input_data: Any) -> str:
        """Generate cache key"""
        if isinstance(input_data, torch.Tensor):
            data_hash = hash(input_data.numpy().tobytes())
        else:
            data_hash = hash(str(input_data))
        return f"{model_name}_{data_hash}"
    
    def get(self, model_name: str, input_data: Any) -> Optional[Dict[str, Any]]:
        """Get cached result"""
        key = self._generate_key(model_name, input_data)
        
        with self.lock:
            if key in self.cache:
                current_time = time.time()
                if current_time - self.access_times[key] < self.ttl_seconds:
                    self.access_times[key] = current_time
                    self.stats['hits'] += 1
                    result = self.cache[key].copy()
                    result['cached'] = True
                    return result
                else:
                    # Expired
                    del self.cache[key]
                    del self.access_times[key]
            
            self.stats['misses'] += 1
            return None
    
    def put(self, model_name: str, input_data: Any, result: Dict[str, Any]):
        """Cache result"""
        key = self._generate_key(model_name, input_data)
        
        with self.lock:
            # Check if cache is full
            if len(self.cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
                self.stats['evictions'] += 1
            
            self.cache[key] = result
            self.access_times[key] = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / max(total_requests, 1)
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'evictions': self.stats['evictions'],
                'ttl_seconds': self.ttl_seconds
            }
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()

class DistributedInferenceServer:
    """Main distributed inference server"""
    
    def __init__(self, port: int = 8080, num_workers: int = 4):
        self.port = port
        self.num_workers = num_workers
        self.model_registry = OptimizedModelRegistry()
        self.load_balancer = LoadBalancer(num_workers)
        self.cache = InferenceCache()
        self.app = web.Application()
        self.setup_routes()
        
        # Performance monitoring
        self.request_count = 0
        self.total_response_time = 0.0
        self.start_time = time.time()
        
        # Initialize workers
        self.initialize_workers()
        
    def initialize_workers(self):
        """Initialize inference workers"""
        for i in range(self.num_workers):
            worker = InferenceWorker(i, self.model_registry)
            self.load_balancer.add_worker(worker)
        
        logger.info(f"Initialized {self.num_workers} inference workers")
    
    def setup_routes(self):
        """Setup HTTP routes"""
        self.app.router.add_post('/predict', self.handle_predict)
        self.app.router.add_get('/health', self.handle_health)
        self.app.router.add_get('/stats', self.handle_stats)
        self.app.router.add_get('/models', self.handle_models)
        self.app.router.add_post('/load_model', self.handle_load_model)
    
    async def handle_predict(self, request: web.Request) -> web.Response:
        """Handle prediction request"""
        try:
            data = await request.json()
            model_name = data.get('model_name')
            input_data = data.get('input_data')
            use_cache = data.get('use_cache', True)
            
            if not model_name or input_data is None:
                return web.json_response({'error': 'Missing model_name or input_data'}, status=400)
            
            start_time = time.perf_counter()
            
            # Check cache first
            if use_cache:
                cached_result = self.cache.get(model_name, input_data)
                if cached_result:
                    return web.json_response(cached_result)
            
            # Get worker and process request
            worker = self.load_balancer.get_next_worker()
            self.load_balancer.update_worker_load(worker.worker_id, 1)
            
            inference_request = {
                'model_name': model_name,
                'input_data': input_data
            }
            
            result = worker.process_request(inference_request)
            
            # Cache result if successful
            if use_cache and result['success']:
                self.cache.put(model_name, input_data, result)
            
            end_time = time.perf_counter()
            
            # Update server stats
            self.request_count += 1
            self.total_response_time += (end_time - start_time) * 1000
            
            # Update worker load
            self.load_balancer.update_worker_load(worker.worker_id, -1)
            
            return web.json_response(result)
            
        except Exception as e:
            logger.error(f"Error handling prediction: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def handle_health(self, request: web.Request) -> web.Response:
        """Handle health check"""
        return web.json_response({
            'status': 'healthy',
            'uptime_seconds': time.time() - self.start_time,
            'models_loaded': len(self.model_registry.list_models())
        })
    
    async def handle_stats(self, request: web.Request) -> web.Response:
        """Handle statistics request"""
        avg_response_time = self.total_response_time / max(self.request_count, 1)
        
        stats = {
            'server': {
                'requests_handled': self.request_count,
                'avg_response_time_ms': avg_response_time,
                'uptime_seconds': time.time() - self.start_time,
                'requests_per_second': self.request_count / max(time.time() - self.start_time, 1)
            },
            'cache': self.cache.get_stats(),
            'load_balancer': self.load_balancer.get_load_stats(),
            'workers': [worker.get_stats() for worker in self.load_balancer.workers],
            'system': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage_percent': psutil.disk_usage('/').percent
            }
        }
        
        return web.json_response(stats)
    
    async def handle_models(self, request: web.Request) -> web.Response:
        """Handle models list request"""
        models = {}
        for model_name in self.model_registry.list_models():
            metadata = self.model_registry.get_metadata(model_name)
            models[model_name] = metadata
        
        return web.json_response({
            'models': models,
            'count': len(models)
        })
    
    async def handle_load_model(self, request: web.Request) -> web.Response:
        """Handle load model request"""
        try:
            data = await request.json()
            model_path = data.get('model_path')
            model_name = data.get('model_name')
            
            if not model_path or not model_name:
                return web.json_response({'error': 'Missing model_path or model_name'}, status=400)
            
            # Load model
            model = torch.jit.load(model_path)
            model.eval()
            
            # Register model
            metadata = {
                'name': model_name,
                'path': model_path,
                'loaded_at': time.time(),
                'size_mb': os.path.getsize(model_path) / (1024 * 1024)
            }
            
            self.model_registry.register_model(model_name, model, metadata)
            
            return web.json_response({'success': True, 'message': f'Model {model_name} loaded successfully'})
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    def load_models_from_directory(self, model_dir: Path):
        """Load all models from directory"""
        model_dir = Path(model_dir)
        
        if not model_dir.exists():
            logger.warning(f"Model directory {model_dir} does not exist")
            return
        
        for model_file in model_dir.glob("*.pt"):
            try:
                model_name = model_file.stem
                model = torch.jit.load(str(model_file))
                model.eval()
                
                metadata = {
                    'name': model_name,
                    'path': str(model_file),
                    'loaded_at': time.time(),
                    'size_mb': model_file.stat().st_size / (1024 * 1024)
                }
                
                self.model_registry.register_model(model_name, model, metadata)
                
            except Exception as e:
                logger.error(f"Failed to load model {model_file}: {e}")
    
    async def start_server(self):
        """Start the inference server"""
        logger.info(f"Starting distributed inference server on port {self.port}")
        
        # Setup uvloop for better performance
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        
        logger.info(f"Server running on http://0.0.0.0:{self.port}")
        logger.info(f"Health check: http://0.0.0.0:{self.port}/health")
        logger.info(f"Stats: http://0.0.0.0:{self.port}/stats")

class PerformanceOptimizer:
    """Performance optimization system"""
    
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.performance_results = {}
        
    def create_optimized_models(self) -> Dict[str, nn.Module]:
        """Create all optimized models"""
        from performance_optimization_system import JITOptimizedTacticalSystem, JITOptimizedStrategicSystem
        from performance_optimization_system import JITOptimizedTacticalActor, JITOptimizedStrategicActor, JITOptimizedCritic
        
        models = {
            'tactical_system': JITOptimizedTacticalSystem(),
            'strategic_system': JITOptimizedStrategicSystem(),
            'fvg_agent': JITOptimizedTacticalActor("fvg"),
            'momentum_agent': JITOptimizedTacticalActor("momentum"),
            'entry_agent': JITOptimizedTacticalActor("entry"),
            'mlmi_agent': JITOptimizedStrategicActor(input_dim=4),
            'nwrqk_agent': JITOptimizedStrategicActor(input_dim=6),
            'mmd_agent': JITOptimizedStrategicActor(input_dim=3),
            'critic': JITOptimizedCritic(state_dim=13)
        }
        
        for model in models.values():
            model.eval()
        
        return models
    
    def compile_models_jit(self, models: Dict[str, nn.Module]) -> Dict[str, torch.jit.ScriptModule]:
        """Compile models using TorchScript"""
        compiled_models = {}
        
        # Example inputs
        tactical_input = torch.randn(1, 60, 7)
        mlmi_input = torch.randn(1, 4)
        nwrqk_input = torch.randn(1, 6)
        mmd_input = torch.randn(1, 3)
        critic_input = torch.randn(1, 13)
        
        # Compile tactical system
        try:
            with torch.no_grad():
                traced_tactical = torch.jit.trace(models['tactical_system'], tactical_input)
                compiled_models['tactical_system'] = torch.jit.optimize_for_inference(traced_tactical)
        except Exception as e:
            logger.error(f"Failed to compile tactical system: {e}")
            compiled_models['tactical_system'] = models['tactical_system']
        
        # Compile strategic system  
        try:
            with torch.no_grad():
                traced_strategic = torch.jit.trace(models['strategic_system'], (mlmi_input, nwrqk_input, mmd_input))
                compiled_models['strategic_system'] = torch.jit.optimize_for_inference(traced_strategic)
        except Exception as e:
            logger.error(f"Failed to compile strategic system: {e}")
            compiled_models['strategic_system'] = models['strategic_system']
        
        # Compile individual agents
        agent_inputs = {
            'fvg_agent': tactical_input,
            'momentum_agent': tactical_input,
            'entry_agent': tactical_input,
            'mlmi_agent': mlmi_input,
            'nwrqk_agent': nwrqk_input,
            'mmd_agent': mmd_input,
            'critic': critic_input
        }
        
        for agent_name, agent_model in models.items():
            if agent_name in ['tactical_system', 'strategic_system']:
                continue
            
            try:
                with torch.no_grad():
                    traced_agent = torch.jit.trace(agent_model, agent_inputs[agent_name])
                    compiled_models[agent_name] = torch.jit.optimize_for_inference(traced_agent)
            except Exception as e:
                logger.error(f"Failed to compile {agent_name}: {e}")
                compiled_models[agent_name] = agent_model
        
        return compiled_models
    
    def benchmark_models(self, models: Dict[str, Any], num_iterations: int = 1000) -> Dict[str, PerformanceMetrics]:
        """Benchmark model performance"""
        results = {}
        
        # Test inputs
        tactical_input = torch.randn(1, 60, 7)
        mlmi_input = torch.randn(1, 4)
        nwrqk_input = torch.randn(1, 6)
        mmd_input = torch.randn(1, 3)
        critic_input = torch.randn(1, 13)
        
        for model_name, model in models.items():
            # Determine input type
            if 'tactical' in model_name or model_name in ['fvg_agent', 'momentum_agent', 'entry_agent']:
                test_input = tactical_input
            elif model_name == 'strategic_system':
                test_input = (mlmi_input, nwrqk_input, mmd_input)
            elif model_name == 'mlmi_agent':
                test_input = mlmi_input
            elif model_name == 'nwrqk_agent':
                test_input = nwrqk_input
            elif model_name == 'mmd_agent':
                test_input = mmd_input
            else:  # critic
                test_input = critic_input
            
            # Warm up
            with torch.no_grad():
                for _ in range(20):
                    if isinstance(test_input, tuple):
                        _ = model(*test_input)
                    else:
                        _ = model(test_input)
            
            # Benchmark
            latencies = []
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    start_time = time.perf_counter()
                    
                    if isinstance(test_input, tuple):
                        _ = model(*test_input)
                    else:
                        _ = model(test_input)
                    
                    end_time = time.perf_counter()
                    latencies.append((end_time - start_time) * 1000)
            
            # Calculate model size
            model_size = sum(p.numel() * 4 for p in model.parameters() if hasattr(model, 'parameters')) / (1024 * 1024)
            
            # Calculate metrics
            avg_latency = np.mean(latencies)
            p99_latency = np.percentile(latencies, 99)
            
            results[model_name] = PerformanceMetrics(
                avg_latency_ms=avg_latency,
                p99_latency_ms=p99_latency,
                throughput_qps=1000 / avg_latency,
                memory_usage_mb=psutil.virtual_memory().percent,
                cpu_usage_percent=psutil.cpu_percent(),
                meets_target=p99_latency < 1.0,
                model_size_mb=model_size
            )
        
        return results
    
    def save_models(self, models: Dict[str, Any], save_dir: Path = Path("models/optimized")):
        """Save optimized models"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in models.items():
            model_path = save_dir / f"{model_name}.pt"
            torch.jit.save(model, str(model_path))
            logger.info(f"Saved {model_name} to {model_path}")
        
        # Save performance results
        if self.performance_results:
            results_path = save_dir / "performance_results.json"
            with open(results_path, 'w') as f:
                results_dict = {}
                for model_name, metrics in self.performance_results.items():
                    results_dict[model_name] = metrics.to_dict()
                json.dump(results_dict, f, indent=2)
            logger.info(f"Saved performance results to {results_path}")

async def main():
    """Main function to start distributed inference system"""
    print("🚀 DISTRIBUTED INFERENCE SYSTEM STARTING")
    print("=" * 60)
    
    # Create and optimize models
    optimizer = PerformanceOptimizer()
    models = optimizer.create_optimized_models()
    compiled_models = optimizer.compile_models_jit(models)
    
    # Benchmark models
    results = optimizer.benchmark_models(compiled_models)
    optimizer.performance_results = results
    
    # Save models
    optimizer.save_models(compiled_models)
    
    # Start inference server
    server = DistributedInferenceServer(port=8080, num_workers=4)
    server.load_models_from_directory(Path("models/optimized"))
    
    # Performance summary
    print("\n🎯 PERFORMANCE SUMMARY")
    print("-" * 40)
    
    for model_name, metrics in results.items():
        status = "✅ READY" if metrics.meets_target else "❌ SLOW"
        print(f"{model_name:20}: {metrics.p99_latency_ms:5.2f}ms {status}")
    
    print("\n🌐 STARTING SERVER...")
    await server.start_server()
    
    # Keep server running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")

if __name__ == "__main__":
    asyncio.run(main())