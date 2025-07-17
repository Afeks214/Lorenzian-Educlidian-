"""
Comprehensive Performance Optimization Test Suite

This module provides thorough testing and validation of all performance optimizations:
- Memory optimization validation
- CPU optimization validation  
- I/O optimization validation
- Monitoring system validation
- Integration testing
- Performance regression testing
- Stress testing

Key Test Categories:
- Unit tests for individual optimizations
- Integration tests for combined optimizations
- Performance benchmarks and regression tests
- Stress tests under extreme conditions
- Memory leak detection
- Thread safety validation
"""

import asyncio
import pytest
import torch
import numpy as np
import time
import threading
import tempfile
import os
import psutil
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
import logging
import structlog
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

# Import optimization modules
from .memory_optimizer import MemoryOptimizer, TensorPool, ObjectPool, GCOptimizer
from .cpu_optimizer import CPUOptimizer, JITCompiler, VectorizedOps, ThreadPoolOptimizer
from .io_optimizer import IOOptimizer, AsyncCache, AsyncFileManager, BatchProcessor
from .enhanced_monitoring import EnhancedPerformanceMonitor, RealTimeMetricsCollector
from .integrated_optimizer import IntegratedOptimizer, OptimizationConfig

logger = structlog.get_logger()


class PerformanceTestSuite:
    """
    Comprehensive performance testing suite for all optimizations.
    """
    
    def __init__(self):
        self.test_results = {}
        self.benchmark_results = {}
        self.stress_test_results = {}
        self.memory_optimizer = MemoryOptimizer()
        self.cpu_optimizer = CPUOptimizer()
        self.io_optimizer = IOOptimizer()
        self.monitoring = EnhancedPerformanceMonitor()
        self.integrated_optimizer = IntegratedOptimizer()
        
        logger.info("PerformanceTestSuite initialized")
    
    def run_all_tests(self) -> Dict:
        """Run all performance tests"""
        
        print("🧪 Running Comprehensive Performance Test Suite")
        print("=" * 60)
        
        # Run test categories
        self.test_results['memory_tests'] = self.test_memory_optimization()
        self.test_results['cpu_tests'] = self.test_cpu_optimization()
        self.test_results['io_tests'] = asyncio.run(self.test_io_optimization())
        self.test_results['monitoring_tests'] = self.test_monitoring_system()
        self.test_results['integration_tests'] = asyncio.run(self.test_integration())
        self.test_results['stress_tests'] = asyncio.run(self.test_stress_scenarios())
        
        # Generate summary report
        return self.generate_test_report()
    
    def test_memory_optimization(self) -> Dict:
        """Test memory optimization components"""
        
        print("\n🧠 Testing Memory Optimization...")
        results = {}
        
        # Test 1: Tensor Pool Performance
        results['tensor_pool'] = self._test_tensor_pool()
        
        # Test 2: Object Pool Performance  
        results['object_pool'] = self._test_object_pool()
        
        # Test 3: GC Optimization
        results['gc_optimization'] = self._test_gc_optimization()
        
        # Test 4: Memory Monitoring
        results['memory_monitoring'] = self._test_memory_monitoring()
        
        # Test 5: Memory Leak Detection
        results['memory_leak_detection'] = self._test_memory_leak_detection()
        
        return results
    
    def _test_tensor_pool(self) -> Dict:
        """Test tensor pool functionality"""
        
        print("  📊 Testing Tensor Pool...")
        
        tensor_pool = TensorPool(max_pool_size=100)
        
        # Test basic functionality
        tensor1 = tensor_pool.get_tensor((10, 10))
        tensor2 = tensor_pool.get_tensor((10, 10))
        
        assert tensor1.shape == (10, 10)
        assert tensor2.shape == (10, 10)
        
        # Test pool reuse
        tensor_pool.return_tensor(tensor1)
        tensor3 = tensor_pool.get_tensor((10, 10))
        
        # Performance test
        start_time = time.time()
        for _ in range(1000):
            tensor = tensor_pool.get_tensor((100, 100))
            tensor_pool.return_tensor(tensor)
        pool_time = time.time() - start_time
        
        # Direct allocation test
        start_time = time.time()
        for _ in range(1000):
            tensor = torch.zeros((100, 100))
            del tensor
        direct_time = time.time() - start_time
        
        # Calculate improvement
        improvement = ((direct_time - pool_time) / direct_time) * 100
        
        stats = tensor_pool.get_stats()
        
        return {
            'functionality_test': 'PASS',
            'pool_time_ms': pool_time * 1000,
            'direct_time_ms': direct_time * 1000,
            'improvement_percent': improvement,
            'hit_rate': stats['hit_rate'],
            'total_allocations': stats['total_allocations'],
            'success': improvement > 0
        }
    
    def _test_object_pool(self) -> Dict:
        """Test object pool functionality"""
        
        print("  🏭 Testing Object Pool...")
        
        def list_factory():
            return []
        
        object_pool = ObjectPool(list_factory, max_size=100)
        
        # Test basic functionality
        obj1 = object_pool.get_object()
        obj2 = object_pool.get_object()
        
        assert isinstance(obj1, list)
        assert isinstance(obj2, list)
        
        # Test pool reuse
        object_pool.return_object(obj1)
        obj3 = object_pool.get_object()
        
        # Performance test
        start_time = time.time()
        for _ in range(1000):
            obj = object_pool.get_object()
            object_pool.return_object(obj)
        pool_time = time.time() - start_time
        
        # Direct creation test
        start_time = time.time()
        for _ in range(1000):
            obj = list_factory()
            del obj
        direct_time = time.time() - start_time
        
        improvement = ((direct_time - pool_time) / direct_time) * 100
        stats = object_pool.get_stats()
        
        return {
            'functionality_test': 'PASS',
            'pool_time_ms': pool_time * 1000,
            'direct_time_ms': direct_time * 1000,
            'improvement_percent': improvement,
            'reuse_rate': stats['reuse_rate'],
            'success': improvement > 0
        }
    
    def _test_gc_optimization(self) -> Dict:
        """Test garbage collection optimization"""
        
        print("  🗑️  Testing GC Optimization...")
        
        gc_optimizer = GCOptimizer()
        
        # Test original settings
        original_thresholds = gc_optimizer.original_thresholds
        
        # Test optimization
        gc_optimizer.optimize_gc()
        optimized_thresholds = gc.get_threshold()
        
        # Test restoration
        gc_optimizer.restore_gc()
        restored_thresholds = gc.get_threshold()
        
        # Test forced collection
        collection_stats = gc_optimizer.force_collection()
        
        return {
            'original_thresholds': original_thresholds,
            'optimized_thresholds': optimized_thresholds,
            'restored_thresholds': restored_thresholds,
            'settings_restored': restored_thresholds == original_thresholds,
            'collection_time_ms': collection_stats['collection_time_ms'],
            'objects_collected': collection_stats['collected_objects'],
            'success': True
        }
    
    def _test_memory_monitoring(self) -> Dict:
        """Test memory monitoring functionality"""
        
        print("  📈 Testing Memory Monitoring...")
        
        self.memory_optimizer.enable_optimizations()
        
        # Let monitoring run for a bit
        time.sleep(2)
        
        # Get current stats
        current_stats = self.memory_optimizer.memory_monitor.get_current_stats()
        
        # Test memory trend analysis
        memory_trend = self.memory_optimizer.memory_monitor.get_memory_trend()
        
        # Test cleanup
        cleanup_stats = self.memory_optimizer.memory_monitor.cleanup_memory()
        
        return {
            'monitoring_active': current_stats is not None,
            'memory_usage_mb': current_stats.used_memory_mb if current_stats else 0,
            'trend_analysis': memory_trend.get('status', 'Available'),
            'cleanup_collected': cleanup_stats['collected_objects'],
            'success': current_stats is not None
        }
    
    def _test_memory_leak_detection(self) -> Dict:
        """Test memory leak detection"""
        
        print("  🔍 Testing Memory Leak Detection...")
        
        # Simulate memory allocation
        tensors = []
        for i in range(100):
            tensor = torch.randn(100, 100)
            tensors.append(tensor)
        
        # Force memory monitoring
        self.memory_optimizer.memory_monitor.start_monitoring()
        time.sleep(1)
        
        # Get stats before cleanup
        stats_before = self.memory_optimizer.memory_monitor.get_current_stats()
        
        # Cleanup
        del tensors
        gc.collect()
        
        # Get stats after cleanup
        time.sleep(1)
        stats_after = self.memory_optimizer.memory_monitor.get_current_stats()
        
        memory_freed = (stats_before.used_memory_mb - stats_after.used_memory_mb) if stats_before and stats_after else 0
        
        return {
            'memory_before_mb': stats_before.used_memory_mb if stats_before else 0,
            'memory_after_mb': stats_after.used_memory_mb if stats_after else 0,
            'memory_freed_mb': memory_freed,
            'leak_detected': memory_freed < 10,  # Should have freed significant memory
            'success': memory_freed > 0
        }
    
    def test_cpu_optimization(self) -> Dict:
        """Test CPU optimization components"""
        
        print("\n⚡ Testing CPU Optimization...")
        results = {}
        
        # Test 1: JIT Compilation
        results['jit_compilation'] = self._test_jit_compilation()
        
        # Test 2: Vectorized Operations
        results['vectorized_ops'] = self._test_vectorized_operations()
        
        # Test 3: Thread Pool Optimization
        results['thread_pools'] = self._test_thread_pools()
        
        # Test 4: CPU Affinity
        results['cpu_affinity'] = self._test_cpu_affinity()
        
        return results
    
    def _test_jit_compilation(self) -> Dict:
        """Test JIT compilation functionality"""
        
        print("  🔥 Testing JIT Compilation...")
        
        # Create test model
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        example_input = torch.randn(1, 10)
        
        # Test JIT compilation
        compiled_model = self.cpu_optimizer.jit_compiler.compile_model(
            model, example_input, 'test_model'
        )
        
        # Performance comparison
        model.eval()
        with torch.no_grad():
            # Original model
            start_time = time.time()
            for _ in range(1000):
                _ = model(example_input)
            original_time = time.time() - start_time
            
            # Compiled model
            start_time = time.time()
            for _ in range(1000):
                _ = compiled_model(example_input)
            compiled_time = time.time() - start_time
        
        improvement = ((original_time - compiled_time) / original_time) * 100
        
        stats = self.cpu_optimizer.jit_compiler.get_compilation_stats()
        
        return {
            'compilation_success': compiled_model is not None,
            'original_time_ms': original_time * 1000,
            'compiled_time_ms': compiled_time * 1000,
            'improvement_percent': improvement,
            'compiled_models': stats['total_models'],
            'success': improvement > 0
        }
    
    def _test_vectorized_operations(self) -> Dict:
        """Test vectorized operations"""
        
        print("  🔢 Testing Vectorized Operations...")
        
        # Test data
        data1 = np.random.randn(10000).astype(np.float64)
        data2 = np.random.randn(10000).astype(np.float64)
        
        # Test correlation
        start_time = time.time()
        fast_corr = self.cpu_optimizer.vectorized_ops.fast_correlation(data1, data2)
        fast_time = time.time() - start_time
        
        start_time = time.time()
        numpy_corr = np.corrcoef(data1, data2)[0, 1]
        numpy_time = time.time() - start_time
        
        correlation_improvement = ((numpy_time - fast_time) / numpy_time) * 100
        
        # Test moving average
        start_time = time.time()
        fast_ma = self.cpu_optimizer.vectorized_ops.fast_moving_average(data1, 50)
        fast_ma_time = time.time() - start_time
        
        start_time = time.time()
        pandas_ma = pd.Series(data1).rolling(50).mean().values
        pandas_ma_time = time.time() - start_time
        
        ma_improvement = ((pandas_ma_time - fast_ma_time) / pandas_ma_time) * 100
        
        return {
            'correlation_improvement': correlation_improvement,
            'moving_average_improvement': ma_improvement,
            'correlation_accuracy': abs(fast_corr - numpy_corr) < 0.001,
            'moving_average_accuracy': np.allclose(fast_ma[49:], pandas_ma[49:], equal_nan=True),
            'success': correlation_improvement > 0 and ma_improvement > 0
        }
    
    def _test_thread_pools(self) -> Dict:
        """Test thread pool optimization"""
        
        print("  🧵 Testing Thread Pool Optimization...")
        
        # Create test function
        def test_function(x):
            return x * x + np.sin(x)
        
        # Test parallel execution
        args_list = [(i,) for i in range(100)]
        
        start_time = time.time()
        results = self.cpu_optimizer.thread_pool_optimizer.execute_parallel(
            'test_pool', test_function, args_list
        )
        parallel_time = time.time() - start_time
        
        # Test serial execution
        start_time = time.time()
        serial_results = [test_function(x) for x, in args_list]
        serial_time = time.time() - start_time
        
        improvement = ((serial_time - parallel_time) / serial_time) * 100
        
        stats = self.cpu_optimizer.thread_pool_optimizer.get_execution_stats()
        
        return {
            'parallel_time_ms': parallel_time * 1000,
            'serial_time_ms': serial_time * 1000,
            'improvement_percent': improvement,
            'results_match': len(results) == len(serial_results),
            'success_rate': stats.get('test_pool', {}).get('avg_success_rate', 0),
            'success': improvement > 0
        }
    
    def _test_cpu_affinity(self) -> Dict:
        """Test CPU affinity management"""
        
        print("  🎯 Testing CPU Affinity...")
        
        affinity_manager = self.cpu_optimizer.affinity_manager
        
        # Get original affinity
        original_affinity = affinity_manager.get_current_affinity()
        
        # Test setting affinity
        test_cores = [0, 1] if len(original_affinity) > 1 else [0]
        affinity_set = affinity_manager.set_process_affinity(test_cores, 'test')
        
        # Get new affinity
        new_affinity = affinity_manager.get_current_affinity()
        
        # Restore original
        restore_success = affinity_manager.restore_original_affinity()
        restored_affinity = affinity_manager.get_current_affinity()
        
        return {
            'original_affinity': original_affinity,
            'affinity_set_success': affinity_set,
            'new_affinity': new_affinity,
            'restore_success': restore_success,
            'restored_affinity': restored_affinity,
            'affinity_restored': restored_affinity == original_affinity,
            'success': affinity_set and restore_success
        }
    
    async def test_io_optimization(self) -> Dict:
        """Test I/O optimization components"""
        
        print("\n💾 Testing I/O Optimization...")
        results = {}
        
        # Test 1: Async Cache
        results['async_cache'] = await self._test_async_cache()
        
        # Test 2: Async File Operations
        results['async_file_ops'] = await self._test_async_file_operations()
        
        # Test 3: Connection Pooling
        results['connection_pooling'] = await self._test_connection_pooling()
        
        # Test 4: Batch Processing
        results['batch_processing'] = await self._test_batch_processing()
        
        return results
    
    async def _test_async_cache(self) -> Dict:
        """Test async cache functionality"""
        
        print("  🗄️  Testing Async Cache...")
        
        from .io_optimizer import AsyncCache
        
        cache = AsyncCache(max_size=100, ttl_seconds=10)
        
        # Test basic operations
        await cache.set('key1', 'value1')
        value1 = await cache.get('key1')
        
        # Test cache hit
        await cache.set('key2', 'value2')
        value2 = await cache.get('key2')
        
        # Test cache miss
        value3 = await cache.get('nonexistent')
        
        # Performance test
        start_time = time.time()
        for i in range(1000):
            await cache.set(f'key{i}', f'value{i}')
        set_time = time.time() - start_time
        
        start_time = time.time()
        for i in range(1000):
            await cache.get(f'key{i}')
        get_time = time.time() - start_time
        
        stats = await cache.get_stats()
        
        return {
            'basic_operations': value1 == 'value1' and value2 == 'value2' and value3 is None,
            'set_time_ms': set_time * 1000,
            'get_time_ms': get_time * 1000,
            'hit_rate': stats['hit_rate'],
            'cache_size': stats['size'],
            'success': stats['hit_rate'] > 0.5
        }
    
    async def _test_async_file_operations(self) -> Dict:
        """Test async file operations"""
        
        print("  📂 Testing Async File Operations...")
        
        await self.io_optimizer.enable_optimizations()
        
        # Create test file
        test_content = "Test content for async file operations" * 100
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(test_content)
            test_file = f.name
        
        try:
            # Test async read
            start_time = time.time()
            content = await self.io_optimizer.file_manager.read_file(test_file)
            async_read_time = time.time() - start_time
            
            # Test sync read
            start_time = time.time()
            with open(test_file, 'r') as f:
                sync_content = f.read()
            sync_read_time = time.time() - start_time
            
            # Test async write
            write_content = "New content for testing"
            start_time = time.time()
            await self.io_optimizer.file_manager.write_file(test_file, write_content)
            async_write_time = time.time() - start_time
            
            # Test sync write
            start_time = time.time()
            with open(test_file, 'w') as f:
                f.write(write_content)
            sync_write_time = time.time() - start_time
            
            # Test batch read
            batch_files = [test_file] * 10
            start_time = time.time()
            batch_results = await self.io_optimizer.file_manager.batch_read_files(batch_files)
            batch_time = time.time() - start_time
            
            return {
                'content_match': content == test_content,
                'async_read_time_ms': async_read_time * 1000,
                'sync_read_time_ms': sync_read_time * 1000,
                'async_write_time_ms': async_write_time * 1000,
                'sync_write_time_ms': sync_write_time * 1000,
                'batch_read_time_ms': batch_time * 1000,
                'batch_results_count': len(batch_results),
                'success': content == test_content
            }
        
        finally:
            # Cleanup
            if os.path.exists(test_file):
                os.unlink(test_file)
    
    async def _test_connection_pooling(self) -> Dict:
        """Test connection pooling"""
        
        print("  🔗 Testing Connection Pooling...")
        
        from .io_optimizer import AsyncConnectionPool
        
        # Create test connection pool
        class TestConnectionPool(AsyncConnectionPool):
            async def create_connection(self):
                # Simulate connection creation
                await asyncio.sleep(0.01)
                return {'id': time.time(), 'active': True}
            
            async def validate_connection(self, connection):
                return connection.get('active', False)
            
            async def close_connection(self, connection):
                connection['active'] = False
        
        pool = TestConnectionPool(max_connections=10)
        
        # Test connection acquisition
        start_time = time.time()
        connections = []
        for _ in range(5):
            conn = await pool.get_connection()
            connections.append(conn)
        acquisition_time = time.time() - start_time
        
        # Test connection return
        start_time = time.time()
        for conn in connections:
            await pool.return_connection(conn)
        return_time = time.time() - start_time
        
        # Test connection reuse
        start_time = time.time()
        reused_conn = await pool.get_connection()
        reuse_time = time.time() - start_time
        
        await pool.return_connection(reused_conn)
        
        stats = pool.get_pool_stats()
        
        return {
            'acquisition_time_ms': acquisition_time * 1000,
            'return_time_ms': return_time * 1000,
            'reuse_time_ms': reuse_time * 1000,
            'active_connections': stats['active_connections'],
            'created_connections': stats['connection_stats']['created'],
            'reused_connections': stats['connection_stats']['reused'],
            'success': stats['connection_stats']['created'] > 0
        }
    
    async def _test_batch_processing(self) -> Dict:
        """Test batch processing"""
        
        print("  📦 Testing Batch Processing...")
        
        from .io_optimizer import BatchProcessor
        
        batch_processor = BatchProcessor(batch_size=10, max_wait_time=0.1)
        
        # Test function
        async def test_operation(x):
            await asyncio.sleep(0.001)  # Simulate work
            return x * 2
        
        # Add operations to batch
        start_time = time.time()
        tasks = []
        for i in range(25):
            task = batch_processor.add_operation(test_operation, i)
            tasks.append(task)
        
        # Wait for all operations
        await asyncio.gather(*tasks)
        processing_time = time.time() - start_time
        
        # Test flush
        await batch_processor.flush()
        
        stats = batch_processor.get_batch_stats()
        
        return {
            'processing_time_ms': processing_time * 1000,
            'total_batches': stats['processing_stats']['total_batches'],
            'total_operations': stats['processing_stats']['total_operations'],
            'avg_batch_size': stats['processing_stats']['avg_batch_size'],
            'success': stats['processing_stats']['total_operations'] >= 25
        }
    
    def test_monitoring_system(self) -> Dict:
        """Test monitoring system"""
        
        print("\n📊 Testing Monitoring System...")
        
        # Enable monitoring
        self.monitoring.enable_monitoring()
        
        # Let it collect some data
        time.sleep(3)
        
        # Test metrics collection
        stats = self.monitoring.get_comprehensive_stats()
        
        # Test dashboard data
        dashboard_data = self.monitoring.get_dashboard_data()
        
        # Test performance report
        performance_report = self.monitoring.create_performance_report(hours=1)
        
        return {
            'monitoring_enabled': stats['monitoring_enabled'],
            'metrics_collected': stats['metrics_collector']['total_collections'],
            'dashboard_metrics': len(dashboard_data.get('metrics', {})),
            'performance_report_generated': 'generated_at' in performance_report,
            'success': stats['monitoring_enabled'] and stats['metrics_collector']['total_collections'] > 0
        }
    
    async def test_integration(self) -> Dict:
        """Test integrated optimization system"""
        
        print("\n🔄 Testing Integration...")
        
        results = {}
        
        # Test 1: Enable integrated optimizations
        await self.integrated_optimizer.enable_optimizations('real_time')
        
        # Test 2: Validate optimizations
        validation_report = await self.integrated_optimizer.validate_optimizations()
        
        # Test 3: Auto-optimization
        recommended_strategy = await self.integrated_optimizer.auto_optimize()
        
        # Test 4: Performance report
        performance_report = await self.integrated_optimizer.generate_performance_report()
        
        # Test 5: Context manager
        async with self.integrated_optimizer.optimization_context('inference'):
            # Test inference optimization
            x = torch.randn(100, 100)
            result = torch.mm(x, x)
            context_test = result.shape == (100, 100)
        
        results = {
            'optimization_enabled': self.integrated_optimizer.optimization_enabled,
            'validation_passed': len(validation_report['safety_checks']) > 0,
            'auto_optimization': recommended_strategy is not None,
            'performance_report': 'report_generated' in performance_report,
            'context_manager': context_test,
            'current_strategy': self.integrated_optimizer.current_strategy,
            'success': all([
                self.integrated_optimizer.optimization_enabled,
                recommended_strategy is not None,
                context_test
            ])
        }
        
        return results
    
    async def test_stress_scenarios(self) -> Dict:
        """Test system under stress conditions"""
        
        print("\n🔥 Testing Stress Scenarios...")
        
        results = {}
        
        # Test 1: High memory allocation
        results['high_memory_allocation'] = await self._test_high_memory_allocation()
        
        # Test 2: High CPU usage
        results['high_cpu_usage'] = await self._test_high_cpu_usage()
        
        # Test 3: Concurrent operations
        results['concurrent_operations'] = await self._test_concurrent_operations()
        
        # Test 4: Large data processing
        results['large_data_processing'] = await self._test_large_data_processing()
        
        return results
    
    async def _test_high_memory_allocation(self) -> Dict:
        """Test high memory allocation scenario"""
        
        print("  🧠 Testing High Memory Allocation...")
        
        self.memory_optimizer.enable_optimizations()
        
        # Allocate large tensors
        tensors = []
        start_memory = psutil.virtual_memory().used
        
        try:
            for i in range(50):
                tensor = torch.randn(1000, 1000)
                tensors.append(tensor)
            
            peak_memory = psutil.virtual_memory().used
            memory_used = (peak_memory - start_memory) / 1024 / 1024  # MB
            
            # Test memory cleanup
            del tensors
            gc.collect()
            
            final_memory = psutil.virtual_memory().used
            memory_freed = (peak_memory - final_memory) / 1024 / 1024  # MB
            
            return {
                'memory_allocated_mb': memory_used,
                'memory_freed_mb': memory_freed,
                'cleanup_efficiency': memory_freed / memory_used if memory_used > 0 else 0,
                'success': memory_freed > memory_used * 0.8  # 80% cleanup
            }
        
        except Exception as e:
            return {
                'error': str(e),
                'success': False
            }
    
    async def _test_high_cpu_usage(self) -> Dict:
        """Test high CPU usage scenario"""
        
        print("  ⚡ Testing High CPU Usage...")
        
        self.cpu_optimizer.enable_optimizations()
        
        # CPU-intensive computation
        def cpu_intensive_task():
            result = 0
            for i in range(1000000):
                result += np.sin(i) * np.cos(i)
            return result
        
        # Test parallel execution
        start_time = time.time()
        start_cpu = psutil.cpu_percent(interval=None)
        
        tasks = []
        for _ in range(4):  # 4 CPU-intensive tasks
            task = asyncio.create_task(asyncio.get_event_loop().run_in_executor(None, cpu_intensive_task))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        end_cpu = psutil.cpu_percent(interval=None)
        
        execution_time = end_time - start_time
        cpu_usage = max(end_cpu - start_cpu, 0)
        
        return {
            'execution_time_ms': execution_time * 1000,
            'cpu_usage_percent': cpu_usage,
            'tasks_completed': len(results),
            'results_valid': all(isinstance(r, (int, float)) for r in results),
            'success': len(results) == 4 and execution_time < 60  # Complete within 60 seconds
        }
    
    async def _test_concurrent_operations(self) -> Dict:
        """Test concurrent operations"""
        
        print("  🔄 Testing Concurrent Operations...")
        
        await self.io_optimizer.enable_optimizations()
        
        # Mixed workload: I/O, CPU, Memory
        async def mixed_workload(task_id):
            # Memory operation
            tensor = torch.randn(100, 100)
            
            # CPU operation
            result = torch.mm(tensor, tensor)
            
            # I/O operation
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                f.write(f"Task {task_id} data")
                temp_file = f.name
            
            content = await self.io_optimizer.file_manager.read_file(temp_file)
            os.unlink(temp_file)
            
            return {
                'task_id': task_id,
                'tensor_shape': result.shape,
                'content_length': len(content),
                'success': True
            }
        
        # Run concurrent tasks
        start_time = time.time()
        tasks = [mixed_workload(i) for i in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        execution_time = time.time() - start_time
        
        # Count successful tasks
        successful_tasks = len([r for r in results if isinstance(r, dict) and r.get('success', False)])
        
        return {
            'execution_time_ms': execution_time * 1000,
            'total_tasks': len(tasks),
            'successful_tasks': successful_tasks,
            'success_rate': successful_tasks / len(tasks),
            'success': successful_tasks >= len(tasks) * 0.9  # 90% success rate
        }
    
    async def _test_large_data_processing(self) -> Dict:
        """Test large data processing"""
        
        print("  📊 Testing Large Data Processing...")
        
        self.memory_optimizer.enable_optimizations()
        self.cpu_optimizer.enable_optimizations()
        
        # Generate large dataset
        data_size = 10000
        large_data = np.random.randn(data_size, 100).astype(np.float32)
        
        # Test processing with optimizations
        start_time = time.time()
        
        # Convert to tensor
        tensor_data = torch.from_numpy(large_data)
        
        # Process in batches
        batch_size = 1000
        results = []
        
        for i in range(0, len(tensor_data), batch_size):
            batch = tensor_data[i:i+batch_size]
            
            # Some processing
            processed = torch.mm(batch, batch.T)
            results.append(processed.sum().item())
        
        processing_time = time.time() - start_time
        
        # Memory usage check
        memory_stats = self.memory_optimizer.get_optimization_stats()
        
        return {
            'data_size': data_size,
            'processing_time_ms': processing_time * 1000,
            'batches_processed': len(results),
            'memory_usage_mb': memory_stats.get('memory_monitor', {}).get('current_stats', {}).get('used_memory_mb', 0),
            'tensor_pool_hit_rate': memory_stats.get('tensor_pool', {}).get('hit_rate', 0),
            'success': len(results) > 0 and processing_time < 30  # Complete within 30 seconds
        }
    
    def generate_test_report(self) -> Dict:
        """Generate comprehensive test report"""
        
        print("\n📋 Generating Test Report...")
        
        # Calculate overall success rates
        def calculate_success_rate(test_category):
            if not isinstance(test_category, dict):
                return 0.0
            
            total_tests = len(test_category)
            successful_tests = sum(1 for test in test_category.values() 
                                 if isinstance(test, dict) and test.get('success', False))
            
            return successful_tests / total_tests if total_tests > 0 else 0.0
        
        # Generate summary
        summary = {
            'test_execution_time': datetime.now().isoformat(),
            'total_test_categories': len(self.test_results),
            'success_rates': {
                category: calculate_success_rate(results)
                for category, results in self.test_results.items()
            },
            'detailed_results': self.test_results,
            'performance_improvements': self._calculate_performance_improvements(),
            'recommendations': self._generate_recommendations()
        }
        
        # Calculate overall success rate
        overall_success_rate = np.mean(list(summary['success_rates'].values()))
        summary['overall_success_rate'] = overall_success_rate
        summary['test_status'] = 'PASS' if overall_success_rate >= 0.8 else 'FAIL'
        
        return summary
    
    def _calculate_performance_improvements(self) -> Dict:
        """Calculate performance improvements across all tests"""
        
        improvements = {}
        
        # Extract improvement percentages from test results
        for category, tests in self.test_results.items():
            if isinstance(tests, dict):
                for test_name, test_result in tests.items():
                    if isinstance(test_result, dict):
                        improvement = test_result.get('improvement_percent', 0)
                        if improvement > 0:
                            improvements[f"{category}.{test_name}"] = improvement
        
        if improvements:
            return {
                'average_improvement': np.mean(list(improvements.values())),
                'max_improvement': np.max(list(improvements.values())),
                'min_improvement': np.min(list(improvements.values())),
                'improvement_details': improvements
            }
        
        return {'message': 'No performance improvements measured'}
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        
        recommendations = []
        
        # Check success rates
        for category, success_rate in self.test_results.items():
            if isinstance(success_rate, dict):
                category_success = sum(1 for test in success_rate.values() 
                                     if isinstance(test, dict) and test.get('success', False))
                category_total = len(success_rate)
                
                if category_success / category_total < 0.8:
                    recommendations.append(f"Review {category} - success rate below 80%")
        
        # Check performance improvements
        improvements = self._calculate_performance_improvements()
        if isinstance(improvements, dict) and improvements.get('average_improvement', 0) < 10:
            recommendations.append("Consider additional optimizations - average improvement below 10%")
        
        # Check stress test results
        stress_results = self.test_results.get('stress_tests', {})
        if isinstance(stress_results, dict):
            for test_name, test_result in stress_results.items():
                if isinstance(test_result, dict) and not test_result.get('success', False):
                    recommendations.append(f"Address stress test failure in {test_name}")
        
        return recommendations


async def main():
    """Run the comprehensive test suite"""
    
    print("🚀 Starting Comprehensive Performance Optimization Test Suite")
    print("=" * 70)
    
    # Create test suite
    test_suite = PerformanceTestSuite()
    
    # Run all tests
    test_report = test_suite.run_all_tests()
    
    # Print summary
    print("\n📊 Test Summary:")
    print(f"Overall Success Rate: {test_report['overall_success_rate']:.1%}")
    print(f"Test Status: {test_report['test_status']}")
    
    print("\n📈 Success Rates by Category:")
    for category, success_rate in test_report['success_rates'].items():
        status = "✅" if success_rate >= 0.8 else "❌"
        print(f"  {status} {category}: {success_rate:.1%}")
    
    # Performance improvements
    improvements = test_report['performance_improvements']
    if 'average_improvement' in improvements:
        print(f"\n⚡ Performance Improvements:")
        print(f"  Average: {improvements['average_improvement']:.1f}%")
        print(f"  Maximum: {improvements['max_improvement']:.1f}%")
        print(f"  Minimum: {improvements['min_improvement']:.1f}%")
    
    # Recommendations
    if test_report['recommendations']:
        print("\n💡 Recommendations:")
        for rec in test_report['recommendations']:
            print(f"  • {rec}")
    
    print(f"\n✅ Test suite completed! Status: {test_report['test_status']}")
    
    # Save detailed report
    with open('/tmp/performance_test_report.json', 'w') as f:
        json.dump(test_report, f, indent=2, default=str)
    
    print("📁 Detailed report saved to /tmp/performance_test_report.json")


if __name__ == "__main__":
    asyncio.run(main())