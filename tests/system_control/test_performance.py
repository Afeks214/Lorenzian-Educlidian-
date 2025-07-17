"""
Performance impact tests for the Trading System Controller.

This module tests the performance characteristics of the master switch system,
including latency, throughput, resource usage, and scalability under load.
"""

import pytest
import time
import threading
import asyncio
import psutil
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch
import tempfile
import os

from src.core.trading_system_controller import (
    TradingSystemController,
    SystemState,
    ComponentStatus
)


class TestPerformanceMetrics:
    """Test suite for performance metrics and benchmarking."""
    
    @pytest.fixture
    def controller(self):
        """Create controller for performance testing."""
        controller = TradingSystemController(
            max_concurrent_operations=100,
            heartbeat_timeout=5.0
        )
        yield controller
        controller.shutdown(timeout=10.0)
    
    def test_startup_latency(self, controller):
        """Test system startup latency."""
        # Register components
        for i in range(10):
            controller.register_component(f"component_{i}", health_check_interval=1.0)
        
        # Measure startup time
        start_times = []
        for _ in range(10):
            start_time = time.perf_counter()
            result = controller.start_system(timeout=10.0)
            end_time = time.perf_counter()
            
            assert result is True
            start_times.append(end_time - start_time)
            
            # Stop system for next iteration
            controller.stop_system(timeout=10.0)
        
        # Analyze startup performance
        avg_startup_time = statistics.mean(start_times)
        max_startup_time = max(start_times)
        min_startup_time = min(start_times)
        
        # Performance assertions
        assert avg_startup_time < 1.0, f"Average startup time too high: {avg_startup_time:.3f}s"
        assert max_startup_time < 2.0, f"Maximum startup time too high: {max_startup_time:.3f}s"
        assert min_startup_time > 0.001, f"Minimum startup time suspiciously low: {min_startup_time:.3f}s"
        
        print(f"Startup latency - Avg: {avg_startup_time:.3f}s, Min: {min_startup_time:.3f}s, Max: {max_startup_time:.3f}s")
    
    def test_shutdown_latency(self, controller):
        """Test system shutdown latency."""
        # Register components
        for i in range(10):
            controller.register_component(f"component_{i}", health_check_interval=1.0)
        
        # Measure shutdown times
        shutdown_times = []
        for _ in range(10):
            controller.start_system(timeout=10.0)
            
            start_time = time.perf_counter()
            result = controller.stop_system(timeout=10.0)
            end_time = time.perf_counter()
            
            assert result is True
            shutdown_times.append(end_time - start_time)
        
        # Analyze shutdown performance
        avg_shutdown_time = statistics.mean(shutdown_times)
        max_shutdown_time = max(shutdown_times)
        
        # Performance assertions
        assert avg_shutdown_time < 1.0, f"Average shutdown time too high: {avg_shutdown_time:.3f}s"
        assert max_shutdown_time < 2.0, f"Maximum shutdown time too high: {max_shutdown_time:.3f}s"
        
        print(f"Shutdown latency - Avg: {avg_shutdown_time:.3f}s, Max: {max_shutdown_time:.3f}s")
    
    def test_emergency_stop_latency(self, controller):
        """Test emergency stop latency."""
        # Register components
        for i in range(10):
            controller.register_component(f"component_{i}", health_check_interval=1.0)
        
        # Measure emergency stop times
        emergency_times = []
        for _ in range(10):
            controller.start_system(timeout=10.0)
            
            start_time = time.perf_counter()
            result = controller.emergency_stop(reason="performance_test")
            end_time = time.perf_counter()
            
            assert result is True
            emergency_times.append(end_time - start_time)
            
            # Reset failsafe for next iteration
            controller.reset_failsafe(force=True)
        
        # Analyze emergency stop performance
        avg_emergency_time = statistics.mean(emergency_times)
        max_emergency_time = max(emergency_times)
        
        # Emergency stop should be very fast
        assert avg_emergency_time < 0.1, f"Average emergency stop time too high: {avg_emergency_time:.3f}s"
        assert max_emergency_time < 0.5, f"Maximum emergency stop time too high: {max_emergency_time:.3f}s"
        
        print(f"Emergency stop latency - Avg: {avg_emergency_time:.3f}s, Max: {max_emergency_time:.3f}s")
    
    def test_component_registration_performance(self, controller):
        """Test component registration performance."""
        # Measure registration times for different numbers of components
        component_counts = [10, 50, 100, 500, 1000]
        
        for count in component_counts:
            start_time = time.perf_counter()
            
            for i in range(count):
                result = controller.register_component(f"perf_component_{i}", health_check_interval=10.0)
                assert result is True
            
            end_time = time.perf_counter()
            registration_time = end_time - start_time
            
            # Performance assertions
            time_per_component = registration_time / count
            assert time_per_component < 0.001, f"Registration time per component too high: {time_per_component:.6f}s"
            
            print(f"Registered {count} components in {registration_time:.3f}s ({time_per_component:.6f}s per component)")
            
            # Cleanup for next iteration
            for i in range(count):
                controller.unregister_component(f"perf_component_{i}")
    
    def test_component_status_update_performance(self, controller):
        """Test component status update performance."""
        # Register components
        component_count = 100
        for i in range(component_count):
            controller.register_component(f"status_component_{i}", health_check_interval=10.0)
        
        # Measure status update times
        update_times = []
        
        for _ in range(100):  # 100 update cycles
            start_time = time.perf_counter()
            
            for i in range(component_count):
                result = controller.update_component_status(f"status_component_{i}", ComponentStatus.HEALTHY)
                assert result is True
            
            end_time = time.perf_counter()
            update_times.append(end_time - start_time)
        
        # Analyze update performance
        avg_update_time = statistics.mean(update_times)
        time_per_update = avg_update_time / component_count
        
        # Performance assertions
        assert time_per_update < 0.001, f"Status update time per component too high: {time_per_update:.6f}s"
        
        print(f"Status update performance - Avg: {avg_update_time:.3f}s for {component_count} components")
    
    def test_state_transition_performance(self, controller):
        """Test state transition performance."""
        # Register component
        controller.register_component("test_component")
        
        # Measure state transition times
        transition_times = []
        
        for _ in range(50):  # 50 full cycles
            # Start
            start_time = time.perf_counter()
            result = controller.start_system(timeout=10.0)
            assert result is True
            
            # Pause
            result = controller.pause_system(timeout=10.0)
            assert result is True
            
            # Resume
            result = controller.resume_system(timeout=10.0)
            assert result is True
            
            # Stop
            result = controller.stop_system(timeout=10.0)
            assert result is True
            
            end_time = time.perf_counter()
            transition_times.append(end_time - start_time)
        
        # Analyze transition performance
        avg_cycle_time = statistics.mean(transition_times)
        
        # Performance assertions
        assert avg_cycle_time < 0.5, f"State transition cycle time too high: {avg_cycle_time:.3f}s"
        
        print(f"State transition cycle time - Avg: {avg_cycle_time:.3f}s")
    
    def test_concurrent_operation_performance(self, controller):
        """Test performance under concurrent operations."""
        # Register components
        for i in range(50):
            controller.register_component(f"concurrent_component_{i}", health_check_interval=5.0)
        
        controller.start_system(timeout=10.0)
        
        # Define concurrent operations
        def register_and_update():
            component_name = f"temp_component_{threading.get_ident()}"
            controller.register_component(component_name, health_check_interval=10.0)
            
            for _ in range(10):
                controller.update_component_status(component_name, ComponentStatus.HEALTHY)
                time.sleep(0.001)  # Small delay
            
            controller.unregister_component(component_name)
        
        # Measure concurrent performance
        start_time = time.perf_counter()
        
        # Use ThreadPoolExecutor for concurrent operations
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(register_and_update) for _ in range(100)]
            
            # Wait for all operations to complete
            for future in as_completed(futures):
                future.result()
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Performance assertions
        assert total_time < 10.0, f"Concurrent operations took too long: {total_time:.3f}s"
        
        print(f"Concurrent operations completed in {total_time:.3f}s")
    
    def test_memory_usage_performance(self, controller):
        """Test memory usage under load."""
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Register many components
        component_count = 1000
        for i in range(component_count):
            controller.register_component(f"memory_component_{i}", health_check_interval=30.0)
        
        # Start system
        controller.start_system(timeout=10.0)
        
        # Update all components multiple times
        for _ in range(10):
            for i in range(component_count):
                controller.update_component_status(f"memory_component_{i}", ComponentStatus.HEALTHY)
        
        # Measure memory usage after operations
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Performance assertions
        memory_per_component = memory_increase / component_count
        assert memory_per_component < 0.1, f"Memory usage per component too high: {memory_per_component:.3f}MB"
        
        print(f"Memory usage - Initial: {initial_memory:.2f}MB, Final: {final_memory:.2f}MB, Increase: {memory_increase:.2f}MB")
    
    def test_cpu_usage_performance(self, controller):
        """Test CPU usage under load."""
        # Register components
        for i in range(100):
            controller.register_component(f"cpu_component_{i}", health_check_interval=0.1)
        
        controller.start_system(timeout=10.0)
        
        # Start CPU monitoring
        process = psutil.Process(os.getpid())
        cpu_samples = []
        
        def monitor_cpu():
            for _ in range(50):  # Sample for 5 seconds
                cpu_percent = process.cpu_percent(interval=0.1)
                cpu_samples.append(cpu_percent)
        
        # Start monitoring in background
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Generate load
        for _ in range(100):
            for i in range(100):
                controller.update_component_status(f"cpu_component_{i}", ComponentStatus.HEALTHY)
            time.sleep(0.01)
        
        # Wait for monitoring to complete
        monitor_thread.join()
        
        # Analyze CPU usage
        avg_cpu = statistics.mean(cpu_samples)
        max_cpu = max(cpu_samples)
        
        # Performance assertions (adjust thresholds based on system)
        assert avg_cpu < 50.0, f"Average CPU usage too high: {avg_cpu:.2f}%"
        assert max_cpu < 80.0, f"Maximum CPU usage too high: {max_cpu:.2f}%"
        
        print(f"CPU usage - Avg: {avg_cpu:.2f}%, Max: {max_cpu:.2f}%")
    
    def test_event_handling_performance(self, controller):
        """Test event handling performance."""
        # Set up event handlers
        event_counts = {"system_started": 0, "system_stopped": 0, "component_health_timeout": 0}
        
        def count_events(event_type):
            def handler(data):
                event_counts[event_type] += 1
            return handler
        
        controller.add_event_handler("system_started", count_events("system_started"))
        controller.add_event_handler("system_stopped", count_events("system_stopped"))
        controller.add_event_handler("component_health_timeout", count_events("component_health_timeout"))
        
        # Register component
        controller.register_component("event_component", health_check_interval=0.1)
        
        # Measure event handling performance
        start_time = time.perf_counter()
        
        # Generate events
        for _ in range(10):
            controller.start_system(timeout=10.0)
            controller.stop_system(timeout=10.0)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Check events were handled
        assert event_counts["system_started"] == 10
        assert event_counts["system_stopped"] == 10
        
        # Performance assertions
        time_per_cycle = total_time / 10
        assert time_per_cycle < 0.5, f"Event handling cycle time too high: {time_per_cycle:.3f}s"
        
        print(f"Event handling performance - {time_per_cycle:.3f}s per cycle")
    
    def test_state_history_performance(self, controller):
        """Test state history performance with many transitions."""
        # Register component
        controller.register_component("history_component")
        
        # Generate many state transitions
        start_time = time.perf_counter()
        
        for _ in range(100):
            controller.start_system(timeout=10.0)
            controller.pause_system(timeout=10.0)
            controller.resume_system(timeout=10.0)
            controller.stop_system(timeout=10.0)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Check state history was maintained
        history = controller.get_state_history()
        assert len(history) >= 400  # 4 transitions per cycle * 100 cycles
        
        # Performance assertions
        transition_time = total_time / 400
        assert transition_time < 0.01, f"State transition time too high: {transition_time:.6f}s"
        
        print(f"State history performance - {transition_time:.6f}s per transition")
    
    def test_scalability_with_many_components(self, controller):
        """Test scalability with large number of components."""
        component_counts = [100, 500, 1000, 2000]
        
        for count in component_counts:
            # Register components
            registration_start = time.perf_counter()
            for i in range(count):
                controller.register_component(f"scale_component_{i}", health_check_interval=30.0)
            registration_end = time.perf_counter()
            
            # Start system
            start_time = time.perf_counter()
            result = controller.start_system(timeout=30.0)
            end_time = time.perf_counter()
            
            assert result is True
            
            # Measure system health check
            health_start = time.perf_counter()
            is_healthy = controller.is_healthy()
            health_end = time.perf_counter()
            
            # Stop system
            stop_start = time.perf_counter()
            result = controller.stop_system(timeout=30.0)
            stop_end = time.perf_counter()
            
            assert result is True
            
            # Performance analysis
            registration_time = registration_end - registration_start
            startup_time = end_time - start_time
            health_time = health_end - health_start
            shutdown_time = stop_end - stop_start
            
            print(f"Scalability test with {count} components:")
            print(f"  Registration: {registration_time:.3f}s")
            print(f"  Startup: {startup_time:.3f}s")
            print(f"  Health check: {health_time:.6f}s")
            print(f"  Shutdown: {shutdown_time:.3f}s")
            
            # Performance assertions
            assert startup_time < 5.0, f"Startup time too high with {count} components: {startup_time:.3f}s"
            assert health_time < 0.1, f"Health check time too high with {count} components: {health_time:.6f}s"
            assert shutdown_time < 5.0, f"Shutdown time too high with {count} components: {shutdown_time:.3f}s"
            
            # Cleanup
            for i in range(count):
                controller.unregister_component(f"scale_component_{i}")


class TestResourceUsage:
    """Test resource usage characteristics."""
    
    @pytest.fixture
    def controller(self):
        """Create controller for resource testing."""
        controller = TradingSystemController()
        yield controller
        controller.shutdown(timeout=10.0)
    
    def test_thread_usage(self, controller):
        """Test thread usage patterns."""
        # Get initial thread count
        initial_threads = threading.active_count()
        
        # Register components and start system
        for i in range(10):
            controller.register_component(f"thread_component_{i}", health_check_interval=1.0)
        
        controller.start_system(timeout=10.0)
        
        # Check thread count after startup
        after_start_threads = threading.active_count()
        
        # Stop system
        controller.stop_system(timeout=10.0)
        
        # Check thread count after shutdown
        after_stop_threads = threading.active_count()
        
        # Analysis
        start_thread_increase = after_start_threads - initial_threads
        stop_thread_cleanup = after_start_threads - after_stop_threads
        
        print(f"Thread usage - Initial: {initial_threads}, After start: {after_start_threads}, After stop: {after_stop_threads}")
        
        # Assertions
        assert start_thread_increase <= 2, f"Too many threads created: {start_thread_increase}"
        assert stop_thread_cleanup >= 0, "Threads not cleaned up properly"
    
    def test_file_descriptor_usage(self, controller):
        """Test file descriptor usage."""
        # Get initial file descriptor count
        process = psutil.Process(os.getpid())
        initial_fds = process.num_fds()
        
        # Register components and start system
        for i in range(50):
            controller.register_component(f"fd_component_{i}", health_check_interval=2.0)
        
        controller.start_system(timeout=10.0)
        
        # Check file descriptor count after startup
        after_start_fds = process.num_fds()
        
        # Stop system
        controller.stop_system(timeout=10.0)
        
        # Check file descriptor count after shutdown
        after_stop_fds = process.num_fds()
        
        # Analysis
        fd_increase = after_start_fds - initial_fds
        fd_cleanup = after_start_fds - after_stop_fds
        
        print(f"File descriptor usage - Initial: {initial_fds}, After start: {after_start_fds}, After stop: {after_stop_fds}")
        
        # Assertions
        assert fd_increase <= 10, f"Too many file descriptors opened: {fd_increase}"
        assert fd_cleanup >= 0, "File descriptors not cleaned up properly"
    
    def test_memory_leak_detection(self, controller):
        """Test for memory leaks over many operations."""
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform many operations
        for cycle in range(10):
            # Register components
            for i in range(100):
                controller.register_component(f"leak_test_{cycle}_{i}", health_check_interval=10.0)
            
            # Start and stop system
            controller.start_system(timeout=10.0)
            controller.stop_system(timeout=10.0)
            
            # Unregister components
            for i in range(100):
                controller.unregister_component(f"leak_test_{cycle}_{i}")
            
            # Check memory usage periodically
            if cycle % 5 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory
                
                print(f"Memory after {cycle + 1} cycles: {current_memory:.2f}MB (increase: {memory_increase:.2f}MB)")
                
                # Memory should not grow significantly
                assert memory_increase < 50.0, f"Potential memory leak detected: {memory_increase:.2f}MB increase"
    
    def test_performance_under_stress(self, controller):
        """Test performance under stress conditions."""
        # Create stress conditions
        stress_components = 500
        stress_cycles = 20
        
        # Register many components
        for i in range(stress_components):
            controller.register_component(f"stress_component_{i}", health_check_interval=0.5)
        
        # Measure performance under stress
        start_time = time.perf_counter()
        
        for cycle in range(stress_cycles):
            # Start system
            controller.start_system(timeout=15.0)
            
            # Update all components
            for i in range(stress_components):
                controller.update_component_status(f"stress_component_{i}", ComponentStatus.HEALTHY)
            
            # Pause and resume
            controller.pause_system(timeout=10.0)
            controller.resume_system(timeout=10.0)
            
            # Stop system
            controller.stop_system(timeout=15.0)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Performance analysis
        time_per_cycle = total_time / stress_cycles
        operations_per_second = (stress_cycles * stress_components) / total_time
        
        print(f"Stress test results:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Time per cycle: {time_per_cycle:.2f}s")
        print(f"  Operations per second: {operations_per_second:.2f}")
        
        # Performance assertions
        assert time_per_cycle < 2.0, f"Stress test cycle time too high: {time_per_cycle:.2f}s"
        assert operations_per_second > 100, f"Operations per second too low: {operations_per_second:.2f}"


class TestLatencyMeasurement:
    """Test specific latency measurements for critical operations."""
    
    @pytest.fixture
    def controller(self):
        """Create controller for latency testing."""
        controller = TradingSystemController()
        yield controller
        controller.shutdown(timeout=5.0)
    
    def test_emergency_stop_latency_under_load(self, controller):
        """Test emergency stop latency under system load."""
        # Register many components
        for i in range(200):
            controller.register_component(f"load_component_{i}", health_check_interval=0.1)
        
        controller.start_system(timeout=10.0)
        
        # Create background load
        def background_load():
            for _ in range(1000):
                for i in range(200):
                    controller.update_component_status(f"load_component_{i}", ComponentStatus.HEALTHY)
                time.sleep(0.001)
        
        load_thread = threading.Thread(target=background_load)
        load_thread.start()
        
        # Wait for load to build up
        time.sleep(0.5)
        
        # Measure emergency stop latency
        emergency_times = []
        for _ in range(5):
            start_time = time.perf_counter()
            result = controller.emergency_stop(reason="latency_test")
            end_time = time.perf_counter()
            
            assert result is True
            emergency_times.append(end_time - start_time)
            
            # Reset for next test
            controller.reset_failsafe(force=True)
            controller.start_system(timeout=5.0)
        
        load_thread.join(timeout=2.0)
        
        # Analyze latency
        avg_latency = statistics.mean(emergency_times)
        max_latency = max(emergency_times)
        
        print(f"Emergency stop latency under load - Avg: {avg_latency:.4f}s, Max: {max_latency:.4f}s")
        
        # Emergency stop should remain fast even under load
        assert avg_latency < 0.1, f"Emergency stop latency too high under load: {avg_latency:.4f}s"
        assert max_latency < 0.2, f"Maximum emergency stop latency too high: {max_latency:.4f}s"
    
    def test_component_health_check_latency(self, controller):
        """Test component health check latency."""
        # Register components with different health check intervals
        for i in range(100):
            controller.register_component(f"health_component_{i}", health_check_interval=0.1)
        
        controller.start_system(timeout=10.0)
        
        # Set all components to healthy
        for i in range(100):
            controller.update_component_status(f"health_component_{i}", ComponentStatus.HEALTHY)
        
        # Measure health check latency
        health_check_times = []
        for _ in range(20):
            start_time = time.perf_counter()
            controller._check_component_health()
            end_time = time.perf_counter()
            
            health_check_times.append(end_time - start_time)
        
        # Analyze latency
        avg_health_check_time = statistics.mean(health_check_times)
        max_health_check_time = max(health_check_times)
        
        print(f"Health check latency - Avg: {avg_health_check_time:.4f}s, Max: {max_health_check_time:.4f}s")
        
        # Health checks should be fast
        assert avg_health_check_time < 0.01, f"Health check latency too high: {avg_health_check_time:.4f}s"
        assert max_health_check_time < 0.05, f"Maximum health check latency too high: {max_health_check_time:.4f}s"
    
    def test_state_query_latency(self, controller):
        """Test state query operation latency."""
        # Register components
        for i in range(50):
            controller.register_component(f"query_component_{i}", health_check_interval=1.0)
        
        controller.start_system(timeout=10.0)
        
        # Measure state query latency
        query_operations = [
            ("get_state", lambda: controller.get_state()),
            ("get_all_components", lambda: controller.get_all_components()),
            ("get_performance_metrics", lambda: controller.get_performance_metrics()),
            ("is_healthy", lambda: controller.is_healthy()),
            ("get_state_history", lambda: controller.get_state_history(limit=10))
        ]
        
        for operation_name, operation_func in query_operations:
            query_times = []
            
            for _ in range(100):
                start_time = time.perf_counter()
                result = operation_func()
                end_time = time.perf_counter()
                
                query_times.append(end_time - start_time)
                assert result is not None  # Basic validation
            
            avg_query_time = statistics.mean(query_times)
            max_query_time = max(query_times)
            
            print(f"{operation_name} latency - Avg: {avg_query_time:.6f}s, Max: {max_query_time:.6f}s")
            
            # Query operations should be very fast
            assert avg_query_time < 0.001, f"{operation_name} latency too high: {avg_query_time:.6f}s"
            assert max_query_time < 0.01, f"Maximum {operation_name} latency too high: {max_query_time:.6f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])