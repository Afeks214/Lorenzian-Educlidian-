"""
RDMA Testing Simulation Framework
===============================

Simulates RDMA (Remote Direct Memory Access) operations for ultra-low latency
testing. Provides realistic latency modeling and performance validation.
"""

import time
import threading
import queue
import socket
import struct
import random
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from .nanosecond_timer import NanosecondTimer


class RDMAOperation(Enum):
    """RDMA operation types"""
    READ = "read"
    WRITE = "write"
    SEND = "send"
    RECV = "recv"
    ATOMIC_CAS = "atomic_cas"
    ATOMIC_ADD = "atomic_add"


@dataclass
class RDMAMessage:
    """RDMA message structure"""
    operation: RDMAOperation
    source_id: int
    destination_id: int
    data: bytes
    size: int
    timestamp_ns: int
    sequence_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RDMAConnection:
    """RDMA connection configuration"""
    connection_id: int
    local_id: int
    remote_id: int
    queue_depth: int
    max_message_size: int
    latency_profile: Dict[str, float]
    bandwidth_gbps: float
    reliability: float = 0.999


@dataclass
class RDMAPerformanceMetrics:
    """RDMA performance metrics"""
    operation_type: RDMAOperation
    total_operations: int
    successful_operations: int
    failed_operations: int
    min_latency_ns: int
    max_latency_ns: int
    avg_latency_ns: float
    p95_latency_ns: int
    p99_latency_ns: int
    throughput_ops_per_sec: float
    bandwidth_utilization: float
    error_rate: float


class RDMASimulator:
    """
    RDMA simulation framework for ultra-low latency testing
    
    Features:
    - Realistic RDMA operation simulation
    - Configurable latency profiles
    - Multi-connection support
    - Performance metrics collection
    - Error injection and testing
    """
    
    def __init__(self, timer: NanosecondTimer):
        self.timer = timer
        self.connections: Dict[int, RDMAConnection] = {}
        self.message_queues: Dict[int, queue.Queue] = {}
        self.performance_metrics: Dict[str, RDMAPerformanceMetrics] = {}
        self.active_connections: Dict[int, bool] = {}
        self.sequence_numbers: Dict[int, int] = {}
        self.executor = ThreadPoolExecutor(max_workers=16)
        
        # Default latency profiles for different scenarios
        self.default_latency_profiles = {
            'local_loopback': {'base_ns': 100, 'variance_ns': 20},
            'local_network': {'base_ns': 500, 'variance_ns': 100},
            'remote_network': {'base_ns': 2000, 'variance_ns': 500},
            'wan': {'base_ns': 10000, 'variance_ns': 2000}
        }
    
    def create_connection(self, connection_id: int, local_id: int, remote_id: int,
                         queue_depth: int = 1024, max_message_size: int = 4096,
                         latency_profile: str = 'local_network',
                         bandwidth_gbps: float = 10.0) -> RDMAConnection:
        """Create a new RDMA connection"""
        
        latency_config = self.default_latency_profiles.get(
            latency_profile, 
            self.default_latency_profiles['local_network']
        )
        
        connection = RDMAConnection(
            connection_id=connection_id,
            local_id=local_id,
            remote_id=remote_id,
            queue_depth=queue_depth,
            max_message_size=max_message_size,
            latency_profile=latency_config,
            bandwidth_gbps=bandwidth_gbps
        )
        
        self.connections[connection_id] = connection
        self.message_queues[connection_id] = queue.Queue(maxsize=queue_depth)
        self.active_connections[connection_id] = True
        self.sequence_numbers[connection_id] = 0
        
        return connection
    
    def simulate_rdma_operation(self, connection_id: int, operation: RDMAOperation,
                               data: bytes, blocking: bool = True) -> Optional[RDMAMessage]:
        """Simulate an RDMA operation"""
        
        if connection_id not in self.connections:
            raise ValueError(f"Connection {connection_id} not found")
        
        connection = self.connections[connection_id]
        
        # Create message
        message = RDMAMessage(
            operation=operation,
            source_id=connection.local_id,
            destination_id=connection.remote_id,
            data=data,
            size=len(data),
            timestamp_ns=time.perf_counter_ns(),
            sequence_number=self._get_next_sequence_number(connection_id)
        )
        
        # Simulate operation based on type
        if operation == RDMAOperation.WRITE:
            return self._simulate_write(connection, message, blocking)
        elif operation == RDMAOperation.READ:
            return self._simulate_read(connection, message, blocking)
        elif operation == RDMAOperation.SEND:
            return self._simulate_send(connection, message, blocking)
        elif operation == RDMAOperation.RECV:
            return self._simulate_recv(connection, message, blocking)
        elif operation == RDMAOperation.ATOMIC_CAS:
            return self._simulate_atomic_cas(connection, message, blocking)
        elif operation == RDMAOperation.ATOMIC_ADD:
            return self._simulate_atomic_add(connection, message, blocking)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _simulate_write(self, connection: RDMAConnection, message: RDMAMessage,
                       blocking: bool) -> Optional[RDMAMessage]:
        """Simulate RDMA write operation"""
        operation_name = f"rdma_write_{connection.connection_id}"
        
        with self.timer.measure(operation_name):
            # Simulate network latency
            latency_ns = self._calculate_latency(connection.latency_profile)
            
            # Simulate bandwidth limitation
            transfer_time_ns = self._calculate_transfer_time(
                message.size, connection.bandwidth_gbps
            )
            
            # Total time is latency + transfer time
            total_time_ns = latency_ns + transfer_time_ns
            
            # Simulate the delay
            self._simulate_delay(total_time_ns)
            
            # Simulate potential failures
            if random.random() > connection.reliability:
                raise RuntimeError("RDMA write failed")
            
            message.metadata['completed_at'] = time.perf_counter_ns()
            message.metadata['simulated_latency_ns'] = latency_ns
            message.metadata['transfer_time_ns'] = transfer_time_ns
            
        return message
    
    def _simulate_read(self, connection: RDMAConnection, message: RDMAMessage,
                      blocking: bool) -> Optional[RDMAMessage]:
        """Simulate RDMA read operation"""
        operation_name = f"rdma_read_{connection.connection_id}"
        
        with self.timer.measure(operation_name):
            # Read operations have round-trip latency
            latency_ns = self._calculate_latency(connection.latency_profile) * 2
            
            # Simulate bandwidth for data transfer
            transfer_time_ns = self._calculate_transfer_time(
                message.size, connection.bandwidth_gbps
            )
            
            total_time_ns = latency_ns + transfer_time_ns
            self._simulate_delay(total_time_ns)
            
            if random.random() > connection.reliability:
                raise RuntimeError("RDMA read failed")
            
            # Simulate reading data
            read_data = self._generate_test_data(message.size)
            message.data = read_data
            message.metadata['completed_at'] = time.perf_counter_ns()
            message.metadata['simulated_latency_ns'] = latency_ns
            message.metadata['transfer_time_ns'] = transfer_time_ns
            
        return message
    
    def _simulate_send(self, connection: RDMAConnection, message: RDMAMessage,
                      blocking: bool) -> Optional[RDMAMessage]:
        """Simulate RDMA send operation"""
        operation_name = f"rdma_send_{connection.connection_id}"
        
        with self.timer.measure(operation_name):
            # Send operations are one-way
            latency_ns = self._calculate_latency(connection.latency_profile)
            transfer_time_ns = self._calculate_transfer_time(
                message.size, connection.bandwidth_gbps
            )
            
            total_time_ns = latency_ns + transfer_time_ns
            self._simulate_delay(total_time_ns)
            
            if random.random() > connection.reliability:
                raise RuntimeError("RDMA send failed")
            
            # Queue message for receiver
            if not self.message_queues[connection.connection_id].full():
                self.message_queues[connection.connection_id].put(message)
            
            message.metadata['completed_at'] = time.perf_counter_ns()
            message.metadata['simulated_latency_ns'] = latency_ns
            message.metadata['transfer_time_ns'] = transfer_time_ns
            
        return message
    
    def _simulate_recv(self, connection: RDMAConnection, message: RDMAMessage,
                      blocking: bool) -> Optional[RDMAMessage]:
        """Simulate RDMA receive operation"""
        operation_name = f"rdma_recv_{connection.connection_id}"
        
        with self.timer.measure(operation_name):
            try:
                # Try to receive message from queue
                if blocking:
                    received_message = self.message_queues[connection.connection_id].get(timeout=1.0)
                else:
                    received_message = self.message_queues[connection.connection_id].get_nowait()
                
                received_message.metadata['received_at'] = time.perf_counter_ns()
                return received_message
                
            except queue.Empty:
                return None
    
    def _simulate_atomic_cas(self, connection: RDMAConnection, message: RDMAMessage,
                           blocking: bool) -> Optional[RDMAMessage]:
        """Simulate RDMA atomic compare-and-swap operation"""
        operation_name = f"rdma_atomic_cas_{connection.connection_id}"
        
        with self.timer.measure(operation_name):
            # Atomic operations have round-trip latency
            latency_ns = self._calculate_latency(connection.latency_profile) * 2
            
            # Atomic operations are typically small (8 bytes)
            atomic_size = 8
            transfer_time_ns = self._calculate_transfer_time(
                atomic_size, connection.bandwidth_gbps
            )
            
            total_time_ns = latency_ns + transfer_time_ns
            self._simulate_delay(total_time_ns)
            
            if random.random() > connection.reliability:
                raise RuntimeError("RDMA atomic CAS failed")
            
            # Simulate CAS result (success/failure)
            cas_success = random.random() > 0.1  # 90% success rate
            
            message.metadata['completed_at'] = time.perf_counter_ns()
            message.metadata['cas_success'] = cas_success
            message.metadata['simulated_latency_ns'] = latency_ns
            message.metadata['transfer_time_ns'] = transfer_time_ns
            
        return message
    
    def _simulate_atomic_add(self, connection: RDMAConnection, message: RDMAMessage,
                           blocking: bool) -> Optional[RDMAMessage]:
        """Simulate RDMA atomic add operation"""
        operation_name = f"rdma_atomic_add_{connection.connection_id}"
        
        with self.timer.measure(operation_name):
            # Similar to CAS but always succeeds
            latency_ns = self._calculate_latency(connection.latency_profile) * 2
            atomic_size = 8
            transfer_time_ns = self._calculate_transfer_time(
                atomic_size, connection.bandwidth_gbps
            )
            
            total_time_ns = latency_ns + transfer_time_ns
            self._simulate_delay(total_time_ns)
            
            if random.random() > connection.reliability:
                raise RuntimeError("RDMA atomic add failed")
            
            # Simulate previous value
            previous_value = random.randint(0, 1000000)
            
            message.metadata['completed_at'] = time.perf_counter_ns()
            message.metadata['previous_value'] = previous_value
            message.metadata['simulated_latency_ns'] = latency_ns
            message.metadata['transfer_time_ns'] = transfer_time_ns
            
        return message
    
    def _calculate_latency(self, latency_profile: Dict[str, float]) -> int:
        """Calculate simulated latency with variance"""
        base_ns = latency_profile['base_ns']
        variance_ns = latency_profile['variance_ns']
        
        # Add random variance (normal distribution)
        actual_latency = max(0, random.normalvariate(base_ns, variance_ns))
        return int(actual_latency)
    
    def _calculate_transfer_time(self, size_bytes: int, bandwidth_gbps: float) -> int:
        """Calculate transfer time based on size and bandwidth"""
        if bandwidth_gbps <= 0:
            return 0
        
        # Convert bandwidth from Gbps to bytes per nanosecond
        bytes_per_ns = (bandwidth_gbps * 1e9) / 8 / 1e9
        
        # Calculate transfer time in nanoseconds
        transfer_time_ns = size_bytes / bytes_per_ns
        return int(transfer_time_ns)
    
    def _simulate_delay(self, delay_ns: int):
        """Simulate processing delay"""
        if delay_ns > 0:
            # For very small delays, use busy waiting for accuracy
            if delay_ns < 1000000:  # Less than 1ms
                start = time.perf_counter_ns()
                while time.perf_counter_ns() - start < delay_ns:
                    pass
            else:
                # For larger delays, use sleep
                time.sleep(delay_ns / 1e9)
    
    def _generate_test_data(self, size: int) -> bytes:
        """Generate test data of specified size"""
        return bytes(random.randint(0, 255) for _ in range(size))
    
    def _get_next_sequence_number(self, connection_id: int) -> int:
        """Get next sequence number for connection"""
        self.sequence_numbers[connection_id] += 1
        return self.sequence_numbers[connection_id]
    
    def benchmark_rdma_performance(self, connection_id: int, operation: RDMAOperation,
                                 message_size: int, num_operations: int = 1000) -> RDMAPerformanceMetrics:
        """Benchmark RDMA performance for specific operation"""
        
        if connection_id not in self.connections:
            raise ValueError(f"Connection {connection_id} not found")
        
        connection = self.connections[connection_id]
        test_data = self._generate_test_data(message_size)
        
        successful_operations = 0
        failed_operations = 0
        latencies = []
        
        start_time = time.perf_counter_ns()
        
        for i in range(num_operations):
            try:
                message = self.simulate_rdma_operation(
                    connection_id, operation, test_data, blocking=True
                )
                
                if message:
                    successful_operations += 1
                    
                    # Calculate operation latency
                    if 'completed_at' in message.metadata:
                        latency_ns = message.metadata['completed_at'] - message.timestamp_ns
                        latencies.append(latency_ns)
                
            except Exception:
                failed_operations += 1
        
        end_time = time.perf_counter_ns()
        total_time_ns = end_time - start_time
        
        # Calculate performance metrics
        if latencies:
            latencies.sort()
            min_latency = min(latencies)
            max_latency = max(latencies)
            avg_latency = sum(latencies) / len(latencies)
            p95_latency = latencies[int(0.95 * len(latencies))]
            p99_latency = latencies[int(0.99 * len(latencies))]
        else:
            min_latency = max_latency = avg_latency = p95_latency = p99_latency = 0
        
        throughput_ops_per_sec = num_operations / (total_time_ns / 1e9)
        
        # Calculate bandwidth utilization
        total_bytes = message_size * successful_operations
        actual_bandwidth_gbps = (total_bytes * 8) / (total_time_ns / 1e9) / 1e9
        bandwidth_utilization = actual_bandwidth_gbps / connection.bandwidth_gbps
        
        error_rate = failed_operations / num_operations
        
        metrics = RDMAPerformanceMetrics(
            operation_type=operation,
            total_operations=num_operations,
            successful_operations=successful_operations,
            failed_operations=failed_operations,
            min_latency_ns=min_latency,
            max_latency_ns=max_latency,
            avg_latency_ns=avg_latency,
            p95_latency_ns=p95_latency,
            p99_latency_ns=p99_latency,
            throughput_ops_per_sec=throughput_ops_per_sec,
            bandwidth_utilization=bandwidth_utilization,
            error_rate=error_rate
        )
        
        metrics_key = f"{connection_id}_{operation.value}_{message_size}"
        self.performance_metrics[metrics_key] = metrics
        
        return metrics
    
    def run_comprehensive_benchmark(self, connection_id: int) -> Dict[str, RDMAPerformanceMetrics]:
        """Run comprehensive RDMA benchmark"""
        
        operations = [
            RDMAOperation.WRITE,
            RDMAOperation.READ,
            RDMAOperation.SEND,
            RDMAOperation.ATOMIC_CAS,
            RDMAOperation.ATOMIC_ADD
        ]
        
        message_sizes = [64, 256, 1024, 4096]
        results = {}
        
        for operation in operations:
            for size in message_sizes:
                try:
                    metrics = self.benchmark_rdma_performance(
                        connection_id, operation, size, num_operations=500
                    )
                    key = f"{operation.value}_{size}b"
                    results[key] = metrics
                except Exception as e:
                    results[f"{operation.value}_{size}b"] = f"Error: {str(e)}"
        
        return results
    
    def validate_rdma_latency_requirements(self, connection_id: int, 
                                         max_latency_ns: int) -> Dict[str, Any]:
        """Validate RDMA latency requirements"""
        
        validation_results = {
            'connection_id': connection_id,
            'max_latency_requirement_ns': max_latency_ns,
            'validation_passed': True,
            'violations': [],
            'operation_results': {}
        }
        
        # Test critical operations
        critical_operations = [
            (RDMAOperation.WRITE, 64),
            (RDMAOperation.READ, 64),
            (RDMAOperation.ATOMIC_CAS, 8)
        ]
        
        for operation, size in critical_operations:
            try:
                metrics = self.benchmark_rdma_performance(
                    connection_id, operation, size, num_operations=100
                )
                
                operation_key = f"{operation.value}_{size}b"
                validation_results['operation_results'][operation_key] = metrics
                
                # Check if latency requirements are met
                if metrics.avg_latency_ns > max_latency_ns:
                    validation_results['validation_passed'] = False
                    validation_results['violations'].append(
                        f"{operation_key}: Average latency {metrics.avg_latency_ns}ns "
                        f"exceeds requirement {max_latency_ns}ns"
                    )
                
                if metrics.p95_latency_ns > max_latency_ns * 1.5:
                    validation_results['validation_passed'] = False
                    validation_results['violations'].append(
                        f"{operation_key}: P95 latency {metrics.p95_latency_ns}ns "
                        f"exceeds 1.5x requirement {max_latency_ns * 1.5}ns"
                    )
                
            except Exception as e:
                validation_results['validation_passed'] = False
                validation_results['violations'].append(
                    f"{operation.value}_{size}b: Test failed with error: {str(e)}"
                )
        
        return validation_results
    
    def close_connection(self, connection_id: int):
        """Close RDMA connection"""
        if connection_id in self.connections:
            self.active_connections[connection_id] = False
            del self.connections[connection_id]
            del self.message_queues[connection_id]
            del self.sequence_numbers[connection_id]
    
    def cleanup(self):
        """Cleanup all resources"""
        for connection_id in list(self.connections.keys()):
            self.close_connection(connection_id)
        
        self.executor.shutdown(wait=True)