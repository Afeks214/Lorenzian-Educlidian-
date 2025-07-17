"""
Parallel Test Executor

Provides concurrent test execution with resource allocation management,
load balancing, and intelligent scheduling for adversarial testing.
"""

import asyncio
import multiprocessing
import threading
import time
import psutil
import logging
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json
import pickle
import os
import sys
import resource
import signal
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    docker = None
import subprocess
import shutil

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.event_bus import EventBus
from src.core.events import Event


class ExecutionMode(Enum):
    THREAD = "thread"
    PROCESS = "process"
    CONTAINER = "container"
    ASYNC = "async"


class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"


@dataclass
class ResourceQuota:
    """Resource quota specification"""
    cpu_cores: float = 1.0
    memory_mb: int = 1024
    disk_mb: int = 1024
    network_mbps: float = 100.0
    gpu_count: int = 0
    max_duration_seconds: int = 3600
    priority: int = 1


@dataclass
class ExecutionContext:
    """Context for test execution"""
    execution_id: str
    test_name: str
    execution_mode: ExecutionMode
    resource_quota: ResourceQuota
    environment: Dict[str, str] = field(default_factory=dict)
    working_directory: str = ""
    timeout: float = 300.0
    retry_count: int = 3
    isolation_level: str = "none"  # none, process, container
    
    # Runtime metadata
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    actual_resources: Dict[str, float] = field(default_factory=dict)
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerNode:
    """Worker node for parallel execution"""
    node_id: str
    capacity: ResourceQuota
    current_load: ResourceQuota = field(default_factory=ResourceQuota)
    active_tasks: List[str] = field(default_factory=list)
    health_status: str = "healthy"
    last_heartbeat: Optional[datetime] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class ResourceMonitor:
    """Monitor system resources for allocation decisions"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.resource_history = deque(maxlen=100)
        self.allocation_history = deque(maxlen=1000)
        self.monitoring_active = False
        self.monitoring_task = None
        self.lock = threading.Lock()
        
    async def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                
                with self.lock:
                    self.resource_history.append({
                        'timestamp': datetime.now(),
                        'metrics': metrics
                    })
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logging.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(5)
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free_gb = disk.free / (1024**3)
            
            # Network metrics
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            # Load average
            try:
                load_avg = os.getloadavg()
                load_1min = load_avg[0]
                load_5min = load_avg[1]
                load_15min = load_avg[2]
            except (OSError, AttributeError):
                load_1min = load_5min = load_15min = 0.0
            
            return {
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'memory_percent': memory_percent,
                'memory_available_gb': memory_available_gb,
                'disk_percent': disk_percent,
                'disk_free_gb': disk_free_gb,
                'network_bytes_sent': network_bytes_sent,
                'network_bytes_recv': network_bytes_recv,
                'load_1min': load_1min,
                'load_5min': load_5min,
                'load_15min': load_15min
            }
        except Exception as e:
            logging.error(f"Error collecting system metrics: {e}")
            return {}
    
    def get_resource_availability(self) -> Dict[str, float]:
        """Get current resource availability"""
        with self.lock:
            if not self.resource_history:
                return {}
            
            latest = self.resource_history[-1]['metrics']
            
            return {
                'cpu_available_percent': 100 - latest.get('cpu_percent', 0),
                'memory_available_gb': latest.get('memory_available_gb', 0),
                'disk_available_gb': latest.get('disk_free_gb', 0),
                'cpu_cores_available': latest.get('cpu_count', 1) * (100 - latest.get('cpu_percent', 0)) / 100,
                'load_factor': latest.get('load_1min', 0) / latest.get('cpu_count', 1)
            }
    
    def record_allocation(self, execution_id: str, quota: ResourceQuota, action: str):
        """Record resource allocation/deallocation"""
        with self.lock:
            self.allocation_history.append({
                'timestamp': datetime.now(),
                'execution_id': execution_id,
                'quota': quota,
                'action': action,
                'system_metrics': self._collect_system_metrics()
            })


class ContainerManager:
    """Manage Docker containers for isolated test execution"""
    
    def __init__(self):
        self.containers = {}
        self.docker_client = None
        self.lock = threading.Lock()
        
        # Initialize Docker client
        if DOCKER_AVAILABLE:
            try:
                self.docker_client = docker.from_env()
            except Exception as e:
                logging.warning(f"Docker not available: {e}")
                self.docker_client = None
        else:
            self.docker_client = None
    
    def is_available(self) -> bool:
        """Check if Docker is available"""
        return self.docker_client is not None
    
    async def create_container(self, execution_id: str, image: str = "python:3.9",
                             resource_quota: ResourceQuota = None,
                             environment: Dict[str, str] = None) -> str:
        """Create a new container for test execution"""
        if not self.is_available():
            raise RuntimeError("Docker not available")
        
        quota = resource_quota or ResourceQuota()
        env = environment or {}
        
        # Container configuration
        container_config = {
            'image': image,
            'name': f"test_executor_{execution_id}",
            'environment': env,
            'detach': True,
            'remove': True,
            'cpu_period': 100000,
            'cpu_quota': int(quota.cpu_cores * 100000),
            'mem_limit': f"{quota.memory_mb}m",
            'network_mode': 'bridge',
            'working_dir': '/workspace'
        }
        
        # Add volume mounts
        mounts = [
            docker.types.Mount(
                target='/workspace',
                source=os.getcwd(),
                type='bind'
            )
        ]
        container_config['mounts'] = mounts
        
        try:
            with self.lock:
                container = self.docker_client.containers.run(**container_config)
                self.containers[execution_id] = container
                return container.id
        except Exception as e:
            logging.error(f"Error creating container: {e}")
            raise
    
    async def execute_in_container(self, execution_id: str, command: str,
                                  working_dir: str = "/workspace") -> Tuple[int, str, str]:
        """Execute command in container"""
        if execution_id not in self.containers:
            raise ValueError(f"Container {execution_id} not found")
        
        container = self.containers[execution_id]
        
        try:
            # Execute command
            result = container.exec_run(command, workdir=working_dir)
            stdout = result.output.decode('utf-8')
            exit_code = result.exit_code
            
            return exit_code, stdout, ""
        except Exception as e:
            logging.error(f"Error executing in container: {e}")
            return 1, "", str(e)
    
    async def cleanup_container(self, execution_id: str):
        """Clean up container"""
        if execution_id not in self.containers:
            return
        
        container = self.containers[execution_id]
        
        try:
            container.stop(timeout=10)
            container.remove()
        except Exception as e:
            logging.error(f"Error cleaning up container: {e}")
        finally:
            with self.lock:
                if execution_id in self.containers:
                    del self.containers[execution_id]
    
    def get_container_stats(self, execution_id: str) -> Dict:
        """Get container resource statistics"""
        if execution_id not in self.containers:
            return {}
        
        container = self.containers[execution_id]
        
        try:
            stats = container.stats(stream=False)
            return {
                'cpu_percent': self._calculate_cpu_percent(stats),
                'memory_usage_mb': stats['memory_stats']['usage'] / (1024**2),
                'memory_limit_mb': stats['memory_stats']['limit'] / (1024**2),
                'network_rx_bytes': stats['networks']['eth0']['rx_bytes'],
                'network_tx_bytes': stats['networks']['eth0']['tx_bytes']
            }
        except Exception as e:
            logging.error(f"Error getting container stats: {e}")
            return {}
    
    def _calculate_cpu_percent(self, stats: Dict) -> float:
        """Calculate CPU percentage from container stats"""
        try:
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            if system_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * \
                             len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100
                return cpu_percent
        except (KeyError, ZeroDivisionError):
            pass
        
        return 0.0


class LoadBalancer:
    """Load balancer for distributing tests across worker nodes"""
    
    def __init__(self):
        self.nodes = {}
        self.scheduling_strategy = "least_loaded"
        self.lock = threading.Lock()
    
    def register_node(self, node: WorkerNode):
        """Register a worker node"""
        with self.lock:
            self.nodes[node.node_id] = node
    
    def unregister_node(self, node_id: str):
        """Unregister a worker node"""
        with self.lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
    
    def select_node(self, resource_quota: ResourceQuota) -> Optional[WorkerNode]:
        """Select best node for task execution"""
        with self.lock:
            if not self.nodes:
                return None
            
            # Filter nodes that can handle the resource requirements
            suitable_nodes = []
            for node in self.nodes.values():
                if self._can_handle_task(node, resource_quota):
                    suitable_nodes.append(node)
            
            if not suitable_nodes:
                return None
            
            # Select based on strategy
            if self.scheduling_strategy == "least_loaded":
                return min(suitable_nodes, key=lambda n: self._calculate_load(n))
            elif self.scheduling_strategy == "round_robin":
                return suitable_nodes[0]  # Simplified round-robin
            else:
                return suitable_nodes[0]
    
    def _can_handle_task(self, node: WorkerNode, quota: ResourceQuota) -> bool:
        """Check if node can handle the task"""
        available_cpu = node.capacity.cpu_cores - node.current_load.cpu_cores
        available_memory = node.capacity.memory_mb - node.current_load.memory_mb
        
        return (available_cpu >= quota.cpu_cores and
                available_memory >= quota.memory_mb and
                node.health_status == "healthy")
    
    def _calculate_load(self, node: WorkerNode) -> float:
        """Calculate node load factor"""
        cpu_load = node.current_load.cpu_cores / node.capacity.cpu_cores
        memory_load = node.current_load.memory_mb / node.capacity.memory_mb
        
        return max(cpu_load, memory_load)
    
    def allocate_resources(self, node_id: str, quota: ResourceQuota) -> bool:
        """Allocate resources on a node"""
        with self.lock:
            if node_id not in self.nodes:
                return False
            
            node = self.nodes[node_id]
            
            if not self._can_handle_task(node, quota):
                return False
            
            # Allocate resources
            node.current_load.cpu_cores += quota.cpu_cores
            node.current_load.memory_mb += quota.memory_mb
            node.current_load.disk_mb += quota.disk_mb
            
            return True
    
    def deallocate_resources(self, node_id: str, quota: ResourceQuota):
        """Deallocate resources from a node"""
        with self.lock:
            if node_id not in self.nodes:
                return
            
            node = self.nodes[node_id]
            
            # Deallocate resources
            node.current_load.cpu_cores = max(0, node.current_load.cpu_cores - quota.cpu_cores)
            node.current_load.memory_mb = max(0, node.current_load.memory_mb - quota.memory_mb)
            node.current_load.disk_mb = max(0, node.current_load.disk_mb - quota.disk_mb)
    
    def get_cluster_status(self) -> Dict:
        """Get cluster status"""
        with self.lock:
            total_capacity = ResourceQuota()
            total_used = ResourceQuota()
            
            for node in self.nodes.values():
                total_capacity.cpu_cores += node.capacity.cpu_cores
                total_capacity.memory_mb += node.capacity.memory_mb
                total_capacity.disk_mb += node.capacity.disk_mb
                
                total_used.cpu_cores += node.current_load.cpu_cores
                total_used.memory_mb += node.current_load.memory_mb
                total_used.disk_mb += node.current_load.disk_mb
            
            return {
                'total_nodes': len(self.nodes),
                'healthy_nodes': len([n for n in self.nodes.values() if n.health_status == "healthy"]),
                'total_capacity': total_capacity,
                'total_used': total_used,
                'utilization': {
                    'cpu': (total_used.cpu_cores / total_capacity.cpu_cores * 100) if total_capacity.cpu_cores > 0 else 0,
                    'memory': (total_used.memory_mb / total_capacity.memory_mb * 100) if total_capacity.memory_mb > 0 else 0,
                    'disk': (total_used.disk_mb / total_capacity.disk_mb * 100) if total_capacity.disk_mb > 0 else 0
                }
            }


class ParallelExecutor:
    """
    Main parallel executor that coordinates test execution across multiple
    worker nodes with resource management and load balancing.
    """
    
    def __init__(self, max_workers: int = None, enable_containers: bool = False):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.enable_containers = enable_containers
        
        # Core components
        self.resource_monitor = ResourceMonitor()
        self.container_manager = ContainerManager() if enable_containers else None
        self.load_balancer = LoadBalancer()
        
        # Execution tracking
        self.active_executions = {}
        self.execution_history = deque(maxlen=1000)
        self.execution_queue = asyncio.Queue()
        
        # Thread and process pools
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        
        # Event system
        self.event_bus = EventBus()
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_task = None
        
        # Performance metrics
        self.metrics = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0.0,
            'total_execution_time': 0.0,
            'resource_efficiency': 0.0
        }
        
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Initialize local worker node
        self._initialize_local_node()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('parallel_executor.log'),
                logging.StreamHandler()
            ]
        )
    
    def _initialize_local_node(self):
        """Initialize local worker node"""
        # Get system capacity
        cpu_cores = psutil.cpu_count()
        memory_mb = psutil.virtual_memory().total // (1024**2)
        disk_mb = psutil.disk_usage('/').free // (1024**2)
        
        local_node = WorkerNode(
            node_id="local",
            capacity=ResourceQuota(
                cpu_cores=cpu_cores,
                memory_mb=memory_mb,
                disk_mb=disk_mb
            ),
            health_status="healthy",
            last_heartbeat=datetime.now()
        )
        
        self.load_balancer.register_node(local_node)
        self.logger.info(f"Initialized local node with {cpu_cores} CPU cores, {memory_mb}MB memory")
    
    async def start(self):
        """Start the parallel executor"""
        self.monitoring_active = True
        
        # Start resource monitoring
        await self.resource_monitor.start_monitoring()
        
        # Start execution monitoring
        self.monitoring_task = asyncio.create_task(self._execution_monitoring_loop())
        
        self.logger.info("Parallel executor started")
    
    async def stop(self):
        """Stop the parallel executor"""
        self.monitoring_active = False
        
        # Stop monitoring
        await self.resource_monitor.stop_monitoring()
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup active executions
        for execution_id in list(self.active_executions.keys()):
            await self.cancel_execution(execution_id)
        
        # Shutdown thread pools
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        self.logger.info("Parallel executor stopped")
    
    async def _execution_monitoring_loop(self):
        """Monitor execution progress and health"""
        while self.monitoring_active:
            try:
                # Check execution health
                await self._check_execution_health()
                
                # Update metrics
                self._update_metrics()
                
                # Emit status updates
                await self._emit_status_updates()
                
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error in execution monitoring: {e}")
                await asyncio.sleep(10)
    
    async def _check_execution_health(self):
        """Check health of active executions"""
        current_time = datetime.now()
        
        for execution_id, context in list(self.active_executions.items()):
            # Check for timeouts
            if (context.start_time and 
                (current_time - context.start_time).total_seconds() > context.timeout):
                
                self.logger.warning(f"Execution {execution_id} timed out")
                await self.cancel_execution(execution_id)
            
            # Update resource usage
            if context.execution_mode == ExecutionMode.CONTAINER and self.container_manager:
                stats = self.container_manager.get_container_stats(execution_id)
                context.actual_resources.update(stats)
    
    def _update_metrics(self):
        """Update performance metrics"""
        if self.execution_history:
            total_time = sum(
                (ctx.end_time - ctx.start_time).total_seconds()
                for ctx in self.execution_history
                if ctx.end_time and ctx.start_time
            )
            
            self.metrics['total_executions'] = len(self.execution_history)
            self.metrics['successful_executions'] = sum(
                1 for ctx in self.execution_history if ctx.exit_code == 0
            )
            self.metrics['failed_executions'] = sum(
                1 for ctx in self.execution_history if ctx.exit_code != 0
            )
            self.metrics['average_execution_time'] = total_time / len(self.execution_history)
            self.metrics['total_execution_time'] = total_time
    
    async def _emit_status_updates(self):
        """Emit status updates"""
        status = {
            'active_executions': len(self.active_executions),
            'cluster_status': self.load_balancer.get_cluster_status(),
            'resource_availability': self.resource_monitor.get_resource_availability(),
            'metrics': self.metrics
        }
        
        await self.event_bus.emit(Event(
            type="executor_status_update",
            data=status
        ))
    
    async def execute_test(self, test_function: Callable, test_args: Tuple = (),
                          test_kwargs: Dict = None, execution_mode: ExecutionMode = ExecutionMode.ASYNC,
                          resource_quota: ResourceQuota = None, 
                          environment: Dict[str, str] = None,
                          timeout: float = 300.0) -> ExecutionContext:
        """Execute a test with specified parameters"""
        
        execution_id = f"exec_{int(time.time())}_{len(self.active_executions)}"
        test_kwargs = test_kwargs or {}
        resource_quota = resource_quota or ResourceQuota()
        environment = environment or {}
        
        # Create execution context
        context = ExecutionContext(
            execution_id=execution_id,
            test_name=test_function.__name__,
            execution_mode=execution_mode,
            resource_quota=resource_quota,
            environment=environment,
            timeout=timeout
        )
        
        # Select worker node
        node = self.load_balancer.select_node(resource_quota)
        if not node:
            raise RuntimeError("No suitable worker node available")
        
        # Allocate resources
        if not self.load_balancer.allocate_resources(node.node_id, resource_quota):
            raise RuntimeError("Failed to allocate resources")
        
        # Record allocation
        self.resource_monitor.record_allocation(execution_id, resource_quota, "allocate")
        
        # Add to active executions
        self.active_executions[execution_id] = context
        
        # Emit execution started event
        await self.event_bus.emit(Event(
            type="execution_started",
            data={
                "execution_id": execution_id,
                "test_name": context.test_name,
                "execution_mode": execution_mode.value,
                "node_id": node.node_id,
                "resource_quota": resource_quota.__dict__
            }
        ))
        
        try:
            # Execute based on mode
            if execution_mode == ExecutionMode.ASYNC:
                await self._execute_async(context, test_function, test_args, test_kwargs)
            elif execution_mode == ExecutionMode.THREAD:
                await self._execute_thread(context, test_function, test_args, test_kwargs)
            elif execution_mode == ExecutionMode.PROCESS:
                await self._execute_process(context, test_function, test_args, test_kwargs)
            elif execution_mode == ExecutionMode.CONTAINER:
                await self._execute_container(context, test_function, test_args, test_kwargs)
            
        except Exception as e:
            context.exit_code = 1
            context.stderr = str(e)
            self.logger.error(f"Execution {execution_id} failed: {e}")
        
        finally:
            # Complete execution
            await self._complete_execution(context, node)
        
        return context
    
    async def _execute_async(self, context: ExecutionContext, test_function: Callable,
                           test_args: Tuple, test_kwargs: Dict):
        """Execute test in async mode"""
        context.start_time = datetime.now()
        
        try:
            if asyncio.iscoroutinefunction(test_function):
                result = await test_function(*test_args, **test_kwargs)
            else:
                result = test_function(*test_args, **test_kwargs)
            
            context.stdout = str(result)
            context.exit_code = 0
            
        except Exception as e:
            context.stderr = str(e)
            context.exit_code = 1
            raise
        
        finally:
            context.end_time = datetime.now()
    
    async def _execute_thread(self, context: ExecutionContext, test_function: Callable,
                            test_args: Tuple, test_kwargs: Dict):
        """Execute test in thread mode"""
        context.start_time = datetime.now()
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.thread_pool,
                lambda: test_function(*test_args, **test_kwargs)
            )
            
            context.stdout = str(result)
            context.exit_code = 0
            
        except Exception as e:
            context.stderr = str(e)
            context.exit_code = 1
            raise
        
        finally:
            context.end_time = datetime.now()
    
    async def _execute_process(self, context: ExecutionContext, test_function: Callable,
                             test_args: Tuple, test_kwargs: Dict):
        """Execute test in process mode"""
        context.start_time = datetime.now()
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.process_pool,
                lambda: test_function(*test_args, **test_kwargs)
            )
            
            context.stdout = str(result)
            context.exit_code = 0
            
        except Exception as e:
            context.stderr = str(e)
            context.exit_code = 1
            raise
        
        finally:
            context.end_time = datetime.now()
    
    async def _execute_container(self, context: ExecutionContext, test_function: Callable,
                               test_args: Tuple, test_kwargs: Dict):
        """Execute test in container mode"""
        if not self.container_manager or not self.container_manager.is_available():
            raise RuntimeError("Container execution not available")
        
        context.start_time = datetime.now()
        
        try:
            # Create container
            container_id = await self.container_manager.create_container(
                context.execution_id,
                resource_quota=context.resource_quota,
                environment=context.environment
            )
            
            # Serialize test function and arguments
            test_data = {
                'function': pickle.dumps(test_function),
                'args': test_args,
                'kwargs': test_kwargs
            }
            
            # Create temporary script
            script_path = f"/tmp/test_script_{context.execution_id}.py"
            with open(script_path, 'w') as f:
                f.write(f"""
import pickle
import sys
import json

# Load test data
test_data = {repr(test_data)}

try:
    # Deserialize and execute
    test_function = pickle.loads(test_data['function'])
    result = test_function(*test_data['args'], **test_data['kwargs'])
    
    print(json.dumps({{'result': str(result), 'success': True}}))
    sys.exit(0)
    
except Exception as e:
    print(json.dumps({{'error': str(e), 'success': False}}))
    sys.exit(1)
""")
            
            # Execute in container
            exit_code, stdout, stderr = await self.container_manager.execute_in_container(
                context.execution_id,
                f"python {script_path}"
            )
            
            context.exit_code = exit_code
            context.stdout = stdout
            context.stderr = stderr
            
            # Parse result
            try:
                result_data = json.loads(stdout)
                if result_data.get('success'):
                    context.exit_code = 0
                else:
                    context.exit_code = 1
                    context.stderr = result_data.get('error', 'Unknown error')
            except json.JSONDecodeError:
                pass
            
        except Exception as e:
            context.stderr = str(e)
            context.exit_code = 1
            raise
        
        finally:
            context.end_time = datetime.now()
            
            # Cleanup
            try:
                await self.container_manager.cleanup_container(context.execution_id)
                if os.path.exists(script_path):
                    os.remove(script_path)
            except Exception as e:
                self.logger.error(f"Error cleaning up container: {e}")
    
    async def _complete_execution(self, context: ExecutionContext, node: WorkerNode):
        """Complete execution and cleanup"""
        # Deallocate resources
        self.load_balancer.deallocate_resources(node.node_id, context.resource_quota)
        
        # Record deallocation
        self.resource_monitor.record_allocation(
            context.execution_id, 
            context.resource_quota, 
            "deallocate"
        )
        
        # Remove from active executions
        if context.execution_id in self.active_executions:
            del self.active_executions[context.execution_id]
        
        # Add to history
        self.execution_history.append(context)
        
        # Emit completion event
        await self.event_bus.emit(Event(
            type="execution_completed",
            data={
                "execution_id": context.execution_id,
                "test_name": context.test_name,
                "exit_code": context.exit_code,
                "execution_time": (context.end_time - context.start_time).total_seconds() if context.end_time and context.start_time else 0,
                "resource_usage": context.actual_resources
            }
        ))
        
        self.logger.info(f"Execution {context.execution_id} completed with exit code {context.exit_code}")
    
    async def cancel_execution(self, execution_id: str):
        """Cancel an active execution"""
        if execution_id not in self.active_executions:
            return
        
        context = self.active_executions[execution_id]
        
        # Cleanup based on execution mode
        if context.execution_mode == ExecutionMode.CONTAINER and self.container_manager:
            await self.container_manager.cleanup_container(execution_id)
        
        # Mark as cancelled
        context.exit_code = -1
        context.stderr = "Execution cancelled"
        context.end_time = datetime.now()
        
        # Remove from active executions
        del self.active_executions[execution_id]
        
        # Add to history
        self.execution_history.append(context)
        
        self.logger.info(f"Execution {execution_id} cancelled")
    
    async def execute_batch(self, test_functions: List[Callable], 
                           execution_mode: ExecutionMode = ExecutionMode.ASYNC,
                           max_parallel: int = None) -> List[ExecutionContext]:
        """Execute a batch of tests in parallel"""
        max_parallel = max_parallel or self.max_workers
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def execute_single(test_func):
            async with semaphore:
                return await self.execute_test(test_func, execution_mode=execution_mode)
        
        tasks = [execute_single(func) for func in test_functions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        execution_contexts = []
        for result in results:
            if isinstance(result, Exception):
                # Create error context
                error_context = ExecutionContext(
                    execution_id=f"error_{int(time.time())}",
                    test_name="unknown",
                    execution_mode=execution_mode,
                    resource_quota=ResourceQuota(),
                    exit_code=1,
                    stderr=str(result)
                )
                execution_contexts.append(error_context)
            else:
                execution_contexts.append(result)
        
        return execution_contexts
    
    def get_execution_status(self, execution_id: str) -> Optional[ExecutionContext]:
        """Get status of an execution"""
        return self.active_executions.get(execution_id)
    
    def get_system_status(self) -> Dict:
        """Get system status"""
        return {
            'active_executions': len(self.active_executions),
            'total_executions': len(self.execution_history),
            'cluster_status': self.load_balancer.get_cluster_status(),
            'resource_availability': self.resource_monitor.get_resource_availability(),
            'metrics': self.metrics,
            'container_support': self.container_manager is not None and self.container_manager.is_available()
        }


# Example usage and testing
def example_cpu_intensive_test(duration: int = 5) -> str:
    """Example CPU-intensive test"""
    import time
    import math
    
    start_time = time.time()
    result = 0
    
    while time.time() - start_time < duration:
        result += math.sqrt(time.time())
    
    return f"CPU test completed: {result:.2f}"


def example_memory_intensive_test(size_mb: int = 100) -> str:
    """Example memory-intensive test"""
    # Allocate memory
    data = [0] * (size_mb * 1024 * 1024 // 8)  # 8 bytes per int
    
    # Do some work
    for i in range(len(data)):
        data[i] = i % 1000
    
    return f"Memory test completed: {len(data)} elements"


async def example_async_test(delay: float = 1.0) -> str:
    """Example async test"""
    await asyncio.sleep(delay)
    return f"Async test completed after {delay}s"


async def demo_parallel_executor():
    """Demonstration of parallel executor"""
    executor = ParallelExecutor(max_workers=4, enable_containers=False)
    
    # Start executor
    await executor.start()
    
    print("=== PARALLEL EXECUTOR DEMO ===")
    
    # Execute single test
    print("\n1. Single test execution:")
    context1 = await executor.execute_test(
        example_cpu_intensive_test,
        test_args=(2,),
        execution_mode=ExecutionMode.THREAD
    )
    print(f"Test: {context1.test_name}, Exit code: {context1.exit_code}")
    print(f"Output: {context1.stdout}")
    
    # Execute batch of tests
    print("\n2. Batch test execution:")
    test_functions = [
        lambda: example_cpu_intensive_test(1),
        lambda: example_memory_intensive_test(10),
        lambda: example_async_test(0.5)
    ]
    
    batch_results = await executor.execute_batch(
        test_functions,
        execution_mode=ExecutionMode.ASYNC,
        max_parallel=3
    )
    
    print(f"Batch completed: {len(batch_results)} tests")
    for i, result in enumerate(batch_results):
        print(f"  Test {i+1}: {result.test_name}, Exit code: {result.exit_code}")
    
    # Get system status
    print("\n3. System status:")
    status = executor.get_system_status()
    print(f"Active executions: {status['active_executions']}")
    print(f"Total executions: {status['total_executions']}")
    print(f"CPU utilization: {status['cluster_status']['utilization']['cpu']:.1f}%")
    print(f"Memory utilization: {status['cluster_status']['utilization']['memory']:.1f}%")
    
    # Wait a bit to see monitoring in action
    await asyncio.sleep(5)
    
    # Stop executor
    await executor.stop()
    
    print("\n=== DEMO COMPLETED ===")


if __name__ == "__main__":
    asyncio.run(demo_parallel_executor())