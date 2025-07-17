"""
Distributed Load Generator for Million TPS Testing
Agent Epsilon - Phase 3A Implementation

Capabilities:
- Distributed load generation across multiple nodes
- Real-time coordination and synchronization
- Dynamic load balancing
- Failure detection and recovery
"""

import asyncio
import time
import json
import uuid
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiohttp
import redis.asyncio as redis
import psutil
import numpy as np
from datetime import datetime, timedelta
import websockets
import multiprocessing as mp

logger = logging.getLogger(__name__)

@dataclass
class LoadNode:
    """Represents a single load generation node"""
    node_id: str
    hostname: str
    port: int
    max_connections: int
    status: str = "idle"
    current_load: int = 0
    last_heartbeat: float = 0
    
@dataclass
class LoadConfiguration:
    """Load test configuration"""
    target_tps: int
    duration_seconds: int
    ramp_up_seconds: int
    ramp_down_seconds: int
    test_type: str
    endpoints: List[str]
    request_distribution: Dict[str, float]
    

class DistributedLoadGenerator:
    """
    Distributed Load Generator for Million TPS Testing
    
    Orchestrates load generation across multiple nodes to achieve
    million TPS throughput with real-time coordination.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/3"):
        self.redis_url = redis_url
        self.redis_client = None
        self.node_id = str(uuid.uuid4())
        self.hostname = psutil.net_if_addrs().get('lo', [{'addr': 'localhost'}])[0]['addr']
        
        # Load generation parameters
        self.nodes: Dict[str, LoadNode] = {}
        self.is_coordinator = False
        self.load_config: Optional[LoadConfiguration] = None
        
        # Performance tracking
        self.total_requests = 0
        self.total_errors = 0
        self.latency_samples = []
        self.throughput_samples = []
        
        # Resource monitoring
        self.cpu_usage = []
        self.memory_usage = []
        self.network_usage = []
        
        # Coordination
        self.coordination_channel = "load_test_coordination"
        self.metrics_channel = "load_test_metrics"
        self.heartbeat_interval = 5.0
        
    async def initialize(self):
        """Initialize the distributed load generator"""
        logger.info(f"Initializing distributed load generator node {self.node_id}")
        
        # Connect to Redis
        self.redis_client = redis.from_url(self.redis_url)
        await self.redis_client.ping()
        
        # Register this node
        await self._register_node()
        
        # Start heartbeat
        asyncio.create_task(self._heartbeat_loop())
        
        # Start metrics collection
        asyncio.create_task(self._collect_metrics())
        
        logger.info(f"Node {self.node_id} initialized successfully")
        
    async def _register_node(self):
        """Register this node with the coordination system"""
        node_info = {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "port": 8080,
            "max_connections": 10000,
            "status": "ready",
            "timestamp": time.time()
        }
        
        await self.redis_client.hset(
            "load_nodes",
            self.node_id,
            json.dumps(node_info)
        )
        
        # Publish node registration
        await self.redis_client.publish(
            self.coordination_channel,
            json.dumps({
                "type": "node_registration",
                "node_id": self.node_id,
                "node_info": node_info
            })
        )
        
    async def _heartbeat_loop(self):
        """Maintain node heartbeat"""
        while True:
            try:
                # Update node status
                await self.redis_client.hset(
                    "load_nodes",
                    self.node_id,
                    json.dumps({
                        "node_id": self.node_id,
                        "hostname": self.hostname,
                        "status": "healthy",
                        "current_load": self.total_requests,
                        "timestamp": time.time(),
                        "cpu_usage": psutil.cpu_percent(),
                        "memory_usage": psutil.virtual_memory().percent
                    })
                )
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Heartbeat failed: {e}")
                await asyncio.sleep(1)
                
    async def _collect_metrics(self):
        """Collect system metrics"""
        while True:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                
                # Network stats
                net_io = psutil.net_io_counters()
                
                self.cpu_usage.append(cpu_percent)
                self.memory_usage.append(memory_percent)
                
                # Keep only last 1000 samples
                if len(self.cpu_usage) > 1000:
                    self.cpu_usage = self.cpu_usage[-1000:]
                    self.memory_usage = self.memory_usage[-1000:]
                
                # Publish metrics
                metrics = {
                    "node_id": self.node_id,
                    "timestamp": time.time(),
                    "cpu_usage": cpu_percent,
                    "memory_usage": memory_percent,
                    "total_requests": self.total_requests,
                    "total_errors": self.total_errors,
                    "throughput": len(self.throughput_samples)
                }
                
                await self.redis_client.publish(
                    self.metrics_channel,
                    json.dumps(metrics)
                )
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Metrics collection failed: {e}")
                await asyncio.sleep(1)
                
    async def become_coordinator(self):
        """Become the test coordinator"""
        logger.info(f"Node {self.node_id} becoming coordinator")
        
        self.is_coordinator = True
        
        # Subscribe to coordination channel
        pubsub = self.redis_client.pubsub()
        await pubsub.subscribe(self.coordination_channel)
        
        # Start coordination loop
        asyncio.create_task(self._coordination_loop(pubsub))
        
    async def _coordination_loop(self, pubsub):
        """Main coordination loop"""
        while self.is_coordinator:
            try:
                message = await pubsub.get_message(timeout=1)
                
                if message and message["type"] == "message":
                    data = json.loads(message["data"])
                    await self._handle_coordination_message(data)
                    
            except Exception as e:
                logger.error(f"Coordination loop error: {e}")
                await asyncio.sleep(1)
                
    async def _handle_coordination_message(self, data):
        """Handle coordination messages"""
        msg_type = data.get("type")
        
        if msg_type == "node_registration":
            await self._handle_node_registration(data)
        elif msg_type == "load_test_start":
            await self._handle_load_test_start(data)
        elif msg_type == "load_test_stop":
            await self._handle_load_test_stop(data)
        elif msg_type == "node_failure":
            await self._handle_node_failure(data)
            
    async def _handle_node_registration(self, data):
        """Handle new node registration"""
        node_info = data["node_info"]
        node_id = node_info["node_id"]
        
        self.nodes[node_id] = LoadNode(
            node_id=node_id,
            hostname=node_info["hostname"],
            port=node_info["port"],
            max_connections=node_info["max_connections"],
            status="ready",
            last_heartbeat=time.time()
        )
        
        logger.info(f"Registered node {node_id} - Total nodes: {len(self.nodes)}")
        
    async def start_distributed_load_test(self, config: LoadConfiguration):
        """Start distributed load test"""
        if not self.is_coordinator:
            raise ValueError("Only coordinator can start load tests")
            
        logger.info(f"Starting distributed load test: {config.target_tps} TPS")
        
        self.load_config = config
        
        # Calculate load distribution
        load_distribution = self._calculate_load_distribution(config)
        
        # Send start commands to all nodes
        for node_id, node_load in load_distribution.items():
            await self._send_load_command(node_id, node_load)
            
        # Start monitoring
        asyncio.create_task(self._monitor_test_progress())
        
    def _calculate_load_distribution(self, config: LoadConfiguration) -> Dict[str, int]:
        """Calculate load distribution across nodes"""
        active_nodes = [n for n in self.nodes.values() if n.status == "ready"]
        
        if not active_nodes:
            raise ValueError("No active nodes available")
            
        # Equal distribution with capacity consideration
        total_capacity = sum(node.max_connections for node in active_nodes)
        
        distribution = {}
        remaining_load = config.target_tps
        
        for node in active_nodes:
            node_percentage = node.max_connections / total_capacity
            node_load = int(config.target_tps * node_percentage)
            
            distribution[node.node_id] = min(node_load, remaining_load)
            remaining_load -= distribution[node.node_id]
            
            if remaining_load <= 0:
                break
                
        return distribution
        
    async def _send_load_command(self, node_id: str, target_load: int):
        """Send load command to specific node"""
        command = {
            "type": "start_load",
            "node_id": node_id,
            "target_tps": target_load,
            "config": {
                "duration_seconds": self.load_config.duration_seconds,
                "ramp_up_seconds": self.load_config.ramp_up_seconds,
                "endpoints": self.load_config.endpoints,
                "request_distribution": self.load_config.request_distribution
            }
        }
        
        await self.redis_client.publish(
            self.coordination_channel,
            json.dumps(command)
        )
        
    async def _monitor_test_progress(self):
        """Monitor test progress and performance"""
        start_time = time.time()
        
        while time.time() - start_time < self.load_config.duration_seconds:
            # Collect metrics from all nodes
            metrics = await self._collect_node_metrics()
            
            # Calculate aggregate metrics
            total_tps = sum(m.get("throughput", 0) for m in metrics)
            total_errors = sum(m.get("total_errors", 0) for m in metrics)
            avg_latency = np.mean([m.get("avg_latency", 0) for m in metrics if m.get("avg_latency")])
            
            # Log progress
            elapsed = time.time() - start_time
            logger.info(f"Progress: {elapsed:.1f}s - TPS: {total_tps:,.0f} "
                       f"Errors: {total_errors:,} Latency: {avg_latency:.2f}ms")
            
            # Check for failures
            await self._check_node_health(metrics)
            
            await asyncio.sleep(5)
            
    async def _collect_node_metrics(self) -> List[Dict]:
        """Collect metrics from all nodes"""
        metrics = []
        
        for node_id in self.nodes:
            try:
                node_metrics = await self.redis_client.hget("node_metrics", node_id)
                if node_metrics:
                    metrics.append(json.loads(node_metrics))
                    
            except Exception as e:
                logger.warning(f"Failed to collect metrics from node {node_id}: {e}")
                
        return metrics
        
    async def _check_node_health(self, metrics: List[Dict]):
        """Check node health and handle failures"""
        current_time = time.time()
        
        for node_id, node in self.nodes.items():
            # Check heartbeat
            if current_time - node.last_heartbeat > self.heartbeat_interval * 3:
                logger.warning(f"Node {node_id} heartbeat timeout")
                await self._handle_node_failure({"node_id": node_id})
                
            # Check resource usage
            node_metrics = next((m for m in metrics if m.get("node_id") == node_id), None)
            
            if node_metrics:
                cpu_usage = node_metrics.get("cpu_usage", 0)
                memory_usage = node_metrics.get("memory_usage", 0)
                
                if cpu_usage > 90 or memory_usage > 90:
                    logger.warning(f"Node {node_id} resource usage critical: "
                                 f"CPU: {cpu_usage}% Memory: {memory_usage}%")
                    
    async def _handle_node_failure(self, data):
        """Handle node failure"""
        node_id = data["node_id"]
        
        if node_id in self.nodes:
            logger.error(f"Node {node_id} failed - redistributing load")
            
            failed_node = self.nodes[node_id]
            failed_load = failed_node.current_load
            
            # Remove failed node
            del self.nodes[node_id]
            
            # Redistribute load to healthy nodes
            if failed_load > 0 and self.nodes:
                await self._redistribute_load(failed_load)
                
    async def _redistribute_load(self, additional_load: int):
        """Redistribute load after node failure"""
        healthy_nodes = [n for n in self.nodes.values() if n.status == "ready"]
        
        if not healthy_nodes:
            logger.error("No healthy nodes available for load redistribution")
            return
            
        # Distribute additional load
        load_per_node = additional_load // len(healthy_nodes)
        
        for node in healthy_nodes:
            new_load = node.current_load + load_per_node
            
            await self._send_load_command(node.node_id, new_load)
            
    async def execute_load_worker(self, target_tps: int, config: Dict):
        """Execute load generation as a worker node"""
        logger.info(f"Starting load worker: {target_tps} TPS")
        
        # Calculate request interval
        request_interval = 1.0 / target_tps if target_tps > 0 else 0.1
        
        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(limit=1000, limit_per_host=100)
        
        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=connector
        ) as session:
            
            # Start load generation
            await self._generate_load(session, request_interval, config)
            
    async def _generate_load(self, session: aiohttp.ClientSession, 
                           interval: float, config: Dict):
        """Generate load requests"""
        endpoints = config["endpoints"]
        request_distribution = config["request_distribution"]
        
        # Create request tasks
        tasks = []
        
        for i in range(1000):  # Create 1000 concurrent workers
            task = asyncio.create_task(
                self._request_worker(session, endpoints, request_distribution, interval)
            )
            tasks.append(task)
            
        # Wait for all tasks
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def _request_worker(self, session: aiohttp.ClientSession,
                            endpoints: List[str], distribution: Dict[str, float],
                            interval: float):
        """Individual request worker"""
        while True:
            try:
                # Select endpoint based on distribution
                endpoint = np.random.choice(endpoints, p=list(distribution.values()))
                
                # Record start time
                start_time = time.perf_counter()
                
                # Make request
                async with session.get(endpoint) as response:
                    await response.read()
                    
                    # Record latency
                    latency = (time.perf_counter() - start_time) * 1000
                    self.latency_samples.append(latency)
                    
                    # Update counters
                    self.total_requests += 1
                    
                    if response.status >= 400:
                        self.total_errors += 1
                        
                # Maintain interval
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.total_errors += 1
                logger.debug(f"Request error: {e}")
                await asyncio.sleep(interval)
                
    async def generate_million_tps_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        if not self.is_coordinator:
            raise ValueError("Only coordinator can generate reports")
            
        # Collect final metrics
        final_metrics = await self._collect_node_metrics()
        
        # Calculate aggregate statistics
        total_requests = sum(m.get("total_requests", 0) for m in final_metrics)
        total_errors = sum(m.get("total_errors", 0) for m in final_metrics)
        
        # Latency statistics
        all_latencies = []
        for metrics in final_metrics:
            if metrics.get("latency_samples"):
                all_latencies.extend(metrics["latency_samples"])
                
        latency_stats = {}
        if all_latencies:
            latency_stats = {
                "min_ms": float(np.min(all_latencies)),
                "max_ms": float(np.max(all_latencies)),
                "mean_ms": float(np.mean(all_latencies)),
                "median_ms": float(np.median(all_latencies)),
                "p95_ms": float(np.percentile(all_latencies, 95)),
                "p99_ms": float(np.percentile(all_latencies, 99)),
                "p99_9_ms": float(np.percentile(all_latencies, 99.9))
            }
            
        # Resource usage statistics
        resource_stats = {
            "max_cpu_usage": max(m.get("cpu_usage", 0) for m in final_metrics),
            "max_memory_usage": max(m.get("memory_usage", 0) for m in final_metrics),
            "avg_cpu_usage": np.mean([m.get("cpu_usage", 0) for m in final_metrics]),
            "avg_memory_usage": np.mean([m.get("memory_usage", 0) for m in final_metrics])
        }
        
        # Calculate actual TPS
        duration = self.load_config.duration_seconds if self.load_config else 60
        actual_tps = total_requests / duration
        
        # Determine test result
        target_tps = self.load_config.target_tps if self.load_config else 1000000
        success_rate = (total_requests - total_errors) / total_requests if total_requests > 0 else 0
        
        test_passed = (
            actual_tps >= target_tps * 0.95 and  # 95% of target TPS
            success_rate >= 0.999 and           # 99.9% success rate
            latency_stats.get("p99_ms", 0) <= 100  # P99 latency under 100ms
        )
        
        return {
            "test_configuration": {
                "target_tps": target_tps,
                "duration_seconds": duration,
                "total_nodes": len(self.nodes),
                "endpoints": self.load_config.endpoints if self.load_config else []
            },
            "performance_results": {
                "actual_tps": actual_tps,
                "total_requests": total_requests,
                "total_errors": total_errors,
                "success_rate": success_rate,
                "latency_statistics": latency_stats,
                "resource_usage": resource_stats
            },
            "test_validation": {
                "passed": test_passed,
                "tps_target_met": actual_tps >= target_tps * 0.95,
                "success_rate_met": success_rate >= 0.999,
                "latency_target_met": latency_stats.get("p99_ms", 0) <= 100
            },
            "node_performance": final_metrics,
            "timestamp": datetime.utcnow().isoformat(),
            "test_id": str(uuid.uuid4())
        }
        
    async def cleanup(self):
        """Cleanup resources"""
        if self.redis_client:
            # Unregister node
            await self.redis_client.hdel("load_nodes", self.node_id)
            
            # Close connection
            await self.redis_client.close()
            
        logger.info(f"Node {self.node_id} cleanup completed")