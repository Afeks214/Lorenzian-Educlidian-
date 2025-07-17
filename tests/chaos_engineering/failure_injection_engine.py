"""
Failure Injection Engine for Advanced Chaos Engineering
=====================================================

This module provides sophisticated failure injection capabilities for comprehensive
chaos engineering testing. It supports gradual failure escalation, realistic
failure patterns, and coordinated multi-component failures.

Key Features:
- Gradual failure escalation with controllable blast radius
- Realistic failure patterns based on production scenarios
- Coordinated multi-component failures for complex testing
- Automatic rollback and safety mechanisms
- Detailed failure tracking and analysis
"""

import asyncio
import time
import random
import logging
import json
import threading
import subprocess
import psutil
import signal
import socket
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import aioredis
import httpx
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures that can be injected."""
    # Network failures
    NETWORK_PARTITION = "network_partition"
    NETWORK_LATENCY = "network_latency"
    NETWORK_PACKET_LOSS = "network_packet_loss"
    NETWORK_BANDWIDTH_LIMIT = "network_bandwidth_limit"
    DNS_FAILURE = "dns_failure"
    
    # Service failures
    SERVICE_CRASH = "service_crash"
    SERVICE_HANG = "service_hang"
    SERVICE_SLOW_RESPONSE = "service_slow_response"
    SERVICE_ERROR_INJECTION = "service_error_injection"
    API_TIMEOUT = "api_timeout"
    
    # Resource failures
    MEMORY_LEAK = "memory_leak"
    CPU_SPIKE = "cpu_spike"
    DISK_FULL = "disk_full"
    FILE_DESCRIPTOR_EXHAUSTION = "file_descriptor_exhaustion"
    THREAD_POOL_EXHAUSTION = "thread_pool_exhaustion"
    
    # Data failures
    DATA_CORRUPTION = "data_corruption"
    DATABASE_CONNECTION_LOSS = "database_connection_loss"
    CACHE_INVALIDATION = "cache_invalidation"
    MESSAGE_QUEUE_FAILURE = "message_queue_failure"
    
    # Infrastructure failures
    CONTAINER_KILL = "container_kill"
    VOLUME_UNMOUNT = "volume_unmount"
    CONFIGURATION_CORRUPTION = "configuration_corruption"
    SECURITY_CERTIFICATE_EXPIRY = "security_certificate_expiry"


class FailureEscalationLevel(Enum):
    """Escalation levels for failure injection."""
    MINIMAL = "minimal"        # 1-5% impact
    LOW = "low"               # 5-15% impact
    MEDIUM = "medium"         # 15-40% impact
    HIGH = "high"             # 40-70% impact
    CRITICAL = "critical"     # 70-100% impact


class FailurePattern(Enum):
    """Common failure patterns observed in production."""
    GRADUAL_DEGRADATION = "gradual_degradation"
    SUDDEN_FAILURE = "sudden_failure"
    INTERMITTENT_FAILURE = "intermittent_failure"
    CASCADING_FAILURE = "cascading_failure"
    RECOVERY_FAILURE = "recovery_failure"
    BROWN_OUT = "brown_out"
    THUNDERING_HERD = "thundering_herd"


@dataclass
class FailureSpec:
    """Specification for a failure injection."""
    failure_type: FailureType
    target_component: str
    escalation_level: FailureEscalationLevel
    pattern: FailurePattern
    duration_seconds: int
    blast_radius: float  # 0.0 to 1.0
    recovery_time_seconds: int
    side_effects: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    rollback_strategy: str = ""
    safety_limits: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailureExecution:
    """Tracking information for an active failure injection."""
    spec: FailureSpec
    injection_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "ACTIVE"
    escalation_steps: List[Dict[str, Any]] = field(default_factory=list)
    side_effects_triggered: List[str] = field(default_factory=list)
    cleanup_actions: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class NetworkFailureInjector:
    """Specialized injector for network-related failures."""
    
    def __init__(self):
        self.active_rules = {}
        self.rule_lock = threading.Lock()
        
    async def inject_network_partition(self, spec: FailureSpec, injection_id: str) -> Dict[str, Any]:
        """Inject network partition between components."""
        try:
            target_component = spec.target_component
            
            # Map component to port/IP
            component_config = self._get_component_network_config(target_component)
            if not component_config:
                return {"success": False, "error": f"Unknown component: {target_component}"}
            
            # Calculate partition scope based on escalation level
            partition_scope = self._calculate_partition_scope(spec.escalation_level)
            
            # Apply network rules
            rules_applied = []
            
            for target_port in component_config.get("ports", []):
                if spec.pattern == FailurePattern.GRADUAL_DEGRADATION:
                    # Gradually increase packet loss
                    for step in range(1, 6):
                        loss_percentage = (step * 20) * partition_scope
                        rule = await self._apply_packet_loss_rule(target_port, loss_percentage)
                        rules_applied.append(rule)
                        await asyncio.sleep(spec.duration_seconds / 10)
                        
                elif spec.pattern == FailurePattern.SUDDEN_FAILURE:
                    # Immediate complete partition
                    rule = await self._apply_network_block_rule(target_port)
                    rules_applied.append(rule)
                    
                elif spec.pattern == FailurePattern.INTERMITTENT_FAILURE:
                    # Intermittent connectivity issues
                    await self._apply_intermittent_partition(target_port, spec.duration_seconds)
            
            with self.rule_lock:
                self.active_rules[injection_id] = {
                    "rules": rules_applied,
                    "component": target_component,
                    "ports": component_config.get("ports", [])
                }
            
            return {
                "success": True,
                "injection_id": injection_id,
                "rules_applied": len(rules_applied),
                "partition_scope": partition_scope
            }
            
        except Exception as e:
            logger.error(f"Network partition injection failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_component_network_config(self, component: str) -> Dict[str, Any]:
        """Get network configuration for a component."""
        config_map = {
            "tactical": {"ports": [8001], "service": "tactical-marl"},
            "strategic": {"ports": [8002], "service": "strategic-marl"},
            "redis": {"ports": [6379], "service": "redis"},
            "database": {"ports": [5432], "service": "postgresql"},
            "api": {"ports": [8000], "service": "api-gateway"},
            "event_bus": {"ports": [9092], "service": "kafka"}
        }
        return config_map.get(component, {})
    
    def _calculate_partition_scope(self, escalation_level: FailureEscalationLevel) -> float:
        """Calculate partition scope based on escalation level."""
        scope_map = {
            FailureEscalationLevel.MINIMAL: 0.1,
            FailureEscalationLevel.LOW: 0.3,
            FailureEscalationLevel.MEDIUM: 0.5,
            FailureEscalationLevel.HIGH: 0.8,
            FailureEscalationLevel.CRITICAL: 1.0
        }
        return scope_map.get(escalation_level, 0.5)
    
    async def _apply_packet_loss_rule(self, port: int, loss_percentage: float) -> Dict[str, Any]:
        """Apply packet loss rule to a specific port."""
        try:
            # Use tc (traffic control) for packet loss
            interface = self._get_network_interface()
            
            # Create qdisc with packet loss
            cmd = [
                "tc", "qdisc", "add", "dev", interface, "root", "handle", "1:",
                "netem", "loss", f"{loss_percentage}%"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            return {
                "type": "packet_loss",
                "port": port,
                "loss_percentage": loss_percentage,
                "interface": interface,
                "command": " ".join(cmd),
                "success": result.returncode == 0,
                "error": result.stderr if result.returncode != 0 else None
            }
            
        except Exception as e:
            logger.error(f"Failed to apply packet loss rule: {e}")
            return {"type": "packet_loss", "success": False, "error": str(e)}
    
    async def _apply_network_block_rule(self, port: int) -> Dict[str, Any]:
        """Apply network block rule to a specific port."""
        try:
            # Use iptables to block traffic
            cmd = ["iptables", "-A", "INPUT", "-p", "tcp", "--dport", str(port), "-j", "DROP"]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            return {
                "type": "network_block",
                "port": port,
                "command": " ".join(cmd),
                "success": result.returncode == 0,
                "error": result.stderr if result.returncode != 0 else None
            }
            
        except Exception as e:
            logger.error(f"Failed to apply network block rule: {e}")
            return {"type": "network_block", "success": False, "error": str(e)}
    
    async def _apply_intermittent_partition(self, port: int, duration_seconds: int):
        """Apply intermittent network partition."""
        try:
            end_time = time.time() + duration_seconds
            
            while time.time() < end_time:
                # Block for 5-15 seconds
                block_duration = random.uniform(5, 15)
                await self._apply_network_block_rule(port)
                await asyncio.sleep(block_duration)
                
                # Unblock for 10-30 seconds
                unblock_duration = random.uniform(10, 30)
                await self._remove_network_block_rule(port)
                await asyncio.sleep(unblock_duration)
                
        except Exception as e:
            logger.error(f"Intermittent partition failed: {e}")
    
    async def _remove_network_block_rule(self, port: int):
        """Remove network block rule."""
        try:
            cmd = ["iptables", "-D", "INPUT", "-p", "tcp", "--dport", str(port), "-j", "DROP"]
            subprocess.run(cmd, capture_output=True, text=True)
        except Exception as e:
            logger.error(f"Failed to remove network block rule: {e}")
    
    def _get_network_interface(self) -> str:
        """Get primary network interface."""
        try:
            # Get default route interface
            result = subprocess.run(
                ["ip", "route", "show", "default"],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                # Parse output to get interface
                parts = result.stdout.strip().split()
                if "dev" in parts:
                    dev_index = parts.index("dev")
                    if dev_index + 1 < len(parts):
                        return parts[dev_index + 1]
            
            return "eth0"  # Default fallback
            
        except Exception:
            return "eth0"
    
    async def cleanup_injection(self, injection_id: str) -> bool:
        """Clean up network injection."""
        try:
            with self.rule_lock:
                if injection_id not in self.active_rules:
                    return False
                
                rule_info = self.active_rules[injection_id]
                
                # Remove iptables rules
                for port in rule_info.get("ports", []):
                    await self._remove_network_block_rule(port)
                
                # Remove tc rules
                interface = self._get_network_interface()
                subprocess.run(["tc", "qdisc", "del", "dev", interface, "root"], capture_output=True)
                
                # Remove from active rules
                del self.active_rules[injection_id]
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup network injection: {e}")
            return False


class ServiceFailureInjector:
    """Specialized injector for service-related failures."""
    
    def __init__(self):
        self.active_injections = {}
        self.injection_lock = threading.Lock()
        
    async def inject_service_crash(self, spec: FailureSpec, injection_id: str) -> Dict[str, Any]:
        """Inject service crash with configurable pattern."""
        try:
            target_component = spec.target_component
            
            # Find target processes
            target_processes = self._find_target_processes(target_component)
            
            if not target_processes:
                return {"success": False, "error": f"No processes found for component: {target_component}"}
            
            crash_results = []
            
            if spec.pattern == FailurePattern.GRADUAL_DEGRADATION:
                # Gradually kill processes
                for i, pid in enumerate(target_processes):
                    await asyncio.sleep(spec.duration_seconds / len(target_processes))
                    result = await self._kill_process(pid, signal.SIGTERM)
                    crash_results.append(result)
                    
            elif spec.pattern == FailurePattern.SUDDEN_FAILURE:
                # Kill all processes immediately
                for pid in target_processes:
                    result = await self._kill_process(pid, signal.SIGKILL)
                    crash_results.append(result)
                    
            elif spec.pattern == FailurePattern.INTERMITTENT_FAILURE:
                # Intermittent crashes
                await self._apply_intermittent_crashes(target_processes, spec.duration_seconds)
                
            elif spec.pattern == FailurePattern.CASCADING_FAILURE:
                # Trigger cascading failures
                await self._trigger_cascading_failures(target_component, target_processes)
            
            with self.injection_lock:
                self.active_injections[injection_id] = {
                    "component": target_component,
                    "processes": target_processes,
                    "crash_results": crash_results,
                    "pattern": spec.pattern
                }
            
            return {
                "success": True,
                "injection_id": injection_id,
                "processes_affected": len(target_processes),
                "crash_results": crash_results
            }
            
        except Exception as e:
            logger.error(f"Service crash injection failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def inject_service_hang(self, spec: FailureSpec, injection_id: str) -> Dict[str, Any]:
        """Inject service hang/deadlock conditions."""
        try:
            target_component = spec.target_component
            
            # Create hang conditions based on component type
            if target_component == "tactical":
                hang_result = await self._create_tactical_hang()
            elif target_component == "strategic":
                hang_result = await self._create_strategic_hang()
            elif target_component == "database":
                hang_result = await self._create_database_hang()
            else:
                hang_result = await self._create_generic_hang(target_component)
            
            with self.injection_lock:
                self.active_injections[injection_id] = {
                    "component": target_component,
                    "hang_type": hang_result.get("hang_type"),
                    "hang_details": hang_result
                }
            
            return {
                "success": True,
                "injection_id": injection_id,
                "hang_result": hang_result
            }
            
        except Exception as e:
            logger.error(f"Service hang injection failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def inject_slow_response(self, spec: FailureSpec, injection_id: str) -> Dict[str, Any]:
        """Inject slow response times."""
        try:
            target_component = spec.target_component
            
            # Calculate delay based on escalation level
            base_delay = self._calculate_response_delay(spec.escalation_level)
            
            # Apply delay injection
            delay_result = await self._apply_response_delay(target_component, base_delay, spec.pattern)
            
            with self.injection_lock:
                self.active_injections[injection_id] = {
                    "component": target_component,
                    "delay_ms": base_delay,
                    "delay_result": delay_result
                }
            
            return {
                "success": True,
                "injection_id": injection_id,
                "delay_ms": base_delay,
                "delay_result": delay_result
            }
            
        except Exception as e:
            logger.error(f"Slow response injection failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _find_target_processes(self, component: str) -> List[int]:
        """Find processes for a target component."""
        try:
            target_processes = []
            
            # Component to process name mapping
            process_patterns = {
                "tactical": ["tactical_main", "tactical-marl"],
                "strategic": ["strategic_main", "strategic-marl"],
                "redis": ["redis-server"],
                "database": ["postgres"],
                "api": ["api_main", "uvicorn"],
                "event_bus": ["kafka"]
            }
            
            patterns = process_patterns.get(component, [component])
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    proc_info = proc.info
                    cmdline = ' '.join(proc_info['cmdline'] or [])
                    
                    for pattern in patterns:
                        if pattern.lower() in cmdline.lower() or pattern.lower() in proc_info['name'].lower():
                            target_processes.append(proc_info['pid'])
                            break
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return target_processes
            
        except Exception as e:
            logger.error(f"Failed to find target processes: {e}")
            return []
    
    async def _kill_process(self, pid: int, signal_type: int) -> Dict[str, Any]:
        """Kill a specific process."""
        try:
            os.kill(pid, signal_type)
            return {"pid": pid, "signal": signal_type, "success": True}
        except ProcessLookupError:
            return {"pid": pid, "signal": signal_type, "success": False, "error": "Process not found"}
        except PermissionError:
            return {"pid": pid, "signal": signal_type, "success": False, "error": "Permission denied"}
        except Exception as e:
            return {"pid": pid, "signal": signal_type, "success": False, "error": str(e)}
    
    async def _apply_intermittent_crashes(self, processes: List[int], duration_seconds: int):
        """Apply intermittent crashes to processes."""
        try:
            end_time = time.time() + duration_seconds
            
            while time.time() < end_time:
                # Randomly select a process to crash
                if processes:
                    pid = random.choice(processes)
                    await self._kill_process(pid, signal.SIGTERM)
                    
                    # Wait before next crash
                    await asyncio.sleep(random.uniform(10, 30))
                    
        except Exception as e:
            logger.error(f"Intermittent crashes failed: {e}")
    
    async def _trigger_cascading_failures(self, component: str, processes: List[int]):
        """Trigger cascading failures."""
        try:
            # Kill primary process first
            if processes:
                await self._kill_process(processes[0], signal.SIGKILL)
                
            # Wait for dependent services to fail
            await asyncio.sleep(5)
            
            # Trigger failures in dependent components
            dependent_components = self._get_dependent_components(component)
            
            for dependent in dependent_components:
                dependent_processes = self._find_target_processes(dependent)
                if dependent_processes:
                    await self._kill_process(dependent_processes[0], signal.SIGTERM)
                    await asyncio.sleep(2)
                    
        except Exception as e:
            logger.error(f"Cascading failures failed: {e}")
    
    def _get_dependent_components(self, component: str) -> List[str]:
        """Get components that depend on the given component."""
        dependencies = {
            "redis": ["tactical", "strategic", "api"],
            "database": ["api", "strategic"],
            "tactical": ["strategic"],
            "api": [],
            "strategic": []
        }
        return dependencies.get(component, [])
    
    async def _create_tactical_hang(self) -> Dict[str, Any]:
        """Create hang condition in tactical service."""
        try:
            # Send hang signal to tactical service
            result = await self._send_hang_signal("tactical", "deadlock_simulation")
            return {"hang_type": "tactical_deadlock", "result": result}
        except Exception as e:
            return {"hang_type": "tactical_deadlock", "error": str(e)}
    
    async def _create_strategic_hang(self) -> Dict[str, Any]:
        """Create hang condition in strategic service."""
        try:
            result = await self._send_hang_signal("strategic", "consensus_hang")
            return {"hang_type": "strategic_consensus_hang", "result": result}
        except Exception as e:
            return {"hang_type": "strategic_consensus_hang", "error": str(e)}
    
    async def _create_database_hang(self) -> Dict[str, Any]:
        """Create hang condition in database."""
        try:
            # Create long-running query
            result = await self._execute_long_running_query()
            return {"hang_type": "database_long_query", "result": result}
        except Exception as e:
            return {"hang_type": "database_long_query", "error": str(e)}
    
    async def _create_generic_hang(self, component: str) -> Dict[str, Any]:
        """Create generic hang condition."""
        try:
            # Send SIGSTOP to pause process
            processes = self._find_target_processes(component)
            if processes:
                pid = processes[0]
                os.kill(pid, signal.SIGSTOP)
                return {"hang_type": "process_pause", "pid": pid, "success": True}
            else:
                return {"hang_type": "process_pause", "success": False, "error": "No processes found"}
        except Exception as e:
            return {"hang_type": "process_pause", "error": str(e)}
    
    async def _send_hang_signal(self, component: str, hang_type: str) -> Dict[str, Any]:
        """Send hang signal to component."""
        try:
            # This would integrate with service-specific hang mechanisms
            # For now, simulate by sending HTTP request to hang endpoint
            service_ports = {"tactical": 8001, "strategic": 8002}
            port = service_ports.get(component)
            
            if port:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"http://localhost:{port}/chaos/hang",
                        json={"hang_type": hang_type},
                        timeout=1.0
                    )
                    return {"success": response.status_code == 200}
            
            return {"success": False, "error": "Unknown component"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_long_running_query(self) -> Dict[str, Any]:
        """Execute a long-running database query."""
        try:
            # This would connect to actual database and run expensive query
            # For now, simulate
            return {"success": True, "query": "SELECT pg_sleep(300)"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _calculate_response_delay(self, escalation_level: FailureEscalationLevel) -> int:
        """Calculate response delay in milliseconds."""
        delay_map = {
            FailureEscalationLevel.MINIMAL: 100,    # 100ms
            FailureEscalationLevel.LOW: 500,        # 500ms
            FailureEscalationLevel.MEDIUM: 2000,    # 2s
            FailureEscalationLevel.HIGH: 10000,     # 10s
            FailureEscalationLevel.CRITICAL: 30000  # 30s
        }
        return delay_map.get(escalation_level, 1000)
    
    async def _apply_response_delay(self, component: str, delay_ms: int, pattern: FailurePattern) -> Dict[str, Any]:
        """Apply response delay to component."""
        try:
            # This would integrate with service-specific delay mechanisms
            service_ports = {"tactical": 8001, "strategic": 8002, "api": 8000}
            port = service_ports.get(component)
            
            if port:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"http://localhost:{port}/chaos/delay",
                        json={"delay_ms": delay_ms, "pattern": pattern.value},
                        timeout=2.0
                    )
                    return {"success": response.status_code == 200, "delay_ms": delay_ms}
            
            return {"success": False, "error": "Unknown component"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def cleanup_injection(self, injection_id: str) -> bool:
        """Clean up service injection."""
        try:
            with self.injection_lock:
                if injection_id not in self.active_injections:
                    return False
                
                injection_info = self.active_injections[injection_id]
                component = injection_info.get("component")
                
                # Resume paused processes
                if "hang_type" in injection_info:
                    processes = self._find_target_processes(component)
                    for pid in processes:
                        try:
                            os.kill(pid, signal.SIGCONT)
                        except ProcessLookupError:
                            pass
                
                # Remove delay injection
                if "delay_ms" in injection_info:
                    await self._remove_response_delay(component)
                
                # Remove from active injections
                del self.active_injections[injection_id]
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup service injection: {e}")
            return False
    
    async def _remove_response_delay(self, component: str):
        """Remove response delay from component."""
        try:
            service_ports = {"tactical": 8001, "strategic": 8002, "api": 8000}
            port = service_ports.get(component)
            
            if port:
                async with httpx.AsyncClient() as client:
                    await client.delete(
                        f"http://localhost:{port}/chaos/delay",
                        timeout=2.0
                    )
        except Exception as e:
            logger.error(f"Failed to remove response delay: {e}")


class ResourceFailureInjector:
    """Specialized injector for resource-related failures."""
    
    def __init__(self):
        self.active_injections = {}
        self.injection_lock = threading.Lock()
        
    async def inject_memory_leak(self, spec: FailureSpec, injection_id: str) -> Dict[str, Any]:
        """Inject memory leak simulation."""
        try:
            # Calculate memory leak rate based on escalation level
            leak_rate_mb = self._calculate_leak_rate(spec.escalation_level)
            
            # Start memory leak simulation
            leak_task = asyncio.create_task(
                self._simulate_memory_leak(leak_rate_mb, spec.duration_seconds)
            )
            
            with self.injection_lock:
                self.active_injections[injection_id] = {
                    "type": "memory_leak",
                    "leak_rate_mb": leak_rate_mb,
                    "leak_task": leak_task,
                    "memory_blocks": []
                }
            
            return {
                "success": True,
                "injection_id": injection_id,
                "leak_rate_mb": leak_rate_mb
            }
            
        except Exception as e:
            logger.error(f"Memory leak injection failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def inject_cpu_spike(self, spec: FailureSpec, injection_id: str) -> Dict[str, Any]:
        """Inject CPU spike simulation."""
        try:
            # Calculate CPU load based on escalation level
            cpu_load = self._calculate_cpu_load(spec.escalation_level)
            
            # Start CPU spike simulation
            cpu_tasks = []
            for i in range(psutil.cpu_count()):
                task = asyncio.create_task(
                    self._simulate_cpu_spike(cpu_load, spec.duration_seconds)
                )
                cpu_tasks.append(task)
            
            with self.injection_lock:
                self.active_injections[injection_id] = {
                    "type": "cpu_spike",
                    "cpu_load": cpu_load,
                    "cpu_tasks": cpu_tasks
                }
            
            return {
                "success": True,
                "injection_id": injection_id,
                "cpu_load": cpu_load,
                "cpu_tasks": len(cpu_tasks)
            }
            
        except Exception as e:
            logger.error(f"CPU spike injection failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def inject_disk_full(self, spec: FailureSpec, injection_id: str) -> Dict[str, Any]:
        """Inject disk full simulation."""
        try:
            # Calculate disk usage based on escalation level
            target_usage = self._calculate_disk_usage(spec.escalation_level)
            
            # Create large files to fill disk
            fill_result = await self._fill_disk_space(target_usage)
            
            with self.injection_lock:
                self.active_injections[injection_id] = {
                    "type": "disk_full",
                    "target_usage": target_usage,
                    "created_files": fill_result.get("created_files", []),
                    "bytes_written": fill_result.get("bytes_written", 0)
                }
            
            return {
                "success": True,
                "injection_id": injection_id,
                "target_usage": target_usage,
                "fill_result": fill_result
            }
            
        except Exception as e:
            logger.error(f"Disk full injection failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _calculate_leak_rate(self, escalation_level: FailureEscalationLevel) -> int:
        """Calculate memory leak rate in MB per second."""
        rate_map = {
            FailureEscalationLevel.MINIMAL: 1,      # 1 MB/s
            FailureEscalationLevel.LOW: 5,          # 5 MB/s
            FailureEscalationLevel.MEDIUM: 20,      # 20 MB/s
            FailureEscalationLevel.HIGH: 50,        # 50 MB/s
            FailureEscalationLevel.CRITICAL: 100    # 100 MB/s
        }
        return rate_map.get(escalation_level, 10)
    
    async def _simulate_memory_leak(self, leak_rate_mb: int, duration_seconds: int):
        """Simulate memory leak."""
        try:
            memory_blocks = []
            end_time = time.time() + duration_seconds
            
            while time.time() < end_time:
                # Allocate memory block
                block_size = leak_rate_mb * 1024 * 1024  # Convert to bytes
                memory_block = bytearray(block_size)
                memory_blocks.append(memory_block)
                
                # Wait 1 second before next allocation
                await asyncio.sleep(1)
                
                # Check memory limits
                memory_info = psutil.virtual_memory()
                if memory_info.percent > 95:  # Safety limit
                    logger.warning("Memory safety limit reached, stopping leak simulation")
                    break
            
            # Keep reference to prevent garbage collection
            await asyncio.sleep(duration_seconds)
            
        except Exception as e:
            logger.error(f"Memory leak simulation failed: {e}")
    
    def _calculate_cpu_load(self, escalation_level: FailureEscalationLevel) -> float:
        """Calculate CPU load percentage."""
        load_map = {
            FailureEscalationLevel.MINIMAL: 0.1,    # 10%
            FailureEscalationLevel.LOW: 0.3,        # 30%
            FailureEscalationLevel.MEDIUM: 0.6,     # 60%
            FailureEscalationLevel.HIGH: 0.9,       # 90%
            FailureEscalationLevel.CRITICAL: 1.0    # 100%
        }
        return load_map.get(escalation_level, 0.5)
    
    async def _simulate_cpu_spike(self, cpu_load: float, duration_seconds: int):
        """Simulate CPU spike."""
        try:
            end_time = time.time() + duration_seconds
            
            while time.time() < end_time:
                # CPU intensive work
                work_time = 0.1 * cpu_load
                sleep_time = 0.1 * (1 - cpu_load)
                
                # Busy work
                start_work = time.time()
                while time.time() - start_work < work_time:
                    # Prime number calculation
                    n = random.randint(1000, 100000)
                    for i in range(2, int(n**0.5) + 1):
                        if n % i == 0:
                            break
                
                # Sleep to control load
                await asyncio.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f"CPU spike simulation failed: {e}")
    
    def _calculate_disk_usage(self, escalation_level: FailureEscalationLevel) -> float:
        """Calculate target disk usage percentage."""
        usage_map = {
            FailureEscalationLevel.MINIMAL: 0.8,    # 80%
            FailureEscalationLevel.LOW: 0.85,       # 85%
            FailureEscalationLevel.MEDIUM: 0.9,     # 90%
            FailureEscalationLevel.HIGH: 0.95,      # 95%
            FailureEscalationLevel.CRITICAL: 0.98   # 98%
        }
        return usage_map.get(escalation_level, 0.9)
    
    async def _fill_disk_space(self, target_usage: float) -> Dict[str, Any]:
        """Fill disk space to target usage."""
        try:
            # Create temporary directory
            temp_dir = Path("/tmp/chaos_disk_fill")
            temp_dir.mkdir(exist_ok=True)
            
            # Get current disk usage
            disk_usage = psutil.disk_usage("/")
            current_usage = disk_usage.used / disk_usage.total
            
            if current_usage >= target_usage:
                return {"success": True, "already_at_target": True}
            
            # Calculate space to fill
            space_to_fill = int((target_usage - current_usage) * disk_usage.total)
            
            # Create files
            created_files = []
            bytes_written = 0
            file_size = 100 * 1024 * 1024  # 100MB per file
            
            while bytes_written < space_to_fill:
                file_path = temp_dir / f"fill_file_{len(created_files)}.tmp"
                
                with open(file_path, "wb") as f:
                    remaining = min(file_size, space_to_fill - bytes_written)
                    f.write(b"0" * remaining)
                    bytes_written += remaining
                
                created_files.append(str(file_path))
                
                # Check if we've reached the target
                current_disk_usage = psutil.disk_usage("/")
                current_usage = current_disk_usage.used / current_disk_usage.total
                
                if current_usage >= target_usage:
                    break
            
            return {
                "success": True,
                "created_files": created_files,
                "bytes_written": bytes_written,
                "target_usage": target_usage,
                "actual_usage": current_usage
            }
            
        except Exception as e:
            logger.error(f"Disk fill failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def cleanup_injection(self, injection_id: str) -> bool:
        """Clean up resource injection."""
        try:
            with self.injection_lock:
                if injection_id not in self.active_injections:
                    return False
                
                injection_info = self.active_injections[injection_id]
                injection_type = injection_info.get("type")
                
                if injection_type == "memory_leak":
                    # Cancel leak task
                    leak_task = injection_info.get("leak_task")
                    if leak_task and not leak_task.cancelled():
                        leak_task.cancel()
                
                elif injection_type == "cpu_spike":
                    # Cancel CPU tasks
                    cpu_tasks = injection_info.get("cpu_tasks", [])
                    for task in cpu_tasks:
                        if not task.cancelled():
                            task.cancel()
                
                elif injection_type == "disk_full":
                    # Remove created files
                    created_files = injection_info.get("created_files", [])
                    for file_path in created_files:
                        try:
                            os.remove(file_path)
                        except FileNotFoundError:
                            pass
                
                # Remove from active injections
                del self.active_injections[injection_id]
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup resource injection: {e}")
            return False


class FailureInjectionEngine:
    """Main engine for coordinating failure injections."""
    
    def __init__(self):
        self.network_injector = NetworkFailureInjector()
        self.service_injector = ServiceFailureInjector()
        self.resource_injector = ResourceFailureInjector()
        
        self.active_executions = {}
        self.execution_lock = threading.Lock()
        
        logger.info("Failure Injection Engine initialized")
    
    async def execute_failure_spec(self, spec: FailureSpec) -> FailureExecution:
        """Execute a failure specification."""
        injection_id = f"failure_{spec.failure_type.value}_{int(time.time())}"
        
        execution = FailureExecution(
            spec=spec,
            injection_id=injection_id,
            start_time=datetime.now()
        )
        
        with self.execution_lock:
            self.active_executions[injection_id] = execution
        
        try:
            # Route to appropriate injector
            if spec.failure_type in [FailureType.NETWORK_PARTITION, FailureType.NETWORK_LATENCY, 
                                   FailureType.NETWORK_PACKET_LOSS, FailureType.NETWORK_BANDWIDTH_LIMIT]:
                result = await self.network_injector.inject_network_partition(spec, injection_id)
                
            elif spec.failure_type in [FailureType.SERVICE_CRASH, FailureType.SERVICE_HANG, 
                                     FailureType.SERVICE_SLOW_RESPONSE, FailureType.SERVICE_ERROR_INJECTION]:
                if spec.failure_type == FailureType.SERVICE_CRASH:
                    result = await self.service_injector.inject_service_crash(spec, injection_id)
                elif spec.failure_type == FailureType.SERVICE_HANG:
                    result = await self.service_injector.inject_service_hang(spec, injection_id)
                elif spec.failure_type == FailureType.SERVICE_SLOW_RESPONSE:
                    result = await self.service_injector.inject_slow_response(spec, injection_id)
                else:
                    result = {"success": False, "error": "Unsupported service failure type"}
                    
            elif spec.failure_type in [FailureType.MEMORY_LEAK, FailureType.CPU_SPIKE, 
                                     FailureType.DISK_FULL, FailureType.FILE_DESCRIPTOR_EXHAUSTION]:
                if spec.failure_type == FailureType.MEMORY_LEAK:
                    result = await self.resource_injector.inject_memory_leak(spec, injection_id)
                elif spec.failure_type == FailureType.CPU_SPIKE:
                    result = await self.resource_injector.inject_cpu_spike(spec, injection_id)
                elif spec.failure_type == FailureType.DISK_FULL:
                    result = await self.resource_injector.inject_disk_full(spec, injection_id)
                else:
                    result = {"success": False, "error": "Unsupported resource failure type"}
                    
            else:
                result = {"success": False, "error": f"Unsupported failure type: {spec.failure_type}"}
            
            # Update execution status
            if result.get("success"):
                execution.status = "ACTIVE"
                execution.metrics = result
                
                # Schedule cleanup
                asyncio.create_task(self._schedule_cleanup(execution))
                
            else:
                execution.status = "FAILED"
                execution.error_message = result.get("error")
                execution.end_time = datetime.now()
            
            return execution
            
        except Exception as e:
            execution.status = "FAILED"
            execution.error_message = str(e)
            execution.end_time = datetime.now()
            
            logger.error(f"Failure execution failed: {e}")
            return execution
    
    async def _schedule_cleanup(self, execution: FailureExecution):
        """Schedule cleanup after failure duration."""
        try:
            # Wait for failure duration
            await asyncio.sleep(execution.spec.duration_seconds)
            
            # Execute cleanup
            await self.cleanup_failure(execution.injection_id)
            
        except Exception as e:
            logger.error(f"Scheduled cleanup failed: {e}")
    
    async def cleanup_failure(self, injection_id: str) -> bool:
        """Clean up a specific failure injection."""
        try:
            with self.execution_lock:
                if injection_id not in self.active_executions:
                    return False
                
                execution = self.active_executions[injection_id]
                
                # Route cleanup to appropriate injector
                cleanup_success = False
                
                if execution.spec.failure_type in [FailureType.NETWORK_PARTITION, FailureType.NETWORK_LATENCY, 
                                                 FailureType.NETWORK_PACKET_LOSS, FailureType.NETWORK_BANDWIDTH_LIMIT]:
                    cleanup_success = await self.network_injector.cleanup_injection(injection_id)
                    
                elif execution.spec.failure_type in [FailureType.SERVICE_CRASH, FailureType.SERVICE_HANG, 
                                                   FailureType.SERVICE_SLOW_RESPONSE, FailureType.SERVICE_ERROR_INJECTION]:
                    cleanup_success = await self.service_injector.cleanup_injection(injection_id)
                    
                elif execution.spec.failure_type in [FailureType.MEMORY_LEAK, FailureType.CPU_SPIKE, 
                                                   FailureType.DISK_FULL, FailureType.FILE_DESCRIPTOR_EXHAUSTION]:
                    cleanup_success = await self.resource_injector.cleanup_injection(injection_id)
                
                # Update execution status
                execution.status = "CLEANED_UP" if cleanup_success else "CLEANUP_FAILED"
                execution.end_time = datetime.now()
                
                # Remove from active executions
                del self.active_executions[injection_id]
                
                return cleanup_success
                
        except Exception as e:
            logger.error(f"Failure cleanup failed: {e}")
            return False
    
    async def cleanup_all_failures(self) -> int:
        """Clean up all active failure injections."""
        cleanup_count = 0
        
        with self.execution_lock:
            active_ids = list(self.active_executions.keys())
        
        for injection_id in active_ids:
            if await self.cleanup_failure(injection_id):
                cleanup_count += 1
        
        return cleanup_count
    
    def get_active_executions(self) -> List[FailureExecution]:
        """Get all active failure executions."""
        with self.execution_lock:
            return list(self.active_executions.values())
    
    def get_execution_status(self, injection_id: str) -> Optional[FailureExecution]:
        """Get status of a specific failure execution."""
        with self.execution_lock:
            return self.active_executions.get(injection_id)
    
    async def get_engine_status(self) -> Dict[str, Any]:
        """Get engine status."""
        with self.execution_lock:
            active_count = len(self.active_executions)
            active_types = [exec.spec.failure_type.value for exec in self.active_executions.values()]
        
        return {
            "engine_version": "1.0.0",
            "active_executions": active_count,
            "active_failure_types": active_types,
            "supported_failure_types": [ft.value for ft in FailureType],
            "supported_escalation_levels": [el.value for el in FailureEscalationLevel],
            "supported_patterns": [fp.value for fp in FailurePattern]
        }


# Example usage
async def main():
    """Demonstrate the Failure Injection Engine."""
    engine = FailureInjectionEngine()
    
    # Create failure specifications
    specs = [
        FailureSpec(
            failure_type=FailureType.NETWORK_PARTITION,
            target_component="redis",
            escalation_level=FailureEscalationLevel.MEDIUM,
            pattern=FailurePattern.SUDDEN_FAILURE,
            duration_seconds=30,
            blast_radius=0.5,
            recovery_time_seconds=10
        ),
        FailureSpec(
            failure_type=FailureType.SERVICE_CRASH,
            target_component="tactical",
            escalation_level=FailureEscalationLevel.HIGH,
            pattern=FailurePattern.CASCADING_FAILURE,
            duration_seconds=45,
            blast_radius=0.8,
            recovery_time_seconds=20
        ),
        FailureSpec(
            failure_type=FailureType.MEMORY_LEAK,
            target_component="system",
            escalation_level=FailureEscalationLevel.LOW,
            pattern=FailurePattern.GRADUAL_DEGRADATION,
            duration_seconds=60,
            blast_radius=0.3,
            recovery_time_seconds=15
        )
    ]
    
    # Execute failures
    executions = []
    for spec in specs:
        execution = await engine.execute_failure_spec(spec)
        executions.append(execution)
        print(f"Executed {spec.failure_type.value}: {execution.status}")
    
    # Monitor executions
    await asyncio.sleep(10)
    
    # Get status
    status = await engine.get_engine_status()
    print(f"Engine Status: {json.dumps(status, indent=2)}")
    
    # Cleanup
    cleanup_count = await engine.cleanup_all_failures()
    print(f"Cleaned up {cleanup_count} failures")


if __name__ == "__main__":
    asyncio.run(main())