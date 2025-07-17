#!/usr/bin/env python3
"""
Automated Recovery Orchestrator
==============================

Central orchestration system for automated recovery actions.
Coordinates different recovery strategies and executes recovery scripts
based on failure types and system state.

Features:
- Intelligent recovery strategy selection
- Parallel and sequential recovery execution
- Recovery success validation
- Rollback mechanisms
- Recovery history and analytics
"""

import asyncio
import logging
import time
import json
import subprocess
import yaml
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
import threading

# Internal imports
from src.monitoring.health_monitor_v2 import HealthStatus, PredictionType, HealthAssessment
from src.core.event_bus import EventBus
from src.core.events import Event

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    RESTART_SERVICE = "restart_service"
    SCALE_HORIZONTALLY = "scale_horizontally"
    INCREASE_RESOURCES = "increase_resources"
    ENABLE_CIRCUIT_BREAKER = "enable_circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAILOVER_TO_BACKUP = "failover_to_backup"
    CLEAR_CACHE = "clear_cache"
    CLEANUP_RESOURCES = "cleanup_resources"
    ROLLBACK_DEPLOYMENT = "rollback_deployment"
    CUSTOM_SCRIPT = "custom_script"


class RecoveryStatus(Enum):
    """Recovery execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLBACK_REQUIRED = "rollback_required"


@dataclass
class RecoveryAction:
    """Definition of a recovery action."""
    strategy: RecoveryStrategy
    service_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    rollback_action: Optional['RecoveryAction'] = None
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class RecoveryExecution:
    """Tracking of recovery execution."""
    action_id: str
    action: RecoveryAction
    status: RecoveryStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    output: Optional[str] = None
    error: Optional[str] = None
    success_metrics: Dict[str, Any] = field(default_factory=dict)


class ServiceRestartRecovery:
    """Handles service restart recovery actions."""
    
    def __init__(self):
        self.restart_history = {}
    
    async def execute(self, service_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute service restart."""
        restart_type = parameters.get('restart_type', 'graceful')
        
        try:
            if restart_type == 'graceful':
                result = await self._graceful_restart(service_name, parameters)
            elif restart_type == 'force':
                result = await self._force_restart(service_name, parameters)
            else:
                result = await self._rolling_restart(service_name, parameters)
            
            # Track restart
            self.restart_history[service_name] = {
                'timestamp': datetime.utcnow(),
                'type': restart_type,
                'success': result['success']
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Service restart failed for {service_name}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _graceful_restart(self, service_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform graceful service restart."""
        # Send SIGTERM and wait for graceful shutdown
        cmd = f"kubectl rollout restart deployment/{service_name}"
        
        try:
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Wait for rollout to complete
                await self._wait_for_rollout(service_name)
                return {
                    'success': True,
                    'output': stdout.decode(),
                    'restart_type': 'graceful'
                }
            else:
                return {
                    'success': False,
                    'error': stderr.decode(),
                    'restart_type': 'graceful'
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _force_restart(self, service_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform force restart."""
        # Delete pods to force restart
        cmd = f"kubectl delete pods -l app={service_name}"
        
        try:
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return {
                    'success': True,
                    'output': stdout.decode(),
                    'restart_type': 'force'
                }
            else:
                return {
                    'success': False,
                    'error': stderr.decode(),
                    'restart_type': 'force'
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _rolling_restart(self, service_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform rolling restart."""
        # Patch deployment to trigger rolling update
        cmd = f"kubectl patch deployment {service_name} -p " + \
              "'{\"spec\":{\"template\":{\"metadata\":{\"annotations\":{\"date\":\"`date +'%s'`\"}}}}}'"
        
        try:
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                await self._wait_for_rollout(service_name)
                return {
                    'success': True,
                    'output': stdout.decode(),
                    'restart_type': 'rolling'
                }
            else:
                return {
                    'success': False,
                    'error': stderr.decode(),
                    'restart_type': 'rolling'
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _wait_for_rollout(self, service_name: str, timeout: int = 300):
        """Wait for rollout to complete."""
        cmd = f"kubectl rollout status deployment/{service_name} --timeout={timeout}s"
        
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        await process.communicate()


class HorizontalScalingRecovery:
    """Handles horizontal scaling recovery actions."""
    
    async def execute(self, service_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute horizontal scaling."""
        current_replicas = parameters.get('current_replicas', 1)
        target_replicas = parameters.get('target_replicas', current_replicas + 1)
        max_replicas = parameters.get('max_replicas', 10)
        
        # Ensure we don't exceed maximum
        target_replicas = min(target_replicas, max_replicas)
        
        try:
            cmd = f"kubectl scale deployment {service_name} --replicas={target_replicas}"
            
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Wait for scaling to complete
                await self._wait_for_scaling(service_name, target_replicas)
                return {
                    'success': True,
                    'output': stdout.decode(),
                    'old_replicas': current_replicas,
                    'new_replicas': target_replicas
                }
            else:
                return {
                    'success': False,
                    'error': stderr.decode()
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _wait_for_scaling(self, service_name: str, expected_replicas: int, timeout: int = 300):
        """Wait for scaling to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            cmd = f"kubectl get deployment {service_name} -o jsonpath='{{.status.readyReplicas}}'"
            
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                try:
                    ready_replicas = int(stdout.decode().strip())
                    if ready_replicas >= expected_replicas:
                        return
                except ValueError:
                    pass
            
            await asyncio.sleep(5)
        
        logger.warning(f"Scaling timeout for {service_name}")


class ResourceIncreaseRecovery:
    """Handles resource increase recovery actions."""
    
    async def execute(self, service_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute resource increase."""
        resource_type = parameters.get('resource_type', 'memory')
        increase_percentage = parameters.get('increase_percentage', 50)
        
        try:
            if resource_type == 'memory':
                return await self._increase_memory(service_name, increase_percentage)
            elif resource_type == 'cpu':
                return await self._increase_cpu(service_name, increase_percentage)
            else:
                return {'success': False, 'error': f'Unknown resource type: {resource_type}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _increase_memory(self, service_name: str, increase_percentage: int) -> Dict[str, Any]:
        """Increase memory resources."""
        # Get current memory limit
        cmd = f"kubectl get deployment {service_name} -o jsonpath='{{.spec.template.spec.containers[0].resources.limits.memory}}'"
        
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            return {'success': False, 'error': 'Failed to get current memory limit'}
        
        current_memory = stdout.decode().strip()
        
        # Parse memory value (e.g., "512Mi" -> 512)
        if current_memory.endswith('Mi'):
            memory_mb = int(current_memory[:-2])
        elif current_memory.endswith('Gi'):
            memory_mb = int(float(current_memory[:-2]) * 1024)
        else:
            return {'success': False, 'error': f'Unknown memory format: {current_memory}'}
        
        # Calculate new memory
        new_memory_mb = int(memory_mb * (1 + increase_percentage / 100))
        new_memory = f"{new_memory_mb}Mi"
        
        # Update deployment
        patch = {
            'spec': {
                'template': {
                    'spec': {
                        'containers': [{
                            'name': service_name,
                            'resources': {
                                'limits': {
                                    'memory': new_memory
                                },
                                'requests': {
                                    'memory': f"{int(new_memory_mb * 0.8)}Mi"
                                }
                            }
                        }]
                    }
                }
            }
        }
        
        patch_json = json.dumps(patch)
        cmd = f"kubectl patch deployment {service_name} -p '{patch_json}'"
        
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            return {
                'success': True,
                'output': stdout.decode(),
                'old_memory': current_memory,
                'new_memory': new_memory
            }
        else:
            return {
                'success': False,
                'error': stderr.decode()
            }
    
    async def _increase_cpu(self, service_name: str, increase_percentage: int) -> Dict[str, Any]:
        """Increase CPU resources."""
        # Similar implementation for CPU
        # Get current CPU limit
        cmd = f"kubectl get deployment {service_name} -o jsonpath='{{.spec.template.spec.containers[0].resources.limits.cpu}}'"
        
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            return {'success': False, 'error': 'Failed to get current CPU limit'}
        
        current_cpu = stdout.decode().strip()
        
        # Parse CPU value (e.g., "500m" -> 0.5)
        if current_cpu.endswith('m'):
            cpu_millicores = int(current_cpu[:-1])
        else:
            cpu_millicores = int(float(current_cpu) * 1000)
        
        # Calculate new CPU
        new_cpu_millicores = int(cpu_millicores * (1 + increase_percentage / 100))
        new_cpu = f"{new_cpu_millicores}m"
        
        # Update deployment
        patch = {
            'spec': {
                'template': {
                    'spec': {
                        'containers': [{
                            'name': service_name,
                            'resources': {
                                'limits': {
                                    'cpu': new_cpu
                                },
                                'requests': {
                                    'cpu': f"{int(new_cpu_millicores * 0.8)}m"
                                }
                            }
                        }]
                    }
                }
            }
        }
        
        patch_json = json.dumps(patch)
        cmd = f"kubectl patch deployment {service_name} -p '{patch_json}'"
        
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            return {
                'success': True,
                'output': stdout.decode(),
                'old_cpu': current_cpu,
                'new_cpu': new_cpu
            }
        else:
            return {
                'success': False,
                'error': stderr.decode()
            }


class CacheCleanupRecovery:
    """Handles cache cleanup recovery actions."""
    
    async def execute(self, service_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cache cleanup."""
        cache_type = parameters.get('cache_type', 'redis')
        cleanup_strategy = parameters.get('cleanup_strategy', 'selective')
        
        try:
            if cache_type == 'redis':
                return await self._cleanup_redis_cache(service_name, cleanup_strategy, parameters)
            elif cache_type == 'memory':
                return await self._cleanup_memory_cache(service_name, cleanup_strategy, parameters)
            else:
                return {'success': False, 'error': f'Unknown cache type: {cache_type}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _cleanup_redis_cache(self, service_name: str, strategy: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Cleanup Redis cache."""
        redis_host = parameters.get('redis_host', 'redis')
        redis_port = parameters.get('redis_port', 6379)
        
        if strategy == 'selective':
            # Remove keys matching service pattern
            pattern = parameters.get('key_pattern', f"{service_name}:*")
            cmd = f"redis-cli -h {redis_host} -p {redis_port} --scan --pattern '{pattern}' | xargs redis-cli -h {redis_host} -p {redis_port} del"
        elif strategy == 'expired':
            # Remove expired keys
            cmd = f"redis-cli -h {redis_host} -p {redis_port} eval \"return redis.call('del', unpack(redis.call('keys', 'expired:*')))\" 0"
        else:
            # Full flush (dangerous!)
            cmd = f"redis-cli -h {redis_host} -p {redis_port} flushdb"
        
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            return {
                'success': True,
                'output': stdout.decode(),
                'cache_type': 'redis',
                'strategy': strategy
            }
        else:
            return {
                'success': False,
                'error': stderr.decode()
            }
    
    async def _cleanup_memory_cache(self, service_name: str, strategy: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Cleanup memory cache by restarting service."""
        # Memory cache cleanup typically requires service restart
        restart_recovery = ServiceRestartRecovery()
        result = await restart_recovery.execute(service_name, {'restart_type': 'graceful'})
        
        if result['success']:
            result['cache_type'] = 'memory'
            result['strategy'] = strategy
        
        return result


class RecoveryOrchestrator:
    """Central orchestrator for automated recovery actions."""
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        self.event_bus = event_bus
        self.recovery_handlers = {}
        self.active_recoveries = {}
        self.recovery_history = []
        self.config = {}
        self._lock = threading.Lock()
        
        # Initialize recovery handlers
        self._initialize_recovery_handlers()
        
        # Load configuration
        self._load_configuration()
        
        logger.info("Recovery orchestrator initialized")
    
    def _initialize_recovery_handlers(self):
        """Initialize recovery strategy handlers."""
        self.recovery_handlers = {
            RecoveryStrategy.RESTART_SERVICE: ServiceRestartRecovery(),
            RecoveryStrategy.SCALE_HORIZONTALLY: HorizontalScalingRecovery(),
            RecoveryStrategy.INCREASE_RESOURCES: ResourceIncreaseRecovery(),
            RecoveryStrategy.CLEAR_CACHE: CacheCleanupRecovery(),
        }
    
    def _load_configuration(self):
        """Load recovery configuration."""
        config_path = Path(__file__).parent / "recovery_config.yaml"
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Failed to load recovery config: {e}")
                self.config = {}
        
        # Set defaults
        if not self.config:
            self.config = {
                'default_timeout': 300,
                'max_concurrent_recoveries': 5,
                'recovery_strategies': {
                    'high_memory_usage': [
                        {'strategy': 'clear_cache', 'priority': 1},
                        {'strategy': 'increase_resources', 'priority': 2},
                        {'strategy': 'restart_service', 'priority': 3}
                    ],
                    'high_cpu_usage': [
                        {'strategy': 'scale_horizontally', 'priority': 1},
                        {'strategy': 'increase_resources', 'priority': 2}
                    ],
                    'service_failure': [
                        {'strategy': 'restart_service', 'priority': 1},
                        {'strategy': 'scale_horizontally', 'priority': 2}
                    ]
                }
            }
    
    async def trigger_recovery(
        self,
        service_name: str,
        failure_type: str,
        health_assessment: Optional[HealthAssessment] = None,
        custom_actions: Optional[List[RecoveryAction]] = None
    ) -> str:
        """Trigger recovery process for a service."""
        recovery_id = f"recovery_{service_name}_{int(time.time())}"
        
        with self._lock:
            if len(self.active_recoveries) >= self.config.get('max_concurrent_recoveries', 5):
                logger.warning(f"Maximum concurrent recoveries reached, queuing {recovery_id}")
                return recovery_id
            
            # Determine recovery actions
            if custom_actions:
                actions = custom_actions
            else:
                actions = self._determine_recovery_actions(service_name, failure_type, health_assessment)
            
            if not actions:
                logger.warning(f"No recovery actions determined for {service_name} with failure type {failure_type}")
                return recovery_id
            
            # Create recovery execution
            self.active_recoveries[recovery_id] = {
                'service_name': service_name,
                'failure_type': failure_type,
                'actions': actions,
                'start_time': datetime.utcnow(),
                'status': 'pending'
            }
        
        # Start recovery process
        asyncio.create_task(self._execute_recovery(recovery_id))
        
        logger.info(f"Recovery triggered for {service_name}: {recovery_id}")
        return recovery_id
    
    def _determine_recovery_actions(
        self,
        service_name: str,
        failure_type: str,
        health_assessment: Optional[HealthAssessment]
    ) -> List[RecoveryAction]:
        """Determine appropriate recovery actions based on failure type and assessment."""
        actions = []
        
        # Get strategies from configuration
        strategies = self.config.get('recovery_strategies', {}).get(failure_type, [])
        
        for strategy_config in strategies:
            strategy = RecoveryStrategy(strategy_config['strategy'])
            parameters = strategy_config.get('parameters', {})
            
            # Customize parameters based on health assessment
            if health_assessment:
                parameters = self._customize_parameters(strategy, parameters, health_assessment)
            
            action = RecoveryAction(
                strategy=strategy,
                service_name=service_name,
                parameters=parameters,
                timeout_seconds=strategy_config.get('timeout', self.config.get('default_timeout', 300))
            )
            
            actions.append(action)
        
        # Sort by priority
        actions.sort(key=lambda a: next(
            (s.get('priority', 999) for s in strategies if s['strategy'] == a.strategy.value),
            999
        ))
        
        return actions
    
    def _customize_parameters(
        self,
        strategy: RecoveryStrategy,
        base_parameters: Dict[str, Any],
        health_assessment: HealthAssessment
    ) -> Dict[str, Any]:
        """Customize recovery parameters based on health assessment."""
        parameters = base_parameters.copy()
        metrics = health_assessment.current_metrics
        
        if strategy == RecoveryStrategy.SCALE_HORIZONTALLY:
            # Determine scaling based on current load
            if metrics.cpu_percent > 90:
                parameters['target_replicas'] = parameters.get('target_replicas', 1) + 2
            elif metrics.cpu_percent > 80:
                parameters['target_replicas'] = parameters.get('target_replicas', 1) + 1
        
        elif strategy == RecoveryStrategy.INCREASE_RESOURCES:
            # Determine resource increase based on usage
            if metrics.memory_percent > 90:
                parameters['resource_type'] = 'memory'
                parameters['increase_percentage'] = 100  # Double memory
            elif metrics.cpu_percent > 90:
                parameters['resource_type'] = 'cpu'
                parameters['increase_percentage'] = 50   # 50% more CPU
        
        elif strategy == RecoveryStrategy.CLEAR_CACHE:
            # Determine cache cleanup strategy
            if metrics.memory_percent > 95:
                parameters['cleanup_strategy'] = 'full'
            else:
                parameters['cleanup_strategy'] = 'selective'
        
        return parameters
    
    async def _execute_recovery(self, recovery_id: str):
        """Execute recovery actions for a recovery session."""
        recovery_info = self.active_recoveries.get(recovery_id)
        if not recovery_info:
            return
        
        service_name = recovery_info['service_name']
        actions = recovery_info['actions']
        
        recovery_info['status'] = 'running'
        recovery_info['executions'] = []
        
        try:
            # Send recovery started event
            if self.event_bus:
                await self.event_bus.publish(Event(
                    type="recovery_started",
                    data={
                        'recovery_id': recovery_id,
                        'service_name': service_name,
                        'failure_type': recovery_info['failure_type'],
                        'actions_count': len(actions)
                    }
                ))
            
            # Execute actions sequentially
            for i, action in enumerate(actions):
                action_id = f"{recovery_id}_action_{i}"
                
                execution = RecoveryExecution(
                    action_id=action_id,
                    action=action,
                    status=RecoveryStatus.RUNNING,
                    start_time=datetime.utcnow()
                )
                
                recovery_info['executions'].append(execution)
                
                logger.info(f"Executing recovery action {i+1}/{len(actions)} for {service_name}: {action.strategy.value}")
                
                # Execute the action
                handler = self.recovery_handlers.get(action.strategy)
                if handler:
                    try:
                        result = await asyncio.wait_for(
                            handler.execute(action.service_name, action.parameters),
                            timeout=action.timeout_seconds
                        )
                        
                        execution.end_time = datetime.utcnow()
                        
                        if result.get('success', False):
                            execution.status = RecoveryStatus.SUCCESS
                            execution.output = result.get('output', '')
                            execution.success_metrics = result
                            
                            logger.info(f"Recovery action {action.strategy.value} succeeded for {service_name}")
                            
                            # If this action succeeded, we might be done
                            if await self._validate_recovery_success(service_name, action):
                                recovery_info['status'] = 'success'
                                break
                        else:
                            execution.status = RecoveryStatus.FAILED
                            execution.error = result.get('error', 'Unknown error')
                            
                            logger.error(f"Recovery action {action.strategy.value} failed for {service_name}: {execution.error}")
                            
                            # Try next action
                            continue
                    
                    except asyncio.TimeoutError:
                        execution.status = RecoveryStatus.FAILED
                        execution.error = f"Action timed out after {action.timeout_seconds} seconds"
                        execution.end_time = datetime.utcnow()
                        
                        logger.error(f"Recovery action {action.strategy.value} timed out for {service_name}")
                        continue
                    
                    except Exception as e:
                        execution.status = RecoveryStatus.FAILED
                        execution.error = str(e)
                        execution.end_time = datetime.utcnow()
                        
                        logger.error(f"Recovery action {action.strategy.value} failed with exception for {service_name}: {e}")
                        continue
                else:
                    execution.status = RecoveryStatus.FAILED
                    execution.error = f"No handler for strategy {action.strategy.value}"
                    execution.end_time = datetime.utcnow()
                    
                    logger.error(f"No handler found for recovery strategy {action.strategy.value}")
            
            # Check final status
            if recovery_info['status'] != 'success':
                recovery_info['status'] = 'failed'
            
            # Send completion event
            if self.event_bus:
                await self.event_bus.publish(Event(
                    type="recovery_completed",
                    data={
                        'recovery_id': recovery_id,
                        'service_name': service_name,
                        'status': recovery_info['status'],
                        'duration_seconds': (datetime.utcnow() - recovery_info['start_time']).total_seconds(),
                        'actions_executed': len(recovery_info['executions']),
                        'actions_successful': len([e for e in recovery_info['executions'] if e.status == RecoveryStatus.SUCCESS])
                    }
                ))
        
        finally:
            # Move to history and clean up
            recovery_info['end_time'] = datetime.utcnow()
            self.recovery_history.append(recovery_info)
            
            with self._lock:
                if recovery_id in self.active_recoveries:
                    del self.active_recoveries[recovery_id]
            
            logger.info(f"Recovery {recovery_id} completed with status: {recovery_info['status']}")
    
    async def _validate_recovery_success(self, service_name: str, action: RecoveryAction) -> bool:
        """Validate that recovery action was successful."""
        # Wait a bit for changes to take effect
        await asyncio.sleep(10)
        
        # Basic validation - check if service is responding
        try:
            cmd = f"kubectl get pods -l app={service_name} --field-selector=status.phase=Running"
            
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                running_pods = len(stdout.decode().strip().split('\n')) - 1  # Subtract header
                return running_pods > 0
            
        except Exception as e:
            logger.error(f"Failed to validate recovery for {service_name}: {e}")
        
        return False
    
    def get_recovery_status(self, recovery_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a recovery session."""
        if recovery_id in self.active_recoveries:
            return self.active_recoveries[recovery_id]
        
        # Check history
        for recovery in self.recovery_history:
            if recovery_id in str(recovery):
                return recovery
        
        return None
    
    def get_active_recoveries(self) -> Dict[str, Any]:
        """Get all active recovery sessions."""
        with self._lock:
            return self.active_recoveries.copy()
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        total_recoveries = len(self.recovery_history)
        successful_recoveries = len([r for r in self.recovery_history if r.get('status') == 'success'])
        
        # Strategy success rates
        strategy_stats = {}
        for recovery in self.recovery_history:
            for execution in recovery.get('executions', []):
                strategy = execution.action.strategy.value
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {'total': 0, 'successful': 0}
                
                strategy_stats[strategy]['total'] += 1
                if execution.status == RecoveryStatus.SUCCESS:
                    strategy_stats[strategy]['successful'] += 1
        
        # Calculate success rates
        for stats in strategy_stats.values():
            stats['success_rate'] = stats['successful'] / stats['total'] if stats['total'] > 0 else 0
        
        return {
            'total_recoveries': total_recoveries,
            'successful_recoveries': successful_recoveries,
            'success_rate': successful_recoveries / total_recoveries if total_recoveries > 0 else 0,
            'active_recoveries': len(self.active_recoveries),
            'strategy_statistics': strategy_stats,
            'last_24h_recoveries': len([
                r for r in self.recovery_history 
                if datetime.utcnow() - r['start_time'] < timedelta(hours=24)
            ])
        }


# Global recovery orchestrator instance
recovery_orchestrator = None


def get_recovery_orchestrator() -> RecoveryOrchestrator:
    """Get global recovery orchestrator instance."""
    global recovery_orchestrator
    
    if recovery_orchestrator is None:
        recovery_orchestrator = RecoveryOrchestrator()
    
    return recovery_orchestrator


if __name__ == "__main__":
    # Example usage
    async def main():
        orchestrator = RecoveryOrchestrator()
        
        # Trigger a recovery
        recovery_id = await orchestrator.trigger_recovery(
            service_name="strategic-agent",
            failure_type="high_memory_usage"
        )
        
        # Monitor recovery
        while True:
            status = orchestrator.get_recovery_status(recovery_id)
            if status and status.get('status') not in ['pending', 'running']:
                break
            await asyncio.sleep(5)
        
        print(f"Recovery completed: {status}")
        print(f"Statistics: {orchestrator.get_recovery_statistics()}")
    
    asyncio.run(main())