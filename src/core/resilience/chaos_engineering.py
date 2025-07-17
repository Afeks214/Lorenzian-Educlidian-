"""
Chaos Engineering Framework
===========================

Comprehensive chaos engineering framework for testing system resilience
through controlled failure injection and fault simulation.

Features:
- Multiple failure injection types
- Controlled chaos experiments
- Automated resilience testing
- Failure scenario simulation
- Recovery validation
- Metrics collection and analysis
"""

import asyncio
import random
import time
import logging
from typing import Dict, List, Optional, Callable, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures that can be injected."""
    NETWORK_DELAY = "network_delay"
    NETWORK_FAILURE = "network_failure"
    SERVICE_UNAVAILABLE = "service_unavailable"
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    MEMORY_LEAK = "memory_leak"
    CPU_SPIKE = "cpu_spike"
    DISK_FAILURE = "disk_failure"
    PARTIAL_FAILURE = "partial_failure"
    CASCADING_FAILURE = "cascading_failure"
    DEPENDENCY_FAILURE = "dependency_failure"


class ExperimentPhase(Enum):
    """Phases of chaos experiment."""
    SETUP = "setup"
    BASELINE = "baseline"
    INJECTION = "injection"
    RECOVERY = "recovery"
    VALIDATION = "validation"
    CLEANUP = "cleanup"


@dataclass
class ChaosConfig:
    """Configuration for chaos engineering experiments."""
    # Experiment settings
    experiment_duration: int = 300  # 5 minutes
    baseline_duration: int = 60     # 1 minute
    recovery_duration: int = 60     # 1 minute
    
    # Failure injection settings
    failure_probability: float = 0.1
    failure_intensity: float = 0.5
    failure_duration: int = 30
    
    # Safety settings
    enable_safety_checks: bool = True
    max_concurrent_experiments: int = 1
    emergency_stop_threshold: float = 0.8
    
    # Targeting settings
    target_services: List[str] = field(default_factory=list)
    exclude_services: List[str] = field(default_factory=list)
    target_percentage: float = 0.1
    
    # Monitoring settings
    metrics_collection: bool = True
    detailed_logging: bool = True
    
    # Validation settings
    validate_recovery: bool = True
    recovery_threshold: float = 0.9
    
    # Scheduling settings
    enable_scheduling: bool = False
    schedule_interval: int = 3600  # 1 hour
    business_hours_only: bool = True


@dataclass
class FailureInjection:
    """Configuration for a specific failure injection."""
    failure_type: FailureType
    target_service: str
    intensity: float
    duration: int
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Conditions
    condition: Optional[Callable] = None
    trigger_probability: float = 1.0
    
    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Metadata
    experiment_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ExperimentMetrics:
    """Metrics collected during chaos experiments."""
    experiment_id: str
    phase: ExperimentPhase
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # System metrics
    system_health_score: float = 0.0
    error_rate: float = 0.0
    response_time: float = 0.0
    throughput: float = 0.0
    
    # Resilience metrics
    recovery_time: float = 0.0
    failure_detection_time: float = 0.0
    circuit_breaker_activations: int = 0
    retry_attempts: int = 0
    
    # Service metrics
    service_availability: Dict[str, float] = field(default_factory=dict)
    service_performance: Dict[str, float] = field(default_factory=dict)
    
    # Experiment results
    experiment_success: bool = False
    recovery_success: bool = False
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class ChaosEngineer:
    """
    Chaos engineering framework for resilience testing.
    
    Provides:
    - Controlled failure injection
    - Automated experiment execution
    - Metrics collection and analysis
    - Recovery validation
    - Safety mechanisms
    """
    
    def __init__(
        self,
        config: ChaosConfig,
        health_monitor: Optional[Any] = None,
        circuit_breaker_manager: Optional[Any] = None,
        bulkhead_manager: Optional[Any] = None
    ):
        """Initialize chaos engineer."""
        self.config = config
        self.health_monitor = health_monitor
        self.circuit_breaker_manager = circuit_breaker_manager
        self.bulkhead_manager = bulkhead_manager
        
        # Experiment tracking
        self.active_experiments: Dict[str, Dict[str, Any]] = {}
        self.experiment_history: List[ExperimentMetrics] = []
        
        # Failure injection registry
        self.failure_injectors: Dict[FailureType, Callable] = {
            FailureType.NETWORK_DELAY: self._inject_network_delay,
            FailureType.NETWORK_FAILURE: self._inject_network_failure,
            FailureType.SERVICE_UNAVAILABLE: self._inject_service_unavailable,
            FailureType.TIMEOUT: self._inject_timeout,
            FailureType.EXCEPTION: self._inject_exception,
            FailureType.RESOURCE_EXHAUSTION: self._inject_resource_exhaustion,
            FailureType.MEMORY_LEAK: self._inject_memory_leak,
            FailureType.CPU_SPIKE: self._inject_cpu_spike,
            FailureType.PARTIAL_FAILURE: self._inject_partial_failure,
            FailureType.CASCADING_FAILURE: self._inject_cascading_failure,
            FailureType.DEPENDENCY_FAILURE: self._inject_dependency_failure
        }
        
        # Service registry
        self.registered_services: Dict[str, Dict[str, Any]] = {}
        
        # Safety mechanisms
        self.safety_locks: Dict[str, asyncio.Lock] = {}
        self.emergency_stop_triggered: bool = False
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.scheduler_task: Optional[asyncio.Task] = None
        
        logger.info("Chaos engineer initialized")
    
    async def initialize(self):
        """Initialize chaos engineer."""
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Start scheduler if enabled
        if self.config.enable_scheduling:
            self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        logger.info("Chaos engineer initialized")
    
    async def close(self):
        """Close chaos engineer."""
        # Stop all active experiments
        await self._emergency_stop()
        
        # Cancel background tasks
        if self.monitoring_task:
            self.monitoring_task.cancel()
        if self.scheduler_task:
            self.scheduler_task.cancel()
        
        logger.info("Chaos engineer closed")
    
    def register_service(
        self,
        service_name: str,
        service_instance: Any,
        failure_modes: List[FailureType],
        custom_injectors: Optional[Dict[FailureType, Callable]] = None
    ):
        """Register a service for chaos testing."""
        self.registered_services[service_name] = {
            'instance': service_instance,
            'failure_modes': failure_modes,
            'custom_injectors': custom_injectors or {},
            'registered_at': datetime.now()
        }
        
        # Create safety lock
        self.safety_locks[service_name] = asyncio.Lock()
        
        logger.info(f"Registered service for chaos testing: {service_name}")
    
    async def run_experiment(
        self,
        experiment_name: str,
        failure_injections: List[FailureInjection],
        custom_config: Optional[ChaosConfig] = None
    ) -> ExperimentMetrics:
        """Run a chaos engineering experiment."""
        config = custom_config or self.config
        experiment_id = f"{experiment_name}_{int(time.time())}"
        
        # Safety check
        if len(self.active_experiments) >= config.max_concurrent_experiments:
            raise ChaosExperimentError("Maximum concurrent experiments reached")
        
        if self.emergency_stop_triggered:
            raise ChaosExperimentError("Emergency stop is active")
        
        # Initialize experiment
        experiment_metrics = ExperimentMetrics(
            experiment_id=experiment_id,
            phase=ExperimentPhase.SETUP,
            start_time=datetime.now()
        )
        
        self.active_experiments[experiment_id] = {
            'name': experiment_name,
            'config': config,
            'injections': failure_injections,
            'metrics': experiment_metrics,
            'start_time': time.time()
        }
        
        logger.info(f"Starting chaos experiment: {experiment_name} ({experiment_id})")
        
        try:
            # Phase 1: Setup
            await self._experiment_setup(experiment_id)
            
            # Phase 2: Baseline measurement
            await self._measure_baseline(experiment_id)
            
            # Phase 3: Failure injection
            await self._inject_failures(experiment_id)
            
            # Phase 4: Recovery monitoring
            await self._monitor_recovery(experiment_id)
            
            # Phase 5: Validation
            await self._validate_experiment(experiment_id)
            
            # Phase 6: Cleanup
            await self._cleanup_experiment(experiment_id)
            
            experiment_metrics.experiment_success = True
            experiment_metrics.end_time = datetime.now()
            
            logger.info(f"Chaos experiment completed successfully: {experiment_id}")
            
        except Exception as e:
            experiment_metrics.experiment_success = False
            experiment_metrics.end_time = datetime.now()
            experiment_metrics.findings.append(f"Experiment failed: {str(e)}")
            
            # Emergency cleanup
            await self._emergency_cleanup(experiment_id)
            
            logger.error(f"Chaos experiment failed: {experiment_id} - {e}")
            raise
        
        finally:
            # Remove from active experiments
            self.active_experiments.pop(experiment_id, None)
            
            # Store in history
            self.experiment_history.append(experiment_metrics)
        
        return experiment_metrics
    
    async def _experiment_setup(self, experiment_id: str):
        """Set up chaos experiment."""
        experiment = self.active_experiments[experiment_id]
        experiment['metrics'].phase = ExperimentPhase.SETUP
        
        # Validate target services
        for injection in experiment['injections']:
            if injection.target_service not in self.registered_services:
                raise ChaosExperimentError(f"Service not registered: {injection.target_service}")
        
        # Set experiment ID on injections
        for injection in experiment['injections']:
            injection.experiment_id = experiment_id
        
        logger.info(f"Experiment setup completed: {experiment_id}")
    
    async def _measure_baseline(self, experiment_id: str):
        """Measure baseline system performance."""
        experiment = self.active_experiments[experiment_id]
        experiment['metrics'].phase = ExperimentPhase.BASELINE
        
        config = experiment['config']
        
        # Collect baseline metrics
        baseline_start = time.time()
        baseline_metrics = []
        
        while time.time() - baseline_start < config.baseline_duration:
            metrics = await self._collect_system_metrics()
            baseline_metrics.append(metrics)
            await asyncio.sleep(1)
        
        # Calculate baseline averages
        if baseline_metrics:
            experiment['baseline'] = {
                'system_health_score': sum(m['system_health_score'] for m in baseline_metrics) / len(baseline_metrics),
                'error_rate': sum(m['error_rate'] for m in baseline_metrics) / len(baseline_metrics),
                'response_time': sum(m['response_time'] for m in baseline_metrics) / len(baseline_metrics),
                'throughput': sum(m['throughput'] for m in baseline_metrics) / len(baseline_metrics)
            }
        
        logger.info(f"Baseline measurement completed: {experiment_id}")
    
    async def _inject_failures(self, experiment_id: str):
        """Inject failures according to experiment configuration."""
        experiment = self.active_experiments[experiment_id]
        experiment['metrics'].phase = ExperimentPhase.INJECTION
        
        # Start all failure injections
        injection_tasks = []
        
        for injection in experiment['injections']:
            # Check if injection should be triggered
            if injection.condition and not injection.condition():
                continue
            
            if random.random() > injection.trigger_probability:
                continue
            
            # Create injection task
            task = asyncio.create_task(self._execute_injection(injection))
            injection_tasks.append(task)
        
        # Wait for all injections to complete
        if injection_tasks:
            await asyncio.gather(*injection_tasks, return_exceptions=True)
        
        logger.info(f"Failure injection completed: {experiment_id}")
    
    async def _execute_injection(self, injection: FailureInjection):
        """Execute a single failure injection."""
        logger.info(f"Executing failure injection: {injection.failure_type.value} on {injection.target_service}")
        
        # Get service instance
        service_info = self.registered_services.get(injection.target_service)
        if not service_info:
            logger.error(f"Service not found: {injection.target_service}")
            return
        
        # Check if failure type is supported
        if injection.failure_type not in service_info['failure_modes']:
            logger.error(f"Failure type {injection.failure_type.value} not supported for {injection.target_service}")
            return
        
        # Get injector function
        injector = service_info['custom_injectors'].get(
            injection.failure_type,
            self.failure_injectors.get(injection.failure_type)
        )
        
        if not injector:
            logger.error(f"No injector found for failure type: {injection.failure_type.value}")
            return
        
        # Execute injection with safety lock
        async with self.safety_locks[injection.target_service]:
            try:
                await injector(injection)
            except Exception as e:
                logger.error(f"Failure injection error: {e}")
    
    async def _inject_network_delay(self, injection: FailureInjection):
        """Inject network delay."""
        delay = injection.parameters.get('delay', 0.1)
        
        # Simulate network delay
        await asyncio.sleep(delay * injection.intensity)
        
        logger.info(f"Network delay injected: {delay * injection.intensity}s")
    
    async def _inject_network_failure(self, injection: FailureInjection):
        """Inject network failure."""
        failure_rate = injection.parameters.get('failure_rate', 0.5)
        
        # Simulate network failure
        if random.random() < failure_rate * injection.intensity:
            raise ConnectionError("Simulated network failure")
        
        logger.info(f"Network failure injected with rate: {failure_rate * injection.intensity}")
    
    async def _inject_service_unavailable(self, injection: FailureInjection):
        """Inject service unavailable error."""
        # Simulate service unavailable
        if random.random() < injection.intensity:
            raise Exception("Service unavailable (503)")
        
        logger.info("Service unavailable error injected")
    
    async def _inject_timeout(self, injection: FailureInjection):
        """Inject timeout error."""
        timeout = injection.parameters.get('timeout', 5.0)
        
        # Simulate timeout
        await asyncio.sleep(timeout * injection.intensity)
        raise asyncio.TimeoutError("Simulated timeout")
    
    async def _inject_exception(self, injection: FailureInjection):
        """Inject general exception."""
        exception_type = injection.parameters.get('exception_type', 'RuntimeError')
        message = injection.parameters.get('message', 'Simulated exception')
        
        # Raise specified exception
        if exception_type == 'RuntimeError':
            raise RuntimeError(message)
        elif exception_type == 'ValueError':
            raise ValueError(message)
        else:
            raise Exception(message)
    
    async def _inject_resource_exhaustion(self, injection: FailureInjection):
        """Inject resource exhaustion."""
        resource_type = injection.parameters.get('resource_type', 'memory')
        
        if resource_type == 'memory':
            # Simulate memory exhaustion
            data = [0] * int(1000000 * injection.intensity)
            await asyncio.sleep(injection.duration)
            del data
        
        logger.info(f"Resource exhaustion injected: {resource_type}")
    
    async def _inject_memory_leak(self, injection: FailureInjection):
        """Inject memory leak."""
        # Simulate memory leak
        leak_size = int(100000 * injection.intensity)
        leaked_data = [0] * leak_size
        
        # Keep reference to simulate leak
        injection.parameters['leaked_data'] = leaked_data
        
        logger.info(f"Memory leak injected: {leak_size} objects")
    
    async def _inject_cpu_spike(self, injection: FailureInjection):
        """Inject CPU spike."""
        duration = injection.duration * injection.intensity
        
        # Simulate CPU spike
        start_time = time.time()
        while time.time() - start_time < duration:
            # Busy wait to consume CPU
            pass
        
        logger.info(f"CPU spike injected for {duration}s")
    
    async def _inject_partial_failure(self, injection: FailureInjection):
        """Inject partial failure."""
        failure_percentage = injection.parameters.get('failure_percentage', 0.5)
        
        # Simulate partial failure
        if random.random() < failure_percentage * injection.intensity:
            await self._inject_exception(injection)
        
        logger.info(f"Partial failure injected: {failure_percentage * injection.intensity:.1%}")
    
    async def _inject_cascading_failure(self, injection: FailureInjection):
        """Inject cascading failure."""
        # Trigger failures in dependent services
        dependent_services = injection.parameters.get('dependent_services', [])
        
        for service in dependent_services:
            if service in self.registered_services:
                # Create cascading injection
                cascading_injection = FailureInjection(
                    failure_type=FailureType.SERVICE_UNAVAILABLE,
                    target_service=service,
                    intensity=injection.intensity * 0.8,
                    duration=injection.duration // 2
                )
                
                await self._execute_injection(cascading_injection)
        
        logger.info(f"Cascading failure injected to {len(dependent_services)} services")
    
    async def _inject_dependency_failure(self, injection: FailureInjection):
        """Inject dependency failure."""
        dependency_name = injection.parameters.get('dependency', 'database')
        
        # Simulate dependency failure
        if random.random() < injection.intensity:
            raise Exception(f"Dependency '{dependency_name}' is unavailable")
        
        logger.info(f"Dependency failure injected: {dependency_name}")
    
    async def _monitor_recovery(self, experiment_id: str):
        """Monitor system recovery after failure injection."""
        experiment = self.active_experiments[experiment_id]
        experiment['metrics'].phase = ExperimentPhase.RECOVERY
        
        config = experiment['config']
        recovery_start = time.time()
        recovery_metrics = []
        
        while time.time() - recovery_start < config.recovery_duration:
            metrics = await self._collect_system_metrics()
            recovery_metrics.append(metrics)
            
            # Check for recovery
            if metrics['system_health_score'] > config.recovery_threshold:
                experiment['metrics'].recovery_time = time.time() - recovery_start
                experiment['metrics'].recovery_success = True
                break
            
            await asyncio.sleep(1)
        
        # Store recovery metrics
        experiment['recovery_metrics'] = recovery_metrics
        
        logger.info(f"Recovery monitoring completed: {experiment_id}")
    
    async def _validate_experiment(self, experiment_id: str):
        """Validate experiment results."""
        experiment = self.active_experiments[experiment_id]
        experiment['metrics'].phase = ExperimentPhase.VALIDATION
        
        baseline = experiment.get('baseline', {})
        recovery_metrics = experiment.get('recovery_metrics', [])
        
        if not baseline or not recovery_metrics:
            experiment['metrics'].findings.append("Insufficient data for validation")
            return
        
        # Calculate recovery performance
        final_metrics = recovery_metrics[-1] if recovery_metrics else {}
        
        # Compare with baseline
        health_recovery = final_metrics.get('system_health_score', 0) / baseline.get('system_health_score', 1)
        response_time_impact = final_metrics.get('response_time', 0) / baseline.get('response_time', 1)
        
        # Generate findings
        findings = []
        recommendations = []
        
        if health_recovery < 0.9:
            findings.append(f"System health recovery incomplete: {health_recovery:.1%}")
            recommendations.append("Review recovery procedures and circuit breaker settings")
        
        if response_time_impact > 1.5:
            findings.append(f"Response time degraded: {response_time_impact:.1f}x baseline")
            recommendations.append("Optimize retry mechanisms and timeouts")
        
        if experiment['metrics'].recovery_success:
            findings.append("System successfully recovered from failure")
        else:
            findings.append("System failed to recover within expected timeframe")
            recommendations.append("Implement faster failure detection and recovery")
        
        experiment['metrics'].findings = findings
        experiment['metrics'].recommendations = recommendations
        
        logger.info(f"Experiment validation completed: {experiment_id}")
    
    async def _cleanup_experiment(self, experiment_id: str):
        """Clean up experiment resources."""
        experiment = self.active_experiments[experiment_id]
        experiment['metrics'].phase = ExperimentPhase.CLEANUP
        
        # Clean up any leaked resources
        for injection in experiment['injections']:
            if 'leaked_data' in injection.parameters:
                del injection.parameters['leaked_data']
        
        logger.info(f"Experiment cleanup completed: {experiment_id}")
    
    async def _emergency_cleanup(self, experiment_id: str):
        """Emergency cleanup of experiment."""
        try:
            await self._cleanup_experiment(experiment_id)
        except Exception as e:
            logger.error(f"Emergency cleanup failed: {e}")
    
    async def _emergency_stop(self):
        """Emergency stop all experiments."""
        self.emergency_stop_triggered = True
        
        # Cancel all active experiments
        for experiment_id in list(self.active_experiments.keys()):
            await self._emergency_cleanup(experiment_id)
            self.active_experiments.pop(experiment_id, None)
        
        logger.warning("Emergency stop triggered - all experiments stopped")
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        metrics = {
            'system_health_score': 1.0,
            'error_rate': 0.0,
            'response_time': 0.1,
            'throughput': 100.0,
            'timestamp': time.time()
        }
        
        # Collect from health monitor
        if self.health_monitor:
            try:
                health_summary = self.health_monitor.get_system_health_summary()
                metrics['system_health_score'] = health_summary.get('system_health_score', 1.0) / 100.0
            except Exception as e:
                logger.error(f"Failed to collect health metrics: {e}")
        
        # Collect from circuit breakers
        if self.circuit_breaker_manager:
            try:
                # Aggregate circuit breaker metrics
                pass
            except Exception as e:
                logger.error(f"Failed to collect circuit breaker metrics: {e}")
        
        return metrics
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while True:
            try:
                await asyncio.sleep(10)
                await self._monitor_safety()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    async def _monitor_safety(self):
        """Monitor safety conditions."""
        if not self.config.enable_safety_checks:
            return
        
        # Check system health
        metrics = await self._collect_system_metrics()
        
        if metrics['system_health_score'] < self.config.emergency_stop_threshold:
            logger.critical("Emergency stop triggered due to low system health")
            await self._emergency_stop()
    
    async def _scheduler_loop(self):
        """Background scheduler loop."""
        while True:
            try:
                await asyncio.sleep(self.config.schedule_interval)
                
                # Check if we should run scheduled experiments
                if self._should_run_scheduled_experiment():
                    await self._run_scheduled_experiment()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
    
    def _should_run_scheduled_experiment(self) -> bool:
        """Check if scheduled experiment should run."""
        if not self.config.enable_scheduling:
            return False
        
        # Check business hours
        if self.config.business_hours_only:
            current_hour = datetime.now().hour
            if not (9 <= current_hour <= 17):
                return False
        
        # Check if any experiments are running
        if self.active_experiments:
            return False
        
        return True
    
    async def _run_scheduled_experiment(self):
        """Run a scheduled chaos experiment."""
        # Create basic failure injection
        injection = FailureInjection(
            failure_type=FailureType.NETWORK_DELAY,
            target_service=random.choice(list(self.registered_services.keys())),
            intensity=0.3,
            duration=30
        )
        
        try:
            await self.run_experiment("scheduled_chaos", [injection])
        except Exception as e:
            logger.error(f"Scheduled experiment failed: {e}")
    
    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get status of a specific experiment."""
        experiment = self.active_experiments.get(experiment_id)
        if not experiment:
            return {}
        
        return {
            'experiment_id': experiment_id,
            'name': experiment['name'],
            'phase': experiment['metrics'].phase.value,
            'start_time': experiment['start_time'],
            'running_time': time.time() - experiment['start_time'],
            'injections': [
                {
                    'type': inj.failure_type.value,
                    'target': inj.target_service,
                    'intensity': inj.intensity
                }
                for inj in experiment['injections']
            ]
        }
    
    def get_all_experiments_status(self) -> Dict[str, Any]:
        """Get status of all experiments."""
        return {
            'active_experiments': {
                exp_id: self.get_experiment_status(exp_id)
                for exp_id in self.active_experiments.keys()
            },
            'experiment_count': len(self.active_experiments),
            'emergency_stop_triggered': self.emergency_stop_triggered,
            'registered_services': list(self.registered_services.keys())
        }
    
    def get_experiment_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get experiment history."""
        return [
            {
                'experiment_id': exp.experiment_id,
                'phase': exp.phase.value,
                'start_time': exp.start_time.isoformat(),
                'end_time': exp.end_time.isoformat() if exp.end_time else None,
                'success': exp.experiment_success,
                'recovery_success': exp.recovery_success,
                'recovery_time': exp.recovery_time,
                'findings': exp.findings,
                'recommendations': exp.recommendations
            }
            for exp in self.experiment_history[-limit:]
        ]


class ChaosExperimentError(Exception):
    """Exception raised during chaos experiments."""
    pass


class ChaosInjectionError(Exception):
    """Exception raised during failure injection."""
    pass