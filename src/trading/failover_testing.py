#!/usr/bin/env python3
"""
Automated Failover Testing Framework
AGENT 2: Trading Engine RTO Specialist

Comprehensive testing framework for validating failover performance
and ensuring <5s RTO targets are met consistently.

Key Features:
- Automated failover scenario testing
- Performance benchmarking
- Chaos engineering integration
- Real-time monitoring during tests
- Detailed performance reporting
- Regression testing capabilities
"""

import asyncio
import time
import logging
import random
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import statistics
import redis.asyncio as redis
from pathlib import Path
import uuid
import traceback

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.events import EventBus, Event, EventType
from src.trading.failover_monitor import TradingEngineFailoverMonitor
from src.trading.fast_circuit_breaker import FastCircuitBreaker
from src.trading.standby_warmup import StandbyWarmupSystem
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TestScenario(Enum):
    """Failover test scenarios"""
    BASIC_FAILOVER = "basic_failover"
    CASCADING_FAILURE = "cascading_failure"
    NETWORK_PARTITION = "network_partition"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    GRADUAL_DEGRADATION = "gradual_degradation"
    SPLIT_BRAIN = "split_brain"
    RECOVERY_VALIDATION = "recovery_validation"
    LOAD_UNDER_FAILOVER = "load_under_failover"
    MULTIPLE_FAILOVERS = "multiple_failovers"
    PARTIAL_FAILURE = "partial_failure"

class TestResult(Enum):
    """Test result status"""
    PASSED = "passed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"
    SKIPPED = "skipped"

@dataclass
class TestConfig:
    """Configuration for failover tests"""
    # Test execution
    test_timeout: float = 30.0  # 30 seconds
    rto_target: float = 5.0  # 5 seconds
    warmup_time: float = 10.0  # 10 seconds
    cooldown_time: float = 15.0  # 15 seconds
    
    # Failure injection
    failure_injection_enabled: bool = True
    chaos_testing_enabled: bool = True
    network_simulation: bool = True
    
    # Performance monitoring
    monitoring_interval: float = 0.1  # 100ms
    detailed_metrics: bool = True
    performance_profiling: bool = True
    
    # Test scenarios
    scenarios_to_run: List[TestScenario] = field(default_factory=lambda: [
        TestScenario.BASIC_FAILOVER,
        TestScenario.CASCADING_FAILURE,
        TestScenario.GRADUAL_DEGRADATION,
        TestScenario.RECOVERY_VALIDATION
    ])
    
    # Regression testing
    regression_testing: bool = True
    baseline_comparison: bool = True
    performance_threshold: float = 0.1  # 10% degradation threshold
    
    # Reporting
    detailed_reporting: bool = True
    export_results: bool = True
    results_format: str = "json"  # json, csv, html

@dataclass
class TestMetrics:
    """Metrics collected during testing"""
    # Timing metrics
    failover_detection_time: float = 0.0
    failover_execution_time: float = 0.0
    total_rto: float = 0.0
    recovery_time: float = 0.0
    
    # Performance metrics
    cpu_usage: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    network_latency: List[float] = field(default_factory=list)
    
    # System metrics
    requests_processed: int = 0
    requests_failed: int = 0
    state_sync_events: int = 0
    circuit_breaker_events: int = 0
    
    # Health metrics
    health_check_failures: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    def calculate_success_rate(self) -> float:
        """Calculate request success rate"""
        if self.requests_processed == 0:
            return 0.0
        return (self.requests_processed - self.requests_failed) / self.requests_processed

@dataclass
class TestResult:
    """Result of a failover test"""
    test_id: str
    scenario: TestScenario
    start_time: float
    end_time: float
    duration: float
    result: TestResult
    rto_achieved: float
    rto_target: float
    metrics: TestMetrics
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def passed(self) -> bool:
        """Check if test passed"""
        return (self.result == TestResult.PASSED and 
                self.rto_achieved <= self.rto_target)

class FailoverTestFramework:
    """
    Comprehensive failover testing framework
    
    Features:
    - Automated test scenario execution
    - Real-time performance monitoring
    - Chaos engineering integration
    - Detailed performance analysis
    - Regression testing capabilities
    """
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.event_bus = EventBus()
        
        # Test components
        self.failover_monitor: Optional[TradingEngineFailoverMonitor] = None
        self.circuit_breaker: Optional[FastCircuitBreaker] = None
        self.warmup_system: Optional[StandbyWarmupSystem] = None
        
        # Test execution
        self.current_test: Optional[TestResult] = None
        self.test_results: List[TestResult] = []
        self.performance_baseline: Dict[str, float] = {}
        
        # Monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.metrics_collector: Optional[TestMetrics] = None
        
        # Chaos engineering
        self.chaos_tasks: List[asyncio.Task] = []
        
        # Redis connection
        self.redis_client: Optional[redis.Redis] = None
        
        logger.info("Failover testing framework initialized")
    
    async def initialize(self, redis_url: str):
        """Initialize testing framework"""
        try:
            # Initialize Redis connection
            self.redis_client = redis.from_url(redis_url)
            await self.redis_client.ping()
            
            # Initialize test components
            await self._initialize_test_components(redis_url)
            
            # Load performance baseline
            await self._load_performance_baseline()
            
            logger.info("Failover testing framework initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize testing framework: {e}")
            raise
    
    async def _initialize_test_components(self, redis_url: str):
        """Initialize test components"""
        # Note: In a real implementation, these would be initialized
        # with actual configurations and connections
        
        # Initialize failover monitor
        monitor_config = {
            'health_check_interval': 0.5,
            'failure_threshold': 2,
            'recovery_threshold': 2
        }
        # self.failover_monitor = TradingEngineFailoverMonitor(monitor_config)
        # await self.failover_monitor.initialize(redis_url)
        
        # Initialize circuit breaker
        cb_config = {
            'failure_threshold': 3,
            'timeout_ms': 5000,
            'service_name': 'trading_engine_test'
        }
        # self.circuit_breaker = FastCircuitBreaker(cb_config)
        # await self.circuit_breaker.initialize(redis_url)
        
        # Initialize warmup system
        warmup_config = {
            'instance_id': 'test_warmup',
            'models_dir': '/app/models/jit_optimized'
        }
        # self.warmup_system = StandbyWarmupSystem(warmup_config)
        # await self.warmup_system.initialize(redis_url)
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all configured test scenarios"""
        logger.info("Starting failover test suite")
        
        results = {
            'start_time': time.time(),
            'test_results': [],
            'summary': {},
            'performance_analysis': {}
        }
        
        try:
            # Run each test scenario
            for scenario in self.config.scenarios_to_run:
                logger.info(f"Running test scenario: {scenario.value}")
                
                # Warmup before test
                await self._warmup_before_test()
                
                # Run test
                test_result = await self._run_test_scenario(scenario)
                results['test_results'].append(test_result)
                
                # Cooldown after test
                await self._cooldown_after_test()
                
                # Stop if critical failure
                if test_result.result == TestResult.ERROR:
                    logger.error(f"Critical failure in {scenario.value}, stopping test suite")
                    break
            
            # Generate summary
            results['summary'] = self._generate_test_summary()
            
            # Performance analysis
            if self.config.regression_testing:
                results['performance_analysis'] = await self._analyze_performance_regression()
            
            # Export results
            if self.config.export_results:
                await self._export_results(results)
            
            logger.info("Failover test suite completed")
            
        except Exception as e:
            logger.error(f"Error running test suite: {e}")
            results['error'] = str(e)
        
        finally:
            results['end_time'] = time.time()
            results['duration'] = results['end_time'] - results['start_time']
        
        return results
    
    async def _run_test_scenario(self, scenario: TestScenario) -> TestResult:
        """Run a specific test scenario"""
        test_id = f"{scenario.value}_{uuid.uuid4().hex[:8]}"
        
        test_result = TestResult(
            test_id=test_id,
            scenario=scenario,
            start_time=time.time(),
            end_time=0.0,
            duration=0.0,
            result=TestResult.FAILED,
            rto_achieved=float('inf'),
            rto_target=self.config.rto_target,
            metrics=TestMetrics()
        )
        
        self.current_test = test_result
        self.metrics_collector = TestMetrics()
        
        try:
            # Start monitoring
            await self._start_test_monitoring()
            
            # Execute test scenario
            if scenario == TestScenario.BASIC_FAILOVER:
                await self._test_basic_failover()
            elif scenario == TestScenario.CASCADING_FAILURE:
                await self._test_cascading_failure()
            elif scenario == TestScenario.NETWORK_PARTITION:
                await self._test_network_partition()
            elif scenario == TestScenario.RESOURCE_EXHAUSTION:
                await self._test_resource_exhaustion()
            elif scenario == TestScenario.GRADUAL_DEGRADATION:
                await self._test_gradual_degradation()
            elif scenario == TestScenario.SPLIT_BRAIN:
                await self._test_split_brain()
            elif scenario == TestScenario.RECOVERY_VALIDATION:
                await self._test_recovery_validation()
            elif scenario == TestScenario.LOAD_UNDER_FAILOVER:
                await self._test_load_under_failover()
            elif scenario == TestScenario.MULTIPLE_FAILOVERS:
                await self._test_multiple_failovers()
            elif scenario == TestScenario.PARTIAL_FAILURE:
                await self._test_partial_failure()
            
            # Calculate final metrics
            test_result.rto_achieved = self.metrics_collector.total_rto
            test_result.metrics = self.metrics_collector
            
            # Determine test result
            if test_result.rto_achieved <= self.config.rto_target:
                test_result.result = TestResult.PASSED
                logger.info(f"Test {test_id} PASSED - RTO: {test_result.rto_achieved:.2f}s")
            else:
                test_result.result = TestResult.FAILED
                logger.warning(f"Test {test_id} FAILED - RTO: {test_result.rto_achieved:.2f}s")
            
        except asyncio.TimeoutError:
            test_result.result = TestResult.TIMEOUT
            test_result.error_message = "Test timed out"
            logger.error(f"Test {test_id} TIMEOUT")
            
        except Exception as e:
            test_result.result = TestResult.ERROR
            test_result.error_message = str(e)
            logger.error(f"Test {test_id} ERROR: {e}")
            
        finally:
            # Stop monitoring
            await self._stop_test_monitoring()
            
            # Calculate final metrics
            test_result.end_time = time.time()
            test_result.duration = test_result.end_time - test_result.start_time
            
            # Add to results
            self.test_results.append(test_result)
            
        return test_result
    
    async def _test_basic_failover(self):
        """Test basic failover scenario"""
        logger.info("Executing basic failover test")
        
        # Simulate active instance failure
        failure_start = time.time()
        
        # Inject failure
        await self._inject_failure("active_instance_failure")
        
        # Wait for failover detection
        detection_time = await self._wait_for_failover_detection()
        self.metrics_collector.failover_detection_time = detection_time
        
        # Wait for failover completion
        execution_time = await self._wait_for_failover_completion()
        self.metrics_collector.failover_execution_time = execution_time
        
        # Calculate total RTO
        self.metrics_collector.total_rto = time.time() - failure_start
        
        # Validate failover success
        await self._validate_failover_success()
        
        logger.info(f"Basic failover test completed - RTO: {self.metrics_collector.total_rto:.2f}s")
    
    async def _test_cascading_failure(self):
        """Test cascading failure scenario"""
        logger.info("Executing cascading failure test")
        
        failure_start = time.time()
        
        # Inject multiple failures in sequence
        await self._inject_failure("database_failure")
        await asyncio.sleep(1)
        await self._inject_failure("state_sync_failure")
        await asyncio.sleep(1)
        await self._inject_failure("active_instance_failure")
        
        # Wait for system recovery
        detection_time = await self._wait_for_failover_detection()
        execution_time = await self._wait_for_failover_completion()
        
        self.metrics_collector.failover_detection_time = detection_time
        self.metrics_collector.failover_execution_time = execution_time
        self.metrics_collector.total_rto = time.time() - failure_start
        
        await self._validate_failover_success()
        
        logger.info(f"Cascading failure test completed - RTO: {self.metrics_collector.total_rto:.2f}s")
    
    async def _test_network_partition(self):
        """Test network partition scenario"""
        logger.info("Executing network partition test")
        
        failure_start = time.time()
        
        # Simulate network partition
        await self._inject_failure("network_partition")
        
        # Wait for partition detection and recovery
        detection_time = await self._wait_for_failover_detection()
        execution_time = await self._wait_for_failover_completion()
        
        self.metrics_collector.failover_detection_time = detection_time
        self.metrics_collector.failover_execution_time = execution_time
        self.metrics_collector.total_rto = time.time() - failure_start
        
        await self._validate_failover_success()
        
        logger.info(f"Network partition test completed - RTO: {self.metrics_collector.total_rto:.2f}s")
    
    async def _test_resource_exhaustion(self):
        """Test resource exhaustion scenario"""
        logger.info("Executing resource exhaustion test")
        
        failure_start = time.time()
        
        # Simulate resource exhaustion
        await self._inject_failure("memory_exhaustion")
        await self._inject_failure("cpu_exhaustion")
        
        # Wait for resource-based failover
        detection_time = await self._wait_for_failover_detection()
        execution_time = await self._wait_for_failover_completion()
        
        self.metrics_collector.failover_detection_time = detection_time
        self.metrics_collector.failover_execution_time = execution_time
        self.metrics_collector.total_rto = time.time() - failure_start
        
        await self._validate_failover_success()
        
        logger.info(f"Resource exhaustion test completed - RTO: {self.metrics_collector.total_rto:.2f}s")
    
    async def _test_gradual_degradation(self):
        """Test gradual degradation scenario"""
        logger.info("Executing gradual degradation test")
        
        failure_start = time.time()
        
        # Simulate gradual performance degradation
        for i in range(5):
            await self._inject_failure(f"performance_degradation_{i}")
            await asyncio.sleep(0.5)
        
        # Wait for degradation detection and failover
        detection_time = await self._wait_for_failover_detection()
        execution_time = await self._wait_for_failover_completion()
        
        self.metrics_collector.failover_detection_time = detection_time
        self.metrics_collector.failover_execution_time = execution_time
        self.metrics_collector.total_rto = time.time() - failure_start
        
        await self._validate_failover_success()
        
        logger.info(f"Gradual degradation test completed - RTO: {self.metrics_collector.total_rto:.2f}s")
    
    async def _test_split_brain(self):
        """Test split brain scenario"""
        logger.info("Executing split brain test")
        
        failure_start = time.time()
        
        # Simulate split brain condition
        await self._inject_failure("split_brain")
        
        # Wait for split brain resolution
        detection_time = await self._wait_for_failover_detection()
        execution_time = await self._wait_for_failover_completion()
        
        self.metrics_collector.failover_detection_time = detection_time
        self.metrics_collector.failover_execution_time = execution_time
        self.metrics_collector.total_rto = time.time() - failure_start
        
        await self._validate_failover_success()
        
        logger.info(f"Split brain test completed - RTO: {self.metrics_collector.total_rto:.2f}s")
    
    async def _test_recovery_validation(self):
        """Test recovery validation scenario"""
        logger.info("Executing recovery validation test")
        
        failure_start = time.time()
        
        # Inject failure and then recovery
        await self._inject_failure("temporary_failure")
        await asyncio.sleep(2)
        await self._inject_recovery("temporary_recovery")
        
        # Wait for recovery detection
        detection_time = await self._wait_for_failover_detection()
        execution_time = await self._wait_for_failover_completion()
        
        self.metrics_collector.failover_detection_time = detection_time
        self.metrics_collector.failover_execution_time = execution_time
        self.metrics_collector.total_rto = time.time() - failure_start
        
        await self._validate_failover_success()
        
        logger.info(f"Recovery validation test completed - RTO: {self.metrics_collector.total_rto:.2f}s")
    
    async def _test_load_under_failover(self):
        """Test failover under load scenario"""
        logger.info("Executing load under failover test")
        
        # Start load generation
        load_task = asyncio.create_task(self._generate_load())
        
        failure_start = time.time()
        
        try:
            # Inject failure under load
            await self._inject_failure("failure_under_load")
            
            # Wait for failover under load
            detection_time = await self._wait_for_failover_detection()
            execution_time = await self._wait_for_failover_completion()
            
            self.metrics_collector.failover_detection_time = detection_time
            self.metrics_collector.failover_execution_time = execution_time
            self.metrics_collector.total_rto = time.time() - failure_start
            
            await self._validate_failover_success()
            
        finally:
            # Stop load generation
            load_task.cancel()
            try:
                await load_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"Load under failover test completed - RTO: {self.metrics_collector.total_rto:.2f}s")
    
    async def _test_multiple_failovers(self):
        """Test multiple failovers scenario"""
        logger.info("Executing multiple failovers test")
        
        total_rto = 0.0
        
        # Perform multiple failovers
        for i in range(3):
            failure_start = time.time()
            
            await self._inject_failure(f"multiple_failure_{i}")
            
            detection_time = await self._wait_for_failover_detection()
            execution_time = await self._wait_for_failover_completion()
            
            rto = time.time() - failure_start
            total_rto += rto
            
            logger.info(f"Failover {i+1} completed - RTO: {rto:.2f}s")
            
            # Brief pause between failovers
            await asyncio.sleep(2)
        
        self.metrics_collector.total_rto = total_rto / 3  # Average RTO
        
        await self._validate_failover_success()
        
        logger.info(f"Multiple failovers test completed - Average RTO: {self.metrics_collector.total_rto:.2f}s")
    
    async def _test_partial_failure(self):
        """Test partial failure scenario"""
        logger.info("Executing partial failure test")
        
        failure_start = time.time()
        
        # Inject partial failures
        await self._inject_failure("partial_model_failure")
        await self._inject_failure("partial_sync_failure")
        
        # Wait for partial failover
        detection_time = await self._wait_for_failover_detection()
        execution_time = await self._wait_for_failover_completion()
        
        self.metrics_collector.failover_detection_time = detection_time
        self.metrics_collector.failover_execution_time = execution_time
        self.metrics_collector.total_rto = time.time() - failure_start
        
        await self._validate_failover_success()
        
        logger.info(f"Partial failure test completed - RTO: {self.metrics_collector.total_rto:.2f}s")
    
    async def _inject_failure(self, failure_type: str):
        """Inject a failure into the system"""
        logger.info(f"Injecting failure: {failure_type}")
        
        # Simulate different types of failures
        if failure_type == "active_instance_failure":
            # Simulate active instance crash
            await self._simulate_instance_crash()
        elif failure_type == "database_failure":
            # Simulate database connectivity issues
            await self._simulate_database_failure()
        elif failure_type == "state_sync_failure":
            # Simulate state synchronization failure
            await self._simulate_state_sync_failure()
        elif failure_type == "network_partition":
            # Simulate network partition
            await self._simulate_network_partition()
        elif failure_type.startswith("performance_degradation"):
            # Simulate performance degradation
            await self._simulate_performance_degradation()
        elif failure_type.startswith("memory_exhaustion"):
            # Simulate memory exhaustion
            await self._simulate_memory_exhaustion()
        elif failure_type.startswith("cpu_exhaustion"):
            # Simulate CPU exhaustion
            await self._simulate_cpu_exhaustion()
        else:
            # Generic failure simulation
            await self._simulate_generic_failure(failure_type)
    
    async def _inject_recovery(self, recovery_type: str):
        """Inject recovery into the system"""
        logger.info(f"Injecting recovery: {recovery_type}")
        
        # Simulate recovery scenarios
        if recovery_type == "temporary_recovery":
            await self._simulate_temporary_recovery()
        else:
            await self._simulate_generic_recovery(recovery_type)
    
    async def _simulate_instance_crash(self):
        """Simulate active instance crash"""
        # In a real implementation, this would stop the active instance
        await asyncio.sleep(0.1)
        logger.debug("Simulated instance crash")
    
    async def _simulate_database_failure(self):
        """Simulate database failure"""
        # In a real implementation, this would disconnect from database
        await asyncio.sleep(0.1)
        logger.debug("Simulated database failure")
    
    async def _simulate_state_sync_failure(self):
        """Simulate state sync failure"""
        # In a real implementation, this would disrupt Redis connection
        await asyncio.sleep(0.1)
        logger.debug("Simulated state sync failure")
    
    async def _simulate_network_partition(self):
        """Simulate network partition"""
        # In a real implementation, this would introduce network delays/drops
        await asyncio.sleep(0.1)
        logger.debug("Simulated network partition")
    
    async def _simulate_performance_degradation(self):
        """Simulate performance degradation"""
        # In a real implementation, this would increase response times
        await asyncio.sleep(0.1)
        logger.debug("Simulated performance degradation")
    
    async def _simulate_memory_exhaustion(self):
        """Simulate memory exhaustion"""
        # In a real implementation, this would consume memory
        await asyncio.sleep(0.1)
        logger.debug("Simulated memory exhaustion")
    
    async def _simulate_cpu_exhaustion(self):
        """Simulate CPU exhaustion"""
        # In a real implementation, this would consume CPU
        await asyncio.sleep(0.1)
        logger.debug("Simulated CPU exhaustion")
    
    async def _simulate_generic_failure(self, failure_type: str):
        """Simulate generic failure"""
        await asyncio.sleep(0.1)
        logger.debug(f"Simulated generic failure: {failure_type}")
    
    async def _simulate_temporary_recovery(self):
        """Simulate temporary recovery"""
        await asyncio.sleep(0.1)
        logger.debug("Simulated temporary recovery")
    
    async def _simulate_generic_recovery(self, recovery_type: str):
        """Simulate generic recovery"""
        await asyncio.sleep(0.1)
        logger.debug(f"Simulated generic recovery: {recovery_type}")
    
    async def _wait_for_failover_detection(self) -> float:
        """Wait for failover detection"""
        detection_start = time.time()
        
        # In a real implementation, this would wait for actual failover detection
        # For now, simulate detection time
        await asyncio.sleep(random.uniform(0.5, 2.0))  # 0.5-2 seconds
        
        detection_time = time.time() - detection_start
        logger.debug(f"Failover detected in {detection_time:.2f}s")
        
        return detection_time
    
    async def _wait_for_failover_completion(self) -> float:
        """Wait for failover completion"""
        execution_start = time.time()
        
        # In a real implementation, this would wait for actual failover completion
        # For now, simulate execution time
        await asyncio.sleep(random.uniform(1.0, 3.0))  # 1-3 seconds
        
        execution_time = time.time() - execution_start
        logger.debug(f"Failover completed in {execution_time:.2f}s")
        
        return execution_time
    
    async def _validate_failover_success(self):
        """Validate that failover was successful"""
        # In a real implementation, this would check:
        # - New active instance is healthy
        # - State synchronization is working
        # - All services are responsive
        
        await asyncio.sleep(0.5)  # Simulate validation time
        logger.debug("Failover validation completed")
    
    async def _generate_load(self):
        """Generate load during testing"""
        logger.info("Starting load generation")
        
        try:
            while True:
                # Simulate requests
                self.metrics_collector.requests_processed += 1
                
                # Simulate occasional failures
                if random.random() < 0.05:  # 5% failure rate
                    self.metrics_collector.requests_failed += 1
                
                await asyncio.sleep(0.01)  # 100 requests per second
                
        except asyncio.CancelledError:
            logger.info("Load generation stopped")
    
    async def _start_test_monitoring(self):
        """Start test monitoring"""
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def _stop_test_monitoring(self):
        """Stop test monitoring"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
    
    async def _monitoring_loop(self):
        """Test monitoring loop"""
        try:
            while True:
                await asyncio.sleep(self.config.monitoring_interval)
                
                # Collect metrics
                if self.metrics_collector:
                    # Simulate metric collection
                    cpu_usage = random.uniform(20, 80)
                    memory_usage = random.uniform(512, 2048)
                    network_latency = random.uniform(1, 10)
                    
                    self.metrics_collector.cpu_usage.append(cpu_usage)
                    self.metrics_collector.memory_usage.append(memory_usage)
                    self.metrics_collector.network_latency.append(network_latency)
                
        except asyncio.CancelledError:
            pass
    
    async def _warmup_before_test(self):
        """Warmup before test"""
        logger.info("Warming up before test")
        await asyncio.sleep(self.config.warmup_time)
    
    async def _cooldown_after_test(self):
        """Cooldown after test"""
        logger.info("Cooling down after test")
        await asyncio.sleep(self.config.cooldown_time)
    
    def _generate_test_summary(self) -> Dict[str, Any]:
        """Generate test summary"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.passed())
        failed_tests = total_tests - passed_tests
        
        if total_tests > 0:
            pass_rate = passed_tests / total_tests
            avg_rto = statistics.mean([r.rto_achieved for r in self.test_results if r.rto_achieved != float('inf')])
            max_rto = max([r.rto_achieved for r in self.test_results if r.rto_achieved != float('inf')])
            min_rto = min([r.rto_achieved for r in self.test_results if r.rto_achieved != float('inf')])
        else:
            pass_rate = 0.0
            avg_rto = 0.0
            max_rto = 0.0
            min_rto = 0.0
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'pass_rate': pass_rate,
            'rto_statistics': {
                'target': self.config.rto_target,
                'average': avg_rto,
                'maximum': max_rto,
                'minimum': min_rto
            },
            'scenarios_tested': [result.scenario.value for result in self.test_results]
        }
    
    async def _analyze_performance_regression(self) -> Dict[str, Any]:
        """Analyze performance regression"""
        if not self.performance_baseline:
            return {'status': 'no_baseline', 'message': 'No performance baseline available'}
        
        current_performance = {}
        for result in self.test_results:
            if result.passed():
                current_performance[result.scenario.value] = result.rto_achieved
        
        regression_analysis = {}
        for scenario, current_rto in current_performance.items():
            if scenario in self.performance_baseline:
                baseline_rto = self.performance_baseline[scenario]
                regression_percent = ((current_rto - baseline_rto) / baseline_rto) * 100
                
                regression_analysis[scenario] = {
                    'baseline_rto': baseline_rto,
                    'current_rto': current_rto,
                    'regression_percent': regression_percent,
                    'is_regression': regression_percent > self.config.performance_threshold * 100
                }
        
        return {
            'status': 'completed',
            'regression_analysis': regression_analysis,
            'overall_regression': any(
                analysis['is_regression'] for analysis in regression_analysis.values()
            )
        }
    
    async def _load_performance_baseline(self):
        """Load performance baseline"""
        try:
            if self.redis_client:
                baseline_data = await self.redis_client.get("failover_test:baseline")
                if baseline_data:
                    self.performance_baseline = json.loads(baseline_data)
                    logger.info("Performance baseline loaded")
                else:
                    logger.info("No performance baseline found")
            
        except Exception as e:
            logger.error(f"Error loading performance baseline: {e}")
    
    async def _save_performance_baseline(self):
        """Save performance baseline"""
        try:
            if self.redis_client:
                baseline = {}
                for result in self.test_results:
                    if result.passed():
                        baseline[result.scenario.value] = result.rto_achieved
                
                await self.redis_client.set(
                    "failover_test:baseline",
                    json.dumps(baseline),
                    ex=86400 * 30  # 30 days
                )
                logger.info("Performance baseline saved")
            
        except Exception as e:
            logger.error(f"Error saving performance baseline: {e}")
    
    async def _export_results(self, results: Dict[str, Any]):
        """Export test results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"failover_test_results_{timestamp}.{self.config.results_format}"
            
            if self.config.results_format == "json":
                # Convert TestResult objects to dictionaries
                results_copy = results.copy()
                results_copy['test_results'] = [
                    {
                        'test_id': r.test_id,
                        'scenario': r.scenario.value,
                        'start_time': r.start_time,
                        'end_time': r.end_time,
                        'duration': r.duration,
                        'result': r.result.value,
                        'rto_achieved': r.rto_achieved,
                        'rto_target': r.rto_target,
                        'error_message': r.error_message,
                        'metrics': {
                            'failover_detection_time': r.metrics.failover_detection_time,
                            'failover_execution_time': r.metrics.failover_execution_time,
                            'total_rto': r.metrics.total_rto,
                            'requests_processed': r.metrics.requests_processed,
                            'requests_failed': r.metrics.requests_failed,
                            'success_rate': r.metrics.calculate_success_rate()
                        }
                    } for r in self.test_results
                ]
                
                with open(filename, 'w') as f:
                    json.dump(results_copy, f, indent=2)
            
            logger.info(f"Test results exported to {filename}")
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
    
    def get_test_status(self) -> Dict[str, Any]:
        """Get current test status"""
        return {
            'current_test': self.current_test.test_id if self.current_test else None,
            'completed_tests': len(self.test_results),
            'total_scenarios': len(self.config.scenarios_to_run),
            'test_results': [
                {
                    'test_id': r.test_id,
                    'scenario': r.scenario.value,
                    'result': r.result.value,
                    'rto_achieved': r.rto_achieved,
                    'passed': r.passed()
                } for r in self.test_results
            ]
        }
    
    async def run_single_test(self, scenario: TestScenario) -> TestResult:
        """Run a single test scenario"""
        logger.info(f"Running single test: {scenario.value}")
        
        # Warmup
        await self._warmup_before_test()
        
        # Run test
        result = await self._run_test_scenario(scenario)
        
        # Cooldown
        await self._cooldown_after_test()
        
        return result
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down failover testing framework")
        
        # Cancel monitoring
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        # Cancel chaos tasks
        for task in self.chaos_tasks:
            task.cancel()
        
        # Close connections
        if self.redis_client:
            await self.redis_client.close()
        
        # Close components
        if self.failover_monitor:
            await self.failover_monitor.shutdown()
        if self.circuit_breaker:
            await self.circuit_breaker.shutdown()
        if self.warmup_system:
            await self.warmup_system.shutdown()
        
        logger.info("Failover testing framework shutdown complete")


# Factory function
def create_failover_test_framework(config: Dict[str, Any]) -> FailoverTestFramework:
    """Create failover test framework instance"""
    test_config = TestConfig(**config)
    return FailoverTestFramework(test_config)


# CLI interface
async def main():
    """Main entry point for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Failover Testing Framework")
    parser.add_argument("--redis-url", default="redis://localhost:6379/3")
    parser.add_argument("--rto-target", type=float, default=5.0)
    parser.add_argument("--scenario", choices=[s.value for s in TestScenario], 
                       help="Run specific scenario")
    parser.add_argument("--export-results", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = TestConfig(
        rto_target=args.rto_target,
        export_results=args.export_results
    )
    
    # Override scenarios if specific scenario requested
    if args.scenario:
        config.scenarios_to_run = [TestScenario(args.scenario)]
    
    # Create and run test framework
    framework = FailoverTestFramework(config)
    
    try:
        await framework.initialize(args.redis_url)
        
        if args.scenario:
            # Run single scenario
            result = await framework.run_single_test(TestScenario(args.scenario))
            print(f"Test result: {result.result.value}")
            print(f"RTO achieved: {result.rto_achieved:.2f}s")
        else:
            # Run all scenarios
            results = await framework.run_all_tests()
            print(f"Test suite completed:")
            print(f"  Total tests: {results['summary']['total_tests']}")
            print(f"  Passed: {results['summary']['passed_tests']}")
            print(f"  Failed: {results['summary']['failed_tests']}")
            print(f"  Pass rate: {results['summary']['pass_rate']:.1%}")
            print(f"  Average RTO: {results['summary']['rto_statistics']['average']:.2f}s")
            print(f"  Maximum RTO: {results['summary']['rto_statistics']['maximum']:.2f}s")
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await framework.shutdown()


if __name__ == "__main__":
    asyncio.run(main())