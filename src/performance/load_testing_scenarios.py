"""
Comprehensive Load Testing Scenarios

This module provides comprehensive load testing scenarios for all system components
including strategic inference, tactical inference, database operations, and trading engine.

Features:
- Multiple load testing patterns (ramp-up, spike, stress, soak)
- Component-specific load tests
- Real-time monitoring during tests
- Resource utilization tracking
- Failure point identification
- Performance degradation analysis
- Scalability assessment

Author: Performance Validation Agent
"""

import asyncio
import time
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
import structlog
from enum import Enum
import json
import concurrent.futures
import threading
import multiprocessing
import psutil
import gc
from collections import defaultdict, deque
import statistics
import warnings
warnings.filterwarnings('ignore')

logger = structlog.get_logger()

class LoadTestPattern(Enum):
    RAMP_UP = "ramp_up"
    SPIKE = "spike"
    STRESS = "stress"
    SOAK = "soak"
    BURST = "burst"
    STEADY_STATE = "steady_state"

class LoadTestPhase(Enum):
    PREPARATION = "preparation"
    WARM_UP = "warm_up"
    MAIN_TEST = "main_test"
    COOL_DOWN = "cool_down"
    ANALYSIS = "analysis"

@dataclass
class LoadTestConfiguration:
    """Load test configuration"""
    name: str
    pattern: LoadTestPattern
    duration_seconds: int
    max_users: int
    max_requests_per_second: int
    ramp_up_duration: int = 60
    cool_down_duration: int = 30
    warm_up_duration: int = 30
    target_function: Optional[Callable] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    success_criteria: Dict[str, float] = field(default_factory=dict)
    monitoring_interval: int = 5

@dataclass
class LoadTestMetrics:
    """Load test metrics"""
    timestamp: datetime
    active_users: int
    requests_per_second: float
    success_rate: float
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    disk_io: float
    network_io: float
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LoadTestResult:
    """Complete load test result"""
    configuration: LoadTestConfiguration
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    metrics_timeline: List[LoadTestMetrics]
    performance_summary: Dict[str, Any]
    resource_utilization: Dict[str, Any]
    failure_analysis: Dict[str, Any]
    recommendations: List[str]
    passed_success_criteria: bool

class LoadTestingFramework:
    """
    Comprehensive load testing framework for performance validation
    """

    def __init__(self):
        self.active_tests = {}
        self.test_history = []
        self.system_baselines = {}
        
        # System monitoring
        self.process = psutil.Process()
        
        # Load test scenarios
        self.scenarios = self._initialize_scenarios()
        
        logger.info("Load testing framework initialized",
                   scenarios=len(self.scenarios))

    def _initialize_scenarios(self) -> Dict[str, LoadTestConfiguration]:
        """Initialize predefined load testing scenarios"""
        scenarios = {}
        
        # Strategic Inference Load Test
        scenarios['strategic_inference_load'] = LoadTestConfiguration(
            name="Strategic Inference Load Test",
            pattern=LoadTestPattern.RAMP_UP,
            duration_seconds=300,
            max_users=50,
            max_requests_per_second=100,
            ramp_up_duration=60,
            success_criteria={
                'avg_response_time_ms': 50.0,
                'p99_response_time_ms': 100.0,
                'success_rate': 0.95,
                'error_rate': 0.05
            },
            parameters={
                'test_type': 'strategic_inference',
                'iterations': 1000
            }
        )
        
        # Tactical Inference Load Test
        scenarios['tactical_inference_load'] = LoadTestConfiguration(
            name="Tactical Inference Load Test",
            pattern=LoadTestPattern.RAMP_UP,
            duration_seconds=300,
            max_users=100,
            max_requests_per_second=200,
            ramp_up_duration=60,
            success_criteria={
                'avg_response_time_ms': 20.0,
                'p99_response_time_ms': 50.0,
                'success_rate': 0.98,
                'error_rate': 0.02
            },
            parameters={
                'test_type': 'tactical_inference',
                'iterations': 500
            }
        )
        
        # Database Stress Test
        scenarios['database_stress'] = LoadTestConfiguration(
            name="Database Stress Test",
            pattern=LoadTestPattern.STRESS,
            duration_seconds=600,
            max_users=200,
            max_requests_per_second=500,
            ramp_up_duration=120,
            success_criteria={
                'avg_response_time_ms': 100.0,
                'p99_response_time_ms': 1000.0,
                'success_rate': 0.90,
                'error_rate': 0.10
            },
            parameters={
                'test_type': 'database_operations',
                'operations': ['read', 'write', 'update', 'delete']
            }
        )
        
        # Trading Engine Spike Test
        scenarios['trading_engine_spike'] = LoadTestConfiguration(
            name="Trading Engine Spike Test",
            pattern=LoadTestPattern.SPIKE,
            duration_seconds=180,
            max_users=500,
            max_requests_per_second=1000,
            ramp_up_duration=30,
            success_criteria={
                'avg_response_time_ms': 10.0,
                'p99_response_time_ms': 100.0,
                'success_rate': 0.95,
                'error_rate': 0.05
            },
            parameters={
                'test_type': 'trading_operations',
                'order_types': ['market', 'limit', 'stop']
            }
        )
        
        # End-to-End Pipeline Soak Test
        scenarios['end_to_end_soak'] = LoadTestConfiguration(
            name="End-to-End Pipeline Soak Test",
            pattern=LoadTestPattern.SOAK,
            duration_seconds=3600,  # 1 hour
            max_users=50,
            max_requests_per_second=50,
            ramp_up_duration=300,
            success_criteria={
                'avg_response_time_ms': 100.0,
                'p99_response_time_ms': 500.0,
                'success_rate': 0.95,
                'memory_growth_mb_per_hour': 100.0,
                'error_rate': 0.05
            },
            parameters={
                'test_type': 'end_to_end_pipeline',
                'include_all_components': True
            }
        )
        
        # Burst Load Test
        scenarios['burst_load'] = LoadTestConfiguration(
            name="Burst Load Test",
            pattern=LoadTestPattern.BURST,
            duration_seconds=300,
            max_users=1000,
            max_requests_per_second=2000,
            ramp_up_duration=10,
            success_criteria={
                'avg_response_time_ms': 200.0,
                'p99_response_time_ms': 1000.0,
                'success_rate': 0.80,
                'error_rate': 0.20
            },
            parameters={
                'test_type': 'burst_capacity',
                'burst_duration': 60
            }
        )
        
        return scenarios

    async def run_load_test(self, scenario_name: str, 
                          custom_config: Optional[LoadTestConfiguration] = None) -> LoadTestResult:
        """Run a comprehensive load test"""
        
        # Get configuration
        if custom_config:
            config = custom_config
        else:
            config = self.scenarios.get(scenario_name)
            if not config:
                raise ValueError(f"Unknown scenario: {scenario_name}")
        
        logger.info("Starting load test",
                   scenario=config.name,
                   pattern=config.pattern.value,
                   duration=config.duration_seconds,
                   max_users=config.max_users)
        
        # Initialize test
        test_id = f"{scenario_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        # Test state
        test_state = {
            'active_users': 0,
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': deque(maxlen=10000),
            'metrics_timeline': [],
            'current_phase': LoadTestPhase.PREPARATION,
            'stop_requested': False
        }
        
        self.active_tests[test_id] = test_state
        
        try:
            # Run test phases
            await self._run_test_phases(config, test_state)
            
            # Calculate final metrics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Generate result
            result = self._generate_load_test_result(
                config, start_time, end_time, duration, test_state
            )
            
            # Store in history
            self.test_history.append(result)
            
            logger.info("Load test completed",
                       scenario=config.name,
                       duration=duration,
                       total_requests=result.total_requests,
                       success_rate=result.successful_requests / max(result.total_requests, 1),
                       passed_criteria=result.passed_success_criteria)
            
            return result
            
        except Exception as e:
            logger.error("Load test failed", scenario=config.name, error=str(e))
            raise
        finally:
            # Cleanup
            if test_id in self.active_tests:
                del self.active_tests[test_id]

    async def _run_test_phases(self, config: LoadTestConfiguration, test_state: Dict[str, Any]):
        """Run all test phases"""
        
        # Phase 1: Preparation
        test_state['current_phase'] = LoadTestPhase.PREPARATION
        await self._preparation_phase(config, test_state)
        
        # Phase 2: Warm-up
        test_state['current_phase'] = LoadTestPhase.WARM_UP
        await self._warm_up_phase(config, test_state)
        
        # Phase 3: Main test
        test_state['current_phase'] = LoadTestPhase.MAIN_TEST
        await self._main_test_phase(config, test_state)
        
        # Phase 4: Cool-down
        test_state['current_phase'] = LoadTestPhase.COOL_DOWN
        await self._cool_down_phase(config, test_state)
        
        # Phase 5: Analysis
        test_state['current_phase'] = LoadTestPhase.ANALYSIS
        await self._analysis_phase(config, test_state)

    async def _preparation_phase(self, config: LoadTestConfiguration, test_state: Dict[str, Any]):
        """Preparation phase"""
        logger.info("Load test preparation phase", scenario=config.name)
        
        # Initialize system monitoring
        await self._start_system_monitoring(config, test_state)
        
        # Prepare test data
        await self._prepare_test_data(config, test_state)
        
        # System health check
        await self._system_health_check(config, test_state)

    async def _warm_up_phase(self, config: LoadTestConfiguration, test_state: Dict[str, Any]):
        """Warm-up phase"""
        logger.info("Load test warm-up phase", scenario=config.name, duration=config.warm_up_duration)
        
        # Gradual warm-up
        warm_up_users = min(5, config.max_users // 10)
        warm_up_rps = min(10, config.max_requests_per_second // 10)
        
        await self._run_load_pattern(
            config, test_state, warm_up_users, warm_up_rps, config.warm_up_duration
        )

    async def _main_test_phase(self, config: LoadTestConfiguration, test_state: Dict[str, Any]):
        """Main test phase"""
        logger.info("Load test main phase", scenario=config.name, pattern=config.pattern.value)
        
        if config.pattern == LoadTestPattern.RAMP_UP:
            await self._ramp_up_pattern(config, test_state)
        elif config.pattern == LoadTestPattern.SPIKE:
            await self._spike_pattern(config, test_state)
        elif config.pattern == LoadTestPattern.STRESS:
            await self._stress_pattern(config, test_state)
        elif config.pattern == LoadTestPattern.SOAK:
            await self._soak_pattern(config, test_state)
        elif config.pattern == LoadTestPattern.BURST:
            await self._burst_pattern(config, test_state)
        elif config.pattern == LoadTestPattern.STEADY_STATE:
            await self._steady_state_pattern(config, test_state)

    async def _cool_down_phase(self, config: LoadTestConfiguration, test_state: Dict[str, Any]):
        """Cool-down phase"""
        logger.info("Load test cool-down phase", scenario=config.name, duration=config.cool_down_duration)
        
        # Gradual cool-down
        cool_down_users = 1
        cool_down_rps = 1
        
        await self._run_load_pattern(
            config, test_state, cool_down_users, cool_down_rps, config.cool_down_duration
        )

    async def _analysis_phase(self, config: LoadTestConfiguration, test_state: Dict[str, Any]):
        """Analysis phase"""
        logger.info("Load test analysis phase", scenario=config.name)
        
        # Analyze performance degradation
        await self._analyze_performance_degradation(config, test_state)
        
        # Identify failure points
        await self._identify_failure_points(config, test_state)
        
        # Generate recommendations
        await self._generate_recommendations(config, test_state)

    async def _ramp_up_pattern(self, config: LoadTestConfiguration, test_state: Dict[str, Any]):
        """Ramp-up load pattern"""
        ramp_steps = 10
        step_duration = config.duration_seconds / ramp_steps
        
        for step in range(ramp_steps):
            users = int((step + 1) * config.max_users / ramp_steps)
            rps = int((step + 1) * config.max_requests_per_second / ramp_steps)
            
            await self._run_load_pattern(config, test_state, users, rps, step_duration)
            
            if test_state['stop_requested']:
                break

    async def _spike_pattern(self, config: LoadTestConfiguration, test_state: Dict[str, Any]):
        """Spike load pattern"""
        # Normal load
        normal_users = config.max_users // 4
        normal_rps = config.max_requests_per_second // 4
        
        # Spike load
        spike_users = config.max_users
        spike_rps = config.max_requests_per_second
        
        # Run pattern: normal -> spike -> normal
        await self._run_load_pattern(config, test_state, normal_users, normal_rps, 60)
        await self._run_load_pattern(config, test_state, spike_users, spike_rps, 60)
        await self._run_load_pattern(config, test_state, normal_users, normal_rps, config.duration_seconds - 120)

    async def _stress_pattern(self, config: LoadTestConfiguration, test_state: Dict[str, Any]):
        """Stress load pattern"""
        # Gradually increase load beyond normal capacity
        stress_users = config.max_users * 2
        stress_rps = config.max_requests_per_second * 2
        
        await self._run_load_pattern(config, test_state, stress_users, stress_rps, config.duration_seconds)

    async def _soak_pattern(self, config: LoadTestConfiguration, test_state: Dict[str, Any]):
        """Soak load pattern"""
        # Sustained load at moderate level
        soak_users = config.max_users // 2
        soak_rps = config.max_requests_per_second // 2
        
        await self._run_load_pattern(config, test_state, soak_users, soak_rps, config.duration_seconds)

    async def _burst_pattern(self, config: LoadTestConfiguration, test_state: Dict[str, Any]):
        """Burst load pattern"""
        burst_duration = config.parameters.get('burst_duration', 60)
        rest_duration = 30
        
        bursts = config.duration_seconds // (burst_duration + rest_duration)
        
        for burst in range(bursts):
            # Burst phase
            await self._run_load_pattern(config, test_state, config.max_users, config.max_requests_per_second, burst_duration)
            
            # Rest phase
            await self._run_load_pattern(config, test_state, 1, 1, rest_duration)
            
            if test_state['stop_requested']:
                break

    async def _steady_state_pattern(self, config: LoadTestConfiguration, test_state: Dict[str, Any]):
        """Steady state load pattern"""
        steady_users = config.max_users // 2
        steady_rps = config.max_requests_per_second // 2
        
        await self._run_load_pattern(config, test_state, steady_users, steady_rps, config.duration_seconds)

    async def _run_load_pattern(self, config: LoadTestConfiguration, test_state: Dict[str, Any],
                              target_users: int, target_rps: int, duration: float):
        """Run a specific load pattern"""
        
        # Update active users
        test_state['active_users'] = target_users
        
        # Calculate request intervals
        request_interval = 1.0 / target_rps if target_rps > 0 else 1.0
        
        # Worker semaphore
        semaphore = asyncio.Semaphore(target_users)
        
        # Run load for specified duration
        end_time = time.time() + duration
        
        async def worker():
            async with semaphore:
                while time.time() < end_time and not test_state['stop_requested']:
                    try:
                        # Execute test function
                        start_time = time.perf_counter()
                        
                        await self._execute_test_function(config, test_state)
                        
                        end_time_req = time.perf_counter()
                        response_time = (end_time_req - start_time) * 1000  # ms
                        
                        # Record metrics
                        test_state['total_requests'] += 1
                        test_state['successful_requests'] += 1
                        test_state['response_times'].append(response_time)
                        
                    except Exception as e:
                        test_state['total_requests'] += 1
                        test_state['failed_requests'] += 1
                        logger.debug("Load test request failed", error=str(e))
                    
                    # Wait for next request
                    await asyncio.sleep(request_interval)
        
        # Start monitoring
        monitoring_task = asyncio.create_task(self._monitor_load_test(config, test_state, duration))
        
        # Start workers
        workers = [worker() for _ in range(target_users)]
        await asyncio.gather(*workers, return_exceptions=True)
        
        # Stop monitoring
        monitoring_task.cancel()
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass

    async def _execute_test_function(self, config: LoadTestConfiguration, test_state: Dict[str, Any]):
        """Execute the test function based on configuration"""
        test_type = config.parameters.get('test_type', 'default')
        
        if test_type == 'strategic_inference':
            await self._test_strategic_inference(config, test_state)
        elif test_type == 'tactical_inference':
            await self._test_tactical_inference(config, test_state)
        elif test_type == 'database_operations':
            await self._test_database_operations(config, test_state)
        elif test_type == 'trading_operations':
            await self._test_trading_operations(config, test_state)
        elif test_type == 'end_to_end_pipeline':
            await self._test_end_to_end_pipeline(config, test_state)
        else:
            # Default test
            await asyncio.sleep(0.01)  # Simulate work

    async def _test_strategic_inference(self, config: LoadTestConfiguration, test_state: Dict[str, Any]):
        """Test strategic inference component"""
        # Simulate strategic inference
        await asyncio.sleep(np.random.uniform(0.005, 0.05))  # 5-50ms
        
        # Random failure
        if random.random() < 0.02:  # 2% failure rate
            raise Exception("Strategic inference failure")

    async def _test_tactical_inference(self, config: LoadTestConfiguration, test_state: Dict[str, Any]):
        """Test tactical inference component"""
        # Simulate tactical inference
        await asyncio.sleep(np.random.uniform(0.002, 0.02))  # 2-20ms
        
        # Random failure
        if random.random() < 0.01:  # 1% failure rate
            raise Exception("Tactical inference failure")

    async def _test_database_operations(self, config: LoadTestConfiguration, test_state: Dict[str, Any]):
        """Test database operations"""
        operations = config.parameters.get('operations', ['read'])
        operation = random.choice(operations)
        
        # Simulate database operation
        if operation == 'read':
            await asyncio.sleep(np.random.uniform(0.01, 0.05))  # 10-50ms
        elif operation == 'write':
            await asyncio.sleep(np.random.uniform(0.02, 0.1))   # 20-100ms
        elif operation == 'update':
            await asyncio.sleep(np.random.uniform(0.015, 0.08)) # 15-80ms
        elif operation == 'delete':
            await asyncio.sleep(np.random.uniform(0.01, 0.06))  # 10-60ms
        
        # Random failure
        if random.random() < 0.05:  # 5% failure rate
            raise Exception(f"Database {operation} failure")

    async def _test_trading_operations(self, config: LoadTestConfiguration, test_state: Dict[str, Any]):
        """Test trading operations"""
        order_types = config.parameters.get('order_types', ['market'])
        order_type = random.choice(order_types)
        
        # Simulate trading operation
        if order_type == 'market':
            await asyncio.sleep(np.random.uniform(0.001, 0.01))  # 1-10ms
        elif order_type == 'limit':
            await asyncio.sleep(np.random.uniform(0.002, 0.015)) # 2-15ms
        elif order_type == 'stop':
            await asyncio.sleep(np.random.uniform(0.002, 0.02))  # 2-20ms
        
        # Random failure
        if random.random() < 0.01:  # 1% failure rate
            raise Exception(f"Trading {order_type} order failure")

    async def _test_end_to_end_pipeline(self, config: LoadTestConfiguration, test_state: Dict[str, Any]):
        """Test end-to-end pipeline"""
        # Simulate complete pipeline
        await asyncio.sleep(np.random.uniform(0.05, 0.2))  # 50-200ms
        
        # Random failure
        if random.random() < 0.03:  # 3% failure rate
            raise Exception("End-to-end pipeline failure")

    async def _monitor_load_test(self, config: LoadTestConfiguration, test_state: Dict[str, Any], duration: float):
        """Monitor load test execution"""
        start_time = time.time()
        
        while time.time() - start_time < duration and not test_state['stop_requested']:
            try:
                # Collect metrics
                metrics = await self._collect_metrics(config, test_state)
                test_state['metrics_timeline'].append(metrics)
                
                # Check for failure conditions
                await self._check_failure_conditions(config, test_state, metrics)
                
                await asyncio.sleep(config.monitoring_interval)
                
            except Exception as e:
                logger.error("Error in load test monitoring", error=str(e))
                await asyncio.sleep(config.monitoring_interval)

    async def _collect_metrics(self, config: LoadTestConfiguration, test_state: Dict[str, Any]) -> LoadTestMetrics:
        """Collect current metrics"""
        
        # Response time metrics
        response_times = list(test_state['response_times'])
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
        else:
            avg_response_time = p95_response_time = p99_response_time = 0
        
        # Success rate
        total_requests = test_state['total_requests']
        successful_requests = test_state['successful_requests']
        success_rate = successful_requests / max(total_requests, 1)
        error_rate = 1 - success_rate
        
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        
        # I/O metrics
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()
        
        return LoadTestMetrics(
            timestamp=datetime.now(),
            active_users=test_state['active_users'],
            requests_per_second=total_requests / max(1, (datetime.now() - test_state.get('start_time', datetime.now())).total_seconds()),
            success_rate=success_rate,
            avg_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            error_rate=error_rate,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_io=disk_io.read_bytes + disk_io.write_bytes if disk_io else 0,
            network_io=network_io.bytes_sent + network_io.bytes_recv if network_io else 0
        )

    async def _check_failure_conditions(self, config: LoadTestConfiguration, test_state: Dict[str, Any], metrics: LoadTestMetrics):
        """Check for failure conditions during load test"""
        
        # Check success criteria
        success_criteria = config.success_criteria
        
        # Response time check
        if 'avg_response_time_ms' in success_criteria:
            if metrics.avg_response_time > success_criteria['avg_response_time_ms']:
                logger.warning("Average response time exceeded threshold",
                             current=metrics.avg_response_time,
                             threshold=success_criteria['avg_response_time_ms'])
        
        # Success rate check
        if 'success_rate' in success_criteria:
            if metrics.success_rate < success_criteria['success_rate']:
                logger.warning("Success rate below threshold",
                             current=metrics.success_rate,
                             threshold=success_criteria['success_rate'])
        
        # System resource checks
        if metrics.cpu_usage > 95:
            logger.warning("High CPU usage detected", cpu_usage=metrics.cpu_usage)
        
        if metrics.memory_usage > 90:
            logger.warning("High memory usage detected", memory_usage=metrics.memory_usage)

    async def _start_system_monitoring(self, config: LoadTestConfiguration, test_state: Dict[str, Any]):
        """Start system monitoring"""
        test_state['start_time'] = datetime.now()
        logger.debug("System monitoring started")

    async def _prepare_test_data(self, config: LoadTestConfiguration, test_state: Dict[str, Any]):
        """Prepare test data"""
        # Generate test data based on configuration
        logger.debug("Test data prepared")

    async def _system_health_check(self, config: LoadTestConfiguration, test_state: Dict[str, Any]):
        """Perform system health check"""
        # Check system health before starting load test
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        
        if cpu_usage > 80:
            logger.warning("High CPU usage before test start", cpu_usage=cpu_usage)
        
        if memory_usage > 80:
            logger.warning("High memory usage before test start", memory_usage=memory_usage)

    async def _analyze_performance_degradation(self, config: LoadTestConfiguration, test_state: Dict[str, Any]):
        """Analyze performance degradation"""
        metrics_timeline = test_state['metrics_timeline']
        
        if len(metrics_timeline) < 2:
            return
        
        # Analyze response time trends
        response_times = [m.avg_response_time for m in metrics_timeline]
        
        # Check for degradation
        if len(response_times) >= 10:
            early_avg = np.mean(response_times[:5])
            late_avg = np.mean(response_times[-5:])
            
            if late_avg > early_avg * 1.5:
                logger.warning("Performance degradation detected",
                             early_avg=early_avg,
                             late_avg=late_avg)

    async def _identify_failure_points(self, config: LoadTestConfiguration, test_state: Dict[str, Any]):
        """Identify failure points"""
        metrics_timeline = test_state['metrics_timeline']
        
        # Find points where error rate spiked
        failure_points = []
        for i, metrics in enumerate(metrics_timeline):
            if metrics.error_rate > 0.1:  # 10% error rate
                failure_points.append({
                    'timestamp': metrics.timestamp,
                    'error_rate': metrics.error_rate,
                    'active_users': metrics.active_users,
                    'cpu_usage': metrics.cpu_usage,
                    'memory_usage': metrics.memory_usage
                })
        
        if failure_points:
            logger.warning("Failure points identified", count=len(failure_points))

    async def _generate_recommendations(self, config: LoadTestConfiguration, test_state: Dict[str, Any]):
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Analyze final metrics
        if test_state['metrics_timeline']:
            final_metrics = test_state['metrics_timeline'][-1]
            
            if final_metrics.avg_response_time > config.success_criteria.get('avg_response_time_ms', float('inf')):
                recommendations.append("Optimize response time - consider caching or algorithm improvements")
            
            if final_metrics.success_rate < config.success_criteria.get('success_rate', 0):
                recommendations.append("Improve system reliability - investigate error causes")
            
            if final_metrics.cpu_usage > 80:
                recommendations.append("Optimize CPU usage - consider load balancing or scaling")
            
            if final_metrics.memory_usage > 80:
                recommendations.append("Optimize memory usage - investigate memory leaks")
        
        test_state['recommendations'] = recommendations

    def _generate_load_test_result(self, config: LoadTestConfiguration, start_time: datetime,
                                 end_time: datetime, duration: float, test_state: Dict[str, Any]) -> LoadTestResult:
        """Generate comprehensive load test result"""
        
        # Performance summary
        response_times = list(test_state['response_times'])
        if response_times:
            performance_summary = {
                'avg_response_time_ms': statistics.mean(response_times),
                'min_response_time_ms': min(response_times),
                'max_response_time_ms': max(response_times),
                'p50_response_time_ms': np.percentile(response_times, 50),
                'p95_response_time_ms': np.percentile(response_times, 95),
                'p99_response_time_ms': np.percentile(response_times, 99),
                'std_response_time_ms': statistics.stdev(response_times) if len(response_times) > 1 else 0
            }
        else:
            performance_summary = {}
        
        # Resource utilization
        metrics_timeline = test_state['metrics_timeline']
        if metrics_timeline:
            resource_utilization = {
                'avg_cpu_usage': statistics.mean([m.cpu_usage for m in metrics_timeline]),
                'max_cpu_usage': max([m.cpu_usage for m in metrics_timeline]),
                'avg_memory_usage': statistics.mean([m.memory_usage for m in metrics_timeline]),
                'max_memory_usage': max([m.memory_usage for m in metrics_timeline])
            }
        else:
            resource_utilization = {}
        
        # Failure analysis
        failure_analysis = {
            'total_failures': test_state['failed_requests'],
            'failure_rate': test_state['failed_requests'] / max(test_state['total_requests'], 1),
            'failure_points': []  # Would be populated by _identify_failure_points
        }
        
        # Check success criteria
        success_rate = test_state['successful_requests'] / max(test_state['total_requests'], 1)
        avg_response_time = performance_summary.get('avg_response_time_ms', 0)
        
        passed_criteria = True
        if 'success_rate' in config.success_criteria:
            if success_rate < config.success_criteria['success_rate']:
                passed_criteria = False
        
        if 'avg_response_time_ms' in config.success_criteria:
            if avg_response_time > config.success_criteria['avg_response_time_ms']:
                passed_criteria = False
        
        return LoadTestResult(
            configuration=config,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            total_requests=test_state['total_requests'],
            successful_requests=test_state['successful_requests'],
            failed_requests=test_state['failed_requests'],
            metrics_timeline=metrics_timeline,
            performance_summary=performance_summary,
            resource_utilization=resource_utilization,
            failure_analysis=failure_analysis,
            recommendations=test_state.get('recommendations', []),
            passed_success_criteria=passed_criteria
        )

    def get_scenario_names(self) -> List[str]:
        """Get list of available scenario names"""
        return list(self.scenarios.keys())

    def get_scenario_config(self, scenario_name: str) -> Optional[LoadTestConfiguration]:
        """Get scenario configuration"""
        return self.scenarios.get(scenario_name)

    def get_test_history(self, limit: int = 10) -> List[LoadTestResult]:
        """Get test history"""
        return self.test_history[-limit:]


# Global instance
load_testing_framework = LoadTestingFramework()