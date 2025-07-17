"""
Load Testing Suite for Execution Engine MARL System
==================================================

Advanced load testing implementation for validating the execution engine under
high-throughput, high-concurrency production conditions.

Load Test Scenarios:
- Sustained Load: 100+ RPS for extended periods
- Burst Load: 500+ RPS for short durations
- Concurrency Test: 50+ concurrent requests
- Stress Test: Progressive load increase until failure
- Endurance Test: 24-hour continuous operation simulation

Author: Agent 5 - Integration Validation & Production Certification
Date: 2025-07-13
Mission: 200% Production Load Testing Certification
"""

import asyncio
import time
import numpy as np
import statistics
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import json
import concurrent.futures
import threading
import multiprocessing
from dataclasses import dataclass, field
import structlog
import psutil
import gc

# Load testing imports
from src.execution.unified_execution_marl_system import (
    UnifiedExecutionMARLSystem, ExecutionDecision, DEFAULT_CONFIG
)
from src.execution.execution_context_processor import (
    ExecutionContextProcessor, RawMarketData, RawPortfolioData, DEFAULT_PROCESSOR_CONFIG
)
from src.execution.agents.centralized_critic import ExecutionContext, MarketFeatures

logger = structlog.get_logger()


@dataclass
class LoadTestMetrics:
    """Comprehensive load test metrics"""
    # Test configuration
    test_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    target_rps: int
    concurrent_requests: int
    
    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    
    # Performance metrics
    actual_rps: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    
    # System resource metrics
    peak_cpu_percent: float = 0.0
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    avg_memory_mb: float = 0.0
    
    # Error analysis
    error_types: Dict[str, int] = field(default_factory=dict)
    error_rate: float = 0.0
    
    # Throughput over time
    throughput_timeline: List[Tuple[float, float]] = field(default_factory=list)  # (time, rps)
    latency_timeline: List[Tuple[float, float]] = field(default_factory=list)   # (time, p95_latency)
    
    # Performance requirements compliance
    latency_compliant: bool = False
    throughput_compliant: bool = False
    error_rate_compliant: bool = False
    overall_compliant: bool = False


@dataclass
class StressTestResult:
    """Stress test results showing system breaking point"""
    max_sustained_rps: float
    max_burst_rps: float
    max_concurrent_users: int
    breaking_point_rps: float
    failure_mode: str
    recovery_time_seconds: float
    degradation_curve: List[Tuple[float, float]]  # (load, performance)


class LoadTestRunner:
    """
    Advanced load test runner for execution engine
    
    Provides comprehensive load testing capabilities including:
    - Sustained load testing
    - Burst testing
    - Concurrency testing
    - Stress testing to failure
    - Performance regression testing
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize load test runner"""
        self.config = config or {}
        
        # Test configuration
        self.test_timeout = self.config.get('test_timeout', 300)  # 5 minutes default
        self.warmup_duration = self.config.get('warmup_duration', 10)  # 10 seconds warmup
        self.cooldown_duration = self.config.get('cooldown_duration', 5)  # 5 seconds cooldown
        
        # Performance targets
        self.latency_target_ms = 0.5  # 500μs = 0.5ms
        self.target_rps = 100
        self.max_error_rate = 0.01  # 1%
        
        # Resource monitoring
        self.monitor_interval = 1.0  # seconds
        self.system_monitor_active = False
        self.resource_metrics = []
        
        # Test data cache for performance
        self.test_data_cache = []
        self._initialize_test_data_cache()
        
        logger.info("LoadTestRunner initialized",
                   test_timeout=self.test_timeout,
                   warmup_duration=self.warmup_duration,
                   latency_target_ms=self.latency_target_ms,
                   target_rps=self.target_rps)
    
    def _initialize_test_data_cache(self):
        """Pre-generate test data for better performance"""
        logger.info("Initializing test data cache...")
        
        cache_size = 1000
        for i in range(cache_size):
            market_data, portfolio_data = self._create_test_data(i)
            self.test_data_cache.append((market_data, portfolio_data))
        
        logger.info(f"Test data cache initialized with {cache_size} entries")
    
    async def run_load_test_suite(self) -> Dict[str, Any]:
        """
        Run comprehensive load test suite
        
        Returns:
            Complete load test results
        """
        logger.info("🚀 Starting comprehensive load test suite")
        
        suite_start_time = time.time()
        test_results = {}
        
        try:
            # Initialize test systems
            execution_system = await self._initialize_test_system()
            
            # Test 1: Baseline Performance Test
            logger.info("📊 Running baseline performance test...")
            test_results['baseline'] = await self._run_baseline_test(execution_system)
            
            # Test 2: Sustained Load Test
            logger.info("⚡ Running sustained load test...")
            test_results['sustained_load'] = await self._run_sustained_load_test(execution_system)
            
            # Test 3: Burst Load Test
            logger.info("💥 Running burst load test...")
            test_results['burst_load'] = await self._run_burst_load_test(execution_system)
            
            # Test 4: Concurrency Test
            logger.info("🔀 Running concurrency test...")
            test_results['concurrency'] = await self._run_concurrency_test(execution_system)
            
            # Test 5: Stress Test
            logger.info("🔥 Running stress test...")
            test_results['stress_test'] = await self._run_stress_test(execution_system)
            
            # Test 6: Endurance Test (shortened for demo)
            logger.info("🏃 Running endurance test...")
            test_results['endurance'] = await self._run_endurance_test(execution_system)
            
            # Cleanup
            await execution_system.shutdown()
            
            # Compile suite results
            suite_duration = time.time() - suite_start_time
            
            suite_results = {
                'suite_metadata': {
                    'start_time': datetime.fromtimestamp(suite_start_time).isoformat(),
                    'duration_seconds': suite_duration,
                    'tests_completed': len(test_results),
                    'environment': self._get_environment_info()
                },
                'test_results': test_results,
                'suite_analysis': self._analyze_suite_results(test_results),
                'production_certification': self._assess_production_certification(test_results)
            }
            
            logger.info("✅ Load test suite completed",
                       duration_seconds=suite_duration,
                       tests_completed=len(test_results))
            
            return suite_results
            
        except Exception as e:
            logger.error("❌ Load test suite failed", error=str(e))
            return {
                'error': f"Load test suite failed: {str(e)}",
                'duration_seconds': time.time() - suite_start_time,
                'partial_results': test_results
            }
    
    async def _initialize_test_system(self) -> UnifiedExecutionMARLSystem:
        """Initialize execution system for load testing"""
        # Use optimized configuration for load testing
        test_config = DEFAULT_CONFIG.copy()
        test_config['max_workers'] = min(20, multiprocessing.cpu_count() * 2)
        
        return UnifiedExecutionMARLSystem(test_config)
    
    async def _run_baseline_test(self, execution_system: UnifiedExecutionMARLSystem) -> LoadTestMetrics:
        """Run baseline performance test with moderate load"""
        test_name = "baseline_performance"
        target_rps = 50
        duration = 30
        concurrent_requests = 10
        
        logger.info(f"Starting {test_name} test", 
                   target_rps=target_rps, 
                   duration=duration,
                   concurrent_requests=concurrent_requests)
        
        return await self._execute_load_test(
            execution_system=execution_system,
            test_name=test_name,
            target_rps=target_rps,
            duration_seconds=duration,
            concurrent_requests=concurrent_requests
        )
    
    async def _run_sustained_load_test(self, execution_system: UnifiedExecutionMARLSystem) -> LoadTestMetrics:
        """Run sustained load test at target RPS"""
        test_name = "sustained_load"
        target_rps = 100
        duration = 60  # 1 minute sustained
        concurrent_requests = 20
        
        logger.info(f"Starting {test_name} test", 
                   target_rps=target_rps, 
                   duration=duration,
                   concurrent_requests=concurrent_requests)
        
        return await self._execute_load_test(
            execution_system=execution_system,
            test_name=test_name,
            target_rps=target_rps,
            duration_seconds=duration,
            concurrent_requests=concurrent_requests
        )
    
    async def _run_burst_load_test(self, execution_system: UnifiedExecutionMARLSystem) -> LoadTestMetrics:
        """Run burst load test with high RPS for short duration"""
        test_name = "burst_load"
        target_rps = 500
        duration = 15  # 15 seconds burst
        concurrent_requests = 50
        
        logger.info(f"Starting {test_name} test", 
                   target_rps=target_rps, 
                   duration=duration,
                   concurrent_requests=concurrent_requests)
        
        return await self._execute_load_test(
            execution_system=execution_system,
            test_name=test_name,
            target_rps=target_rps,
            duration_seconds=duration,
            concurrent_requests=concurrent_requests
        )
    
    async def _run_concurrency_test(self, execution_system: UnifiedExecutionMARLSystem) -> Dict[str, Any]:
        """Run concurrency test with varying concurrent request levels"""
        test_name = "concurrency_scaling"
        concurrency_levels = [10, 25, 50, 75, 100]
        duration = 20  # seconds per level
        target_rps = 75  # Moderate RPS to focus on concurrency
        
        concurrency_results = {}
        
        for concurrency in concurrency_levels:
            logger.info(f"Testing concurrency level: {concurrency}")
            
            test_result = await self._execute_load_test(
                execution_system=execution_system,
                test_name=f"{test_name}_{concurrency}",
                target_rps=target_rps,
                duration_seconds=duration,
                concurrent_requests=concurrency
            )
            
            concurrency_results[f'concurrency_{concurrency}'] = test_result
            
            # Short cooldown between levels
            await asyncio.sleep(2)
        
        # Analyze concurrency scaling
        scaling_analysis = self._analyze_concurrency_scaling(concurrency_results)
        
        return {
            'test_name': test_name,
            'concurrency_results': concurrency_results,
            'scaling_analysis': scaling_analysis,
            'max_supported_concurrency': scaling_analysis['max_sustainable_concurrency']
        }
    
    async def _run_stress_test(self, execution_system: UnifiedExecutionMARLSystem) -> StressTestResult:
        """Run stress test to find system breaking point"""
        test_name = "stress_test"
        logger.info(f"Starting {test_name} - finding system breaking point")
        
        # Progressive load increase
        load_levels = [50, 100, 200, 300, 500, 750, 1000, 1500]
        duration_per_level = 10  # seconds
        concurrent_requests = 50
        
        stress_results = []
        breaking_point_found = False
        breaking_point_rps = 0
        failure_mode = "none"
        
        for target_rps in load_levels:
            logger.info(f"Stress testing at {target_rps} RPS")
            
            try:
                test_result = await self._execute_load_test(
                    execution_system=execution_system,
                    test_name=f"{test_name}_{target_rps}",
                    target_rps=target_rps,
                    duration_seconds=duration_per_level,
                    concurrent_requests=concurrent_requests,
                    timeout_per_request=5.0  # Shorter timeout for stress test
                )
                
                stress_results.append((target_rps, test_result))
                
                # Check for failure conditions
                if (test_result.error_rate > 0.1 or  # >10% error rate
                    test_result.p95_latency_ms > 5.0 or  # >5ms latency
                    test_result.actual_rps < target_rps * 0.8):  # <80% of target RPS
                    
                    breaking_point_found = True
                    breaking_point_rps = target_rps
                    
                    if test_result.error_rate > 0.1:
                        failure_mode = "high_error_rate"
                    elif test_result.p95_latency_ms > 5.0:
                        failure_mode = "latency_degradation"
                    else:
                        failure_mode = "throughput_degradation"
                    
                    logger.warning(f"Breaking point found at {target_rps} RPS", 
                                  failure_mode=failure_mode)
                    break
                
            except Exception as e:
                logger.error(f"Stress test failed at {target_rps} RPS", error=str(e))
                breaking_point_found = True
                breaking_point_rps = target_rps
                failure_mode = "system_failure"
                break
            
            # Short recovery period
            await asyncio.sleep(1)
        
        # Determine maximum sustainable performance
        max_sustained_rps = 0
        max_burst_rps = 0
        
        for target_rps, result in stress_results:
            if result.error_rate <= 0.05 and result.p95_latency_ms <= 1.0:  # Sustained criteria
                max_sustained_rps = max(max_sustained_rps, result.actual_rps)
            
            if result.error_rate <= 0.1:  # Burst criteria (more lenient)
                max_burst_rps = max(max_burst_rps, result.actual_rps)
        
        # Create degradation curve
        degradation_curve = [
            (target_rps, result.actual_rps / target_rps)  # Performance ratio
            for target_rps, result in stress_results
        ]
        
        return StressTestResult(
            max_sustained_rps=max_sustained_rps,
            max_burst_rps=max_burst_rps,
            max_concurrent_users=concurrent_requests,
            breaking_point_rps=breaking_point_rps,
            failure_mode=failure_mode,
            recovery_time_seconds=0.0,  # Would measure in real test
            degradation_curve=degradation_curve
        )
    
    async def _run_endurance_test(self, execution_system: UnifiedExecutionMARLSystem) -> LoadTestMetrics:
        """Run endurance test (shortened for demo)"""
        test_name = "endurance"
        target_rps = 75  # Moderate load for endurance
        duration = 120  # 2 minutes (normally would be hours)
        concurrent_requests = 25
        
        logger.info(f"Starting {test_name} test", 
                   target_rps=target_rps, 
                   duration=duration,
                   concurrent_requests=concurrent_requests)
        
        return await self._execute_load_test(
            execution_system=execution_system,
            test_name=test_name,
            target_rps=target_rps,
            duration_seconds=duration,
            concurrent_requests=concurrent_requests
        )
    
    async def _execute_load_test(self,
                               execution_system: UnifiedExecutionMARLSystem,
                               test_name: str,
                               target_rps: int,
                               duration_seconds: int,
                               concurrent_requests: int,
                               timeout_per_request: float = 10.0) -> LoadTestMetrics:
        """Execute a single load test with specified parameters"""
        
        # Initialize metrics
        metrics = LoadTestMetrics(
            test_name=test_name,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=duration_seconds,
            target_rps=target_rps,
            concurrent_requests=concurrent_requests
        )
        
        # Start system monitoring
        await self._start_system_monitoring()
        
        # Warmup phase
        if self.warmup_duration > 0:
            logger.info(f"Warmup phase: {self.warmup_duration} seconds")
            await self._warmup_phase(execution_system)
        
        # Main test execution
        test_start = time.time()
        
        # Request tracking
        request_semaphore = asyncio.Semaphore(concurrent_requests)
        request_times = []
        request_results = []
        request_count = 0
        
        # Request rate control
        request_interval = 1.0 / target_rps
        
        async def execute_single_request(request_id: int) -> Dict[str, Any]:
            """Execute single request with timing"""
            async with request_semaphore:
                request_start = time.perf_counter()
                
                try:
                    # Get test data
                    market_data, portfolio_data = self.test_data_cache[request_id % len(self.test_data_cache)]
                    
                    # Create execution context
                    context_processor = ExecutionContextProcessor(DEFAULT_PROCESSOR_CONFIG)
                    execution_context_tensor = await context_processor.process_execution_context(
                        market_data, portfolio_data
                    )
                    execution_context = self._tensor_to_execution_context(execution_context_tensor)
                    market_features = self._create_market_features(market_data)
                    
                    # Execute decision
                    decision = await asyncio.wait_for(
                        execution_system.execute_unified_decision(execution_context, market_features),
                        timeout=timeout_per_request
                    )
                    
                    request_time = (time.perf_counter() - request_start) * 1000  # ms
                    
                    await context_processor.shutdown()
                    
                    return {
                        'success': True,
                        'latency_ms': request_time,
                        'fill_rate': decision.fill_rate,
                        'error': None
                    }
                    
                except asyncio.TimeoutError:
                    request_time = (time.perf_counter() - request_start) * 1000
                    return {
                        'success': False,
                        'latency_ms': request_time,
                        'fill_rate': 0.0,
                        'error': 'timeout'
                    }
                except Exception as e:
                    request_time = (time.perf_counter() - request_start) * 1000
                    return {
                        'success': False,
                        'latency_ms': request_time,
                        'fill_rate': 0.0,
                        'error': str(e)
                    }
        
        # Execute test with rate limiting
        test_tasks = []
        
        while time.time() - test_start < duration_seconds:
            batch_start = time.time()
            
            # Create batch of requests
            batch_size = min(concurrent_requests, target_rps)
            batch_tasks = [
                execute_single_request(request_count + i)
                for i in range(batch_size)
            ]
            
            # Execute batch
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    request_results.append({
                        'success': False,
                        'latency_ms': 0,
                        'fill_rate': 0.0,
                        'error': str(result)
                    })
                else:
                    request_results.append(result)
                
                request_count += 1
            
            # Rate limiting
            batch_time = time.time() - batch_start
            if batch_time < request_interval:
                await asyncio.sleep(request_interval - batch_time)
            
            # Periodic metrics collection
            if request_count % (target_rps // 4) == 0:  # 4 times per second
                current_time = time.time() - test_start
                current_rps = request_count / current_time if current_time > 0 else 0
                
                # Recent latencies for timeline
                recent_latencies = [r['latency_ms'] for r in request_results[-target_rps:] if r['success']]
                current_p95 = np.percentile(recent_latencies, 95) if recent_latencies else 0
                
                metrics.throughput_timeline.append((current_time, current_rps))
                metrics.latency_timeline.append((current_time, current_p95))
        
        # Stop system monitoring
        await self._stop_system_monitoring()
        
        # Calculate final metrics
        metrics.end_time = datetime.now()
        metrics.duration_seconds = time.time() - test_start
        metrics.total_requests = len(request_results)
        
        # Success/failure counts
        successful_results = [r for r in request_results if r['success']]
        failed_results = [r for r in request_results if not r['success']]
        timeout_results = [r for r in failed_results if r['error'] == 'timeout']
        
        metrics.successful_requests = len(successful_results)
        metrics.failed_requests = len(failed_results)
        metrics.timeout_requests = len(timeout_results)
        
        # Performance metrics
        metrics.actual_rps = metrics.total_requests / metrics.duration_seconds
        
        if successful_results:
            successful_latencies = [r['latency_ms'] for r in successful_results]
            metrics.avg_latency_ms = np.mean(successful_latencies)
            metrics.p50_latency_ms = np.percentile(successful_latencies, 50)
            metrics.p95_latency_ms = np.percentile(successful_latencies, 95)
            metrics.p99_latency_ms = np.percentile(successful_latencies, 99)
            metrics.max_latency_ms = np.max(successful_latencies)
        
        # Error analysis
        metrics.error_rate = len(failed_results) / metrics.total_requests if metrics.total_requests > 0 else 0
        error_types = {}
        for result in failed_results:
            error_type = result['error'] or 'unknown'
            error_types[error_type] = error_types.get(error_type, 0) + 1
        metrics.error_types = error_types
        
        # System resource metrics
        if self.resource_metrics:
            cpu_values = [m['cpu_percent'] for m in self.resource_metrics]
            memory_values = [m['memory_mb'] for m in self.resource_metrics]
            
            metrics.peak_cpu_percent = max(cpu_values)
            metrics.peak_memory_mb = max(memory_values)
            metrics.avg_cpu_percent = np.mean(cpu_values)
            metrics.avg_memory_mb = np.mean(memory_values)
        
        # Compliance assessment
        metrics.latency_compliant = metrics.p95_latency_ms <= self.latency_target_ms
        metrics.throughput_compliant = metrics.actual_rps >= target_rps * 0.9  # 90% of target
        metrics.error_rate_compliant = metrics.error_rate <= self.max_error_rate
        metrics.overall_compliant = all([
            metrics.latency_compliant,
            metrics.throughput_compliant,
            metrics.error_rate_compliant
        ])
        
        logger.info(f"Load test '{test_name}' completed",
                   actual_rps=metrics.actual_rps,
                   p95_latency_ms=metrics.p95_latency_ms,
                   error_rate=metrics.error_rate,
                   compliant=metrics.overall_compliant)
        
        return metrics
    
    async def _warmup_phase(self, execution_system: UnifiedExecutionMARLSystem):
        """Execute warmup phase to stabilize system performance"""
        warmup_rps = 10
        warmup_requests = int(self.warmup_duration * warmup_rps)
        
        for i in range(warmup_requests):
            try:
                market_data, portfolio_data = self.test_data_cache[i % len(self.test_data_cache)]
                
                context_processor = ExecutionContextProcessor(DEFAULT_PROCESSOR_CONFIG)
                execution_context_tensor = await context_processor.process_execution_context(
                    market_data, portfolio_data
                )
                execution_context = self._tensor_to_execution_context(execution_context_tensor)
                market_features = self._create_market_features(market_data)
                
                await execution_system.execute_unified_decision(execution_context, market_features)
                await context_processor.shutdown()
                
                await asyncio.sleep(1.0 / warmup_rps)
                
            except Exception as e:
                logger.warning(f"Warmup request failed: {e}")
    
    async def _start_system_monitoring(self):
        """Start system resource monitoring"""
        self.system_monitor_active = True
        self.resource_metrics = []
        
        async def monitor_resources():
            while self.system_monitor_active:
                try:
                    cpu_percent = psutil.cpu_percent(interval=None)
                    memory_info = psutil.virtual_memory()
                    memory_mb = memory_info.used / (1024 * 1024)
                    
                    self.resource_metrics.append({
                        'timestamp': time.time(),
                        'cpu_percent': cpu_percent,
                        'memory_mb': memory_mb,
                        'memory_percent': memory_info.percent
                    })
                    
                    await asyncio.sleep(self.monitor_interval)
                    
                except Exception as e:
                    logger.warning(f"Resource monitoring error: {e}")
                    break
        
        # Start monitoring task
        asyncio.create_task(monitor_resources())
    
    async def _stop_system_monitoring(self):
        """Stop system resource monitoring"""
        self.system_monitor_active = False
        await asyncio.sleep(self.monitor_interval * 2)  # Wait for monitor to stop
    
    def _create_test_data(self, index: int) -> Tuple[RawMarketData, RawPortfolioData]:
        """Create test data for load testing"""
        base_price = 16800 + (index % 100) * 10
        
        market_data = RawMarketData(
            timestamp=datetime.now(),
            price=base_price + np.random.normal(0, 5),
            bid=base_price - 0.5,
            ask=base_price + 0.5,
            volume=np.random.randint(1000, 5000),
            bid_volume=np.random.randint(500, 2500),
            ask_volume=np.random.randint(500, 2500),
            bid_depth=np.random.uniform(10000, 30000),
            ask_depth=np.random.uniform(10000, 30000),
            tick_direction=np.random.choice([-1, 0, 1]),
            trade_size=np.random.uniform(1, 50)
        )
        
        portfolio_data = RawPortfolioData(
            timestamp=datetime.now(),
            portfolio_value=100000 + np.random.uniform(-2500, 2500),
            available_capital=50000 + np.random.uniform(-1250, 1250),
            current_position=np.random.uniform(-0.2, 0.2),
            unrealized_pnl=np.random.normal(0, 500),
            var_estimate=np.random.uniform(0.015, 0.035),
            sharpe_ratio=np.random.uniform(0.5, 2.0),
            max_drawdown=np.random.uniform(0.0, 0.1),
            current_drawdown=np.random.uniform(0.0, 0.05)
        )
        
        return market_data, portfolio_data
    
    def _tensor_to_execution_context(self, tensor: torch.Tensor) -> ExecutionContext:
        """Convert tensor to ExecutionContext (simplified)"""
        values = tensor.numpy()
        
        return ExecutionContext(
            portfolio_value=values[0] * 100000.0,
            available_capital=values[1] * values[0] * 100000.0,
            current_position=values[2],
            unrealized_pnl=values[3] * values[0] * 100000.0,
            realized_pnl=0.0,
            var_estimate=values[4],
            expected_return=values[5],
            volatility=values[6],
            sharpe_ratio=values[13] * 3.0,
            max_drawdown=0.0,
            drawdown_current=values[14],
            time_since_last_trade=values[11] * 24.0 * 3600.0,
            risk_budget_used=values[2],
            correlation_risk=values[9],
            liquidity_score=1.0 - values[7]
        )
    
    def _create_market_features(self, market_data: RawMarketData) -> MarketFeatures:
        """Create MarketFeatures from market data (simplified)"""
        return MarketFeatures(
            buy_volume=market_data.ask_volume,
            sell_volume=market_data.bid_volume,
            order_flow_imbalance=(market_data.ask_volume - market_data.bid_volume) / (market_data.ask_volume + market_data.bid_volume + 1),
            support_level=market_data.bid,
            resistance_level=market_data.ask,
            atm_vol=0.15,
            realized_garch=0.12,
            correlation_spy=0.5
        )
    
    def _analyze_concurrency_scaling(self, concurrency_results: Dict[str, LoadTestMetrics]) -> Dict[str, Any]:
        """Analyze concurrency scaling performance"""
        concurrency_levels = []
        throughput_values = []
        latency_values = []
        error_rates = []
        
        for key, metrics in concurrency_results.items():
            concurrency = int(key.split('_')[1])
            concurrency_levels.append(concurrency)
            throughput_values.append(metrics.actual_rps)
            latency_values.append(metrics.p95_latency_ms)
            error_rates.append(metrics.error_rate)
        
        # Find maximum sustainable concurrency
        max_sustainable = 0
        for i, (concurrency, error_rate, latency) in enumerate(zip(concurrency_levels, error_rates, latency_values)):
            if error_rate <= 0.05 and latency <= 1.0:  # Sustainable criteria
                max_sustainable = concurrency
        
        # Calculate scaling efficiency
        scaling_efficiency = []
        for i in range(1, len(throughput_values)):
            efficiency = throughput_values[i] / throughput_values[0]  # Relative to baseline
            scaling_efficiency.append(efficiency)
        
        return {
            'max_sustainable_concurrency': max_sustainable,
            'peak_throughput_rps': max(throughput_values),
            'scaling_efficiency': np.mean(scaling_efficiency) if scaling_efficiency else 1.0,
            'latency_degradation_slope': np.polyfit(concurrency_levels, latency_values, 1)[0],
            'throughput_saturation_point': self._find_saturation_point(concurrency_levels, throughput_values)
        }
    
    def _find_saturation_point(self, x_values: List[float], y_values: List[float]) -> float:
        """Find the point where throughput saturates"""
        if len(x_values) < 3:
            return x_values[-1] if x_values else 0
        
        # Find point where derivative becomes negative or very small
        derivatives = []
        for i in range(1, len(y_values)):
            derivative = (y_values[i] - y_values[i-1]) / (x_values[i] - x_values[i-1])
            derivatives.append(derivative)
        
        # Find first point where derivative drops significantly
        for i, derivative in enumerate(derivatives):
            if derivative < 0.1:  # Very small improvement
                return x_values[i + 1]
        
        return x_values[-1]  # No saturation found
    
    def _analyze_suite_results(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall suite results"""
        all_compliant = True
        performance_summary = {}
        
        for test_name, result in test_results.items():
            if isinstance(result, LoadTestMetrics):
                performance_summary[test_name] = {
                    'actual_rps': result.actual_rps,
                    'p95_latency_ms': result.p95_latency_ms,
                    'error_rate': result.error_rate,
                    'compliant': result.overall_compliant
                }
                if not result.overall_compliant:
                    all_compliant = False
        
        return {
            'all_tests_compliant': all_compliant,
            'performance_summary': performance_summary,
            'system_limits': self._extract_system_limits(test_results),
            'recommendations': self._generate_recommendations(test_results)
        }
    
    def _extract_system_limits(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract system performance limits from test results"""
        limits = {
            'max_sustained_rps': 0,
            'max_burst_rps': 0,
            'max_concurrent_requests': 0,
            'min_latency_ms': float('inf'),
            'max_error_free_rps': 0
        }
        
        for test_name, result in test_results.items():
            if isinstance(result, LoadTestMetrics):
                if result.error_rate <= 0.01:  # Error-free performance
                    limits['max_error_free_rps'] = max(limits['max_error_free_rps'], result.actual_rps)
                
                limits['min_latency_ms'] = min(limits['min_latency_ms'], result.p95_latency_ms)
                
                if 'sustained' in test_name and result.overall_compliant:
                    limits['max_sustained_rps'] = max(limits['max_sustained_rps'], result.actual_rps)
                
                if 'burst' in test_name and result.error_rate <= 0.1:
                    limits['max_burst_rps'] = max(limits['max_burst_rps'], result.actual_rps)
            
            elif isinstance(result, dict) and 'max_supported_concurrency' in result:
                limits['max_concurrent_requests'] = result['max_supported_concurrency']
            
            elif isinstance(result, StressTestResult):
                limits['max_sustained_rps'] = max(limits['max_sustained_rps'], result.max_sustained_rps)
                limits['max_burst_rps'] = max(limits['max_burst_rps'], result.max_burst_rps)
        
        return limits
    
    def _generate_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Analyze results for recommendations
        latency_issues = []
        throughput_issues = []
        error_issues = []
        
        for test_name, result in test_results.items():
            if isinstance(result, LoadTestMetrics):
                if not result.latency_compliant:
                    latency_issues.append(test_name)
                if not result.throughput_compliant:
                    throughput_issues.append(test_name)
                if not result.error_rate_compliant:
                    error_issues.append(test_name)
        
        # Generate specific recommendations
        if latency_issues:
            recommendations.append(f"Optimize latency for tests: {', '.join(latency_issues)}")
            recommendations.append("Consider async processing optimizations and caching")
        
        if throughput_issues:
            recommendations.append(f"Improve throughput for tests: {', '.join(throughput_issues)}")
            recommendations.append("Consider horizontal scaling and load balancing")
        
        if error_issues:
            recommendations.append(f"Reduce error rates for tests: {', '.join(error_issues)}")
            recommendations.append("Implement better error handling and retry mechanisms")
        
        if not recommendations:
            recommendations.append("System performance meets all requirements - ready for production")
        
        return recommendations
    
    def _assess_production_certification(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess production readiness certification"""
        requirements_met = {
            'sustained_load': False,
            'burst_capacity': False,
            'concurrency_support': False,
            'stress_resilience': False,
            'error_tolerance': False
        }
        
        # Check individual requirements
        for test_name, result in test_results.items():
            if 'sustained' in test_name and isinstance(result, LoadTestMetrics):
                requirements_met['sustained_load'] = result.overall_compliant
            
            elif 'burst' in test_name and isinstance(result, LoadTestMetrics):
                requirements_met['burst_capacity'] = result.error_rate <= 0.1
            
            elif 'concurrency' in test_name and isinstance(result, dict):
                requirements_met['concurrency_support'] = result.get('max_supported_concurrency', 0) >= 50
            
            elif 'stress' in test_name and isinstance(result, StressTestResult):
                requirements_met['stress_resilience'] = result.max_sustained_rps >= 100
        
        # Overall error tolerance
        all_error_rates = [
            result.error_rate for result in test_results.values() 
            if isinstance(result, LoadTestMetrics)
        ]
        if all_error_rates:
            requirements_met['error_tolerance'] = max(all_error_rates) <= 0.05
        
        # Calculate certification level
        compliance_rate = sum(requirements_met.values()) / len(requirements_met)
        
        if compliance_rate == 1.0:
            certification = "FULLY CERTIFIED FOR PRODUCTION"
        elif compliance_rate >= 0.8:
            certification = "CONDITIONALLY CERTIFIED - MINOR OPTIMIZATIONS NEEDED"
        elif compliance_rate >= 0.6:
            certification = "PRE-PRODUCTION - SIGNIFICANT IMPROVEMENTS REQUIRED"
        else:
            certification = "NOT CERTIFIED - MAJOR PERFORMANCE ISSUES"
        
        return {
            'certification_level': certification,
            'compliance_rate': compliance_rate,
            'requirements_met': requirements_met,
            'production_ready': compliance_rate >= 0.8,
            'blocking_issues': [req for req, met in requirements_met.items() if not met]
        }
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """Get environment information for test context"""
        return {
            'cpu_count': multiprocessing.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}",
            'platform': sys.platform
        }


# Standalone execution for load testing
if __name__ == "__main__":
    import sys
    
    async def main():
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
        # Run load test suite
        runner = LoadTestRunner()
        results = await runner.run_load_test_suite()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"load_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Load test completed. Results saved to {filename}")
        print(f"Production Certification: {results.get('production_certification', {}).get('certification_level', 'Unknown')}")
    
    asyncio.run(main())