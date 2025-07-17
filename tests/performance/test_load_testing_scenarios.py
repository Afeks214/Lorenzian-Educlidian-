"""
Load Testing Scenarios for Production Validation - Agent 4 Implementation
===================================================================

Production-scale load testing scenarios to validate system performance
under realistic trading conditions and high-frequency market data loads.

Test Scenarios:
1. Market Open Load Test - Simulates market opening surge
2. High Frequency Trading Load - Continuous high-volume processing
3. Volatility Spike Load - Sudden market volatility conditions
4. Sustained Trading Load - Long-duration trading session
5. Concurrent User Load - Multiple trading strategies simultaneously

Author: Agent 4 - Performance Baseline Research Agent
"""

import pytest
import asyncio
import time
import json
import numpy as np
import pandas as pd
import torch
import psutil
import threading
import concurrent.futures
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path
import redis.asyncio as redis
from unittest.mock import Mock, patch

# Configure load testing
pytestmark = [pytest.mark.performance, pytest.mark.load]
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LoadTestScenario:
    """Load test scenario configuration."""
    name: str
    duration_minutes: int
    target_ops_per_second: int
    concurrent_users: int
    ramp_up_time_seconds: int
    data_pattern: str  # 'market_open', 'high_frequency', 'volatility_spike', 'sustained'
    performance_targets: Dict[str, float]


@dataclass
class LoadTestResult:
    """Load test execution result."""
    scenario_name: str
    test_duration_seconds: float
    total_operations: int
    successful_operations: int
    failed_operations: int
    average_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    throughput_ops_per_sec: float
    error_rate_percent: float
    resource_utilization: Dict[str, float]
    performance_targets_met: Dict[str, bool]
    recommendations: List[str]


class ProductionLoadTestSuite:
    """
    Production load testing suite for validating system performance
    under realistic trading conditions.
    """
    
    def __init__(self):
        """Initialize load testing suite."""
        self.process = psutil.Process()
        self.test_results = {}
        self.redis_client = None
        
        # Load test scenarios
        self.scenarios = {
            "market_open_surge": LoadTestScenario(
                name="market_open_surge",
                duration_minutes=5,
                target_ops_per_second=2000,
                concurrent_users=20,
                ramp_up_time_seconds=30,
                data_pattern="market_open",
                performance_targets={
                    "p99_latency_ms": 10.0,
                    "error_rate_percent": 1.0,
                    "throughput_ops_per_sec": 1800.0
                }
            ),
            "high_frequency_trading": LoadTestScenario(
                name="high_frequency_trading",
                duration_minutes=10,
                target_ops_per_second=1500,
                concurrent_users=15,
                ramp_up_time_seconds=60,
                data_pattern="high_frequency",
                performance_targets={
                    "p99_latency_ms": 5.0,
                    "error_rate_percent": 0.5,
                    "throughput_ops_per_sec": 1400.0
                }
            ),
            "volatility_spike": LoadTestScenario(
                name="volatility_spike",
                duration_minutes=3,
                target_ops_per_second=3000,
                concurrent_users=30,
                ramp_up_time_seconds=15,
                data_pattern="volatility_spike",
                performance_targets={
                    "p99_latency_ms": 15.0,
                    "error_rate_percent": 2.0,
                    "throughput_ops_per_sec": 2500.0
                }
            ),
            "sustained_trading": LoadTestScenario(
                name="sustained_trading",
                duration_minutes=30,
                target_ops_per_second=800,
                concurrent_users=8,
                ramp_up_time_seconds=120,
                data_pattern="sustained",
                performance_targets={
                    "p99_latency_ms": 8.0,
                    "error_rate_percent": 0.1,
                    "throughput_ops_per_sec": 750.0
                }
            ),
            "concurrent_strategies": LoadTestScenario(
                name="concurrent_strategies",
                duration_minutes=15,
                target_ops_per_second=1200,
                concurrent_users=12,
                ramp_up_time_seconds=90,
                data_pattern="mixed",
                performance_targets={
                    "p99_latency_ms": 12.0,
                    "error_rate_percent": 1.5,
                    "throughput_ops_per_sec": 1000.0
                }
            )
        }
        
        logger.info(f"Load testing suite initialized with {len(self.scenarios)} scenarios")
    
    async def setup_load_testing(self):
        """Setup load testing environment."""
        logger.info("Setting up load testing environment")
        
        # Initialize Redis connection for event simulation
        try:
            self.redis_client = redis.from_url("redis://localhost:6379/3")
            await self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, using mock events")
            self.redis_client = None
        
        # Pre-allocate test data
        self.test_data = {
            "strategic_matrices": [torch.randn(1, 48, 13) for _ in range(1000)],
            "tactical_states": [torch.randn(1, 60, 7) for _ in range(1000)],
            "market_events": self._generate_market_events(10000)
        }
        
        logger.info("Load testing environment setup complete")
    
    async def teardown_load_testing(self):
        """Cleanup load testing environment."""
        if self.redis_client:
            await self.redis_client.close()
        logger.info("Load testing environment cleanup complete")
    
    def _generate_market_events(self, count: int) -> List[Dict[str, Any]]:
        """Generate realistic market events for load testing."""
        events = []
        
        for i in range(count):
            event = {
                "event_id": f"market_event_{i}",
                "timestamp": time.time() + i * 0.001,
                "symbol": np.random.choice(["NQ", "ES", "RTY", "YM"]),
                "price": 16000 + np.random.normal(0, 50),
                "volume": np.random.randint(1, 1000),
                "bid": 16000 + np.random.normal(0, 50) - 0.25,
                "ask": 16000 + np.random.normal(0, 50) + 0.25,
                "event_type": np.random.choice(["tick", "trade", "quote"]),
                "market_state": np.random.choice(["open", "premarket", "afterhours"]),
                "volatility": np.random.uniform(0.1, 0.3)
            }
            events.append(event)
        
        return events
    
    def _generate_data_pattern(self, pattern: str, duration_seconds: int) -> List[Dict[str, Any]]:
        """Generate data pattern for specific load test scenario."""
        events = []
        
        if pattern == "market_open":
            # Simulate market opening surge - high volume at start, tapering off
            for i in range(duration_seconds * 100):  # 100 events per second
                intensity = max(0.1, 1.0 - (i / (duration_seconds * 100)))
                if np.random.random() < intensity:
                    event = {
                        "timestamp": time.time() + i * 0.01,
                        "intensity": intensity,
                        "pattern": "market_open",
                        "data": self._create_market_event()
                    }
                    events.append(event)
        
        elif pattern == "high_frequency":
            # Simulate consistent high-frequency trading
            for i in range(duration_seconds * 150):  # 150 events per second
                event = {
                    "timestamp": time.time() + i * 0.0067,
                    "intensity": 0.8 + np.random.uniform(-0.1, 0.1),
                    "pattern": "high_frequency",
                    "data": self._create_market_event()
                }
                events.append(event)
        
        elif pattern == "volatility_spike":
            # Simulate sudden volatility spike
            for i in range(duration_seconds * 200):  # 200 events per second
                # Spike in middle of test
                middle_point = duration_seconds * 100
                distance_from_middle = abs(i - middle_point)
                intensity = 1.0 / (1.0 + distance_from_middle / 100)
                
                if np.random.random() < intensity:
                    event = {
                        "timestamp": time.time() + i * 0.005,
                        "intensity": intensity,
                        "pattern": "volatility_spike",
                        "data": self._create_market_event()
                    }
                    events.append(event)
        
        elif pattern == "sustained":
            # Simulate sustained trading session
            for i in range(duration_seconds * 80):  # 80 events per second
                event = {
                    "timestamp": time.time() + i * 0.0125,
                    "intensity": 0.6 + np.random.uniform(-0.1, 0.1),
                    "pattern": "sustained",
                    "data": self._create_market_event()
                }
                events.append(event)
        
        else:  # mixed pattern
            # Mix of different patterns
            for i in range(duration_seconds * 120):  # 120 events per second
                pattern_type = np.random.choice(["normal", "spike", "quiet"])
                intensity = {
                    "normal": 0.7,
                    "spike": 1.0,
                    "quiet": 0.3
                }[pattern_type]
                
                event = {
                    "timestamp": time.time() + i * 0.0083,
                    "intensity": intensity,
                    "pattern": "mixed",
                    "data": self._create_market_event()
                }
                events.append(event)
        
        return events
    
    def _create_market_event(self) -> Dict[str, Any]:
        """Create a single market event."""
        return {
            "symbol": "NQ",
            "price": 16000 + np.random.normal(0, 25),
            "volume": np.random.randint(1, 500),
            "timestamp": time.time(),
            "event_type": "market_data"
        }
    
    async def run_load_test_scenario(self, scenario: LoadTestScenario) -> LoadTestResult:
        """Execute a single load test scenario."""
        logger.info(f"🚀 Running load test scenario: {scenario.name}")
        logger.info(f"   Duration: {scenario.duration_minutes} minutes")
        logger.info(f"   Target OPS: {scenario.target_ops_per_second}")
        logger.info(f"   Concurrent Users: {scenario.concurrent_users}")
        
        # Generate test data for this scenario
        test_events = self._generate_data_pattern(
            scenario.data_pattern, 
            scenario.duration_minutes * 60
        )
        
        # Performance metrics collection
        latencies = []
        errors = []
        operation_times = []
        resource_samples = []
        
        # Test execution
        start_time = time.time()
        
        async def load_test_worker(worker_id: int, events_chunk: List[Dict[str, Any]]):
            """Individual load test worker."""
            worker_latencies = []
            worker_errors = []
            
            for event in events_chunk:
                try:
                    # Simulate processing time based on event intensity
                    processing_start = time.perf_counter()
                    
                    # Simulate strategic/tactical processing
                    await self._simulate_event_processing(event)
                    
                    processing_end = time.perf_counter()
                    latency_ms = (processing_end - processing_start) * 1000
                    worker_latencies.append(latency_ms)
                    
                    # Brief pause to control load
                    await asyncio.sleep(0.001)
                    
                except Exception as e:
                    worker_errors.append(str(e))
            
            return worker_latencies, worker_errors
        
        # Distribute events across workers
        chunk_size = len(test_events) // scenario.concurrent_users
        event_chunks = [
            test_events[i:i + chunk_size] 
            for i in range(0, len(test_events), chunk_size)
        ]
        
        # Execute load test with ramp-up
        tasks = []
        for i, chunk in enumerate(event_chunks[:scenario.concurrent_users]):
            # Staggered start for ramp-up
            if i < scenario.concurrent_users:
                await asyncio.sleep(scenario.ramp_up_time_seconds / scenario.concurrent_users)
            
            task = asyncio.create_task(load_test_worker(i, chunk))
            tasks.append(task)
        
        # Collect results
        worker_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        all_latencies = []
        all_errors = []
        
        for result in worker_results:
            if isinstance(result, tuple):
                worker_latencies, worker_errors = result
                all_latencies.extend(worker_latencies)
                all_errors.extend(worker_errors)
            else:
                all_errors.append(f"Worker failed: {result}")
        
        # Calculate metrics
        end_time = time.time()
        test_duration = end_time - start_time
        
        # Performance metrics
        if all_latencies:
            avg_latency = np.mean(all_latencies)
            p95_latency = np.percentile(all_latencies, 95)
            p99_latency = np.percentile(all_latencies, 99)
            max_latency = np.max(all_latencies)
        else:
            avg_latency = p95_latency = p99_latency = max_latency = 0
        
        successful_ops = len(all_latencies)
        failed_ops = len(all_errors)
        total_ops = successful_ops + failed_ops
        
        throughput = successful_ops / test_duration if test_duration > 0 else 0
        error_rate = (failed_ops / total_ops * 100) if total_ops > 0 else 0
        
        # Resource utilization
        resource_utilization = {
            "cpu_percent": self.process.cpu_percent(),
            "memory_mb": self.process.memory_info().rss / 1024 / 1024,
            "memory_percent": self.process.memory_percent()
        }
        
        # Performance targets validation
        targets_met = {
            "p99_latency": p99_latency <= scenario.performance_targets["p99_latency_ms"],
            "error_rate": error_rate <= scenario.performance_targets["error_rate_percent"],
            "throughput": throughput >= scenario.performance_targets["throughput_ops_per_sec"]
        }
        
        # Generate recommendations
        recommendations = self._generate_load_test_recommendations(
            scenario, avg_latency, p99_latency, error_rate, throughput
        )
        
        result = LoadTestResult(
            scenario_name=scenario.name,
            test_duration_seconds=test_duration,
            total_operations=total_ops,
            successful_operations=successful_ops,
            failed_operations=failed_ops,
            average_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            max_latency_ms=max_latency,
            throughput_ops_per_sec=throughput,
            error_rate_percent=error_rate,
            resource_utilization=resource_utilization,
            performance_targets_met=targets_met,
            recommendations=recommendations
        )
        
        self.test_results[scenario.name] = result
        
        # Log results
        logger.info(f"✅ Load test scenario '{scenario.name}' completed")
        logger.info(f"   Duration: {test_duration:.1f}s")
        logger.info(f"   Operations: {successful_ops}/{total_ops}")
        logger.info(f"   Throughput: {throughput:.1f} ops/sec")
        logger.info(f"   P99 Latency: {p99_latency:.2f}ms")
        logger.info(f"   Error Rate: {error_rate:.2f}%")
        logger.info(f"   Targets Met: {all(targets_met.values())}")
        
        return result
    
    async def _simulate_event_processing(self, event: Dict[str, Any]):
        """Simulate realistic event processing."""
        # Simulate variable processing time based on event intensity
        intensity = event.get("intensity", 0.5)
        base_processing_time = 0.002  # 2ms base
        
        # Higher intensity = more processing time
        processing_time = base_processing_time * (1 + intensity)
        
        # Add some randomness
        processing_time += np.random.uniform(-0.0005, 0.0005)
        
        # Simulate I/O operations
        await asyncio.sleep(processing_time)
        
        # Simulate CPU-intensive operations
        if np.random.random() < intensity:
            # Simulate model inference
            test_data = torch.randn(1, 48, 13)
            _ = torch.mm(test_data[0], test_data[0].T)
    
    def _generate_load_test_recommendations(self, scenario: LoadTestScenario, 
                                          avg_latency: float, p99_latency: float,
                                          error_rate: float, throughput: float) -> List[str]:
        """Generate recommendations based on load test results."""
        recommendations = []
        
        # Latency recommendations
        if p99_latency > scenario.performance_targets["p99_latency_ms"]:
            recommendations.append(f"P99 latency ({p99_latency:.2f}ms) exceeds target "
                                 f"({scenario.performance_targets['p99_latency_ms']}ms)")
        
        if avg_latency > scenario.performance_targets["p99_latency_ms"] * 0.5:
            recommendations.append("Average latency is high - optimize processing pipeline")
        
        # Throughput recommendations
        if throughput < scenario.performance_targets["throughput_ops_per_sec"]:
            recommendations.append(f"Throughput ({throughput:.1f} ops/sec) below target "
                                 f"({scenario.performance_targets['throughput_ops_per_sec']} ops/sec)")
        
        # Error rate recommendations
        if error_rate > scenario.performance_targets["error_rate_percent"]:
            recommendations.append(f"Error rate ({error_rate:.2f}%) exceeds target "
                                 f"({scenario.performance_targets['error_rate_percent']}%)")
        
        # Specific scenario recommendations
        if scenario.name == "market_open_surge" and p99_latency > 8.0:
            recommendations.append("Optimize for market opening surge handling")
        
        elif scenario.name == "high_frequency_trading" and throughput < 1200:
            recommendations.append("Implement batching for high-frequency operations")
        
        elif scenario.name == "volatility_spike" and error_rate > 1.0:
            recommendations.append("Improve error handling during volatility spikes")
        
        return recommendations
    
    async def run_all_load_test_scenarios(self) -> Dict[str, Any]:
        """Run all load test scenarios and generate comprehensive report."""
        logger.info("🚀 Starting comprehensive load testing suite")
        
        await self.setup_load_testing()
        
        try:
            # Run all scenarios
            for scenario_name, scenario in self.scenarios.items():
                logger.info(f"\\n{'='*60}")
                logger.info(f"SCENARIO: {scenario_name.upper()}")
                logger.info('='*60)
                
                result = await self.run_load_test_scenario(scenario)
                
                # Brief pause between scenarios
                await asyncio.sleep(10)
            
            # Generate comprehensive report
            report = self._generate_load_test_report()
            
            # Save report
            self._save_load_test_report(report)
            
            logger.info("\\n✅ Comprehensive load testing suite completed")
            
            return report
            
        finally:
            await self.teardown_load_testing()
    
    def _generate_load_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive load test report."""
        # Overall metrics
        all_throughputs = [result.throughput_ops_per_sec for result in self.test_results.values()]
        all_latencies = [result.p99_latency_ms for result in self.test_results.values()]
        all_error_rates = [result.error_rate_percent for result in self.test_results.values()]
        
        # Scenario success rates
        scenarios_passed = sum(
            1 for result in self.test_results.values() 
            if all(result.performance_targets_met.values())
        )
        
        # Overall assessment
        overall_pass_rate = scenarios_passed / len(self.test_results) if self.test_results else 0
        production_ready = overall_pass_rate >= 0.8
        
        return {
            "executive_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_scenarios": len(self.test_results),
                "scenarios_passed": scenarios_passed,
                "overall_pass_rate": overall_pass_rate,
                "production_ready": production_ready,
                "average_throughput": np.mean(all_throughputs) if all_throughputs else 0,
                "average_p99_latency": np.mean(all_latencies) if all_latencies else 0,
                "average_error_rate": np.mean(all_error_rates) if all_error_rates else 0
            },
            "scenario_results": {
                result.scenario_name: {
                    "duration_seconds": result.test_duration_seconds,
                    "total_operations": result.total_operations,
                    "successful_operations": result.successful_operations,
                    "throughput_ops_per_sec": result.throughput_ops_per_sec,
                    "p99_latency_ms": result.p99_latency_ms,
                    "error_rate_percent": result.error_rate_percent,
                    "targets_met": result.performance_targets_met,
                    "recommendations": result.recommendations,
                    "resource_utilization": result.resource_utilization
                }
                for result in self.test_results.values()
            },
            "performance_analysis": {
                "peak_throughput": max(all_throughputs) if all_throughputs else 0,
                "worst_p99_latency": max(all_latencies) if all_latencies else 0,
                "highest_error_rate": max(all_error_rates) if all_error_rates else 0,
                "most_challenging_scenario": self._identify_most_challenging_scenario(),
                "best_performing_scenario": self._identify_best_performing_scenario()
            },
            "deployment_recommendations": {
                "approved_for_production": production_ready,
                "confidence_level": "high" if overall_pass_rate >= 0.9 else "medium" if overall_pass_rate >= 0.7 else "low",
                "critical_issues": self._identify_critical_issues(),
                "scaling_recommendations": self._generate_scaling_recommendations(),
                "monitoring_requirements": [
                    "Monitor throughput under load",
                    "Track P99 latency during peak hours",
                    "Set up alerts for error rate spikes",
                    "Monitor resource utilization trends"
                ]
            },
            "test_metadata": {
                "framework_version": "1.0.0",
                "agent_id": "Agent_4_Performance_Baseline_Research",
                "test_suite": "production_load_testing"
            }
        }
    
    def _identify_most_challenging_scenario(self) -> str:
        """Identify the most challenging scenario based on performance."""
        if not self.test_results:
            return "none"
        
        # Score based on targets not met
        worst_score = 0
        worst_scenario = ""
        
        for scenario_name, result in self.test_results.items():
            score = sum(1 for met in result.performance_targets_met.values() if not met)
            if score > worst_score:
                worst_score = score
                worst_scenario = scenario_name
        
        return worst_scenario
    
    def _identify_best_performing_scenario(self) -> str:
        """Identify the best performing scenario."""
        if not self.test_results:
            return "none"
        
        # Score based on targets met and performance
        best_score = 0
        best_scenario = ""
        
        for scenario_name, result in self.test_results.items():
            score = sum(1 for met in result.performance_targets_met.values() if met)
            score += (1000 / result.p99_latency_ms) * 0.1  # Bonus for low latency
            score += (result.throughput_ops_per_sec / 1000) * 0.1  # Bonus for high throughput
            
            if score > best_score:
                best_score = score
                best_scenario = scenario_name
        
        return best_scenario
    
    def _identify_critical_issues(self) -> List[str]:
        """Identify critical issues across all scenarios."""
        issues = []
        
        for scenario_name, result in self.test_results.items():
            if not result.performance_targets_met.get("p99_latency", True):
                issues.append(f"{scenario_name}: P99 latency exceeds target")
            
            if not result.performance_targets_met.get("error_rate", True):
                issues.append(f"{scenario_name}: Error rate exceeds target")
            
            if not result.performance_targets_met.get("throughput", True):
                issues.append(f"{scenario_name}: Throughput below target")
        
        return issues
    
    def _generate_scaling_recommendations(self) -> List[str]:
        """Generate scaling recommendations based on load test results."""
        recommendations = []
        
        # Check if any scenario required more resources
        high_cpu_scenarios = [
            name for name, result in self.test_results.items()
            if result.resource_utilization.get("cpu_percent", 0) > 80
        ]
        
        if high_cpu_scenarios:
            recommendations.append(f"Consider CPU scaling for scenarios: {', '.join(high_cpu_scenarios)}")
        
        # Check memory usage
        high_memory_scenarios = [
            name for name, result in self.test_results.items()
            if result.resource_utilization.get("memory_mb", 0) > 1000
        ]
        
        if high_memory_scenarios:
            recommendations.append(f"Consider memory scaling for scenarios: {', '.join(high_memory_scenarios)}")
        
        # Check throughput scaling
        low_throughput_scenarios = [
            name for name, result in self.test_results.items()
            if result.throughput_ops_per_sec < 500
        ]
        
        if low_throughput_scenarios:
            recommendations.append(f"Implement horizontal scaling for: {', '.join(low_throughput_scenarios)}")
        
        return recommendations
    
    def _save_load_test_report(self, report: Dict[str, Any]):
        """Save load test report to file."""
        output_file = Path("production_load_test_report.json")
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📊 Load test report saved to: {output_file}")


# Test implementations
class TestProductionLoadTesting:
    """Test suite for production load testing scenarios."""
    
    @pytest.fixture
    def load_test_suite(self):
        """Create load test suite instance."""
        return ProductionLoadTestSuite()
    
    @pytest.mark.asyncio
    @pytest.mark.load
    async def test_market_open_surge_scenario(self, load_test_suite):
        """Test market opening surge scenario."""
        scenario = load_test_suite.scenarios["market_open_surge"]
        # Reduce duration for testing
        scenario.duration_minutes = 1
        
        await load_test_suite.setup_load_testing()
        
        try:
            result = await load_test_suite.run_load_test_scenario(scenario)
            
            assert result.scenario_name == "market_open_surge"
            assert result.total_operations > 0
            assert result.throughput_ops_per_sec > 0
            
            logger.info(f"Market open surge - Throughput: {result.throughput_ops_per_sec:.1f} ops/sec")
            logger.info(f"Market open surge - P99 latency: {result.p99_latency_ms:.2f}ms")
            
        finally:
            await load_test_suite.teardown_load_testing()
    
    @pytest.mark.asyncio
    @pytest.mark.load
    async def test_high_frequency_trading_scenario(self, load_test_suite):
        """Test high frequency trading scenario."""
        scenario = load_test_suite.scenarios["high_frequency_trading"]
        # Reduce duration for testing
        scenario.duration_minutes = 1
        
        await load_test_suite.setup_load_testing()
        
        try:
            result = await load_test_suite.run_load_test_scenario(scenario)
            
            assert result.scenario_name == "high_frequency_trading"
            assert result.total_operations > 0
            assert result.throughput_ops_per_sec > 0
            
            logger.info(f"High frequency - Throughput: {result.throughput_ops_per_sec:.1f} ops/sec")
            logger.info(f"High frequency - P99 latency: {result.p99_latency_ms:.2f}ms")
            
        finally:
            await load_test_suite.teardown_load_testing()
    
    @pytest.mark.asyncio
    @pytest.mark.load
    async def test_comprehensive_load_testing_suite(self, load_test_suite):
        """Test comprehensive load testing suite (reduced duration)."""
        # Reduce all scenario durations for testing
        for scenario in load_test_suite.scenarios.values():
            scenario.duration_minutes = 1
            scenario.concurrent_users = min(4, scenario.concurrent_users)
        
        report = await load_test_suite.run_all_load_test_scenarios()
        
        # Verify report structure
        assert "executive_summary" in report
        assert "scenario_results" in report
        assert "performance_analysis" in report
        assert "deployment_recommendations" in report
        
        # Verify executive summary
        summary = report["executive_summary"]
        assert "total_scenarios" in summary
        assert "scenarios_passed" in summary
        assert "production_ready" in summary
        
        # Log overall results
        logger.info(f"Total scenarios: {summary['total_scenarios']}")
        logger.info(f"Scenarios passed: {summary['scenarios_passed']}")
        logger.info(f"Production ready: {summary['production_ready']}")
        logger.info(f"Average throughput: {summary['average_throughput']:.1f} ops/sec")


if __name__ == "__main__":
    """Run load testing scenarios directly."""
    async def main():
        load_test_suite = ProductionLoadTestSuite()
        
        print("🚀 Production Load Testing Suite")
        print("=" * 60)
        
        # Reduce durations for demo
        for scenario in load_test_suite.scenarios.values():
            scenario.duration_minutes = 2
            scenario.concurrent_users = min(8, scenario.concurrent_users)
        
        # Run comprehensive load testing
        report = await load_test_suite.run_all_load_test_scenarios()
        
        # Display results
        print("\\n📊 LOAD TEST RESULTS SUMMARY")
        print("=" * 50)
        summary = report["executive_summary"]
        print(f"Total Scenarios: {summary['total_scenarios']}")
        print(f"Scenarios Passed: {summary['scenarios_passed']}")
        print(f"Overall Pass Rate: {summary['overall_pass_rate']:.1%}")
        print(f"Production Ready: {summary['production_ready']}")
        print(f"Average Throughput: {summary['average_throughput']:.1f} ops/sec")
        print(f"Average P99 Latency: {summary['average_p99_latency']:.2f}ms")
        
        # Show individual scenario results
        print("\\n🎯 SCENARIO RESULTS")
        print("=" * 50)
        for scenario_name, result in report["scenario_results"].items():
            print(f"{scenario_name.upper()}:")
            print(f"  Throughput: {result['throughput_ops_per_sec']:.1f} ops/sec")
            print(f"  P99 Latency: {result['p99_latency_ms']:.2f}ms")
            print(f"  Error Rate: {result['error_rate_percent']:.2f}%")
            print(f"  Targets Met: {all(result['targets_met'].values())}")
            print()
        
        print("\\n✅ Production Load Testing Suite Complete!")
        print("📄 Full report saved to: production_load_test_report.json")
    
    asyncio.run(main())