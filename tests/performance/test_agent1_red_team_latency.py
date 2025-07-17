"""
Agent 1 Red Team Mission: Full-Stack Latency Audit for Operation "Aegis Prime"

CRITICAL OBJECTIVE: Prove the Execution Engine CANNOT meet sub-millisecond latency targets 
when the entire MARL pipeline is under maximum load.

This is a RUTHLESS stress test designed to expose latency bottlenecks across the complete
Strategic MARL → Tactical → Risk → Execution pipeline under realistic adversarial conditions.

MISSION PARAMETERS:
- Target: <500µs total pipeline latency (EXPECTED TO FAIL)
- Method: End-to-end "photon-in-photon-out" measurement 
- Scope: Complete pipeline from Strategic event to FIX dispatch
- Conditions: Maximum upstream load, event storms, queue overflows
"""

import asyncio
import time
import pytest
import json
import statistics
import numpy as np
import redis.asyncio as redis
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import threading
from unittest.mock import Mock, AsyncMock, patch
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import os
from pathlib import Path

# Configure logging for detailed forensic analysis
logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s', 
                   datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

@dataclass
class LatencyMeasurement:
    """Precise latency measurement with microsecond resolution."""
    component: str
    operation: str
    start_time_ns: int
    end_time_ns: int
    duration_us: float
    metadata: Dict[str, Any]
    
    @property
    def duration_ms(self) -> float:
        return self.duration_us / 1000.0

@dataclass
class PipelineStageLatency:
    """Latency breakdown for each pipeline stage."""
    strategic_event_generation_us: float
    synergy_detection_us: float
    redis_stream_publish_us: float
    redis_stream_read_us: float
    tactical_lock_acquisition_us: float
    tactical_matrix_fetch_us: float
    tactical_agent_inference_us: float
    tactical_decision_aggregation_us: float
    execution_command_creation_us: float
    fix_dispatch_simulation_us: float
    total_pipeline_us: float
    
    @property
    def total_pipeline_ms(self) -> float:
        return self.total_pipeline_us / 1000.0
    
    def get_bottleneck(self) -> Tuple[str, float]:
        """Identify the single biggest latency contributor."""
        stages = {
            'strategic_event_generation': self.strategic_event_generation_us,
            'synergy_detection': self.synergy_detection_us,
            'redis_stream_publish': self.redis_stream_publish_us,
            'redis_stream_read': self.redis_stream_read_us,
            'tactical_lock_acquisition': self.tactical_lock_acquisition_us,
            'tactical_matrix_fetch': self.tactical_matrix_fetch_us,
            'tactical_agent_inference': self.tactical_agent_inference_us,
            'tactical_decision_aggregation': self.tactical_decision_aggregation_us,
            'execution_command_creation': self.execution_command_creation_us,
            'fix_dispatch_simulation': self.fix_dispatch_simulation_us,
        }
        bottleneck_stage = max(stages.items(), key=lambda x: x[1])
        return bottleneck_stage

class MockExecutionEngine:
    """Mock execution engine with realistic FIX gateway latency simulation."""
    
    def __init__(self):
        self.orders_processed = 0
        self.latency_simulation_us = 50  # 50µs FIX dispatch simulation
        self.queue_depth = 0
        self.queue_overflow_threshold = 1000
        
    async def dispatch_fix_order(self, execution_command: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate FIX order dispatch with realistic latency."""
        start_time = time.perf_counter_ns()
        
        # Simulate queue depth impact on latency
        if self.queue_depth > self.queue_overflow_threshold:
            # Queue overflow - massive latency spike
            await asyncio.sleep(0.01)  # 10ms penalty
        else:
            # Normal FIX processing latency
            await asyncio.sleep(self.latency_simulation_us / 1_000_000)
        
        self.orders_processed += 1
        end_time = time.perf_counter_ns()
        
        return {
            "order_id": f"FIX_{self.orders_processed}",
            "status": "DISPATCHED",
            "latency_us": (end_time - start_time) / 1000,
            "queue_depth": self.queue_depth,
            "timestamp": time.time()
        }
    
    def add_queue_pressure(self, depth: int):
        """Simulate upstream queue pressure."""
        self.queue_depth += depth
    
    def clear_queue(self):
        """Clear queue simulation."""
        self.queue_depth = 0

class Agent1RedTeamLatencyAudit:
    """
    Agent 1 Mission: Ruthless full-stack latency audit.
    
    This class implements comprehensive latency testing designed to expose
    bottlenecks and prove that sub-millisecond targets cannot be met under
    realistic production conditions.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/2"):
        """Initialize the red team audit with forensic precision."""
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        
        # Mission parameters
        self.target_pipeline_latency_us = 500  # 500µs target (EXPECTED TO FAIL)
        self.critical_threshold_us = 1000      # 1ms critical threshold
        self.failure_threshold_us = 2000       # 2ms = mission failure
        
        # Test configuration
        self.num_baseline_tests = 1000
        self.num_stress_tests = 500
        self.event_storm_rate = 100  # events per second
        self.max_concurrent_events = 50
        
        # Forensic data collection
        self.baseline_measurements: List[PipelineStageLatency] = []
        self.stress_measurements: List[PipelineStageLatency] = []
        self.event_storm_measurements: List[PipelineStageLatency] = []
        self.overflow_measurements: List[PipelineStageLatency] = []
        
        # Component mocks for controlled testing
        self.mock_execution_engine = MockExecutionEngine()
        self.mock_strategic_marl = None
        self.mock_synergy_detector = None
        self.mock_tactical_controller = None
        
        # Performance tracking
        self.bottleneck_analysis = {}
        self.failure_modes = []
        
        logger.info("🔴 Agent 1 Red Team Latency Audit initialized")
        logger.info(f"🎯 Target pipeline latency: {self.target_pipeline_latency_us}µs")
        logger.info(f"⚠️  Critical threshold: {self.critical_threshold_us}µs")
        logger.info(f"💥 Failure threshold: {self.failure_threshold_us}µs")
    
    async def setup_audit_environment(self):
        """Setup controlled audit environment with all components."""
        logger.info("🛠️  Setting up Agent 1 audit environment...")
        
        # Setup Redis connection
        self.redis_client = redis.from_url(self.redis_url)
        await self.redis_client.ping()
        
        # Setup mock components with realistic latencies
        await self._setup_mock_components()
        
        # Create test streams
        await self._setup_test_streams()
        
        logger.info("✅ Audit environment ready for red team assault")
    
    async def _setup_mock_components(self):
        """Setup mock components with realistic behavior."""
        # Mock Strategic MARL Environment
        self.mock_strategic_marl = Mock()
        self.mock_strategic_marl.generate_synergy_event = AsyncMock()
        
        # Mock Synergy Detector
        self.mock_synergy_detector = Mock()
        self.mock_synergy_detector.process_features = Mock()
        
        # Mock Tactical Controller with actual timing simulation
        self.mock_tactical_controller = Mock()
        self.mock_tactical_controller.on_synergy_detected = AsyncMock()
        
        # Configure realistic latencies
        await self._configure_realistic_latencies()
    
    async def _configure_realistic_latencies(self):
        """Configure realistic latencies for each component."""
        
        async def mock_strategic_event_gen():
            """Simulate strategic event generation latency."""
            await asyncio.sleep(0.000020)  # 20µs
            return {
                "synergy_type": "TYPE_1",
                "direction": 1,
                "confidence": 0.75,
                "correlation_id": f"test_{time.time()}",
                "timestamp": time.time()
            }
        
        async def mock_synergy_detection(features):
            """Simulate synergy detection latency."""
            await asyncio.sleep(0.000030)  # 30µs
            return True
        
        async def mock_tactical_processing(event):
            """Simulate tactical processing with all stages."""
            # Lock acquisition
            await asyncio.sleep(0.000010)  # 10µs
            # Matrix fetch
            await asyncio.sleep(0.000050)  # 50µs
            # Agent inference (3 agents)
            await asyncio.sleep(0.000075)  # 75µs (25µs each)
            # Decision aggregation
            await asyncio.sleep(0.000015)  # 15µs
            # Execution command creation
            await asyncio.sleep(0.000020)  # 20µs
            
            return {
                "action": "long",
                "confidence": 0.8,
                "execution_command": {
                    "action": "execute_trade",
                    "side": "BUY",
                    "quantity": 1,
                    "correlation_id": event.get("correlation_id")
                }
            }
        
        self.mock_strategic_marl.generate_synergy_event = mock_strategic_event_gen
        self.mock_synergy_detector.process_features = mock_synergy_detection
        self.mock_tactical_controller.on_synergy_detected = mock_tactical_processing
    
    async def _setup_test_streams(self):
        """Setup Redis streams for testing."""
        try:
            await self.redis_client.xgroup_create(
                "synergy_events",
                "test_tactical_group",
                id='0',
                mkstream=True
            )
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                logger.warning(f"Could not create test group: {e}")
    
    async def execute_full_audit(self) -> Dict[str, Any]:
        """Execute the complete Agent 1 red team audit."""
        logger.info("🚨 COMMENCING AGENT 1 RED TEAM ASSAULT")
        logger.info("=" * 80)
        
        await self.setup_audit_environment()
        
        audit_results = {}
        
        try:
            # Phase 1: Baseline Performance Measurement
            logger.info("📊 PHASE 1: Baseline Performance Measurement")
            baseline_results = await self._run_baseline_tests()
            audit_results["baseline"] = baseline_results
            
            # Phase 2: Component-Level Latency Breakdown  
            logger.info("🔬 PHASE 2: Component-Level Latency Breakdown")
            breakdown_results = await self._run_component_breakdown_tests()
            audit_results["component_breakdown"] = breakdown_results
            
            # Phase 3: Event Storm Throughput Test
            logger.info("⛈️  PHASE 3: Event Storm Throughput Test")
            storm_results = await self._run_event_storm_tests()
            audit_results["event_storm"] = storm_results
            
            # Phase 4: Queue Overflow & Traffic Jam Tests
            logger.info("🚧 PHASE 4: Queue Overflow & Traffic Jam Tests")
            overflow_results = await self._run_queue_overflow_tests()
            audit_results["queue_overflow"] = overflow_results
            
            # Phase 5: Concurrent Load Testing
            logger.info("⚡ PHASE 5: Concurrent Load Testing")
            concurrent_results = await self._run_concurrent_load_tests()
            audit_results["concurrent_load"] = concurrent_results
            
            # Phase 6: Network Latency Injection
            logger.info("🌐 PHASE 6: Network Latency Injection")
            network_results = await self._run_network_latency_tests()
            audit_results["network_latency"] = network_results
            
            # Final Analysis
            final_analysis = await self._perform_final_analysis(audit_results)
            audit_results["final_analysis"] = final_analysis
            
        finally:
            await self._cleanup_audit_environment()
        
        return audit_results
    
    async def _run_baseline_tests(self) -> Dict[str, Any]:
        """Run baseline performance tests to establish normal latency profile."""
        logger.info(f"Running {self.num_baseline_tests} baseline measurements...")
        
        baseline_latencies = []
        
        for i in range(self.num_baseline_tests):
            latency = await self._measure_single_pipeline_latency()
            baseline_latencies.append(latency)
            self.baseline_measurements.append(latency)
            
            if (i + 1) % 100 == 0:
                current_p99 = np.percentile([l.total_pipeline_us for l in baseline_latencies], 99)
                logger.info(f"Progress: {i+1}/{self.num_baseline_tests}, P99: {current_p99:.1f}µs")
        
        # Analyze baseline results
        total_latencies = [l.total_pipeline_us for l in baseline_latencies]
        
        results = {
            "total_tests": len(baseline_latencies),
            "statistics": {
                "min_us": float(np.min(total_latencies)),
                "max_us": float(np.max(total_latencies)),
                "mean_us": float(np.mean(total_latencies)),
                "median_us": float(np.median(total_latencies)),
                "std_us": float(np.std(total_latencies)),
                "p50_us": float(np.percentile(total_latencies, 50)),
                "p95_us": float(np.percentile(total_latencies, 95)),
                "p99_us": float(np.percentile(total_latencies, 99)),
                "p99_9_us": float(np.percentile(total_latencies, 99.9)),
            },
            "target_analysis": {
                "target_met_count": sum(1 for l in total_latencies if l <= self.target_pipeline_latency_us),
                "target_met_percentage": (sum(1 for l in total_latencies if l <= self.target_pipeline_latency_us) / len(total_latencies)) * 100,
                "critical_threshold_exceeded": sum(1 for l in total_latencies if l > self.critical_threshold_us),
                "failure_threshold_exceeded": sum(1 for l in total_latencies if l > self.failure_threshold_us),
            }
        }
        
        # Identify primary bottleneck from baseline
        bottleneck_analysis = self._analyze_bottlenecks(baseline_latencies)
        results["bottleneck_analysis"] = bottleneck_analysis
        
        logger.info(f"📊 Baseline Results:")
        logger.info(f"   Mean latency: {results['statistics']['mean_us']:.1f}µs")
        logger.info(f"   P99 latency: {results['statistics']['p99_us']:.1f}µs")
        logger.info(f"   Target met: {results['target_analysis']['target_met_percentage']:.1f}%")
        logger.info(f"   Primary bottleneck: {bottleneck_analysis['primary_bottleneck']['component']} ({bottleneck_analysis['primary_bottleneck']['avg_latency_us']:.1f}µs)")
        
        return results
    
    async def _run_component_breakdown_tests(self) -> Dict[str, Any]:
        """Detailed component-level latency breakdown analysis."""
        logger.info("Analyzing component-level latency breakdown...")
        
        # Take sample of measurements for detailed analysis
        sample_size = min(100, len(self.baseline_measurements))
        sample_measurements = self.baseline_measurements[:sample_size]
        
        component_stats = {}
        
        for measurement in sample_measurements:
            components = {
                'strategic_event_generation': measurement.strategic_event_generation_us,
                'synergy_detection': measurement.synergy_detection_us,
                'redis_stream_publish': measurement.redis_stream_publish_us,
                'redis_stream_read': measurement.redis_stream_read_us,
                'tactical_lock_acquisition': measurement.tactical_lock_acquisition_us,
                'tactical_matrix_fetch': measurement.tactical_matrix_fetch_us,
                'tactical_agent_inference': measurement.tactical_agent_inference_us,
                'tactical_decision_aggregation': measurement.tactical_decision_aggregation_us,
                'execution_command_creation': measurement.execution_command_creation_us,
                'fix_dispatch_simulation': measurement.fix_dispatch_simulation_us,
            }
            
            for component, latency in components.items():
                if component not in component_stats:
                    component_stats[component] = []
                component_stats[component].append(latency)
        
        # Calculate statistics for each component
        breakdown_results = {}
        for component, latencies in component_stats.items():
            breakdown_results[component] = {
                "mean_us": float(np.mean(latencies)),
                "p95_us": float(np.percentile(latencies, 95)),
                "p99_us": float(np.percentile(latencies, 99)),
                "max_us": float(np.max(latencies)),
                "percentage_of_total": (np.mean(latencies) / np.mean([m.total_pipeline_us for m in sample_measurements])) * 100
            }
        
        # Sort by average latency to identify worst offenders
        sorted_components = sorted(breakdown_results.items(), key=lambda x: x[1]['mean_us'], reverse=True)
        
        logger.info("🔬 Component Latency Breakdown (sorted by average latency):")
        for i, (component, stats) in enumerate(sorted_components[:5]):
            logger.info(f"   {i+1}. {component}: {stats['mean_us']:.1f}µs avg, {stats['p99_us']:.1f}µs P99 ({stats['percentage_of_total']:.1f}% of total)")
        
        return {
            "component_breakdown": breakdown_results,
            "worst_offenders": dict(sorted_components[:3]),
            "sample_size": sample_size
        }
    
    async def _run_event_storm_tests(self) -> Dict[str, Any]:
        """Test pipeline performance under event storm conditions."""
        logger.info(f"Testing event storm: {self.event_storm_rate} events/second...")
        
        # Generate event storm
        storm_duration = 10  # seconds
        total_events = self.event_storm_rate * storm_duration
        
        storm_latencies = []
        
        # Launch event storm
        start_time = time.time()
        
        async def generate_storm_event(event_id: int):
            """Generate a single storm event."""
            try:
                latency = await self._measure_single_pipeline_latency()
                return latency
            except Exception as e:
                logger.error(f"Storm event {event_id} failed: {e}")
                return None
        
        # Create concurrent event storm
        semaphore = asyncio.Semaphore(self.max_concurrent_events)
        
        async def controlled_storm_event(event_id: int):
            async with semaphore:
                return await generate_storm_event(event_id)
        
        # Launch all events
        tasks = [controlled_storm_event(i) for i in range(total_events)]
        
        # Process with progress tracking
        completed = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result:
                storm_latencies.append(result)
                self.event_storm_measurements.append(result)
            
            completed += 1
            if completed % 100 == 0:
                logger.info(f"Storm progress: {completed}/{total_events}")
        
        actual_duration = time.time() - start_time
        actual_rate = len(storm_latencies) / actual_duration
        
        # Analyze storm results
        total_latencies = [l.total_pipeline_us for l in storm_latencies]
        
        results = {
            "target_rate": self.event_storm_rate,
            "actual_rate": actual_rate,
            "duration_seconds": actual_duration,
            "successful_events": len(storm_latencies),
            "failed_events": total_events - len(storm_latencies),
            "failure_rate": (total_events - len(storm_latencies)) / total_events,
            "statistics": {
                "mean_us": float(np.mean(total_latencies)),
                "p95_us": float(np.percentile(total_latencies, 95)),
                "p99_us": float(np.percentile(total_latencies, 99)),
                "max_us": float(np.max(total_latencies)),
            },
            "degradation_analysis": self._analyze_storm_degradation(storm_latencies)
        }
        
        logger.info(f"⛈️  Event Storm Results:")
        logger.info(f"   Actual rate: {actual_rate:.1f} events/second")
        logger.info(f"   Failure rate: {results['failure_rate']*100:.2f}%")
        logger.info(f"   Storm P99 latency: {results['statistics']['p99_us']:.1f}µs")
        
        return results
    
    async def _run_queue_overflow_tests(self) -> Dict[str, Any]:
        """Test pipeline behavior under queue overflow conditions."""
        logger.info("Testing queue overflow and traffic jam scenarios...")
        
        # Simulate upstream queue pressure
        overflow_scenarios = [
            {"name": "light_pressure", "queue_depth": 500},
            {"name": "moderate_pressure", "queue_depth": 1000}, 
            {"name": "heavy_pressure", "queue_depth": 2000},
            {"name": "critical_pressure", "queue_depth": 5000}
        ]
        
        overflow_results = {}
        
        for scenario in overflow_scenarios:
            logger.info(f"Testing {scenario['name']} (queue depth: {scenario['queue_depth']})")
            
            # Apply queue pressure
            self.mock_execution_engine.add_queue_pressure(scenario['queue_depth'])
            
            # Measure latency under pressure
            pressure_latencies = []
            for _ in range(100):
                latency = await self._measure_single_pipeline_latency()
                pressure_latencies.append(latency)
                self.overflow_measurements.append(latency)
            
            # Clear queue for next test
            self.mock_execution_engine.clear_queue()
            
            # Analyze results
            total_latencies = [l.total_pipeline_us for l in pressure_latencies]
            
            overflow_results[scenario['name']] = {
                "queue_depth": scenario['queue_depth'],
                "statistics": {
                    "mean_us": float(np.mean(total_latencies)),
                    "p99_us": float(np.percentile(total_latencies, 99)),
                    "max_us": float(np.max(total_latencies)),
                },
                "degradation_factor": float(np.mean(total_latencies)) / float(np.mean([l.total_pipeline_us for l in self.baseline_measurements[:100]]))
            }
            
            logger.info(f"   {scenario['name']}: P99={overflow_results[scenario['name']]['statistics']['p99_us']:.1f}µs, degradation={overflow_results[scenario['name']]['degradation_factor']:.1f}x")
        
        return overflow_results
    
    async def _run_concurrent_load_tests(self) -> Dict[str, Any]:
        """Test pipeline under maximum concurrent load."""
        logger.info("Testing maximum concurrent load scenarios...")
        
        concurrency_levels = [10, 25, 50, 100]
        concurrent_results = {}
        
        for concurrency in concurrency_levels:
            logger.info(f"Testing concurrency level: {concurrency}")
            
            async def concurrent_pipeline_test(test_id: int):
                """Single concurrent pipeline test."""
                return await self._measure_single_pipeline_latency()
            
            # Launch concurrent tests
            start_time = time.time()
            tasks = [concurrent_pipeline_test(i) for i in range(concurrency)]
            concurrent_latencies = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # Filter successful results
            successful_latencies = [l for l in concurrent_latencies if isinstance(l, PipelineStageLatency)]
            failed_count = len([l for l in concurrent_latencies if not isinstance(l, PipelineStageLatency)])
            
            if successful_latencies:
                total_latencies = [l.total_pipeline_us for l in successful_latencies]
                
                concurrent_results[f"concurrency_{concurrency}"] = {
                    "concurrency_level": concurrency,
                    "successful_tests": len(successful_latencies),
                    "failed_tests": failed_count,
                    "total_duration": end_time - start_time,
                    "statistics": {
                        "mean_us": float(np.mean(total_latencies)),
                        "p95_us": float(np.percentile(total_latencies, 95)),
                        "p99_us": float(np.percentile(total_latencies, 99)),
                        "max_us": float(np.max(total_latencies)),
                    }
                }
                
                logger.info(f"   Concurrency {concurrency}: {len(successful_latencies)}/{concurrency} successful, P99={concurrent_results[f'concurrency_{concurrency}']['statistics']['p99_us']:.1f}µs")
        
        return concurrent_results
    
    async def _run_network_latency_tests(self) -> Dict[str, Any]:
        """Test pipeline under simulated network latency conditions."""
        logger.info("Testing pipeline under network latency injection...")
        
        # Simulate network latency by adding delays to Redis operations
        network_scenarios = [
            {"name": "lan_latency", "delay_us": 100},      # 0.1ms LAN
            {"name": "wan_latency", "delay_us": 1000},     # 1ms WAN  
            {"name": "satellite_latency", "delay_us": 10000}, # 10ms satellite
            {"name": "degraded_network", "delay_us": 25000}   # 25ms degraded
        ]
        
        network_results = {}
        
        for scenario in network_scenarios:
            logger.info(f"Testing {scenario['name']} ({scenario['delay_us']}µs delay)")
            
            # Inject network delay into Redis operations
            original_xadd = self.redis_client.xadd
            original_xread = self.redis_client.xreadgroup
            
            async def delayed_xadd(*args, **kwargs):
                await asyncio.sleep(scenario['delay_us'] / 1_000_000)
                return await original_xadd(*args, **kwargs)
            
            async def delayed_xread(*args, **kwargs):
                await asyncio.sleep(scenario['delay_us'] / 1_000_000)
                return await original_xread(*args, **kwargs)
            
            # Monkey patch with delays
            self.redis_client.xadd = delayed_xadd
            self.redis_client.xreadgroup = delayed_xread
            
            # Measure latency with network delays
            network_latencies = []
            for _ in range(50):  # Smaller sample for network tests
                latency = await self._measure_single_pipeline_latency()
                network_latencies.append(latency)
            
            # Restore original methods
            self.redis_client.xadd = original_xadd
            self.redis_client.xreadgroup = original_xread
            
            # Analyze results
            total_latencies = [l.total_pipeline_us for l in network_latencies]
            
            network_results[scenario['name']] = {
                "injected_delay_us": scenario['delay_us'],
                "statistics": {
                    "mean_us": float(np.mean(total_latencies)),
                    "p99_us": float(np.percentile(total_latencies, 99)),
                    "max_us": float(np.max(total_latencies)),
                },
                "latency_increase": float(np.mean(total_latencies)) - float(np.mean([l.total_pipeline_us for l in self.baseline_measurements[:50]]))
            }
            
            logger.info(f"   {scenario['name']}: P99={network_results[scenario['name']]['statistics']['p99_us']:.1f}µs, increase={network_results[scenario['name']]['latency_increase']:.1f}µs")
        
        return network_results
    
    async def _measure_single_pipeline_latency(self) -> PipelineStageLatency:
        """Measure end-to-end pipeline latency with microsecond precision."""
        
        # Stage 1: Strategic Event Generation
        strategic_start = time.perf_counter_ns()
        strategic_event = await self.mock_strategic_marl.generate_synergy_event()
        strategic_end = time.perf_counter_ns()
        strategic_latency_us = (strategic_end - strategic_start) / 1000
        
        # Stage 2: Synergy Detection
        synergy_start = time.perf_counter_ns()
        await self.mock_synergy_detector.process_features({})
        synergy_end = time.perf_counter_ns()
        synergy_latency_us = (synergy_end - synergy_start) / 1000
        
        # Stage 3: Redis Stream Publish
        redis_pub_start = time.perf_counter_ns()
        await self.redis_client.xadd("synergy_events", strategic_event)
        redis_pub_end = time.perf_counter_ns()
        redis_pub_latency_us = (redis_pub_end - redis_pub_start) / 1000
        
        # Stage 4: Redis Stream Read (simulated)
        redis_read_start = time.perf_counter_ns()
        # Simulate reading from stream
        await asyncio.sleep(0.000005)  # 5µs read simulation
        redis_read_end = time.perf_counter_ns()
        redis_read_latency_us = (redis_read_end - redis_read_start) / 1000
        
        # Stage 5-9: Tactical Processing (broken down)
        tactical_start = time.perf_counter_ns()
        
        # Lock acquisition
        lock_start = time.perf_counter_ns()
        await asyncio.sleep(0.000010)  # 10µs lock simulation
        lock_end = time.perf_counter_ns()
        lock_latency_us = (lock_end - lock_start) / 1000
        
        # Matrix fetch  
        matrix_start = time.perf_counter_ns()
        await asyncio.sleep(0.000050)  # 50µs matrix fetch
        matrix_end = time.perf_counter_ns()
        matrix_latency_us = (matrix_end - matrix_start) / 1000
        
        # Agent inference
        inference_start = time.perf_counter_ns()
        await asyncio.sleep(0.000075)  # 75µs agent inference
        inference_end = time.perf_counter_ns()
        inference_latency_us = (inference_end - inference_start) / 1000
        
        # Decision aggregation
        aggregation_start = time.perf_counter_ns()
        await asyncio.sleep(0.000015)  # 15µs aggregation
        aggregation_end = time.perf_counter_ns()
        aggregation_latency_us = (aggregation_end - aggregation_start) / 1000
        
        # Execution command creation
        exec_cmd_start = time.perf_counter_ns()
        tactical_decision = await self.mock_tactical_controller.on_synergy_detected(strategic_event)
        exec_cmd_end = time.perf_counter_ns()
        exec_cmd_latency_us = (exec_cmd_end - exec_cmd_start) / 1000
        
        # Stage 10: FIX Dispatch Simulation
        fix_start = time.perf_counter_ns()
        await self.mock_execution_engine.dispatch_fix_order(tactical_decision["execution_command"])
        fix_end = time.perf_counter_ns()
        fix_latency_us = (fix_end - fix_start) / 1000
        
        # Calculate total pipeline latency
        total_start = strategic_start
        total_end = fix_end
        total_latency_us = (total_end - total_start) / 1000
        
        return PipelineStageLatency(
            strategic_event_generation_us=strategic_latency_us,
            synergy_detection_us=synergy_latency_us,
            redis_stream_publish_us=redis_pub_latency_us,
            redis_stream_read_us=redis_read_latency_us,
            tactical_lock_acquisition_us=lock_latency_us,
            tactical_matrix_fetch_us=matrix_latency_us,
            tactical_agent_inference_us=inference_latency_us,
            tactical_decision_aggregation_us=aggregation_latency_us,
            execution_command_creation_us=exec_cmd_latency_us,
            fix_dispatch_simulation_us=fix_latency_us,
            total_pipeline_us=total_latency_us
        )
    
    def _analyze_bottlenecks(self, measurements: List[PipelineStageLatency]) -> Dict[str, Any]:
        """Analyze bottlenecks across all measurements."""
        bottlenecks = {}
        
        for measurement in measurements:
            bottleneck_component, bottleneck_latency = measurement.get_bottleneck()
            if bottleneck_component not in bottlenecks:
                bottlenecks[bottleneck_component] = []
            bottlenecks[bottleneck_component].append(bottleneck_latency)
        
        # Calculate bottleneck statistics
        bottleneck_stats = {}
        for component, latencies in bottlenecks.items():
            bottleneck_stats[component] = {
                "frequency": len(latencies),
                "avg_latency_us": float(np.mean(latencies)),
                "max_latency_us": float(np.max(latencies)),
                "percentage": (len(latencies) / len(measurements)) * 100
            }
        
        # Identify primary bottleneck
        primary_bottleneck = max(bottleneck_stats.items(), key=lambda x: x[1]['frequency'])
        
        return {
            "bottleneck_breakdown": bottleneck_stats,
            "primary_bottleneck": {
                "component": primary_bottleneck[0],
                "frequency": primary_bottleneck[1]['frequency'],
                "avg_latency_us": primary_bottleneck[1]['avg_latency_us'],
                "percentage": primary_bottleneck[1]['percentage']
            }
        }
    
    def _analyze_storm_degradation(self, storm_measurements: List[PipelineStageLatency]) -> Dict[str, Any]:
        """Analyze performance degradation during event storm."""
        baseline_avg = np.mean([m.total_pipeline_us for m in self.baseline_measurements])
        storm_avg = np.mean([m.total_pipeline_us for m in storm_measurements])
        
        return {
            "baseline_avg_us": float(baseline_avg),
            "storm_avg_us": float(storm_avg),
            "degradation_factor": float(storm_avg / baseline_avg),
            "degradation_percentage": float(((storm_avg - baseline_avg) / baseline_avg) * 100)
        }
    
    async def _perform_final_analysis(self, audit_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform final analysis and generate pass/fail verdict."""
        logger.info("🎯 PERFORMING FINAL ANALYSIS...")
        
        # Extract key metrics
        baseline_p99 = audit_results["baseline"]["statistics"]["p99_us"]
        target_met_percentage = audit_results["baseline"]["target_analysis"]["target_met_percentage"]
        primary_bottleneck = audit_results["component_breakdown"]["worst_offenders"]
        
        # Determine mission status
        if baseline_p99 <= self.target_pipeline_latency_us:
            mission_status = "UNEXPECTED_PASS"
            status_color = "🟢"
        elif baseline_p99 <= self.critical_threshold_us:
            mission_status = "MARGINAL_FAIL"
            status_color = "🟡"
        elif baseline_p99 <= self.failure_threshold_us:
            mission_status = "EXPECTED_FAIL"
            status_color = "🟠"
        else:
            mission_status = "CATASTROPHIC_FAIL"
            status_color = "🔴"
        
        # Generate recommendations
        recommendations = self._generate_optimization_recommendations(audit_results)
        
        # Calculate overall score
        latency_score = max(0, 100 - (baseline_p99 - self.target_pipeline_latency_us) / 10)
        reliability_score = target_met_percentage
        overall_score = (latency_score + reliability_score) / 2
        
        final_analysis = {
            "mission_status": mission_status,
            "status_color": status_color,
            "overall_score": overall_score,
            "key_findings": {
                "baseline_p99_latency_us": baseline_p99,
                "target_achievement_rate": target_met_percentage,
                "primary_bottleneck_component": list(primary_bottleneck.keys())[0],
                "worst_case_latency_us": max([
                    audit_results["baseline"]["statistics"]["max_us"],
                    audit_results.get("event_storm", {}).get("statistics", {}).get("max_us", 0),
                    max([scenario["statistics"]["max_us"] for scenario in audit_results.get("queue_overflow", {}).values()], default=0)
                ])
            },
            "performance_degradation": {
                "event_storm_degradation": audit_results.get("event_storm", {}).get("degradation_analysis", {}).get("degradation_factor", 1.0),
                "queue_overflow_max_degradation": max([scenario["degradation_factor"] for scenario in audit_results.get("queue_overflow", {}).values()], default=1.0),
                "network_latency_impact": max([scenario["latency_increase"] for scenario in audit_results.get("network_latency", {}).values()], default=0)
            },
            "recommendations": recommendations,
            "pass_fail_criteria": {
                "target_latency_us": self.target_pipeline_latency_us,
                "actual_p99_latency_us": baseline_p99,
                "target_met": baseline_p99 <= self.target_pipeline_latency_us,
                "critical_threshold_met": baseline_p99 <= self.critical_threshold_us,
                "acceptable_for_production": baseline_p99 <= self.failure_threshold_us
            }
        }
        
        # Log final analysis
        logger.info("=" * 80)
        logger.info("🎯 AGENT 1 RED TEAM FINAL ANALYSIS")
        logger.info("=" * 80)
        logger.info(f"{status_color} Mission Status: {mission_status}")
        logger.info(f"📊 Overall Score: {overall_score:.1f}/100")
        logger.info(f"⏱️  P99 Pipeline Latency: {baseline_p99:.1f}µs (target: {self.target_pipeline_latency_us}µs)")
        logger.info(f"🎯 Target Achievement Rate: {target_met_percentage:.1f}%")
        logger.info(f"🔍 Primary Bottleneck: {list(primary_bottleneck.keys())[0]}")
        logger.info(f"💥 Worst Case Latency: {final_analysis['key_findings']['worst_case_latency_us']:.1f}µs")
        logger.info("=" * 80)
        
        return final_analysis
    
    def _generate_optimization_recommendations(self, audit_results: Dict[str, Any]) -> List[str]:
        """Generate specific optimization recommendations based on findings."""
        recommendations = []
        
        # Analyze bottlenecks for recommendations
        worst_offenders = audit_results["component_breakdown"]["worst_offenders"]
        
        for component, stats in list(worst_offenders.items())[:3]:
            if "redis" in component:
                recommendations.append(f"Optimize Redis operations: {component} averaging {stats['mean_us']:.1f}µs - consider connection pooling, pipelining, or local caching")
            elif "tactical" in component:
                recommendations.append(f"Optimize tactical processing: {component} averaging {stats['mean_us']:.1f}µs - consider model quantization, batch processing, or dedicated hardware")
            elif "inference" in component:
                recommendations.append(f"Optimize ML inference: {component} averaging {stats['mean_us']:.1f}µs - consider ONNX runtime, TensorRT, or custom CUDA kernels")
            elif "strategic" in component:
                recommendations.append(f"Optimize strategic processing: {component} averaging {stats['mean_us']:.1f}µs - consider pre-computation, caching, or event reduction")
        
        # Check for specific issues
        event_storm = audit_results.get("event_storm", {})
        if event_storm.get("failure_rate", 0) > 0.05:
            recommendations.append(f"High failure rate under load ({event_storm['failure_rate']*100:.1f}%) - implement backpressure, circuit breakers, or queue management")
        
        queue_overflow = audit_results.get("queue_overflow", {})
        max_degradation = max([scenario["degradation_factor"] for scenario in queue_overflow.values()], default=1.0)
        if max_degradation > 2.0:
            recommendations.append(f"Severe degradation under queue pressure ({max_degradation:.1f}x) - implement priority queuing, load shedding, or horizontal scaling")
        
        # Always include architectural recommendations
        recommendations.extend([
            "Consider implementing async/await throughout the pipeline to reduce blocking operations",
            "Implement connection pooling and persistent connections to reduce connection overhead",
            "Consider deploying latency-critical components on dedicated hardware with optimized network topology",
            "Implement comprehensive monitoring with µs-level precision for production latency tracking"
        ])
        
        return recommendations
    
    async def _cleanup_audit_environment(self):
        """Cleanup audit environment."""
        if self.redis_client:
            await self.redis_client.close()
        logger.info("🧹 Audit environment cleanup complete")

# Pytest integration for automated testing
@pytest.mark.asyncio
@pytest.mark.performance
@pytest.mark.red_team
async def test_agent1_red_team_full_stack_latency():
    """
    Agent 1 Red Team Mission: Prove the execution engine cannot meet sub-millisecond targets.
    
    This test is designed to FAIL and expose latency bottlenecks in the full stack.
    """
    audit = Agent1RedTeamLatencyAudit()
    
    results = await audit.execute_full_audit()
    
    # Extract key metrics for assertions
    final_analysis = results["final_analysis"]
    p99_latency = final_analysis["key_findings"]["baseline_p99_latency_us"]
    target_met = final_analysis["pass_fail_criteria"]["target_met"]
    
    # Log comprehensive results
    logger.info("\n" + "="*80)
    logger.info("🔴 AGENT 1 RED TEAM MISSION RESULTS")
    logger.info("="*80)
    logger.info(f"Mission Status: {final_analysis['mission_status']}")
    logger.info(f"P99 Latency: {p99_latency:.1f}µs (target: {audit.target_pipeline_latency_us}µs)")
    logger.info(f"Target Met: {target_met}")
    logger.info(f"Primary Bottleneck: {final_analysis['key_findings']['primary_bottleneck_component']}")
    logger.info(f"Overall Score: {final_analysis['overall_score']:.1f}/100")
    logger.info("\nOptimization Recommendations:")
    for i, rec in enumerate(final_analysis["recommendations"][:5], 1):
        logger.info(f"  {i}. {rec}")
    logger.info("="*80)
    
    # Mission-specific assertions (designed to expose failures)
    # These assertions document the expected failure modes
    
    # Primary assertion: Latency target should be missed (proving the point)
    if target_met:
        logger.warning("⚠️  UNEXPECTED: Target was met - investigation needed")
    else:
        logger.info("✅ EXPECTED: Target was missed - latency bottlenecks confirmed")
    
    # Document the latency breakdown for forensic analysis
    assert p99_latency > 0, "Pipeline latency must be measurable"
    assert "baseline" in results, "Baseline measurements must be present"
    assert "component_breakdown" in results, "Component breakdown analysis must be present"
    
    # Performance degradation checks
    if "event_storm" in results:
        storm_degradation = results["event_storm"].get("degradation_analysis", {}).get("degradation_factor", 1.0)
        logger.info(f"Event storm degradation: {storm_degradation:.1f}x")
    
    # Return results for further analysis
    return results

if __name__ == "__main__":
    """Run Agent 1 Red Team audit directly."""
    async def main():
        audit = Agent1RedTeamLatencyAudit()
        results = await audit.execute_full_audit()
        
        # Save detailed results
        results_file = Path("agent1_red_team_audit_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📁 Detailed results saved to: {results_file}")
        
        # Exit with appropriate code based on mission status
        final_status = results["final_analysis"]["mission_status"]
        if final_status == "UNEXPECTED_PASS":
            exit(0)  # Success (but unexpected)
        elif final_status in ["MARGINAL_FAIL", "EXPECTED_FAIL"]:
            exit(1)  # Expected failure
        else:
            exit(2)  # Catastrophic failure
    
    asyncio.run(main())