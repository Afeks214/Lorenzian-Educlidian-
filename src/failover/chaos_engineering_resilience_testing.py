"""
Chaos Engineering Resilience Testing Framework
===========================================

This module provides comprehensive chaos engineering tests for resilience validation
with automated RTO/RPO validation, recovery testing, and performance regression detection.

Key Features:
- Automated chaos injection across all system components
- RTO/RPO validation for various failure scenarios
- Self-healing system validation
- Performance regression detection
- Comprehensive recovery validation
- Production-ready chaos testing pipeline

Integration with:
- Database failover testing
- Trading engine failover testing
- Existing chaos engineering framework
- Performance monitoring systems
"""

import asyncio
import time
import logging
import json
import traceback
import random
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import uuid
import numpy as np
from contextlib import asynccontextmanager

# Import existing frameworks
from .database_failover_testing import DatabaseFailoverTester, FailoverTestConfig, FailoverType
from .trading_engine_failover_testing import TradingEngineFailoverTester, TradingEngineFailoverConfig, TradingEngineFailoverType
from ..core.resilience.resilience_manager import ResilienceManager, ResilienceConfig
from ..core.event_bus import EventBus, Event, EventType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChaosTestSuite(Enum):
    """Chaos test suite categories."""
    DATABASE_RESILIENCE = "database_resilience"
    TRADING_ENGINE_RESILIENCE = "trading_engine_resilience"
    NETWORK_RESILIENCE = "network_resilience"
    PERFORMANCE_RESILIENCE = "performance_resilience"
    RECOVERY_VALIDATION = "recovery_validation"
    INTEGRATION_RESILIENCE = "integration_resilience"
    FULL_SYSTEM_RESILIENCE = "full_system_resilience"


class ChaosTestPriority(Enum):
    """Priority levels for chaos tests."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ChaosTestTarget:
    """Target for chaos testing."""
    component: str
    service_name: str
    endpoint: Optional[str] = None
    process_name: Optional[str] = None
    config_path: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    
    # RTO/RPO targets
    rto_target_seconds: float = 30.0
    rpo_target_seconds: float = 1.0
    
    # Test parameters
    failure_modes: List[str] = field(default_factory=list)
    recovery_methods: List[str] = field(default_factory=list)
    validation_criteria: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChaosTestScenario:
    """Comprehensive chaos test scenario."""
    test_id: str
    name: str
    description: str
    suite: ChaosTestSuite
    priority: ChaosTestPriority
    
    # Test configuration
    targets: List[ChaosTestTarget] = field(default_factory=list)
    duration_seconds: int = 300
    max_concurrent_failures: int = 1
    
    # Failure injection
    failure_probability: float = 1.0
    failure_duration_seconds: int = 60
    cascading_failures: bool = False
    
    # Recovery validation
    enable_recovery_validation: bool = True
    enable_performance_validation: bool = True
    enable_data_consistency_validation: bool = True
    
    # Success criteria
    max_rto_seconds: float = 30.0
    max_rpo_seconds: float = 1.0
    min_availability_percent: float = 99.0
    max_performance_degradation_percent: float = 20.0
    
    # Execution tracking
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: str = "pending"
    results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChaosTestResult:
    """Result of chaos test execution."""
    test_id: str
    scenario: ChaosTestScenario
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Metrics
    rto_achieved: Dict[str, float] = field(default_factory=dict)
    rpo_achieved: Dict[str, float] = field(default_factory=dict)
    availability_achieved: float = 0.0
    performance_degradation: float = 0.0
    
    # Detailed results
    failure_injection_results: List[Dict[str, Any]] = field(default_factory=list)
    recovery_validation_results: List[Dict[str, Any]] = field(default_factory=list)
    performance_test_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # System state
    pre_test_state: Dict[str, Any] = field(default_factory=dict)
    post_test_state: Dict[str, Any] = field(default_factory=dict)
    system_changes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Issues
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def success(self) -> bool:
        """Check if test was successful."""
        return (
            self.status == "completed" and
            len(self.errors) == 0 and
            all(rto <= self.scenario.max_rto_seconds for rto in self.rto_achieved.values()) and
            all(rpo <= self.scenario.max_rpo_seconds for rpo in self.rpo_achieved.values()) and
            self.availability_achieved >= self.scenario.min_availability_percent and
            self.performance_degradation <= self.scenario.max_performance_degradation_percent
        )


class ChaosTestOrchestrator:
    """Orchestrator for chaos engineering tests."""
    
    def __init__(self, resilience_manager: ResilienceManager):
        self.resilience_manager = resilience_manager
        self.event_bus = EventBus()
        self.active_tests: Dict[str, ChaosTestResult] = {}
        self.test_history: List[ChaosTestResult] = []
        self.test_lock = threading.Lock()
        
    async def initialize(self):
        """Initialize chaos test orchestrator."""
        await self.event_bus.initialize()
        logger.info("Chaos test orchestrator initialized")
    
    async def close(self):
        """Close orchestrator."""
        await self.event_bus.close()
        logger.info("Chaos test orchestrator closed")
    
    def create_test_scenarios(self) -> List[ChaosTestScenario]:
        """Create comprehensive test scenarios."""
        scenarios = []
        
        # Database resilience scenarios
        scenarios.extend(self._create_database_scenarios())
        
        # Trading engine resilience scenarios
        scenarios.extend(self._create_trading_engine_scenarios())
        
        # Network resilience scenarios
        scenarios.extend(self._create_network_scenarios())
        
        # Performance resilience scenarios
        scenarios.extend(self._create_performance_scenarios())
        
        # Recovery validation scenarios
        scenarios.extend(self._create_recovery_scenarios())
        
        # Integration resilience scenarios
        scenarios.extend(self._create_integration_scenarios())
        
        # Full system resilience scenarios
        scenarios.extend(self._create_full_system_scenarios())
        
        return scenarios
    
    def _create_database_scenarios(self) -> List[ChaosTestScenario]:
        """Create database resilience scenarios."""
        scenarios = []
        
        # Primary database failure
        scenarios.append(ChaosTestScenario(
            test_id="db_primary_failure_001",
            name="Database Primary Failure",
            description="Test database failover when primary node fails",
            suite=ChaosTestSuite.DATABASE_RESILIENCE,
            priority=ChaosTestPriority.CRITICAL,
            targets=[
                ChaosTestTarget(
                    component="database",
                    service_name="postgresql-primary",
                    process_name="postgres",
                    rto_target_seconds=15.0,
                    rpo_target_seconds=1.0,
                    failure_modes=["process_kill", "network_partition", "disk_failure"],
                    recovery_methods=["patroni_failover", "standby_promotion"]
                )
            ],
            duration_seconds=120,
            max_rto_seconds=15.0,
            max_rpo_seconds=1.0
        ))
        
        # Database connection exhaustion
        scenarios.append(ChaosTestScenario(
            test_id="db_connection_exhaustion_001",
            name="Database Connection Exhaustion",
            description="Test system behavior when database connections are exhausted",
            suite=ChaosTestSuite.DATABASE_RESILIENCE,
            priority=ChaosTestPriority.HIGH,
            targets=[
                ChaosTestTarget(
                    component="database",
                    service_name="postgresql-primary",
                    rto_target_seconds=30.0,
                    failure_modes=["connection_flooding"],
                    recovery_methods=["connection_pool_reset", "database_restart"]
                )
            ],
            duration_seconds=180,
            max_rto_seconds=30.0
        ))
        
        # Patroni cluster split-brain
        scenarios.append(ChaosTestScenario(
            test_id="db_split_brain_001",
            name="Patroni Cluster Split-Brain",
            description="Test Patroni cluster behavior during split-brain scenario",
            suite=ChaosTestSuite.DATABASE_RESILIENCE,
            priority=ChaosTestPriority.HIGH,
            targets=[
                ChaosTestTarget(
                    component="patroni",
                    service_name="patroni-cluster",
                    rto_target_seconds=45.0,
                    failure_modes=["etcd_partition", "network_split"],
                    recovery_methods=["manual_failover", "cluster_rebuild"]
                )
            ],
            duration_seconds=300,
            max_rto_seconds=45.0
        ))
        
        return scenarios
    
    def _create_trading_engine_scenarios(self) -> List[ChaosTestScenario]:
        """Create trading engine resilience scenarios."""
        scenarios = []
        
        # Tactical agent failure
        scenarios.append(ChaosTestScenario(
            test_id="tactical_agent_failure_001",
            name="Tactical Agent Failure",
            description="Test tactical agent recovery and state restoration",
            suite=ChaosTestSuite.TRADING_ENGINE_RESILIENCE,
            priority=ChaosTestPriority.CRITICAL,
            targets=[
                ChaosTestTarget(
                    component="tactical_agent",
                    service_name="tactical-marl",
                    endpoint="http://localhost:8001",
                    rto_target_seconds=30.0,
                    failure_modes=["process_kill", "model_corruption", "state_corruption"],
                    recovery_methods=["agent_restart", "model_recompilation", "state_recovery"]
                )
            ],
            duration_seconds=180,
            max_rto_seconds=30.0
        ))
        
        # Strategic agent failure
        scenarios.append(ChaosTestScenario(
            test_id="strategic_agent_failure_001",
            name="Strategic Agent Failure",
            description="Test strategic agent recovery and coordination",
            suite=ChaosTestSuite.TRADING_ENGINE_RESILIENCE,
            priority=ChaosTestPriority.CRITICAL,
            targets=[
                ChaosTestTarget(
                    component="strategic_agent",
                    service_name="strategic-marl",
                    endpoint="http://localhost:8002",
                    rto_target_seconds=45.0,
                    failure_modes=["process_kill", "memory_exhaustion"],
                    recovery_methods=["agent_restart", "coordination_recovery"]
                )
            ],
            duration_seconds=240,
            max_rto_seconds=45.0
        ))
        
        # MARL coordination failure
        scenarios.append(ChaosTestScenario(
            test_id="marl_coordination_failure_001",
            name="MARL Coordination Failure",
            description="Test multi-agent coordination recovery",
            suite=ChaosTestSuite.TRADING_ENGINE_RESILIENCE,
            priority=ChaosTestPriority.HIGH,
            targets=[
                ChaosTestTarget(
                    component="marl_coordinator",
                    service_name="marl-coordinator",
                    rto_target_seconds=60.0,
                    failure_modes=["coordinator_kill", "communication_failure"],
                    recovery_methods=["coordinator_restart", "agent_re-registration"]
                )
            ],
            duration_seconds=300,
            max_rto_seconds=60.0
        ))
        
        # JIT model corruption
        scenarios.append(ChaosTestScenario(
            test_id="jit_model_corruption_001",
            name="JIT Model Corruption",
            description="Test JIT model recovery and recompilation",
            suite=ChaosTestSuite.TRADING_ENGINE_RESILIENCE,
            priority=ChaosTestPriority.MEDIUM,
            targets=[
                ChaosTestTarget(
                    component="jit_models",
                    service_name="model-cache",
                    rto_target_seconds=120.0,
                    failure_modes=["model_corruption", "cache_corruption"],
                    recovery_methods=["model_recompilation", "cache_rebuild"]
                )
            ],
            duration_seconds=360,
            max_rto_seconds=120.0
        ))
        
        return scenarios
    
    def _create_network_scenarios(self) -> List[ChaosTestScenario]:
        """Create network resilience scenarios."""
        scenarios = []
        
        # Network partition between services
        scenarios.append(ChaosTestScenario(
            test_id="network_partition_001",
            name="Inter-Service Network Partition",
            description="Test system behavior during network partition",
            suite=ChaosTestSuite.NETWORK_RESILIENCE,
            priority=ChaosTestPriority.HIGH,
            targets=[
                ChaosTestTarget(
                    component="network",
                    service_name="inter-service-communication",
                    rto_target_seconds=30.0,
                    failure_modes=["network_partition", "latency_injection"],
                    recovery_methods=["network_healing", "service_discovery"]
                )
            ],
            duration_seconds=180,
            max_rto_seconds=30.0
        ))
        
        # High network latency
        scenarios.append(ChaosTestScenario(
            test_id="high_latency_001",
            name="High Network Latency",
            description="Test system performance under high network latency",
            suite=ChaosTestSuite.NETWORK_RESILIENCE,
            priority=ChaosTestPriority.MEDIUM,
            targets=[
                ChaosTestTarget(
                    component="network",
                    service_name="network-layer",
                    rto_target_seconds=60.0,
                    failure_modes=["latency_injection", "packet_loss"],
                    recovery_methods=["adaptive_timeout", "circuit_breaker"]
                )
            ],
            duration_seconds=300,
            max_performance_degradation_percent=30.0
        ))
        
        return scenarios
    
    def _create_performance_scenarios(self) -> List[ChaosTestScenario]:
        """Create performance resilience scenarios."""
        scenarios = []
        
        # Memory exhaustion
        scenarios.append(ChaosTestScenario(
            test_id="memory_exhaustion_001",
            name="System Memory Exhaustion",
            description="Test system behavior under memory pressure",
            suite=ChaosTestSuite.PERFORMANCE_RESILIENCE,
            priority=ChaosTestPriority.HIGH,
            targets=[
                ChaosTestTarget(
                    component="system_memory",
                    service_name="memory-manager",
                    rto_target_seconds=60.0,
                    failure_modes=["memory_exhaustion", "memory_leak"],
                    recovery_methods=["memory_cleanup", "process_restart"]
                )
            ],
            duration_seconds=240,
            max_rto_seconds=60.0
        ))
        
        # CPU overload
        scenarios.append(ChaosTestScenario(
            test_id="cpu_overload_001",
            name="CPU Overload",
            description="Test system performance under CPU pressure",
            suite=ChaosTestSuite.PERFORMANCE_RESILIENCE,
            priority=ChaosTestPriority.MEDIUM,
            targets=[
                ChaosTestTarget(
                    component="cpu",
                    service_name="cpu-scheduler",
                    rto_target_seconds=30.0,
                    failure_modes=["cpu_overload", "cpu_throttling"],
                    recovery_methods=["load_balancing", "priority_adjustment"]
                )
            ],
            duration_seconds=180,
            max_performance_degradation_percent=40.0
        ))
        
        # Disk I/O saturation
        scenarios.append(ChaosTestScenario(
            test_id="disk_io_saturation_001",
            name="Disk I/O Saturation",
            description="Test system behavior under disk I/O pressure",
            suite=ChaosTestSuite.PERFORMANCE_RESILIENCE,
            priority=ChaosTestPriority.MEDIUM,
            targets=[
                ChaosTestTarget(
                    component="disk_io",
                    service_name="disk-manager",
                    rto_target_seconds=45.0,
                    failure_modes=["disk_saturation", "disk_full"],
                    recovery_methods=["io_throttling", "disk_cleanup"]
                )
            ],
            duration_seconds=300,
            max_rto_seconds=45.0
        ))
        
        return scenarios
    
    def _create_recovery_scenarios(self) -> List[ChaosTestScenario]:
        """Create recovery validation scenarios."""
        scenarios = []
        
        # Self-healing validation
        scenarios.append(ChaosTestScenario(
            test_id="self_healing_001",
            name="Self-Healing Validation",
            description="Test system self-healing capabilities",
            suite=ChaosTestSuite.RECOVERY_VALIDATION,
            priority=ChaosTestPriority.HIGH,
            targets=[
                ChaosTestTarget(
                    component="self_healing",
                    service_name="healing-manager",
                    rto_target_seconds=90.0,
                    failure_modes=["multiple_failures", "cascading_failures"],
                    recovery_methods=["automatic_healing", "assisted_recovery"]
                )
            ],
            duration_seconds=360,
            max_rto_seconds=90.0
        ))
        
        # Backup and restore validation
        scenarios.append(ChaosTestScenario(
            test_id="backup_restore_001",
            name="Backup and Restore Validation",
            description="Test backup and restore mechanisms",
            suite=ChaosTestSuite.RECOVERY_VALIDATION,
            priority=ChaosTestPriority.MEDIUM,
            targets=[
                ChaosTestTarget(
                    component="backup_system",
                    service_name="backup-manager",
                    rto_target_seconds=300.0,
                    rpo_target_seconds=30.0,
                    failure_modes=["data_corruption", "complete_data_loss"],
                    recovery_methods=["backup_restore", "point_in_time_recovery"]
                )
            ],
            duration_seconds=600,
            max_rto_seconds=300.0,
            max_rpo_seconds=30.0
        ))
        
        return scenarios
    
    def _create_integration_scenarios(self) -> List[ChaosTestScenario]:
        """Create integration resilience scenarios."""
        scenarios = []
        
        # Cross-component failure
        scenarios.append(ChaosTestScenario(
            test_id="cross_component_failure_001",
            name="Cross-Component Failure",
            description="Test system behavior during cross-component failures",
            suite=ChaosTestSuite.INTEGRATION_RESILIENCE,
            priority=ChaosTestPriority.HIGH,
            targets=[
                ChaosTestTarget(
                    component="database",
                    service_name="postgresql-primary",
                    rto_target_seconds=30.0,
                    failure_modes=["process_kill"],
                    dependencies=["tactical_agent", "strategic_agent"]
                ),
                ChaosTestTarget(
                    component="redis",
                    service_name="redis-server",
                    rto_target_seconds=15.0,
                    failure_modes=["process_kill"],
                    dependencies=["agent_communication"]
                )
            ],
            duration_seconds=240,
            max_concurrent_failures=2,
            cascading_failures=True,
            max_rto_seconds=45.0
        ))
        
        # Event bus failure
        scenarios.append(ChaosTestScenario(
            test_id="event_bus_failure_001",
            name="Event Bus Failure",
            description="Test system behavior when event bus fails",
            suite=ChaosTestSuite.INTEGRATION_RESILIENCE,
            priority=ChaosTestPriority.HIGH,
            targets=[
                ChaosTestTarget(
                    component="event_bus",
                    service_name="event-bus",
                    rto_target_seconds=20.0,
                    failure_modes=["bus_corruption", "message_loss"],
                    recovery_methods=["bus_restart", "message_replay"]
                )
            ],
            duration_seconds=180,
            max_rto_seconds=20.0
        ))
        
        return scenarios
    
    def _create_full_system_scenarios(self) -> List[ChaosTestScenario]:
        """Create full system resilience scenarios."""
        scenarios = []
        
        # Complete system failure
        scenarios.append(ChaosTestScenario(
            test_id="complete_system_failure_001",
            name="Complete System Failure",
            description="Test complete system recovery from total failure",
            suite=ChaosTestSuite.FULL_SYSTEM_RESILIENCE,
            priority=ChaosTestPriority.CRITICAL,
            targets=[
                ChaosTestTarget(
                    component="full_system",
                    service_name="trading-system",
                    rto_target_seconds=300.0,
                    rpo_target_seconds=5.0,
                    failure_modes=["complete_shutdown", "power_failure"],
                    recovery_methods=["full_system_restart", "disaster_recovery"]
                )
            ],
            duration_seconds=900,
            max_rto_seconds=300.0,
            max_rpo_seconds=5.0
        ))
        
        # Cascading failure simulation
        scenarios.append(ChaosTestScenario(
            test_id="cascading_failure_001",
            name="Cascading Failure Simulation",
            description="Test system resilience against cascading failures",
            suite=ChaosTestSuite.FULL_SYSTEM_RESILIENCE,
            priority=ChaosTestPriority.HIGH,
            targets=[
                ChaosTestTarget(
                    component="database",
                    service_name="postgresql-primary",
                    rto_target_seconds=30.0,
                    failure_modes=["process_kill"]
                ),
                ChaosTestTarget(
                    component="tactical_agent",
                    service_name="tactical-marl",
                    rto_target_seconds=45.0,
                    failure_modes=["process_kill"]
                ),
                ChaosTestTarget(
                    component="strategic_agent",
                    service_name="strategic-marl",
                    rto_target_seconds=60.0,
                    failure_modes=["process_kill"]
                )
            ],
            duration_seconds=360,
            max_concurrent_failures=3,
            cascading_failures=True,
            max_rto_seconds=90.0
        ))
        
        return scenarios
    
    async def run_chaos_test(self, scenario: ChaosTestScenario) -> ChaosTestResult:
        """Run a single chaos test scenario."""
        result = ChaosTestResult(
            test_id=scenario.test_id,
            scenario=scenario,
            status="running",
            start_time=datetime.now()
        )
        
        with self.test_lock:
            self.active_tests[scenario.test_id] = result
        
        logger.info(f"Starting chaos test: {scenario.name}")
        
        try:
            # Phase 1: Pre-test state capture
            await self._capture_pre_test_state(result)
            
            # Phase 2: Execute failure injection
            await self._execute_failure_injection(result)
            
            # Phase 3: Monitor recovery
            await self._monitor_recovery(result)
            
            # Phase 4: Validate recovery
            await self._validate_recovery(result)
            
            # Phase 5: Performance validation
            await self._validate_performance(result)
            
            # Phase 6: Post-test state capture
            await self._capture_post_test_state(result)
            
            # Phase 7: Cleanup
            await self._cleanup_test(result)
            
            result.status = "completed"
            result.end_time = datetime.now()
            
            # Calculate final metrics
            self._calculate_final_metrics(result)
            
            logger.info(f"Chaos test completed: {scenario.name}")
            logger.info(f"Test Success: {result.success()}")
            
        except Exception as e:
            result.status = "failed"
            result.end_time = datetime.now()
            result.errors.append(f"Test failed: {str(e)}")
            
            logger.error(f"Chaos test failed: {scenario.name} - {str(e)}")
            
            # Emergency cleanup
            await self._emergency_cleanup(result)
        
        finally:
            with self.test_lock:
                self.active_tests.pop(scenario.test_id, None)
                self.test_history.append(result)
        
        return result
    
    async def _capture_pre_test_state(self, result: ChaosTestResult):
        """Capture system state before test."""
        try:
            # Capture system metrics
            system_state = await self.resilience_manager.get_system_status()
            result.pre_test_state["system_status"] = system_state
            
            # Capture service health
            service_health = {}
            for target in result.scenario.targets:
                if target.endpoint:
                    try:
                        import httpx
                        async with httpx.AsyncClient() as client:
                            response = await client.get(f"{target.endpoint}/health", timeout=5.0)
                            service_health[target.component] = {
                                "healthy": response.status_code == 200,
                                "response_time": response.elapsed.total_seconds()
                            }
                    except Exception as e:
                        service_health[target.component] = {
                            "healthy": False,
                            "error": str(e)
                        }
            
            result.pre_test_state["service_health"] = service_health
            
            # Capture performance baseline
            performance_baseline = await self._measure_performance_baseline()
            result.pre_test_state["performance_baseline"] = performance_baseline
            
        except Exception as e:
            result.warnings.append(f"Pre-test state capture failed: {str(e)}")
    
    async def _execute_failure_injection(self, result: ChaosTestResult):
        """Execute failure injection for all targets."""
        try:
            injection_results = []
            
            for target in result.scenario.targets:
                logger.info(f"Injecting failure for target: {target.component}")
                
                # Choose failure mode
                failure_mode = random.choice(target.failure_modes) if target.failure_modes else "generic_failure"
                
                # Execute failure injection
                injection_result = await self._inject_failure(target, failure_mode)
                injection_result["target"] = target.component
                injection_result["failure_mode"] = failure_mode
                injection_results.append(injection_result)
                
                # Wait between injections if multiple targets
                if len(result.scenario.targets) > 1:
                    await asyncio.sleep(5)
            
            result.failure_injection_results = injection_results
            
        except Exception as e:
            result.errors.append(f"Failure injection failed: {str(e)}")
    
    async def _inject_failure(self, target: ChaosTestTarget, failure_mode: str) -> Dict[str, Any]:
        """Inject specific failure for a target."""
        try:
            injection_result = {
                "success": False,
                "start_time": time.time(),
                "method": failure_mode,
                "details": {}
            }
            
            if failure_mode == "process_kill":
                # Kill process
                import psutil
                killed_pids = []
                
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        cmdline = ' '.join(proc.info['cmdline'] or [])
                        if target.process_name and target.process_name in proc.info['name'].lower():
                            psutil.Process(proc.info['pid']).kill()
                            killed_pids.append(proc.info['pid'])
                        elif target.service_name and target.service_name in cmdline.lower():
                            psutil.Process(proc.info['pid']).kill()
                            killed_pids.append(proc.info['pid'])
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                injection_result["success"] = len(killed_pids) > 0
                injection_result["details"]["killed_pids"] = killed_pids
                
            elif failure_mode == "network_partition":
                # Create network partition
                import subprocess
                
                if target.endpoint:
                    # Extract port from endpoint
                    import urllib.parse
                    parsed = urllib.parse.urlparse(target.endpoint)
                    port = parsed.port or 80
                    
                    # Block port
                    cmd = ["iptables", "-A", "INPUT", "-p", "tcp", "--dport", str(port), "-j", "DROP"]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    injection_result["success"] = result.returncode == 0
                    injection_result["details"]["blocked_port"] = port
                    injection_result["details"]["command"] = ' '.join(cmd)
                
            elif failure_mode == "memory_exhaustion":
                # Simulate memory exhaustion
                import numpy as np
                
                try:
                    # Allocate large memory chunks
                    memory_chunks = []
                    for i in range(10):
                        chunk = np.random.random((100 * 1024 * 1024 // 8,))  # 100MB chunks
                        memory_chunks.append(chunk)
                        await asyncio.sleep(0.1)
                    
                    injection_result["success"] = True
                    injection_result["details"]["memory_chunks"] = len(memory_chunks)
                    injection_result["details"]["memory_bombs"] = memory_chunks  # Keep reference
                    
                except Exception as e:
                    injection_result["details"]["error"] = str(e)
                
            elif failure_mode == "cpu_overload":
                # Generate CPU load
                async def cpu_load():
                    end_time = time.time() + 60  # 1 minute of CPU load
                    while time.time() < end_time:
                        # CPU-intensive calculation
                        sum(i * i for i in range(10000))
                        await asyncio.sleep(0.001)
                
                # Start CPU load task
                cpu_task = asyncio.create_task(cpu_load())
                injection_result["success"] = True
                injection_result["details"]["cpu_task"] = cpu_task
                
            elif failure_mode == "disk_saturation":
                # Fill disk space
                import tempfile
                import os
                
                try:
                    temp_files = []
                    for i in range(5):
                        temp_file = tempfile.NamedTemporaryFile(delete=False)
                        temp_file.write(b'0' * (100 * 1024 * 1024))  # 100MB per file
                        temp_file.close()
                        temp_files.append(temp_file.name)
                    
                    injection_result["success"] = True
                    injection_result["details"]["temp_files"] = temp_files
                    
                except Exception as e:
                    injection_result["details"]["error"] = str(e)
                
            else:
                injection_result["details"]["error"] = f"Unknown failure mode: {failure_mode}"
            
            injection_result["end_time"] = time.time()
            injection_result["duration"] = injection_result["end_time"] - injection_result["start_time"]
            
            return injection_result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "method": failure_mode,
                "start_time": time.time(),
                "end_time": time.time()
            }
    
    async def _monitor_recovery(self, result: ChaosTestResult):
        """Monitor system recovery after failure injection."""
        try:
            recovery_start = time.time()
            max_monitor_time = result.scenario.duration_seconds
            
            recovery_data = {
                "monitoring_start": recovery_start,
                "status_samples": [],
                "recovery_events": []
            }
            
            while time.time() - recovery_start < max_monitor_time:
                current_time = time.time()
                
                # Sample system status
                system_status = await self.resilience_manager.get_system_status()
                recovery_data["status_samples"].append({
                    "timestamp": current_time,
                    "system_status": system_status
                })
                
                # Check for recovery events
                recovery_events = await self._detect_recovery_events(result.scenario.targets)
                if recovery_events:
                    recovery_data["recovery_events"].extend(recovery_events)
                
                # Check if full recovery achieved
                if await self._is_full_recovery_achieved(result.scenario.targets):
                    recovery_data["full_recovery_time"] = current_time - recovery_start
                    break
                
                await asyncio.sleep(5)  # Sample every 5 seconds
            
            result.recovery_validation_results.append(recovery_data)
            
        except Exception as e:
            result.errors.append(f"Recovery monitoring failed: {str(e)}")
    
    async def _detect_recovery_events(self, targets: List[ChaosTestTarget]) -> List[Dict[str, Any]]:
        """Detect recovery events for targets."""
        events = []
        
        for target in targets:
            if target.endpoint:
                try:
                    import httpx
                    async with httpx.AsyncClient() as client:
                        response = await client.get(f"{target.endpoint}/health", timeout=5.0)
                        if response.status_code == 200:
                            events.append({
                                "event": "service_recovery",
                                "target": target.component,
                                "timestamp": time.time(),
                                "response_time": response.elapsed.total_seconds()
                            })
                except Exception:
                    pass
        
        return events
    
    async def _is_full_recovery_achieved(self, targets: List[ChaosTestTarget]) -> bool:
        """Check if full recovery is achieved for all targets."""
        for target in targets:
            if target.endpoint:
                try:
                    import httpx
                    async with httpx.AsyncClient() as client:
                        response = await client.get(f"{target.endpoint}/health", timeout=5.0)
                        if response.status_code != 200:
                            return False
                except Exception:
                    return False
        
        return True
    
    async def _validate_recovery(self, result: ChaosTestResult):
        """Validate recovery completeness."""
        try:
            validation_results = []
            
            for target in result.scenario.targets:
                target_validation = {
                    "target": target.component,
                    "rto_achieved": 0.0,
                    "rpo_achieved": 0.0,
                    "recovery_validated": False
                }
                
                # Calculate RTO
                recovery_events = [
                    event for event in result.recovery_validation_results[0].get("recovery_events", [])
                    if event.get("target") == target.component
                ]
                
                if recovery_events:
                    first_recovery = min(recovery_events, key=lambda x: x["timestamp"])
                    target_validation["rto_achieved"] = (
                        first_recovery["timestamp"] - result.recovery_validation_results[0]["monitoring_start"]
                    )
                    target_validation["recovery_validated"] = True
                
                # Validate against target RTO
                if target_validation["rto_achieved"] <= target.rto_target_seconds:
                    result.rto_achieved[target.component] = target_validation["rto_achieved"]
                else:
                    result.warnings.append(
                        f"RTO target missed for {target.component}: "
                        f"{target_validation['rto_achieved']:.2f}s > {target.rto_target_seconds}s"
                    )
                
                validation_results.append(target_validation)
            
            result.recovery_validation_results.append({
                "validation_results": validation_results,
                "timestamp": time.time()
            })
            
        except Exception as e:
            result.errors.append(f"Recovery validation failed: {str(e)}")
    
    async def _validate_performance(self, result: ChaosTestResult):
        """Validate post-recovery performance."""
        try:
            # Measure current performance
            current_performance = await self._measure_performance_baseline()
            
            # Compare with baseline
            baseline_performance = result.pre_test_state.get("performance_baseline", {})
            
            performance_degradation = 0.0
            
            if baseline_performance and current_performance:
                # Calculate degradation based on response times
                baseline_avg = baseline_performance.get("average_response_time", 0.0)
                current_avg = current_performance.get("average_response_time", 0.0)
                
                if baseline_avg > 0:
                    performance_degradation = ((current_avg - baseline_avg) / baseline_avg) * 100
            
            result.performance_degradation = max(0, performance_degradation)
            
            result.performance_test_results.append({
                "baseline_performance": baseline_performance,
                "current_performance": current_performance,
                "performance_degradation": performance_degradation,
                "timestamp": time.time()
            })
            
        except Exception as e:
            result.errors.append(f"Performance validation failed: {str(e)}")
    
    async def _measure_performance_baseline(self) -> Dict[str, Any]:
        """Measure system performance baseline."""
        try:
            # Measure system metrics
            import psutil
            
            baseline = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "network_io": psutil.net_io_counters()._asdict(),
                "timestamp": time.time()
            }
            
            # Measure service response times
            service_endpoints = [
                "http://localhost:8001/health",
                "http://localhost:8002/health",
                "http://localhost:8003/health"
            ]
            
            response_times = []
            
            for endpoint in service_endpoints:
                try:
                    import httpx
                    async with httpx.AsyncClient() as client:
                        start_time = time.time()
                        response = await client.get(endpoint, timeout=5.0)
                        response_time = time.time() - start_time
                        
                        if response.status_code == 200:
                            response_times.append(response_time)
                            
                except Exception:
                    pass
            
            if response_times:
                baseline["average_response_time"] = sum(response_times) / len(response_times)
                baseline["max_response_time"] = max(response_times)
                baseline["min_response_time"] = min(response_times)
            
            return baseline
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _capture_post_test_state(self, result: ChaosTestResult):
        """Capture system state after test."""
        try:
            # Capture final system state
            system_state = await self.resilience_manager.get_system_status()
            result.post_test_state["system_status"] = system_state
            
            # Calculate system changes
            pre_state = result.pre_test_state.get("system_status", {})
            changes = self._calculate_system_changes(pre_state, system_state)
            result.system_changes = changes
            
        except Exception as e:
            result.warnings.append(f"Post-test state capture failed: {str(e)}")
    
    def _calculate_system_changes(self, pre_state: Dict[str, Any], post_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate changes between pre and post test states."""
        changes = []
        
        # Compare service states
        pre_services = pre_state.get("services", {})
        post_services = post_state.get("services", {})
        
        for service_name, post_service in post_services.items():
            pre_service = pre_services.get(service_name, {})
            
            if pre_service.get("healthy") != post_service.get("healthy"):
                changes.append({
                    "type": "service_health_change",
                    "service": service_name,
                    "before": pre_service.get("healthy"),
                    "after": post_service.get("healthy")
                })
        
        return changes
    
    async def _cleanup_test(self, result: ChaosTestResult):
        """Clean up after test execution."""
        try:
            # Clean up failure injections
            for injection_result in result.failure_injection_results:
                await self._cleanup_injection(injection_result)
            
            # Wait for system to stabilize
            await asyncio.sleep(30)
            
        except Exception as e:
            result.warnings.append(f"Test cleanup failed: {str(e)}")
    
    async def _cleanup_injection(self, injection_result: Dict[str, Any]):
        """Clean up a specific failure injection."""
        try:
            method = injection_result.get("method")
            details = injection_result.get("details", {})
            
            if method == "network_partition":
                # Remove iptables rules
                import subprocess
                blocked_port = details.get("blocked_port")
                if blocked_port:
                    cmd = ["iptables", "-D", "INPUT", "-p", "tcp", "--dport", str(blocked_port), "-j", "DROP"]
                    subprocess.run(cmd, capture_output=True)
                    
            elif method == "memory_exhaustion":
                # Memory will be freed automatically when reference is removed
                pass
                
            elif method == "cpu_overload":
                # Cancel CPU task
                cpu_task = details.get("cpu_task")
                if cpu_task and not cpu_task.done():
                    cpu_task.cancel()
                    
            elif method == "disk_saturation":
                # Remove temporary files
                import os
                temp_files = details.get("temp_files", [])
                for temp_file in temp_files:
                    try:
                        os.remove(temp_file)
                    except FileNotFoundError:
                        pass
                        
        except Exception as e:
            logger.warning(f"Failed to cleanup injection: {e}")
    
    async def _emergency_cleanup(self, result: ChaosTestResult):
        """Emergency cleanup for failed tests."""
        try:
            logger.warning("Performing emergency cleanup")
            
            # Reset iptables
            import subprocess
            subprocess.run(["iptables", "-F"], capture_output=True)
            
            # Restart critical services
            critical_services = [
                "tactical-agent",
                "strategic-agent", 
                "redis",
                "postgresql"
            ]
            
            for service in critical_services:
                subprocess.run(["systemctl", "restart", service], capture_output=True)
            
            # Wait for services to stabilize
            await asyncio.sleep(60)
            
        except Exception as e:
            logger.error(f"Emergency cleanup failed: {e}")
    
    def _calculate_final_metrics(self, result: ChaosTestResult):
        """Calculate final test metrics."""
        try:
            # Calculate availability
            total_samples = len(result.recovery_validation_results[0].get("status_samples", []))
            healthy_samples = 0
            
            for sample in result.recovery_validation_results[0].get("status_samples", []):
                system_status = sample.get("system_status", {})
                services = system_status.get("services", {})
                
                if all(service.get("healthy", False) for service in services.values()):
                    healthy_samples += 1
            
            if total_samples > 0:
                result.availability_achieved = (healthy_samples / total_samples) * 100
            
            # Set overall RTO/RPO achievements
            if result.rto_achieved:
                result.rto_achieved["overall"] = max(result.rto_achieved.values())
            
            if result.rpo_achieved:
                result.rpo_achieved["overall"] = max(result.rpo_achieved.values())
            
        except Exception as e:
            result.warnings.append(f"Final metrics calculation failed: {str(e)}")
    
    async def run_chaos_test_suite(self, suite: ChaosTestSuite, priority: Optional[ChaosTestPriority] = None) -> List[ChaosTestResult]:
        """Run a complete chaos test suite."""
        scenarios = self.create_test_scenarios()
        
        # Filter scenarios by suite and priority
        filtered_scenarios = [
            scenario for scenario in scenarios
            if scenario.suite == suite and (priority is None or scenario.priority == priority)
        ]
        
        logger.info(f"Running chaos test suite: {suite.value} with {len(filtered_scenarios)} scenarios")
        
        results = []
        
        for scenario in filtered_scenarios:
            logger.info(f"Running scenario: {scenario.name}")
            
            # Add delay between scenarios
            if results:
                await asyncio.sleep(30)
            
            result = await self.run_chaos_test(scenario)
            results.append(result)
            
            # Stop on critical failure
            if not result.success() and scenario.priority == ChaosTestPriority.CRITICAL:
                logger.error(f"Critical test failed: {scenario.name}")
                break
        
        # Generate suite summary
        successful_tests = sum(1 for result in results if result.success())
        
        logger.info(f"Chaos test suite completed: {successful_tests}/{len(results)} tests passed")
        
        return results
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of all tests."""
        active_count = len(self.active_tests)
        total_tests = len(self.test_history)
        successful_tests = sum(1 for test in self.test_history if test.success())
        
        return {
            "active_tests": active_count,
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "test_history": [
                {
                    "test_id": test.test_id,
                    "scenario_name": test.scenario.name,
                    "status": test.status,
                    "success": test.success(),
                    "duration": (test.end_time - test.start_time).total_seconds() if test.end_time else 0
                }
                for test in self.test_history
            ]
        }


# Example usage
async def main():
    """Demonstrate chaos engineering resilience testing."""
    # Create resilience manager
    resilience_config = ResilienceConfig(
        service_name="trading_system",
        environment="testing"
    )
    
    resilience_manager = ResilienceManager(resilience_config)
    await resilience_manager.initialize()
    
    # Create chaos test orchestrator
    orchestrator = ChaosTestOrchestrator(resilience_manager)
    await orchestrator.initialize()
    
    try:
        # Run database resilience tests
        db_results = await orchestrator.run_chaos_test_suite(
            ChaosTestSuite.DATABASE_RESILIENCE,
            ChaosTestPriority.CRITICAL
        )
        
        print(f"Database resilience tests: {len(db_results)} tests")
        
        # Run trading engine resilience tests
        trading_results = await orchestrator.run_chaos_test_suite(
            ChaosTestSuite.TRADING_ENGINE_RESILIENCE,
            ChaosTestPriority.HIGH
        )
        
        print(f"Trading engine resilience tests: {len(trading_results)} tests")
        
        # Get test summary
        summary = orchestrator.get_test_summary()
        print(f"Test Summary: {summary}")
        
    finally:
        await orchestrator.close()
        await resilience_manager.close()


if __name__ == "__main__":
    asyncio.run(main())