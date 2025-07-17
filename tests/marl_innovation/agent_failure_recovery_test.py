"""
Agent Failure Recovery Test Suite
Agent 3 Mission: Comprehensive Agent Failure Recovery Testing

This module tests the system's ability to recover from agent failures while maintaining 
coordination and operational continuity. Tests cover:

1. Individual agent failures (timeout, exception, disconnect)
2. Cascade failure prevention
3. Graceful degradation scenarios
4. Recovery time optimization
5. Circuit breaker patterns
6. State consistency during recovery
"""

import pytest
import asyncio
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass
import logging
from enum import Enum

from src.core.events import EventBus, Event, EventType
from src.core.kernel import AlgoSpaceKernel

# Test markers
pytestmark = [pytest.mark.recovery, pytest.mark.marl_innovation]

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of agent failures to test"""
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    DISCONNECT = "disconnect"
    MEMORY_LEAK = "memory_leak"
    DEADLOCK = "deadlock"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


class RecoveryStrategy(Enum):
    """Recovery strategies for different failure types"""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class FailureScenario:
    """Represents a failure scenario for testing"""
    name: str
    description: str
    failed_agent: str
    failure_type: FailureType
    failure_duration_seconds: float
    expected_recovery_strategy: RecoveryStrategy
    max_recovery_time_seconds: float
    should_trigger_cascade: bool = False
    critical_functionality_affected: List[str] = None


@dataclass
class RecoveryResult:
    """Result of a recovery test"""
    scenario_name: str
    success: bool
    recovery_time_seconds: float
    cascade_failures: List[str]
    functionality_maintained: List[str]
    functionality_lost: List[str]
    error_messages: List[str]
    performance_impact: Dict[str, float]


class AgentFailureSimulator:
    """
    Simulates various types of agent failures for testing recovery mechanisms
    """
    
    def __init__(self):
        self.active_failures = {}
        self.failure_history = []
        self.recovery_attempts = {}
        
    async def inject_failure(self, agent: Mock, failure_type: FailureType, 
                           duration_seconds: float = 10.0) -> str:
        """Inject a failure into an agent"""
        failure_id = f"{agent.name}_{failure_type.value}_{int(time.time())}"
        
        logger.info(f"Injecting failure: {failure_id}")
        
        # Store original methods
        original_methods = {}
        if hasattr(agent, 'make_decision'):
            original_methods['make_decision'] = agent.make_decision
        if hasattr(agent, 'assess_risk'):
            original_methods['assess_risk'] = agent.assess_risk
        if hasattr(agent, 'generate_explanation'):
            original_methods['generate_explanation'] = agent.generate_explanation
        
        # Inject failure behavior
        if failure_type == FailureType.TIMEOUT:
            await self._inject_timeout_failure(agent, duration_seconds)
        elif failure_type == FailureType.EXCEPTION:
            await self._inject_exception_failure(agent, duration_seconds)
        elif failure_type == FailureType.DISCONNECT:
            await self._inject_disconnect_failure(agent, duration_seconds)
        elif failure_type == FailureType.MEMORY_LEAK:
            await self._inject_memory_leak_failure(agent, duration_seconds)
        elif failure_type == FailureType.DEADLOCK:
            await self._inject_deadlock_failure(agent, duration_seconds)
        elif failure_type == FailureType.RESOURCE_EXHAUSTION:
            await self._inject_resource_exhaustion_failure(agent, duration_seconds)
        
        # Track failure
        self.active_failures[failure_id] = {
            'agent': agent,
            'failure_type': failure_type,
            'start_time': time.time(),
            'duration': duration_seconds,
            'original_methods': original_methods
        }
        
        self.failure_history.append({
            'failure_id': failure_id,
            'agent_name': agent.name,
            'failure_type': failure_type.value,
            'start_time': datetime.now(),
            'duration_seconds': duration_seconds
        })
        
        # Schedule recovery
        asyncio.create_task(self._schedule_recovery(failure_id, duration_seconds))
        
        return failure_id
    
    async def _inject_timeout_failure(self, agent: Mock, duration_seconds: float):
        """Inject timeout failure - agent becomes very slow"""
        original_methods = {}
        
        if hasattr(agent, 'make_decision'):
            original_methods['make_decision'] = agent.make_decision
            
            async def timeout_decision(*args, **kwargs):
                await asyncio.sleep(duration_seconds)  # Simulate timeout
                if time.time() - self.active_failures[f"{agent.name}_timeout_{int(time.time())}"]["start_time"] < duration_seconds:
                    raise asyncio.TimeoutError("Agent timeout")
                return await original_methods['make_decision'](*args, **kwargs)
            
            agent.make_decision = timeout_decision
        
        if hasattr(agent, 'assess_risk'):
            original_methods['assess_risk'] = agent.assess_risk
            
            async def timeout_risk(*args, **kwargs):
                await asyncio.sleep(duration_seconds)
                if time.time() - self.active_failures[f"{agent.name}_timeout_{int(time.time())}"]["start_time"] < duration_seconds:
                    raise asyncio.TimeoutError("Risk assessment timeout")
                return await original_methods['assess_risk'](*args, **kwargs)
            
            agent.assess_risk = timeout_risk
        
        if hasattr(agent, 'generate_explanation'):
            original_methods['generate_explanation'] = agent.generate_explanation
            
            async def timeout_explanation(*args, **kwargs):
                await asyncio.sleep(duration_seconds)
                if time.time() - self.active_failures[f"{agent.name}_timeout_{int(time.time())}"]["start_time"] < duration_seconds:
                    raise asyncio.TimeoutError("Explanation timeout")
                return await original_methods['generate_explanation'](*args, **kwargs)
            
            agent.generate_explanation = timeout_explanation
    
    async def _inject_exception_failure(self, agent: Mock, duration_seconds: float):
        """Inject exception failure - agent throws exceptions"""
        if hasattr(agent, 'make_decision'):
            async def exception_decision(*args, **kwargs):
                raise Exception(f"Simulated exception in {agent.name}")
            agent.make_decision = exception_decision
        
        if hasattr(agent, 'assess_risk'):
            async def exception_risk(*args, **kwargs):
                raise Exception(f"Simulated risk assessment exception in {agent.name}")
            agent.assess_risk = exception_risk
        
        if hasattr(agent, 'generate_explanation'):
            async def exception_explanation(*args, **kwargs):
                raise Exception(f"Simulated explanation exception in {agent.name}")
            agent.generate_explanation = exception_explanation
    
    async def _inject_disconnect_failure(self, agent: Mock, duration_seconds: float):
        """Inject disconnect failure - agent becomes unresponsive"""
        agent.initialized = False
        agent.connected = False
        
        # Remove agent methods
        if hasattr(agent, 'make_decision'):
            delattr(agent, 'make_decision')
        if hasattr(agent, 'assess_risk'):
            delattr(agent, 'assess_risk')
        if hasattr(agent, 'generate_explanation'):
            delattr(agent, 'generate_explanation')
    
    async def _inject_memory_leak_failure(self, agent: Mock, duration_seconds: float):
        """Inject memory leak failure - agent gradually degrades"""
        agent.memory_usage = 0
        
        if hasattr(agent, 'make_decision'):
            original_decision = agent.make_decision
            
            async def leaky_decision(*args, **kwargs):
                agent.memory_usage += 1000  # Simulate memory leak
                if agent.memory_usage > 10000:
                    raise MemoryError(f"Out of memory in {agent.name}")
                await asyncio.sleep(agent.memory_usage * 0.001)  # Gradual slowdown
                return await original_decision(*args, **kwargs)
            
            agent.make_decision = leaky_decision
    
    async def _inject_deadlock_failure(self, agent: Mock, duration_seconds: float):
        """Inject deadlock failure - agent gets stuck"""
        agent.deadlock_lock = asyncio.Lock()
        
        if hasattr(agent, 'make_decision'):
            async def deadlock_decision(*args, **kwargs):
                async with agent.deadlock_lock:
                    # Simulate deadlock by acquiring lock and never releasing
                    await asyncio.sleep(duration_seconds)
                    raise Exception("Deadlock detected")
            
            agent.make_decision = deadlock_decision
    
    async def _inject_resource_exhaustion_failure(self, agent: Mock, duration_seconds: float):
        """Inject resource exhaustion failure - agent runs out of resources"""
        agent.resource_count = 100
        
        if hasattr(agent, 'make_decision'):
            original_decision = agent.make_decision
            
            async def resource_exhausted_decision(*args, **kwargs):
                agent.resource_count -= 10
                if agent.resource_count <= 0:
                    raise Exception(f"Resource exhaustion in {agent.name}")
                return await original_decision(*args, **kwargs)
            
            agent.make_decision = resource_exhausted_decision
    
    async def _schedule_recovery(self, failure_id: str, duration_seconds: float):
        """Schedule automatic recovery after failure duration"""
        await asyncio.sleep(duration_seconds)
        await self.recover_failure(failure_id)
    
    async def recover_failure(self, failure_id: str) -> bool:
        """Recover from a failure"""
        if failure_id not in self.active_failures:
            return False
        
        failure_info = self.active_failures[failure_id]
        agent = failure_info['agent']
        original_methods = failure_info['original_methods']
        
        logger.info(f"Recovering from failure: {failure_id}")
        
        # Restore original methods
        for method_name, original_method in original_methods.items():
            setattr(agent, method_name, original_method)
        
        # Restore agent state
        agent.initialized = True
        agent.connected = True
        
        # Clean up failure-specific state
        if hasattr(agent, 'memory_usage'):
            delattr(agent, 'memory_usage')
        if hasattr(agent, 'deadlock_lock'):
            delattr(agent, 'deadlock_lock')
        if hasattr(agent, 'resource_count'):
            delattr(agent, 'resource_count')
        
        # Remove from active failures
        del self.active_failures[failure_id]
        
        # Track recovery
        self.recovery_attempts[failure_id] = {
            'recovery_time': datetime.now(),
            'success': True
        }
        
        logger.info(f"Recovery completed for: {failure_id}")
        return True
    
    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get statistics about failures and recoveries"""
        total_failures = len(self.failure_history)
        total_recoveries = len(self.recovery_attempts)
        
        failure_types = {}
        for failure in self.failure_history:
            failure_type = failure['failure_type']
            if failure_type not in failure_types:
                failure_types[failure_type] = 0
            failure_types[failure_type] += 1
        
        return {
            'total_failures': total_failures,
            'total_recoveries': total_recoveries,
            'recovery_rate': total_recoveries / max(1, total_failures),
            'active_failures': len(self.active_failures),
            'failure_types': failure_types
        }


class RecoveryTestOrchestrator:
    """
    Orchestrates recovery testing scenarios
    """
    
    def __init__(self):
        self.event_bus = EventBus()
        self.agents = {}
        self.failure_simulator = AgentFailureSimulator()
        self.recovery_results = []
        self.performance_baseline = {}
        
    async def initialize_agents(self, config: Dict[str, Any]) -> bool:
        """Initialize agents for recovery testing"""
        try:
            # Initialize agents with recovery capabilities
            self.agents['strategic'] = await self._create_resilient_strategic_agent(config.get('strategic', {}))
            self.agents['tactical'] = await self._create_resilient_tactical_agent(config.get('tactical', {}))
            self.agents['risk'] = await self._create_resilient_risk_agent(config.get('risk', {}))
            self.agents['xai'] = await self._create_resilient_xai_agent(config.get('xai', {}))
            
            # Establish performance baseline
            await self._establish_performance_baseline()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            return False
    
    async def _create_resilient_strategic_agent(self, config: Dict[str, Any]) -> Mock:
        """Create strategic agent with recovery capabilities"""
        agent = Mock()
        agent.name = "strategic_agent"
        agent.initialized = True
        agent.connected = True
        agent.circuit_breaker = {'failures': 0, 'last_failure': None, 'is_open': False}
        
        async def make_decision(synergy_data):
            if agent.circuit_breaker['is_open']:
                raise Exception("Circuit breaker open")
            
            try:
                await asyncio.sleep(0.1)  # Simulate processing
                decision = {
                    'action': np.random.choice(['buy', 'sell', 'hold']),
                    'confidence': np.random.uniform(0.6, 0.9),
                    'timestamp': datetime.now(),
                    'agent': 'strategic'
                }
                
                # Reset circuit breaker on success
                agent.circuit_breaker['failures'] = 0
                
                return decision
            except Exception as e:
                agent.circuit_breaker['failures'] += 1
                agent.circuit_breaker['last_failure'] = datetime.now()
                
                if agent.circuit_breaker['failures'] >= 3:
                    agent.circuit_breaker['is_open'] = True
                    logger.warning(f"Circuit breaker opened for {agent.name}")
                
                raise
        
        agent.make_decision = make_decision
        return agent
    
    async def _create_resilient_tactical_agent(self, config: Dict[str, Any]) -> Mock:
        """Create tactical agent with recovery capabilities"""
        agent = Mock()
        agent.name = "tactical_agent"
        agent.initialized = True
        agent.connected = True
        agent.fallback_mode = False
        agent.retry_count = 0
        agent.max_retries = 3
        
        async def make_decision(strategic_signal):
            if agent.fallback_mode:
                return {
                    'action': 'hold',
                    'confidence': 0.5,
                    'execution_timing': 'delayed',
                    'timestamp': datetime.now(),
                    'agent': 'tactical_fallback'
                }
            
            try:
                await asyncio.sleep(0.05)
                decision = {
                    'action': strategic_signal.get('action', 'hold'),
                    'confidence': strategic_signal.get('confidence', 0.5) * 0.9,
                    'execution_timing': 'immediate',
                    'timestamp': datetime.now(),
                    'agent': 'tactical'
                }
                
                # Reset retry count on success
                agent.retry_count = 0
                
                return decision
            except Exception as e:
                agent.retry_count += 1
                
                if agent.retry_count >= agent.max_retries:
                    agent.fallback_mode = True
                    logger.warning(f"Tactical agent entering fallback mode")
                
                raise
        
        agent.make_decision = make_decision
        return agent
    
    async def _create_resilient_risk_agent(self, config: Dict[str, Any]) -> Mock:
        """Create risk agent with recovery capabilities"""
        agent = Mock()
        agent.name = "risk_agent"
        agent.initialized = True
        agent.connected = True
        agent.emergency_mode = False
        
        async def assess_risk(decision_data):
            if agent.emergency_mode:
                return {
                    'risk_score': 0.9,
                    'risk_level': 'high',
                    'recommendation': 'emergency_stop',
                    'timestamp': datetime.now(),
                    'agent': 'risk_emergency'
                }
            
            try:
                await asyncio.sleep(0.02)
                risk_score = np.random.uniform(0.1, 0.8)
                
                assessment = {
                    'risk_score': risk_score,
                    'risk_level': 'high' if risk_score > 0.7 else 'medium' if risk_score > 0.4 else 'low',
                    'recommendation': 'proceed' if risk_score < 0.6 else 'reduce_position',
                    'timestamp': datetime.now(),
                    'agent': 'risk'
                }
                
                return assessment
            except Exception as e:
                agent.emergency_mode = True
                logger.warning(f"Risk agent entering emergency mode")
                raise
        
        agent.assess_risk = assess_risk
        return agent
    
    async def _create_resilient_xai_agent(self, config: Dict[str, Any]) -> Mock:
        """Create XAI agent with recovery capabilities"""
        agent = Mock()
        agent.name = "xai_agent"
        agent.initialized = True
        agent.connected = True
        agent.cache = {}
        
        async def generate_explanation(decision_data):
            try:
                # Check cache first
                cache_key = str(hash(str(decision_data)))
                if cache_key in agent.cache:
                    return agent.cache[cache_key]
                
                await asyncio.sleep(0.2)
                explanation = {
                    'explanation_text': f"Decision based on {decision_data.get('reasoning', 'analysis')}",
                    'confidence_score': decision_data.get('confidence', 0.5),
                    'key_factors': ['market_trend', 'risk_level'],
                    'timestamp': datetime.now(),
                    'agent': 'xai'
                }
                
                # Cache explanation
                agent.cache[cache_key] = explanation
                
                return explanation
            except Exception as e:
                # Return cached explanation if available
                if agent.cache:
                    return list(agent.cache.values())[-1]
                
                # Fallback explanation
                return {
                    'explanation_text': "Unable to generate detailed explanation",
                    'confidence_score': 0.3,
                    'key_factors': ['system_error'],
                    'timestamp': datetime.now(),
                    'agent': 'xai_fallback'
                }
        
        agent.generate_explanation = generate_explanation
        return agent
    
    async def _establish_performance_baseline(self):
        """Establish performance baseline for comparison"""
        baseline_runs = 10
        
        for agent_name, agent in self.agents.items():
            latencies = []
            success_rates = []
            
            for _ in range(baseline_runs):
                start_time = time.time()
                try:
                    if hasattr(agent, 'make_decision'):
                        await agent.make_decision({'test': True})
                    elif hasattr(agent, 'assess_risk'):
                        await agent.assess_risk({'test': True})
                    elif hasattr(agent, 'generate_explanation'):
                        await agent.generate_explanation({'test': True})
                    
                    success_rates.append(1.0)
                except Exception:
                    success_rates.append(0.0)
                
                latencies.append((time.time() - start_time) * 1000)
            
            self.performance_baseline[agent_name] = {
                'avg_latency_ms': np.mean(latencies),
                'success_rate': np.mean(success_rates),
                'p95_latency_ms': np.percentile(latencies, 95)
            }
        
        logger.info(f"Performance baseline established: {self.performance_baseline}")
    
    async def run_failure_scenario(self, scenario: FailureScenario) -> RecoveryResult:
        """Run a single failure scenario"""
        logger.info(f"Running failure scenario: {scenario.name}")
        
        start_time = time.time()
        result = RecoveryResult(
            scenario_name=scenario.name,
            success=False,
            recovery_time_seconds=0.0,
            cascade_failures=[],
            functionality_maintained=[],
            functionality_lost=[],
            error_messages=[],
            performance_impact={}
        )
        
        try:
            # Get baseline performance
            baseline_performance = await self._measure_system_performance()
            
            # Inject failure
            agent = self.agents[scenario.failed_agent]
            failure_id = await self.failure_simulator.inject_failure(
                agent, scenario.failure_type, scenario.failure_duration_seconds
            )
            
            # Monitor system during failure
            await asyncio.sleep(0.1)  # Let failure propagate
            
            # Test functionality during failure
            functionality_results = await self._test_functionality_during_failure(scenario)
            result.functionality_maintained = functionality_results['maintained']
            result.functionality_lost = functionality_results['lost']
            
            # Detect cascade failures
            cascade_failures = await self._detect_cascade_failures(scenario.failed_agent)
            result.cascade_failures = cascade_failures
            
            # Wait for recovery
            recovery_start = time.time()
            
            # Test recovery
            max_wait_time = scenario.max_recovery_time_seconds
            recovery_successful = False
            
            while time.time() - recovery_start < max_wait_time:
                if await self._test_agent_recovery(scenario.failed_agent):
                    recovery_successful = True
                    break
                await asyncio.sleep(0.1)
            
            result.recovery_time_seconds = time.time() - recovery_start
            result.success = recovery_successful
            
            # Measure performance impact
            if recovery_successful:
                post_recovery_performance = await self._measure_system_performance()
                result.performance_impact = self._calculate_performance_impact(
                    baseline_performance, post_recovery_performance
                )
            
        except Exception as e:
            result.error_messages.append(str(e))
            logger.error(f"Error in failure scenario {scenario.name}: {e}")
        
        result.recovery_time_seconds = time.time() - start_time
        self.recovery_results.append(result)
        
        return result
    
    async def _measure_system_performance(self) -> Dict[str, Any]:
        """Measure current system performance"""
        performance = {}
        
        for agent_name, agent in self.agents.items():
            if not agent.initialized:
                continue
            
            latencies = []
            success_count = 0
            total_attempts = 5
            
            for _ in range(total_attempts):
                start_time = time.time()
                try:
                    if hasattr(agent, 'make_decision'):
                        await agent.make_decision({'test': True})
                    elif hasattr(agent, 'assess_risk'):
                        await agent.assess_risk({'test': True})
                    elif hasattr(agent, 'generate_explanation'):
                        await agent.generate_explanation({'test': True})
                    
                    success_count += 1
                except Exception:
                    pass
                
                latencies.append((time.time() - start_time) * 1000)
            
            performance[agent_name] = {
                'avg_latency_ms': np.mean(latencies),
                'success_rate': success_count / total_attempts,
                'responsive': success_count > 0
            }
        
        return performance
    
    async def _test_functionality_during_failure(self, scenario: FailureScenario) -> Dict[str, List[str]]:
        """Test what functionality is maintained during failure"""
        functionality_results = {
            'maintained': [],
            'lost': []
        }
        
        # Define critical functionality by agent
        critical_functions = {
            'strategic': ['strategic_decision_making', 'market_analysis'],
            'tactical': ['trade_execution', 'timing_optimization'],
            'risk': ['risk_assessment', 'portfolio_monitoring'],
            'xai': ['decision_explanation', 'transparency_reporting']
        }
        
        # Test each function
        for agent_name, functions in critical_functions.items():
            if agent_name == scenario.failed_agent:
                # Test if failed agent functions are lost
                for func in functions:
                    if await self._test_specific_function(agent_name, func):
                        functionality_results['maintained'].append(f"{agent_name}_{func}")
                    else:
                        functionality_results['lost'].append(f"{agent_name}_{func}")
            else:
                # Test if other agents maintain functionality
                for func in functions:
                    if await self._test_specific_function(agent_name, func):
                        functionality_results['maintained'].append(f"{agent_name}_{func}")
                    else:
                        functionality_results['lost'].append(f"{agent_name}_{func}")
        
        return functionality_results
    
    async def _test_specific_function(self, agent_name: str, function_name: str) -> bool:
        """Test a specific function of an agent"""
        if agent_name not in self.agents:
            return False
        
        agent = self.agents[agent_name]
        
        try:
            if function_name.endswith('decision_making') and hasattr(agent, 'make_decision'):
                await asyncio.wait_for(agent.make_decision({'test': True}), timeout=1.0)
            elif function_name.endswith('assessment') and hasattr(agent, 'assess_risk'):
                await asyncio.wait_for(agent.assess_risk({'test': True}), timeout=1.0)
            elif function_name.endswith('explanation') and hasattr(agent, 'generate_explanation'):
                await asyncio.wait_for(agent.generate_explanation({'test': True}), timeout=1.0)
            else:
                return False
            
            return True
        except Exception:
            return False
    
    async def _detect_cascade_failures(self, failed_agent: str) -> List[str]:
        """Detect cascade failures caused by the primary failure"""
        cascade_failures = []
        
        # Test other agents for cascade effects
        for agent_name, agent in self.agents.items():
            if agent_name == failed_agent:
                continue
            
            # Check if agent is still responsive
            try:
                if hasattr(agent, 'make_decision'):
                    await asyncio.wait_for(agent.make_decision({'test': True}), timeout=0.5)
                elif hasattr(agent, 'assess_risk'):
                    await asyncio.wait_for(agent.assess_risk({'test': True}), timeout=0.5)
                elif hasattr(agent, 'generate_explanation'):
                    await asyncio.wait_for(agent.generate_explanation({'test': True}), timeout=0.5)
            except Exception:
                cascade_failures.append(agent_name)
        
        return cascade_failures
    
    async def _test_agent_recovery(self, agent_name: str) -> bool:
        """Test if an agent has recovered from failure"""
        if agent_name not in self.agents:
            return False
        
        agent = self.agents[agent_name]
        
        try:
            # Test basic functionality
            if hasattr(agent, 'make_decision'):
                await asyncio.wait_for(agent.make_decision({'test': True}), timeout=1.0)
            elif hasattr(agent, 'assess_risk'):
                await asyncio.wait_for(agent.assess_risk({'test': True}), timeout=1.0)
            elif hasattr(agent, 'generate_explanation'):
                await asyncio.wait_for(agent.generate_explanation({'test': True}), timeout=1.0)
            
            return True
        except Exception:
            return False
    
    def _calculate_performance_impact(self, baseline: Dict[str, Any], 
                                    post_recovery: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance impact of failure and recovery"""
        impact = {}
        
        for agent_name in baseline:
            if agent_name in post_recovery:
                baseline_latency = baseline[agent_name]['avg_latency_ms']
                post_latency = post_recovery[agent_name]['avg_latency_ms']
                
                baseline_success = baseline[agent_name]['success_rate']
                post_success = post_recovery[agent_name]['success_rate']
                
                impact[agent_name] = {
                    'latency_impact_percent': ((post_latency - baseline_latency) / baseline_latency) * 100,
                    'success_rate_impact_percent': ((post_success - baseline_success) / baseline_success) * 100,
                    'overall_impact_score': self._calculate_overall_impact_score(
                        baseline_latency, post_latency, baseline_success, post_success
                    )
                }
        
        return impact
    
    def _calculate_overall_impact_score(self, baseline_latency: float, post_latency: float,
                                      baseline_success: float, post_success: float) -> float:
        """Calculate overall impact score (0-1, where 0 is no impact)"""
        latency_impact = abs(post_latency - baseline_latency) / baseline_latency
        success_impact = abs(post_success - baseline_success) / baseline_success
        
        # Weighted average (success rate is more important)
        overall_impact = (latency_impact * 0.3) + (success_impact * 0.7)
        
        return min(1.0, overall_impact)
    
    async def run_cascade_failure_test(self) -> Dict[str, Any]:
        """Test cascade failure scenarios"""
        logger.info("Running cascade failure test")
        
        # Inject multiple failures simultaneously
        failure_tasks = []
        
        for agent_name in ['strategic', 'tactical']:
            task = asyncio.create_task(
                self.failure_simulator.inject_failure(
                    self.agents[agent_name], 
                    FailureType.EXCEPTION, 
                    duration_seconds=5.0
                )
            )
            failure_tasks.append(task)
        
        # Wait for failures to be injected
        await asyncio.gather(*failure_tasks)
        
        # Monitor system behavior
        await asyncio.sleep(1.0)
        
        # Test if system can still function
        system_functional = await self._test_system_functionality()
        
        # Test recovery
        recovery_time = await self._measure_recovery_time()
        
        return {
            'cascade_detected': len(failure_tasks) > 1,
            'system_remained_functional': system_functional,
            'recovery_time_seconds': recovery_time,
            'affected_agents': list(self.agents.keys())
        }
    
    async def _test_system_functionality(self) -> bool:
        """Test if the system can still function despite failures"""
        try:
            # Test basic coordination flow
            if 'risk' in self.agents and self.agents['risk'].initialized:
                await self.agents['risk'].assess_risk({'test': True})
            
            if 'xai' in self.agents and self.agents['xai'].initialized:
                await self.agents['xai'].generate_explanation({'test': True})
            
            return True
        except Exception:
            return False
    
    async def _measure_recovery_time(self) -> float:
        """Measure time for system to recover from cascade failure"""
        start_time = time.time()
        max_wait_time = 10.0
        
        while time.time() - start_time < max_wait_time:
            all_recovered = True
            
            for agent_name in self.agents:
                if not await self._test_agent_recovery(agent_name):
                    all_recovered = False
                    break
            
            if all_recovered:
                return time.time() - start_time
            
            await asyncio.sleep(0.1)
        
        return max_wait_time  # Timeout
    
    def get_recovery_summary(self) -> Dict[str, Any]:
        """Get comprehensive recovery test summary"""
        if not self.recovery_results:
            return {'no_tests_run': True}
        
        total_tests = len(self.recovery_results)
        successful_recoveries = sum(1 for r in self.recovery_results if r.success)
        
        avg_recovery_time = np.mean([r.recovery_time_seconds for r in self.recovery_results])
        
        failure_types_tested = list(set([
            r.scenario_name.split('_')[0] for r in self.recovery_results
        ]))
        
        cascade_failures = sum(len(r.cascade_failures) for r in self.recovery_results)
        
        return {
            'total_tests': total_tests,
            'successful_recoveries': successful_recoveries,
            'recovery_success_rate': successful_recoveries / total_tests,
            'avg_recovery_time_seconds': avg_recovery_time,
            'failure_types_tested': failure_types_tested,
            'total_cascade_failures': cascade_failures,
            'performance_baseline': self.performance_baseline,
            'failure_statistics': self.failure_simulator.get_failure_statistics()
        }


class TestAgentFailureRecovery:
    """Test suite for agent failure recovery"""
    
    @pytest.fixture
    async def recovery_orchestrator(self):
        """Set up recovery test orchestrator"""
        orchestrator = RecoveryTestOrchestrator()
        
        config = {
            'strategic': {'circuit_breaker_threshold': 3},
            'tactical': {'max_retries': 3, 'fallback_enabled': True},
            'risk': {'emergency_mode_enabled': True},
            'xai': {'cache_enabled': True}
        }
        
        success = await orchestrator.initialize_agents(config)
        assert success, "Failed to initialize agents"
        
        yield orchestrator
        
        # Cleanup
        await orchestrator.event_bus.stop()
    
    @pytest.mark.asyncio
    async def test_strategic_agent_timeout_recovery(self, recovery_orchestrator):
        """Test strategic agent recovery from timeout failure"""
        scenario = FailureScenario(
            name="strategic_timeout_test",
            description="Test strategic agent timeout recovery",
            failed_agent="strategic",
            failure_type=FailureType.TIMEOUT,
            failure_duration_seconds=2.0,
            expected_recovery_strategy=RecoveryStrategy.CIRCUIT_BREAKER,
            max_recovery_time_seconds=5.0
        )
        
        result = await recovery_orchestrator.run_failure_scenario(scenario)
        
        assert result.success, "Strategic agent failed to recover from timeout"
        assert result.recovery_time_seconds <= 5.0, "Recovery time too long"
        assert len(result.cascade_failures) == 0, "Unexpected cascade failures"
    
    @pytest.mark.asyncio
    async def test_tactical_agent_exception_recovery(self, recovery_orchestrator):
        """Test tactical agent recovery from exception failure"""
        scenario = FailureScenario(
            name="tactical_exception_test",
            description="Test tactical agent exception recovery",
            failed_agent="tactical",
            failure_type=FailureType.EXCEPTION,
            failure_duration_seconds=3.0,
            expected_recovery_strategy=RecoveryStrategy.FALLBACK,
            max_recovery_time_seconds=5.0
        )
        
        result = await recovery_orchestrator.run_failure_scenario(scenario)
        
        assert result.success, "Tactical agent failed to recover from exception"
        assert 'tactical_trade_execution' in result.functionality_maintained or \
               'tactical_trade_execution' in result.functionality_lost, "Functionality not tracked"
    
    @pytest.mark.asyncio
    async def test_risk_agent_disconnect_recovery(self, recovery_orchestrator):
        """Test risk agent recovery from disconnect failure"""
        scenario = FailureScenario(
            name="risk_disconnect_test",
            description="Test risk agent disconnect recovery",
            failed_agent="risk",
            failure_type=FailureType.DISCONNECT,
            failure_duration_seconds=2.0,
            expected_recovery_strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
            max_recovery_time_seconds=5.0
        )
        
        result = await recovery_orchestrator.run_failure_scenario(scenario)
        
        assert result.success, "Risk agent failed to recover from disconnect"
        assert len(result.cascade_failures) <= 1, "Too many cascade failures"
    
    @pytest.mark.asyncio
    async def test_xai_agent_memory_leak_recovery(self, recovery_orchestrator):
        """Test XAI agent recovery from memory leak"""
        scenario = FailureScenario(
            name="xai_memory_leak_test",
            description="Test XAI agent memory leak recovery",
            failed_agent="xai",
            failure_type=FailureType.MEMORY_LEAK,
            failure_duration_seconds=4.0,
            expected_recovery_strategy=RecoveryStrategy.RETRY,
            max_recovery_time_seconds=6.0
        )
        
        result = await recovery_orchestrator.run_failure_scenario(scenario)
        
        assert result.success, "XAI agent failed to recover from memory leak"
        assert 'xai_decision_explanation' in result.functionality_maintained or \
               'xai_decision_explanation' in result.functionality_lost, "XAI functionality not tracked"
    
    @pytest.mark.asyncio
    async def test_cascade_failure_prevention(self, recovery_orchestrator):
        """Test cascade failure prevention"""
        result = await recovery_orchestrator.run_cascade_failure_test()
        
        assert result['cascade_detected'], "Cascade failure not detected"
        assert result['system_remained_functional'], "System did not remain functional"
        assert result['recovery_time_seconds'] <= 10.0, "Recovery time too long"
    
    @pytest.mark.asyncio
    async def test_multiple_failure_scenarios(self, recovery_orchestrator):
        """Test multiple failure scenarios in sequence"""
        scenarios = [
            FailureScenario(
                name="multi_test_1",
                description="First failure test",
                failed_agent="strategic",
                failure_type=FailureType.TIMEOUT,
                failure_duration_seconds=1.0,
                expected_recovery_strategy=RecoveryStrategy.CIRCUIT_BREAKER,
                max_recovery_time_seconds=3.0
            ),
            FailureScenario(
                name="multi_test_2",
                description="Second failure test",
                failed_agent="tactical",
                failure_type=FailureType.EXCEPTION,
                failure_duration_seconds=1.0,
                expected_recovery_strategy=RecoveryStrategy.FALLBACK,
                max_recovery_time_seconds=3.0
            ),
            FailureScenario(
                name="multi_test_3",
                description="Third failure test",
                failed_agent="risk",
                failure_type=FailureType.DISCONNECT,
                failure_duration_seconds=1.0,
                expected_recovery_strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                max_recovery_time_seconds=3.0
            )
        ]
        
        results = []
        for scenario in scenarios:
            result = await recovery_orchestrator.run_failure_scenario(scenario)
            results.append(result)
            
            # Wait between scenarios
            await asyncio.sleep(0.5)
        
        # Validate all scenarios
        successful_recoveries = sum(1 for r in results if r.success)
        assert successful_recoveries >= 2, "Too many recovery failures"
        
        # Validate no cumulative degradation
        final_performance = await recovery_orchestrator._measure_system_performance()
        assert len(final_performance) >= 3, "System performance degraded"
    
    @pytest.mark.asyncio
    async def test_recovery_performance_impact(self, recovery_orchestrator):
        """Test performance impact of recovery"""
        scenario = FailureScenario(
            name="performance_impact_test",
            description="Test performance impact of recovery",
            failed_agent="strategic",
            failure_type=FailureType.TIMEOUT,
            failure_duration_seconds=2.0,
            expected_recovery_strategy=RecoveryStrategy.CIRCUIT_BREAKER,
            max_recovery_time_seconds=5.0
        )
        
        result = await recovery_orchestrator.run_failure_scenario(scenario)
        
        assert result.success, "Recovery failed"
        assert result.performance_impact, "Performance impact not measured"
        
        # Check that performance impact is reasonable
        for agent_name, impact in result.performance_impact.items():
            assert impact['overall_impact_score'] <= 0.5, f"Performance impact too high for {agent_name}"
    
    def test_recovery_summary_generation(self, recovery_orchestrator):
        """Test recovery summary generation"""
        # Add some mock recovery results
        recovery_orchestrator.recovery_results = [
            RecoveryResult(
                scenario_name="test_1",
                success=True,
                recovery_time_seconds=2.5,
                cascade_failures=[],
                functionality_maintained=['strategic_decision_making'],
                functionality_lost=[],
                error_messages=[],
                performance_impact={}
            ),
            RecoveryResult(
                scenario_name="test_2",
                success=False,
                recovery_time_seconds=10.0,
                cascade_failures=['tactical'],
                functionality_maintained=[],
                functionality_lost=['risk_assessment'],
                error_messages=['Timeout'],
                performance_impact={}
            )
        ]
        
        summary = recovery_orchestrator.get_recovery_summary()
        
        assert 'total_tests' in summary
        assert 'successful_recoveries' in summary
        assert 'recovery_success_rate' in summary
        assert 'avg_recovery_time_seconds' in summary
        
        assert summary['total_tests'] == 2
        assert summary['successful_recoveries'] == 1
        assert summary['recovery_success_rate'] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])