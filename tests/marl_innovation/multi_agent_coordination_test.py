"""
Multi-Agent Coordination Test Suite
Agent 3 Mission: Comprehensive Multi-Agent Coordination Testing

This module tests the coordination mechanisms between the four main agents:
1. Strategic Agent (30-minute timeframe)
2. Tactical Agent (5-minute timeframe)  
3. Risk Management Agent
4. XAI Engine

Tests cover:
- Event bus communication patterns
- Decision synchronization protocols
- State management consistency
- Performance under concurrent operations
- Failure recovery scenarios
"""

import pytest
import asyncio
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import threading
import logging

from src.core.events import EventBus, Event, EventType
from src.core.kernel import AlgoSpaceKernel
from src.agents.strategic_marl_component import StrategicMARLComponent
from src.tactical.controller import TacticalMARLController
from src.risk.agents.risk_monitor_agent import RiskMonitorAgent
from src.xai.core.integration_interfaces import XAICoreEngineOrchestrator

# Test markers
pytestmark = [pytest.mark.coordination, pytest.mark.marl_innovation]

logger = logging.getLogger(__name__)


@dataclass
class CoordinationTestScenario:
    """Test scenario for multi-agent coordination"""
    name: str
    description: str
    agents: List[str]
    event_sequence: List[Dict[str, Any]]
    expected_coordination_pattern: Dict[str, Any]
    timeout_seconds: float = 30.0
    performance_targets: Dict[str, float] = None


@dataclass
class AgentMessage:
    """Represents a message between agents"""
    source_agent: str
    target_agent: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime
    correlation_id: str


class MultiAgentCoordinationOrchestrator:
    """
    Orchestrates multi-agent coordination tests
    
    This class manages the lifecycle of coordination tests, including:
    - Agent initialization and configuration
    - Event bus setup and monitoring
    - Coordination pattern validation
    - Performance measurement
    - Failure injection and recovery testing
    """
    
    def __init__(self):
        self.event_bus = EventBus()
        self.agents = {}
        self.coordination_history = []
        self.performance_metrics = {}
        self.active_scenarios = {}
        
        # Message tracking
        self.message_log = []
        self.coordination_patterns = {}
        
        # Performance tracking
        self.decision_latencies = {}
        self.throughput_metrics = {}
        
        logger.info("MultiAgentCoordinationOrchestrator initialized")
    
    async def initialize_agents(self, config: Dict[str, Any]) -> bool:
        """Initialize all agents for coordination testing"""
        try:
            # Initialize Strategic Agent
            strategic_config = config.get('strategic', {})
            self.agents['strategic'] = await self._create_mock_strategic_agent(strategic_config)
            
            # Initialize Tactical Agent
            tactical_config = config.get('tactical', {})
            self.agents['tactical'] = await self._create_mock_tactical_agent(tactical_config)
            
            # Initialize Risk Management Agent
            risk_config = config.get('risk', {})
            self.agents['risk'] = await self._create_mock_risk_agent(risk_config)
            
            # Initialize XAI Engine
            xai_config = config.get('xai', {})
            self.agents['xai'] = await self._create_mock_xai_agent(xai_config)
            
            # Wire event subscriptions
            await self._setup_event_subscriptions()
            
            logger.info("All agents initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            return False
    
    async def _create_mock_strategic_agent(self, config: Dict[str, Any]) -> Mock:
        """Create mock strategic agent with realistic behavior"""
        agent = Mock()
        agent.name = "strategic_agent"
        agent.timeframe = "30m"
        agent.initialized = True
        agent.performance_metrics = {'decisions_made': 0, 'avg_confidence': 0.0}
        
        # Mock decision making
        async def make_decision(synergy_data):
            await asyncio.sleep(0.1)  # Simulate processing time
            decision = {
                'action': np.random.choice(['buy', 'sell', 'hold']),
                'confidence': np.random.uniform(0.6, 0.9),
                'reasoning': 'Strategic analysis complete',
                'timestamp': datetime.now(),
                'agent': 'strategic'
            }
            
            # Publish strategic decision event
            event = self.event_bus.create_event(
                EventType.STRATEGIC_DECISION,
                decision,
                source='strategic_agent'
            )
            self.event_bus.publish(event)
            
            return decision
        
        agent.make_decision = make_decision
        return agent
    
    async def _create_mock_tactical_agent(self, config: Dict[str, Any]) -> Mock:
        """Create mock tactical agent with realistic behavior"""
        agent = Mock()
        agent.name = "tactical_agent"
        agent.timeframe = "5m"
        agent.initialized = True
        agent.performance_metrics = {'decisions_made': 0, 'avg_latency_ms': 0.0}
        
        # Mock decision making
        async def make_decision(strategic_signal):
            await asyncio.sleep(0.05)  # Faster tactical response
            decision = {
                'action': strategic_signal.get('action', 'hold'),
                'confidence': strategic_signal.get('confidence', 0.5) * 0.9,
                'execution_timing': 'immediate',
                'timestamp': datetime.now(),
                'agent': 'tactical'
            }
            
            # Publish tactical decision event
            event = self.event_bus.create_event(
                EventType.EXECUTE_TRADE,
                decision,
                source='tactical_agent'
            )
            self.event_bus.publish(event)
            
            return decision
        
        agent.make_decision = make_decision
        return agent
    
    async def _create_mock_risk_agent(self, config: Dict[str, Any]) -> Mock:
        """Create mock risk management agent with realistic behavior"""
        agent = Mock()
        agent.name = "risk_agent"
        agent.initialized = True
        agent.performance_metrics = {'risk_assessments': 0, 'alerts_generated': 0}
        
        # Mock risk assessment
        async def assess_risk(decision_data):
            await asyncio.sleep(0.02)  # Fast risk assessment
            risk_score = np.random.uniform(0.1, 0.8)
            
            assessment = {
                'risk_score': risk_score,
                'risk_level': 'high' if risk_score > 0.7 else 'medium' if risk_score > 0.4 else 'low',
                'recommendation': 'proceed' if risk_score < 0.6 else 'reduce_position',
                'timestamp': datetime.now(),
                'agent': 'risk'
            }
            
            # Publish risk assessment event
            event_type = EventType.RISK_BREACH if risk_score > 0.8 else EventType.RISK_UPDATE
            event = self.event_bus.create_event(
                event_type,
                assessment,
                source='risk_agent'
            )
            self.event_bus.publish(event)
            
            return assessment
        
        agent.assess_risk = assess_risk
        return agent
    
    async def _create_mock_xai_agent(self, config: Dict[str, Any]) -> Mock:
        """Create mock XAI engine with realistic behavior"""
        agent = Mock()
        agent.name = "xai_agent"
        agent.initialized = True
        agent.performance_metrics = {'explanations_generated': 0, 'avg_generation_time_ms': 0.0}
        
        # Mock explanation generation
        async def generate_explanation(decision_data):
            await asyncio.sleep(0.2)  # Explanation generation takes time
            explanation = {
                'explanation_text': f"Decision based on {decision_data.get('reasoning', 'analysis')}",
                'confidence_score': decision_data.get('confidence', 0.5),
                'key_factors': ['market_trend', 'risk_level', 'historical_patterns'],
                'timestamp': datetime.now(),
                'agent': 'xai'
            }
            
            # Publish explanation event
            event = self.event_bus.create_event(
                EventType.XAI_EXPLANATION_GENERATED,
                explanation,
                source='xai_agent'
            )
            self.event_bus.publish(event)
            
            return explanation
        
        agent.generate_explanation = generate_explanation
        return agent
    
    async def _setup_event_subscriptions(self):
        """Set up event subscriptions for agent coordination"""
        
        # Strategic → Tactical coordination
        def on_strategic_decision(event):
            asyncio.create_task(self._handle_strategic_decision(event))
        
        # Risk monitoring for all decisions
        def on_trade_execution(event):
            asyncio.create_task(self._handle_trade_execution(event))
        
        # XAI explanations for critical decisions
        def on_critical_decision(event):
            asyncio.create_task(self._handle_critical_decision(event))
        
        # Set up subscriptions
        self.event_bus.subscribe(EventType.STRATEGIC_DECISION, on_strategic_decision)
        self.event_bus.subscribe(EventType.EXECUTE_TRADE, on_trade_execution)
        self.event_bus.subscribe(EventType.RISK_BREACH, on_critical_decision)
        
        # Track coordination messages
        self.event_bus.subscribe(EventType.STRATEGIC_DECISION, self._log_coordination_message)
        self.event_bus.subscribe(EventType.EXECUTE_TRADE, self._log_coordination_message)
        self.event_bus.subscribe(EventType.RISK_UPDATE, self._log_coordination_message)
        self.event_bus.subscribe(EventType.XAI_EXPLANATION_GENERATED, self._log_coordination_message)
    
    async def _handle_strategic_decision(self, event):
        """Handle strategic decision and trigger tactical response"""
        try:
            if 'tactical' in self.agents:
                await self.agents['tactical'].make_decision(event.payload)
                
                # Record coordination
                self.coordination_history.append({
                    'type': 'strategic_to_tactical',
                    'timestamp': datetime.now(),
                    'source_event': event.event_type.value,
                    'payload': event.payload
                })
                
        except Exception as e:
            logger.error(f"Error in strategic→tactical coordination: {e}")
    
    async def _handle_trade_execution(self, event):
        """Handle trade execution and trigger risk assessment"""
        try:
            if 'risk' in self.agents:
                await self.agents['risk'].assess_risk(event.payload)
                
                # Record coordination
                self.coordination_history.append({
                    'type': 'tactical_to_risk',
                    'timestamp': datetime.now(),
                    'source_event': event.event_type.value,
                    'payload': event.payload
                })
                
        except Exception as e:
            logger.error(f"Error in tactical→risk coordination: {e}")
    
    async def _handle_critical_decision(self, event):
        """Handle critical decisions and trigger XAI explanation"""
        try:
            if 'xai' in self.agents:
                await self.agents['xai'].generate_explanation(event.payload)
                
                # Record coordination
                self.coordination_history.append({
                    'type': 'critical_to_xai',
                    'timestamp': datetime.now(),
                    'source_event': event.event_type.value,
                    'payload': event.payload
                })
                
        except Exception as e:
            logger.error(f"Error in critical→XAI coordination: {e}")
    
    def _log_coordination_message(self, event):
        """Log coordination messages for analysis"""
        message = AgentMessage(
            source_agent=event.source,
            target_agent='broadcast',
            message_type=event.event_type.value,
            payload=event.payload,
            timestamp=event.timestamp,
            correlation_id=str(id(event))
        )
        
        self.message_log.append(message)
    
    async def run_coordination_scenario(self, scenario: CoordinationTestScenario) -> Dict[str, Any]:
        """Run a coordination test scenario"""
        logger.info(f"Running coordination scenario: {scenario.name}")
        
        start_time = time.time()
        results = {
            'scenario_name': scenario.name,
            'start_time': start_time,
            'success': False,
            'coordination_events': [],
            'performance_metrics': {},
            'errors': []
        }
        
        try:
            # Clear previous coordination history
            self.coordination_history.clear()
            self.message_log.clear()
            
            # Execute event sequence
            for event_spec in scenario.event_sequence:
                await self._execute_event(event_spec)
                await asyncio.sleep(0.1)  # Small delay between events
            
            # Wait for coordination to complete
            await asyncio.sleep(1.0)
            
            # Analyze coordination patterns
            coordination_analysis = self._analyze_coordination_patterns(scenario)
            results['coordination_events'] = coordination_analysis
            
            # Calculate performance metrics
            results['performance_metrics'] = self._calculate_performance_metrics()
            
            # Validate expected coordination pattern
            if self._validate_coordination_pattern(scenario.expected_coordination_pattern):
                results['success'] = True
            
            results['duration_seconds'] = time.time() - start_time
            
        except Exception as e:
            logger.error(f"Error in coordination scenario {scenario.name}: {e}")
            results['errors'].append(str(e))
        
        return results
    
    async def _execute_event(self, event_spec: Dict[str, Any]):
        """Execute a single event in the coordination scenario"""
        event_type = EventType(event_spec['type'])
        payload = event_spec.get('payload', {})
        source = event_spec.get('source', 'test_orchestrator')
        
        event = self.event_bus.create_event(event_type, payload, source)
        self.event_bus.publish(event)
    
    def _analyze_coordination_patterns(self, scenario: CoordinationTestScenario) -> List[Dict[str, Any]]:
        """Analyze coordination patterns from the scenario execution"""
        patterns = []
        
        # Group coordination events by type
        coordination_types = {}
        for coord in self.coordination_history:
            coord_type = coord['type']
            if coord_type not in coordination_types:
                coordination_types[coord_type] = []
            coordination_types[coord_type].append(coord)
        
        # Analyze each coordination type
        for coord_type, events in coordination_types.items():
            pattern = {
                'type': coord_type,
                'count': len(events),
                'avg_latency_ms': 0.0,
                'success_rate': 1.0
            }
            
            # Calculate timing metrics
            if len(events) > 1:
                latencies = []
                for i in range(1, len(events)):
                    latency = (events[i]['timestamp'] - events[i-1]['timestamp']).total_seconds() * 1000
                    latencies.append(latency)
                
                pattern['avg_latency_ms'] = np.mean(latencies)
                pattern['max_latency_ms'] = np.max(latencies)
                pattern['min_latency_ms'] = np.min(latencies)
            
            patterns.append(pattern)
        
        return patterns
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics for the coordination test"""
        metrics = {
            'total_coordination_events': len(self.coordination_history),
            'total_messages': len(self.message_log),
            'coordination_types': len(set(coord['type'] for coord in self.coordination_history)),
            'avg_coordination_latency_ms': 0.0,
            'message_throughput_per_second': 0.0
        }
        
        # Calculate average coordination latency
        if len(self.coordination_history) > 0:
            start_time = self.coordination_history[0]['timestamp']
            end_time = self.coordination_history[-1]['timestamp']
            total_duration = (end_time - start_time).total_seconds()
            
            if total_duration > 0:
                metrics['avg_coordination_latency_ms'] = (total_duration / len(self.coordination_history)) * 1000
                metrics['message_throughput_per_second'] = len(self.message_log) / total_duration
        
        return metrics
    
    def _validate_coordination_pattern(self, expected_pattern: Dict[str, Any]) -> bool:
        """Validate that the coordination pattern matches expectations"""
        if not expected_pattern:
            return True
        
        # Check minimum coordination events
        min_events = expected_pattern.get('min_coordination_events', 0)
        if len(self.coordination_history) < min_events:
            return False
        
        # Check required coordination types
        required_types = expected_pattern.get('required_coordination_types', [])
        actual_types = set(coord['type'] for coord in self.coordination_history)
        
        for required_type in required_types:
            if required_type not in actual_types:
                return False
        
        # Check maximum latency
        max_latency = expected_pattern.get('max_latency_ms', float('inf'))
        if len(self.coordination_history) > 1:
            for i in range(1, len(self.coordination_history)):
                latency = (self.coordination_history[i]['timestamp'] - 
                          self.coordination_history[i-1]['timestamp']).total_seconds() * 1000
                if latency > max_latency:
                    return False
        
        return True
    
    async def test_concurrent_coordination(self, num_concurrent_scenarios: int = 5) -> Dict[str, Any]:
        """Test coordination under concurrent load"""
        logger.info(f"Testing concurrent coordination with {num_concurrent_scenarios} scenarios")
        
        # Create multiple concurrent scenarios
        scenarios = []
        for i in range(num_concurrent_scenarios):
            scenario = CoordinationTestScenario(
                name=f"concurrent_test_{i}",
                description=f"Concurrent coordination test {i}",
                agents=['strategic', 'tactical', 'risk', 'xai'],
                event_sequence=[
                    {
                        'type': 'SYNERGY_DETECTED',
                        'payload': {'synergy_type': 'TYPE_1', 'confidence': 0.8},
                        'source': f'test_{i}'
                    }
                ],
                expected_coordination_pattern={
                    'min_coordination_events': 2,
                    'required_coordination_types': ['strategic_to_tactical', 'tactical_to_risk'],
                    'max_latency_ms': 500
                }
            )
            scenarios.append(scenario)
        
        # Run scenarios concurrently
        start_time = time.time()
        tasks = [self.run_coordination_scenario(scenario) for scenario in scenarios]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze concurrent performance
        concurrent_metrics = {
            'total_scenarios': num_concurrent_scenarios,
            'successful_scenarios': sum(1 for r in results if isinstance(r, dict) and r.get('success', False)),
            'failed_scenarios': sum(1 for r in results if isinstance(r, Exception) or not r.get('success', False)),
            'total_duration_seconds': time.time() - start_time,
            'avg_scenario_duration_seconds': np.mean([r['duration_seconds'] for r in results if isinstance(r, dict)]),
            'coordination_conflicts': self._detect_coordination_conflicts()
        }
        
        return {
            'concurrent_metrics': concurrent_metrics,
            'scenario_results': results
        }
    
    def _detect_coordination_conflicts(self) -> List[Dict[str, Any]]:
        """Detect potential coordination conflicts"""
        conflicts = []
        
        # Check for rapid sequential coordination events that might conflict
        for i in range(1, len(self.coordination_history)):
            current = self.coordination_history[i]
            previous = self.coordination_history[i-1]
            
            time_diff = (current['timestamp'] - previous['timestamp']).total_seconds() * 1000
            
            # Flag as potential conflict if events are too close together
            if time_diff < 10:  # 10ms threshold
                conflicts.append({
                    'type': 'rapid_sequential_coordination',
                    'time_difference_ms': time_diff,
                    'event1': previous,
                    'event2': current
                })
        
        return conflicts
    
    async def test_failure_recovery(self, failure_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test coordination recovery from agent failures"""
        logger.info(f"Testing failure recovery with {len(failure_scenarios)} failure scenarios")
        
        recovery_results = []
        
        for scenario in failure_scenarios:
            result = await self._test_single_failure_scenario(scenario)
            recovery_results.append(result)
        
        return {
            'failure_scenarios_tested': len(failure_scenarios),
            'recovery_results': recovery_results,
            'overall_recovery_success_rate': sum(1 for r in recovery_results if r['recovered']) / len(recovery_results)
        }
    
    async def _test_single_failure_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single failure scenario"""
        failed_agent = scenario['failed_agent']
        failure_type = scenario['failure_type']
        
        logger.info(f"Testing failure scenario: {failed_agent} - {failure_type}")
        
        # Simulate agent failure
        if failed_agent in self.agents:
            if failure_type == 'timeout':
                # Make agent very slow
                original_method = self.agents[failed_agent].make_decision
                async def slow_decision(*args, **kwargs):
                    await asyncio.sleep(10)  # Simulate timeout
                    return await original_method(*args, **kwargs)
                self.agents[failed_agent].make_decision = slow_decision
            
            elif failure_type == 'exception':
                # Make agent throw exceptions
                async def failing_decision(*args, **kwargs):
                    raise Exception(f"Simulated failure in {failed_agent}")
                self.agents[failed_agent].make_decision = failing_decision
            
            elif failure_type == 'disconnect':
                # Remove agent from coordination
                self.agents[failed_agent].initialized = False
        
        # Test coordination with failed agent
        test_scenario = CoordinationTestScenario(
            name=f"failure_test_{failed_agent}_{failure_type}",
            description=f"Test coordination with {failed_agent} failure",
            agents=['strategic', 'tactical', 'risk', 'xai'],
            event_sequence=[
                {
                    'type': 'SYNERGY_DETECTED',
                    'payload': {'synergy_type': 'TYPE_1', 'confidence': 0.8},
                    'source': 'failure_test'
                }
            ],
            expected_coordination_pattern={
                'min_coordination_events': 1,  # Reduced expectations due to failure
                'max_latency_ms': 1000
            }
        )
        
        # Run the test
        start_time = time.time()
        result = await self.run_coordination_scenario(test_scenario)
        
        # Analyze recovery
        recovered = self._analyze_recovery_behavior(result, failed_agent)
        
        return {
            'failed_agent': failed_agent,
            'failure_type': failure_type,
            'recovered': recovered,
            'coordination_maintained': result.get('success', False),
            'recovery_time_seconds': time.time() - start_time
        }
    
    def _analyze_recovery_behavior(self, result: Dict[str, Any], failed_agent: str) -> bool:
        """Analyze if the system recovered from agent failure"""
        # Check if other agents continued coordinating
        coordination_events = result.get('coordination_events', [])
        
        # Recovery is successful if:
        # 1. At least some coordination still occurred
        # 2. No cascade failures (other agents still responding)
        # 3. System didn't completely freeze
        
        if len(coordination_events) > 0:
            # Check if coordination types not involving failed agent still worked
            working_types = [event['type'] for event in coordination_events]
            
            # Define coordination types that should work even with specific agent failures
            expected_working_types = {
                'strategic': ['tactical_to_risk', 'critical_to_xai'],
                'tactical': ['strategic_to_tactical', 'critical_to_xai'],
                'risk': ['strategic_to_tactical', 'critical_to_xai'],
                'xai': ['strategic_to_tactical', 'tactical_to_risk']
            }
            
            expected_for_failure = expected_working_types.get(failed_agent, [])
            working_expected = any(coord_type in working_types for coord_type in expected_for_failure)
            
            return working_expected
        
        return False
    
    def get_coordination_summary(self) -> Dict[str, Any]:
        """Get comprehensive coordination summary"""
        return {
            'total_coordination_events': len(self.coordination_history),
            'total_messages': len(self.message_log),
            'coordination_types': list(set(coord['type'] for coord in self.coordination_history)),
            'agent_participation': {
                agent_name: sum(1 for msg in self.message_log if msg.source_agent == agent_name)
                for agent_name in self.agents.keys()
            },
            'message_timeline': [
                {
                    'timestamp': msg.timestamp.isoformat(),
                    'source': msg.source_agent,
                    'type': msg.message_type
                }
                for msg in self.message_log[-20:]  # Last 20 messages
            ]
        }


class TestMultiAgentCoordination:
    """Test suite for multi-agent coordination"""
    
    @pytest.fixture
    async def coordination_orchestrator(self):
        """Set up coordination orchestrator for testing"""
        orchestrator = MultiAgentCoordinationOrchestrator()
        
        config = {
            'strategic': {'timeframe': '30m', 'confidence_threshold': 0.7},
            'tactical': {'timeframe': '5m', 'latency_target_ms': 100},
            'risk': {'risk_threshold': 0.8, 'emergency_threshold': 0.9},
            'xai': {'explanation_timeout_ms': 500}
        }
        
        success = await orchestrator.initialize_agents(config)
        assert success, "Failed to initialize agents"
        
        yield orchestrator
        
        # Cleanup
        await orchestrator.event_bus.stop()
    
    @pytest.mark.asyncio
    async def test_strategic_tactical_coordination(self, coordination_orchestrator):
        """Test Strategic → Tactical agent coordination"""
        scenario = CoordinationTestScenario(
            name="strategic_tactical_coordination",
            description="Test strategic agent triggering tactical response",
            agents=['strategic', 'tactical'],
            event_sequence=[
                {
                    'type': 'SYNERGY_DETECTED',
                    'payload': {
                        'synergy_type': 'TYPE_1_BULLISH',
                        'confidence': 0.85,
                        'matrix_data': np.random.randn(48, 13).tolist()
                    },
                    'source': 'synergy_detector'
                }
            ],
            expected_coordination_pattern={
                'min_coordination_events': 2,
                'required_coordination_types': ['strategic_to_tactical', 'tactical_to_risk'],
                'max_latency_ms': 200
            }
        )
        
        result = await coordination_orchestrator.run_coordination_scenario(scenario)
        
        assert result['success'], "Strategic-Tactical coordination failed"
        assert len(result['coordination_events']) >= 2, "Insufficient coordination events"
        
        # Verify coordination chain
        coordination_types = [event['type'] for event in result['coordination_events']]
        assert 'strategic_to_tactical' in coordination_types
        assert 'tactical_to_risk' in coordination_types
    
    @pytest.mark.asyncio
    async def test_risk_management_intervention(self, coordination_orchestrator):
        """Test Risk Management agent intervention in coordination"""
        scenario = CoordinationTestScenario(
            name="risk_intervention",
            description="Test risk management intervention",
            agents=['strategic', 'tactical', 'risk'],
            event_sequence=[
                {
                    'type': 'SYNERGY_DETECTED',
                    'payload': {
                        'synergy_type': 'TYPE_1_BULLISH',
                        'confidence': 0.9,
                        'risk_score': 0.85  # High risk
                    },
                    'source': 'synergy_detector'
                }
            ],
            expected_coordination_pattern={
                'min_coordination_events': 3,
                'required_coordination_types': ['strategic_to_tactical', 'tactical_to_risk'],
                'max_latency_ms': 300
            }
        )
        
        result = await coordination_orchestrator.run_coordination_scenario(scenario)
        
        assert result['success'], "Risk intervention coordination failed"
        
        # Check that risk agent was involved
        coordination_types = [event['type'] for event in result['coordination_events']]
        assert 'tactical_to_risk' in coordination_types
    
    @pytest.mark.asyncio
    async def test_xai_explanation_coordination(self, coordination_orchestrator):
        """Test XAI explanation coordination for critical decisions"""
        scenario = CoordinationTestScenario(
            name="xai_explanation",
            description="Test XAI explanation for critical decisions",
            agents=['strategic', 'tactical', 'risk', 'xai'],
            event_sequence=[
                {
                    'type': 'RISK_BREACH',
                    'payload': {
                        'risk_score': 0.92,
                        'risk_type': 'correlation_spike',
                        'decision_context': {
                            'action': 'emergency_stop',
                            'confidence': 0.95
                        }
                    },
                    'source': 'risk_monitor'
                }
            ],
            expected_coordination_pattern={
                'min_coordination_events': 1,
                'required_coordination_types': ['critical_to_xai'],
                'max_latency_ms': 600
            }
        )
        
        result = await coordination_orchestrator.run_coordination_scenario(scenario)
        
        assert result['success'], "XAI explanation coordination failed"
        
        # Check that XAI agent generated explanation
        coordination_types = [event['type'] for event in result['coordination_events']]
        assert 'critical_to_xai' in coordination_types
    
    @pytest.mark.asyncio
    async def test_concurrent_coordination_load(self, coordination_orchestrator):
        """Test coordination under concurrent load"""
        result = await coordination_orchestrator.test_concurrent_coordination(num_concurrent_scenarios=3)
        
        concurrent_metrics = result['concurrent_metrics']
        
        # Validate concurrent performance
        assert concurrent_metrics['successful_scenarios'] >= 2, "Too many concurrent failures"
        assert concurrent_metrics['total_duration_seconds'] < 10, "Concurrent coordination too slow"
        
        # Check for coordination conflicts
        conflicts = concurrent_metrics['coordination_conflicts']
        assert len(conflicts) <= 1, "Too many coordination conflicts detected"
    
    @pytest.mark.asyncio
    async def test_agent_failure_recovery(self, coordination_orchestrator):
        """Test coordination recovery from agent failures"""
        failure_scenarios = [
            {'failed_agent': 'tactical', 'failure_type': 'timeout'},
            {'failed_agent': 'risk', 'failure_type': 'exception'},
            {'failed_agent': 'xai', 'failure_type': 'disconnect'}
        ]
        
        result = await coordination_orchestrator.test_failure_recovery(failure_scenarios)
        
        assert result['failure_scenarios_tested'] == 3
        assert result['overall_recovery_success_rate'] >= 0.6, "Recovery success rate too low"
        
        # Check individual recovery results
        for recovery_result in result['recovery_results']:
            assert recovery_result['recovery_time_seconds'] < 5, "Recovery time too long"
    
    @pytest.mark.asyncio
    async def test_coordination_performance_benchmarks(self, coordination_orchestrator):
        """Test coordination performance benchmarks"""
        # Run multiple scenarios to establish performance baseline
        scenarios = []
        for i in range(5):
            scenario = CoordinationTestScenario(
                name=f"performance_test_{i}",
                description=f"Performance benchmark {i}",
                agents=['strategic', 'tactical', 'risk', 'xai'],
                event_sequence=[
                    {
                        'type': 'SYNERGY_DETECTED',
                        'payload': {'synergy_type': 'TYPE_1', 'confidence': 0.8},
                        'source': f'perf_test_{i}'
                    }
                ],
                expected_coordination_pattern={
                    'min_coordination_events': 2,
                    'max_latency_ms': 100
                }
            )
            scenarios.append(scenario)
        
        # Run scenarios sequentially to measure individual performance
        results = []
        for scenario in scenarios:
            result = await coordination_orchestrator.run_coordination_scenario(scenario)
            results.append(result)
        
        # Analyze performance
        avg_duration = np.mean([r['duration_seconds'] for r in results])
        avg_coordination_events = np.mean([len(r['coordination_events']) for r in results])
        
        # Performance assertions
        assert avg_duration < 2.0, f"Average coordination duration too high: {avg_duration}s"
        assert avg_coordination_events >= 2, f"Too few coordination events: {avg_coordination_events}"
        
        # Check coordination latency
        all_latencies = []
        for result in results:
            for event in result['coordination_events']:
                if 'avg_latency_ms' in event:
                    all_latencies.append(event['avg_latency_ms'])
        
        if all_latencies:
            avg_latency = np.mean(all_latencies)
            assert avg_latency < 150, f"Average coordination latency too high: {avg_latency}ms"
    
    @pytest.mark.asyncio
    async def test_coordination_state_consistency(self, coordination_orchestrator):
        """Test state consistency across agents during coordination"""
        scenario = CoordinationTestScenario(
            name="state_consistency",
            description="Test state consistency across agents",
            agents=['strategic', 'tactical', 'risk', 'xai'],
            event_sequence=[
                {
                    'type': 'SYNERGY_DETECTED',
                    'payload': {
                        'synergy_type': 'TYPE_1_BULLISH',
                        'confidence': 0.8,
                        'correlation_id': 'test_consistency_123'
                    },
                    'source': 'synergy_detector'
                }
            ],
            expected_coordination_pattern={
                'min_coordination_events': 2,
                'required_coordination_types': ['strategic_to_tactical', 'tactical_to_risk'],
                'max_latency_ms': 200
            }
        )
        
        result = await coordination_orchestrator.run_coordination_scenario(scenario)
        
        assert result['success'], "State consistency test failed"
        
        # Verify correlation ID propagation
        message_log = coordination_orchestrator.message_log
        correlation_ids = [msg.correlation_id for msg in message_log if hasattr(msg, 'correlation_id')]
        
        # Check that correlation context is maintained
        assert len(set(correlation_ids)) <= 2, "Too many correlation IDs - state inconsistency"
    
    def test_coordination_summary_generation(self, coordination_orchestrator):
        """Test coordination summary generation"""
        # Add some mock coordination history
        coordination_orchestrator.coordination_history = [
            {'type': 'strategic_to_tactical', 'timestamp': datetime.now()},
            {'type': 'tactical_to_risk', 'timestamp': datetime.now()},
            {'type': 'critical_to_xai', 'timestamp': datetime.now()}
        ]
        
        summary = coordination_orchestrator.get_coordination_summary()
        
        assert 'total_coordination_events' in summary
        assert 'coordination_types' in summary
        assert 'agent_participation' in summary
        assert 'message_timeline' in summary
        
        assert summary['total_coordination_events'] == 3
        assert len(summary['coordination_types']) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])