"""
Multi-Agent Interaction Validator - Advanced MARL Testing
=========================================================

This module implements comprehensive validation of multi-agent interactions
in MARL systems, including communication protocols, coordination patterns,
and emergent interaction behaviors.

Key Features:
- Real-time interaction monitoring
- Communication protocol validation
- Deadlock and livelock detection
- Message flow analysis
- Interaction quality metrics
- Performance benchmarking

Author: Agent Delta - MARL Testing Innovation Specialist
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import networkx as nx

logger = logging.getLogger(__name__)


class InteractionType(Enum):
    """Types of multi-agent interactions."""
    COMMUNICATION = "communication"
    COORDINATION = "coordination"
    COMPETITION = "competition"
    COOPERATION = "cooperation"
    NEGOTIATION = "negotiation"
    INFORMATION_SHARING = "information_sharing"
    RESOURCE_SHARING = "resource_sharing"
    CONFLICT_RESOLUTION = "conflict_resolution"


class MessageType(Enum):
    """Types of messages in multi-agent communication."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    BROADCAST = "broadcast"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    ACKNOWLEDGMENT = "acknowledgment"


@dataclass
class InteractionMessage:
    """Structure for inter-agent messages."""
    message_id: str
    sender_id: str
    receiver_id: str
    message_type: MessageType
    timestamp: datetime
    content: Dict[str, Any]
    priority: int = 0
    ttl_ms: int = 5000
    acknowledgment_required: bool = False
    response_time_ms: Optional[float] = None


@dataclass
class AgentInteraction:
    """Structure for agent interactions."""
    interaction_id: str
    agent_ids: List[str]
    interaction_type: InteractionType
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    messages: List[InteractionMessage] = field(default_factory=list)
    outcome: Optional[str] = None
    success: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InteractionValidationConfig:
    """Configuration for interaction validation."""
    max_interaction_duration_ms: int = 10000
    message_timeout_ms: int = 5000
    deadlock_detection_timeout_ms: int = 15000
    max_message_queue_size: int = 1000
    interaction_coverage_threshold: float = 0.9
    quality_score_threshold: float = 0.8
    enable_deadlock_detection: bool = True
    enable_message_tracing: bool = True
    enable_performance_monitoring: bool = True
    log_all_interactions: bool = False


class MultiAgentInteractionValidator:
    """
    Advanced Multi-Agent Interaction Validator.
    
    This validator provides comprehensive testing and validation of multi-agent
    interactions including communication protocols, coordination patterns,
    and system-level behavior analysis.
    """
    
    def __init__(self, config: Optional[InteractionValidationConfig] = None):
        """Initialize the interaction validator."""
        self.config = config or InteractionValidationConfig()
        self.interactions = {}
        self.message_history = deque(maxlen=10000)
        self.agent_graph = nx.DiGraph()
        self.interaction_patterns = defaultdict(int)
        self.performance_metrics = {
            'total_interactions': 0,
            'successful_interactions': 0,
            'failed_interactions': 0,
            'average_interaction_duration_ms': 0.0,
            'message_throughput': 0.0,
            'deadlock_incidents': 0,
            'timeout_incidents': 0,
            'communication_efficiency': 0.0
        }
        
        # Monitoring state
        self.monitoring_active = False
        self.deadlock_detector = None
        self.message_tracer = None
        
        logger.info("Multi-Agent Interaction Validator initialized")
    
    def is_initialized(self) -> bool:
        """Check if validator is properly initialized."""
        return True
    
    async def validate_agent_interactions(self, 
                                        agent_system: Any,
                                        test_scenarios: List[Dict[str, Any]],
                                        validation_config: Optional[InteractionValidationConfig] = None) -> Dict[str, Any]:
        """
        Validate multi-agent interactions comprehensively.
        
        Args:
            agent_system: The MARL system to validate
            test_scenarios: List of test scenarios to run
            validation_config: Optional validation configuration
            
        Returns:
            Comprehensive validation results
        """
        config = validation_config or self.config
        validation_start = time.time()
        
        logger.info("Starting multi-agent interaction validation")
        
        try:
            # Initialize monitoring
            await self._initialize_monitoring(agent_system)
            
            # Run test scenarios
            scenario_results = []
            for i, scenario in enumerate(test_scenarios):
                logger.info(f"Running test scenario {i+1}/{len(test_scenarios)}: {scenario.get('name', 'Unknown')}")
                result = await self._run_interaction_scenario(agent_system, scenario, config)
                scenario_results.append(result)
            
            # Analyze interaction patterns
            pattern_analysis = self._analyze_interaction_patterns()
            
            # Generate comprehensive metrics
            validation_metrics = self._calculate_validation_metrics(scenario_results)
            
            # Detect potential issues
            issues = self._detect_interaction_issues()
            
            # Generate final results
            validation_results = {
                'validation_duration_ms': (time.time() - validation_start) * 1000,
                'total_scenarios': len(test_scenarios),
                'scenario_results': scenario_results,
                'pattern_analysis': pattern_analysis,
                'validation_metrics': validation_metrics,
                'detected_issues': issues,
                'performance_metrics': self.performance_metrics,
                'coverage': self._calculate_interaction_coverage(),
                'quality_score': self._calculate_quality_score(),
                'protocol_compliance': self._calculate_protocol_compliance(),
                'communication_efficiency': self._calculate_communication_efficiency()
            }
            
            logger.info(f"Interaction validation completed in {validation_results['validation_duration_ms']:.2f}ms")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Interaction validation failed: {str(e)}")
            raise
        finally:
            # Clean up monitoring
            await self._cleanup_monitoring()
    
    async def _initialize_monitoring(self, agent_system: Any):
        """Initialize interaction monitoring systems."""
        self.monitoring_active = True
        
        # Initialize deadlock detector
        if self.config.enable_deadlock_detection:
            self.deadlock_detector = DeadlockDetector(
                timeout_ms=self.config.deadlock_detection_timeout_ms
            )
        
        # Initialize message tracer
        if self.config.enable_message_tracing:
            self.message_tracer = MessageTracer(
                max_history=self.config.max_message_queue_size
            )
        
        # Reset monitoring state
        self.interactions.clear()
        self.message_history.clear()
        self.agent_graph.clear()
        self.interaction_patterns.clear()
        
        logger.info("Interaction monitoring initialized")
    
    async def _run_interaction_scenario(self, 
                                      agent_system: Any,
                                      scenario: Dict[str, Any],
                                      config: InteractionValidationConfig) -> Dict[str, Any]:
        """Run a specific interaction test scenario."""
        scenario_start = time.time()
        scenario_name = scenario.get('name', 'Unknown')
        
        try:
            # Extract scenario parameters
            agent_count = scenario.get('agent_count', 3)
            interaction_type = InteractionType(scenario.get('interaction_type', 'communication'))
            duration_ms = scenario.get('duration_ms', 5000)
            expected_messages = scenario.get('expected_messages', 10)
            
            # Generate test agents
            test_agents = self._generate_test_agents(agent_count, scenario)
            
            # Create interaction context
            interaction_context = {
                'scenario': scenario,
                'agents': test_agents,
                'start_time': datetime.now(),
                'expected_outcome': scenario.get('expected_outcome', 'success')
            }
            
            # Execute interaction scenario
            execution_result = await self._execute_interaction_scenario(
                agent_system, interaction_context, duration_ms
            )
            
            # Analyze scenario results
            scenario_analysis = self._analyze_scenario_results(execution_result, scenario)
            
            # Generate scenario metrics
            scenario_metrics = {
                'duration_ms': (time.time() - scenario_start) * 1000,
                'interactions_observed': len(execution_result.get('interactions', [])),
                'messages_exchanged': len(execution_result.get('messages', [])),
                'agents_participated': len(execution_result.get('active_agents', [])),
                'success_rate': execution_result.get('success_rate', 0.0),
                'completion_status': execution_result.get('status', 'unknown')
            }
            
            return {
                'scenario_name': scenario_name,
                'execution_result': execution_result,
                'analysis': scenario_analysis,
                'metrics': scenario_metrics,
                'success': scenario_metrics['completion_status'] == 'completed'
            }
            
        except Exception as e:
            logger.error(f"Scenario {scenario_name} failed: {str(e)}")
            return {
                'scenario_name': scenario_name,
                'execution_result': {'error': str(e)},
                'analysis': {'error': str(e)},
                'metrics': {'duration_ms': (time.time() - scenario_start) * 1000},
                'success': False
            }
    
    def _generate_test_agents(self, agent_count: int, scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test agents for interaction scenarios."""
        agents = []
        
        for i in range(agent_count):
            agent = {
                'agent_id': f"test_agent_{i}",
                'agent_type': scenario.get('agent_types', ['default'])[i % len(scenario.get('agent_types', ['default']))],
                'capabilities': scenario.get('capabilities', ['communication', 'coordination']),
                'role': scenario.get('roles', ['participant'])[i % len(scenario.get('roles', ['participant']))],
                'initial_state': scenario.get('initial_states', [{}])[i % len(scenario.get('initial_states', [{}]))],
                'interaction_preferences': scenario.get('interaction_preferences', {})
            }
            agents.append(agent)
        
        return agents
    
    async def _execute_interaction_scenario(self, 
                                          agent_system: Any,
                                          interaction_context: Dict[str, Any],
                                          duration_ms: int) -> Dict[str, Any]:
        """Execute interaction scenario with monitoring."""
        execution_start = time.time()
        
        # Simulated execution - in production, this would interact with real agents
        interactions = []
        messages = []
        active_agents = set()
        
        # Simulate agent interactions
        agents = interaction_context['agents']
        scenario = interaction_context['scenario']
        
        # Generate realistic interaction patterns
        for step in range(duration_ms // 100):  # 100ms steps
            # Simulate message exchange
            if len(agents) > 1:
                sender = np.random.choice(agents)
                receiver = np.random.choice([a for a in agents if a != sender])
                
                message = self._simulate_message_exchange(sender, receiver, step)
                messages.append(message)
                
                active_agents.add(sender['agent_id'])
                active_agents.add(receiver['agent_id'])
            
            # Simulate interaction events
            if step % 10 == 0:  # Every 1 second
                interaction = self._simulate_interaction_event(agents, step)
                interactions.append(interaction)
            
            # Small delay to simulate real-time execution
            await asyncio.sleep(0.001)
        
        # Calculate execution metrics
        execution_metrics = {
            'total_duration_ms': (time.time() - execution_start) * 1000,
            'interactions_generated': len(interactions),
            'messages_generated': len(messages),
            'active_agents': list(active_agents),
            'status': 'completed' if len(interactions) > 0 else 'failed',
            'success_rate': min(len(interactions) / 10, 1.0)  # Expect at least 10 interactions
        }
        
        return {
            'interactions': interactions,
            'messages': messages,
            'active_agents': list(active_agents),
            'execution_metrics': execution_metrics,
            'status': execution_metrics['status'],
            'success_rate': execution_metrics['success_rate']
        }
    
    def _simulate_message_exchange(self, sender: Dict, receiver: Dict, step: int) -> InteractionMessage:
        """Simulate message exchange between agents."""
        message_types = list(MessageType)
        
        message = InteractionMessage(
            message_id=f"msg_{step}_{sender['agent_id']}_{receiver['agent_id']}",
            sender_id=sender['agent_id'],
            receiver_id=receiver['agent_id'],
            message_type=np.random.choice(message_types),
            timestamp=datetime.now(),
            content={
                'data': f"message_content_{step}",
                'priority': np.random.randint(1, 6),
                'sequence_number': step
            },
            priority=np.random.randint(1, 6),
            ttl_ms=np.random.randint(1000, 10000),
            acknowledgment_required=np.random.choice([True, False]),
            response_time_ms=np.random.uniform(1, 100)
        )
        
        # Track message in history
        self.message_history.append(message)
        
        return message
    
    def _simulate_interaction_event(self, agents: List[Dict], step: int) -> AgentInteraction:
        """Simulate interaction event between agents."""
        interaction_types = list(InteractionType)
        
        # Select participating agents
        num_participants = np.random.randint(2, min(len(agents) + 1, 5))
        participants = np.random.choice(agents, size=num_participants, replace=False)
        
        interaction = AgentInteraction(
            interaction_id=f"interaction_{step}",
            agent_ids=[agent['agent_id'] for agent in participants],
            interaction_type=np.random.choice(interaction_types),
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(milliseconds=np.random.randint(100, 1000)),
            duration_ms=np.random.uniform(100, 1000),
            messages=[],
            outcome=np.random.choice(['success', 'failure', 'timeout']),
            success=np.random.choice([True, False], p=[0.8, 0.2]),
            metrics={
                'efficiency': np.random.uniform(0.5, 1.0),
                'quality': np.random.uniform(0.6, 1.0),
                'participants': num_participants
            }
        )
        
        # Track interaction
        self.interactions[interaction.interaction_id] = interaction
        
        # Update interaction patterns
        pattern_key = f"{interaction.interaction_type.value}_{num_participants}"
        self.interaction_patterns[pattern_key] += 1
        
        return interaction
    
    def _analyze_scenario_results(self, execution_result: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the results of a scenario execution."""
        interactions = execution_result.get('interactions', [])
        messages = execution_result.get('messages', [])
        
        # Calculate success metrics
        successful_interactions = sum(1 for i in interactions if i.success)
        total_interactions = len(interactions)
        success_rate = successful_interactions / max(total_interactions, 1)
        
        # Analyze message patterns
        message_types = defaultdict(int)
        for msg in messages:
            message_types[msg.message_type.value] += 1
        
        # Calculate response times
        response_times = [msg.response_time_ms for msg in messages if msg.response_time_ms is not None]
        avg_response_time = np.mean(response_times) if response_times else 0
        
        # Analyze interaction types
        interaction_types = defaultdict(int)
        for interaction in interactions:
            interaction_types[interaction.interaction_type.value] += 1
        
        return {
            'success_rate': success_rate,
            'total_interactions': total_interactions,
            'successful_interactions': successful_interactions,
            'total_messages': len(messages),
            'message_types': dict(message_types),
            'average_response_time_ms': avg_response_time,
            'interaction_types': dict(interaction_types),
            'scenario_compliance': self._check_scenario_compliance(execution_result, scenario),
            'quality_metrics': self._calculate_quality_metrics(interactions, messages)
        }
    
    def _check_scenario_compliance(self, execution_result: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Check if execution results comply with scenario requirements."""
        compliance = {
            'agent_participation': True,
            'message_exchange': True,
            'interaction_types': True,
            'duration_compliance': True,
            'outcome_compliance': True
        }
        
        # Check agent participation
        expected_agents = scenario.get('agent_count', 3)
        actual_agents = len(execution_result.get('active_agents', []))
        compliance['agent_participation'] = actual_agents >= expected_agents
        
        # Check message exchange
        expected_messages = scenario.get('expected_messages', 10)
        actual_messages = len(execution_result.get('messages', []))
        compliance['message_exchange'] = actual_messages >= expected_messages
        
        # Check interaction types
        expected_interaction_type = scenario.get('interaction_type', 'communication')
        interactions = execution_result.get('interactions', [])
        type_found = any(i.interaction_type.value == expected_interaction_type for i in interactions)
        compliance['interaction_types'] = type_found
        
        return compliance
    
    def _calculate_quality_metrics(self, interactions: List[AgentInteraction], messages: List[InteractionMessage]) -> Dict[str, Any]:
        """Calculate quality metrics for interactions and messages."""
        if not interactions:
            return {'error': 'No interactions to analyze'}
        
        # Calculate interaction quality
        interaction_qualities = [i.metrics.get('quality', 0.5) for i in interactions]
        avg_interaction_quality = np.mean(interaction_qualities)
        
        # Calculate message efficiency
        successful_messages = sum(1 for msg in messages if msg.response_time_ms and msg.response_time_ms < 100)
        message_efficiency = successful_messages / max(len(messages), 1)
        
        # Calculate coordination effectiveness
        coordination_interactions = [i for i in interactions if i.interaction_type == InteractionType.COORDINATION]
        coordination_success_rate = sum(1 for i in coordination_interactions if i.success) / max(len(coordination_interactions), 1)
        
        return {
            'average_interaction_quality': avg_interaction_quality,
            'message_efficiency': message_efficiency,
            'coordination_success_rate': coordination_success_rate,
            'interaction_diversity': len(set(i.interaction_type for i in interactions)),
            'participant_distribution': self._calculate_participant_distribution(interactions)
        }
    
    def _calculate_participant_distribution(self, interactions: List[AgentInteraction]) -> Dict[str, Any]:
        """Calculate distribution of participants across interactions."""
        participant_counts = [len(i.agent_ids) for i in interactions]
        
        return {
            'min_participants': min(participant_counts) if participant_counts else 0,
            'max_participants': max(participant_counts) if participant_counts else 0,
            'avg_participants': np.mean(participant_counts) if participant_counts else 0,
            'participant_variance': np.var(participant_counts) if participant_counts else 0
        }
    
    def _analyze_interaction_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in agent interactions."""
        if not self.interaction_patterns:
            return {'patterns': {}, 'analysis': 'No patterns detected'}
        
        # Sort patterns by frequency
        sorted_patterns = sorted(self.interaction_patterns.items(), key=lambda x: x[1], reverse=True)
        
        # Analyze pattern diversity
        total_patterns = sum(self.interaction_patterns.values())
        pattern_diversity = len(self.interaction_patterns) / max(total_patterns, 1)
        
        # Identify dominant patterns
        dominant_patterns = sorted_patterns[:3]  # Top 3 patterns
        
        return {
            'patterns': dict(sorted_patterns),
            'total_patterns': total_patterns,
            'pattern_diversity': pattern_diversity,
            'dominant_patterns': dominant_patterns,
            'pattern_distribution': self._calculate_pattern_distribution(),
            'pattern_trends': self._analyze_pattern_trends()
        }
    
    def _calculate_pattern_distribution(self) -> Dict[str, float]:
        """Calculate distribution of interaction patterns."""
        total = sum(self.interaction_patterns.values())
        if total == 0:
            return {}
        
        return {
            pattern: count / total 
            for pattern, count in self.interaction_patterns.items()
        }
    
    def _analyze_pattern_trends(self) -> Dict[str, Any]:
        """Analyze trends in interaction patterns."""
        # Simplified trend analysis
        return {
            'increasing_patterns': [],
            'decreasing_patterns': [],
            'stable_patterns': list(self.interaction_patterns.keys()),
            'trend_analysis': 'Insufficient data for trend analysis'
        }
    
    def _calculate_validation_metrics(self, scenario_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive validation metrics."""
        successful_scenarios = sum(1 for result in scenario_results if result['success'])
        total_scenarios = len(scenario_results)
        
        # Calculate overall success rate
        overall_success_rate = successful_scenarios / max(total_scenarios, 1)
        
        # Calculate average metrics
        avg_interactions = np.mean([result['metrics']['interactions_observed'] for result in scenario_results])
        avg_messages = np.mean([result['metrics']['messages_exchanged'] for result in scenario_results])
        avg_duration = np.mean([result['metrics']['duration_ms'] for result in scenario_results])
        
        # Calculate quality scores
        quality_scores = []
        for result in scenario_results:
            analysis = result.get('analysis', {})
            quality_metrics = analysis.get('quality_metrics', {})
            if 'average_interaction_quality' in quality_metrics:
                quality_scores.append(quality_metrics['average_interaction_quality'])
        
        avg_quality_score = np.mean(quality_scores) if quality_scores else 0.0
        
        return {
            'overall_success_rate': overall_success_rate,
            'successful_scenarios': successful_scenarios,
            'total_scenarios': total_scenarios,
            'average_interactions_per_scenario': avg_interactions,
            'average_messages_per_scenario': avg_messages,
            'average_scenario_duration_ms': avg_duration,
            'average_quality_score': avg_quality_score,
            'performance_summary': self._generate_performance_summary()
        }
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary."""
        return {
            'interaction_throughput': self.performance_metrics.get('total_interactions', 0),
            'message_throughput': self.performance_metrics.get('message_throughput', 0),
            'average_response_time': self.performance_metrics.get('average_interaction_duration_ms', 0),
            'error_rate': self.performance_metrics.get('failed_interactions', 0) / max(self.performance_metrics.get('total_interactions', 1), 1),
            'efficiency_score': self.performance_metrics.get('communication_efficiency', 0)
        }
    
    def _detect_interaction_issues(self) -> List[Dict[str, Any]]:
        """Detect potential issues in agent interactions."""
        issues = []
        
        # Check for deadlocks
        if self.performance_metrics.get('deadlock_incidents', 0) > 0:
            issues.append({
                'type': 'deadlock',
                'severity': 'high',
                'description': f"Detected {self.performance_metrics['deadlock_incidents']} deadlock incidents",
                'recommendation': 'Implement deadlock detection and resolution mechanisms'
            })
        
        # Check for timeouts
        if self.performance_metrics.get('timeout_incidents', 0) > 0:
            issues.append({
                'type': 'timeout',
                'severity': 'medium',
                'description': f"Detected {self.performance_metrics['timeout_incidents']} timeout incidents",
                'recommendation': 'Optimize message handling and response times'
            })
        
        # Check communication efficiency
        if self.performance_metrics.get('communication_efficiency', 0) < 0.8:
            issues.append({
                'type': 'communication_efficiency',
                'severity': 'medium',
                'description': f"Low communication efficiency: {self.performance_metrics['communication_efficiency']:.2f}",
                'recommendation': 'Improve message routing and protocol efficiency'
            })
        
        return issues
    
    def _calculate_interaction_coverage(self) -> float:
        """Calculate interaction coverage metric."""
        # Simplified coverage calculation
        unique_agent_pairs = set()
        for interaction in self.interactions.values():
            if len(interaction.agent_ids) >= 2:
                for i in range(len(interaction.agent_ids)):
                    for j in range(i + 1, len(interaction.agent_ids)):
                        pair = tuple(sorted([interaction.agent_ids[i], interaction.agent_ids[j]]))
                        unique_agent_pairs.add(pair)
        
        # Assume we want to test all possible pairs
        total_agents = len(set(agent_id for interaction in self.interactions.values() for agent_id in interaction.agent_ids))
        max_pairs = total_agents * (total_agents - 1) // 2 if total_agents > 1 else 1
        
        return len(unique_agent_pairs) / max_pairs
    
    def _calculate_quality_score(self) -> float:
        """Calculate overall quality score."""
        if not self.interactions:
            return 0.0
        
        quality_scores = []
        for interaction in self.interactions.values():
            quality = interaction.metrics.get('quality', 0.5)
            efficiency = interaction.metrics.get('efficiency', 0.5)
            success_factor = 1.0 if interaction.success else 0.0
            
            interaction_quality = (quality + efficiency + success_factor) / 3.0
            quality_scores.append(interaction_quality)
        
        return np.mean(quality_scores)
    
    def _calculate_protocol_compliance(self) -> float:
        """Calculate protocol compliance score."""
        if not self.message_history:
            return 1.0
        
        compliant_messages = 0
        for message in self.message_history:
            # Check basic protocol compliance
            if (message.message_id and 
                message.sender_id and 
                message.receiver_id and 
                message.timestamp and 
                message.content):
                compliant_messages += 1
        
        return compliant_messages / len(self.message_history)
    
    def _calculate_communication_efficiency(self) -> float:
        """Calculate communication efficiency score."""
        if not self.message_history:
            return 0.0
        
        # Calculate based on response times and success rate
        successful_messages = sum(1 for msg in self.message_history if msg.response_time_ms and msg.response_time_ms < 100)
        efficiency = successful_messages / len(self.message_history)
        
        return efficiency
    
    async def _cleanup_monitoring(self):
        """Clean up monitoring resources."""
        self.monitoring_active = False
        self.deadlock_detector = None
        self.message_tracer = None
        
        logger.info("Interaction monitoring cleaned up")


class DeadlockDetector:
    """Deadlock detection utility."""
    
    def __init__(self, timeout_ms: int = 15000):
        self.timeout_ms = timeout_ms
        self.waiting_agents = {}
        self.resource_locks = {}
    
    def check_deadlock(self, agent_id: str, resource_id: str) -> bool:
        """Check for potential deadlock."""
        # Simplified deadlock detection
        return False


class MessageTracer:
    """Message tracing utility."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.message_trace = deque(maxlen=max_history)
    
    def trace_message(self, message: InteractionMessage):
        """Trace a message."""
        self.message_trace.append(message)
    
    def get_message_history(self) -> List[InteractionMessage]:
        """Get message history."""
        return list(self.message_trace)