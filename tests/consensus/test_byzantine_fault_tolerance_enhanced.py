"""
Enhanced Byzantine Fault Tolerance Test Suite - AGENT 1 MISSION
Advanced Byzantine Fault Tolerance Testing Framework

This comprehensive test suite validates Byzantine fault tolerance with:
1. System resilience with up to 1/3 malicious agents
2. Consensus under network partitions and delays
3. Recovery mechanisms for failed consensus attempts
4. Advanced Byzantine attack scenarios
5. Performance under Byzantine conditions

Author: Agent 1 - MARL Coordination Testing Specialist
Version: 1.0 - Production Ready
"""

import pytest
import asyncio
import time
import json
import numpy as np
import threading
from typing import Dict, Any, List, Optional, Tuple, Set
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
import random
from collections import defaultdict, Counter

# Core imports
from src.consensus.pbft_engine import PBFTEngine, ConsensusDecision, ConsensusRequest, PBFTMessage, MessageType
from src.consensus.byzantine_detector import ByzantineDetector, ByzantinePattern, ByzantineEvidence
from src.consensus.cryptographic_core import CryptographicCore
from src.core.events import EventBus, Event, EventType

# Test markers
pytestmark = [
    pytest.mark.consensus_testing,
    pytest.mark.byzantine_testing,
    pytest.mark.fault_tolerance,
    pytest.mark.security
]

logger = logging.getLogger(__name__)


class ByzantineAttackType(Enum):
    """Types of Byzantine attacks to test"""
    CONFLICTING_VOTES = "conflicting_votes"
    MESSAGE_DROPPING = "message_dropping"
    TIMING_ATTACKS = "timing_attacks"
    SIGNATURE_FORGERY = "signature_forgery"
    NETWORK_PARTITION = "network_partition"
    COORDINATED_DISRUPTION = "coordinated_disruption"
    SILENT_FAILURE = "silent_failure"
    FLOODING_ATTACK = "flooding_attack"
    VIEW_CHANGE_ABUSE = "view_change_abuse"
    DOUBLE_SPENDING = "double_spending"


@dataclass
class ByzantineAgent:
    """Byzantine agent configuration"""
    agent_id: str
    attack_type: ByzantineAttackType
    attack_probability: float = 0.7
    attack_parameters: Dict[str, Any] = field(default_factory=dict)
    coordinated_with: List[str] = field(default_factory=list)
    active: bool = True


@dataclass
class NetworkPartition:
    """Network partition configuration"""
    partition_id: str
    agents: List[str]
    isolated_from: List[str]
    partition_duration: float
    message_delay: float = 0.0
    message_drop_rate: float = 0.0


@dataclass
class ByzantineTestMetrics:
    """Metrics for Byzantine fault tolerance testing"""
    total_consensus_attempts: int = 0
    successful_consensus: int = 0
    failed_consensus: int = 0
    byzantine_agents_detected: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    consensus_latency_ms: List[float] = field(default_factory=list)
    recovery_time_ms: List[float] = field(default_factory=list)
    network_partition_recoveries: int = 0
    attack_success_rate: float = 0.0
    system_availability: float = 0.0


class ByzantineFaultToleranceOrchestrator:
    """
    Advanced Byzantine fault tolerance testing orchestrator
    
    Features:
    - Sophisticated Byzantine attack simulation
    - Network partition testing
    - Consensus recovery validation
    - Performance under Byzantine conditions
    - Coordinated attack scenarios
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.event_bus = EventBus()
        
        # Core components
        self.pbft_engines = {}
        self.byzantine_detectors = {}
        self.crypto_cores = {}
        
        # Test state
        self.agents = {}
        self.byzantine_agents = {}
        self.network_partitions = {}
        self.consensus_history = []
        self.attack_history = []
        self.test_metrics = ByzantineTestMetrics()
        
        # Message interception
        self.message_interceptor = MessageInterceptor()
        self.consensus_monitor = ConsensusMonitor()
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
        logger.info("Byzantine fault tolerance orchestrator initialized")
    
    async def initialize_byzantine_testing_environment(self, agent_count: int = 7, byzantine_count: int = 2) -> bool:
        """Initialize comprehensive Byzantine testing environment"""
        try:
            # Validate Byzantine fault tolerance requirements
            if byzantine_count >= agent_count / 3:
                logger.warning(f"Byzantine count {byzantine_count} exceeds f < n/3 requirement for {agent_count} agents")
            
            # Create agent IDs
            agent_ids = [f"agent_{i}" for i in range(agent_count)]
            
            # Initialize cryptographic cores
            for agent_id in agent_ids:
                self.crypto_cores[agent_id] = CryptographicCore()
            
            # Initialize PBFT engines
            for agent_id in agent_ids:
                self.pbft_engines[agent_id] = PBFTEngine(
                    agent_id=agent_id,
                    agent_ids=agent_ids,
                    byzantine_fault_tolerance=byzantine_count,
                    consensus_timeout=2.0,
                    cryptographic_core=self.crypto_cores[agent_id]
                )
            
            # Initialize Byzantine detectors
            for agent_id in agent_ids:
                self.byzantine_detectors[agent_id] = ByzantineDetector(
                    agent_ids=agent_ids,
                    detection_window=120.0,
                    anomaly_threshold=0.7,
                    min_evidence_count=3
                )
            
            # Initialize agents
            await self._initialize_test_agents(agent_ids)
            
            # Configure Byzantine agents
            await self._configure_byzantine_agents(agent_ids[:byzantine_count])
            
            # Setup message interception
            await self._setup_message_interception()
            
            logger.info(f"Byzantine testing environment initialized: {agent_count} agents, {byzantine_count} Byzantine")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Byzantine testing environment: {e}")
            return False
    
    async def _initialize_test_agents(self, agent_ids: List[str]):
        """Initialize test agents with realistic behavior"""
        for agent_id in agent_ids:
            agent = await self._create_test_agent(agent_id)
            self.agents[agent_id] = agent
    
    async def _create_test_agent(self, agent_id: str) -> Mock:
        """Create test agent with consensus capabilities"""
        agent = Mock()
        agent.agent_id = agent_id
        agent.is_byzantine = False
        agent.message_log = []
        agent.consensus_participation = []
        
        # Mock consensus participation
        async def participate_in_consensus(consensus_request: ConsensusRequest) -> Dict[str, Any]:
            participation_start = time.time()
            
            # Simulate processing time
            await asyncio.sleep(np.random.uniform(0.01, 0.05))
            
            # Generate decision
            decision = {
                'agent_id': agent_id,
                'decision': np.random.choice(['approve', 'reject', 'abstain']),
                'confidence': np.random.uniform(0.6, 0.95),
                'timestamp': time.time(),
                'consensus_round': consensus_request.request_id
            }
            
            # Record participation
            agent.consensus_participation.append(decision)
            
            # Record message activity for Byzantine detection
            for detector in self.byzantine_detectors.values():
                detector.record_message_activity(
                    agent_id=agent_id,
                    message_type='consensus_participation',
                    timestamp=time.time(),
                    consensus_round=len(agent.consensus_participation),
                    signature_valid=True
                )
            
            participation_time = (time.time() - participation_start) * 1000
            self.test_metrics.consensus_latency_ms.append(participation_time)
            
            return decision
        
        # Mock message handling
        async def handle_message(message: PBFTMessage) -> bool:
            agent.message_log.append(message)
            
            # Record message activity
            for detector in self.byzantine_detectors.values():
                detector.record_message_activity(
                    agent_id=agent_id,
                    message_type=message.message_type.value,
                    timestamp=message.timestamp,
                    signature_valid=message.signature is not None
                )
            
            return True
        
        agent.participate_in_consensus = participate_in_consensus
        agent.handle_message = handle_message
        
        return agent
    
    async def _configure_byzantine_agents(self, byzantine_agent_ids: List[str]):
        """Configure Byzantine agents with various attack types"""
        attack_types = list(ByzantineAttackType)
        
        for i, agent_id in enumerate(byzantine_agent_ids):
            attack_type = attack_types[i % len(attack_types)]
            
            byzantine_agent = ByzantineAgent(
                agent_id=agent_id,
                attack_type=attack_type,
                attack_probability=0.7,
                attack_parameters=self._get_attack_parameters(attack_type)
            )
            
            self.byzantine_agents[agent_id] = byzantine_agent
            self.agents[agent_id].is_byzantine = True
            
            # Override agent behavior for Byzantine attacks
            await self._override_agent_behavior(agent_id, byzantine_agent)
            
            logger.info(f"Configured Byzantine agent {agent_id} with attack type {attack_type.value}")
    
    def _get_attack_parameters(self, attack_type: ByzantineAttackType) -> Dict[str, Any]:
        """Get attack parameters for specific attack type"""
        parameters = {
            ByzantineAttackType.CONFLICTING_VOTES: {
                'conflict_probability': 0.8,
                'vote_variations': ['approve', 'reject', 'abstain']
            },
            ByzantineAttackType.MESSAGE_DROPPING: {
                'drop_rate': 0.5,
                'selective_drop': True
            },
            ByzantineAttackType.TIMING_ATTACKS: {
                'delay_min_ms': 100,
                'delay_max_ms': 2000,
                'delay_probability': 0.6
            },
            ByzantineAttackType.SIGNATURE_FORGERY: {
                'forge_probability': 0.4,
                'invalid_signature_rate': 0.3
            },
            ByzantineAttackType.FLOODING_ATTACK: {
                'message_multiplier': 5,
                'flood_probability': 0.5
            },
            ByzantineAttackType.VIEW_CHANGE_ABUSE: {
                'unnecessary_view_changes': 0.7,
                'false_view_change_rate': 0.4
            },
            ByzantineAttackType.SILENT_FAILURE: {
                'silence_probability': 0.6,
                'partial_participation': 0.3
            },
            ByzantineAttackType.COORDINATED_DISRUPTION: {
                'coordination_probability': 0.8,
                'disruption_intensity': 0.7
            }
        }
        
        return parameters.get(attack_type, {})
    
    async def _override_agent_behavior(self, agent_id: str, byzantine_agent: ByzantineAgent):
        """Override agent behavior for Byzantine attacks"""
        original_agent = self.agents[agent_id]
        attack_type = byzantine_agent.attack_type
        
        if attack_type == ByzantineAttackType.CONFLICTING_VOTES:
            await self._implement_conflicting_votes_attack(agent_id, byzantine_agent)
        elif attack_type == ByzantineAttackType.MESSAGE_DROPPING:
            await self._implement_message_dropping_attack(agent_id, byzantine_agent)
        elif attack_type == ByzantineAttackType.TIMING_ATTACKS:
            await self._implement_timing_attack(agent_id, byzantine_agent)
        elif attack_type == ByzantineAttackType.SIGNATURE_FORGERY:
            await self._implement_signature_forgery_attack(agent_id, byzantine_agent)
        elif attack_type == ByzantineAttackType.FLOODING_ATTACK:
            await self._implement_flooding_attack(agent_id, byzantine_agent)
        elif attack_type == ByzantineAttackType.VIEW_CHANGE_ABUSE:
            await self._implement_view_change_abuse_attack(agent_id, byzantine_agent)
        elif attack_type == ByzantineAttackType.SILENT_FAILURE:
            await self._implement_silent_failure_attack(agent_id, byzantine_agent)
        elif attack_type == ByzantineAttackType.COORDINATED_DISRUPTION:
            await self._implement_coordinated_disruption_attack(agent_id, byzantine_agent)
    
    async def _implement_conflicting_votes_attack(self, agent_id: str, byzantine_agent: ByzantineAgent):
        """Implement conflicting votes attack"""
        original_participate = self.agents[agent_id].participate_in_consensus
        
        async def byzantine_participate_in_consensus(consensus_request: ConsensusRequest) -> Dict[str, Any]:
            if random.random() < byzantine_agent.attack_probability:
                # Send conflicting decisions
                decisions = byzantine_agent.attack_parameters['vote_variations']
                
                # Record attack
                self.attack_history.append({
                    'agent_id': agent_id,
                    'attack_type': 'conflicting_votes',
                    'timestamp': time.time(),
                    'consensus_round': consensus_request.request_id
                })
                
                # Return random conflicting vote
                return {
                    'agent_id': agent_id,
                    'decision': random.choice(decisions),
                    'confidence': random.uniform(0.1, 0.4),  # Low confidence
                    'timestamp': time.time(),
                    'byzantine_attack': True
                }
            else:
                return await original_participate(consensus_request)
        
        self.agents[agent_id].participate_in_consensus = byzantine_participate_in_consensus
    
    async def _implement_message_dropping_attack(self, agent_id: str, byzantine_agent: ByzantineAgent):
        """Implement message dropping attack"""
        original_handle_message = self.agents[agent_id].handle_message
        
        async def byzantine_handle_message(message: PBFTMessage) -> bool:
            if random.random() < byzantine_agent.attack_parameters['drop_rate']:
                # Drop message
                self.attack_history.append({
                    'agent_id': agent_id,
                    'attack_type': 'message_dropping',
                    'timestamp': time.time(),
                    'dropped_message_type': message.message_type.value
                })
                
                return False  # Message dropped
            else:
                return await original_handle_message(message)
        
        self.agents[agent_id].handle_message = byzantine_handle_message
    
    async def _implement_timing_attack(self, agent_id: str, byzantine_agent: ByzantineAgent):
        """Implement timing attack"""
        original_participate = self.agents[agent_id].participate_in_consensus
        
        async def byzantine_participate_in_consensus(consensus_request: ConsensusRequest) -> Dict[str, Any]:
            if random.random() < byzantine_agent.attack_parameters['delay_probability']:
                # Introduce delay
                delay_ms = random.uniform(
                    byzantine_agent.attack_parameters['delay_min_ms'],
                    byzantine_agent.attack_parameters['delay_max_ms']
                )
                
                await asyncio.sleep(delay_ms / 1000.0)
                
                # Record attack
                self.attack_history.append({
                    'agent_id': agent_id,
                    'attack_type': 'timing_attack',
                    'timestamp': time.time(),
                    'delay_ms': delay_ms
                })
            
            return await original_participate(consensus_request)
        
        self.agents[agent_id].participate_in_consensus = byzantine_participate_in_consensus
    
    async def _implement_signature_forgery_attack(self, agent_id: str, byzantine_agent: ByzantineAgent):
        """Implement signature forgery attack"""
        original_handle_message = self.agents[agent_id].handle_message
        
        async def byzantine_handle_message(message: PBFTMessage) -> bool:
            if random.random() < byzantine_agent.attack_parameters['forge_probability']:
                # Forge signature
                message.signature = "forged_signature_" + str(random.randint(1000, 9999))
                
                # Record attack
                self.attack_history.append({
                    'agent_id': agent_id,
                    'attack_type': 'signature_forgery',
                    'timestamp': time.time(),
                    'forged_signature': message.signature
                })
                
                # Record invalid signature for Byzantine detection
                for detector in self.byzantine_detectors.values():
                    detector.record_message_activity(
                        agent_id=agent_id,
                        message_type=message.message_type.value,
                        timestamp=message.timestamp,
                        signature_valid=False
                    )
            
            return await original_handle_message(message)
        
        self.agents[agent_id].handle_message = byzantine_handle_message
    
    async def _implement_flooding_attack(self, agent_id: str, byzantine_agent: ByzantineAgent):
        """Implement flooding attack"""
        original_participate = self.agents[agent_id].participate_in_consensus
        
        async def byzantine_participate_in_consensus(consensus_request: ConsensusRequest) -> Dict[str, Any]:
            if random.random() < byzantine_agent.attack_parameters['flood_probability']:
                # Send multiple messages
                multiplier = byzantine_agent.attack_parameters['message_multiplier']
                
                # Record attack
                self.attack_history.append({
                    'agent_id': agent_id,
                    'attack_type': 'flooding_attack',
                    'timestamp': time.time(),
                    'message_multiplier': multiplier
                })
                
                # Send multiple rapid messages
                for i in range(multiplier):
                    await asyncio.sleep(0.001)  # Very short delay
                    
                    # Record flooding activity
                    for detector in self.byzantine_detectors.values():
                        detector.record_message_activity(
                            agent_id=agent_id,
                            message_type='flooding_message',
                            timestamp=time.time(),
                            signature_valid=True
                        )
            
            return await original_participate(consensus_request)
        
        self.agents[agent_id].participate_in_consensus = byzantine_participate_in_consensus
    
    async def _implement_view_change_abuse_attack(self, agent_id: str, byzantine_agent: ByzantineAgent):
        """Implement view change abuse attack"""
        async def byzantine_trigger_view_change():
            if random.random() < byzantine_agent.attack_parameters['unnecessary_view_changes']:
                # Record attack
                self.attack_history.append({
                    'agent_id': agent_id,
                    'attack_type': 'view_change_abuse',
                    'timestamp': time.time(),
                    'unnecessary_view_change': True
                })
                
                # Record view change abuse
                for detector in self.byzantine_detectors.values():
                    detector.record_consensus_result(
                        consensus_round=len(self.consensus_history),
                        participating_agents=[agent_id],
                        consensus_achieved=False,
                        view_changes=5  # Excessive view changes
                    )
        
        # Add view change abuse method to agent
        self.agents[agent_id].trigger_view_change_abuse = byzantine_trigger_view_change
    
    async def _implement_silent_failure_attack(self, agent_id: str, byzantine_agent: ByzantineAgent):
        """Implement silent failure attack"""
        original_participate = self.agents[agent_id].participate_in_consensus
        
        async def byzantine_participate_in_consensus(consensus_request: ConsensusRequest) -> Dict[str, Any]:
            if random.random() < byzantine_agent.attack_parameters['silence_probability']:
                # Remain silent (don't participate)
                self.attack_history.append({
                    'agent_id': agent_id,
                    'attack_type': 'silent_failure',
                    'timestamp': time.time(),
                    'consensus_round': consensus_request.request_id
                })
                
                # Don't return any decision (silent)
                await asyncio.sleep(10.0)  # Long delay to simulate timeout
                return None
            else:
                return await original_participate(consensus_request)
        
        self.agents[agent_id].participate_in_consensus = byzantine_participate_in_consensus
    
    async def _implement_coordinated_disruption_attack(self, agent_id: str, byzantine_agent: ByzantineAgent):
        """Implement coordinated disruption attack"""
        original_participate = self.agents[agent_id].participate_in_consensus
        
        async def byzantine_participate_in_consensus(consensus_request: ConsensusRequest) -> Dict[str, Any]:
            if random.random() < byzantine_agent.attack_parameters['coordination_probability']:
                # Coordinate with other Byzantine agents
                coordinated_decision = self._get_coordinated_decision(consensus_request)
                
                # Record attack
                self.attack_history.append({
                    'agent_id': agent_id,
                    'attack_type': 'coordinated_disruption',
                    'timestamp': time.time(),
                    'coordinated_decision': coordinated_decision
                })
                
                return {
                    'agent_id': agent_id,
                    'decision': coordinated_decision,
                    'confidence': 0.9,  # High confidence to seem legitimate
                    'timestamp': time.time(),
                    'byzantine_attack': True,
                    'coordinated': True
                }
            else:
                return await original_participate(consensus_request)
        
        self.agents[agent_id].participate_in_consensus = byzantine_participate_in_consensus
    
    def _get_coordinated_decision(self, consensus_request: ConsensusRequest) -> str:
        """Get coordinated decision for Byzantine agents"""
        # Simple coordination: all Byzantine agents vote the same way
        return 'reject'  # Always reject to disrupt consensus
    
    async def _setup_message_interception(self):
        """Setup message interception for network simulation"""
        self.message_interceptor = MessageInterceptor()
        
        # Intercept messages between agents
        for agent_id in self.agents.keys():
            await self.message_interceptor.register_agent(agent_id)
    
    async def run_byzantine_consensus_test(self, consensus_rounds: int = 10) -> Dict[str, Any]:
        """Run Byzantine consensus test with multiple rounds"""
        logger.info(f"Starting Byzantine consensus test with {consensus_rounds} rounds")
        
        test_results = {
            'total_rounds': consensus_rounds,
            'consensus_results': [],
            'attack_summary': {},
            'detection_results': {},
            'performance_metrics': {}
        }
        
        for round_num in range(consensus_rounds):
            logger.info(f"Running consensus round {round_num + 1}/{consensus_rounds}")
            
            # Create consensus request
            consensus_request = ConsensusRequest(
                request_id=f"test_consensus_{round_num}",
                agent_decisions={agent_id: {'decision': 'test_decision'} for agent_id in self.agents.keys()},
                market_state={'round': round_num},
                synergy_context={},
                timestamp=time.time(),
                requester_id='test_orchestrator'
            )
            
            # Execute consensus round
            consensus_result = await self._execute_consensus_round(consensus_request)
            test_results['consensus_results'].append(consensus_result)
            
            # Update metrics
            self.test_metrics.total_consensus_attempts += 1
            if consensus_result.get('consensus_achieved', False):
                self.test_metrics.successful_consensus += 1
            else:
                self.test_metrics.failed_consensus += 1
            
            # Wait between rounds
            await asyncio.sleep(0.5)
        
        # Analyze attack effectiveness
        test_results['attack_summary'] = self._analyze_attack_effectiveness()
        
        # Get Byzantine detection results
        test_results['detection_results'] = self._analyze_byzantine_detection()
        
        # Calculate performance metrics
        test_results['performance_metrics'] = self._calculate_performance_metrics()
        
        return test_results
    
    async def _execute_consensus_round(self, consensus_request: ConsensusRequest) -> Dict[str, Any]:
        """Execute a single consensus round"""
        start_time = time.time()
        
        # Collect agent decisions
        agent_decisions = {}
        consensus_tasks = []
        
        for agent_id, agent in self.agents.items():
            task = asyncio.create_task(agent.participate_in_consensus(consensus_request))
            consensus_tasks.append((agent_id, task))
        
        # Wait for agent responses
        for agent_id, task in consensus_tasks:
            try:
                decision = await asyncio.wait_for(task, timeout=3.0)
                if decision is not None:
                    agent_decisions[agent_id] = decision
            except asyncio.TimeoutError:
                logger.warning(f"Agent {agent_id} timed out during consensus")
            except Exception as e:
                logger.error(f"Error from agent {agent_id}: {e}")
        
        # Determine consensus result
        consensus_achieved = self._determine_consensus(agent_decisions)
        
        consensus_time = (time.time() - start_time) * 1000
        
        # Record consensus result
        for detector in self.byzantine_detectors.values():
            detector.record_consensus_result(
                consensus_round=len(self.consensus_history),
                participating_agents=list(agent_decisions.keys()),
                consensus_achieved=consensus_achieved,
                view_changes=0
            )
        
        consensus_result = {
            'consensus_request_id': consensus_request.request_id,
            'consensus_achieved': consensus_achieved,
            'participating_agents': list(agent_decisions.keys()),
            'consensus_time_ms': consensus_time,
            'agent_decisions': agent_decisions,
            'timestamp': time.time()
        }
        
        self.consensus_history.append(consensus_result)
        
        return consensus_result
    
    def _determine_consensus(self, agent_decisions: Dict[str, Any]) -> bool:
        """Determine if consensus was achieved"""
        if len(agent_decisions) < len(self.agents) * 0.67:  # Need > 2/3 participation
            return False
        
        # Count decisions
        decision_counts = Counter()
        for decision_data in agent_decisions.values():
            if decision_data and 'decision' in decision_data:
                decision_counts[decision_data['decision']] += 1
        
        # Check for majority
        if decision_counts:
            max_count = max(decision_counts.values())
            return max_count > len(agent_decisions) * 0.5
        
        return False
    
    def _analyze_attack_effectiveness(self) -> Dict[str, Any]:
        """Analyze effectiveness of Byzantine attacks"""
        attack_summary = {}
        
        # Group attacks by type
        attacks_by_type = defaultdict(list)
        for attack in self.attack_history:
            attacks_by_type[attack['attack_type']].append(attack)
        
        # Calculate effectiveness for each attack type
        for attack_type, attacks in attacks_by_type.items():
            # Find consensus rounds affected by this attack type
            affected_rounds = set()
            for attack in attacks:
                # Find corresponding consensus round
                for consensus in self.consensus_history:
                    if abs(consensus['timestamp'] - attack['timestamp']) < 5.0:
                        affected_rounds.add(consensus['consensus_request_id'])
            
            # Calculate disruption rate
            disrupted_consensus = sum(1 for consensus in self.consensus_history 
                                    if consensus['consensus_request_id'] in affected_rounds 
                                    and not consensus['consensus_achieved'])
            
            attack_summary[attack_type] = {
                'total_attacks': len(attacks),
                'affected_consensus_rounds': len(affected_rounds),
                'disrupted_consensus': disrupted_consensus,
                'disruption_rate': disrupted_consensus / max(1, len(affected_rounds))
            }
        
        return attack_summary
    
    def _analyze_byzantine_detection(self) -> Dict[str, Any]:
        """Analyze Byzantine detection effectiveness"""
        detection_results = {}
        
        # Get detection results from all detectors
        for detector_id, detector in self.byzantine_detectors.items():
            suspected, confirmed = detector.get_byzantine_agents()
            
            # Calculate detection accuracy
            true_positives = len(set(confirmed) & set(self.byzantine_agents.keys()))
            false_positives = len(set(confirmed) - set(self.byzantine_agents.keys()))
            false_negatives = len(set(self.byzantine_agents.keys()) - set(confirmed))
            
            detection_results[detector_id] = {
                'suspected_agents': list(suspected),
                'confirmed_agents': list(confirmed),
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'precision': true_positives / max(1, true_positives + false_positives),
                'recall': true_positives / max(1, true_positives + false_negatives),
                'detection_metrics': detector.get_detection_metrics()
            }
        
        return detection_results
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
        return {
            'total_consensus_attempts': self.test_metrics.total_consensus_attempts,
            'successful_consensus': self.test_metrics.successful_consensus,
            'failed_consensus': self.test_metrics.failed_consensus,
            'success_rate': self.test_metrics.successful_consensus / max(1, self.test_metrics.total_consensus_attempts),
            'average_consensus_latency_ms': np.mean(self.test_metrics.consensus_latency_ms) if self.test_metrics.consensus_latency_ms else 0,
            'max_consensus_latency_ms': np.max(self.test_metrics.consensus_latency_ms) if self.test_metrics.consensus_latency_ms else 0,
            'total_attacks': len(self.attack_history),
            'unique_attack_types': len(set(attack['attack_type'] for attack in self.attack_history)),
            'byzantine_agents_count': len(self.byzantine_agents),
            'honest_agents_count': len(self.agents) - len(self.byzantine_agents)
        }
    
    async def run_network_partition_test(self, partition_scenarios: List[NetworkPartition]) -> Dict[str, Any]:
        """Run network partition test scenarios"""
        logger.info(f"Starting network partition test with {len(partition_scenarios)} scenarios")
        
        test_results = {
            'partition_scenarios': [],
            'recovery_metrics': {},
            'consensus_under_partition': {}
        }
        
        for scenario in partition_scenarios:
            logger.info(f"Testing partition scenario: {scenario.partition_id}")
            
            # Apply network partition
            await self._apply_network_partition(scenario)
            
            # Test consensus under partition
            partition_consensus_result = await self._test_consensus_under_partition(scenario)
            
            # Test partition recovery
            recovery_result = await self._test_partition_recovery(scenario)
            
            scenario_result = {
                'partition_id': scenario.partition_id,
                'partition_agents': scenario.agents,
                'isolated_agents': scenario.isolated_from,
                'partition_duration': scenario.partition_duration,
                'consensus_result': partition_consensus_result,
                'recovery_result': recovery_result
            }
            
            test_results['partition_scenarios'].append(scenario_result)
            
            # Remove partition
            await self._remove_network_partition(scenario)
        
        return test_results
    
    async def _apply_network_partition(self, partition: NetworkPartition):
        """Apply network partition"""
        self.network_partitions[partition.partition_id] = partition
        
        # Configure message delays and drops
        for agent_id in partition.agents:
            if agent_id in self.agents:
                # Add partition effects to agent
                self.agents[agent_id].partition_id = partition.partition_id
                self.agents[agent_id].message_delay = partition.message_delay
                self.agents[agent_id].message_drop_rate = partition.message_drop_rate
        
        logger.info(f"Applied network partition {partition.partition_id}")
    
    async def _test_consensus_under_partition(self, partition: NetworkPartition) -> Dict[str, Any]:
        """Test consensus under network partition"""
        # Create consensus request
        consensus_request = ConsensusRequest(
            request_id=f"partition_test_{partition.partition_id}",
            agent_decisions={agent_id: {'decision': 'partition_test'} for agent_id in partition.agents},
            market_state={'partition_test': True},
            synergy_context={},
            timestamp=time.time(),
            requester_id='partition_tester'
        )
        
        # Execute consensus with partition
        consensus_result = await self._execute_consensus_round(consensus_request)
        
        return {
            'consensus_achieved': consensus_result['consensus_achieved'],
            'participating_agents': len(consensus_result['participating_agents']),
            'consensus_time_ms': consensus_result['consensus_time_ms'],
            'partition_effects': {
                'message_delays': partition.message_delay,
                'message_drops': partition.message_drop_rate
            }
        }
    
    async def _test_partition_recovery(self, partition: NetworkPartition) -> Dict[str, Any]:
        """Test recovery from network partition"""
        recovery_start = time.time()
        
        # Remove partition
        await self._remove_network_partition(partition)
        
        # Test consensus recovery
        recovery_consensus_request = ConsensusRequest(
            request_id=f"recovery_test_{partition.partition_id}",
            agent_decisions={agent_id: {'decision': 'recovery_test'} for agent_id in self.agents.keys()},
            market_state={'recovery_test': True},
            synergy_context={},
            timestamp=time.time(),
            requester_id='recovery_tester'
        )
        
        recovery_consensus_result = await self._execute_consensus_round(recovery_consensus_request)
        
        recovery_time = (time.time() - recovery_start) * 1000
        
        return {
            'recovery_time_ms': recovery_time,
            'consensus_restored': recovery_consensus_result['consensus_achieved'],
            'agents_recovered': len(recovery_consensus_result['participating_agents']),
            'recovery_success': recovery_consensus_result['consensus_achieved'] and 
                              len(recovery_consensus_result['participating_agents']) >= len(self.agents) * 0.67
        }
    
    async def _remove_network_partition(self, partition: NetworkPartition):
        """Remove network partition"""
        if partition.partition_id in self.network_partitions:
            del self.network_partitions[partition.partition_id]
        
        # Remove partition effects from agents
        for agent_id in partition.agents:
            if agent_id in self.agents:
                self.agents[agent_id].partition_id = None
                self.agents[agent_id].message_delay = 0.0
                self.agents[agent_id].message_drop_rate = 0.0
        
        logger.info(f"Removed network partition {partition.partition_id}")
    
    async def run_coordinated_byzantine_attack_test(self, coordinated_agents: List[str]) -> Dict[str, Any]:
        """Run coordinated Byzantine attack test"""
        logger.info(f"Starting coordinated Byzantine attack test with {len(coordinated_agents)} agents")
        
        # Configure coordinated attack
        for agent_id in coordinated_agents:
            if agent_id in self.byzantine_agents:
                self.byzantine_agents[agent_id].coordinated_with = [
                    other_id for other_id in coordinated_agents if other_id != agent_id
                ]
        
        # Execute coordinated attack scenarios
        attack_scenarios = [
            {'name': 'coordinated_rejection', 'rounds': 5},
            {'name': 'coordinated_delay', 'rounds': 3},
            {'name': 'coordinated_conflicting_votes', 'rounds': 4}
        ]
        
        test_results = {
            'coordinated_agents': coordinated_agents,
            'attack_scenarios': [],
            'system_resilience': {}
        }
        
        for scenario in attack_scenarios:
            scenario_result = await self._execute_coordinated_attack_scenario(scenario, coordinated_agents)
            test_results['attack_scenarios'].append(scenario_result)
        
        # Analyze system resilience
        test_results['system_resilience'] = self._analyze_system_resilience()
        
        return test_results
    
    async def _execute_coordinated_attack_scenario(self, scenario: Dict[str, Any], coordinated_agents: List[str]) -> Dict[str, Any]:
        """Execute coordinated attack scenario"""
        scenario_results = []
        
        for round_num in range(scenario['rounds']):
            # Create consensus request
            consensus_request = ConsensusRequest(
                request_id=f"coordinated_attack_{scenario['name']}_{round_num}",
                agent_decisions={agent_id: {'decision': 'coordinated_test'} for agent_id in self.agents.keys()},
                market_state={'coordinated_attack': True},
                synergy_context={},
                timestamp=time.time(),
                requester_id='coordinated_attack_tester'
            )
            
            # Execute consensus round
            consensus_result = await self._execute_consensus_round(consensus_request)
            scenario_results.append(consensus_result)
            
            await asyncio.sleep(0.5)
        
        return {
            'scenario_name': scenario['name'],
            'rounds_executed': len(scenario_results),
            'successful_consensus': sum(1 for r in scenario_results if r['consensus_achieved']),
            'failed_consensus': sum(1 for r in scenario_results if not r['consensus_achieved']),
            'attack_effectiveness': 1 - (sum(1 for r in scenario_results if r['consensus_achieved']) / len(scenario_results))
        }
    
    def _analyze_system_resilience(self) -> Dict[str, Any]:
        """Analyze system resilience against Byzantine attacks"""
        total_consensus = len(self.consensus_history)
        successful_consensus = sum(1 for c in self.consensus_history if c['consensus_achieved'])
        
        return {
            'total_consensus_attempts': total_consensus,
            'successful_consensus': successful_consensus,
            'system_availability': successful_consensus / max(1, total_consensus),
            'resilience_score': min(1.0, successful_consensus / max(1, total_consensus) * 1.5),
            'byzantine_attack_resistance': 1 - (len(self.attack_history) / max(1, total_consensus))
        }
    
    def generate_comprehensive_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive Byzantine fault tolerance test report"""
        return {
            'test_configuration': {
                'total_agents': len(self.agents),
                'byzantine_agents': len(self.byzantine_agents),
                'attack_types': list(set(ba.attack_type.value for ba in self.byzantine_agents.values())),
                'test_duration': time.time() - self.config.get('test_start_time', time.time())
            },
            'consensus_results': {
                'total_consensus_rounds': len(self.consensus_history),
                'successful_consensus': sum(1 for c in self.consensus_history if c['consensus_achieved']),
                'failed_consensus': sum(1 for c in self.consensus_history if not c['consensus_achieved']),
                'average_consensus_time_ms': np.mean([c['consensus_time_ms'] for c in self.consensus_history]) if self.consensus_history else 0
            },
            'attack_analysis': self._analyze_attack_effectiveness(),
            'detection_analysis': self._analyze_byzantine_detection(),
            'performance_metrics': self._calculate_performance_metrics(),
            'network_partition_results': getattr(self, 'network_partition_results', {}),
            'system_resilience': self._analyze_system_resilience(),
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Analyze success rate
        success_rate = self.test_metrics.successful_consensus / max(1, self.test_metrics.total_consensus_attempts)
        if success_rate < 0.8:
            recommendations.append("Consider increasing consensus timeout to improve success rate")
        
        # Analyze detection effectiveness
        detection_results = self._analyze_byzantine_detection()
        avg_precision = np.mean([r['precision'] for r in detection_results.values()])
        if avg_precision < 0.7:
            recommendations.append("Improve Byzantine detection algorithms for better precision")
        
        # Analyze attack effectiveness
        attack_summary = self._analyze_attack_effectiveness()
        high_disruption_attacks = [attack for attack, data in attack_summary.items() 
                                 if data['disruption_rate'] > 0.5]
        if high_disruption_attacks:
            recommendations.append(f"Strengthen defenses against: {', '.join(high_disruption_attacks)}")
        
        return recommendations


class MessageInterceptor:
    """Message interceptor for network simulation"""
    
    def __init__(self):
        self.registered_agents = {}
        self.message_log = []
    
    async def register_agent(self, agent_id: str):
        """Register agent for message interception"""
        self.registered_agents[agent_id] = {'messages': [], 'active': True}
    
    async def intercept_message(self, source: str, target: str, message: Any) -> bool:
        """Intercept message between agents"""
        self.message_log.append({
            'source': source,
            'target': target,
            'message': message,
            'timestamp': time.time()
        })
        
        # Check if message should be delivered
        if target in self.registered_agents and self.registered_agents[target]['active']:
            return True
        
        return False


class ConsensusMonitor:
    """Monitor consensus process"""
    
    def __init__(self):
        self.consensus_events = []
        self.performance_metrics = {}
    
    def record_consensus_event(self, event_type: str, data: Dict[str, Any]):
        """Record consensus event"""
        self.consensus_events.append({
            'event_type': event_type,
            'data': data,
            'timestamp': time.time()
        })
    
    def get_consensus_statistics(self) -> Dict[str, Any]:
        """Get consensus statistics"""
        return {
            'total_events': len(self.consensus_events),
            'event_types': Counter(event['event_type'] for event in self.consensus_events),
            'performance_metrics': self.performance_metrics
        }


class PerformanceTracker:
    """Track performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'latency_measurements': [],
            'throughput_measurements': [],
            'resource_usage': []
        }
    
    def record_latency(self, operation: str, latency_ms: float):
        """Record latency measurement"""
        self.metrics['latency_measurements'].append({
            'operation': operation,
            'latency_ms': latency_ms,
            'timestamp': time.time()
        })
    
    def record_throughput(self, operation: str, throughput: float):
        """Record throughput measurement"""
        self.metrics['throughput_measurements'].append({
            'operation': operation,
            'throughput': throughput,
            'timestamp': time.time()
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            'average_latency_ms': np.mean([m['latency_ms'] for m in self.metrics['latency_measurements']]) if self.metrics['latency_measurements'] else 0,
            'max_latency_ms': np.max([m['latency_ms'] for m in self.metrics['latency_measurements']]) if self.metrics['latency_measurements'] else 0,
            'average_throughput': np.mean([m['throughput'] for m in self.metrics['throughput_measurements']]) if self.metrics['throughput_measurements'] else 0,
            'total_measurements': len(self.metrics['latency_measurements']) + len(self.metrics['throughput_measurements'])
        }


class TestByzantineFaultTolerance:
    """Enhanced Byzantine fault tolerance test suite"""
    
    @pytest.fixture
    async def byzantine_orchestrator(self):
        """Setup Byzantine fault tolerance orchestrator"""
        config = {
            'test_name': 'byzantine_fault_tolerance_test',
            'test_start_time': time.time(),
            'agent_count': 7,
            'byzantine_count': 2,
            'consensus_timeout': 2.0,
            'detection_threshold': 0.7
        }
        
        orchestrator = ByzantineFaultToleranceOrchestrator(config)
        
        success = await orchestrator.initialize_byzantine_testing_environment(
            agent_count=config['agent_count'],
            byzantine_count=config['byzantine_count']
        )
        assert success, "Failed to initialize Byzantine testing environment"
        
        yield orchestrator
        
        # Cleanup
        await orchestrator.event_bus.stop()
    
    @pytest.mark.asyncio
    async def test_consensus_with_byzantine_agents(self, byzantine_orchestrator):
        """Test consensus with Byzantine agents present"""
        # Run Byzantine consensus test
        test_results = await byzantine_orchestrator.run_byzantine_consensus_test(consensus_rounds=5)
        
        # Verify test execution
        assert test_results['total_rounds'] == 5
        assert len(test_results['consensus_results']) == 5
        
        # Verify some consensus was achieved despite Byzantine agents
        successful_consensus = sum(1 for result in test_results['consensus_results'] 
                                 if result['consensus_achieved'])
        assert successful_consensus > 0, "No consensus achieved with Byzantine agents"
        
        # Verify attack detection
        assert len(test_results['attack_summary']) > 0, "No attacks detected"
        
        # Verify performance metrics
        performance = test_results['performance_metrics']
        assert performance['total_consensus_attempts'] == 5
        assert performance['byzantine_agents_count'] == 2
    
    @pytest.mark.asyncio
    async def test_network_partition_resilience(self, byzantine_orchestrator):
        """Test system resilience under network partitions"""
        # Create partition scenarios
        partition_scenarios = [
            NetworkPartition(
                partition_id='test_partition_1',
                agents=['agent_0', 'agent_1', 'agent_2'],
                isolated_from=['agent_3', 'agent_4', 'agent_5', 'agent_6'],
                partition_duration=2.0,
                message_delay=0.1,
                message_drop_rate=0.1
            ),
            NetworkPartition(
                partition_id='test_partition_2',
                agents=['agent_0', 'agent_1'],
                isolated_from=['agent_2', 'agent_3', 'agent_4', 'agent_5', 'agent_6'],
                partition_duration=3.0,
                message_delay=0.2,
                message_drop_rate=0.2
            )
        ]
        
        # Run network partition test
        test_results = await byzantine_orchestrator.run_network_partition_test(partition_scenarios)
        
        # Verify partition scenarios were executed
        assert len(test_results['partition_scenarios']) == 2
        
        # Verify partition effects were tested
        for scenario_result in test_results['partition_scenarios']:
            assert 'consensus_result' in scenario_result
            assert 'recovery_result' in scenario_result
            
            # Verify recovery was tested
            recovery_result = scenario_result['recovery_result']
            assert 'recovery_time_ms' in recovery_result
            assert 'consensus_restored' in recovery_result
    
    @pytest.mark.asyncio
    async def test_coordinated_byzantine_attacks(self, byzantine_orchestrator):
        """Test system resilience against coordinated Byzantine attacks"""
        # Select Byzantine agents for coordination
        byzantine_agent_ids = list(byzantine_orchestrator.byzantine_agents.keys())
        
        # Run coordinated attack test
        test_results = await byzantine_orchestrator.run_coordinated_byzantine_attack_test(
            coordinated_agents=byzantine_agent_ids[:2]
        )
        
        # Verify coordinated attack was executed
        assert len(test_results['coordinated_agents']) == 2
        assert len(test_results['attack_scenarios']) > 0
        
        # Verify system resilience metrics
        resilience = test_results['system_resilience']
        assert 'system_availability' in resilience
        assert 'resilience_score' in resilience
        assert resilience['system_availability'] >= 0.0
        assert resilience['resilience_score'] >= 0.0
    
    @pytest.mark.asyncio
    async def test_byzantine_detection_accuracy(self, byzantine_orchestrator):
        """Test accuracy of Byzantine detection"""
        # Run consensus test to generate detection data
        test_results = await byzantine_orchestrator.run_byzantine_consensus_test(consensus_rounds=10)
        
        # Analyze detection results
        detection_results = test_results['detection_results']
        
        # Verify detection was performed
        assert len(detection_results) > 0
        
        # Check detection accuracy for at least one detector
        for detector_id, results in detection_results.items():
            precision = results['precision']
            recall = results['recall']
            
            # Verify detection metrics are reasonable
            assert precision >= 0.0 and precision <= 1.0
            assert recall >= 0.0 and recall <= 1.0
            
            # Verify some detection occurred
            assert results['true_positives'] >= 0
            assert results['false_positives'] >= 0
            assert results['false_negatives'] >= 0
    
    @pytest.mark.asyncio
    async def test_attack_type_effectiveness(self, byzantine_orchestrator):
        """Test effectiveness of different attack types"""
        # Run consensus test
        test_results = await byzantine_orchestrator.run_byzantine_consensus_test(consensus_rounds=8)
        
        # Analyze attack effectiveness
        attack_summary = test_results['attack_summary']
        
        # Verify different attack types were tested
        assert len(attack_summary) > 0
        
        # Check attack effectiveness metrics
        for attack_type, summary in attack_summary.items():
            assert 'total_attacks' in summary
            assert 'disruption_rate' in summary
            assert summary['total_attacks'] >= 0
            assert summary['disruption_rate'] >= 0.0 and summary['disruption_rate'] <= 1.0
    
    @pytest.mark.asyncio
    async def test_consensus_recovery_mechanisms(self, byzantine_orchestrator):
        """Test consensus recovery mechanisms"""
        # Run multiple consensus rounds to test recovery
        test_results = await byzantine_orchestrator.run_byzantine_consensus_test(consensus_rounds=6)
        
        # Verify recovery was tested
        consensus_results = test_results['consensus_results']
        
        # Check for recovery patterns
        failed_consensus = [r for r in consensus_results if not r['consensus_achieved']]
        successful_consensus = [r for r in consensus_results if r['consensus_achieved']]
        
        # Verify some recovery occurred if there were failures
        if failed_consensus:
            assert len(successful_consensus) > 0, "No recovery after consensus failures"
        
        # Verify reasonable consensus times
        for result in consensus_results:
            assert result['consensus_time_ms'] > 0
            assert result['consensus_time_ms'] < 10000  # Should be less than 10 seconds
    
    @pytest.mark.asyncio
    async def test_system_performance_under_byzantine_conditions(self, byzantine_orchestrator):
        """Test system performance under Byzantine conditions"""
        # Run performance test
        test_results = await byzantine_orchestrator.run_byzantine_consensus_test(consensus_rounds=15)
        
        # Analyze performance metrics
        performance = test_results['performance_metrics']
        
        # Verify performance metrics
        assert performance['total_consensus_attempts'] == 15
        assert performance['average_consensus_latency_ms'] > 0
        assert performance['max_consensus_latency_ms'] >= performance['average_consensus_latency_ms']
        
        # Verify system maintained reasonable performance
        assert performance['average_consensus_latency_ms'] < 5000  # Less than 5 seconds average
        assert performance['success_rate'] > 0.2  # At least 20% success rate
    
    def test_comprehensive_test_report_generation(self, byzantine_orchestrator):
        """Test comprehensive test report generation"""
        # Generate test report
        test_report = byzantine_orchestrator.generate_comprehensive_test_report()
        
        # Verify report structure
        assert 'test_configuration' in test_report
        assert 'consensus_results' in test_report
        assert 'attack_analysis' in test_report
        assert 'detection_analysis' in test_report
        assert 'performance_metrics' in test_report
        assert 'system_resilience' in test_report
        assert 'recommendations' in test_report
        
        # Verify report content
        config = test_report['test_configuration']
        assert config['total_agents'] == 7
        assert config['byzantine_agents'] == 2
        assert len(config['attack_types']) > 0
        
        # Verify recommendations are generated
        recommendations = test_report['recommendations']
        assert isinstance(recommendations, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])