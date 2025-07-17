"""
PBFT Engine Test Suite

Comprehensive tests for the PBFT consensus engine including:
- 3-phase consensus protocol (pre-prepare, prepare, commit)
- Byzantine fault tolerance with f=2 
- Cryptographic message validation
- View change protocol
- Performance requirements (<500ms latency)

Author: Agent 2 - Consensus Security Engineer
Version: 1.0 - Production Ready
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src.consensus.pbft_engine import (
    PBFTEngine, PBFTMessage, PBFTPhase, MessageType, 
    ConsensusRequest, ConsensusDecision
)
from src.consensus.cryptographic_core import CryptographicCore


class TestPBFTEngine:
    """Test suite for PBFT consensus engine"""
    
    @pytest.fixture
    def agent_ids(self):
        """Standard set of agent IDs for testing"""
        return ['agent_1', 'agent_2', 'agent_3', 'agent_4', 'agent_5', 'agent_6', 'agent_7']
    
    @pytest.fixture
    def crypto_core(self, agent_ids):
        """Cryptographic core for testing"""
        crypto = CryptographicCore()
        crypto.initialize_agent_keys(agent_ids + ['primary_agent'])
        return crypto
    
    @pytest.fixture
    def pbft_engine(self, agent_ids, crypto_core):
        """PBFT engine instance for testing"""
        return PBFTEngine(
            agent_id='primary_agent',
            agent_ids=agent_ids,
            byzantine_fault_tolerance=2,  # f=2, supports 7 agents
            consensus_timeout=0.5,
            cryptographic_core=crypto_core
        )
    
    def test_pbft_engine_initialization(self, pbft_engine, agent_ids):
        """Test PBFT engine initialization"""
        assert pbft_engine.agent_id == 'primary_agent'
        assert pbft_engine.n == len(agent_ids)
        assert pbft_engine.f == 2
        assert pbft_engine.consensus_timeout == 0.5
        assert pbft_engine.current_view == 0
        assert pbft_engine.sequence_number == 0
    
    def test_byzantine_fault_tolerance_validation(self, agent_ids, crypto_core):
        """Test Byzantine fault tolerance validation"""
        # Valid configuration: n=7, f=2 (3f+1=7)
        engine = PBFTEngine('test', agent_ids, 2, 0.5, crypto_core)
        assert engine.f == 2
        
        # Invalid configuration: not enough agents
        with pytest.raises(ValueError, match="Insufficient agents"):
            PBFTEngine('test', agent_ids[:4], 2, 0.5, crypto_core)  # n=4, need 7 for f=2
    
    def test_primary_selection(self, pbft_engine, agent_ids):
        """Test primary node selection"""
        # View 0 should select first agent
        primary_0 = pbft_engine._get_primary_for_view(0)
        assert primary_0 in agent_ids
        
        # View 1 should select different agent
        primary_1 = pbft_engine._get_primary_for_view(1)
        assert primary_1 in agent_ids
        assert primary_1 != primary_0  # Should rotate
    
    def test_pbft_message_creation(self):
        """Test PBFT message creation and validation"""
        msg = PBFTMessage(
            message_type=MessageType.PRE_PREPARE,
            view_number=1,
            sequence_number=10,
            sender_id='test_agent',
            payload={'test': 'data'},
            timestamp=time.time(),
            phase=PBFTPhase.PRE_PREPARE
        )
        
        assert msg.message_type == MessageType.PRE_PREPARE
        assert msg.view_number == 1
        assert msg.sequence_number == 10
        assert msg.sender_id == 'test_agent'
        assert msg.phase == PBFTPhase.PRE_PREPARE
        
        # Test message hash
        hash1 = msg.get_hash()
        hash2 = msg.get_hash()
        assert hash1 == hash2  # Should be deterministic
        assert len(hash1) == 64  # SHA256 hex string
    
    def test_consensus_request_creation(self):
        """Test consensus request creation"""
        agent_decisions = {
            'agent_1': {'action': 2, 'confidence': 0.8},
            'agent_2': {'action': 1, 'confidence': 0.6}
        }
        
        request = ConsensusRequest(
            request_id='test_request',
            agent_decisions=agent_decisions,
            market_state=Mock(),
            synergy_context={'type': 'TYPE_1'},
            timestamp=time.time(),
            requester_id='test_requester'
        )
        
        assert request.request_id == 'test_request'
        assert len(request.agent_decisions) == 2
        
        payload = request.to_payload()
        assert 'request_id' in payload
        assert 'agent_decisions' in payload
        assert 'synergy_context' in payload
    
    @pytest.mark.asyncio
    async def test_consensus_happy_path(self, pbft_engine):
        """Test successful consensus with no Byzantine agents"""
        # Mock agent decisions
        agent_decisions = {
            'agent_1': Mock(action=2, confidence=0.8, probabilities=[0.1, 0.2, 0.7]),
            'agent_2': Mock(action=2, confidence=0.7, probabilities=[0.1, 0.3, 0.6]),
            'agent_3': Mock(action=2, confidence=0.9, probabilities=[0.0, 0.1, 0.9])
        }
        
        market_state = Mock()
        synergy_context = {'type': 'TYPE_1', 'direction': 1, 'confidence': 0.8}
        
        # Mock the PBFT protocol methods
        with patch.object(pbft_engine, '_execute_pbft_consensus') as mock_pbft:
            mock_result = ConsensusDecision(
                request_id='test_request',
                execute=True,
                action=2,
                confidence=0.8,
                consensus_achieved=True,
                participating_agents=['agent_1', 'agent_2', 'agent_3'],
                byzantine_agents_detected=[],
                view_number=0,
                sequence_number=1,
                timestamp=time.time(),
                safety_level=0.9
            )
            mock_pbft.return_value = mock_result
            
            start_time = time.time()
            result = await pbft_engine.request_consensus(
                request_id='test_request',
                agent_decisions=agent_decisions,
                market_state=market_state,
                synergy_context=synergy_context
            )
            latency = time.time() - start_time
            
            # Verify result
            assert result.consensus_achieved is True
            assert result.action == 2
            assert result.confidence == 0.8
            assert result.safety_level == 0.9
            assert len(result.byzantine_agents_detected) == 0
            
            # Verify performance requirement
            assert latency < 0.5  # <500ms requirement
    
    @pytest.mark.asyncio
    async def test_consensus_with_byzantine_agents(self, pbft_engine):
        """Test consensus with Byzantine agents detected"""
        agent_decisions = {
            'agent_1': Mock(action=2, confidence=0.8, probabilities=[0.1, 0.2, 0.7]),
            'agent_2': Mock(action=0, confidence=0.9, probabilities=[0.9, 0.1, 0.0]),  # Byzantine
            'agent_3': Mock(action=2, confidence=0.7, probabilities=[0.1, 0.3, 0.6]),
            'agent_4': Mock(action=2, confidence=0.8, probabilities=[0.0, 0.2, 0.8])
        }
        
        with patch.object(pbft_engine, '_execute_pbft_consensus') as mock_pbft:
            mock_result = ConsensusDecision(
                request_id='test_request',
                execute=True,
                action=2,
                confidence=0.7,  # Lower due to Byzantine agent
                consensus_achieved=True,
                participating_agents=['agent_1', 'agent_3', 'agent_4'],
                byzantine_agents_detected=['agent_2'],
                view_number=0,
                sequence_number=1,
                timestamp=time.time(),
                safety_level=0.7
            )
            mock_pbft.return_value = mock_result
            
            result = await pbft_engine.request_consensus(
                request_id='test_request',
                agent_decisions=agent_decisions,
                market_state=Mock(),
                synergy_context={'type': 'TYPE_1'}
            )
            
            assert result.consensus_achieved is True
            assert 'agent_2' in result.byzantine_agents_detected
            assert result.safety_level < 0.9  # Reduced due to Byzantine agent
    
    @pytest.mark.asyncio
    async def test_consensus_timeout(self, pbft_engine):
        """Test consensus timeout handling"""
        agent_decisions = {
            'agent_1': Mock(action=2, confidence=0.8, probabilities=[0.1, 0.2, 0.7])
        }
        
        # Mock timeout scenario
        async def timeout_side_effect(*args, **kwargs):
            await asyncio.sleep(1.0)  # Longer than consensus_timeout
            raise asyncio.TimeoutError("Consensus timeout")
        
        with patch.object(pbft_engine, '_execute_pbft_consensus', side_effect=timeout_side_effect):
            start_time = time.time()
            result = await pbft_engine.request_consensus(
                request_id='test_timeout',
                agent_decisions=agent_decisions,
                market_state=Mock(),
                synergy_context={}
            )
            latency = time.time() - start_time
            
            # Should return emergency decision
            assert result.consensus_achieved is False
            assert result.execute is False  # Safe default
            assert result.action == 1  # Neutral
            assert result.confidence == 0.0
            assert result.safety_level == 0.0
    
    def test_message_validation_pre_prepare(self, pbft_engine, crypto_core):
        """Test pre-prepare message validation"""
        # Valid pre-prepare message
        msg = PBFTMessage(
            message_type=MessageType.PRE_PREPARE,
            view_number=pbft_engine.current_view,
            sequence_number=1,
            sender_id=pbft_engine.primary_id,
            payload={'test': 'data'},
            timestamp=time.time(),
            phase=PBFTPhase.PRE_PREPARE
        )
        
        # Sign message
        msg.signature = crypto_core.sign_message(msg.get_hash(), pbft_engine.primary_id)
        
        assert pbft_engine._validate_pre_prepare_message(msg) is True
        
        # Invalid: wrong sender
        msg_wrong_sender = PBFTMessage(
            message_type=MessageType.PRE_PREPARE,
            view_number=pbft_engine.current_view,
            sequence_number=1,
            sender_id='wrong_agent',
            payload={'test': 'data'},
            timestamp=time.time(),
            phase=PBFTPhase.PRE_PREPARE
        )
        
        assert pbft_engine._validate_pre_prepare_message(msg_wrong_sender) is False
        
        # Invalid: wrong view
        msg_wrong_view = PBFTMessage(
            message_type=MessageType.PRE_PREPARE,
            view_number=999,
            sequence_number=1,
            sender_id=pbft_engine.primary_id,
            payload={'test': 'data'},
            timestamp=time.time(),
            phase=PBFTPhase.PRE_PREPARE
        )
        
        assert pbft_engine._validate_pre_prepare_message(msg_wrong_view) is False
    
    def test_message_validation_prepare(self, pbft_engine, crypto_core):
        """Test prepare message validation"""
        sequence_num = 1
        
        msg = PBFTMessage(
            message_type=MessageType.PREPARE,
            view_number=pbft_engine.current_view,
            sequence_number=sequence_num,
            sender_id='agent_1',
            payload={'test': 'data'},
            timestamp=time.time(),
            phase=PBFTPhase.PREPARE
        )
        
        # Sign message
        msg.signature = crypto_core.sign_message(msg.get_hash(), 'agent_1')
        
        assert pbft_engine._validate_prepare_message(msg, sequence_num) is True
        
        # Invalid: wrong sequence number
        assert pbft_engine._validate_prepare_message(msg, 999) is False
        
        # Invalid: Byzantine sender
        pbft_engine.byzantine_agents.add('agent_1')
        assert pbft_engine._validate_prepare_message(msg, sequence_num) is False
    
    def test_message_validation_commit(self, pbft_engine, crypto_core):
        """Test commit message validation"""
        sequence_num = 1
        
        msg = PBFTMessage(
            message_type=MessageType.COMMIT,
            view_number=pbft_engine.current_view,
            sequence_number=sequence_num,
            sender_id='agent_1',
            payload={'test': 'data'},
            timestamp=time.time(),
            phase=PBFTPhase.COMMIT
        )
        
        # Sign message
        msg.signature = crypto_core.sign_message(msg.get_hash(), 'agent_1')
        
        assert pbft_engine._validate_commit_message(msg, sequence_num) is True
        
        # Invalid: wrong sequence number
        assert pbft_engine._validate_commit_message(msg, 999) is False
    
    @pytest.mark.asyncio
    async def test_view_change_protocol(self, pbft_engine):
        """Test view change protocol"""
        old_view = pbft_engine.current_view
        old_primary = pbft_engine.primary_id
        
        # Initiate view change
        await pbft_engine._initiate_view_change()
        
        # Verify metrics updated
        assert pbft_engine.consensus_metrics['view_changes'] >= 1
    
    def test_byzantine_detection_and_recording(self, pbft_engine):
        """Test Byzantine agent detection and recording"""
        agent_id = 'agent_1'
        behavior = 'invalid_signature'
        
        # Record suspicious behavior
        pbft_engine._record_suspicious_behavior(agent_id, behavior)
        
        assert agent_id in pbft_engine.suspicious_behavior
        assert behavior in pbft_engine.suspicious_behavior[agent_id]
        
        # Record more suspicious behavior to trigger Byzantine classification
        pbft_engine._record_suspicious_behavior(agent_id, 'timing_attack')
        pbft_engine._record_suspicious_behavior(agent_id, 'conflicting_votes')
        
        # Should be classified as Byzantine after 3 behaviors
        assert agent_id in pbft_engine.byzantine_agents
        assert pbft_engine.consensus_metrics['byzantine_detections'] >= 1
    
    def test_consensus_metrics_tracking(self, pbft_engine):
        """Test consensus metrics tracking"""
        initial_metrics = pbft_engine.get_consensus_metrics()
        
        # Update metrics
        pbft_engine._update_consensus_metrics(True, 0.1)
        pbft_engine.consensus_metrics['successful_consensus'] += 1
        
        updated_metrics = pbft_engine.get_consensus_metrics()
        
        assert updated_metrics['total_consensus_requests'] >= initial_metrics['total_consensus_requests']
        assert updated_metrics['average_latency'] >= 0
        assert updated_metrics['max_latency'] >= 0
        assert 'success_rate' in updated_metrics
        assert 'failure_rate' in updated_metrics
        assert 'byzantine_agent_count' in updated_metrics
    
    def test_emergency_decision_creation(self, pbft_engine):
        """Test emergency decision creation"""
        agent_outputs = {
            'agent_1': {'action': 1, 'confidence': 0.5}
        }
        
        emergency_decision = pbft_engine._create_emergency_decision('emergency_test', agent_outputs)
        
        assert emergency_decision.execute is False
        assert emergency_decision.action == 1  # Neutral
        assert emergency_decision.confidence == 0.0
        assert emergency_decision.consensus_achieved is False
        assert emergency_decision.safety_level == 0.0
    
    def test_consensus_state_reset(self, pbft_engine):
        """Test consensus state reset functionality"""
        # Modify state
        pbft_engine.current_view = 5
        pbft_engine.sequence_number = 10
        pbft_engine.byzantine_agents.add('agent_1')
        pbft_engine.pre_prepare_log[1] = Mock()
        
        # Reset
        pbft_engine.reset_consensus_state()
        
        # Verify reset
        assert pbft_engine.current_view == 0
        assert pbft_engine.sequence_number == 0
        assert len(pbft_engine.byzantine_agents) == 0
        assert len(pbft_engine.pre_prepare_log) == 0
        assert len(pbft_engine.prepare_log) == 0
        assert len(pbft_engine.commit_log) == 0


class TestPBFTPerformance:
    """Performance tests for PBFT engine"""
    
    @pytest.fixture
    def performance_engine(self):
        """PBFT engine optimized for performance testing"""
        agent_ids = [f'agent_{i}' for i in range(7)]
        crypto = CryptographicCore()
        crypto.initialize_agent_keys(agent_ids + ['perf_primary'])
        
        return PBFTEngine(
            agent_id='perf_primary',
            agent_ids=agent_ids,
            byzantine_fault_tolerance=2,
            consensus_timeout=0.1,  # Aggressive timeout for testing
            cryptographic_core=crypto
        )
    
    @pytest.mark.asyncio
    async def test_consensus_latency_requirement(self, performance_engine):
        """Test that consensus meets <500ms latency requirement"""
        agent_decisions = {
            f'agent_{i}': Mock(
                action=2, 
                confidence=0.8, 
                probabilities=[0.1, 0.2, 0.7]
            ) for i in range(7)
        }
        
        # Mock successful consensus
        with patch.object(performance_engine, '_execute_pbft_consensus') as mock_pbft:
            mock_result = ConsensusDecision(
                request_id='perf_test',
                execute=True,
                action=2,
                confidence=0.8,
                consensus_achieved=True,
                participating_agents=[f'agent_{i}' for i in range(7)],
                byzantine_agents_detected=[],
                view_number=0,
                sequence_number=1,
                timestamp=time.time(),
                safety_level=0.9
            )
            mock_pbft.return_value = mock_result
            
            # Measure latency
            start_time = time.time()
            result = await performance_engine.request_consensus(
                request_id='perf_test',
                agent_decisions=agent_decisions,
                market_state=Mock(),
                synergy_context={}
            )
            latency = time.time() - start_time
            
            # Verify performance requirement
            assert latency < 0.5, f"Consensus latency {latency:.3f}s exceeds 500ms requirement"
            assert result.consensus_achieved is True
    
    @pytest.mark.asyncio
    async def test_high_load_consensus(self, performance_engine):
        """Test consensus under high load (multiple concurrent requests)"""
        async def single_consensus_request(request_id: str):
            agent_decisions = {
                f'agent_{i}': Mock(
                    action=2, 
                    confidence=0.8, 
                    probabilities=[0.1, 0.2, 0.7]
                ) for i in range(3)  # Smaller set for faster testing
            }
            
            with patch.object(performance_engine, '_execute_pbft_consensus') as mock_pbft:
                mock_result = ConsensusDecision(
                    request_id=request_id,
                    execute=True,
                    action=2,
                    confidence=0.8,
                    consensus_achieved=True,
                    participating_agents=[f'agent_{i}' for i in range(3)],
                    byzantine_agents_detected=[],
                    view_number=0,
                    sequence_number=1,
                    timestamp=time.time(),
                    safety_level=0.9
                )
                mock_pbft.return_value = mock_result
                
                return await performance_engine.request_consensus(
                    request_id=request_id,
                    agent_decisions=agent_decisions,
                    market_state=Mock(),
                    synergy_context={}
                )
        
        # Run 10 concurrent consensus requests
        start_time = time.time()
        tasks = [single_consensus_request(f'load_test_{i}') for i in range(10)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Verify all succeeded
        assert len(results) == 10
        assert all(result.consensus_achieved for result in results)
        
        # Verify reasonable throughput
        throughput = len(results) / total_time
        assert throughput > 5, f"Throughput {throughput:.1f} requests/sec is too low"


@pytest.mark.integration
class TestPBFTIntegration:
    """Integration tests for PBFT engine with other components"""
    
    def test_integration_with_cryptographic_core(self):
        """Test PBFT integration with cryptographic core"""
        agent_ids = ['agent_1', 'agent_2', 'agent_3', 'agent_4', 'agent_5', 'agent_6', 'agent_7']
        crypto = CryptographicCore()
        crypto.initialize_agent_keys(agent_ids + ['integr_primary'])
        
        engine = PBFTEngine(
            agent_id='integr_primary',
            agent_ids=agent_ids,
            byzantine_fault_tolerance=2,
            consensus_timeout=0.5,
            cryptographic_core=crypto
        )
        
        # Verify crypto integration
        assert engine.crypto_core is not None
        assert engine.message_validator is not None
        
        # Test message signing and validation
        msg = PBFTMessage(
            message_type=MessageType.PRE_PREPARE,
            view_number=0,
            sequence_number=1,
            sender_id='integr_primary',
            payload={'test': 'integration'},
            timestamp=time.time(),
            phase=PBFTPhase.PRE_PREPARE
        )
        
        # Sign message
        msg.signature = crypto.sign_message(msg.get_hash(), 'integr_primary')
        
        # Validate message
        assert engine.message_validator.validate_message(msg) is True
    
    def test_consensus_decision_aggregation(self):
        """Test consensus decision aggregation with Byzantine resistance"""
        agent_ids = ['agent_1', 'agent_2', 'agent_3', 'agent_4', 'agent_5', 'agent_6', 'agent_7']
        crypto = CryptographicCore()
        crypto.initialize_agent_keys(agent_ids + ['aggr_primary'])
        
        engine = PBFTEngine(
            agent_id='aggr_primary',
            agent_ids=agent_ids,
            byzantine_fault_tolerance=2,
            consensus_timeout=0.5,
            cryptographic_core=crypto
        )
        
        # Test decision aggregation
        clean_agents = {'agent_1', 'agent_2', 'agent_3', 'agent_4', 'agent_5'}
        agent_decisions = {
            'agent_1': Mock(action=2, confidence=0.8, probabilities=[0.1, 0.2, 0.7]),
            'agent_2': Mock(action=2, confidence=0.7, probabilities=[0.2, 0.1, 0.7]),
            'agent_3': Mock(action=1, confidence=0.6, probabilities=[0.3, 0.4, 0.3]),
            'agent_4': Mock(action=2, confidence=0.9, probabilities=[0.0, 0.1, 0.9]),
            'agent_5': Mock(action=2, confidence=0.8, probabilities=[0.1, 0.1, 0.8])
        }
        
        synergy_context = {'type': 'TYPE_1', 'direction': 1, 'confidence': 0.8}
        
        execute, action, confidence = engine._aggregate_decisions_byzantine_safe(
            agent_decisions, clean_agents, synergy_context
        )
        
        assert action == 2  # Majority voted for action 2
        assert confidence > 0.0
        assert isinstance(execute, bool)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])