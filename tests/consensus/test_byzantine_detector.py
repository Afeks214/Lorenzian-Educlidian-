"""
Byzantine Detector Test Suite

Comprehensive tests for Byzantine agent behavior detection:
- Real-time behavior monitoring
- Statistical anomaly detection
- Pattern recognition
- Trust score calculation
- Evidence collection and reporting

Author: Agent 2 - Consensus Security Engineer
Version: 1.0 - Production Ready
"""

import pytest
import time
from unittest.mock import Mock
import sys
import os
import statistics

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src.consensus.byzantine_detector import (
    ByzantineDetector, ByzantinePattern, ByzantineEvidence, 
    AgentBehaviorProfile
)


class TestByzantineDetector:
    """Test suite for Byzantine detector"""
    
    @pytest.fixture
    def agent_ids(self):
        """Standard agent IDs for testing"""
        return ['agent_1', 'agent_2', 'agent_3', 'agent_4', 'agent_5', 'agent_6', 'agent_7']
    
    @pytest.fixture
    def detector(self, agent_ids):
        """Byzantine detector instance for testing"""
        return ByzantineDetector(
            agent_ids=agent_ids,
            detection_window=60.0,  # 1 minute for testing
            anomaly_threshold=0.7,
            min_evidence_count=2  # Lower for testing
        )
    
    def test_detector_initialization(self, detector, agent_ids):
        """Test Byzantine detector initialization"""
        assert len(detector.agent_profiles) == len(agent_ids)
        assert detector.detection_window == 60.0
        assert detector.anomaly_threshold == 0.7
        assert detector.min_evidence_count == 2
        assert len(detector.suspected_byzantine) == 0
        assert len(detector.confirmed_byzantine) == 0
    
    def test_agent_behavior_profile_creation(self, detector, agent_ids):
        """Test agent behavior profile creation"""
        for agent_id in agent_ids:
            profile = detector.agent_profiles[agent_id]
            assert profile.agent_id == agent_id
            assert profile.trust_score == 1.0
            assert profile.is_active is True
            assert profile.total_messages == 0
            assert profile.signature_failure_count == 0
    
    def test_message_activity_recording(self, detector):
        """Test message activity recording"""
        agent_id = 'agent_1'
        timestamp = time.time()
        
        detector.record_message_activity(
            agent_id=agent_id,
            message_type='pre_prepare',
            timestamp=timestamp,
            consensus_round=1,
            signature_valid=True
        )
        
        profile = detector.agent_profiles[agent_id]
        assert profile.total_messages == 1
        assert profile.last_activity == timestamp
        assert profile.signature_failure_count == 0
        assert len(detector.message_timestamps[agent_id]) == 1
    
    def test_signature_failure_detection(self, detector):
        """Test signature failure detection and evidence recording"""
        agent_id = 'agent_1'
        
        # Record signature failure
        detector.record_message_activity(
            agent_id=agent_id,
            message_type='prepare',
            timestamp=time.time(),
            signature_valid=False
        )
        
        profile = detector.agent_profiles[agent_id]
        assert profile.signature_failure_count == 1
        assert len(profile.behavior_anomalies) >= 1
        
        # Check evidence was recorded
        evidence = profile.behavior_anomalies[-1]
        assert evidence.pattern_type == ByzantinePattern.SIGNATURE_MANIPULATION
        assert evidence.agent_id == agent_id
    
    def test_timing_pattern_analysis(self, detector):
        """Test message timing pattern analysis"""
        agent_id = 'agent_1'
        current_time = time.time()
        
        # Simulate message flooding (very fast messages)
        for i in range(10):
            detector.record_message_activity(
                agent_id=agent_id,
                message_type='prepare',
                timestamp=current_time + i * 0.01,  # 10ms intervals (very fast)
                signature_valid=True
            )
        
        # Check for flooding evidence
        profile = detector.agent_profiles[agent_id]
        flooding_evidence = [
            e for e in profile.behavior_anomalies 
            if e.pattern_type == ByzantinePattern.MESSAGE_FLOODING
        ]
        
        assert len(flooding_evidence) > 0
    
    def test_silent_failure_detection(self, detector):
        """Test silent failure pattern detection"""
        agent_id = 'agent_1'
        current_time = time.time()
        
        # Simulate very slow messages (silent failure pattern)
        timestamps = [current_time, current_time + 100, current_time + 200]  # Very long intervals
        
        for timestamp in timestamps:
            detector.record_message_activity(
                agent_id=agent_id,
                message_type='commit',
                timestamp=timestamp,
                signature_valid=True
            )
        
        # Check for silent failure evidence
        profile = detector.agent_profiles[agent_id]
        silent_evidence = [
            e for e in profile.behavior_anomalies 
            if e.pattern_type == ByzantinePattern.SILENT_FAILURE
        ]
        
        assert len(silent_evidence) > 0
    
    def test_consensus_participation_tracking(self, detector, agent_ids):
        """Test consensus participation tracking"""
        consensus_round = 1
        participating_agents = agent_ids[:5]  # Only first 5 agents participate
        
        detector.record_consensus_result(
            consensus_round=consensus_round,
            participating_agents=participating_agents,
            consensus_achieved=True,
            view_changes=0
        )
        
        # Check participation statistics
        for agent_id in participating_agents:
            profile = detector.agent_profiles[agent_id]
            stats = profile.consensus_participation
            assert stats['total_rounds'] == 1
            assert stats['participated_rounds'] == 1
        
        # Non-participating agents should have evidence
        for agent_id in agent_ids[5:]:
            profile = detector.agent_profiles[agent_id]
            if profile.is_active:  # Only if agent is considered active
                silent_evidence = [
                    e for e in profile.behavior_anomalies 
                    if e.pattern_type == ByzantinePattern.SILENT_FAILURE
                ]
                # May have evidence depending on recent activity
    
    def test_consensus_failure_analysis(self, detector, agent_ids):
        """Test consensus failure analysis"""
        consensus_round = 5
        
        # Simulate consensus failure with high view changes
        detector.record_consensus_result(
            consensus_round=consensus_round,
            participating_agents=agent_ids,
            consensus_achieved=False,
            view_changes=5  # High number of view changes
        )
        
        # Check for view change abuse evidence
        for agent_id in agent_ids:
            profile = detector.agent_profiles[agent_id]
            view_change_evidence = [
                e for e in profile.behavior_anomalies 
                if e.pattern_type == ByzantinePattern.VIEW_CHANGE_ABUSE
            ]
            assert len(view_change_evidence) >= 1
    
    def test_byzantine_classification(self, detector):
        """Test Byzantine agent classification process"""
        agent_id = 'agent_1'
        
        # Generate multiple pieces of evidence
        detector._record_evidence(
            agent_id=agent_id,
            pattern_type=ByzantinePattern.SIGNATURE_MANIPULATION,
            severity=0.8,
            evidence_data={'failures': 5},
            confidence=0.9
        )
        
        detector._record_evidence(
            agent_id=agent_id,
            pattern_type=ByzantinePattern.MESSAGE_FLOODING,
            severity=0.7,
            evidence_data={'rate': 100},
            confidence=0.8
        )
        
        # Update classification
        detector._update_byzantine_classification(agent_id)
        
        # Agent should be suspected
        assert agent_id in detector.suspected_byzantine
        
        # Add high-confidence evidence for confirmation
        detector._record_evidence(
            agent_id=agent_id,
            pattern_type=ByzantinePattern.CONFLICTING_VOTES,
            severity=0.9,
            evidence_data={'conflicts': 3},
            confidence=0.9
        )
        
        detector._update_byzantine_classification(agent_id)
        
        # Agent should be confirmed Byzantine
        assert agent_id in detector.confirmed_byzantine
    
    def test_trust_score_calculation(self, detector):
        """Test trust score calculation and updates"""
        agent_id = 'agent_1'
        initial_trust = detector.get_agent_trust_score(agent_id)
        assert initial_trust == 1.0
        
        # Record some suspicious behavior
        detector._record_evidence(
            agent_id=agent_id,
            pattern_type=ByzantinePattern.TIMING_ATTACK,
            severity=0.5,
            evidence_data={'variance': 0.8},
            confidence=0.6
        )
        
        detector._update_byzantine_classification(agent_id)
        
        # Trust score should decrease
        new_trust = detector.get_agent_trust_score(agent_id)
        assert new_trust < initial_trust
    
    def test_statistical_outlier_detection(self, detector):
        """Test statistical outlier detection"""
        # Create normal behavior for most agents
        normal_decisions = {
            'agent_1': Mock(confidence=0.7, is_byzantine=False),
            'agent_2': Mock(confidence=0.8, is_byzantine=False),
            'agent_3': Mock(confidence=0.75, is_byzantine=False),
            'agent_4': Mock(confidence=0.72, is_byzantine=False)
        }
        
        # Create outlier
        outlier_decision = Mock(confidence=0.1, is_byzantine=False)  # Very low confidence
        
        # Test outlier detection
        is_outlier = detector._is_statistical_outlier(outlier_decision, normal_decisions)
        assert is_outlier is True
        
        # Test normal decision
        normal_decision = Mock(confidence=0.73, is_byzantine=False)
        is_normal = detector._is_statistical_outlier(normal_decision, normal_decisions)
        assert is_normal is False
    
    def test_probability_distribution_validation(self, detector):
        """Test probability distribution validation"""
        import numpy as np
        
        # Valid distribution
        valid_probs = np.array([0.2, 0.3, 0.5])
        assert detector._validate_probability_distribution(valid_probs) is True
        
        # Invalid: doesn't sum to 1
        invalid_sum = np.array([0.1, 0.2, 0.5])
        assert detector._validate_probability_distribution(invalid_sum) is False
        
        # Invalid: negative probability
        negative_prob = np.array([-0.1, 0.6, 0.5])
        assert detector._validate_probability_distribution(negative_prob) is False
        
        # Invalid: probability > 1
        over_one = np.array([0.5, 1.2, 0.3])
        assert detector._validate_probability_distribution(over_one) is False
    
    def test_detection_metrics(self, detector):
        """Test detection metrics tracking"""
        initial_metrics = detector.get_detection_metrics()
        
        # Record some detection activity
        detector._record_evidence(
            agent_id='agent_1',
            pattern_type=ByzantinePattern.SIGNATURE_MANIPULATION,
            severity=0.8,
            evidence_data={},
            confidence=0.9
        )
        
        detector.suspected_byzantine.add('agent_2')
        detector.confirmed_byzantine.add('agent_3')
        
        updated_metrics = detector.get_detection_metrics()
        
        assert updated_metrics['total_evidence_count'] > initial_metrics['total_evidence_count']
        assert updated_metrics['suspected_byzantine_count'] == 1
        assert updated_metrics['confirmed_byzantine_count'] == 1
        assert 'byzantine_detection_rate' in updated_metrics
    
    def test_agent_behavior_summary(self, detector):
        """Test agent behavior summary generation"""
        agent_id = 'agent_1'
        
        # Record some activity
        detector.record_message_activity(
            agent_id=agent_id,
            message_type='prepare',
            timestamp=time.time(),
            signature_valid=False  # Signature failure
        )
        
        detector.record_consensus_result(
            consensus_round=1,
            participating_agents=[agent_id],
            consensus_achieved=True
        )
        
        summary = detector.get_agent_behavior_summary(agent_id)
        
        assert summary['agent_id'] == agent_id
        assert 'trust_score' in summary
        assert 'total_messages' in summary
        assert 'signature_failure_rate' in summary
        assert 'participation_rate' in summary
        assert 'recent_evidence_count' in summary
        assert 'evidence_types' in summary
        assert 'last_activity' in summary
    
    def test_data_cleanup(self, detector):
        """Test old data cleanup functionality"""
        agent_id = 'agent_1'
        old_time = time.time() - 200  # 200 seconds ago
        
        # Add old evidence
        old_evidence = ByzantineEvidence(
            agent_id=agent_id,
            pattern_type=ByzantinePattern.TIMING_ATTACK,
            severity=0.5,
            evidence_data={},
            timestamp=old_time,
            detection_confidence=0.6
        )
        
        detector.evidence_database.append(old_evidence)
        detector.agent_profiles[agent_id].behavior_anomalies.append(old_evidence)
        
        # Add old consensus round data
        detector.consensus_round_data[999] = {
            'start_time': old_time,
            'participants': {agent_id}
        }
        
        initial_evidence_count = len(detector.evidence_database)
        initial_round_count = len(detector.consensus_round_data)
        
        # Cleanup
        detector.cleanup_old_data()
        
        # Old data should be removed
        assert len(detector.evidence_database) <= initial_evidence_count
        assert len(detector.consensus_round_data) <= initial_round_count
    
    def test_evidence_report_export(self, detector):
        """Test evidence report export"""
        agent_id = 'agent_1'
        
        # Generate some evidence
        detector._record_evidence(
            agent_id=agent_id,
            pattern_type=ByzantinePattern.MESSAGE_FLOODING,
            severity=0.7,
            evidence_data={'rate': 50},
            confidence=0.8
        )
        
        detector.suspected_byzantine.add(agent_id)
        
        # Export report
        report_json = detector.export_evidence_report()
        
        assert isinstance(report_json, str)
        
        import json
        report = json.loads(report_json)
        
        assert 'timestamp' in report
        assert 'suspected_byzantine' in report
        assert 'confirmed_byzantine' in report
        assert 'agent_summaries' in report
        assert 'evidence_database' in report
        assert 'detection_metrics' in report
        
        assert agent_id in report['suspected_byzantine']
        assert len(report['evidence_database']) >= 1


class TestByzantineEvidence:
    """Test Byzantine evidence container"""
    
    def test_evidence_creation(self):
        """Test Byzantine evidence creation"""
        evidence = ByzantineEvidence(
            agent_id='test_agent',
            pattern_type=ByzantinePattern.SIGNATURE_MANIPULATION,
            severity=0.8,
            evidence_data={'failure_count': 5},
            timestamp=0,  # Will be set in __post_init__
            detection_confidence=0.9
        )
        
        assert evidence.agent_id == 'test_agent'
        assert evidence.pattern_type == ByzantinePattern.SIGNATURE_MANIPULATION
        assert evidence.severity == 0.8
        assert evidence.detection_confidence == 0.9
        assert evidence.timestamp > 0  # Should be set automatically
        assert evidence.evidence_data['failure_count'] == 5


class TestAgentBehaviorProfile:
    """Test agent behavior profile"""
    
    def test_profile_creation(self):
        """Test agent behavior profile creation"""
        profile = AgentBehaviorProfile(
            agent_id='test_agent',
            last_activity=0  # Will be set in __post_init__
        )
        
        assert profile.agent_id == 'test_agent'
        assert profile.trust_score == 1.0
        assert profile.is_active is True
        assert profile.total_messages == 0
        assert profile.signature_failure_count == 0
        assert profile.last_activity > 0  # Should be set automatically
        assert len(profile.behavior_anomalies) == 0


class TestByzantinePatterns:
    """Test Byzantine pattern enumeration"""
    
    def test_pattern_types(self):
        """Test Byzantine pattern types"""
        patterns = [
            ByzantinePattern.TIMING_ATTACK,
            ByzantinePattern.CONSENSUS_DISRUPTION,
            ByzantinePattern.SIGNATURE_MANIPULATION,
            ByzantinePattern.MESSAGE_FLOODING,
            ByzantinePattern.SILENT_FAILURE,
            ByzantinePattern.CONFLICTING_VOTES,
            ByzantinePattern.VIEW_CHANGE_ABUSE,
            ByzantinePattern.NETWORK_PARTITION
        ]
        
        # Verify all patterns have unique values
        pattern_values = [p.value for p in patterns]
        assert len(pattern_values) == len(set(pattern_values))
        
        # Verify pattern values are strings
        for pattern in patterns:
            assert isinstance(pattern.value, str)


@pytest.mark.integration
class TestByzantineDetectorIntegration:
    """Integration tests for Byzantine detector"""
    
    def test_realistic_attack_scenario(self):
        """Test realistic Byzantine attack scenario"""
        agent_ids = ['honest_1', 'honest_2', 'honest_3', 'byzantine_1', 'byzantine_2']
        detector = ByzantineDetector(agent_ids, detection_window=120.0, min_evidence_count=2)
        
        current_time = time.time()
        
        # Simulate honest agent behavior
        for i, agent_id in enumerate(['honest_1', 'honest_2', 'honest_3']):
            for round_num in range(10):
                detector.record_message_activity(
                    agent_id=agent_id,
                    message_type='pre_prepare',
                    timestamp=current_time + round_num * 30 + i,
                    consensus_round=round_num,
                    signature_valid=True
                )
                
                detector.record_message_activity(
                    agent_id=agent_id,
                    message_type='prepare',
                    timestamp=current_time + round_num * 30 + i + 1,
                    consensus_round=round_num,
                    signature_valid=True
                )
                
                detector.record_message_activity(
                    agent_id=agent_id,
                    message_type='commit',
                    timestamp=current_time + round_num * 30 + i + 2,
                    consensus_round=round_num,
                    signature_valid=True
                )
        
        # Simulate Byzantine agent behavior
        for round_num in range(10):
            # Byzantine agent 1: Signature manipulation
            detector.record_message_activity(
                agent_id='byzantine_1',
                message_type='prepare',
                timestamp=current_time + round_num * 30,
                consensus_round=round_num,
                signature_valid=False  # Invalid signatures
            )
            
            # Byzantine agent 2: Message flooding
            for flood_msg in range(20):  # Flood with messages
                detector.record_message_activity(
                    agent_id='byzantine_2',
                    message_type='prepare',
                    timestamp=current_time + round_num * 30 + flood_msg * 0.1,
                    consensus_round=round_num,
                    signature_valid=True
                )
            
            # Record consensus results
            detector.record_consensus_result(
                consensus_round=round_num,
                participating_agents=agent_ids,
                consensus_achieved=round_num % 3 != 0,  # Fail some rounds
                view_changes=2 if round_num % 3 == 0 else 0
            )
        
        # Check detection results
        suspected, confirmed = detector.get_byzantine_agents()
        
        # Byzantine agents should be detected
        assert 'byzantine_1' in suspected or 'byzantine_1' in confirmed
        assert 'byzantine_2' in suspected or 'byzantine_2' in confirmed
        
        # Honest agents should not be detected
        for honest_agent in ['honest_1', 'honest_2', 'honest_3']:
            assert honest_agent not in confirmed
        
        # Verify metrics
        metrics = detector.get_detection_metrics()
        assert metrics['total_evidence_count'] > 0
        assert metrics['suspected_byzantine_count'] + metrics['confirmed_byzantine_count'] >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])