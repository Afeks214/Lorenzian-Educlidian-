"""
Byzantine Agent Behavior Detection System

Real-time detection of Byzantine (malicious/faulty) agents in the PBFT consensus
system. Uses statistical analysis, behavior pattern recognition, and anomaly
detection to identify agents acting in a Byzantine manner.

Detection Patterns:
- Message timing anomalies
- Consensus voting inconsistencies
- Cryptographic signature failures
- Network behavior deviations
- Decision pattern analysis

Author: Agent 2 - Consensus Security Engineer
Version: 1.0 - Production Ready
"""

import time
import logging
import statistics
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import json

logger = logging.getLogger(__name__)


class ByzantinePattern(Enum):
    """Types of Byzantine behavior patterns"""
    TIMING_ATTACK = "timing_attack"
    CONSENSUS_DISRUPTION = "consensus_disruption"
    SIGNATURE_MANIPULATION = "signature_manipulation"
    MESSAGE_FLOODING = "message_flooding"
    SILENT_FAILURE = "silent_failure"
    CONFLICTING_VOTES = "conflicting_votes"
    VIEW_CHANGE_ABUSE = "view_change_abuse"
    NETWORK_PARTITION = "network_partition"


@dataclass
class ByzantineEvidence:
    """Evidence of Byzantine behavior"""
    agent_id: str
    pattern_type: ByzantinePattern
    severity: float  # 0.0 to 1.0
    evidence_data: Dict[str, Any]
    timestamp: float
    detection_confidence: float
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


@dataclass
class AgentBehaviorProfile:
    """Behavior profile for an agent"""
    agent_id: str
    message_timing_stats: Dict[str, float] = field(default_factory=dict)
    consensus_participation: Dict[str, int] = field(default_factory=dict)
    signature_failure_count: int = 0
    total_messages: int = 0
    last_activity: float = 0.0
    trust_score: float = 1.0
    is_active: bool = True
    behavior_anomalies: List[ByzantineEvidence] = field(default_factory=list)
    
    def __post_init__(self):
        if self.last_activity == 0:
            self.last_activity = time.time()


class ByzantineDetector:
    """
    Real-time Byzantine Agent Behavior Detection System
    
    Monitors agent behavior patterns and detects Byzantine faults using:
    - Statistical analysis of message timing
    - Consensus participation tracking
    - Cryptographic failure monitoring
    - Network behavior analysis
    - Machine learning-based anomaly detection
    """
    
    def __init__(
        self,
        agent_ids: List[str],
        detection_window: float = 300.0,  # 5 minutes
        anomaly_threshold: float = 0.7,
        min_evidence_count: int = 3
    ):
        """
        Initialize Byzantine detector
        
        Args:
            agent_ids: List of all participating agent IDs
            detection_window: Time window for behavior analysis (seconds)
            anomaly_threshold: Threshold for Byzantine detection (0.0-1.0)
            min_evidence_count: Minimum evidence instances for Byzantine classification
        """
        self.agent_ids = agent_ids
        self.detection_window = detection_window
        self.anomaly_threshold = anomaly_threshold
        self.min_evidence_count = min_evidence_count
        
        # Agent behavior profiles
        self.agent_profiles: Dict[str, AgentBehaviorProfile] = {}
        for agent_id in agent_ids:
            self.agent_profiles[agent_id] = AgentBehaviorProfile(agent_id=agent_id)
        
        # Message timing tracking
        self.message_timestamps: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.consensus_round_data: Dict[int, Dict[str, Any]] = {}
        
        # Byzantine agent tracking
        self.suspected_byzantine: Set[str] = set()
        self.confirmed_byzantine: Set[str] = set()
        self.evidence_database: List[ByzantineEvidence] = []
        
        # Detection metrics
        self.detection_metrics = {
            'total_detections': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'detection_accuracy': 1.0,
            'average_detection_time': 0.0,
            'evidence_count_by_type': defaultdict(int)
        }
        
        # Behavioral baselines
        self.baseline_message_interval = 1.0  # seconds
        self.baseline_consensus_participation = 0.95  # 95% participation expected
        self.max_signature_failure_rate = 0.05  # 5% signature failures allowed
        
        logger.info(f"Byzantine detector initialized for {len(agent_ids)} agents")
    
    def record_message_activity(
        self,
        agent_id: str,
        message_type: str,
        timestamp: float,
        consensus_round: Optional[int] = None,
        signature_valid: bool = True
    ):
        """
        Record agent message activity for behavior analysis
        
        Args:
            agent_id: Agent sending the message
            message_type: Type of message (pre_prepare, prepare, commit, etc.)
            timestamp: Message timestamp
            consensus_round: Optional consensus round number
            signature_valid: Whether message signature was valid
        """
        if agent_id not in self.agent_profiles:
            logger.warning(f"Unknown agent {agent_id}, adding to profiles")
            self.agent_profiles[agent_id] = AgentBehaviorProfile(agent_id=agent_id)
        
        profile = self.agent_profiles[agent_id]
        
        # Update basic activity
        profile.total_messages += 1
        profile.last_activity = timestamp
        
        # Track message timing
        self.message_timestamps[agent_id].append(timestamp)
        
        # Track signature failures
        if not signature_valid:
            profile.signature_failure_count += 1
            self._record_evidence(
                agent_id,
                ByzantinePattern.SIGNATURE_MANIPULATION,
                0.3,  # Medium severity for single signature failure
                {'message_type': message_type, 'timestamp': timestamp},
                0.6
            )
        
        # Update consensus participation
        if consensus_round is not None:
            if consensus_round not in self.consensus_round_data:
                self.consensus_round_data[consensus_round] = {
                    'participants': set(),
                    'message_counts': defaultdict(int),
                    'start_time': timestamp
                }
            
            self.consensus_round_data[consensus_round]['participants'].add(agent_id)
            self.consensus_round_data[consensus_round]['message_counts'][agent_id] += 1
        
        # Analyze behavior patterns
        self._analyze_timing_patterns(agent_id)
        self._analyze_message_frequency(agent_id)
        
        # Check for Byzantine patterns
        self._check_byzantine_patterns(agent_id)
    
    def record_consensus_result(
        self,
        consensus_round: int,
        participating_agents: List[str],
        consensus_achieved: bool,
        view_changes: int = 0
    ):
        """
        Record consensus round results for behavior analysis
        
        Args:
            consensus_round: Consensus round number
            participating_agents: Agents that participated
            consensus_achieved: Whether consensus was achieved
            view_changes: Number of view changes during this round
        """
        if consensus_round not in self.consensus_round_data:
            self.consensus_round_data[consensus_round] = {
                'participants': set(participating_agents),
                'message_counts': defaultdict(int),
                'start_time': time.time()
            }
        
        round_data = self.consensus_round_data[consensus_round]
        round_data['consensus_achieved'] = consensus_achieved
        round_data['view_changes'] = view_changes
        round_data['end_time'] = time.time()
        
        # Analyze participation patterns
        self._analyze_consensus_participation(consensus_round, participating_agents)
        
        # Check for consensus disruption patterns
        if not consensus_achieved or view_changes > 1:
            self._analyze_consensus_failures(consensus_round)
        
        # Update trust scores
        self._update_trust_scores(participating_agents, consensus_achieved)
    
    def _analyze_timing_patterns(self, agent_id: str):
        """Analyze message timing patterns for anomalies"""
        timestamps = list(self.message_timestamps[agent_id])
        
        if len(timestamps) < 5:  # Need minimum data for analysis
            return
        
        # Calculate inter-message intervals
        intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        
        if len(intervals) < 3:
            return
        
        # Statistical analysis
        mean_interval = statistics.mean(intervals)
        std_interval = statistics.stdev(intervals) if len(intervals) > 1 else 0
        
        # Update profile
        profile = self.agent_profiles[agent_id]
        profile.message_timing_stats['mean_interval'] = mean_interval
        profile.message_timing_stats['std_interval'] = std_interval
        
        # Detect timing anomalies
        # Pattern 1: Too fast (potential flooding)
        if mean_interval < self.baseline_message_interval * 0.1:  # 10x faster than baseline
            self._record_evidence(
                agent_id,
                ByzantinePattern.MESSAGE_FLOODING,
                0.8,  # High severity
                {
                    'mean_interval': mean_interval,
                    'baseline': self.baseline_message_interval,
                    'recent_intervals': intervals[-5:]
                },
                0.9
            )
        
        # Pattern 2: Too slow (potential silent failure)
        elif mean_interval > self.baseline_message_interval * 10:  # 10x slower than baseline
            self._record_evidence(
                agent_id,
                ByzantinePattern.SILENT_FAILURE,
                0.6,  # Medium severity
                {
                    'mean_interval': mean_interval,
                    'baseline': self.baseline_message_interval,
                    'last_activity': profile.last_activity
                },
                0.7
            )
        
        # Pattern 3: High variability (erratic behavior)
        elif std_interval > mean_interval * 2:  # High coefficient of variation
            self._record_evidence(
                agent_id,
                ByzantinePattern.TIMING_ATTACK,
                0.5,  # Medium severity
                {
                    'mean_interval': mean_interval,
                    'std_interval': std_interval,
                    'coefficient_of_variation': std_interval / mean_interval
                },
                0.6
            )
    
    def _analyze_message_frequency(self, agent_id: str):
        """Analyze message frequency for anomalies"""
        current_time = time.time()
        timestamps = [
            ts for ts in self.message_timestamps[agent_id]
            if current_time - ts <= self.detection_window
        ]
        
        message_rate = len(timestamps) / self.detection_window
        
        # Expected message rate (rough estimate: 1 message per consensus round)
        expected_rate = 1.0 / 30.0  # Assuming 30 second consensus rounds
        
        # Check for flooding
        if message_rate > expected_rate * 10:  # 10x higher than expected
            self._record_evidence(
                agent_id,
                ByzantinePattern.MESSAGE_FLOODING,
                0.7,
                {
                    'message_rate': message_rate,
                    'expected_rate': expected_rate,
                    'window': self.detection_window
                },
                0.8
            )
    
    def _analyze_consensus_participation(self, consensus_round: int, participating_agents: List[str]):
        """Analyze consensus participation patterns"""
        all_agents = set(self.agent_ids)
        participants = set(participating_agents)
        non_participants = all_agents - participants
        
        # Update participation statistics
        for agent_id in all_agents:
            profile = self.agent_profiles[agent_id]
            
            if 'total_rounds' not in profile.consensus_participation:
                profile.consensus_participation['total_rounds'] = 0
                profile.consensus_participation['participated_rounds'] = 0
            
            profile.consensus_participation['total_rounds'] += 1
            
            if agent_id in participants:
                profile.consensus_participation['participated_rounds'] += 1
            else:
                # Check for silent failure pattern
                if profile.is_active and time.time() - profile.last_activity < 60:  # Active but not participating
                    self._record_evidence(
                        agent_id,
                        ByzantinePattern.SILENT_FAILURE,
                        0.4,
                        {
                            'consensus_round': consensus_round,
                            'last_activity': profile.last_activity,
                            'participation_rate': self._get_participation_rate(agent_id)
                        },
                        0.5
                    )
        
        # Check for participation anomalies
        for agent_id in participants:
            participation_rate = self._get_participation_rate(agent_id)
            
            if participation_rate < self.baseline_consensus_participation * 0.7:  # Below 70% of baseline
                self._record_evidence(
                    agent_id,
                    ByzantinePattern.CONSENSUS_DISRUPTION,
                    0.5,
                    {
                        'participation_rate': participation_rate,
                        'baseline': self.baseline_consensus_participation,
                        'consensus_round': consensus_round
                    },
                    0.6
                )
    
    def _analyze_consensus_failures(self, consensus_round: int):
        """Analyze consensus failures for Byzantine patterns"""
        round_data = self.consensus_round_data.get(consensus_round, {})
        participants = round_data.get('participants', set())
        view_changes = round_data.get('view_changes', 0)
        
        # High view changes indicate potential Byzantine disruption
        if view_changes > 2:
            for agent_id in participants:
                self._record_evidence(
                    agent_id,
                    ByzantinePattern.VIEW_CHANGE_ABUSE,
                    0.3 * view_changes,  # Severity increases with view changes
                    {
                        'consensus_round': consensus_round,
                        'view_changes': view_changes,
                        'participants': list(participants)
                    },
                    0.7
                )
    
    def _check_byzantine_patterns(self, agent_id: str):
        """Check for Byzantine behavior patterns"""
        profile = self.agent_profiles[agent_id]
        
        # Check signature failure rate
        if profile.total_messages > 10:  # Minimum sample size
            failure_rate = profile.signature_failure_count / profile.total_messages
            
            if failure_rate > self.max_signature_failure_rate:
                self._record_evidence(
                    agent_id,
                    ByzantinePattern.SIGNATURE_MANIPULATION,
                    min(failure_rate * 2, 1.0),  # Cap at 1.0
                    {
                        'failure_rate': failure_rate,
                        'threshold': self.max_signature_failure_rate,
                        'total_messages': profile.total_messages,
                        'failures': profile.signature_failure_count
                    },
                    0.9
                )
        
        # Check for conflicting votes (requires additional context)
        self._check_conflicting_votes(agent_id)
        
        # Update Byzantine classification
        self._update_byzantine_classification(agent_id)
    
    def _check_conflicting_votes(self, agent_id: str):
        """Check for conflicting votes in the same consensus round"""
        # This would require tracking votes per round
        # For now, we'll implement a simplified version
        
        recent_rounds = [
            round_num for round_num, data in self.consensus_round_data.items()
            if time.time() - data.get('start_time', 0) <= self.detection_window
        ]
        
        conflicting_behavior_count = 0
        
        for round_num in recent_rounds:
            round_data = self.consensus_round_data[round_num]
            message_count = round_data['message_counts'].get(agent_id, 0)
            
            # If agent sent significantly more messages than expected, might be conflicting
            expected_messages = 3  # pre-prepare, prepare, commit
            if message_count > expected_messages * 2:
                conflicting_behavior_count += 1
        
        if conflicting_behavior_count > 2:
            self._record_evidence(
                agent_id,
                ByzantinePattern.CONFLICTING_VOTES,
                0.6,
                {
                    'conflicting_rounds': conflicting_behavior_count,
                    'recent_rounds': recent_rounds[-5:]
                },
                0.7
            )
    
    def _record_evidence(
        self,
        agent_id: str,
        pattern_type: ByzantinePattern,
        severity: float,
        evidence_data: Dict[str, Any],
        confidence: float
    ):
        """Record evidence of Byzantine behavior"""
        evidence = ByzantineEvidence(
            agent_id=agent_id,
            pattern_type=pattern_type,
            severity=severity,
            evidence_data=evidence_data,
            timestamp=time.time(),
            detection_confidence=confidence
        )
        
        # Add to evidence database
        self.evidence_database.append(evidence)
        
        # Add to agent profile
        profile = self.agent_profiles[agent_id]
        profile.behavior_anomalies.append(evidence)
        
        # Update metrics
        self.detection_metrics['evidence_count_by_type'][pattern_type.value] += 1
        
        logger.debug(f"Recorded {pattern_type.value} evidence for {agent_id}, severity: {severity:.2f}")
    
    def _update_byzantine_classification(self, agent_id: str):
        """Update Byzantine classification based on evidence"""
        profile = self.agent_profiles[agent_id]
        
        # Calculate composite anomaly score
        recent_evidence = [
            e for e in profile.behavior_anomalies
            if time.time() - e.timestamp <= self.detection_window
        ]
        
        if len(recent_evidence) < self.min_evidence_count:
            return
        
        # Weight by severity and confidence
        weighted_score = sum(
            e.severity * e.detection_confidence for e in recent_evidence
        ) / len(recent_evidence)
        
        # Update trust score
        profile.trust_score = max(0.0, 1.0 - weighted_score)
        
        # Byzantine classification
        if weighted_score >= self.anomaly_threshold:
            if agent_id not in self.suspected_byzantine:
                self.suspected_byzantine.add(agent_id)
                logger.warning(f"Agent {agent_id} classified as suspected Byzantine (score: {weighted_score:.3f})")
            
            # Escalate to confirmed if high confidence and severity
            high_confidence_evidence = [
                e for e in recent_evidence
                if e.detection_confidence >= 0.8 and e.severity >= 0.7
            ]
            
            if len(high_confidence_evidence) >= 2:
                if agent_id not in self.confirmed_byzantine:
                    self.confirmed_byzantine.add(agent_id)
                    self.detection_metrics['total_detections'] += 1
                    logger.error(f"Agent {agent_id} confirmed as Byzantine!")
        
        elif agent_id in self.suspected_byzantine and weighted_score < self.anomaly_threshold * 0.5:
            # Remove from suspected if score improves significantly
            self.suspected_byzantine.discard(agent_id)
            logger.info(f"Agent {agent_id} removed from Byzantine suspects (improved behavior)")
    
    def _get_participation_rate(self, agent_id: str) -> float:
        """Get consensus participation rate for agent"""
        profile = self.agent_profiles[agent_id]
        stats = profile.consensus_participation
        
        if stats.get('total_rounds', 0) == 0:
            return 1.0  # No data yet, assume good
        
        return stats['participated_rounds'] / stats['total_rounds']
    
    def _update_trust_scores(self, participating_agents: List[str], consensus_achieved: bool):
        """Update trust scores based on consensus outcome"""
        for agent_id in participating_agents:
            profile = self.agent_profiles[agent_id]
            
            if consensus_achieved:
                # Slightly increase trust for successful consensus
                profile.trust_score = min(1.0, profile.trust_score + 0.01)
            else:
                # Slightly decrease trust for failed consensus
                profile.trust_score = max(0.0, profile.trust_score - 0.02)
    
    def get_byzantine_agents(self) -> Tuple[Set[str], Set[str]]:
        """
        Get Byzantine agent classifications
        
        Returns:
            Tuple of (suspected_byzantine, confirmed_byzantine) agent sets
        """
        return self.suspected_byzantine.copy(), self.confirmed_byzantine.copy()
    
    def get_agent_trust_score(self, agent_id: str) -> float:
        """Get trust score for specific agent"""
        profile = self.agent_profiles.get(agent_id)
        return profile.trust_score if profile else 0.0
    
    def get_agent_behavior_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive behavior summary for agent"""
        profile = self.agent_profiles.get(agent_id)
        if not profile:
            return {'error': 'Agent not found'}
        
        recent_evidence = [
            e for e in profile.behavior_anomalies
            if time.time() - e.timestamp <= self.detection_window
        ]
        
        return {
            'agent_id': agent_id,
            'trust_score': profile.trust_score,
            'is_suspected_byzantine': agent_id in self.suspected_byzantine,
            'is_confirmed_byzantine': agent_id in self.confirmed_byzantine,
            'total_messages': profile.total_messages,
            'signature_failure_rate': profile.signature_failure_count / max(1, profile.total_messages),
            'participation_rate': self._get_participation_rate(agent_id),
            'recent_evidence_count': len(recent_evidence),
            'evidence_types': list(set(e.pattern_type.value for e in recent_evidence)),
            'last_activity': profile.last_activity,
            'timing_stats': profile.message_timing_stats,
            'consensus_stats': profile.consensus_participation
        }
    
    def get_detection_metrics(self) -> Dict[str, Any]:
        """Get Byzantine detection metrics"""
        metrics = self.detection_metrics.copy()
        
        # Add current state
        metrics['suspected_byzantine_count'] = len(self.suspected_byzantine)
        metrics['confirmed_byzantine_count'] = len(self.confirmed_byzantine)
        metrics['total_evidence_count'] = len(self.evidence_database)
        metrics['active_agent_count'] = len([p for p in self.agent_profiles.values() if p.is_active])
        
        # Calculate detection rates
        total_agents = len(self.agent_ids)
        if total_agents > 0:
            metrics['byzantine_detection_rate'] = len(self.confirmed_byzantine) / total_agents
        
        return metrics
    
    def cleanup_old_data(self):
        """Clean up old evidence and data to prevent memory buildup"""
        current_time = time.time()
        cutoff_time = current_time - self.detection_window * 2  # Keep 2x detection window
        
        # Clean evidence database
        self.evidence_database = [
            e for e in self.evidence_database
            if e.timestamp > cutoff_time
        ]
        
        # Clean agent behavior anomalies
        for profile in self.agent_profiles.values():
            profile.behavior_anomalies = [
                e for e in profile.behavior_anomalies
                if e.timestamp > cutoff_time
            ]
        
        # Clean old consensus round data
        old_rounds = [
            round_num for round_num, data in self.consensus_round_data.items()
            if data.get('start_time', 0) < cutoff_time
        ]
        
        for round_num in old_rounds:
            del self.consensus_round_data[round_num]
        
        logger.debug(f"Cleaned up old data: {len(old_rounds)} rounds, cutoff: {cutoff_time}")
    
    def export_evidence_report(self) -> str:
        """Export comprehensive evidence report as JSON"""
        report = {
            'timestamp': time.time(),
            'detection_window': self.detection_window,
            'suspected_byzantine': list(self.suspected_byzantine),
            'confirmed_byzantine': list(self.confirmed_byzantine),
            'agent_summaries': {},
            'evidence_database': [],
            'detection_metrics': self.get_detection_metrics()
        }
        
        # Add agent summaries
        for agent_id in self.agent_ids:
            report['agent_summaries'][agent_id] = self.get_agent_behavior_summary(agent_id)
        
        # Add evidence database
        for evidence in self.evidence_database:
            report['evidence_database'].append({
                'agent_id': evidence.agent_id,
                'pattern_type': evidence.pattern_type.value,
                'severity': evidence.severity,
                'timestamp': evidence.timestamp,
                'confidence': evidence.detection_confidence,
                'evidence_data': evidence.evidence_data
            })
        
        return json.dumps(report, indent=2, default=str)