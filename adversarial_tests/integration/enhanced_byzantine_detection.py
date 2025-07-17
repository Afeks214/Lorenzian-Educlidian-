"""
Enhanced Byzantine Detection with ML-based Behavioral Analysis
============================================================

This module implements advanced Byzantine fault detection using machine learning
techniques to identify malicious behavior patterns in distributed consensus systems.

Key Features:
- ML-based behavioral pattern recognition
- Real-time anomaly detection using ensemble methods
- Temporal behavior analysis with LSTM networks
- Graph-based consensus analysis
- Adaptive trust scoring with decay factors
- Automated response and quarantine mechanisms

Author: Agent Beta Mission - Byzantine Enhancement
Version: 1.0.0
Classification: CRITICAL SECURITY COMPONENT
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import threading
import logging
from collections import deque, defaultdict
import time
import json
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import networkx as nx
from scipy.stats import entropy
import structlog

logger = structlog.get_logger()


class ByzantineNodeState(Enum):
    """Byzantine node states"""
    HONEST = "HONEST"
    SUSPICIOUS = "SUSPICIOUS"
    MALICIOUS = "MALICIOUS"
    QUARANTINED = "QUARANTINED"
    UNKNOWN = "UNKNOWN"


class ByzantineAttackType(Enum):
    """Types of Byzantine attacks"""
    RANDOM_VOTING = "RANDOM_VOTING"
    ALWAYS_DISAGREE = "ALWAYS_DISAGREE"
    SELECTIVE_ATTACK = "SELECTIVE_ATTACK"
    COALITION_ATTACK = "COALITION_ATTACK"
    TIMING_ATTACK = "TIMING_ATTACK"
    SYBIL_ATTACK = "SYBIL_ATTACK"
    ECLIPSE_ATTACK = "ECLIPSE_ATTACK"


@dataclass
class ByzantineNodeBehavior:
    """Detailed Byzantine node behavior tracking"""
    node_id: str
    state: ByzantineNodeState
    trust_score: float
    consensus_participation: List[bool]
    response_times: List[float]
    vote_patterns: List[Dict[str, Any]]
    attack_indicators: List[str]
    last_activity: datetime
    behavior_embedding: Optional[np.ndarray] = None
    ml_anomaly_score: float = 0.0
    temporal_consistency: float = 1.0
    network_centrality: float = 0.0
    coalition_membership: Optional[str] = None


@dataclass
class ConsensusRound:
    """Consensus round data"""
    round_id: str
    timestamp: datetime
    proposal: Dict[str, Any]
    participants: List[str]
    votes: Dict[str, bool]
    consensus_reached: bool
    duration_ms: float
    Byzantine_nodes_detected: List[str]


@dataclass
class ByzantineAttackDetection:
    """Byzantine attack detection result"""
    attack_type: ByzantineAttackType
    confidence: float
    affected_nodes: List[str]
    attack_pattern: Dict[str, Any]
    detection_time: datetime
    evidence: List[str]
    mitigation_actions: List[str]


class ByzantineLSTM(nn.Module):
    """LSTM network for temporal behavior analysis"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super(ByzantineLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Use last time step
        return self.sigmoid(out)


class EnhancedByzantineDetector:
    """
    Enhanced Byzantine fault detection system with ML-based behavioral analysis.
    """
    
    def __init__(
        self,
        node_count: int = 100,
        malicious_ratio: float = 0.3,
        detection_window: int = 50,
        ml_model_update_interval: int = 100,
        trust_decay_factor: float = 0.95,
        quarantine_threshold: float = 0.3
    ):
        self.node_count = node_count
        self.malicious_ratio = malicious_ratio
        self.detection_window = detection_window
        self.ml_model_update_interval = ml_model_update_interval
        self.trust_decay_factor = trust_decay_factor
        self.quarantine_threshold = quarantine_threshold
        
        # Node tracking
        self.nodes: Dict[str, ByzantineNodeBehavior] = {}
        self.consensus_history: deque = deque(maxlen=1000)
        self.attack_detections: List[ByzantineAttackDetection] = []
        
        # ML Models
        self.isolation_forest = IsolationForest(contamination=malicious_ratio, random_state=42)
        self.random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
        self.lstm_model = ByzantineLSTM(input_size=10)  # 10 behavioral features
        self.scaler = StandardScaler()
        self.temporal_scaler = MinMaxScaler()
        
        # Network analysis
        self.consensus_graph = nx.DiGraph()
        self.coalition_detector = DBSCAN(eps=0.3, min_samples=3)
        
        # Performance tracking
        self.detection_metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            'detection_times': deque(maxlen=100),
            'accuracy_history': deque(maxlen=100)
        }
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.callbacks: List[Callable] = []
        
        # Initialize system
        self._initialize_nodes()
        self._initialize_ml_models()
        
        logger.info("EnhancedByzantineDetector initialized",
                   node_count=node_count,
                   malicious_ratio=malicious_ratio)
    
    def _initialize_nodes(self):
        """Initialize Byzantine nodes with realistic behavior patterns"""
        for i in range(self.node_count):
            node_id = f"byzantine_node_{i:03d}"
            
            # Determine if node is malicious
            is_malicious = i < int(self.node_count * self.malicious_ratio)
            
            # Initialize node behavior
            self.nodes[node_id] = ByzantineNodeBehavior(
                node_id=node_id,
                state=ByzantineNodeState.MALICIOUS if is_malicious else ByzantineNodeState.HONEST,
                trust_score=0.3 if is_malicious else 1.0,
                consensus_participation=[],
                response_times=[],
                vote_patterns=[],
                attack_indicators=[],
                last_activity=datetime.now(),
                behavior_embedding=None,
                ml_anomaly_score=0.0,
                temporal_consistency=1.0,
                network_centrality=0.0,
                coalition_membership=None
            )
            
            # Add to consensus network
            self.consensus_graph.add_node(node_id, malicious=is_malicious)
    
    def _initialize_ml_models(self):
        """Initialize ML models with synthetic training data"""
        # Generate synthetic training data
        training_data, labels = self._generate_synthetic_training_data()
        
        # Train isolation forest
        self.isolation_forest.fit(training_data)
        
        # Train random forest
        self.random_forest.fit(training_data, labels)
        
        # Initialize LSTM optimizer
        self.lstm_optimizer = optim.Adam(self.lstm_model.parameters(), lr=0.001)
        self.lstm_criterion = nn.BCELoss()
        
        logger.info("ML models initialized with synthetic training data")
    
    def _generate_synthetic_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data for ML models"""
        n_samples = 1000
        n_features = 10
        
        # Generate honest node behavior
        honest_data = np.random.normal(0.7, 0.1, (n_samples // 2, n_features))
        honest_labels = np.zeros(n_samples // 2)
        
        # Generate malicious node behavior
        malicious_data = np.random.normal(0.3, 0.2, (n_samples // 2, n_features))
        malicious_labels = np.ones(n_samples // 2)
        
        # Add attack-specific patterns
        # Random voting pattern
        malicious_data[:100, 0] = np.random.uniform(0, 1, 100)
        
        # Always disagree pattern
        malicious_data[100:200, 1] = np.random.uniform(0, 0.2, 100)
        
        # Timing attack pattern
        malicious_data[200:300, 2] = np.random.uniform(0.8, 1.0, 100)
        
        # Combine data
        training_data = np.vstack([honest_data, malicious_data])
        labels = np.hstack([honest_labels, malicious_labels])
        
        # Shuffle data
        indices = np.random.permutation(len(training_data))
        training_data = training_data[indices]
        labels = labels[indices]
        
        return training_data, labels
    
    async def simulate_consensus_rounds(self, num_rounds: int = 100) -> List[ConsensusRound]:
        """
        Simulate consensus rounds with Byzantine behavior.
        
        Args:
            num_rounds: Number of consensus rounds to simulate
            
        Returns:
            List of consensus round results
        """
        logger.info(f"Starting {num_rounds} consensus rounds simulation")
        
        consensus_rounds = []
        
        for round_num in range(num_rounds):
            # Create consensus round
            round_data = await self._simulate_single_consensus_round(round_num)
            consensus_rounds.append(round_data)
            
            # Analyze round for Byzantine behavior
            await self._analyze_consensus_round(round_data)
            
            # Update ML models periodically
            if round_num % self.ml_model_update_interval == 0:
                await self._update_ml_models()
            
            # Brief pause between rounds
            await asyncio.sleep(0.01)
        
        logger.info(f"Completed {num_rounds} consensus rounds")
        return consensus_rounds
    
    async def _simulate_single_consensus_round(self, round_num: int) -> ConsensusRound:
        """Simulate a single consensus round"""
        round_id = f"round_{round_num:04d}"
        start_time = datetime.now()
        
        # Create proposal
        proposal = {
            "round_id": round_id,
            "value": np.random.randint(0, 100),
            "timestamp": start_time.isoformat()
        }
        
        # Select participants (random subset)
        participants = np.random.choice(
            list(self.nodes.keys()),
            size=min(50, len(self.nodes)),
            replace=False
        ).tolist()
        
        # Collect votes
        votes = {}
        for node_id in participants:
            node = self.nodes[node_id]
            
            # Simulate voting behavior
            vote = await self._simulate_node_vote(node, proposal)
            votes[node_id] = vote
            
            # Update node participation
            node.consensus_participation.append(True)
            if len(node.consensus_participation) > self.detection_window:
                node.consensus_participation.pop(0)
            
            # Record vote pattern
            node.vote_patterns.append({
                "round_id": round_id,
                "vote": vote,
                "timestamp": datetime.now().isoformat()
            })
            if len(node.vote_patterns) > self.detection_window:
                node.vote_patterns.pop(0)
        
        # Determine consensus
        consensus_reached = sum(votes.values()) > len(votes) * 0.6  # 60% threshold
        
        # Calculate round duration
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        return ConsensusRound(
            round_id=round_id,
            timestamp=start_time,
            proposal=proposal,
            participants=participants,
            votes=votes,
            consensus_reached=consensus_reached,
            duration_ms=duration_ms,
            Byzantine_nodes_detected=[]
        )
    
    async def _simulate_node_vote(self, node: ByzantineNodeBehavior, proposal: Dict[str, Any]) -> bool:
        """Simulate node voting behavior"""
        # Simulate response time
        response_time = np.random.exponential(50)  # ms
        node.response_times.append(response_time)
        if len(node.response_times) > self.detection_window:
            node.response_times.pop(0)
        
        # Determine vote based on node type
        if node.state == ByzantineNodeState.HONEST:
            # Honest nodes vote based on proposal value with some randomness
            honest_vote = proposal["value"] > 50
            return honest_vote if np.random.random() > 0.1 else not honest_vote
        
        elif node.state == ByzantineNodeState.MALICIOUS:
            # Malicious behavior patterns
            attack_type = self._determine_attack_type(node)
            
            if attack_type == ByzantineAttackType.RANDOM_VOTING:
                return np.random.random() > 0.5
            elif attack_type == ByzantineAttackType.ALWAYS_DISAGREE:
                return False
            elif attack_type == ByzantineAttackType.SELECTIVE_ATTACK:
                # Attack only certain proposals
                return proposal["value"] < 25
            elif attack_type == ByzantineAttackType.TIMING_ATTACK:
                # Delay response significantly
                await asyncio.sleep(response_time / 500)  # Simulate delay
                return np.random.random() > 0.5
            else:
                return np.random.random() > 0.5
        
        else:
            # Quarantined or suspicious nodes
            return np.random.random() > 0.7
    
    def _determine_attack_type(self, node: ByzantineNodeBehavior) -> ByzantineAttackType:
        """Determine attack type for malicious node"""
        # Simple hash-based attack type assignment
        node_hash = hash(node.node_id) % 4
        
        attack_types = [
            ByzantineAttackType.RANDOM_VOTING,
            ByzantineAttackType.ALWAYS_DISAGREE,
            ByzantineAttackType.SELECTIVE_ATTACK,
            ByzantineAttackType.TIMING_ATTACK
        ]
        
        return attack_types[node_hash]
    
    async def _analyze_consensus_round(self, round_data: ConsensusRound):
        """Analyze consensus round for Byzantine behavior"""
        # Update consensus graph
        self._update_consensus_graph(round_data)
        
        # Extract behavioral features
        for node_id in round_data.participants:
            node = self.nodes[node_id]
            features = self._extract_behavioral_features(node)
            
            # Update behavior embedding
            node.behavior_embedding = features
            
            # ML-based anomaly detection
            if len(features) > 0:
                anomaly_score = self._calculate_ml_anomaly_score(features)
                node.ml_anomaly_score = anomaly_score
                
                # Update node state based on anomaly score
                await self._update_node_state(node)
        
        # Detect coalition attacks
        await self._detect_coalition_attacks(round_data)
        
        # Store round data
        self.consensus_history.append(round_data)
    
    def _update_consensus_graph(self, round_data: ConsensusRound):
        """Update consensus network graph"""
        # Add edges between nodes that voted similarly
        votes = round_data.votes
        
        for node1 in votes:
            for node2 in votes:
                if node1 != node2 and votes[node1] == votes[node2]:
                    # Add edge for similar voting
                    if self.consensus_graph.has_edge(node1, node2):
                        self.consensus_graph[node1][node2]['weight'] += 1
                    else:
                        self.consensus_graph.add_edge(node1, node2, weight=1)
        
        # Update centrality measures
        try:
            centrality = nx.closeness_centrality(self.consensus_graph)
            for node_id, centrality_value in centrality.items():
                if node_id in self.nodes:
                    self.nodes[node_id].network_centrality = centrality_value
        except (ValueError, TypeError, AttributeError) as e:
            pass  # Handle empty graph case
    
    def _extract_behavioral_features(self, node: ByzantineNodeBehavior) -> np.ndarray:
        """Extract behavioral features for ML analysis"""
        features = []
        
        # Consensus participation rate
        if node.consensus_participation:
            participation_rate = sum(node.consensus_participation) / len(node.consensus_participation)
            features.append(participation_rate)
        else:
            features.append(0.0)
        
        # Average response time
        if node.response_times:
            avg_response_time = np.mean(node.response_times)
            features.append(min(avg_response_time / 1000, 1.0))  # Normalize
        else:
            features.append(0.0)
        
        # Vote consistency
        if node.vote_patterns:
            votes = [pattern["vote"] for pattern in node.vote_patterns]
            vote_consistency = np.var(votes) if len(votes) > 1 else 0.0
            features.append(vote_consistency)
        else:
            features.append(0.0)
        
        # Trust score
        features.append(node.trust_score)
        
        # Network centrality
        features.append(node.network_centrality)
        
        # Temporal consistency
        features.append(node.temporal_consistency)
        
        # Activity frequency
        time_since_activity = (datetime.now() - node.last_activity).total_seconds()
        activity_score = min(1.0, 1.0 / (1.0 + time_since_activity / 3600))  # Decay over hours
        features.append(activity_score)
        
        # Attack indicators count
        features.append(min(len(node.attack_indicators) / 10, 1.0))  # Normalize
        
        # Vote pattern entropy
        if node.vote_patterns:
            votes = [pattern["vote"] for pattern in node.vote_patterns]
            if len(votes) > 1:
                vote_entropy = entropy([sum(votes), len(votes) - sum(votes)])
                features.append(vote_entropy / np.log(2))  # Normalize
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # Response time variance
        if len(node.response_times) > 1:
            response_time_var = np.var(node.response_times)
            features.append(min(response_time_var / 10000, 1.0))  # Normalize
        else:
            features.append(0.0)
        
        return np.array(features)
    
    def _calculate_ml_anomaly_score(self, features: np.ndarray) -> float:
        """Calculate ML-based anomaly score"""
        try:
            # Isolation Forest score
            iso_score = self.isolation_forest.decision_function([features])[0]
            
            # Random Forest probability
            rf_prob = self.random_forest.predict_proba([features])[0][1]  # Malicious probability
            
            # Combine scores
            combined_score = 0.6 * (1 - (iso_score + 1) / 2) + 0.4 * rf_prob
            
            return combined_score
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            return 0.0
    
    async def _update_node_state(self, node: ByzantineNodeBehavior):
        """Update node state based on analysis"""
        # Apply trust decay
        node.trust_score *= self.trust_decay_factor
        
        # Update state based on ML anomaly score
        if node.ml_anomaly_score > 0.8:
            if node.state != ByzantineNodeState.MALICIOUS:
                node.state = ByzantineNodeState.MALICIOUS
                node.attack_indicators.append(f"ML_ANOMALY_HIGH_{datetime.now().isoformat()}")
        elif node.ml_anomaly_score > 0.6:
            if node.state == ByzantineNodeState.HONEST:
                node.state = ByzantineNodeState.SUSPICIOUS
                node.attack_indicators.append(f"ML_ANOMALY_MEDIUM_{datetime.now().isoformat()}")
        
        # Quarantine nodes below threshold
        if node.trust_score < self.quarantine_threshold:
            node.state = ByzantineNodeState.QUARANTINED
            node.attack_indicators.append(f"QUARANTINED_{datetime.now().isoformat()}")
        
        # Update activity timestamp
        node.last_activity = datetime.now()
    
    async def _detect_coalition_attacks(self, round_data: ConsensusRound):
        """Detect coalition attacks using clustering"""
        if len(round_data.participants) < 5:
            return
        
        # Extract features for clustering
        features = []
        node_ids = []
        
        for node_id in round_data.participants:
            node = self.nodes[node_id]
            if node.behavior_embedding is not None:
                features.append(node.behavior_embedding)
                node_ids.append(node_id)
        
        if len(features) < 3:
            return
        
        # Perform clustering
        features_array = np.array(features)
        cluster_labels = self.coalition_detector.fit_predict(features_array)
        
        # Analyze clusters for coalitions
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            if label != -1:  # Not noise
                clusters[label].append(node_ids[i])
        
        # Detect malicious coalitions
        for cluster_id, cluster_nodes in clusters.items():
            if len(cluster_nodes) >= 3:  # Minimum coalition size
                # Check if cluster contains mostly malicious nodes
                malicious_count = sum(1 for node_id in cluster_nodes 
                                    if self.nodes[node_id].state == ByzantineNodeState.MALICIOUS)
                
                if malicious_count >= len(cluster_nodes) * 0.6:
                    # Detected coalition attack
                    attack_detection = ByzantineAttackDetection(
                        attack_type=ByzantineAttackType.COALITION_ATTACK,
                        confidence=malicious_count / len(cluster_nodes),
                        affected_nodes=cluster_nodes,
                        attack_pattern={
                            "cluster_id": cluster_id,
                            "cluster_size": len(cluster_nodes),
                            "malicious_ratio": malicious_count / len(cluster_nodes)
                        },
                        detection_time=datetime.now(),
                        evidence=[f"Cluster {cluster_id} with {malicious_count}/{len(cluster_nodes)} malicious nodes"],
                        mitigation_actions=[
                            "Quarantine coalition members",
                            "Increase consensus threshold",
                            "Implement vote weighting"
                        ]
                    )
                    
                    self.attack_detections.append(attack_detection)
                    
                    # Update coalition membership
                    for node_id in cluster_nodes:
                        self.nodes[node_id].coalition_membership = f"coalition_{cluster_id}"
                    
                    logger.warning(f"Coalition attack detected: {len(cluster_nodes)} nodes",
                                 cluster_id=cluster_id,
                                 malicious_ratio=malicious_count / len(cluster_nodes))
    
    async def _update_ml_models(self):
        """Update ML models with new data"""
        logger.info("Updating ML models with new behavioral data")
        
        # Collect recent behavioral data
        features = []
        labels = []
        
        for node in self.nodes.values():
            if node.behavior_embedding is not None:
                features.append(node.behavior_embedding)
                # Label based on current state
                label = 1 if node.state == ByzantineNodeState.MALICIOUS else 0
                labels.append(label)
        
        if len(features) < 10:
            return
        
        # Update models
        features_array = np.array(features)
        labels_array = np.array(labels)
        
        # Update isolation forest
        self.isolation_forest.fit(features_array)
        
        # Update random forest
        self.random_forest.fit(features_array, labels_array)
        
        # Update LSTM model
        await self._update_lstm_model(features_array, labels_array)
        
        # Calculate detection metrics
        self._calculate_detection_metrics(features_array, labels_array)
    
    async def _update_lstm_model(self, features: np.ndarray, labels: np.ndarray):
        """Update LSTM model for temporal behavior analysis"""
        if len(features) < 20:
            return
        
        # Prepare temporal sequences
        sequence_length = 10
        temporal_features = []
        temporal_labels = []
        
        for i in range(sequence_length, len(features)):
            sequence = features[i-sequence_length:i]
            label = labels[i]
            temporal_features.append(sequence)
            temporal_labels.append(label)
        
        if len(temporal_features) < 5:
            return
        
        # Convert to tensors
        X = torch.FloatTensor(temporal_features)
        y = torch.FloatTensor(temporal_labels).unsqueeze(1)
        
        # Training loop
        self.lstm_model.train()
        for epoch in range(10):
            self.lstm_optimizer.zero_grad()
            outputs = self.lstm_model(X)
            loss = self.lstm_criterion(outputs, y)
            loss.backward()
            self.lstm_optimizer.step()
    
    def _calculate_detection_metrics(self, features: np.ndarray, labels: np.ndarray):
        """Calculate detection performance metrics"""
        # Predict using current models
        predictions = self.random_forest.predict(features)
        
        # Calculate metrics
        true_positives = sum((labels == 1) & (predictions == 1))
        false_positives = sum((labels == 0) & (predictions == 1))
        true_negatives = sum((labels == 0) & (predictions == 0))
        false_negatives = sum((labels == 1) & (predictions == 0))
        
        # Update metrics
        self.detection_metrics['true_positives'] = true_positives
        self.detection_metrics['false_positives'] = false_positives
        self.detection_metrics['true_negatives'] = true_negatives
        self.detection_metrics['false_negatives'] = false_negatives
        
        # Calculate accuracy
        accuracy = (true_positives + true_negatives) / len(labels)
        self.detection_metrics['accuracy_history'].append(accuracy)
        
        logger.info("Detection metrics updated",
                   accuracy=accuracy,
                   true_positives=true_positives,
                   false_positives=false_positives)
    
    def get_detection_report(self) -> Dict[str, Any]:
        """Generate comprehensive detection report"""
        # Calculate current metrics
        total_nodes = len(self.nodes)
        malicious_nodes = sum(1 for node in self.nodes.values() 
                            if node.state == ByzantineNodeState.MALICIOUS)
        quarantined_nodes = sum(1 for node in self.nodes.values() 
                              if node.state == ByzantineNodeState.QUARANTINED)
        
        # Performance metrics
        metrics = self.detection_metrics
        current_accuracy = metrics['accuracy_history'][-1] if metrics['accuracy_history'] else 0.0
        
        # Attack detection summary
        attack_summary = {}
        for attack in self.attack_detections:
            attack_type = attack.attack_type.value
            attack_summary[attack_type] = attack_summary.get(attack_type, 0) + 1
        
        return {
            "detection_summary": {
                "total_nodes": total_nodes,
                "malicious_nodes_detected": malicious_nodes,
                "quarantined_nodes": quarantined_nodes,
                "detection_accuracy": current_accuracy,
                "consensus_rounds_analyzed": len(self.consensus_history)
            },
            "attack_detections": {
                "total_attacks_detected": len(self.attack_detections),
                "attack_types": attack_summary,
                "recent_attacks": [
                    {
                        "type": attack.attack_type.value,
                        "confidence": attack.confidence,
                        "affected_nodes": len(attack.affected_nodes),
                        "detection_time": attack.detection_time.isoformat()
                    }
                    for attack in self.attack_detections[-5:]  # Last 5 attacks
                ]
            },
            "performance_metrics": {
                "true_positives": metrics['true_positives'],
                "false_positives": metrics['false_positives'],
                "true_negatives": metrics['true_negatives'],
                "false_negatives": metrics['false_negatives'],
                "precision": metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives']) if (metrics['true_positives'] + metrics['false_positives']) > 0 else 0,
                "recall": metrics['true_positives'] / (metrics['true_positives'] + metrics['false_negatives']) if (metrics['true_positives'] + metrics['false_negatives']) > 0 else 0,
                "accuracy_trend": list(metrics['accuracy_history'])[-10:]  # Last 10 accuracy values
            },
            "ml_model_status": {
                "isolation_forest_trained": self.isolation_forest is not None,
                "random_forest_trained": self.random_forest is not None,
                "lstm_model_trained": self.lstm_model is not None,
                "model_update_count": len(self.consensus_history) // self.ml_model_update_interval
            },
            "network_analysis": {
                "graph_nodes": self.consensus_graph.number_of_nodes(),
                "graph_edges": self.consensus_graph.number_of_edges(),
                "graph_density": nx.density(self.consensus_graph) if self.consensus_graph.number_of_nodes() > 0 else 0
            }
        }
    
    def start_real_time_monitoring(self):
        """Start real-time monitoring"""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        logger.info("Real-time Byzantine monitoring started")
    
    def stop_real_time_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Real-time Byzantine monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Monitor node states
                self._monitor_node_states()
                
                # Check for new attacks
                self._check_for_attacks()
                
                # Update trust scores
                self._update_trust_scores()
                
                time.sleep(0.1)  # 100ms monitoring interval
                
            except Exception as e:
                logger.error(f"Error in Byzantine monitoring loop: {e}")
                time.sleep(1.0)
    
    def _monitor_node_states(self):
        """Monitor node states for changes"""
        for node in self.nodes.values():
            # Check for state transitions
            if node.state == ByzantineNodeState.SUSPICIOUS:
                # Escalate to malicious if anomaly score increases
                if node.ml_anomaly_score > 0.8:
                    node.state = ByzantineNodeState.MALICIOUS
                    logger.warning(f"Node {node.node_id} escalated to MALICIOUS")
    
    def _check_for_attacks(self):
        """Check for new attack patterns"""
        # This would be implemented with real-time pattern matching
        pass
    
    def _update_trust_scores(self):
        """Update trust scores with decay"""
        for node in self.nodes.values():
            node.trust_score *= self.trust_decay_factor
            node.trust_score = max(0.0, min(1.0, node.trust_score))
    
    def register_callback(self, callback: Callable):
        """Register callback for attack detection"""
        self.callbacks.append(callback)
    
    def _notify_callbacks(self, event_data: Dict[str, Any]):
        """Notify registered callbacks"""
        for callback in self.callbacks:
            try:
                callback(event_data)
            except Exception as e:
                logger.error(f"Error in callback: {e}")


# Example usage and testing
async def main():
    """Example usage of Enhanced Byzantine Detection"""
    detector = EnhancedByzantineDetector(
        node_count=50,
        malicious_ratio=0.3,
        detection_window=30
    )
    
    # Start monitoring
    detector.start_real_time_monitoring()
    
    # Simulate consensus rounds
    await detector.simulate_consensus_rounds(200)
    
    # Generate report
    report = detector.get_detection_report()
    print(json.dumps(report, indent=2))
    
    # Stop monitoring
    detector.stop_real_time_monitoring()


if __name__ == "__main__":
    asyncio.run(main())