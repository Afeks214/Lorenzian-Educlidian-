"""
Emergent Behavior Detector - Advanced MARL Pattern Analysis
===========================================================

This module implements sophisticated detection and analysis of emergent behaviors
in multi-agent reinforcement learning systems using advanced pattern recognition,
statistical analysis, and machine learning techniques.

Key Features:
- Real-time behavior pattern detection
- Anomaly detection for novel behaviors
- Statistical significance testing
- Temporal pattern analysis
- Behavioral clustering and classification
- Emergent property quantification

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
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import networkx as nx
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class BehaviorType(Enum):
    """Types of emergent behaviors."""
    COORDINATION = "coordination"
    COMPETITION = "competition"
    COOPERATION = "cooperation"
    SPECIALIZATION = "specialization"
    SYNCHRONIZATION = "synchronization"
    PHASE_TRANSITION = "phase_transition"
    COLLECTIVE_INTELLIGENCE = "collective_intelligence"
    SWARM_BEHAVIOR = "swarm_behavior"
    HIERARCHICAL_ORGANIZATION = "hierarchical_organization"
    COMMUNICATION_PROTOCOL = "communication_protocol"


class NoveltyLevel(Enum):
    """Levels of behavior novelty."""
    KNOWN = "known"
    VARIANT = "variant"
    NOVEL = "novel"
    UNPRECEDENTED = "unprecedented"


@dataclass
class BehaviorPattern:
    """Structure for behavior patterns."""
    pattern_id: str
    behavior_type: BehaviorType
    agents_involved: List[str]
    detection_time: datetime
    duration_ms: float
    confidence: float
    novelty_level: NoveltyLevel
    features: Dict[str, Any]
    statistical_significance: float
    description: str
    impact_score: float = 0.0
    stability_score: float = 0.0
    reproducibility_score: float = 0.0


@dataclass
class EmergentBehaviorConfig:
    """Configuration for emergent behavior detection."""
    observation_window_ms: int = 30000
    pattern_detection_threshold: float = 0.7
    novelty_threshold: float = 0.8
    statistical_significance_threshold: float = 0.05
    min_pattern_duration_ms: int = 1000
    max_pattern_duration_ms: int = 60000
    clustering_algorithm: str = "dbscan"
    feature_extraction_method: str = "statistical"
    enable_temporal_analysis: bool = True
    enable_network_analysis: bool = True
    enable_statistical_testing: bool = True
    behavior_history_size: int = 1000


class EmergentBehaviorDetector:
    """
    Advanced Emergent Behavior Detector for MARL Systems.
    
    This detector uses multiple analysis techniques to identify, classify,
    and quantify emergent behaviors in multi-agent systems including:
    - Pattern recognition and clustering
    - Statistical analysis and significance testing
    - Temporal behavior analysis
    - Network analysis of agent interactions
    - Novelty detection and classification
    """
    
    def __init__(self, config: Optional[EmergentBehaviorConfig] = None):
        """Initialize the emergent behavior detector."""
        self.config = config or EmergentBehaviorConfig()
        self.behavior_history = deque(maxlen=self.config.behavior_history_size)
        self.known_patterns = {}
        self.feature_extractors = {}
        self.clustering_models = {}
        self.temporal_analyzers = {}
        self.network_analyzer = None
        
        # Analysis state
        self.observation_buffer = deque(maxlen=10000)
        self.pattern_cache = {}
        self.novelty_detector = None
        self.statistical_analyzer = None
        
        # Performance metrics
        self.detection_metrics = {
            'total_patterns_detected': 0,
            'novel_patterns_detected': 0,
            'false_positive_rate': 0.0,
            'detection_accuracy': 0.0,
            'processing_time_ms': 0.0,
            'memory_usage_mb': 0.0
        }
        
        # Initialize components
        self._initialize_components()
        
        logger.info("Emergent Behavior Detector initialized")
    
    def is_initialized(self) -> bool:
        """Check if detector is properly initialized."""
        return self.statistical_analyzer is not None
    
    def _initialize_components(self):
        """Initialize detector components."""
        try:
            # Initialize feature extractors
            self.feature_extractors = {
                'statistical': StatisticalFeatureExtractor(),
                'temporal': TemporalFeatureExtractor(),
                'network': NetworkFeatureExtractor(),
                'behavioral': BehavioralFeatureExtractor()
            }
            
            # Initialize clustering models
            self.clustering_models = {
                'dbscan': DBSCAN(eps=0.5, min_samples=5),
                'kmeans': KMeans(n_clusters=8, random_state=42)
            }
            
            # Initialize analyzers
            self.statistical_analyzer = StatisticalAnalyzer(
                significance_threshold=self.config.statistical_significance_threshold
            )
            
            self.novelty_detector = NoveltyDetector(
                threshold=self.config.novelty_threshold
            )
            
            self.network_analyzer = NetworkAnalyzer()
            
            # Initialize temporal analyzers
            self.temporal_analyzers = {
                'phase_transition': PhaseTransitionAnalyzer(),
                'synchronization': SynchronizationAnalyzer(),
                'trend_analysis': TrendAnalyzer()
            }
            
            logger.info("All emergent behavior detector components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize detector components: {str(e)}")
            raise
    
    async def detect_emergent_behaviors(self, 
                                      agent_system: Any,
                                      observation_window: int = 1000,
                                      behavior_patterns: List[Dict] = None,
                                      detection_config: Optional[EmergentBehaviorConfig] = None) -> Dict[str, Any]:
        """
        Detect emergent behaviors in multi-agent system.
        
        Args:
            agent_system: The MARL system to analyze
            observation_window: Number of observations to analyze
            behavior_patterns: Known behavior patterns for comparison
            detection_config: Optional detection configuration
            
        Returns:
            Comprehensive behavior detection results
        """
        config = detection_config or self.config
        detection_start = time.time()
        
        logger.info(f"Starting emergent behavior detection with {observation_window} observations")
        
        try:
            # Collect system observations
            observations = await self._collect_system_observations(agent_system, observation_window)
            
            # Extract features from observations
            features = self._extract_behavioral_features(observations)
            
            # Detect patterns using multiple methods
            detected_patterns = await self._detect_behavior_patterns(features, observations)
            
            # Classify pattern novelty
            novelty_analysis = self._analyze_pattern_novelty(detected_patterns, behavior_patterns)
            
            # Perform statistical analysis
            statistical_results = self._perform_statistical_analysis(detected_patterns, observations)
            
            # Analyze temporal dynamics
            temporal_analysis = self._analyze_temporal_dynamics(observations)
            
            # Analyze network properties
            network_analysis = self._analyze_network_properties(observations)
            
            # Calculate behavioral metrics
            behavioral_metrics = self._calculate_behavioral_metrics(detected_patterns, observations)
            
            # Generate comprehensive results
            detection_results = {
                'detection_duration_ms': (time.time() - detection_start) * 1000,
                'total_observations': len(observations),
                'behaviors_detected': len(detected_patterns),
                'novel_behaviors': len([p for p in detected_patterns if p.novelty_level in [NoveltyLevel.NOVEL, NoveltyLevel.UNPRECEDENTED]]),
                'detected_patterns': [self._serialize_pattern(p) for p in detected_patterns],
                'novelty_analysis': novelty_analysis,
                'statistical_results': statistical_results,
                'temporal_analysis': temporal_analysis,
                'network_analysis': network_analysis,
                'behavioral_metrics': behavioral_metrics,
                'pattern_accuracy': self._calculate_pattern_accuracy(detected_patterns),
                'behavioral_diversity': self._calculate_behavioral_diversity(detected_patterns),
                'convergence_analysis': self._analyze_convergence(observations),
                'stability_score': self._calculate_stability_score(detected_patterns),
                'detection_performance': self.detection_metrics
            }
            
            # Update behavior history
            self._update_behavior_history(detected_patterns)
            
            logger.info(f"Emergent behavior detection completed: {len(detected_patterns)} patterns detected")
            
            return detection_results
            
        except Exception as e:
            logger.error(f"Emergent behavior detection failed: {str(e)}")
            raise
    
    async def _collect_system_observations(self, agent_system: Any, observation_window: int) -> List[Dict[str, Any]]:
        """Collect observations from the agent system."""
        observations = []
        
        # Simulate system observations - in production, this would collect real data
        for i in range(observation_window):
            observation = {
                'timestamp': datetime.now(),
                'step': i,
                'agent_states': self._simulate_agent_states(agent_system),
                'agent_actions': self._simulate_agent_actions(agent_system),
                'interactions': self._simulate_agent_interactions(agent_system),
                'rewards': self._simulate_rewards(agent_system),
                'environment_state': self._simulate_environment_state(),
                'communication_events': self._simulate_communication_events()
            }
            observations.append(observation)
            
            # Add to observation buffer
            self.observation_buffer.append(observation)
        
        return observations
    
    def _simulate_agent_states(self, agent_system: Any) -> Dict[str, Any]:
        """Simulate agent states for behavior analysis."""
        num_agents = 5  # Typical multi-agent system
        
        states = {}
        for i in range(num_agents):
            agent_id = f"agent_{i}"
            states[agent_id] = {
                'position': np.random.uniform(-10, 10, 2),
                'velocity': np.random.uniform(-1, 1, 2),
                'internal_state': np.random.uniform(0, 1, 10),
                'energy': np.random.uniform(0.5, 1.0),
                'active': np.random.choice([True, False], p=[0.9, 0.1])
            }
        
        return states
    
    def _simulate_agent_actions(self, agent_system: Any) -> Dict[str, Any]:
        """Simulate agent actions for behavior analysis."""
        num_agents = 5
        
        actions = {}
        for i in range(num_agents):
            agent_id = f"agent_{i}"
            actions[agent_id] = {
                'action_type': np.random.choice(['move', 'communicate', 'wait', 'cooperate', 'compete']),
                'action_value': np.random.uniform(0, 1, 3),
                'target_agent': f"agent_{np.random.randint(0, num_agents)}" if np.random.random() > 0.5 else None,
                'confidence': np.random.uniform(0.5, 1.0)
            }
        
        return actions
    
    def _simulate_agent_interactions(self, agent_system: Any) -> List[Dict[str, Any]]:
        """Simulate agent interactions for behavior analysis."""
        num_agents = 5
        interactions = []
        
        # Generate random interactions
        for _ in range(np.random.randint(0, 10)):
            agent_pair = np.random.choice(num_agents, 2, replace=False)
            interaction = {
                'agents': [f"agent_{agent_pair[0]}", f"agent_{agent_pair[1]}"],
                'interaction_type': np.random.choice(['cooperation', 'competition', 'communication', 'coordination']),
                'strength': np.random.uniform(0.1, 1.0),
                'duration': np.random.uniform(100, 1000),
                'outcome': np.random.choice(['success', 'failure', 'partial'])
            }
            interactions.append(interaction)
        
        return interactions
    
    def _simulate_rewards(self, agent_system: Any) -> Dict[str, float]:
        """Simulate rewards for behavior analysis."""
        num_agents = 5
        
        rewards = {}
        for i in range(num_agents):
            agent_id = f"agent_{i}"
            rewards[agent_id] = np.random.uniform(-1, 1)
        
        return rewards
    
    def _simulate_environment_state(self) -> Dict[str, Any]:
        """Simulate environment state for behavior analysis."""
        return {
            'global_state': np.random.uniform(0, 1, 5),
            'resource_availability': np.random.uniform(0, 1, 3),
            'environmental_pressure': np.random.uniform(0, 1),
            'time_step': int(time.time() * 1000) % 10000
        }
    
    def _simulate_communication_events(self) -> List[Dict[str, Any]]:
        """Simulate communication events for behavior analysis."""
        events = []
        
        for _ in range(np.random.randint(0, 5)):
            event = {
                'sender': f"agent_{np.random.randint(0, 5)}",
                'receiver': f"agent_{np.random.randint(0, 5)}",
                'message_type': np.random.choice(['request', 'response', 'broadcast', 'negotiation']),
                'content_size': np.random.randint(10, 1000),
                'priority': np.random.randint(1, 5)
            }
            events.append(event)
        
        return events
    
    def _extract_behavioral_features(self, observations: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Extract behavioral features from observations."""
        features = {}
        
        # Extract features using different methods
        for method, extractor in self.feature_extractors.items():
            try:
                method_features = extractor.extract_features(observations)
                features[method] = method_features
            except Exception as e:
                logger.warning(f"Feature extraction failed for {method}: {str(e)}")
        
        return features
    
    async def _detect_behavior_patterns(self, 
                                      features: Dict[str, np.ndarray], 
                                      observations: List[Dict[str, Any]]) -> List[BehaviorPattern]:
        """Detect behavior patterns using multiple analysis methods."""
        detected_patterns = []
        
        # Use clustering to identify patterns
        for method, feature_data in features.items():
            if feature_data is None or len(feature_data) == 0:
                continue
                
            try:
                # Apply clustering algorithm
                clustering_model = self.clustering_models[self.config.clustering_algorithm]
                
                # Standardize features
                scaler = StandardScaler()
                normalized_features = scaler.fit_transform(feature_data)
                
                # Fit clustering model
                cluster_labels = clustering_model.fit_predict(normalized_features)
                
                # Extract patterns from clusters
                patterns = self._extract_patterns_from_clusters(
                    cluster_labels, normalized_features, observations, method
                )
                
                detected_patterns.extend(patterns)
                
            except Exception as e:
                logger.warning(f"Pattern detection failed for {method}: {str(e)}")
        
        # Remove duplicate patterns
        unique_patterns = self._remove_duplicate_patterns(detected_patterns)
        
        return unique_patterns
    
    def _extract_patterns_from_clusters(self, 
                                      cluster_labels: np.ndarray, 
                                      features: np.ndarray, 
                                      observations: List[Dict[str, Any]], 
                                      method: str) -> List[BehaviorPattern]:
        """Extract behavior patterns from clustering results."""
        patterns = []
        
        unique_labels = set(cluster_labels)
        if -1 in unique_labels:  # Remove noise cluster from DBSCAN
            unique_labels.remove(-1)
        
        for label in unique_labels:
            # Get observations in this cluster
            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_observations = [observations[i] for i in cluster_indices]
            
            if len(cluster_observations) < 3:  # Minimum pattern size
                continue
            
            # Analyze cluster characteristics
            pattern = self._analyze_pattern_characteristics(
                cluster_observations, features[cluster_indices], method, label
            )
            
            if pattern:
                patterns.append(pattern)
        
        return patterns
    
    def _analyze_pattern_characteristics(self, 
                                       cluster_observations: List[Dict[str, Any]], 
                                       cluster_features: np.ndarray, 
                                       method: str, 
                                       label: int) -> Optional[BehaviorPattern]:
        """Analyze characteristics of a detected pattern."""
        try:
            # Calculate pattern properties
            start_time = cluster_observations[0]['timestamp']
            end_time = cluster_observations[-1]['timestamp']
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            # Check minimum duration
            if duration_ms < self.config.min_pattern_duration_ms:
                return None
            
            # Extract involved agents
            involved_agents = set()
            for obs in cluster_observations:
                involved_agents.update(obs['agent_states'].keys())
            
            # Determine behavior type
            behavior_type = self._classify_behavior_type(cluster_observations)
            
            # Calculate confidence
            confidence = self._calculate_pattern_confidence(cluster_features, cluster_observations)
            
            # Calculate statistical significance
            significance = self._calculate_statistical_significance(cluster_observations)
            
            # Generate pattern description
            description = self._generate_pattern_description(behavior_type, involved_agents, cluster_observations)
            
            # Create pattern object
            pattern = BehaviorPattern(
                pattern_id=f"pattern_{method}_{label}_{int(time.time() * 1000)}",
                behavior_type=behavior_type,
                agents_involved=list(involved_agents),
                detection_time=datetime.now(),
                duration_ms=duration_ms,
                confidence=confidence,
                novelty_level=NoveltyLevel.KNOWN,  # Will be updated by novelty analysis
                features=self._extract_pattern_features(cluster_features),
                statistical_significance=significance,
                description=description,
                impact_score=self._calculate_impact_score(cluster_observations),
                stability_score=self._calculate_pattern_stability(cluster_observations),
                reproducibility_score=self._calculate_reproducibility_score(cluster_observations)
            )
            
            return pattern
            
        except Exception as e:
            logger.warning(f"Pattern analysis failed: {str(e)}")
            return None
    
    def _classify_behavior_type(self, observations: List[Dict[str, Any]]) -> BehaviorType:
        """Classify the type of behavior based on observations."""
        # Analyze interaction patterns
        interaction_types = []
        for obs in observations:
            for interaction in obs['interactions']:
                interaction_types.append(interaction['interaction_type'])
        
        # Determine dominant behavior type
        if interaction_types:
            most_common = max(set(interaction_types), key=interaction_types.count)
            
            if most_common == 'cooperation':
                return BehaviorType.COOPERATION
            elif most_common == 'competition':
                return BehaviorType.COMPETITION
            elif most_common == 'coordination':
                return BehaviorType.COORDINATION
            elif most_common == 'communication':
                return BehaviorType.COMMUNICATION_PROTOCOL
        
        # Default classification based on other factors
        return BehaviorType.COLLECTIVE_INTELLIGENCE
    
    def _calculate_pattern_confidence(self, features: np.ndarray, observations: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for a pattern."""
        try:
            # Calculate feature consistency
            feature_consistency = 1.0 - np.std(features, axis=0).mean()
            
            # Calculate temporal consistency
            temporal_consistency = self._calculate_temporal_consistency(observations)
            
            # Calculate agent participation consistency
            participation_consistency = self._calculate_participation_consistency(observations)
            
            # Combine confidence metrics
            confidence = (feature_consistency + temporal_consistency + participation_consistency) / 3.0
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {str(e)}")
            return 0.5
    
    def _calculate_temporal_consistency(self, observations: List[Dict[str, Any]]) -> float:
        """Calculate temporal consistency of behavior."""
        if len(observations) < 2:
            return 0.0
        
        # Calculate time intervals
        intervals = []
        for i in range(1, len(observations)):
            interval = (observations[i]['timestamp'] - observations[i-1]['timestamp']).total_seconds()
            intervals.append(interval)
        
        # Calculate consistency as inverse of variance
        if len(intervals) > 1:
            consistency = 1.0 / (1.0 + np.var(intervals))
        else:
            consistency = 1.0
        
        return min(1.0, consistency)
    
    def _calculate_participation_consistency(self, observations: List[Dict[str, Any]]) -> float:
        """Calculate agent participation consistency."""
        agent_participation = defaultdict(int)
        
        for obs in observations:
            for agent_id in obs['agent_states']:
                if obs['agent_states'][agent_id]['active']:
                    agent_participation[agent_id] += 1
        
        if not agent_participation:
            return 0.0
        
        # Calculate consistency as inverse of participation variance
        participation_rates = [count / len(observations) for count in agent_participation.values()]
        consistency = 1.0 - np.var(participation_rates)
        
        return max(0.0, min(1.0, consistency))
    
    def _calculate_statistical_significance(self, observations: List[Dict[str, Any]]) -> float:
        """Calculate statistical significance of pattern."""
        if len(observations) < 3:
            return 0.0
        
        # Use statistical analyzer
        return self.statistical_analyzer.calculate_significance(observations)
    
    def _generate_pattern_description(self, 
                                    behavior_type: BehaviorType, 
                                    involved_agents: Set[str], 
                                    observations: List[Dict[str, Any]]) -> str:
        """Generate human-readable pattern description."""
        agent_count = len(involved_agents)
        observation_count = len(observations)
        
        description = f"{behavior_type.value.title()} behavior involving {agent_count} agents observed across {observation_count} time steps"
        
        return description
    
    def _extract_pattern_features(self, features: np.ndarray) -> Dict[str, Any]:
        """Extract key features from pattern."""
        if len(features) == 0:
            return {}
        
        return {
            'feature_mean': features.mean(axis=0).tolist(),
            'feature_std': features.std(axis=0).tolist(),
            'feature_min': features.min(axis=0).tolist(),
            'feature_max': features.max(axis=0).tolist(),
            'feature_dimension': features.shape[1],
            'observation_count': features.shape[0]
        }
    
    def _calculate_impact_score(self, observations: List[Dict[str, Any]]) -> float:
        """Calculate impact score of behavior pattern."""
        # Analyze reward changes
        reward_changes = []
        for obs in observations:
            total_reward = sum(obs['rewards'].values())
            reward_changes.append(total_reward)
        
        if len(reward_changes) < 2:
            return 0.0
        
        # Calculate impact as reward improvement
        impact = (reward_changes[-1] - reward_changes[0]) / len(reward_changes)
        
        return max(0.0, min(1.0, impact + 0.5))  # Normalize to [0, 1]
    
    def _calculate_pattern_stability(self, observations: List[Dict[str, Any]]) -> float:
        """Calculate stability score of behavior pattern."""
        if len(observations) < 3:
            return 0.0
        
        # Calculate stability based on consistency of behavior
        stability_metrics = []
        
        # Agent state stability
        for agent_id in observations[0]['agent_states']:
            agent_states = [obs['agent_states'].get(agent_id, {}) for obs in observations]
            agent_stability = self._calculate_agent_stability(agent_states)
            stability_metrics.append(agent_stability)
        
        return np.mean(stability_metrics) if stability_metrics else 0.0
    
    def _calculate_agent_stability(self, agent_states: List[Dict[str, Any]]) -> float:
        """Calculate stability for a single agent."""
        if len(agent_states) < 2:
            return 0.0
        
        # Calculate position stability
        positions = [state.get('position', [0, 0]) for state in agent_states if 'position' in state]
        if len(positions) < 2:
            return 0.0
        
        position_changes = [np.linalg.norm(np.array(positions[i+1]) - np.array(positions[i])) 
                           for i in range(len(positions)-1)]
        
        stability = 1.0 / (1.0 + np.mean(position_changes))
        
        return min(1.0, stability)
    
    def _calculate_reproducibility_score(self, observations: List[Dict[str, Any]]) -> float:
        """Calculate reproducibility score of behavior pattern."""
        # Check for similar patterns in history
        similar_patterns = 0
        
        for historical_pattern in self.behavior_history:
            if self._patterns_are_similar(observations, historical_pattern):
                similar_patterns += 1
        
        # Reproducibility score based on historical occurrences
        reproducibility = min(1.0, similar_patterns / 10.0)
        
        return reproducibility
    
    def _patterns_are_similar(self, current_observations: List[Dict[str, Any]], historical_pattern: BehaviorPattern) -> bool:
        """Check if current observations are similar to historical pattern."""
        # Simplified similarity check
        return (len(current_observations) > 5 and 
                len(historical_pattern.agents_involved) > 2 and
                abs(len(current_observations) - historical_pattern.duration_ms / 100) < 5)
    
    def _remove_duplicate_patterns(self, patterns: List[BehaviorPattern]) -> List[BehaviorPattern]:
        """Remove duplicate patterns from detection results."""
        unique_patterns = []
        
        for pattern in patterns:
            is_duplicate = False
            for existing in unique_patterns:
                if self._are_patterns_similar(pattern, existing):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_patterns.append(pattern)
        
        return unique_patterns
    
    def _are_patterns_similar(self, pattern1: BehaviorPattern, pattern2: BehaviorPattern) -> bool:
        """Check if two patterns are similar."""
        return (pattern1.behavior_type == pattern2.behavior_type and
                set(pattern1.agents_involved) == set(pattern2.agents_involved) and
                abs(pattern1.duration_ms - pattern2.duration_ms) < 1000)
    
    def _analyze_pattern_novelty(self, 
                                patterns: List[BehaviorPattern], 
                                known_patterns: List[Dict] = None) -> Dict[str, Any]:
        """Analyze novelty of detected patterns."""
        if not patterns:
            return {'novel_patterns': 0, 'known_patterns': 0, 'novelty_score': 0.0}
        
        novelty_analysis = {
            'novel_patterns': 0,
            'known_patterns': 0,
            'variant_patterns': 0,
            'unprecedented_patterns': 0,
            'novelty_score': 0.0,
            'pattern_novelty_details': []
        }
        
        for pattern in patterns:
            novelty_level = self.novelty_detector.assess_novelty(pattern, known_patterns)
            pattern.novelty_level = novelty_level
            
            if novelty_level == NoveltyLevel.NOVEL:
                novelty_analysis['novel_patterns'] += 1
            elif novelty_level == NoveltyLevel.UNPRECEDENTED:
                novelty_analysis['unprecedented_patterns'] += 1
            elif novelty_level == NoveltyLevel.VARIANT:
                novelty_analysis['variant_patterns'] += 1
            else:
                novelty_analysis['known_patterns'] += 1
            
            novelty_analysis['pattern_novelty_details'].append({
                'pattern_id': pattern.pattern_id,
                'novelty_level': novelty_level.value,
                'confidence': pattern.confidence
            })
        
        # Calculate overall novelty score
        novelty_score = (novelty_analysis['novel_patterns'] * 1.0 + 
                        novelty_analysis['unprecedented_patterns'] * 1.5 + 
                        novelty_analysis['variant_patterns'] * 0.5) / len(patterns)
        
        novelty_analysis['novelty_score'] = min(1.0, novelty_score)
        
        return novelty_analysis
    
    def _perform_statistical_analysis(self, 
                                    patterns: List[BehaviorPattern], 
                                    observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform statistical analysis on detected patterns."""
        return self.statistical_analyzer.analyze_patterns(patterns, observations)
    
    def _analyze_temporal_dynamics(self, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal dynamics of behaviors."""
        temporal_analysis = {}
        
        for analyzer_name, analyzer in self.temporal_analyzers.items():
            try:
                analysis_result = analyzer.analyze(observations)
                temporal_analysis[analyzer_name] = analysis_result
            except Exception as e:
                logger.warning(f"Temporal analysis failed for {analyzer_name}: {str(e)}")
                temporal_analysis[analyzer_name] = {'error': str(e)}
        
        return temporal_analysis
    
    def _analyze_network_properties(self, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze network properties of agent interactions."""
        return self.network_analyzer.analyze(observations)
    
    def _calculate_behavioral_metrics(self, 
                                    patterns: List[BehaviorPattern], 
                                    observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive behavioral metrics."""
        if not patterns:
            return {'error': 'No patterns to analyze'}
        
        metrics = {
            'pattern_diversity': len(set(p.behavior_type for p in patterns)),
            'average_confidence': np.mean([p.confidence for p in patterns]),
            'average_duration_ms': np.mean([p.duration_ms for p in patterns]),
            'average_impact_score': np.mean([p.impact_score for p in patterns]),
            'average_stability_score': np.mean([p.stability_score for p in patterns]),
            'behavior_type_distribution': self._calculate_behavior_distribution(patterns),
            'agent_participation_stats': self._calculate_agent_participation_stats(patterns),
            'temporal_distribution': self._calculate_temporal_distribution(patterns)
        }
        
        return metrics
    
    def _calculate_behavior_distribution(self, patterns: List[BehaviorPattern]) -> Dict[str, float]:
        """Calculate distribution of behavior types."""
        type_counts = defaultdict(int)
        for pattern in patterns:
            type_counts[pattern.behavior_type.value] += 1
        
        total = len(patterns)
        return {behavior_type: count / total for behavior_type, count in type_counts.items()}
    
    def _calculate_agent_participation_stats(self, patterns: List[BehaviorPattern]) -> Dict[str, Any]:
        """Calculate agent participation statistics."""
        agent_participation = defaultdict(int)
        
        for pattern in patterns:
            for agent_id in pattern.agents_involved:
                agent_participation[agent_id] += 1
        
        if not agent_participation:
            return {'error': 'No agent participation data'}
        
        participation_counts = list(agent_participation.values())
        
        return {
            'total_agents': len(agent_participation),
            'average_participation': np.mean(participation_counts),
            'participation_std': np.std(participation_counts),
            'most_active_agents': sorted(agent_participation.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def _calculate_temporal_distribution(self, patterns: List[BehaviorPattern]) -> Dict[str, Any]:
        """Calculate temporal distribution of patterns."""
        durations = [p.duration_ms for p in patterns]
        
        return {
            'min_duration_ms': min(durations),
            'max_duration_ms': max(durations),
            'average_duration_ms': np.mean(durations),
            'duration_std_ms': np.std(durations),
            'duration_distribution': {
                'short': sum(1 for d in durations if d < 1000),
                'medium': sum(1 for d in durations if 1000 <= d < 10000),
                'long': sum(1 for d in durations if d >= 10000)
            }
        }
    
    def _calculate_pattern_accuracy(self, patterns: List[BehaviorPattern]) -> float:
        """Calculate accuracy of pattern detection."""
        if not patterns:
            return 0.0
        
        # Calculate accuracy based on confidence and statistical significance
        accuracy_scores = []
        for pattern in patterns:
            accuracy = (pattern.confidence * 0.6 + 
                       pattern.statistical_significance * 0.4)
            accuracy_scores.append(accuracy)
        
        return np.mean(accuracy_scores)
    
    def _calculate_behavioral_diversity(self, patterns: List[BehaviorPattern]) -> float:
        """Calculate behavioral diversity score."""
        if not patterns:
            return 0.0
        
        # Calculate diversity based on behavior types and agent participation
        behavior_types = set(p.behavior_type for p in patterns)
        all_agents = set()
        for pattern in patterns:
            all_agents.update(pattern.agents_involved)
        
        # Diversity score combines type diversity and agent diversity
        type_diversity = len(behavior_types) / len(BehaviorType)
        agent_diversity = len(all_agents) / max(len(all_agents), 10)  # Assume max 10 agents
        
        diversity = (type_diversity + agent_diversity) / 2.0
        
        return min(1.0, diversity)
    
    def _analyze_convergence(self, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze convergence properties of the system."""
        if len(observations) < 10:
            return {'error': 'Insufficient data for convergence analysis'}
        
        # Analyze reward convergence
        rewards_over_time = []
        for obs in observations:
            total_reward = sum(obs['rewards'].values())
            rewards_over_time.append(total_reward)
        
        # Calculate convergence metrics
        convergence_analysis = {
            'reward_trend': self._calculate_trend(rewards_over_time),
            'reward_variance': np.var(rewards_over_time),
            'convergence_rate': self._calculate_convergence_rate(rewards_over_time),
            'stability_measure': self._calculate_stability_measure(rewards_over_time),
            'oscillation_frequency': self._calculate_oscillation_frequency(rewards_over_time)
        }
        
        return convergence_analysis
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction."""
        if len(values) < 2:
            return 'unknown'
        
        # Simple linear regression
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_convergence_rate(self, values: List[float]) -> float:
        """Calculate convergence rate."""
        if len(values) < 3:
            return 0.0
        
        # Calculate rate of change
        changes = np.diff(values)
        rate = np.mean(np.abs(changes))
        
        return 1.0 / (1.0 + rate)  # Inverse relationship
    
    def _calculate_stability_measure(self, values: List[float]) -> float:
        """Calculate stability measure."""
        if len(values) < 2:
            return 0.0
        
        # Calculate coefficient of variation
        cv = np.std(values) / (np.mean(values) + 1e-8)
        
        return 1.0 / (1.0 + cv)  # Inverse relationship
    
    def _calculate_oscillation_frequency(self, values: List[float]) -> float:
        """Calculate oscillation frequency."""
        if len(values) < 4:
            return 0.0
        
        # Count direction changes
        changes = np.diff(values)
        direction_changes = np.sum(np.diff(np.sign(changes)) != 0)
        
        return direction_changes / len(values)
    
    def _calculate_stability_score(self, patterns: List[BehaviorPattern]) -> float:
        """Calculate overall stability score."""
        if not patterns:
            return 0.0
        
        stability_scores = [p.stability_score for p in patterns]
        return np.mean(stability_scores)
    
    def _serialize_pattern(self, pattern: BehaviorPattern) -> Dict[str, Any]:
        """Serialize pattern for JSON output."""
        return {
            'pattern_id': pattern.pattern_id,
            'behavior_type': pattern.behavior_type.value,
            'agents_involved': pattern.agents_involved,
            'detection_time': pattern.detection_time.isoformat(),
            'duration_ms': pattern.duration_ms,
            'confidence': pattern.confidence,
            'novelty_level': pattern.novelty_level.value,
            'statistical_significance': pattern.statistical_significance,
            'description': pattern.description,
            'impact_score': pattern.impact_score,
            'stability_score': pattern.stability_score,
            'reproducibility_score': pattern.reproducibility_score,
            'features': pattern.features
        }
    
    def _update_behavior_history(self, patterns: List[BehaviorPattern]):
        """Update behavior history with new patterns."""
        for pattern in patterns:
            self.behavior_history.append(pattern)
        
        # Update detection metrics
        self.detection_metrics['total_patterns_detected'] += len(patterns)
        self.detection_metrics['novel_patterns_detected'] += sum(
            1 for p in patterns if p.novelty_level in [NoveltyLevel.NOVEL, NoveltyLevel.UNPRECEDENTED]
        )


# Supporting classes for feature extraction and analysis
class StatisticalFeatureExtractor:
    """Extract statistical features from observations."""
    
    def extract_features(self, observations: List[Dict[str, Any]]) -> np.ndarray:
        """Extract statistical features."""
        features = []
        
        for obs in observations:
            # Extract agent-based features
            agent_features = []
            for agent_id, state in obs['agent_states'].items():
                agent_features.extend([
                    state.get('energy', 0),
                    int(state.get('active', False)),
                    np.linalg.norm(state.get('position', [0, 0])),
                    np.linalg.norm(state.get('velocity', [0, 0]))
                ])
            
            # Extract interaction features
            interaction_features = [
                len(obs['interactions']),
                sum(1 for i in obs['interactions'] if i['outcome'] == 'success'),
                np.mean([i['strength'] for i in obs['interactions']]) if obs['interactions'] else 0
            ]
            
            # Extract communication features
            comm_features = [
                len(obs['communication_events']),
                np.mean([e['content_size'] for e in obs['communication_events']]) if obs['communication_events'] else 0
            ]
            
            # Combine all features
            obs_features = agent_features + interaction_features + comm_features
            features.append(obs_features)
        
        return np.array(features)


class TemporalFeatureExtractor:
    """Extract temporal features from observations."""
    
    def extract_features(self, observations: List[Dict[str, Any]]) -> np.ndarray:
        """Extract temporal features."""
        if len(observations) < 2:
            return np.array([])
        
        features = []
        
        for i in range(1, len(observations)):
            # Calculate temporal differences
            prev_obs = observations[i-1]
            curr_obs = observations[i]
            
            # Agent state changes
            agent_changes = []
            for agent_id in curr_obs['agent_states']:
                if agent_id in prev_obs['agent_states']:
                    prev_pos = np.array(prev_obs['agent_states'][agent_id].get('position', [0, 0]))
                    curr_pos = np.array(curr_obs['agent_states'][agent_id].get('position', [0, 0]))
                    position_change = np.linalg.norm(curr_pos - prev_pos)
                    agent_changes.append(position_change)
            
            # Interaction changes
            interaction_change = len(curr_obs['interactions']) - len(prev_obs['interactions'])
            
            # Communication changes
            comm_change = len(curr_obs['communication_events']) - len(prev_obs['communication_events'])
            
            temporal_features = [
                np.mean(agent_changes) if agent_changes else 0,
                np.std(agent_changes) if agent_changes else 0,
                interaction_change,
                comm_change
            ]
            
            features.append(temporal_features)
        
        return np.array(features)


class NetworkFeatureExtractor:
    """Extract network-based features from observations."""
    
    def extract_features(self, observations: List[Dict[str, Any]]) -> np.ndarray:
        """Extract network features."""
        features = []
        
        for obs in observations:
            # Build interaction network
            G = nx.Graph()
            
            # Add agents as nodes
            for agent_id in obs['agent_states']:
                G.add_node(agent_id)
            
            # Add interactions as edges
            for interaction in obs['interactions']:
                agents = interaction['agents']
                if len(agents) >= 2:
                    G.add_edge(agents[0], agents[1], weight=interaction['strength'])
            
            # Extract network features
            if G.number_of_nodes() > 0:
                network_features = [
                    G.number_of_nodes(),
                    G.number_of_edges(),
                    nx.density(G),
                    len(list(nx.connected_components(G))),
                    np.mean(list(dict(G.degree()).values())) if G.number_of_nodes() > 0 else 0
                ]
            else:
                network_features = [0, 0, 0, 0, 0]
            
            features.append(network_features)
        
        return np.array(features)


class BehavioralFeatureExtractor:
    """Extract behavioral features from observations."""
    
    def extract_features(self, observations: List[Dict[str, Any]]) -> np.ndarray:
        """Extract behavioral features."""
        features = []
        
        for obs in observations:
            # Extract action-based features
            action_features = []
            for agent_id, action in obs['agent_actions'].items():
                action_features.extend([
                    action.get('confidence', 0),
                    len(action.get('action_value', [])),
                    1 if action.get('target_agent') else 0
                ])
            
            # Extract reward features
            reward_features = [
                sum(obs['rewards'].values()),
                np.mean(list(obs['rewards'].values())),
                np.std(list(obs['rewards'].values())),
                max(obs['rewards'].values()) if obs['rewards'] else 0,
                min(obs['rewards'].values()) if obs['rewards'] else 0
            ]
            
            # Combine features
            behavioral_features = action_features + reward_features
            features.append(behavioral_features)
        
        return np.array(features)


class StatisticalAnalyzer:
    """Statistical analysis for behavior patterns."""
    
    def __init__(self, significance_threshold: float = 0.05):
        self.significance_threshold = significance_threshold
    
    def calculate_significance(self, observations: List[Dict[str, Any]]) -> float:
        """Calculate statistical significance of observations."""
        if len(observations) < 3:
            return 0.0
        
        # Extract rewards for significance testing
        rewards = [sum(obs['rewards'].values()) for obs in observations]
        
        # Perform one-sample t-test against zero
        try:
            t_stat, p_value = stats.ttest_1samp(rewards, 0)
            return 1.0 - p_value  # Convert to significance score
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            return 0.0
    
    def analyze_patterns(self, patterns: List[BehaviorPattern], observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns statistically."""
        if not patterns:
            return {'error': 'No patterns to analyze'}
        
        # Calculate pattern statistics
        confidences = [p.confidence for p in patterns]
        durations = [p.duration_ms for p in patterns]
        impact_scores = [p.impact_score for p in patterns]
        
        analysis = {
            'pattern_count': len(patterns),
            'confidence_stats': {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': min(confidences),
                'max': max(confidences)
            },
            'duration_stats': {
                'mean': np.mean(durations),
                'std': np.std(durations),
                'min': min(durations),
                'max': max(durations)
            },
            'impact_stats': {
                'mean': np.mean(impact_scores),
                'std': np.std(impact_scores),
                'min': min(impact_scores),
                'max': max(impact_scores)
            },
            'significance_tests': self._perform_significance_tests(patterns, observations)
        }
        
        return analysis
    
    def _perform_significance_tests(self, patterns: List[BehaviorPattern], observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform various significance tests."""
        tests = {}
        
        # Test pattern significance
        confidences = [p.confidence for p in patterns]
        if len(confidences) >= 3:
            try:
                t_stat, p_value = stats.ttest_1samp(confidences, 0.5)  # Test against random
                tests['confidence_significance'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < self.significance_threshold
                }
            except (ValueError, TypeError, AttributeError, KeyError) as e:
                tests['confidence_significance'] = {'error': 'Test failed'}
        
        return tests


class NoveltyDetector:
    """Detect novel behaviors."""
    
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
    
    def assess_novelty(self, pattern: BehaviorPattern, known_patterns: List[Dict] = None) -> NoveltyLevel:
        """Assess novelty level of a pattern."""
        if not known_patterns:
            return NoveltyLevel.NOVEL
        
        # Simple novelty assessment based on pattern characteristics
        # In production, this would use more sophisticated ML techniques
        
        if pattern.confidence > 0.9 and pattern.statistical_significance > 0.95:
            return NoveltyLevel.UNPRECEDENTED
        elif pattern.confidence > 0.8:
            return NoveltyLevel.NOVEL
        elif pattern.confidence > 0.6:
            return NoveltyLevel.VARIANT
        else:
            return NoveltyLevel.KNOWN


class NetworkAnalyzer:
    """Analyze network properties of agent interactions."""
    
    def analyze(self, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze network properties."""
        # Build cumulative interaction network
        G = nx.Graph()
        
        for obs in observations:
            # Add agents as nodes
            for agent_id in obs['agent_states']:
                G.add_node(agent_id)
            
            # Add interactions as edges
            for interaction in obs['interactions']:
                agents = interaction['agents']
                if len(agents) >= 2:
                    if G.has_edge(agents[0], agents[1]):
                        G[agents[0]][agents[1]]['weight'] += interaction['strength']
                    else:
                        G.add_edge(agents[0], agents[1], weight=interaction['strength'])
        
        # Calculate network metrics
        if G.number_of_nodes() > 0:
            analysis = {
                'node_count': G.number_of_nodes(),
                'edge_count': G.number_of_edges(),
                'density': nx.density(G),
                'connected_components': len(list(nx.connected_components(G))),
                'average_degree': np.mean(list(dict(G.degree()).values())),
                'clustering_coefficient': nx.average_clustering(G) if G.number_of_edges() > 0 else 0,
                'diameter': nx.diameter(G) if nx.is_connected(G) else 0,
                'centrality_measures': self._calculate_centrality_measures(G)
            }
        else:
            analysis = {'error': 'No network structure detected'}
        
        return analysis
    
    def _calculate_centrality_measures(self, G: nx.Graph) -> Dict[str, Any]:
        """Calculate centrality measures."""
        try:
            centrality = {
                'degree_centrality': nx.degree_centrality(G),
                'betweenness_centrality': nx.betweenness_centrality(G),
                'closeness_centrality': nx.closeness_centrality(G),
                'eigenvector_centrality': nx.eigenvector_centrality(G, max_iter=1000)
            }
            
            # Calculate average centrality scores
            avg_centrality = {}
            for measure, values in centrality.items():
                avg_centrality[f'avg_{measure}'] = np.mean(list(values.values()))
            
            return avg_centrality
            
        except Exception as e:
            return {'error': f'Centrality calculation failed: {str(e)}'}


class PhaseTransitionAnalyzer:
    """Analyze phase transitions in behavior."""
    
    def analyze(self, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze phase transitions."""
        if len(observations) < 10:
            return {'error': 'Insufficient data for phase transition analysis'}
        
        # Extract system state variables
        system_states = []
        for obs in observations:
            state = [
                sum(obs['rewards'].values()),
                len(obs['interactions']),
                len(obs['communication_events']),
                sum(1 for agent in obs['agent_states'].values() if agent.get('active', False))
            ]
            system_states.append(state)
        
        states_array = np.array(system_states)
        
        # Detect phase transitions using change point detection
        transitions = self._detect_change_points(states_array)
        
        return {
            'transition_points': transitions,
            'phase_count': len(transitions) + 1,
            'transition_analysis': self._analyze_transitions(transitions, states_array)
        }
    
    def _detect_change_points(self, states: np.ndarray) -> List[int]:
        """Detect change points in state sequence."""
        # Simple change point detection using variance
        window_size = 10
        transitions = []
        
        for i in range(window_size, len(states) - window_size):
            before = states[i-window_size:i]
            after = states[i:i+window_size]
            
            # Calculate variance difference
            var_before = np.var(before, axis=0)
            var_after = np.var(after, axis=0)
            
            if np.mean(np.abs(var_after - var_before)) > 0.5:
                transitions.append(i)
        
        return transitions
    
    def _analyze_transitions(self, transitions: List[int], states: np.ndarray) -> Dict[str, Any]:
        """Analyze characteristics of transitions."""
        if not transitions:
            return {'message': 'No transitions detected'}
        
        analysis = {
            'transition_frequency': len(transitions) / len(states),
            'average_phase_length': np.mean(np.diff([0] + transitions + [len(states)])),
            'transition_magnitudes': []
        }
        
        # Calculate transition magnitudes
        for t in transitions:
            if t > 0 and t < len(states) - 1:
                magnitude = np.linalg.norm(states[t] - states[t-1])
                analysis['transition_magnitudes'].append(magnitude)
        
        analysis['average_transition_magnitude'] = np.mean(analysis['transition_magnitudes']) if analysis['transition_magnitudes'] else 0
        
        return analysis


class SynchronizationAnalyzer:
    """Analyze synchronization patterns in agent behavior."""
    
    def analyze(self, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze synchronization patterns."""
        if len(observations) < 5:
            return {'error': 'Insufficient data for synchronization analysis'}
        
        # Extract agent positions over time
        agent_positions = defaultdict(list)
        for obs in observations:
            for agent_id, state in obs['agent_states'].items():
                position = state.get('position', [0, 0])
                agent_positions[agent_id].append(position)
        
        # Calculate synchronization metrics
        sync_analysis = {
            'position_synchronization': self._calculate_position_sync(agent_positions),
            'action_synchronization': self._calculate_action_sync(observations),
            'reward_synchronization': self._calculate_reward_sync(observations)
        }
        
        return sync_analysis
    
    def _calculate_position_sync(self, agent_positions: Dict[str, List]) -> Dict[str, Any]:
        """Calculate position synchronization."""
        if len(agent_positions) < 2:
            return {'error': 'Insufficient agents for synchronization analysis'}
        
        # Calculate pairwise correlations
        correlations = []
        agent_ids = list(agent_positions.keys())
        
        for i in range(len(agent_ids)):
            for j in range(i+1, len(agent_ids)):
                pos1 = np.array(agent_positions[agent_ids[i]])
                pos2 = np.array(agent_positions[agent_ids[j]])
                
                if len(pos1) == len(pos2) and len(pos1) > 1:
                    # Calculate correlation for x and y coordinates
                    corr_x = np.corrcoef(pos1[:, 0], pos2[:, 0])[0, 1]
                    corr_y = np.corrcoef(pos1[:, 1], pos2[:, 1])[0, 1]
                    
                    if not np.isnan(corr_x) and not np.isnan(corr_y):
                        correlations.append((corr_x + corr_y) / 2)
        
        return {
            'average_correlation': np.mean(correlations) if correlations else 0,
            'max_correlation': max(correlations) if correlations else 0,
            'synchronization_score': np.mean([abs(c) for c in correlations]) if correlations else 0
        }
    
    def _calculate_action_sync(self, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate action synchronization."""
        # Count simultaneous actions
        simultaneous_actions = 0
        total_timesteps = len(observations)
        
        for obs in observations:
            action_types = [action['action_type'] for action in obs['agent_actions'].values()]
            unique_actions = set(action_types)
            
            if len(unique_actions) == 1 and len(action_types) > 1:
                simultaneous_actions += 1
        
        return {
            'simultaneous_action_rate': simultaneous_actions / total_timesteps,
            'synchronization_score': simultaneous_actions / total_timesteps
        }
    
    def _calculate_reward_sync(self, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate reward synchronization."""
        # Extract agent rewards over time
        agent_rewards = defaultdict(list)
        for obs in observations:
            for agent_id, reward in obs['rewards'].items():
                agent_rewards[agent_id].append(reward)
        
        # Calculate reward correlations
        correlations = []
        agent_ids = list(agent_rewards.keys())
        
        for i in range(len(agent_ids)):
            for j in range(i+1, len(agent_ids)):
                rewards1 = agent_rewards[agent_ids[i]]
                rewards2 = agent_rewards[agent_ids[j]]
                
                if len(rewards1) == len(rewards2) and len(rewards1) > 1:
                    corr = np.corrcoef(rewards1, rewards2)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
        
        return {
            'average_reward_correlation': np.mean(correlations) if correlations else 0,
            'reward_synchronization_score': np.mean([abs(c) for c in correlations]) if correlations else 0
        }


class TrendAnalyzer:
    """Analyze trends in behavior over time."""
    
    def analyze(self, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze behavioral trends."""
        if len(observations) < 3:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Extract time series data
        rewards_over_time = [sum(obs['rewards'].values()) for obs in observations]
        interactions_over_time = [len(obs['interactions']) for obs in observations]
        communication_over_time = [len(obs['communication_events']) for obs in observations]
        
        # Analyze trends
        trends = {
            'reward_trend': self._analyze_trend(rewards_over_time),
            'interaction_trend': self._analyze_trend(interactions_over_time),
            'communication_trend': self._analyze_trend(communication_over_time)
        }
        
        return trends
    
    def _analyze_trend(self, values: List[float]) -> Dict[str, Any]:
        """Analyze trend in values."""
        if len(values) < 2:
            return {'error': 'Insufficient data'}
        
        # Calculate linear regression
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # Determine trend direction
        if slope > 0.01:
            direction = 'increasing'
        elif slope < -0.01:
            direction = 'decreasing'
        else:
            direction = 'stable'
        
        return {
            'direction': direction,
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'strength': abs(slope),
            'confidence': 1 - p_value if p_value < 0.05 else 0
        }