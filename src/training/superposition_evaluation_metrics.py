"""
Superposition Evaluation Metrics for MARL Training

This module implements comprehensive evaluation metrics specifically designed
for assessing the quality of superposition outputs in cascade agent systems.
It provides detailed analysis of superposition properties, coordination quality,
and training progress.

Key Features:
- Superposition quality metrics (entropy, consistency, diversity)
- Coordination evaluation between sequential agents
- Confidence calibration assessment
- Decision stability analysis
- Training progress tracking
- Comprehensive visualization and reporting

Author: AGENT 9 - Superposition-Aware Training Framework
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import math
from collections import deque, defaultdict
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of evaluation metrics"""
    SUPERPOSITION_QUALITY = "superposition_quality"
    COORDINATION_QUALITY = "coordination_quality"
    CONFIDENCE_CALIBRATION = "confidence_calibration"
    DECISION_STABILITY = "decision_stability"
    ENTROPY_ANALYSIS = "entropy_analysis"
    CONSISTENCY_ANALYSIS = "consistency_analysis"
    DIVERSITY_ANALYSIS = "diversity_analysis"
    TEMPORAL_ANALYSIS = "temporal_analysis"


@dataclass
class SuperpositionMetrics:
    """Container for superposition-specific metrics"""
    # Entropy metrics
    entropy_mean: float
    entropy_std: float
    entropy_trend: float
    target_entropy_compliance: float
    
    # Consistency metrics
    consistency_score: float
    consistency_stability: float
    pairwise_consistency: float
    
    # Confidence metrics
    confidence_calibration: float
    confidence_variance: float
    confidence_distribution: List[float]
    
    # Diversity metrics
    decision_diversity: float
    superposition_coverage: float
    effective_superposition_dim: float
    
    # Quality scores
    overall_quality: float
    stability_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'entropy_mean': self.entropy_mean,
            'entropy_std': self.entropy_std,
            'entropy_trend': self.entropy_trend,
            'target_entropy_compliance': self.target_entropy_compliance,
            'consistency_score': self.consistency_score,
            'consistency_stability': self.consistency_stability,
            'pairwise_consistency': self.pairwise_consistency,
            'confidence_calibration': self.confidence_calibration,
            'confidence_variance': self.confidence_variance,
            'confidence_distribution': self.confidence_distribution,
            'decision_diversity': self.decision_diversity,
            'superposition_coverage': self.superposition_coverage,
            'effective_superposition_dim': self.effective_superposition_dim,
            'overall_quality': self.overall_quality,
            'stability_score': self.stability_score
        }


@dataclass
class CoordinationMetrics:
    """Container for coordination-specific metrics"""
    # Sequential coordination
    sequential_alignment: float
    temporal_consistency: float
    information_flow: float
    
    # Pairwise coordination
    pairwise_similarities: Dict[str, float]
    coordination_stability: float
    
    # System-level coordination
    system_coherence: float
    coordination_efficiency: float
    
    # Performance correlation
    coordination_performance_correlation: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'sequential_alignment': self.sequential_alignment,
            'temporal_consistency': self.temporal_consistency,
            'information_flow': self.information_flow,
            'pairwise_similarities': self.pairwise_similarities,
            'coordination_stability': self.coordination_stability,
            'system_coherence': self.system_coherence,
            'coordination_efficiency': self.coordination_efficiency,
            'coordination_performance_correlation': self.coordination_performance_correlation
        }


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report"""
    timestamp: str
    episode_range: Tuple[int, int]
    
    # Agent-specific metrics
    agent_metrics: Dict[str, SuperpositionMetrics]
    
    # System-level metrics
    coordination_metrics: CoordinationMetrics
    
    # Training progress
    training_progress: Dict[str, List[float]]
    
    # Performance correlation
    performance_correlations: Dict[str, float]
    
    # Recommendations
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'episode_range': self.episode_range,
            'agent_metrics': {agent: metrics.to_dict() for agent, metrics in self.agent_metrics.items()},
            'coordination_metrics': self.coordination_metrics.to_dict(),
            'training_progress': self.training_progress,
            'performance_correlations': self.performance_correlations,
            'recommendations': self.recommendations
        }


class SuperpositionQualityAnalyzer:
    """Analyzer for superposition quality metrics"""
    
    def __init__(self, superposition_dim: int, target_entropy: float = 1.0):
        self.superposition_dim = superposition_dim
        self.target_entropy = target_entropy
        self.max_entropy = math.log(superposition_dim)
        
    def analyze_superposition_quality(self, superposition_outputs: List[Dict[str, torch.Tensor]]) -> SuperpositionMetrics:
        """Analyze quality of superposition outputs"""
        # Extract data
        entropies = []
        consistencies = []
        confidence_weights = []
        decision_states = []
        
        for output in superposition_outputs:
            if 'entropy' in output:
                entropies.append(output['entropy'].item() if isinstance(output['entropy'], torch.Tensor) else output['entropy'])
            
            if 'consistency_score' in output:
                consistencies.append(output['consistency_score'].item() if isinstance(output['consistency_score'], torch.Tensor) else output['consistency_score'])
            
            if 'confidence_weights' in output:
                confidence_weights.append(output['confidence_weights'])
            
            if 'decision_states' in output:
                decision_states.append(output['decision_states'])
        
        # Calculate entropy metrics
        entropy_metrics = self._calculate_entropy_metrics(entropies)
        
        # Calculate consistency metrics
        consistency_metrics = self._calculate_consistency_metrics(consistencies, decision_states)
        
        # Calculate confidence metrics
        confidence_metrics = self._calculate_confidence_metrics(confidence_weights)
        
        # Calculate diversity metrics
        diversity_metrics = self._calculate_diversity_metrics(decision_states, confidence_weights)
        
        # Calculate overall quality
        overall_quality = self._calculate_overall_quality(
            entropy_metrics, consistency_metrics, confidence_metrics, diversity_metrics
        )
        
        return SuperpositionMetrics(
            entropy_mean=entropy_metrics['mean'],
            entropy_std=entropy_metrics['std'],
            entropy_trend=entropy_metrics['trend'],
            target_entropy_compliance=entropy_metrics['target_compliance'],
            consistency_score=consistency_metrics['mean'],
            consistency_stability=consistency_metrics['stability'],
            pairwise_consistency=consistency_metrics['pairwise'],
            confidence_calibration=confidence_metrics['calibration'],
            confidence_variance=confidence_metrics['variance'],
            confidence_distribution=confidence_metrics['distribution'],
            decision_diversity=diversity_metrics['diversity'],
            superposition_coverage=diversity_metrics['coverage'],
            effective_superposition_dim=diversity_metrics['effective_dim'],
            overall_quality=overall_quality['overall'],
            stability_score=overall_quality['stability']
        )
    
    def _calculate_entropy_metrics(self, entropies: List[float]) -> Dict[str, float]:
        """Calculate entropy-related metrics"""
        if not entropies:
            return {'mean': 0.0, 'std': 0.0, 'trend': 0.0, 'target_compliance': 0.0}
        
        entropies = np.array(entropies)
        
        # Basic statistics
        mean_entropy = np.mean(entropies)
        std_entropy = np.std(entropies)
        
        # Trend analysis
        if len(entropies) > 10:
            x = np.arange(len(entropies))
            slope, _, _, _, _ = stats.linregress(x, entropies)
            trend = slope
        else:
            trend = 0.0
        
        # Target compliance
        target_compliance = 1.0 - np.mean(np.abs(entropies - self.target_entropy) / self.target_entropy)
        target_compliance = max(0.0, target_compliance)
        
        return {
            'mean': mean_entropy,
            'std': std_entropy,
            'trend': trend,
            'target_compliance': target_compliance
        }
    
    def _calculate_consistency_metrics(self, consistencies: List[float], 
                                     decision_states: List[torch.Tensor]) -> Dict[str, float]:
        """Calculate consistency-related metrics"""
        if not consistencies:
            return {'mean': 0.0, 'stability': 0.0, 'pairwise': 0.0}
        
        consistencies = np.array(consistencies)
        
        # Basic statistics
        mean_consistency = np.mean(consistencies)
        stability = 1.0 / (1.0 + np.std(consistencies))
        
        # Pairwise consistency analysis
        pairwise_consistency = 0.0
        if decision_states:
            pairwise_scores = []
            for states in decision_states:
                if len(states.shape) >= 2:  # [superposition_dim, action_dim]
                    pairwise_score = self._calculate_pairwise_consistency(states)
                    pairwise_scores.append(pairwise_score)
            
            pairwise_consistency = np.mean(pairwise_scores) if pairwise_scores else 0.0
        
        return {
            'mean': mean_consistency,
            'stability': stability,
            'pairwise': pairwise_consistency
        }
    
    def _calculate_pairwise_consistency(self, decision_states: torch.Tensor) -> float:
        """Calculate pairwise consistency between decision states"""
        if len(decision_states.shape) < 2:
            return 0.0
        
        num_states = decision_states.shape[0]
        if num_states < 2:
            return 1.0
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(num_states):
            for j in range(i + 1, num_states):
                state_i = F.softmax(decision_states[i], dim=0)
                state_j = F.softmax(decision_states[j], dim=0)
                
                # Use cosine similarity
                similarity = F.cosine_similarity(state_i, state_j, dim=0).item()
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_confidence_metrics(self, confidence_weights: List[torch.Tensor]) -> Dict[str, float]:
        """Calculate confidence-related metrics"""
        if not confidence_weights:
            return {'calibration': 0.0, 'variance': 0.0, 'distribution': []}
        
        # Concatenate all confidence weights
        all_weights = torch.cat(confidence_weights, dim=0)
        
        # Calculate calibration (how well-distributed confidence is)
        mean_confidence = torch.mean(all_weights, dim=0)
        target_uniform = torch.ones_like(mean_confidence) / len(mean_confidence)
        calibration = 1.0 - torch.mean(torch.abs(mean_confidence - target_uniform)).item()
        
        # Calculate variance
        variance = torch.var(all_weights).item()
        
        # Distribution analysis
        confidence_dist = torch.mean(all_weights, dim=0).tolist()
        
        return {
            'calibration': calibration,
            'variance': variance,
            'distribution': confidence_dist
        }
    
    def _calculate_diversity_metrics(self, decision_states: List[torch.Tensor], 
                                   confidence_weights: List[torch.Tensor]) -> Dict[str, float]:
        """Calculate diversity-related metrics"""
        if not decision_states:
            return {'diversity': 0.0, 'coverage': 0.0, 'effective_dim': 0.0}
        
        # Calculate decision diversity
        diversity_scores = []
        for states in decision_states:
            if len(states.shape) >= 2:
                diversity_score = self._calculate_decision_diversity(states)
                diversity_scores.append(diversity_score)
        
        decision_diversity = np.mean(diversity_scores) if diversity_scores else 0.0
        
        # Calculate superposition coverage
        coverage_scores = []
        for weights in confidence_weights:
            # Effective number of dimensions used
            entropy_weights = -(weights * torch.log(weights + 1e-8)).sum()
            effective_dim = torch.exp(entropy_weights).item()
            coverage = effective_dim / self.superposition_dim
            coverage_scores.append(coverage)
        
        superposition_coverage = np.mean(coverage_scores) if coverage_scores else 0.0
        effective_dim = np.mean([torch.exp(-(w * torch.log(w + 1e-8)).sum()).item() 
                                for w in confidence_weights]) if confidence_weights else 0.0
        
        return {
            'diversity': decision_diversity,
            'coverage': superposition_coverage,
            'effective_dim': effective_dim
        }
    
    def _calculate_decision_diversity(self, decision_states: torch.Tensor) -> float:
        """Calculate diversity of decision states"""
        if len(decision_states.shape) < 2:
            return 0.0
        
        # Convert to probabilities
        probs = F.softmax(decision_states, dim=1)
        
        # Calculate mutual information between decision states
        num_states = probs.shape[0]
        if num_states < 2:
            return 0.0
        
        diversity_scores = []
        for i in range(num_states):
            for j in range(i + 1, num_states):
                # Calculate KL divergence
                kl_div = F.kl_div(torch.log(probs[i] + 1e-8), probs[j], reduction='sum')
                diversity_scores.append(kl_div.item())
        
        return np.mean(diversity_scores) if diversity_scores else 0.0
    
    def _calculate_overall_quality(self, entropy_metrics: Dict[str, float],
                                 consistency_metrics: Dict[str, float],
                                 confidence_metrics: Dict[str, float],
                                 diversity_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate overall quality scores"""
        # Weight different components
        entropy_score = entropy_metrics['target_compliance'] * 0.3
        consistency_score = consistency_metrics['mean'] * 0.3
        confidence_score = confidence_metrics['calibration'] * 0.2
        diversity_score = diversity_metrics['diversity'] * 0.2
        
        overall_quality = entropy_score + consistency_score + confidence_score + diversity_score
        
        # Stability score
        stability_factors = [
            consistency_metrics['stability'],
            1.0 / (1.0 + entropy_metrics['std']),
            1.0 / (1.0 + confidence_metrics['variance'])
        ]
        stability_score = np.mean(stability_factors)
        
        return {
            'overall': overall_quality,
            'stability': stability_score
        }


class CoordinationAnalyzer:
    """Analyzer for coordination quality metrics"""
    
    def __init__(self, agent_names: List[str]):
        self.agent_names = agent_names
        self.num_agents = len(agent_names)
    
    def analyze_coordination(self, agent_outputs: List[Dict[str, Dict[str, torch.Tensor]]],
                           performance_data: List[Dict[str, float]]) -> CoordinationMetrics:
        """Analyze coordination quality between agents"""
        # Extract coordination data
        coordination_data = self._extract_coordination_data(agent_outputs)
        
        # Calculate sequential alignment
        sequential_alignment = self._calculate_sequential_alignment(coordination_data)
        
        # Calculate temporal consistency
        temporal_consistency = self._calculate_temporal_consistency(coordination_data)
        
        # Calculate information flow
        information_flow = self._calculate_information_flow(coordination_data)
        
        # Calculate pairwise similarities
        pairwise_similarities = self._calculate_pairwise_similarities(coordination_data)
        
        # Calculate coordination stability
        coordination_stability = self._calculate_coordination_stability(coordination_data)
        
        # Calculate system coherence
        system_coherence = self._calculate_system_coherence(coordination_data)
        
        # Calculate coordination efficiency
        coordination_efficiency = self._calculate_coordination_efficiency(coordination_data, performance_data)
        
        # Calculate coordination-performance correlation
        coordination_performance_correlation = self._calculate_coordination_performance_correlation(
            coordination_data, performance_data
        )
        
        return CoordinationMetrics(
            sequential_alignment=sequential_alignment,
            temporal_consistency=temporal_consistency,
            information_flow=information_flow,
            pairwise_similarities=pairwise_similarities,
            coordination_stability=coordination_stability,
            system_coherence=system_coherence,
            coordination_efficiency=coordination_efficiency,
            coordination_performance_correlation=coordination_performance_correlation
        )
    
    def _extract_coordination_data(self, agent_outputs: List[Dict[str, Dict[str, torch.Tensor]]]) -> Dict[str, List[torch.Tensor]]:
        """Extract coordination-relevant data from agent outputs"""
        coordination_data = {agent: [] for agent in self.agent_names}
        
        for episode_outputs in agent_outputs:
            for agent_name in self.agent_names:
                if agent_name in episode_outputs:
                    agent_output = episode_outputs[agent_name]
                    
                    # Extract confidence weights as coordination signal
                    if 'confidence_weights' in agent_output:
                        coordination_data[agent_name].append(agent_output['confidence_weights'])
        
        return coordination_data
    
    def _calculate_sequential_alignment(self, coordination_data: Dict[str, List[torch.Tensor]]) -> float:
        """Calculate alignment between sequential agents"""
        if self.num_agents < 2:
            return 1.0
        
        alignment_scores = []
        
        for i in range(len(self.agent_names) - 1):
            current_agent = self.agent_names[i]
            next_agent = self.agent_names[i + 1]
            
            current_data = coordination_data.get(current_agent, [])
            next_data = coordination_data.get(next_agent, [])
            
            # Calculate alignment between consecutive agents
            if current_data and next_data:
                min_len = min(len(current_data), len(next_data))
                agent_alignment = []
                
                for j in range(min_len):
                    if current_data[j].shape == next_data[j].shape:
                        similarity = F.cosine_similarity(
                            current_data[j].flatten(),
                            next_data[j].flatten(),
                            dim=0
                        ).item()
                        agent_alignment.append(similarity)
                
                if agent_alignment:
                    alignment_scores.append(np.mean(agent_alignment))
        
        return np.mean(alignment_scores) if alignment_scores else 0.0
    
    def _calculate_temporal_consistency(self, coordination_data: Dict[str, List[torch.Tensor]]) -> float:
        """Calculate temporal consistency of coordination signals"""
        consistency_scores = []
        
        for agent_name, agent_data in coordination_data.items():
            if len(agent_data) < 2:
                continue
            
            # Calculate consistency across time
            temporal_similarities = []
            for i in range(len(agent_data) - 1):
                if agent_data[i].shape == agent_data[i + 1].shape:
                    similarity = F.cosine_similarity(
                        agent_data[i].flatten(),
                        agent_data[i + 1].flatten(),
                        dim=0
                    ).item()
                    temporal_similarities.append(similarity)
            
            if temporal_similarities:
                consistency_scores.append(np.mean(temporal_similarities))
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _calculate_information_flow(self, coordination_data: Dict[str, List[torch.Tensor]]) -> float:
        """Calculate information flow between agents"""
        if self.num_agents < 2:
            return 0.0
        
        information_flows = []
        
        for i in range(len(self.agent_names) - 1):
            current_agent = self.agent_names[i]
            next_agent = self.agent_names[i + 1]
            
            current_data = coordination_data.get(current_agent, [])
            next_data = coordination_data.get(next_agent, [])
            
            if current_data and next_data:
                # Calculate mutual information (simplified)
                min_len = min(len(current_data), len(next_data))
                
                # Convert to discrete values for MI calculation
                current_discrete = []
                next_discrete = []
                
                for j in range(min_len):
                    if current_data[j].shape == next_data[j].shape:
                        current_discrete.append(torch.argmax(current_data[j]).item())
                        next_discrete.append(torch.argmax(next_data[j]).item())
                
                if current_discrete and next_discrete:
                    mi = mutual_info_score(current_discrete, next_discrete)
                    information_flows.append(mi)
        
        return np.mean(information_flows) if information_flows else 0.0
    
    def _calculate_pairwise_similarities(self, coordination_data: Dict[str, List[torch.Tensor]]) -> Dict[str, float]:
        """Calculate pairwise similarities between agents"""
        similarities = {}
        
        for i in range(len(self.agent_names)):
            for j in range(i + 1, len(self.agent_names)):
                agent_i = self.agent_names[i]
                agent_j = self.agent_names[j]
                
                data_i = coordination_data.get(agent_i, [])
                data_j = coordination_data.get(agent_j, [])
                
                if data_i and data_j:
                    min_len = min(len(data_i), len(data_j))
                    pair_similarities = []
                    
                    for k in range(min_len):
                        if data_i[k].shape == data_j[k].shape:
                            similarity = F.cosine_similarity(
                                data_i[k].flatten(),
                                data_j[k].flatten(),
                                dim=0
                            ).item()
                            pair_similarities.append(similarity)
                    
                    if pair_similarities:
                        similarities[f"{agent_i}_{agent_j}"] = np.mean(pair_similarities)
        
        return similarities
    
    def _calculate_coordination_stability(self, coordination_data: Dict[str, List[torch.Tensor]]) -> float:
        """Calculate stability of coordination patterns"""
        stability_scores = []
        
        for agent_name, agent_data in coordination_data.items():
            if len(agent_data) < 10:
                continue
            
            # Calculate variance in coordination signals
            stacked_data = torch.stack(agent_data)
            variance = torch.var(stacked_data, dim=0)
            mean_variance = torch.mean(variance).item()
            
            # Lower variance indicates higher stability
            stability = 1.0 / (1.0 + mean_variance)
            stability_scores.append(stability)
        
        return np.mean(stability_scores) if stability_scores else 0.0
    
    def _calculate_system_coherence(self, coordination_data: Dict[str, List[torch.Tensor]]) -> float:
        """Calculate overall system coherence"""
        if self.num_agents < 2:
            return 1.0
        
        # Calculate average pairwise similarity across all agents
        similarities = self._calculate_pairwise_similarities(coordination_data)
        
        if not similarities:
            return 0.0
        
        return np.mean(list(similarities.values()))
    
    def _calculate_coordination_efficiency(self, coordination_data: Dict[str, List[torch.Tensor]],
                                         performance_data: List[Dict[str, float]]) -> float:
        """Calculate efficiency of coordination relative to performance"""
        if not performance_data:
            return 0.0
        
        # Calculate coordination effort (variance in coordination signals)
        coordination_effort = 0.0
        total_agents = 0
        
        for agent_name, agent_data in coordination_data.items():
            if agent_data:
                stacked_data = torch.stack(agent_data)
                effort = torch.mean(torch.var(stacked_data, dim=0)).item()
                coordination_effort += effort
                total_agents += 1
        
        if total_agents > 0:
            coordination_effort /= total_agents
        
        # Calculate performance gain
        performance_gains = []
        for perf_data in performance_data:
            total_performance = sum(perf_data.values())
            performance_gains.append(total_performance)
        
        avg_performance = np.mean(performance_gains) if performance_gains else 0.0
        
        # Efficiency = performance / coordination effort
        if coordination_effort > 0:
            efficiency = avg_performance / coordination_effort
        else:
            efficiency = avg_performance
        
        return min(1.0, efficiency)
    
    def _calculate_coordination_performance_correlation(self, coordination_data: Dict[str, List[torch.Tensor]],
                                                      performance_data: List[Dict[str, float]]) -> float:
        """Calculate correlation between coordination and performance"""
        if not performance_data:
            return 0.0
        
        # Calculate coordination scores
        coordination_scores = []
        for i in range(len(performance_data)):
            episode_coordination = 0.0
            
            for agent_name in self.agent_names:
                if agent_name in coordination_data and i < len(coordination_data[agent_name]):
                    # Use entropy of confidence weights as coordination measure
                    weights = coordination_data[agent_name][i]
                    entropy = -(weights * torch.log(weights + 1e-8)).sum().item()
                    episode_coordination += entropy
            
            coordination_scores.append(episode_coordination)
        
        # Calculate performance scores
        performance_scores = [sum(perf_data.values()) for perf_data in performance_data]
        
        # Calculate correlation
        if len(coordination_scores) == len(performance_scores) and len(coordination_scores) > 1:
            correlation, _ = stats.pearsonr(coordination_scores, performance_scores)
            return correlation
        
        return 0.0


class SuperpositionEvaluationSystem:
    """Main evaluation system for superposition-aware training"""
    
    def __init__(self, agent_names: List[str], superposition_dim: int,
                 target_entropy: float = 1.0, evaluation_window: int = 100):
        self.agent_names = agent_names
        self.superposition_dim = superposition_dim
        self.target_entropy = target_entropy
        self.evaluation_window = evaluation_window
        
        # Initialize analyzers
        self.superposition_analyzer = SuperpositionQualityAnalyzer(superposition_dim, target_entropy)
        self.coordination_analyzer = CoordinationAnalyzer(agent_names)
        
        # Data storage
        self.episode_data = deque(maxlen=evaluation_window)
        self.evaluation_history = []
        
        # Metrics tracking
        self.metrics_history = defaultdict(list)
        
        logger.info(f"Initialized SuperpositionEvaluationSystem for {len(agent_names)} agents")
    
    def add_episode_data(self, episode_data: Dict[str, Any]):
        """Add episode data for evaluation"""
        self.episode_data.append(episode_data)
    
    def evaluate_current_state(self) -> EvaluationReport:
        """Evaluate current state of training"""
        if not self.episode_data:
            return self._create_empty_report()
        
        # Extract data for evaluation
        agent_outputs = []
        performance_data = []
        
        for episode in self.episode_data:
            if 'agent_outputs' in episode:
                agent_outputs.append(episode['agent_outputs'])
            
            if 'performance_data' in episode:
                performance_data.append(episode['performance_data'])
        
        # Analyze superposition quality for each agent
        agent_metrics = {}
        for agent_name in self.agent_names:
            agent_superposition_outputs = []
            
            for episode_outputs in agent_outputs:
                if agent_name in episode_outputs:
                    agent_superposition_outputs.append(episode_outputs[agent_name])
            
            if agent_superposition_outputs:
                agent_metrics[agent_name] = self.superposition_analyzer.analyze_superposition_quality(
                    agent_superposition_outputs
                )
        
        # Analyze coordination
        coordination_metrics = self.coordination_analyzer.analyze_coordination(
            agent_outputs, performance_data
        )
        
        # Calculate training progress
        training_progress = self._calculate_training_progress()
        
        # Calculate performance correlations
        performance_correlations = self._calculate_performance_correlations()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(agent_metrics, coordination_metrics)
        
        # Create report
        report = EvaluationReport(
            timestamp=str(pd.Timestamp.now()),
            episode_range=(max(0, len(self.episode_data) - self.evaluation_window), len(self.episode_data)),
            agent_metrics=agent_metrics,
            coordination_metrics=coordination_metrics,
            training_progress=training_progress,
            performance_correlations=performance_correlations,
            recommendations=recommendations
        )
        
        # Store in history
        self.evaluation_history.append(report)
        
        # Update metrics history
        self._update_metrics_history(report)
        
        return report
    
    def _create_empty_report(self) -> EvaluationReport:
        """Create empty evaluation report"""
        return EvaluationReport(
            timestamp=str(pd.Timestamp.now()),
            episode_range=(0, 0),
            agent_metrics={},
            coordination_metrics=CoordinationMetrics(
                sequential_alignment=0.0,
                temporal_consistency=0.0,
                information_flow=0.0,
                pairwise_similarities={},
                coordination_stability=0.0,
                system_coherence=0.0,
                coordination_efficiency=0.0,
                coordination_performance_correlation=0.0
            ),
            training_progress={},
            performance_correlations={},
            recommendations=[]
        )
    
    def _calculate_training_progress(self) -> Dict[str, List[float]]:
        """Calculate training progress metrics"""
        progress = {}
        
        # Extract metrics over time
        for agent_name in self.agent_names:
            agent_progress = {
                'superposition_quality': [],
                'entropy_compliance': [],
                'consistency_score': [],
                'coordination_score': []
            }
            
            for episode in self.episode_data:
                if 'agent_outputs' in episode and agent_name in episode['agent_outputs']:
                    agent_output = episode['agent_outputs'][agent_name]
                    
                    # Extract quality metrics
                    if 'entropy' in agent_output:
                        entropy = agent_output['entropy'].item() if isinstance(agent_output['entropy'], torch.Tensor) else agent_output['entropy']
                        compliance = 1.0 - abs(entropy - self.target_entropy) / self.target_entropy
                        agent_progress['entropy_compliance'].append(compliance)
                    
                    if 'consistency_score' in agent_output:
                        consistency = agent_output['consistency_score'].item() if isinstance(agent_output['consistency_score'], torch.Tensor) else agent_output['consistency_score']
                        agent_progress['consistency_score'].append(consistency)
            
            progress[agent_name] = agent_progress
        
        return progress
    
    def _calculate_performance_correlations(self) -> Dict[str, float]:
        """Calculate correlations between different metrics"""
        correlations = {}
        
        if len(self.evaluation_history) < 2:
            return correlations
        
        # Extract metrics from history
        quality_scores = []
        coordination_scores = []
        performance_scores = []
        
        for report in self.evaluation_history[-10:]:  # Last 10 evaluations
            # Average agent quality
            if report.agent_metrics:
                avg_quality = np.mean([metrics.overall_quality for metrics in report.agent_metrics.values()])
                quality_scores.append(avg_quality)
            
            # Coordination score
            coordination_scores.append(report.coordination_metrics.system_coherence)
            
            # Performance score (from training progress)
            if report.training_progress:
                avg_performance = 0.0
                count = 0
                for agent_progress in report.training_progress.values():
                    if agent_progress.get('entropy_compliance'):
                        avg_performance += np.mean(agent_progress['entropy_compliance'][-10:])
                        count += 1
                
                if count > 0:
                    performance_scores.append(avg_performance / count)
        
        # Calculate correlations
        if len(quality_scores) >= 2 and len(coordination_scores) >= 2:
            corr, _ = stats.pearsonr(quality_scores, coordination_scores)
            correlations['quality_coordination'] = corr
        
        if len(quality_scores) >= 2 and len(performance_scores) >= 2:
            corr, _ = stats.pearsonr(quality_scores, performance_scores)
            correlations['quality_performance'] = corr
        
        if len(coordination_scores) >= 2 and len(performance_scores) >= 2:
            corr, _ = stats.pearsonr(coordination_scores, performance_scores)
            correlations['coordination_performance'] = corr
        
        return correlations
    
    def _generate_recommendations(self, agent_metrics: Dict[str, SuperpositionMetrics],
                                coordination_metrics: CoordinationMetrics) -> List[str]:
        """Generate recommendations based on evaluation"""
        recommendations = []
        
        # Analyze agent-specific issues
        for agent_name, metrics in agent_metrics.items():
            if metrics.entropy_mean < self.target_entropy * 0.8:
                recommendations.append(f"Agent {agent_name}: Increase entropy regularization to encourage exploration")
            
            if metrics.consistency_score < 0.6:
                recommendations.append(f"Agent {agent_name}: Improve consistency penalty to reduce conflicting decisions")
            
            if metrics.confidence_calibration < 0.5:
                recommendations.append(f"Agent {agent_name}: Improve confidence calibration training")
            
            if metrics.overall_quality < 0.6:
                recommendations.append(f"Agent {agent_name}: Overall superposition quality needs improvement")
        
        # Analyze coordination issues
        if coordination_metrics.sequential_alignment < 0.6:
            recommendations.append("Improve sequential alignment through better coordination rewards")
        
        if coordination_metrics.temporal_consistency < 0.5:
            recommendations.append("Enhance temporal consistency with stability regularization")
        
        if coordination_metrics.system_coherence < 0.5:
            recommendations.append("System coherence is low - consider curriculum learning adjustments")
        
        # Performance-based recommendations
        if len(self.evaluation_history) >= 3:
            recent_reports = self.evaluation_history[-3:]
            
            # Check for declining trends
            quality_trend = []
            for report in recent_reports:
                if report.agent_metrics:
                    avg_quality = np.mean([m.overall_quality for m in report.agent_metrics.values()])
                    quality_trend.append(avg_quality)
            
            if len(quality_trend) >= 2:
                if quality_trend[-1] < quality_trend[0] * 0.9:
                    recommendations.append("Quality trend is declining - consider reducing learning rate")
        
        return recommendations
    
    def _update_metrics_history(self, report: EvaluationReport):
        """Update metrics history"""
        # Agent metrics
        for agent_name, metrics in report.agent_metrics.items():
            self.metrics_history[f'{agent_name}_quality'].append(metrics.overall_quality)
            self.metrics_history[f'{agent_name}_entropy'].append(metrics.entropy_mean)
            self.metrics_history[f'{agent_name}_consistency'].append(metrics.consistency_score)
        
        # Coordination metrics
        self.metrics_history['coordination_alignment'].append(report.coordination_metrics.sequential_alignment)
        self.metrics_history['coordination_coherence'].append(report.coordination_metrics.system_coherence)
        self.metrics_history['coordination_efficiency'].append(report.coordination_metrics.coordination_efficiency)
    
    def visualize_evaluation(self, report: EvaluationReport, save_path: Optional[str] = None):
        """Create visualization of evaluation results"""
        import pandas as pd
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Superposition Training Evaluation', fontsize=16)
        
        # Agent quality heatmap
        if report.agent_metrics:
            quality_data = []
            for agent_name, metrics in report.agent_metrics.items():
                quality_data.append([
                    metrics.overall_quality,
                    metrics.entropy_mean / self.target_entropy,
                    metrics.consistency_score,
                    metrics.confidence_calibration,
                    metrics.decision_diversity
                ])
            
            quality_df = pd.DataFrame(
                quality_data,
                index=list(report.agent_metrics.keys()),
                columns=['Overall Quality', 'Entropy', 'Consistency', 'Confidence', 'Diversity']
            )
            
            sns.heatmap(quality_df, annot=True, fmt='.2f', cmap='viridis', ax=axes[0, 0])
            axes[0, 0].set_title('Agent Quality Metrics')
        
        # Coordination metrics
        coord_metrics = [
            report.coordination_metrics.sequential_alignment,
            report.coordination_metrics.temporal_consistency,
            report.coordination_metrics.information_flow,
            report.coordination_metrics.coordination_stability,
            report.coordination_metrics.system_coherence
        ]
        
        coord_labels = ['Sequential\nAlignment', 'Temporal\nConsistency', 'Information\nFlow', 
                       'Coordination\nStability', 'System\nCoherence']
        
        axes[0, 1].bar(coord_labels, coord_metrics)
        axes[0, 1].set_title('Coordination Metrics')
        axes[0, 1].set_ylim(0, 1)
        
        # Training progress
        if report.training_progress:
            for agent_name, progress in report.training_progress.items():
                if progress.get('entropy_compliance'):
                    axes[0, 2].plot(progress['entropy_compliance'], label=f'{agent_name} Entropy')
                if progress.get('consistency_score'):
                    axes[1, 0].plot(progress['consistency_score'], label=f'{agent_name} Consistency')
            
            axes[0, 2].set_title('Entropy Compliance Over Time')
            axes[0, 2].legend()
            axes[1, 0].set_title('Consistency Score Over Time')
            axes[1, 0].legend()
        
        # Performance correlations
        if report.performance_correlations:
            corr_names = list(report.performance_correlations.keys())
            corr_values = list(report.performance_correlations.values())
            
            axes[1, 1].bar(corr_names, corr_values)
            axes[1, 1].set_title('Performance Correlations')
            axes[1, 1].set_ylim(-1, 1)
        
        # Recommendations
        if report.recommendations:
            axes[1, 2].text(0.1, 0.9, 'Recommendations:', transform=axes[1, 2].transAxes, 
                           fontsize=12, fontweight='bold')
            
            for i, rec in enumerate(report.recommendations[:5]):  # Show top 5
                axes[1, 2].text(0.1, 0.8 - i*0.15, f"• {rec}", transform=axes[1, 2].transAxes, 
                               fontsize=10, wrap=True)
        
        axes[1, 2].set_title('Training Recommendations')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def export_detailed_report(self, filepath: str):
        """Export detailed evaluation report"""
        if not self.evaluation_history:
            logger.warning("No evaluation history to export")
            return
        
        # Create comprehensive report
        report_data = {
            'evaluation_summary': {
                'total_evaluations': len(self.evaluation_history),
                'agents_evaluated': self.agent_names,
                'superposition_dim': self.superposition_dim,
                'target_entropy': self.target_entropy,
                'evaluation_window': self.evaluation_window
            },
            'latest_evaluation': self.evaluation_history[-1].to_dict(),
            'historical_metrics': dict(self.metrics_history),
            'trends': self._calculate_trends(),
            'performance_summary': self._calculate_performance_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Detailed evaluation report exported to {filepath}")
    
    def _calculate_trends(self) -> Dict[str, float]:
        """Calculate trends in metrics"""
        trends = {}
        
        for metric_name, values in self.metrics_history.items():
            if len(values) >= 10:
                x = np.arange(len(values))
                slope, _, _, _, _ = stats.linregress(x, values)
                trends[metric_name] = slope
        
        return trends
    
    def _calculate_performance_summary(self) -> Dict[str, Any]:
        """Calculate performance summary"""
        if not self.evaluation_history:
            return {}
        
        latest_report = self.evaluation_history[-1]
        
        return {
            'overall_system_quality': np.mean([m.overall_quality for m in latest_report.agent_metrics.values()]) if latest_report.agent_metrics else 0.0,
            'system_coordination': latest_report.coordination_metrics.system_coherence,
            'training_stability': np.mean([m.stability_score for m in latest_report.agent_metrics.values()]) if latest_report.agent_metrics else 0.0,
            'recommendations_count': len(latest_report.recommendations)
        }
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of evaluation system"""
        return {
            'total_episodes_evaluated': len(self.episode_data),
            'total_evaluations_completed': len(self.evaluation_history),
            'agents_tracked': self.agent_names,
            'latest_performance': self._calculate_performance_summary(),
            'active_recommendations': len(self.evaluation_history[-1].recommendations) if self.evaluation_history else 0
        }