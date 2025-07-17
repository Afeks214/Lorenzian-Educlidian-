"""
Multi-Agent Decision Fusion Engine - Intelligence Integration Layer

Advanced Bayesian inference system for fusing decisions from multiple agents 
with conflict resolution, uncertainty quantification, and meta-learning.

Features:
- Bayesian inference for multi-agent decision fusion
- Weighted voting with dynamic credibility scoring
- Conflict resolution algorithms for disagreeing agents
- Confidence propagation and uncertainty quantification
- Meta-learning for optimal fusion parameter tuning
- Real-time adaptation based on agent performance

Architecture:
- Bayesian Fusion Core: Statistical inference engine
- Credibility Scoring: Dynamic agent performance tracking
- Conflict Resolution: Algorithm for handling disagreements
- Uncertainty Estimation: Confidence interval calculation
- Meta-Learning: Continuous parameter optimization
"""

import asyncio
import threading
import time
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import structlog
from concurrent.futures import ThreadPoolExecutor
import queue
from collections import defaultdict, deque
import json
import math
from abc import ABC, abstractmethod
from scipy import stats
from scipy.optimize import minimize_scalar

logger = structlog.get_logger()


class DecisionType(Enum):
    """Types of decisions to fuse"""
    POSITION_SIZE = "position_size"
    RISK_LEVEL = "risk_level"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    PORTFOLIO_WEIGHT = "portfolio_weight"
    EMERGENCY_ACTION = "emergency_action"
    CRISIS_RESPONSE = "crisis_response"
    HUMAN_OVERRIDE = "human_override"


class FusionMethod(Enum):
    """Fusion methods for different decision types"""
    BAYESIAN_INFERENCE = "bayesian_inference"
    WEIGHTED_AVERAGE = "weighted_average"
    EXPERT_CONSENSUS = "expert_consensus"
    DEMPSTER_SHAFER = "dempster_shafer"
    MONTE_CARLO = "monte_carlo"
    KALMAN_FILTER = "kalman_filter"


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving agent conflicts"""
    HIGHEST_CONFIDENCE = "highest_confidence"
    WEIGHTED_MAJORITY = "weighted_majority"
    HIERARCHICAL_OVERRIDE = "hierarchical_override"
    STATISTICAL_CONSENSUS = "statistical_consensus"
    META_LEARNING_OPTIMIZED = "meta_learning_optimized"


@dataclass
class AgentDecision:
    """Individual agent decision with uncertainty"""
    agent_name: str
    decision_value: Union[float, int, np.ndarray]
    confidence: float
    uncertainty: float
    timestamp: datetime
    decision_type: DecisionType
    supporting_evidence: Dict[str, Any] = field(default_factory=dict)
    risk_factors: List[str] = field(default_factory=list)
    alternative_scenarios: List[Tuple[Any, float]] = field(default_factory=list)


@dataclass
class AgentCredibility:
    """Dynamic credibility scoring for agents"""
    agent_name: str
    base_credibility: float
    recent_performance: deque = field(default_factory=lambda: deque(maxlen=50))
    specialization_weights: Dict[DecisionType, float] = field(default_factory=dict)
    error_history: deque = field(default_factory=lambda: deque(maxlen=100))
    adaptation_rate: float = 0.1
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class FusionResult:
    """Result of decision fusion"""
    fused_decision: Union[float, int, np.ndarray]
    confidence_interval: Tuple[float, float]
    fusion_confidence: float
    participating_agents: List[str]
    fusion_method: FusionMethod
    conflict_detected: bool
    conflict_resolution: Optional[ConflictResolutionStrategy]
    uncertainty_estimate: float
    meta_learning_weight: float
    execution_time_ms: float
    supporting_rationale: str


@dataclass
class ConflictAnalysis:
    """Analysis of conflicts between agents"""
    conflict_level: float  # 0.0 to 1.0
    conflicting_agents: List[Tuple[str, str]]
    disagreement_magnitude: float
    confidence_variance: float
    resolution_strategy: ConflictResolutionStrategy
    resolution_confidence: float


class BayesianFusionCore:
    """Core Bayesian inference engine for decision fusion"""
    
    def __init__(self):
        self.prior_distributions: Dict[DecisionType, Tuple[float, float]] = {}
        self.likelihood_models: Dict[str, Callable] = {}
        self.posterior_cache: Dict[str, Tuple] = {}
        
    def set_prior_distribution(self, decision_type: DecisionType, mean: float, variance: float):
        """Set prior distribution for decision type"""
        self.prior_distributions[decision_type] = (mean, variance)
    
    def compute_bayesian_fusion(self, 
                              decisions: List[AgentDecision],
                              agent_credibilities: Dict[str, AgentCredibility]) -> Tuple[float, float, float]:
        """
        Compute Bayesian fusion of agent decisions
        
        Returns:
            (fused_value, confidence, uncertainty)
        """
        if not decisions:
            return 0.0, 0.0, 1.0
        
        decision_type = decisions[0].decision_type
        
        # Get prior distribution
        if decision_type in self.prior_distributions:
            prior_mean, prior_var = self.prior_distributions[decision_type]
        else:
            # Use empirical prior from decisions
            values = [d.decision_value for d in decisions if isinstance(d.decision_value, (int, float))]
            if values:
                prior_mean = np.mean(values)
                prior_var = np.var(values) if len(values) > 1 else 1.0
            else:
                prior_mean, prior_var = 0.0, 1.0
        
        # Bayesian updating with agent observations
        posterior_precision = 1.0 / prior_var  # Prior precision
        posterior_mean_num = prior_mean / prior_var  # Prior contribution
        
        for decision in decisions:
            if not isinstance(decision.decision_value, (int, float)):
                continue
                
            # Get agent credibility
            credibility = agent_credibilities.get(decision.agent_name)
            if credibility:
                agent_weight = credibility.base_credibility
                recent_perf = np.mean(credibility.recent_performance) if credibility.recent_performance else 0.5
                credibility_factor = agent_weight * (0.5 + recent_perf)
            else:
                credibility_factor = 0.5
            
            # Likelihood precision based on confidence and credibility
            likelihood_precision = credibility_factor * decision.confidence / max(0.01, decision.uncertainty)
            
            # Update posterior
            posterior_precision += likelihood_precision
            posterior_mean_num += decision.decision_value * likelihood_precision
        
        # Final posterior parameters
        posterior_variance = 1.0 / posterior_precision
        posterior_mean = posterior_mean_num / posterior_precision
        
        # Fusion confidence based on precision
        fusion_confidence = min(0.95, posterior_precision / (posterior_precision + 1.0))
        
        # Uncertainty estimate
        uncertainty = math.sqrt(posterior_variance)
        
        return posterior_mean, fusion_confidence, uncertainty
    
    def compute_confidence_interval(self, 
                                  mean: float, 
                                  variance: float, 
                                  confidence_level: float = 0.95) -> Tuple[float, float]:
        """Compute confidence interval for fused decision"""
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        margin = z_score * math.sqrt(variance)
        return (mean - margin, mean + margin)


class CredibilityScorer:
    """Dynamic credibility scoring system for agents"""
    
    def __init__(self):
        self.agent_credibilities: Dict[str, AgentCredibility] = {}
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.meta_learning_enabled = True
        
    def initialize_agent(self, 
                        agent_name: str, 
                        base_credibility: float = 0.5,
                        specializations: Optional[Dict[DecisionType, float]] = None):
        """Initialize credibility tracking for agent"""
        self.agent_credibilities[agent_name] = AgentCredibility(
            agent_name=agent_name,
            base_credibility=base_credibility,
            specialization_weights=specializations or {}
        )
    
    def update_performance(self, 
                         agent_name: str, 
                         decision_type: DecisionType,
                         actual_outcome: float,
                         predicted_outcome: float,
                         confidence: float):
        """Update agent performance based on actual outcomes"""
        if agent_name not in self.agent_credibilities:
            self.initialize_agent(agent_name)
        
        credibility = self.agent_credibilities[agent_name]
        
        # Calculate performance score
        error = abs(actual_outcome - predicted_outcome)
        max_error = max(abs(actual_outcome), abs(predicted_outcome), 1.0)
        performance_score = max(0.0, 1.0 - error / max_error)
        
        # Weight by confidence
        weighted_performance = performance_score * confidence + (1 - confidence) * 0.5
        
        # Update recent performance
        credibility.recent_performance.append(weighted_performance)
        
        # Update specialization weights
        if decision_type not in credibility.specialization_weights:
            credibility.specialization_weights[decision_type] = 0.5
        
        current_weight = credibility.specialization_weights[decision_type]
        credibility.specialization_weights[decision_type] = (
            current_weight * (1 - credibility.adaptation_rate) + 
            performance_score * credibility.adaptation_rate
        )
        
        # Update base credibility with exponential moving average
        if credibility.recent_performance:
            avg_recent_performance = np.mean(credibility.recent_performance)
            credibility.base_credibility = (
                credibility.base_credibility * 0.9 + 
                avg_recent_performance * 0.1
            )
        
        credibility.last_update = datetime.now()
        
        logger.debug("Agent credibility updated",
                    agent=agent_name,
                    performance_score=performance_score,
                    base_credibility=credibility.base_credibility)
    
    def get_agent_weight(self, 
                        agent_name: str, 
                        decision_type: DecisionType,
                        confidence: float) -> float:
        """Get dynamic weight for agent decision"""
        if agent_name not in self.agent_credibilities:
            return 0.5  # Default weight
        
        credibility = self.agent_credibilities[agent_name]
        
        # Base credibility
        weight = credibility.base_credibility
        
        # Specialization bonus
        if decision_type in credibility.specialization_weights:
            specialization_bonus = credibility.specialization_weights[decision_type] - 0.5
            weight += specialization_bonus * 0.3
        
        # Recent performance factor
        if credibility.recent_performance:
            recent_avg = np.mean(list(credibility.recent_performance)[-10:])  # Last 10 decisions
            weight = weight * 0.7 + recent_avg * 0.3
        
        # Confidence scaling
        weight *= (0.5 + confidence * 0.5)
        
        return np.clip(weight, 0.1, 2.0)


class ConflictResolver:
    """Conflict resolution system for disagreeing agents"""
    
    def __init__(self):
        self.resolution_strategies: Dict[ConflictResolutionStrategy, Callable] = {
            ConflictResolutionStrategy.HIGHEST_CONFIDENCE: self._resolve_highest_confidence,
            ConflictResolutionStrategy.WEIGHTED_MAJORITY: self._resolve_weighted_majority,
            ConflictResolutionStrategy.HIERARCHICAL_OVERRIDE: self._resolve_hierarchical,
            ConflictResolutionStrategy.STATISTICAL_CONSENSUS: self._resolve_statistical,
            ConflictResolutionStrategy.META_LEARNING_OPTIMIZED: self._resolve_meta_learning
        }
    
    def analyze_conflicts(self, decisions: List[AgentDecision]) -> ConflictAnalysis:
        """Analyze conflicts between agent decisions"""
        if len(decisions) < 2:
            return ConflictAnalysis(
                conflict_level=0.0,
                conflicting_agents=[],
                disagreement_magnitude=0.0,
                confidence_variance=0.0,
                resolution_strategy=ConflictResolutionStrategy.HIGHEST_CONFIDENCE,
                resolution_confidence=1.0
            )
        
        # Extract numerical values for analysis
        values = []
        confidences = []
        agent_names = []
        
        for decision in decisions:
            if isinstance(decision.decision_value, (int, float)):
                values.append(decision.decision_value)
                confidences.append(decision.confidence)
                agent_names.append(decision.agent_name)
        
        if len(values) < 2:
            return ConflictAnalysis(
                conflict_level=0.0,
                conflicting_agents=[],
                disagreement_magnitude=0.0,
                confidence_variance=0.0,
                resolution_strategy=ConflictResolutionStrategy.HIGHEST_CONFIDENCE,
                resolution_confidence=1.0
            )
        
        # Calculate disagreement metrics
        value_std = np.std(values)
        value_range = max(values) - min(values)
        mean_value = np.mean(values)
        
        # Normalized disagreement magnitude
        disagreement_magnitude = value_range / (abs(mean_value) + 1.0)
        
        # Confidence variance
        confidence_variance = np.var(confidences)
        
        # Conflict level (0.0 to 1.0)
        conflict_level = min(1.0, disagreement_magnitude + confidence_variance)
        
        # Identify conflicting agent pairs
        conflicting_pairs = []
        threshold = disagreement_magnitude * 0.5
        
        for i, (val1, agent1) in enumerate(zip(values, agent_names)):
            for j, (val2, agent2) in enumerate(zip(values, agent_names)):
                if i < j and abs(val1 - val2) > threshold:
                    conflicting_pairs.append((agent1, agent2))
        
        # Select resolution strategy based on conflict characteristics
        if conflict_level > 0.7:
            strategy = ConflictResolutionStrategy.META_LEARNING_OPTIMIZED
        elif confidence_variance > 0.3:
            strategy = ConflictResolutionStrategy.HIGHEST_CONFIDENCE
        elif len(conflicting_pairs) > 2:
            strategy = ConflictResolutionStrategy.STATISTICAL_CONSENSUS
        else:
            strategy = ConflictResolutionStrategy.WEIGHTED_MAJORITY
        
        return ConflictAnalysis(
            conflict_level=conflict_level,
            conflicting_agents=conflicting_pairs,
            disagreement_magnitude=disagreement_magnitude,
            confidence_variance=confidence_variance,
            resolution_strategy=strategy,
            resolution_confidence=1.0 - conflict_level
        )
    
    def resolve_conflict(self, 
                        decisions: List[AgentDecision],
                        conflict_analysis: ConflictAnalysis,
                        agent_weights: Dict[str, float]) -> Tuple[float, float]:
        """Resolve conflict using selected strategy"""
        
        strategy = conflict_analysis.resolution_strategy
        resolver = self.resolution_strategies.get(strategy, self._resolve_weighted_majority)
        
        return resolver(decisions, agent_weights, conflict_analysis)
    
    def _resolve_highest_confidence(self, 
                                  decisions: List[AgentDecision],
                                  agent_weights: Dict[str, float],
                                  conflict: ConflictAnalysis) -> Tuple[float, float]:
        """Resolve by selecting highest confidence decision"""
        if not decisions:
            return 0.0, 0.0
        
        best_decision = max(decisions, key=lambda d: d.confidence)
        return float(best_decision.decision_value), best_decision.confidence
    
    def _resolve_weighted_majority(self, 
                                 decisions: List[AgentDecision],
                                 agent_weights: Dict[str, float],
                                 conflict: ConflictAnalysis) -> Tuple[float, float]:
        """Resolve using weighted average"""
        if not decisions:
            return 0.0, 0.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        confidence_sum = 0.0
        
        for decision in decisions:
            if isinstance(decision.decision_value, (int, float)):
                weight = agent_weights.get(decision.agent_name, 1.0) * decision.confidence
                weighted_sum += decision.decision_value * weight
                total_weight += weight
                confidence_sum += decision.confidence
        
        if total_weight > 0:
            result = weighted_sum / total_weight
            avg_confidence = confidence_sum / len(decisions)
        else:
            result = decisions[0].decision_value
            avg_confidence = decisions[0].confidence
        
        return float(result), avg_confidence
    
    def _resolve_hierarchical(self, 
                            decisions: List[AgentDecision],
                            agent_weights: Dict[str, float],
                            conflict: ConflictAnalysis) -> Tuple[float, float]:
        """Resolve using hierarchical override"""
        if not decisions:
            return 0.0, 0.0
        
        # Sort by agent weight (highest first)
        sorted_decisions = sorted(decisions, 
                                key=lambda d: agent_weights.get(d.agent_name, 0.0), 
                                reverse=True)
        
        best_decision = sorted_decisions[0]
        return float(best_decision.decision_value), best_decision.confidence
    
    def _resolve_statistical(self, 
                           decisions: List[AgentDecision],
                           agent_weights: Dict[str, float],
                           conflict: ConflictAnalysis) -> Tuple[float, float]:
        """Resolve using statistical consensus (median)"""
        if not decisions:
            return 0.0, 0.0
        
        values = [d.decision_value for d in decisions if isinstance(d.decision_value, (int, float))]
        confidences = [d.confidence for d in decisions if isinstance(d.decision_value, (int, float))]
        
        if values:
            result = float(np.median(values))
            avg_confidence = np.mean(confidences)
        else:
            result = float(decisions[0].decision_value)
            avg_confidence = decisions[0].confidence
        
        return result, avg_confidence
    
    def _resolve_meta_learning(self, 
                             decisions: List[AgentDecision],
                             agent_weights: Dict[str, float],
                             conflict: ConflictAnalysis) -> Tuple[float, float]:
        """Resolve using meta-learning optimized weights"""
        # For now, fall back to weighted majority with optimized weights
        return self._resolve_weighted_majority(decisions, agent_weights, conflict)


class DecisionFusionEngine:
    """
    Multi-Agent Decision Fusion Engine
    
    Advanced system for fusing decisions from multiple agents using Bayesian inference,
    conflict resolution, and meta-learning optimization.
    """
    
    def __init__(self):
        self.bayesian_core = BayesianFusionCore()
        self.credibility_scorer = CredibilityScorer()
        self.conflict_resolver = ConflictResolver()
        
        # Fusion configuration
        self.fusion_methods: Dict[DecisionType, FusionMethod] = {
            DecisionType.POSITION_SIZE: FusionMethod.BAYESIAN_INFERENCE,
            DecisionType.RISK_LEVEL: FusionMethod.WEIGHTED_AVERAGE,
            DecisionType.EMERGENCY_ACTION: FusionMethod.EXPERT_CONSENSUS,
            DecisionType.CRISIS_RESPONSE: FusionMethod.EXPERT_CONSENSUS,
            DecisionType.HUMAN_OVERRIDE: FusionMethod.EXPERT_CONSENSUS
        }
        
        # Performance tracking
        self.fusion_count = 0
        self.fusion_times: deque = deque(maxlen=1000)
        self.confidence_history: deque = deque(maxlen=1000)
        self.conflict_rate_history: deque = deque(maxlen=100)
        
        # Meta-learning state
        self.learning_enabled = True
        self.optimization_history: deque = deque(maxlen=50)
        
        logger.info("Decision fusion engine initialized")
    
    def register_agent(self, 
                      agent_name: str, 
                      base_credibility: float = 0.5,
                      specializations: Optional[Dict[DecisionType, float]] = None):
        """Register agent for credibility tracking"""
        self.credibility_scorer.initialize_agent(agent_name, base_credibility, specializations)
        logger.debug("Agent registered for fusion", agent=agent_name, credibility=base_credibility)
    
    def fuse_decisions(self, 
                      decisions: List[AgentDecision],
                      decision_type: DecisionType,
                      context: Optional[Dict[str, Any]] = None) -> FusionResult:
        """
        Fuse multiple agent decisions into single decision
        
        Args:
            decisions: List of agent decisions to fuse
            decision_type: Type of decision being fused
            context: Additional context for fusion
            
        Returns:
            Fusion result with confidence metrics
        """
        start_time = datetime.now()
        
        try:
            if not decisions:
                return self._create_empty_result(decision_type)
            
            # Single decision case
            if len(decisions) == 1:
                return self._create_single_decision_result(decisions[0], start_time)
            
            # Get agent weights
            agent_weights = {}
            for decision in decisions:
                agent_weights[decision.agent_name] = self.credibility_scorer.get_agent_weight(
                    decision.agent_name, decision_type, decision.confidence
                )
            
            # Analyze conflicts
            conflict_analysis = self.conflict_resolver.analyze_conflicts(decisions)
            
            # Select fusion method
            fusion_method = self.fusion_methods.get(decision_type, FusionMethod.BAYESIAN_INFERENCE)
            
            # Perform fusion based on method
            if fusion_method == FusionMethod.BAYESIAN_INFERENCE:
                fused_value, fusion_confidence, uncertainty = self.bayesian_core.compute_bayesian_fusion(
                    decisions, self.credibility_scorer.agent_credibilities
                )
            elif conflict_analysis.conflict_level > 0.5:
                # Use conflict resolution for high conflict situations
                fused_value, fusion_confidence = self.conflict_resolver.resolve_conflict(
                    decisions, conflict_analysis, agent_weights
                )
                uncertainty = conflict_analysis.disagreement_magnitude
            else:
                # Use weighted average for low conflict
                fused_value, fusion_confidence = self._compute_weighted_average(decisions, agent_weights)
                uncertainty = np.std([d.uncertainty for d in decisions])
            
            # Compute confidence interval
            confidence_interval = self.bayesian_core.compute_confidence_interval(
                fused_value, uncertainty**2
            )
            
            # Calculate meta-learning weight
            meta_weight = self._compute_meta_learning_weight(decisions, decision_type)
            
            # Execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create result
            result = FusionResult(
                fused_decision=fused_value,
                confidence_interval=confidence_interval,
                fusion_confidence=fusion_confidence,
                participating_agents=[d.agent_name for d in decisions],
                fusion_method=fusion_method,
                conflict_detected=conflict_analysis.conflict_level > 0.3,
                conflict_resolution=conflict_analysis.resolution_strategy if conflict_analysis.conflict_level > 0.3 else None,
                uncertainty_estimate=uncertainty,
                meta_learning_weight=meta_weight,
                execution_time_ms=execution_time,
                supporting_rationale=self._generate_rationale(decisions, conflict_analysis, fusion_method)
            )
            
            # Update tracking
            self.fusion_count += 1
            self.fusion_times.append(execution_time)
            self.confidence_history.append(fusion_confidence)
            self.conflict_rate_history.append(conflict_analysis.conflict_level)
            
            return result
            
        except Exception as e:
            logger.error("Error in decision fusion", error=str(e))
            return self._create_error_result(str(e), start_time)
    
    def update_agent_performance(self, 
                               agent_name: str,
                               decision_type: DecisionType,
                               predicted_value: float,
                               actual_value: float,
                               confidence: float):
        """Update agent performance based on actual outcomes"""
        self.credibility_scorer.update_performance(
            agent_name, decision_type, actual_value, predicted_value, confidence
        )
    
    def _compute_weighted_average(self, 
                                decisions: List[AgentDecision],
                                agent_weights: Dict[str, float]) -> Tuple[float, float]:
        """Compute weighted average of decisions"""
        total_weight = 0.0
        weighted_sum = 0.0
        confidence_sum = 0.0
        
        for decision in decisions:
            if isinstance(decision.decision_value, (int, float)):
                weight = agent_weights.get(decision.agent_name, 1.0) * decision.confidence
                weighted_sum += decision.decision_value * weight
                total_weight += weight
                confidence_sum += decision.confidence
        
        if total_weight > 0:
            avg_value = weighted_sum / total_weight
            avg_confidence = confidence_sum / len(decisions)
        else:
            avg_value = decisions[0].decision_value
            avg_confidence = decisions[0].confidence
        
        return float(avg_value), avg_confidence
    
    def _compute_meta_learning_weight(self, 
                                    decisions: List[AgentDecision],
                                    decision_type: DecisionType) -> float:
        """Compute meta-learning weight for fusion"""
        if not self.learning_enabled:
            return 1.0
        
        # Base weight on historical performance
        base_weight = 1.0
        
        # Adjust based on agent diversity
        agent_count = len(decisions)
        diversity_bonus = min(0.5, agent_count * 0.1)
        
        # Adjust based on confidence levels
        avg_confidence = np.mean([d.confidence for d in decisions])
        confidence_factor = 0.5 + avg_confidence * 0.5
        
        return base_weight + diversity_bonus * confidence_factor
    
    def _generate_rationale(self, 
                          decisions: List[AgentDecision],
                          conflict_analysis: ConflictAnalysis,
                          fusion_method: FusionMethod) -> str:
        """Generate human-readable rationale for fusion"""
        agent_names = [d.agent_name for d in decisions]
        avg_confidence = np.mean([d.confidence for d in decisions])
        
        rationale = f"Fused {len(decisions)} decisions from agents: {', '.join(agent_names)} "
        rationale += f"using {fusion_method.value} method. "
        rationale += f"Average confidence: {avg_confidence:.2f}. "
        
        if conflict_analysis.conflict_level > 0.3:
            rationale += f"Conflict detected (level: {conflict_analysis.conflict_level:.2f}), "
            rationale += f"resolved using {conflict_analysis.resolution_strategy.value}."
        else:
            rationale += "No significant conflicts detected."
        
        return rationale
    
    def _create_empty_result(self, decision_type: DecisionType) -> FusionResult:
        """Create result for empty decision list"""
        return FusionResult(
            fused_decision=0.0,
            confidence_interval=(0.0, 0.0),
            fusion_confidence=0.0,
            participating_agents=[],
            fusion_method=FusionMethod.WEIGHTED_AVERAGE,
            conflict_detected=False,
            conflict_resolution=None,
            uncertainty_estimate=1.0,
            meta_learning_weight=0.0,
            execution_time_ms=0.0,
            supporting_rationale="No decisions to fuse"
        )
    
    def _create_single_decision_result(self, decision: AgentDecision, start_time: datetime) -> FusionResult:
        """Create result for single decision"""
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return FusionResult(
            fused_decision=decision.decision_value,
            confidence_interval=(decision.decision_value - decision.uncertainty, 
                               decision.decision_value + decision.uncertainty),
            fusion_confidence=decision.confidence,
            participating_agents=[decision.agent_name],
            fusion_method=FusionMethod.WEIGHTED_AVERAGE,
            conflict_detected=False,
            conflict_resolution=None,
            uncertainty_estimate=decision.uncertainty,
            meta_learning_weight=1.0,
            execution_time_ms=execution_time,
            supporting_rationale=f"Single decision from {decision.agent_name}"
        )
    
    def _create_error_result(self, error_msg: str, start_time: datetime) -> FusionResult:
        """Create error result"""
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return FusionResult(
            fused_decision=0.0,
            confidence_interval=(0.0, 0.0),
            fusion_confidence=0.0,
            participating_agents=[],
            fusion_method=FusionMethod.WEIGHTED_AVERAGE,
            conflict_detected=True,
            conflict_resolution=None,
            uncertainty_estimate=1.0,
            meta_learning_weight=0.0,
            execution_time_ms=execution_time,
            supporting_rationale=f"Fusion error: {error_msg}"
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        avg_fusion_time = np.mean(self.fusion_times) if self.fusion_times else 0.0
        avg_confidence = np.mean(self.confidence_history) if self.confidence_history else 0.0
        avg_conflict_rate = np.mean(self.conflict_rate_history) if self.conflict_rate_history else 0.0
        
        agent_metrics = {}
        for agent_name, credibility in self.credibility_scorer.agent_credibilities.items():
            agent_metrics[agent_name] = {
                'base_credibility': credibility.base_credibility,
                'recent_performance_avg': np.mean(credibility.recent_performance) if credibility.recent_performance else 0.0,
                'specializations': dict(credibility.specialization_weights),
                'error_count': len(credibility.error_history)
            }
        
        return {
            'fusion_count': self.fusion_count,
            'avg_fusion_time_ms': avg_fusion_time,
            'avg_confidence': avg_confidence,
            'avg_conflict_rate': avg_conflict_rate,
            'agent_count': len(self.credibility_scorer.agent_credibilities),
            'agent_metrics': agent_metrics,
            'learning_enabled': self.learning_enabled,
            'fusion_methods': {dt.value: fm.value for dt, fm in self.fusion_methods.items()}
        }
    
    def optimize_fusion_parameters(self):
        """Optimize fusion parameters using meta-learning"""
        if not self.learning_enabled or len(self.optimization_history) < 10:
            return
        
        # Placeholder for meta-learning optimization
        # This would analyze historical performance and adjust fusion parameters
        logger.debug("Fusion parameter optimization completed")
    
    def reset_learning_state(self):
        """Reset learning state for fresh start"""
        for credibility in self.credibility_scorer.agent_credibilities.values():
            credibility.recent_performance.clear()
            credibility.error_history.clear()
            credibility.base_credibility = 0.5
        
        self.optimization_history.clear()
        logger.info("Fusion learning state reset")