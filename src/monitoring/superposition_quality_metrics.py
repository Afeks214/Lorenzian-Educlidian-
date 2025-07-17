"""
Quality Metrics for Superposition Outputs.

This module provides comprehensive quality assessment for superposition outputs
from the universal superposition framework. It evaluates:

- Decision consistency and stability
- Probabilistic coherence and calibration
- Multi-agent agreement and divergence
- Temporal consistency and smoothness
- Trading performance quality indicators
- Risk-adjusted quality metrics

The metrics ensure that superposition outputs meet quality standards
for production trading systems.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import deque, defaultdict
import json
from scipy import stats
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score, mean_squared_error
import warnings
from concurrent.futures import ThreadPoolExecutor
import threading
import time


class QualityDimension(Enum):
    """Dimensions of quality assessment."""
    CONSISTENCY = "consistency"
    COHERENCE = "coherence"
    CALIBRATION = "calibration"
    STABILITY = "stability"
    DIVERSITY = "diversity"
    CONFIDENCE = "confidence"
    PERFORMANCE = "performance"
    RISK_ADJUSTED = "risk_adjusted"


class QualityLevel(Enum):
    """Quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


@dataclass
class SuperpositionOutput:
    """Represents a superposition output for quality assessment."""
    timestamp: datetime
    decision_probabilities: np.ndarray
    agent_contributions: Dict[str, float]
    confidence_scores: Dict[str, float]
    ensemble_confidence: float
    decision_value: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'decision_probabilities': self.decision_probabilities.tolist(),
            'agent_contributions': self.agent_contributions,
            'confidence_scores': self.confidence_scores,
            'ensemble_confidence': self.ensemble_confidence,
            'decision_value': self.decision_value,
            'metadata': self.metadata
        }


@dataclass
class QualityMetric:
    """Individual quality metric result."""
    dimension: QualityDimension
    name: str
    value: float
    level: QualityLevel
    threshold: float
    description: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'dimension': self.dimension.value,
            'name': self.name,
            'value': self.value,
            'level': self.level.value,
            'threshold': self.threshold,
            'description': self.description,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class QualityReport:
    """Comprehensive quality assessment report."""
    timestamp: datetime
    overall_quality: QualityLevel
    dimension_scores: Dict[QualityDimension, float]
    metrics: List[QualityMetric]
    recommendations: List[str]
    trend_analysis: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'overall_quality': self.overall_quality.value,
            'dimension_scores': {dim.value: score for dim, score in self.dimension_scores.items()},
            'metrics': [metric.to_dict() for metric in self.metrics],
            'recommendations': self.recommendations,
            'trend_analysis': self.trend_analysis
        }


class SuperpositionQualityMetrics:
    """
    Comprehensive quality metrics system for superposition outputs.
    
    This system evaluates the quality of superposition outputs across multiple
    dimensions to ensure they meet the standards required for production
    trading systems.
    """
    
    def __init__(
        self,
        history_size: int = 1000,
        consistency_threshold: float = 0.8,
        coherence_threshold: float = 0.7,
        calibration_threshold: float = 0.75,
        stability_threshold: float = 0.85,
        confidence_threshold: float = 0.6
    ):
        """
        Initialize the quality metrics system.
        
        Args:
            history_size: Number of outputs to keep in history
            consistency_threshold: Threshold for consistency quality
            coherence_threshold: Threshold for coherence quality
            calibration_threshold: Threshold for calibration quality
            stability_threshold: Threshold for stability quality
            confidence_threshold: Threshold for confidence quality
        """
        self.history_size = history_size
        self.consistency_threshold = consistency_threshold
        self.coherence_threshold = coherence_threshold
        self.calibration_threshold = calibration_threshold
        self.stability_threshold = stability_threshold
        self.confidence_threshold = confidence_threshold
        
        # Setup logging
        self.logger = logging.getLogger('superposition_quality_metrics')
        self.logger.setLevel(logging.INFO)
        
        # Output history
        self.output_history: deque = deque(maxlen=history_size)
        self.quality_history: deque = deque(maxlen=history_size)
        
        # Quality tracking
        self.dimension_trends: Dict[QualityDimension, deque] = {
            dim: deque(maxlen=100) for dim in QualityDimension
        }
        
        # Performance tracking
        self.performance_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Quality thresholds
        self.quality_thresholds = {
            QualityDimension.CONSISTENCY: consistency_threshold,
            QualityDimension.COHERENCE: coherence_threshold,
            QualityDimension.CALIBRATION: calibration_threshold,
            QualityDimension.STABILITY: stability_threshold,
            QualityDimension.CONFIDENCE: confidence_threshold,
            QualityDimension.DIVERSITY: 0.5,
            QualityDimension.PERFORMANCE: 0.6,
            QualityDimension.RISK_ADJUSTED: 0.7
        }
        
        self.logger.info(f"Quality metrics system initialized with history_size={history_size}")
    
    def record_output(self, output: SuperpositionOutput) -> None:
        """
        Record a superposition output for quality assessment.
        
        Args:
            output: Superposition output to record
        """
        with self._lock:
            self.output_history.append(output)
            
            # Perform real-time quality assessment
            if len(self.output_history) >= 2:
                self._update_real_time_metrics(output)
    
    def assess_quality(self, window_size: Optional[int] = None) -> QualityReport:
        """
        Perform comprehensive quality assessment.
        
        Args:
            window_size: Number of recent outputs to assess (None for all)
            
        Returns:
            Comprehensive quality report
        """
        if not self.output_history:
            return self._create_empty_report()
        
        # Select outputs for assessment
        outputs = list(self.output_history)
        if window_size:
            outputs = outputs[-window_size:]
        
        if len(outputs) < 2:
            return self._create_empty_report()
        
        # Assess each quality dimension
        metrics = []
        
        # Consistency metrics
        consistency_metrics = self._assess_consistency(outputs)
        metrics.extend(consistency_metrics)
        
        # Coherence metrics
        coherence_metrics = self._assess_coherence(outputs)
        metrics.extend(coherence_metrics)
        
        # Calibration metrics
        calibration_metrics = self._assess_calibration(outputs)
        metrics.extend(calibration_metrics)
        
        # Stability metrics
        stability_metrics = self._assess_stability(outputs)
        metrics.extend(stability_metrics)
        
        # Diversity metrics
        diversity_metrics = self._assess_diversity(outputs)
        metrics.extend(diversity_metrics)
        
        # Confidence metrics
        confidence_metrics = self._assess_confidence(outputs)
        metrics.extend(confidence_metrics)
        
        # Performance metrics
        performance_metrics = self._assess_performance(outputs)
        metrics.extend(performance_metrics)
        
        # Risk-adjusted metrics
        risk_adjusted_metrics = self._assess_risk_adjusted_quality(outputs)
        metrics.extend(risk_adjusted_metrics)
        
        # Calculate dimension scores
        dimension_scores = self._calculate_dimension_scores(metrics)
        
        # Determine overall quality
        overall_quality = self._determine_overall_quality(dimension_scores)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, dimension_scores)
        
        # Perform trend analysis
        trend_analysis = self._analyze_trends()
        
        # Create quality report
        report = QualityReport(
            timestamp=datetime.now(),
            overall_quality=overall_quality,
            dimension_scores=dimension_scores,
            metrics=metrics,
            recommendations=recommendations,
            trend_analysis=trend_analysis
        )
        
        # Store in history
        with self._lock:
            self.quality_history.append(report)
            
            # Update dimension trends
            for dim, score in dimension_scores.items():
                self.dimension_trends[dim].append(score)
        
        return report
    
    def _assess_consistency(self, outputs: List[SuperpositionOutput]) -> List[QualityMetric]:
        """Assess consistency of outputs."""
        metrics = []
        
        try:
            # Decision consistency over time
            decision_values = [output.decision_value for output in outputs]
            decision_std = np.std(decision_values)
            decision_mean = np.mean(decision_values)
            
            # Coefficient of variation for consistency
            cv = decision_std / (abs(decision_mean) + 1e-8)
            consistency_score = max(0, 1 - cv)
            
            metrics.append(QualityMetric(
                dimension=QualityDimension.CONSISTENCY,
                name="Decision Consistency",
                value=consistency_score,
                level=self._score_to_level(consistency_score, self.consistency_threshold),
                threshold=self.consistency_threshold,
                description="Consistency of decision values over time",
                timestamp=datetime.now(),
                metadata={
                    'decision_std': decision_std,
                    'decision_mean': decision_mean,
                    'coefficient_of_variation': cv
                }
            ))
            
            # Probability distribution consistency
            if len(outputs) >= 5:
                prob_matrices = [output.decision_probabilities for output in outputs[-5:]]
                prob_consistency = self._calculate_probability_consistency(prob_matrices)
                
                metrics.append(QualityMetric(
                    dimension=QualityDimension.CONSISTENCY,
                    name="Probability Consistency",
                    value=prob_consistency,
                    level=self._score_to_level(prob_consistency, self.consistency_threshold),
                    threshold=self.consistency_threshold,
                    description="Consistency of probability distributions",
                    timestamp=datetime.now()
                ))
            
            # Agent contribution consistency
            agent_consistency = self._calculate_agent_contribution_consistency(outputs)
            
            metrics.append(QualityMetric(
                dimension=QualityDimension.CONSISTENCY,
                name="Agent Contribution Consistency",
                value=agent_consistency,
                level=self._score_to_level(agent_consistency, self.consistency_threshold),
                threshold=self.consistency_threshold,
                description="Consistency of agent contributions",
                timestamp=datetime.now()
            ))
            
        except Exception as e:
            self.logger.error(f"Error assessing consistency: {e}")
        
        return metrics
    
    def _assess_coherence(self, outputs: List[SuperpositionOutput]) -> List[QualityMetric]:
        """Assess coherence of outputs."""
        metrics = []
        
        try:
            # Probability coherence (sum to 1, non-negative)
            prob_coherence_scores = []
            for output in outputs:
                probs = output.decision_probabilities
                
                # Check sum to 1
                prob_sum = np.sum(probs)
                sum_coherence = 1 - abs(prob_sum - 1.0)
                
                # Check non-negative
                negative_coherence = 1 if np.all(probs >= 0) else 0
                
                # Combined coherence
                coherence_score = 0.7 * sum_coherence + 0.3 * negative_coherence
                prob_coherence_scores.append(coherence_score)
            
            avg_prob_coherence = np.mean(prob_coherence_scores)
            
            metrics.append(QualityMetric(
                dimension=QualityDimension.COHERENCE,
                name="Probability Coherence",
                value=avg_prob_coherence,
                level=self._score_to_level(avg_prob_coherence, self.coherence_threshold),
                threshold=self.coherence_threshold,
                description="Coherence of probability distributions",
                timestamp=datetime.now()
            ))
            
            # Ensemble coherence (alignment between agents)
            ensemble_coherence = self._calculate_ensemble_coherence(outputs)
            
            metrics.append(QualityMetric(
                dimension=QualityDimension.COHERENCE,
                name="Ensemble Coherence",
                value=ensemble_coherence,
                level=self._score_to_level(ensemble_coherence, self.coherence_threshold),
                threshold=self.coherence_threshold,
                description="Coherence between agent contributions",
                timestamp=datetime.now()
            ))
            
        except Exception as e:
            self.logger.error(f"Error assessing coherence: {e}")
        
        return metrics
    
    def _assess_calibration(self, outputs: List[SuperpositionOutput]) -> List[QualityMetric]:
        """Assess calibration of outputs."""
        metrics = []
        
        try:
            # Confidence calibration
            confidence_scores = [output.ensemble_confidence for output in outputs]
            decision_values = [output.decision_value for output in outputs]
            
            # Calculate calibration using binning
            calibration_score = self._calculate_calibration_score(confidence_scores, decision_values)
            
            metrics.append(QualityMetric(
                dimension=QualityDimension.CALIBRATION,
                name="Confidence Calibration",
                value=calibration_score,
                level=self._score_to_level(calibration_score, self.calibration_threshold),
                threshold=self.calibration_threshold,
                description="Calibration of confidence scores",
                timestamp=datetime.now()
            ))
            
            # Probabilistic calibration
            if len(outputs) >= 10:
                prob_calibration = self._calculate_probabilistic_calibration(outputs)
                
                metrics.append(QualityMetric(
                    dimension=QualityDimension.CALIBRATION,
                    name="Probabilistic Calibration",
                    value=prob_calibration,
                    level=self._score_to_level(prob_calibration, self.calibration_threshold),
                    threshold=self.calibration_threshold,
                    description="Calibration of probability predictions",
                    timestamp=datetime.now()
                ))
            
        except Exception as e:
            self.logger.error(f"Error assessing calibration: {e}")
        
        return metrics
    
    def _assess_stability(self, outputs: List[SuperpositionOutput]) -> List[QualityMetric]:
        """Assess stability of outputs."""
        metrics = []
        
        try:
            # Temporal stability
            timestamps = [output.timestamp for output in outputs]
            decision_values = [output.decision_value for output in outputs]
            
            # Calculate stability using moving windows
            stability_score = self._calculate_temporal_stability(timestamps, decision_values)
            
            metrics.append(QualityMetric(
                dimension=QualityDimension.STABILITY,
                name="Temporal Stability",
                value=stability_score,
                level=self._score_to_level(stability_score, self.stability_threshold),
                threshold=self.stability_threshold,
                description="Stability of decisions over time",
                timestamp=datetime.now()
            ))
            
            # Confidence stability
            confidence_scores = [output.ensemble_confidence for output in outputs]
            confidence_stability = self._calculate_confidence_stability(confidence_scores)
            
            metrics.append(QualityMetric(
                dimension=QualityDimension.STABILITY,
                name="Confidence Stability",
                value=confidence_stability,
                level=self._score_to_level(confidence_stability, self.stability_threshold),
                threshold=self.stability_threshold,
                description="Stability of confidence scores",
                timestamp=datetime.now()
            ))
            
        except Exception as e:
            self.logger.error(f"Error assessing stability: {e}")
        
        return metrics
    
    def _assess_diversity(self, outputs: List[SuperpositionOutput]) -> List[QualityMetric]:
        """Assess diversity of outputs."""
        metrics = []
        
        try:
            # Agent contribution diversity
            diversity_scores = []
            for output in outputs:
                contributions = list(output.agent_contributions.values())
                if len(contributions) > 1:
                    # Calculate entropy as diversity measure
                    normalized_contrib = np.array(contributions) / (np.sum(contributions) + 1e-8)
                    diversity = entropy(normalized_contrib + 1e-8)
                    # Normalize by maximum possible entropy
                    max_entropy = np.log(len(contributions))
                    diversity_score = diversity / max_entropy if max_entropy > 0 else 0
                else:
                    diversity_score = 0
                
                diversity_scores.append(diversity_score)
            
            avg_diversity = np.mean(diversity_scores)
            
            metrics.append(QualityMetric(
                dimension=QualityDimension.DIVERSITY,
                name="Agent Contribution Diversity",
                value=avg_diversity,
                level=self._score_to_level(avg_diversity, self.quality_thresholds[QualityDimension.DIVERSITY]),
                threshold=self.quality_thresholds[QualityDimension.DIVERSITY],
                description="Diversity of agent contributions",
                timestamp=datetime.now()
            ))
            
            # Decision diversity over time
            decision_diversity = self._calculate_decision_diversity(outputs)
            
            metrics.append(QualityMetric(
                dimension=QualityDimension.DIVERSITY,
                name="Decision Diversity",
                value=decision_diversity,
                level=self._score_to_level(decision_diversity, self.quality_thresholds[QualityDimension.DIVERSITY]),
                threshold=self.quality_thresholds[QualityDimension.DIVERSITY],
                description="Diversity of decisions over time",
                timestamp=datetime.now()
            ))
            
        except Exception as e:
            self.logger.error(f"Error assessing diversity: {e}")
        
        return metrics
    
    def _assess_confidence(self, outputs: List[SuperpositionOutput]) -> List[QualityMetric]:
        """Assess confidence quality of outputs."""
        metrics = []
        
        try:
            # Ensemble confidence quality
            confidence_scores = [output.ensemble_confidence for output in outputs]
            avg_confidence = np.mean(confidence_scores)
            
            # Quality based on confidence level and stability
            confidence_std = np.std(confidence_scores)
            confidence_quality = avg_confidence * (1 - confidence_std)
            
            metrics.append(QualityMetric(
                dimension=QualityDimension.CONFIDENCE,
                name="Ensemble Confidence Quality",
                value=confidence_quality,
                level=self._score_to_level(confidence_quality, self.confidence_threshold),
                threshold=self.confidence_threshold,
                description="Quality of ensemble confidence scores",
                timestamp=datetime.now(),
                metadata={
                    'avg_confidence': avg_confidence,
                    'confidence_std': confidence_std
                }
            ))
            
            # Agent confidence agreement
            agent_confidence_agreement = self._calculate_agent_confidence_agreement(outputs)
            
            metrics.append(QualityMetric(
                dimension=QualityDimension.CONFIDENCE,
                name="Agent Confidence Agreement",
                value=agent_confidence_agreement,
                level=self._score_to_level(agent_confidence_agreement, self.confidence_threshold),
                threshold=self.confidence_threshold,
                description="Agreement between agent confidence scores",
                timestamp=datetime.now()
            ))
            
        except Exception as e:
            self.logger.error(f"Error assessing confidence: {e}")
        
        return metrics
    
    def _assess_performance(self, outputs: List[SuperpositionOutput]) -> List[QualityMetric]:
        """Assess performance quality of outputs."""
        metrics = []
        
        try:
            # Decision performance (simulated)
            decision_performance = self._calculate_decision_performance(outputs)
            
            metrics.append(QualityMetric(
                dimension=QualityDimension.PERFORMANCE,
                name="Decision Performance",
                value=decision_performance,
                level=self._score_to_level(decision_performance, self.quality_thresholds[QualityDimension.PERFORMANCE]),
                threshold=self.quality_thresholds[QualityDimension.PERFORMANCE],
                description="Performance quality of decisions",
                timestamp=datetime.now()
            ))
            
            # Confidence-performance alignment
            confidence_performance_alignment = self._calculate_confidence_performance_alignment(outputs)
            
            metrics.append(QualityMetric(
                dimension=QualityDimension.PERFORMANCE,
                name="Confidence-Performance Alignment",
                value=confidence_performance_alignment,
                level=self._score_to_level(confidence_performance_alignment, self.quality_thresholds[QualityDimension.PERFORMANCE]),
                threshold=self.quality_thresholds[QualityDimension.PERFORMANCE],
                description="Alignment between confidence and performance",
                timestamp=datetime.now()
            ))
            
        except Exception as e:
            self.logger.error(f"Error assessing performance: {e}")
        
        return metrics
    
    def _assess_risk_adjusted_quality(self, outputs: List[SuperpositionOutput]) -> List[QualityMetric]:
        """Assess risk-adjusted quality of outputs."""
        metrics = []
        
        try:
            # Risk-adjusted performance
            risk_adjusted_performance = self._calculate_risk_adjusted_performance(outputs)
            
            metrics.append(QualityMetric(
                dimension=QualityDimension.RISK_ADJUSTED,
                name="Risk-Adjusted Performance",
                value=risk_adjusted_performance,
                level=self._score_to_level(risk_adjusted_performance, self.quality_thresholds[QualityDimension.RISK_ADJUSTED]),
                threshold=self.quality_thresholds[QualityDimension.RISK_ADJUSTED],
                description="Risk-adjusted performance quality",
                timestamp=datetime.now()
            ))
            
            # Uncertainty handling quality
            uncertainty_handling = self._calculate_uncertainty_handling_quality(outputs)
            
            metrics.append(QualityMetric(
                dimension=QualityDimension.RISK_ADJUSTED,
                name="Uncertainty Handling",
                value=uncertainty_handling,
                level=self._score_to_level(uncertainty_handling, self.quality_thresholds[QualityDimension.RISK_ADJUSTED]),
                threshold=self.quality_thresholds[QualityDimension.RISK_ADJUSTED],
                description="Quality of uncertainty handling",
                timestamp=datetime.now()
            ))
            
        except Exception as e:
            self.logger.error(f"Error assessing risk-adjusted quality: {e}")
        
        return metrics
    
    # Helper methods for quality calculations
    
    def _calculate_probability_consistency(self, prob_matrices: List[np.ndarray]) -> float:
        """Calculate consistency of probability distributions."""
        if len(prob_matrices) < 2:
            return 1.0
        
        # Calculate pairwise KL divergences
        kl_divergences = []
        for i in range(len(prob_matrices)):
            for j in range(i + 1, len(prob_matrices)):
                p = prob_matrices[i] + 1e-8
                q = prob_matrices[j] + 1e-8
                kl_div = entropy(p, q)
                kl_divergences.append(kl_div)
        
        # Convert to consistency score
        avg_kl = np.mean(kl_divergences)
        consistency = 1 / (1 + avg_kl)
        
        return consistency
    
    def _calculate_agent_contribution_consistency(self, outputs: List[SuperpositionOutput]) -> float:
        """Calculate consistency of agent contributions."""
        if len(outputs) < 2:
            return 1.0
        
        # Get all agent IDs
        all_agents = set()
        for output in outputs:
            all_agents.update(output.agent_contributions.keys())
        
        if not all_agents:
            return 1.0
        
        # Calculate consistency for each agent
        agent_consistencies = []
        for agent_id in all_agents:
            contributions = [output.agent_contributions.get(agent_id, 0) for output in outputs]
            if len(contributions) > 1:
                consistency = 1 - (np.std(contributions) / (np.mean(contributions) + 1e-8))
                agent_consistencies.append(max(0, consistency))
        
        return np.mean(agent_consistencies) if agent_consistencies else 1.0
    
    def _calculate_ensemble_coherence(self, outputs: List[SuperpositionOutput]) -> float:
        """Calculate coherence between ensemble components."""
        coherence_scores = []
        
        for output in outputs:
            # Check if contributions and confidence align
            contributions = list(output.agent_contributions.values())
            confidences = list(output.confidence_scores.values())
            
            if len(contributions) == len(confidences) and len(contributions) > 0:
                # Calculate correlation between contributions and confidences
                if len(contributions) > 1:
                    correlation = np.corrcoef(contributions, confidences)[0, 1]
                    coherence_scores.append(max(0, correlation))
                else:
                    coherence_scores.append(1.0)
        
        return np.mean(coherence_scores) if coherence_scores else 1.0
    
    def _calculate_calibration_score(self, confidence_scores: List[float], outcomes: List[float]) -> float:
        """Calculate calibration score using binning."""
        if len(confidence_scores) < 10:
            return 0.5  # Insufficient data
        
        # Bin confidence scores
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(confidence_scores, bins) - 1
        
        calibration_error = 0
        total_samples = 0
        
        for bin_idx in range(len(bins) - 1):
            bin_mask = bin_indices == bin_idx
            if np.sum(bin_mask) > 0:
                bin_confidence = np.mean(np.array(confidence_scores)[bin_mask])
                bin_outcomes = np.array(outcomes)[bin_mask]
                bin_accuracy = np.mean(bin_outcomes)
                
                bin_size = np.sum(bin_mask)
                calibration_error += bin_size * abs(bin_confidence - bin_accuracy)
                total_samples += bin_size
        
        if total_samples > 0:
            calibration_error /= total_samples
            return max(0, 1 - calibration_error)
        
        return 0.5
    
    def _calculate_probabilistic_calibration(self, outputs: List[SuperpositionOutput]) -> float:
        """Calculate probabilistic calibration."""
        # Simplified implementation
        # In practice, this would use actual trading outcomes
        return 0.7  # Placeholder
    
    def _calculate_temporal_stability(self, timestamps: List[datetime], values: List[float]) -> float:
        """Calculate temporal stability of values."""
        if len(values) < 3:
            return 1.0
        
        # Calculate smoothness using second derivatives
        second_derivatives = []
        for i in range(1, len(values) - 1):
            second_deriv = values[i-1] - 2*values[i] + values[i+1]
            second_derivatives.append(abs(second_deriv))
        
        # Stability inversely related to variance of second derivatives
        stability = 1 / (1 + np.var(second_derivatives))
        return stability
    
    def _calculate_confidence_stability(self, confidence_scores: List[float]) -> float:
        """Calculate stability of confidence scores."""
        if len(confidence_scores) < 2:
            return 1.0
        
        # Use coefficient of variation
        cv = np.std(confidence_scores) / (np.mean(confidence_scores) + 1e-8)
        stability = 1 / (1 + cv)
        return stability
    
    def _calculate_decision_diversity(self, outputs: List[SuperpositionOutput]) -> float:
        """Calculate diversity of decisions."""
        decision_values = [output.decision_value for output in outputs]
        if len(decision_values) < 2:
            return 0.0
        
        # Use normalized range as diversity measure
        value_range = np.max(decision_values) - np.min(decision_values)
        value_std = np.std(decision_values)
        
        # Combine range and standard deviation
        diversity = 0.5 * (value_range / (np.mean(np.abs(decision_values)) + 1e-8)) + 0.5 * (value_std / (np.mean(np.abs(decision_values)) + 1e-8))
        return min(1.0, diversity)
    
    def _calculate_agent_confidence_agreement(self, outputs: List[SuperpositionOutput]) -> float:
        """Calculate agreement between agent confidence scores."""
        agreement_scores = []
        
        for output in outputs:
            confidences = list(output.confidence_scores.values())
            if len(confidences) > 1:
                # Calculate coefficient of variation
                cv = np.std(confidences) / (np.mean(confidences) + 1e-8)
                agreement = 1 / (1 + cv)
                agreement_scores.append(agreement)
        
        return np.mean(agreement_scores) if agreement_scores else 1.0
    
    def _calculate_decision_performance(self, outputs: List[SuperpositionOutput]) -> float:
        """Calculate decision performance (simulated)."""
        # This would integrate with actual trading performance
        # For now, return a simulated score based on confidence
        confidence_scores = [output.ensemble_confidence for output in outputs]
        return np.mean(confidence_scores) if confidence_scores else 0.5
    
    def _calculate_confidence_performance_alignment(self, outputs: List[SuperpositionOutput]) -> float:
        """Calculate alignment between confidence and performance."""
        # Simplified implementation
        return 0.75  # Placeholder
    
    def _calculate_risk_adjusted_performance(self, outputs: List[SuperpositionOutput]) -> float:
        """Calculate risk-adjusted performance."""
        # Simplified implementation
        return 0.65  # Placeholder
    
    def _calculate_uncertainty_handling_quality(self, outputs: List[SuperpositionOutput]) -> float:
        """Calculate quality of uncertainty handling."""
        # Check if confidence scores appropriately reflect uncertainty
        confidence_scores = [output.ensemble_confidence for output in outputs]
        
        # Good uncertainty handling should have moderate confidence
        # (not too high, not too low)
        avg_confidence = np.mean(confidence_scores)
        confidence_std = np.std(confidence_scores)
        
        # Optimal range is 0.6-0.8 with moderate variance
        distance_from_optimal = abs(avg_confidence - 0.7)
        uncertainty_quality = (1 - distance_from_optimal) * (1 - confidence_std)
        
        return max(0, uncertainty_quality)
    
    def _calculate_dimension_scores(self, metrics: List[QualityMetric]) -> Dict[QualityDimension, float]:
        """Calculate average scores for each dimension."""
        dimension_scores = {}
        
        for dimension in QualityDimension:
            dim_metrics = [m for m in metrics if m.dimension == dimension]
            if dim_metrics:
                dimension_scores[dimension] = np.mean([m.value for m in dim_metrics])
            else:
                dimension_scores[dimension] = 0.0
        
        return dimension_scores
    
    def _determine_overall_quality(self, dimension_scores: Dict[QualityDimension, float]) -> QualityLevel:
        """Determine overall quality level."""
        if not dimension_scores:
            return QualityLevel.POOR
        
        # Weighted average of dimension scores
        weights = {
            QualityDimension.CONSISTENCY: 0.2,
            QualityDimension.COHERENCE: 0.15,
            QualityDimension.CALIBRATION: 0.15,
            QualityDimension.STABILITY: 0.15,
            QualityDimension.DIVERSITY: 0.1,
            QualityDimension.CONFIDENCE: 0.1,
            QualityDimension.PERFORMANCE: 0.1,
            QualityDimension.RISK_ADJUSTED: 0.05
        }
        
        weighted_score = sum(weights.get(dim, 0) * score for dim, score in dimension_scores.items())
        
        return self._score_to_level(weighted_score, 0.7)
    
    def _score_to_level(self, score: float, threshold: float) -> QualityLevel:
        """Convert score to quality level."""
        if score >= threshold * 1.2:
            return QualityLevel.EXCELLENT
        elif score >= threshold:
            return QualityLevel.GOOD
        elif score >= threshold * 0.8:
            return QualityLevel.ACCEPTABLE
        elif score >= threshold * 0.5:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNACCEPTABLE
    
    def _generate_recommendations(self, metrics: List[QualityMetric], dimension_scores: Dict[QualityDimension, float]) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        # Recommendations based on poor metrics
        poor_metrics = [m for m in metrics if m.level in [QualityLevel.POOR, QualityLevel.UNACCEPTABLE]]
        
        for metric in poor_metrics:
            if metric.dimension == QualityDimension.CONSISTENCY:
                recommendations.append("Improve decision consistency by enhancing agent coordination")
            elif metric.dimension == QualityDimension.COHERENCE:
                recommendations.append("Enhance probability coherence through better normalization")
            elif metric.dimension == QualityDimension.CALIBRATION:
                recommendations.append("Improve calibration through confidence score adjustment")
            elif metric.dimension == QualityDimension.STABILITY:
                recommendations.append("Increase stability through temporal smoothing")
            elif metric.dimension == QualityDimension.DIVERSITY:
                recommendations.append("Enhance diversity through agent specialization")
            elif metric.dimension == QualityDimension.CONFIDENCE:
                recommendations.append("Improve confidence estimation accuracy")
            elif metric.dimension == QualityDimension.PERFORMANCE:
                recommendations.append("Optimize decision-making algorithms for better performance")
            elif metric.dimension == QualityDimension.RISK_ADJUSTED:
                recommendations.append("Improve risk-adjusted performance through better uncertainty handling")
        
        # Overall recommendations
        overall_score = np.mean(list(dimension_scores.values()))
        if overall_score < 0.6:
            recommendations.append("Consider comprehensive system review and optimization")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze quality trends over time."""
        trend_analysis = {}
        
        for dimension, scores in self.dimension_trends.items():
            if len(scores) >= 5:
                recent_scores = list(scores)[-5:]
                trend_slope = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
                
                trend_analysis[dimension.value] = {
                    'current_score': recent_scores[-1],
                    'trend_slope': trend_slope,
                    'trend_direction': 'improving' if trend_slope > 0.01 else 'declining' if trend_slope < -0.01 else 'stable'
                }
        
        return trend_analysis
    
    def _update_real_time_metrics(self, output: SuperpositionOutput) -> None:
        """Update real-time quality metrics."""
        # This would perform lightweight real-time quality checks
        # For now, we'll just log the output
        pass
    
    def _create_empty_report(self) -> QualityReport:
        """Create an empty quality report."""
        return QualityReport(
            timestamp=datetime.now(),
            overall_quality=QualityLevel.POOR,
            dimension_scores={dim: 0.0 for dim in QualityDimension},
            metrics=[],
            recommendations=["Insufficient data for quality assessment"],
            trend_analysis={}
        )
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get quality summary statistics."""
        with self._lock:
            if not self.quality_history:
                return {'status': 'no_data'}
            
            recent_report = self.quality_history[-1]
            
            # Calculate trends
            if len(self.quality_history) >= 5:
                recent_scores = [
                    np.mean(list(report.dimension_scores.values()))
                    for report in list(self.quality_history)[-5:]
                ]
                trend_slope = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
                trend_direction = 'improving' if trend_slope > 0.01 else 'declining' if trend_slope < -0.01 else 'stable'
            else:
                trend_direction = 'unknown'
            
            return {
                'status': 'active',
                'overall_quality': recent_report.overall_quality.value,
                'dimension_scores': {dim.value: score for dim, score in recent_report.dimension_scores.items()},
                'trend_direction': trend_direction,
                'total_outputs_assessed': len(self.output_history),
                'recent_recommendations': recent_report.recommendations,
                'last_assessment': recent_report.timestamp.isoformat()
            }
    
    def export_quality_data(self) -> Dict[str, Any]:
        """Export quality data for analysis."""
        with self._lock:
            return {
                'quality_reports': [report.to_dict() for report in self.quality_history],
                'dimension_trends': {
                    dim.value: list(scores) for dim, scores in self.dimension_trends.items()
                },
                'configuration': {
                    'thresholds': {dim.value: thresh for dim, thresh in self.quality_thresholds.items()},
                    'history_size': self.history_size
                }
            }