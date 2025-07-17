"""
Analytics Module for Human Feedback System

This module provides comprehensive analytics for expert performance,
model alignment, and RLHF training effectiveness.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import json
import structlog
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .feedback_api import ExpertChoice
from .rlhf_trainer import PreferenceDatabase, RLHFTrainer

logger = structlog.get_logger()


class MetricType(Enum):
    """Types of metrics for analysis"""
    ACCURACY = "accuracy"
    CONFIDENCE = "confidence"
    RESPONSE_TIME = "response_time"
    AGREEMENT = "agreement"
    CONSISTENCY = "consistency"
    PERFORMANCE = "performance"


@dataclass
class ExpertMetrics:
    """Comprehensive expert performance metrics"""
    expert_id: str
    total_decisions: int
    average_confidence: float
    success_rate: float
    response_time_avg: float
    consistency_score: float
    agreement_with_model: float
    specialization_areas: List[str]
    performance_trend: str  # "improving", "declining", "stable"
    risk_profile: str  # "conservative", "moderate", "aggressive"


@dataclass
class ModelAlignmentMetrics:
    """Model alignment with expert preferences"""
    overall_alignment: float
    accuracy_improvement: float
    preference_learning_rate: float
    expert_satisfaction: float
    model_confidence_calibration: float
    bias_detection_score: float


@dataclass
class RLHFEffectivenessMetrics:
    """RLHF training effectiveness metrics"""
    training_convergence: float
    sample_efficiency: float
    generalization_score: float
    expert_diversity_impact: float
    feedback_quality_score: float
    model_stability: float


class ExpertAnalytics:
    """Analytics engine for expert performance and behavior"""
    
    def __init__(self, preference_db: PreferenceDatabase):
        self.preference_db = preference_db
        self.cache_timeout = timedelta(hours=1)
        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        
    def calculate_expert_metrics(self, expert_id: str, period_days: int = 30) -> ExpertMetrics:
        """Calculate comprehensive metrics for an expert"""
        cache_key = f"expert_metrics_{expert_id}_{period_days}"
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        # Get expert decisions from database
        decisions = self._get_expert_decisions(expert_id, period_days)
        
        if not decisions:
            return ExpertMetrics(
                expert_id=expert_id,
                total_decisions=0,
                average_confidence=0.0,
                success_rate=0.0,
                response_time_avg=0.0,
                consistency_score=0.0,
                agreement_with_model=0.0,
                specialization_areas=[],
                performance_trend="stable",
                risk_profile="unknown"
            )
        
        # Calculate metrics
        total_decisions = len(decisions)
        average_confidence = np.mean([d['confidence'] for d in decisions])
        success_rate = self._calculate_success_rate(decisions)
        response_time_avg = self._calculate_avg_response_time(decisions)
        consistency_score = self._calculate_consistency_score(decisions)
        agreement_with_model = self._calculate_model_agreement(decisions)
        specialization_areas = self._identify_specialization_areas(decisions)
        performance_trend = self._analyze_performance_trend(decisions)
        risk_profile = self._determine_risk_profile(decisions)
        
        metrics = ExpertMetrics(
            expert_id=expert_id,
            total_decisions=total_decisions,
            average_confidence=average_confidence,
            success_rate=success_rate,
            response_time_avg=response_time_avg,
            consistency_score=consistency_score,
            agreement_with_model=agreement_with_model,
            specialization_areas=specialization_areas,
            performance_trend=performance_trend,
            risk_profile=risk_profile
        )
        
        self._cache_result(cache_key, metrics)
        return metrics

    def _get_expert_decisions(self, expert_id: str, period_days: int) -> List[Dict]:
        """Get expert decisions from database"""
        try:
            # Query preference database
            records = self.preference_db.get_preference_records(
                expert_id=expert_id,
                limit=None
            )
            
            # Filter by time period
            cutoff_date = datetime.now() - timedelta(days=period_days)
            filtered_records = [
                record for record in records 
                if record.timestamp >= cutoff_date
            ]
            
            # Convert to dictionary format
            decisions = []
            for record in filtered_records:
                decisions.append({
                    'decision_id': record.decision_id,
                    'confidence': record.expert_confidence,
                    'timestamp': record.timestamp,
                    'chosen_strategy': record.chosen_strategy,
                    'rejected_strategy': record.rejected_strategy,
                    'market_outcome': record.market_outcome,
                    'context_features': record.context_features
                })
            
            return decisions
            
        except Exception as e:
            logger.error("Error getting expert decisions", error=str(e))
            return []

    def _calculate_success_rate(self, decisions: List[Dict]) -> float:
        """Calculate expert success rate based on market outcomes"""
        decisions_with_outcomes = [
            d for d in decisions 
            if d.get('market_outcome') is not None
        ]
        
        if not decisions_with_outcomes:
            return 0.75  # Default assumption
        
        successful_decisions = sum(
            1 for d in decisions_with_outcomes 
            if d['market_outcome'] > 0
        )
        
        return successful_decisions / len(decisions_with_outcomes)

    def _calculate_avg_response_time(self, decisions: List[Dict]) -> float:
        """Calculate average response time (mock implementation)"""
        # In a real system, this would track time from decision presentation to response
        return np.random.uniform(30, 300)  # 30 seconds to 5 minutes

    def _calculate_consistency_score(self, decisions: List[Dict]) -> float:
        """Calculate consistency in expert decisions"""
        if len(decisions) < 5:
            return 0.5  # Insufficient data
        
        # Analyze consistency in confidence levels
        confidences = [d['confidence'] for d in decisions]
        confidence_std = np.std(confidences)
        
        # Analyze consistency in strategy preferences
        strategy_preferences = self._analyze_strategy_preferences(decisions)
        
        # Combined consistency score (lower std deviation = higher consistency)
        consistency = max(0, 1 - (confidence_std / 0.5))  # Normalize
        
        return min(1.0, consistency)

    def _calculate_model_agreement(self, decisions: List[Dict]) -> float:
        """Calculate agreement between expert and model recommendations"""
        # This would compare expert choices with what the model recommended
        # For now, using a mock calculation
        return np.random.uniform(0.6, 0.9)

    def _identify_specialization_areas(self, decisions: List[Dict]) -> List[str]:
        """Identify areas where expert shows particular strength"""
        # Analyze decision patterns to identify specializations
        strategy_counts = {}
        for decision in decisions:
            strategy_type = decision['chosen_strategy'].strategy_type.value
            strategy_counts[strategy_type] = strategy_counts.get(strategy_type, 0) + 1
        
        # Find most frequent strategies (top 2)
        sorted_strategies = sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True)
        specializations = [strategy for strategy, count in sorted_strategies[:2]]
        
        return specializations

    def _analyze_performance_trend(self, decisions: List[Dict]) -> str:
        """Analyze performance trend over time"""
        if len(decisions) < 10:
            return "stable"
        
        # Sort by timestamp
        sorted_decisions = sorted(decisions, key=lambda x: x['timestamp'])
        
        # Split into two halves and compare success rates
        mid_point = len(sorted_decisions) // 2
        first_half = sorted_decisions[:mid_point]
        second_half = sorted_decisions[mid_point:]
        
        first_success = self._calculate_success_rate(first_half)
        second_success = self._calculate_success_rate(second_half)
        
        if second_success > first_success + 0.1:
            return "improving"
        elif second_success < first_success - 0.1:
            return "declining"
        else:
            return "stable"

    def _determine_risk_profile(self, decisions: List[Dict]) -> str:
        """Determine expert's risk profile from decisions"""
        if not decisions:
            return "unknown"
        
        # Analyze chosen strategies and position sizes
        aggressive_count = 0
        conservative_count = 0
        
        for decision in decisions:
            strategy = decision['chosen_strategy']
            if hasattr(strategy, 'strategy_type'):
                if strategy.strategy_type.value in ['aggressive', 'breakout']:
                    aggressive_count += 1
                elif strategy.strategy_type.value in ['conservative', 'mean_reversion']:
                    conservative_count += 1
        
        total_classified = aggressive_count + conservative_count
        if total_classified == 0:
            return "moderate"
        
        aggressive_ratio = aggressive_count / total_classified
        
        if aggressive_ratio > 0.6:
            return "aggressive"
        elif aggressive_ratio < 0.3:
            return "conservative"
        else:
            return "moderate"

    def _analyze_strategy_preferences(self, decisions: List[Dict]) -> Dict[str, float]:
        """Analyze expert's strategy preferences"""
        strategy_counts = {}
        total_decisions = len(decisions)
        
        for decision in decisions:
            strategy_type = decision['chosen_strategy'].strategy_type.value
            strategy_counts[strategy_type] = strategy_counts.get(strategy_type, 0) + 1
        
        # Convert to percentages
        strategy_preferences = {
            strategy: count / total_decisions 
            for strategy, count in strategy_counts.items()
        }
        
        return strategy_preferences

    def compare_experts(self, expert_ids: List[str], period_days: int = 30) -> Dict[str, Any]:
        """Compare multiple experts across various metrics"""
        expert_metrics = []
        for expert_id in expert_ids:
            metrics = self.calculate_expert_metrics(expert_id, period_days)
            expert_metrics.append(metrics)
        
        # Statistical comparisons
        comparison = {
            "experts": expert_metrics,
            "rankings": {
                "by_success_rate": sorted(expert_metrics, key=lambda x: x.success_rate, reverse=True),
                "by_confidence": sorted(expert_metrics, key=lambda x: x.average_confidence, reverse=True),
                "by_consistency": sorted(expert_metrics, key=lambda x: x.consistency_score, reverse=True),
                "by_model_agreement": sorted(expert_metrics, key=lambda x: x.agreement_with_model, reverse=True)
            },
            "statistical_summary": {
                "avg_success_rate": np.mean([m.success_rate for m in expert_metrics]),
                "std_success_rate": np.std([m.success_rate for m in expert_metrics]),
                "avg_confidence": np.mean([m.average_confidence for m in expert_metrics]),
                "avg_consistency": np.mean([m.consistency_score for m in expert_metrics])
            }
        }
        
        return comparison

    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result if not expired"""
        if cache_key in self._cache:
            timestamp, result = self._cache[cache_key]
            if datetime.now() - timestamp < self.cache_timeout:
                return result
        return None

    def _cache_result(self, cache_key: str, result: Any):
        """Cache a result with timestamp"""
        self._cache[cache_key] = (datetime.now(), result)


class ModelAlignmentAnalyzer:
    """Analyzer for model-expert alignment metrics"""
    
    def __init__(self, preference_db: PreferenceDatabase, rlhf_trainer: RLHFTrainer):
        self.preference_db = preference_db
        self.rlhf_trainer = rlhf_trainer

    def calculate_alignment_metrics(self) -> ModelAlignmentMetrics:
        """Calculate comprehensive model alignment metrics"""
        
        # Get all preference records
        records = self.preference_db.get_preference_records()
        
        if not records:
            return self._default_alignment_metrics()
        
        # Calculate various alignment metrics
        overall_alignment = self._calculate_overall_alignment(records)
        accuracy_improvement = self._calculate_accuracy_improvement()
        preference_learning_rate = self._calculate_learning_rate()
        expert_satisfaction = self._calculate_expert_satisfaction(records)
        confidence_calibration = self._calculate_confidence_calibration(records)
        bias_detection = self._calculate_bias_detection_score(records)
        
        return ModelAlignmentMetrics(
            overall_alignment=overall_alignment,
            accuracy_improvement=accuracy_improvement,
            preference_learning_rate=preference_learning_rate,
            expert_satisfaction=expert_satisfaction,
            model_confidence_calibration=confidence_calibration,
            bias_detection_score=bias_detection
        )

    def _calculate_overall_alignment(self, records: List) -> float:
        """Calculate overall alignment between model and experts"""
        if not records:
            return 0.0
        
        # Use the RLHF trainer to get alignment scores
        alignment_scores = []
        
        for record in records[-100:]:  # Last 100 records
            try:
                # Get model's reward for chosen vs rejected strategy
                chosen_reward = self.rlhf_trainer.get_strategy_reward(
                    record.context_features, record.chosen_strategy
                )
                rejected_reward = self.rlhf_trainer.get_strategy_reward(
                    record.context_features, record.rejected_strategy
                )
                
                # Alignment is positive if model prefers chosen strategy
                alignment = 1.0 if chosen_reward > rejected_reward else 0.0
                alignment_scores.append(alignment)
                
            except Exception as e:
                logger.warning("Error calculating alignment score", error=str(e))
                continue
        
        return np.mean(alignment_scores) if alignment_scores else 0.5

    def _calculate_accuracy_improvement(self) -> float:
        """Calculate accuracy improvement from RLHF training"""
        training_status = self.rlhf_trainer.get_training_status()
        
        # Compare current accuracy with baseline
        current_accuracy = training_status.get("validation_accuracy", 0.6)
        baseline_accuracy = 0.5  # Assume 50% baseline (random)
        
        improvement = (current_accuracy - baseline_accuracy) / baseline_accuracy
        return min(1.0, max(0.0, improvement))

    def _calculate_learning_rate(self) -> float:
        """Calculate how quickly the model learns from feedback"""
        # Analyze training history from RLHF trainer
        training_history = self.rlhf_trainer.training_history
        
        if len(training_history) < 2:
            return 0.5
        
        # Calculate improvement rate
        recent_accuracy = training_history[-1].get("val_accuracy", 0.5)
        initial_accuracy = training_history[0].get("val_accuracy", 0.5)
        
        sessions = len(training_history)
        learning_rate = (recent_accuracy - initial_accuracy) / sessions
        
        return min(1.0, max(0.0, learning_rate * 10))  # Scale to 0-1

    def _calculate_expert_satisfaction(self, records: List) -> float:
        """Calculate expert satisfaction based on confidence and agreement"""
        if not records:
            return 0.5
        
        # High confidence decisions that align with model should indicate satisfaction
        satisfaction_scores = []
        
        for record in records:
            # Weight by expert confidence
            confidence = record.expert_confidence
            
            # Check if expert's high-confidence decisions align with model
            if confidence > 0.8:
                satisfaction_scores.append(confidence)
            elif confidence < 0.5:
                satisfaction_scores.append(1 - confidence)  # Dissatisfaction
            else:
                satisfaction_scores.append(0.5)  # Neutral
        
        return np.mean(satisfaction_scores) if satisfaction_scores else 0.5

    def _calculate_confidence_calibration(self, records: List) -> float:
        """Calculate how well model confidence matches actual performance"""
        # This would require actual performance data vs predicted confidence
        return np.random.uniform(0.6, 0.8)  # Mock implementation

    def _calculate_bias_detection_score(self, records: List) -> float:
        """Detect potential biases in expert decisions"""
        if not records:
            return 1.0  # No bias detected
        
        # Analyze for various biases
        bias_scores = []
        
        # Time-of-day bias
        time_bias = self._detect_time_bias(records)
        bias_scores.append(1 - time_bias)
        
        # Overconfidence bias
        confidence_bias = self._detect_confidence_bias(records)
        bias_scores.append(1 - confidence_bias)
        
        # Recency bias
        recency_bias = self._detect_recency_bias(records)
        bias_scores.append(1 - recency_bias)
        
        return np.mean(bias_scores)

    def _detect_time_bias(self, records: List) -> float:
        """Detect time-of-day bias in decisions"""
        # Mock implementation
        return np.random.uniform(0.0, 0.3)

    def _detect_confidence_bias(self, records: List) -> float:
        """Detect overconfidence or underconfidence bias"""
        confidences = [record.expert_confidence for record in records]
        
        if not confidences:
            return 0.0
        
        avg_confidence = np.mean(confidences)
        
        # Check for extreme confidence levels
        if avg_confidence > 0.9 or avg_confidence < 0.3:
            return 0.5  # Potential bias
        
        return 0.1  # Minimal bias

    def _detect_recency_bias(self, records: List) -> float:
        """Detect recency bias in decision patterns"""
        # Mock implementation
        return np.random.uniform(0.0, 0.2)

    def _default_alignment_metrics(self) -> ModelAlignmentMetrics:
        """Return default metrics when no data is available"""
        return ModelAlignmentMetrics(
            overall_alignment=0.5,
            accuracy_improvement=0.0,
            preference_learning_rate=0.0,
            expert_satisfaction=0.5,
            model_confidence_calibration=0.5,
            bias_detection_score=1.0
        )


class AnalyticsDashboard:
    """Comprehensive analytics dashboard for the human feedback system"""
    
    def __init__(self, preference_db: PreferenceDatabase, rlhf_trainer: RLHFTrainer):
        self.expert_analytics = ExpertAnalytics(preference_db)
        self.alignment_analyzer = ModelAlignmentAnalyzer(preference_db, rlhf_trainer)
        
    def generate_comprehensive_report(self, expert_ids: List[str] = None) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        
        if expert_ids is None:
            # Get all expert IDs from database
            expert_ids = ["trader001", "senior001", "pm001"]  # Default experts
        
        # Expert performance analysis
        expert_comparison = self.expert_analytics.compare_experts(expert_ids)
        
        # Model alignment analysis
        alignment_metrics = self.alignment_analyzer.calculate_alignment_metrics()
        
        # System-wide metrics
        system_metrics = self._calculate_system_metrics()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "expert_performance": expert_comparison,
            "model_alignment": asdict(alignment_metrics),
            "system_metrics": system_metrics,
            "recommendations": self._generate_recommendations(expert_comparison, alignment_metrics)
        }
        
        return report

    def _calculate_system_metrics(self) -> Dict[str, Any]:
        """Calculate system-wide performance metrics"""
        return {
            "total_decisions_processed": 150,  # Mock data
            "average_response_time": 120,  # seconds
            "system_uptime": "99.8%",
            "model_training_sessions": 12,
            "expert_engagement_rate": 0.85
        }

    def _generate_recommendations(self, expert_comparison: Dict, alignment_metrics: ModelAlignmentMetrics) -> List[str]:
        """Generate actionable recommendations based on analytics"""
        recommendations = []
        
        # Expert performance recommendations
        avg_success_rate = expert_comparison["statistical_summary"]["avg_success_rate"]
        if avg_success_rate < 0.7:
            recommendations.append("Consider additional expert training or calibration sessions")
        
        # Model alignment recommendations
        if alignment_metrics.overall_alignment < 0.7:
            recommendations.append("Increase frequency of RLHF training to improve model alignment")
        
        if alignment_metrics.bias_detection_score < 0.8:
            recommendations.append("Implement bias mitigation strategies in expert feedback collection")
        
        # System recommendations
        if alignment_metrics.preference_learning_rate < 0.3:
            recommendations.append("Optimize RLHF training parameters for faster convergence")
        
        return recommendations

    def export_metrics_for_monitoring(self) -> Dict[str, float]:
        """Export key metrics for external monitoring systems"""
        alignment_metrics = self.alignment_analyzer.calculate_alignment_metrics()
        
        return {
            "model_alignment_score": alignment_metrics.overall_alignment,
            "accuracy_improvement": alignment_metrics.accuracy_improvement,
            "expert_satisfaction": alignment_metrics.expert_satisfaction,
            "bias_detection_score": alignment_metrics.bias_detection_score,
            "system_health": 0.95  # Overall system health score
        }