#!/usr/bin/env python3
"""
🚨 AGENT GAMMA MISSION - ATTACK EFFECTIVENESS ANALYZER
Advanced MARL Attack Development: Effectiveness Measurement and Success Rate Calculation

This module provides comprehensive attack effectiveness measurement and success rate
calculation systems for evaluating MARL attack performance:
- Multi-dimensional effectiveness metrics
- Success rate calculation across different conditions
- Attack performance benchmarking
- Effectiveness trend analysis
- Defensive countermeasure effectiveness assessment

Key Components:
1. Attack Effectiveness Metrics
2. Success Rate Calculation Engine
3. Performance Benchmarking System
4. Trend Analysis Tools
5. Defensive Assessment Framework

MISSION OBJECTIVE: Achieve >80% measurement accuracy and provide actionable insights
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import time
import json
from collections import defaultdict, deque
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Import attack modules for analysis
from . import (
    UnifiedAttackOrchestrator,
    UnifiedAttackResult,
    AttackModule,
    CoordinationAttackResult,
    TemporalAttackResult,
    PolicyGradientAttackResult,
    RegimeAttackResult,
    ScenarioAttackResult
)

@dataclass
class EffectivenessMetrics:
    """Comprehensive effectiveness metrics for attacks."""
    # Core metrics
    success_rate: float
    avg_disruption_score: float
    avg_execution_time_ms: float
    
    # Advanced metrics
    consistency_score: float
    reliability_score: float
    stealth_score: float
    adaptability_score: float
    
    # Performance metrics
    throughput: float  # attacks per second
    resource_efficiency: float
    scalability_score: float
    
    # Defensive metrics
    defense_bypass_rate: float
    false_positive_rate: float
    detection_evasion_rate: float
    
    # Metadata
    sample_size: int
    measurement_period: str
    confidence_interval: Tuple[float, float]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class SuccessRateAnalysis:
    """Success rate analysis across different conditions."""
    overall_success_rate: float
    success_by_attack_type: Dict[str, float]
    success_by_target_condition: Dict[str, float]
    success_by_defense_level: Dict[str, float]
    success_trend: List[float]
    success_confidence_interval: Tuple[float, float]
    statistical_significance: float
    
class EffectivenessCategory(Enum):
    """Categories for effectiveness measurement."""
    DISRUPTION = "disruption"
    STEALTH = "stealth"
    PERSISTENCE = "persistence"
    ADAPTABILITY = "adaptability"
    EFFICIENCY = "efficiency"

class AttackEffectivenessAnalyzer:
    """
    Advanced Attack Effectiveness Analyzer.
    
    This system provides comprehensive effectiveness measurement and success rate
    calculation for MARL attacks with statistical analysis and trend tracking.
    """
    
    def __init__(self, orchestrator: UnifiedAttackOrchestrator = None):
        """
        Initialize the Attack Effectiveness Analyzer.
        
        Args:
            orchestrator: UnifiedAttackOrchestrator instance for analysis
        """
        self.orchestrator = orchestrator or UnifiedAttackOrchestrator()
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Analysis history
        self.effectiveness_history = []
        self.success_rate_history = []
        self.benchmark_results = []
        
        # Metrics configuration
        self.metrics_config = {
            'disruption_weight': 0.3,
            'stealth_weight': 0.2,
            'persistence_weight': 0.2,
            'adaptability_weight': 0.15,
            'efficiency_weight': 0.15,
            'min_sample_size': 10,
            'confidence_level': 0.95
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            'excellent': 0.8,
            'good': 0.6,
            'fair': 0.4,
            'poor': 0.2
        }
        
        # Trend analysis parameters
        self.trend_window = 50
        self.trend_significance_threshold = 0.05
        
        self.logger.info("AttackEffectivenessAnalyzer initialized")
    
    def calculate_effectiveness_metrics(
        self,
        attack_results: List[UnifiedAttackResult],
        target_conditions: Dict[str, Any] = None
    ) -> EffectivenessMetrics:
        """
        Calculate comprehensive effectiveness metrics for a set of attack results.
        
        Args:
            attack_results: List of attack results to analyze
            target_conditions: Target system conditions during attacks
            
        Returns:
            EffectivenessMetrics with comprehensive analysis
        """
        if not attack_results:
            return self._get_empty_metrics()
        
        self.logger.info(f"Calculating effectiveness metrics for {len(attack_results)} attacks")
        
        # Core metrics
        success_rate = self._calculate_success_rate(attack_results)
        avg_disruption = np.mean([r.disruption_score for r in attack_results])
        avg_execution_time = np.mean([r.execution_time_ms for r in attack_results])
        
        # Advanced metrics
        consistency_score = self._calculate_consistency_score(attack_results)
        reliability_score = self._calculate_reliability_score(attack_results)
        stealth_score = self._calculate_stealth_score(attack_results)
        adaptability_score = self._calculate_adaptability_score(attack_results)
        
        # Performance metrics
        throughput = self._calculate_throughput(attack_results)
        resource_efficiency = self._calculate_resource_efficiency(attack_results)
        scalability_score = self._calculate_scalability_score(attack_results)
        
        # Defensive metrics
        defense_bypass_rate = self._calculate_defense_bypass_rate(attack_results, target_conditions)
        false_positive_rate = self._calculate_false_positive_rate(attack_results)
        detection_evasion_rate = self._calculate_detection_evasion_rate(attack_results)
        
        # Statistical analysis
        confidence_interval = self._calculate_confidence_interval(
            [r.disruption_score for r in attack_results]
        )
        
        # Create effectiveness metrics
        metrics = EffectivenessMetrics(
            success_rate=success_rate,
            avg_disruption_score=avg_disruption,
            avg_execution_time_ms=avg_execution_time,
            consistency_score=consistency_score,
            reliability_score=reliability_score,
            stealth_score=stealth_score,
            adaptability_score=adaptability_score,
            throughput=throughput,
            resource_efficiency=resource_efficiency,
            scalability_score=scalability_score,
            defense_bypass_rate=defense_bypass_rate,
            false_positive_rate=false_positive_rate,
            detection_evasion_rate=detection_evasion_rate,
            sample_size=len(attack_results),
            measurement_period=f"{attack_results[0].timestamp} to {attack_results[-1].timestamp}",
            confidence_interval=confidence_interval
        )
        
        # Record metrics
        self.effectiveness_history.append(metrics)
        
        self.logger.info(
            f"Effectiveness metrics calculated: success_rate={success_rate:.3f}, "
            f"avg_disruption={avg_disruption:.3f}, consistency={consistency_score:.3f}"
        )
        
        return metrics
    
    def calculate_success_rate_analysis(
        self,
        attack_results: List[UnifiedAttackResult],
        group_by_conditions: Dict[str, Any] = None
    ) -> SuccessRateAnalysis:
        """
        Calculate comprehensive success rate analysis.
        
        Args:
            attack_results: List of attack results to analyze
            group_by_conditions: Conditions to group success rates by
            
        Returns:
            SuccessRateAnalysis with detailed breakdown
        """
        if not attack_results:
            return self._get_empty_success_analysis()
        
        self.logger.info(f"Calculating success rate analysis for {len(attack_results)} attacks")
        
        # Overall success rate
        overall_success_rate = self._calculate_success_rate(attack_results)
        
        # Success by attack type
        success_by_type = self._calculate_success_by_attack_type(attack_results)
        
        # Success by target condition
        success_by_condition = self._calculate_success_by_condition(attack_results, group_by_conditions)
        
        # Success by defense level (simulated)
        success_by_defense = self._calculate_success_by_defense_level(attack_results)
        
        # Success trend analysis
        success_trend = self._calculate_success_trend(attack_results)
        
        # Statistical analysis
        success_confidence_interval = self._calculate_success_confidence_interval(attack_results)
        statistical_significance = self._calculate_statistical_significance(attack_results)
        
        # Create success rate analysis
        analysis = SuccessRateAnalysis(
            overall_success_rate=overall_success_rate,
            success_by_attack_type=success_by_type,
            success_by_target_condition=success_by_condition,
            success_by_defense_level=success_by_defense,
            success_trend=success_trend,
            success_confidence_interval=success_confidence_interval,
            statistical_significance=statistical_significance
        )
        
        # Record analysis
        self.success_rate_history.append(analysis)
        
        self.logger.info(
            f"Success rate analysis completed: overall_rate={overall_success_rate:.3f}, "
            f"trend_slope={np.polyfit(range(len(success_trend)), success_trend, 1)[0]:.4f}"
        )
        
        return analysis
    
    def benchmark_attack_performance(
        self,
        attack_results: List[UnifiedAttackResult],
        baseline_results: List[UnifiedAttackResult] = None
    ) -> Dict[str, Any]:
        """
        Benchmark attack performance against baseline or historical data.
        
        Args:
            attack_results: Current attack results to benchmark
            baseline_results: Baseline results for comparison
            
        Returns:
            Dictionary with benchmarking results
        """
        if not attack_results:
            return {'error': 'No attack results provided for benchmarking'}
        
        self.logger.info(f"Benchmarking {len(attack_results)} attacks")
        
        # Use historical data as baseline if not provided
        if baseline_results is None:
            baseline_results = self._get_historical_baseline()
        
        # Calculate current metrics
        current_metrics = self._calculate_basic_metrics(attack_results)
        
        # Calculate baseline metrics
        baseline_metrics = self._calculate_basic_metrics(baseline_results) if baseline_results else {}
        
        # Performance comparison
        performance_comparison = {}
        if baseline_metrics:
            for metric, current_value in current_metrics.items():
                baseline_value = baseline_metrics.get(metric, 0)
                if baseline_value > 0:
                    improvement = (current_value - baseline_value) / baseline_value
                    performance_comparison[metric] = {
                        'current': current_value,
                        'baseline': baseline_value,
                        'improvement': improvement,
                        'improvement_percent': improvement * 100
                    }
        
        # Performance ranking
        performance_ranking = self._rank_attack_performance(attack_results)
        
        # Efficiency analysis
        efficiency_analysis = self._analyze_attack_efficiency(attack_results)
        
        # Create benchmark result
        benchmark_result = {
            'timestamp': datetime.now(),
            'sample_size': len(attack_results),
            'current_metrics': current_metrics,
            'baseline_metrics': baseline_metrics,
            'performance_comparison': performance_comparison,
            'performance_ranking': performance_ranking,
            'efficiency_analysis': efficiency_analysis,
            'overall_score': self._calculate_overall_benchmark_score(current_metrics),
            'recommendations': self._generate_benchmark_recommendations(performance_comparison)
        }
        
        # Record benchmark
        self.benchmark_results.append(benchmark_result)
        
        self.logger.info(
            f"Benchmarking completed: overall_score={benchmark_result['overall_score']:.3f}"
        )
        
        return benchmark_result
    
    def analyze_effectiveness_trends(
        self,
        time_window: int = None
    ) -> Dict[str, Any]:
        """
        Analyze effectiveness trends over time.
        
        Args:
            time_window: Number of recent measurements to analyze
            
        Returns:
            Dictionary with trend analysis results
        """
        if not self.effectiveness_history:
            return {'error': 'No effectiveness history available for trend analysis'}
        
        window = time_window or self.trend_window
        recent_history = self.effectiveness_history[-window:]
        
        self.logger.info(f"Analyzing effectiveness trends for {len(recent_history)} measurements")
        
        # Extract time series data
        timestamps = [m.timestamp for m in recent_history]
        success_rates = [m.success_rate for m in recent_history]
        disruption_scores = [m.avg_disruption_score for m in recent_history]
        execution_times = [m.avg_execution_time_ms for m in recent_history]
        
        # Trend analysis
        trend_analysis = {
            'success_rate_trend': self._analyze_trend(success_rates),
            'disruption_score_trend': self._analyze_trend(disruption_scores),
            'execution_time_trend': self._analyze_trend(execution_times),
            'consistency_trend': self._analyze_trend([m.consistency_score for m in recent_history]),
            'reliability_trend': self._analyze_trend([m.reliability_score for m in recent_history])
        }
        
        # Overall trend assessment
        overall_trend = self._assess_overall_trend(trend_analysis)
        
        # Anomaly detection
        anomalies = self._detect_anomalies(recent_history)
        
        # Forecasting
        forecast = self._forecast_effectiveness(recent_history)
        
        return {
            'analysis_period': f"{timestamps[0]} to {timestamps[-1]}",
            'sample_size': len(recent_history),
            'trend_analysis': trend_analysis,
            'overall_trend': overall_trend,
            'anomalies': anomalies,
            'forecast': forecast,
            'statistical_significance': self._calculate_trend_significance(success_rates)
        }
    
    def assess_defensive_countermeasures(
        self,
        attack_results_before: List[UnifiedAttackResult],
        attack_results_after: List[UnifiedAttackResult],
        countermeasure_description: str
    ) -> Dict[str, Any]:
        """
        Assess the effectiveness of defensive countermeasures.
        
        Args:
            attack_results_before: Attack results before countermeasure
            attack_results_after: Attack results after countermeasure
            countermeasure_description: Description of the countermeasure
            
        Returns:
            Dictionary with countermeasure assessment results
        """
        if not attack_results_before or not attack_results_after:
            return {'error': 'Insufficient data for countermeasure assessment'}
        
        self.logger.info(f"Assessing countermeasure effectiveness: {countermeasure_description}")
        
        # Calculate metrics before and after
        metrics_before = self._calculate_basic_metrics(attack_results_before)
        metrics_after = self._calculate_basic_metrics(attack_results_after)
        
        # Calculate impact
        impact_analysis = {}
        for metric, value_before in metrics_before.items():
            value_after = metrics_after.get(metric, 0)
            
            if value_before > 0:
                reduction = (value_before - value_after) / value_before
                impact_analysis[metric] = {
                    'before': value_before,
                    'after': value_after,
                    'reduction': reduction,
                    'reduction_percent': reduction * 100
                }
        
        # Statistical significance testing
        significance_tests = self._perform_significance_tests(
            attack_results_before, attack_results_after
        )
        
        # Countermeasure effectiveness rating
        effectiveness_rating = self._rate_countermeasure_effectiveness(impact_analysis)
        
        # Side effects analysis
        side_effects = self._analyze_countermeasure_side_effects(
            attack_results_before, attack_results_after
        )
        
        return {
            'countermeasure': countermeasure_description,
            'assessment_date': datetime.now(),
            'sample_sizes': {
                'before': len(attack_results_before),
                'after': len(attack_results_after)
            },
            'metrics_before': metrics_before,
            'metrics_after': metrics_after,
            'impact_analysis': impact_analysis,
            'significance_tests': significance_tests,
            'effectiveness_rating': effectiveness_rating,
            'side_effects': side_effects,
            'recommendations': self._generate_countermeasure_recommendations(impact_analysis, effectiveness_rating)
        }
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive effectiveness analysis report."""
        if not self.orchestrator.unified_history:
            return {'error': 'No attack data available for comprehensive report'}
        
        # Get recent attack results
        recent_attacks = self.orchestrator.unified_history[-200:]
        
        # Calculate comprehensive metrics
        effectiveness_metrics = self.calculate_effectiveness_metrics(recent_attacks)
        success_rate_analysis = self.calculate_success_rate_analysis(recent_attacks)
        
        # Benchmark performance
        benchmark_results = self.benchmark_attack_performance(recent_attacks)
        
        # Trend analysis
        trend_analysis = self.analyze_effectiveness_trends()
        
        # Module-specific analysis
        module_analysis = self._analyze_by_module(recent_attacks)
        
        # Attack type analysis
        attack_type_analysis = self._analyze_by_attack_type(recent_attacks)
        
        # Performance insights
        insights = self._generate_performance_insights(
            effectiveness_metrics, success_rate_analysis, benchmark_results
        )
        
        # Recommendations
        recommendations = self._generate_comprehensive_recommendations(
            effectiveness_metrics, success_rate_analysis, trend_analysis
        )
        
        report = {
            'report_metadata': {
                'generated_at': datetime.now(),
                'analysis_period': f"{recent_attacks[0].timestamp} to {recent_attacks[-1].timestamp}",
                'total_attacks_analyzed': len(recent_attacks),
                'report_version': '1.0'
            },
            'executive_summary': {
                'overall_success_rate': success_rate_analysis.overall_success_rate,
                'avg_disruption_score': effectiveness_metrics.avg_disruption_score,
                'performance_rating': self._get_performance_rating(effectiveness_metrics.avg_disruption_score),
                'key_findings': insights[:3],  # Top 3 insights
                'critical_recommendations': recommendations[:3]  # Top 3 recommendations
            },
            'detailed_analysis': {
                'effectiveness_metrics': effectiveness_metrics,
                'success_rate_analysis': success_rate_analysis,
                'benchmark_results': benchmark_results,
                'trend_analysis': trend_analysis,
                'module_analysis': module_analysis,
                'attack_type_analysis': attack_type_analysis
            },
            'insights_and_recommendations': {
                'performance_insights': insights,
                'recommendations': recommendations,
                'risk_assessment': self._assess_attack_risks(recent_attacks),
                'defensive_priorities': self._prioritize_defensive_measures(recent_attacks)
            }
        }
        
        return report
    
    # Helper methods for metric calculations
    def _calculate_success_rate(self, results: List[UnifiedAttackResult]) -> float:
        """Calculate basic success rate."""
        if not results:
            return 0.0
        return sum(1 for r in results if r.success) / len(results)
    
    def _calculate_consistency_score(self, results: List[UnifiedAttackResult]) -> float:
        """Calculate consistency score based on disruption score variance."""
        if not results:
            return 0.0
        
        disruption_scores = [r.disruption_score for r in results]
        if len(disruption_scores) < 2:
            return 1.0
        
        # Lower variance = higher consistency
        variance = np.var(disruption_scores)
        consistency = 1.0 / (1.0 + variance)
        return min(consistency, 1.0)
    
    def _calculate_reliability_score(self, results: List[UnifiedAttackResult]) -> float:
        """Calculate reliability score based on success rate stability."""
        if not results:
            return 0.0
        
        # Calculate success rate over sliding windows
        window_size = min(10, len(results) // 2)
        if window_size < 2:
            return self._calculate_success_rate(results)
        
        success_rates = []
        for i in range(len(results) - window_size + 1):
            window = results[i:i + window_size]
            success_rates.append(self._calculate_success_rate(window))
        
        # Lower variance in success rates = higher reliability
        if len(success_rates) < 2:
            return success_rates[0] if success_rates else 0.0
        
        variance = np.var(success_rates)
        reliability = 1.0 / (1.0 + variance * 10)  # Scale variance impact
        return min(reliability, 1.0)
    
    def _calculate_stealth_score(self, results: List[UnifiedAttackResult]) -> float:
        """Calculate stealth score based on execution time and disruption efficiency."""
        if not results:
            return 0.0
        
        # Stealth = high disruption with low execution time
        disruption_scores = [r.disruption_score for r in results]
        execution_times = [r.execution_time_ms for r in results]
        
        avg_disruption = np.mean(disruption_scores)
        avg_execution_time = np.mean(execution_times)
        
        # Normalize execution time (lower is better for stealth)
        time_factor = 1.0 / (1.0 + avg_execution_time / 1000)  # Convert ms to seconds
        
        stealth_score = avg_disruption * time_factor
        return min(stealth_score, 1.0)
    
    def _calculate_adaptability_score(self, results: List[UnifiedAttackResult]) -> float:
        """Calculate adaptability score based on attack type diversity."""
        if not results:
            return 0.0
        
        # Count unique attack types
        attack_types = set(r.attack_type for r in results)
        modules = set(r.attack_module for r in results)
        
        # More diversity = higher adaptability
        type_diversity = len(attack_types) / max(len(results), 1)
        module_diversity = len(modules) / max(len(results), 1)
        
        adaptability = (type_diversity + module_diversity) / 2.0
        return min(adaptability, 1.0)
    
    def _calculate_throughput(self, results: List[UnifiedAttackResult]) -> float:
        """Calculate attack throughput (attacks per second)."""
        if not results or len(results) < 2:
            return 0.0
        
        # Calculate time span
        start_time = min(r.timestamp for r in results)
        end_time = max(r.timestamp for r in results)
        time_span = (end_time - start_time).total_seconds()
        
        if time_span <= 0:
            return 0.0
        
        return len(results) / time_span
    
    def _calculate_resource_efficiency(self, results: List[UnifiedAttackResult]) -> float:
        """Calculate resource efficiency (disruption per unit time)."""
        if not results:
            return 0.0
        
        total_disruption = sum(r.disruption_score for r in results)
        total_time = sum(r.execution_time_ms for r in results)
        
        if total_time <= 0:
            return 0.0
        
        return total_disruption / (total_time / 1000)  # Per second
    
    def _calculate_scalability_score(self, results: List[UnifiedAttackResult]) -> float:
        """Calculate scalability score based on performance stability with volume."""
        if not results:
            return 0.0
        
        # For now, assume linear scalability
        # In practice, this would measure performance degradation with increased load
        return 0.8  # Placeholder
    
    def _calculate_defense_bypass_rate(self, results: List[UnifiedAttackResult], conditions: Dict[str, Any]) -> float:
        """Calculate defense bypass rate."""
        # Placeholder implementation
        # In practice, this would measure success rate against different defense levels
        return self._calculate_success_rate(results)
    
    def _calculate_false_positive_rate(self, results: List[UnifiedAttackResult]) -> float:
        """Calculate false positive rate."""
        # Placeholder implementation
        # In practice, this would measure incorrect success classifications
        return 0.05  # Assume 5% false positive rate
    
    def _calculate_detection_evasion_rate(self, results: List[UnifiedAttackResult]) -> float:
        """Calculate detection evasion rate."""
        # Placeholder implementation
        # In practice, this would measure how often attacks avoid detection
        return 0.7  # Assume 70% evasion rate
    
    def _calculate_confidence_interval(self, values: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval for values."""
        if not values:
            return (0.0, 0.0)
        
        confidence_level = self.metrics_config['confidence_level']
        mean = np.mean(values)
        std_error = stats.sem(values)
        
        if len(values) < 2:
            return (mean, mean)
        
        interval = stats.t.interval(confidence_level, len(values) - 1, mean, std_error)
        return interval
    
    def _get_empty_metrics(self) -> EffectivenessMetrics:
        """Get empty effectiveness metrics."""
        return EffectivenessMetrics(
            success_rate=0.0,
            avg_disruption_score=0.0,
            avg_execution_time_ms=0.0,
            consistency_score=0.0,
            reliability_score=0.0,
            stealth_score=0.0,
            adaptability_score=0.0,
            throughput=0.0,
            resource_efficiency=0.0,
            scalability_score=0.0,
            defense_bypass_rate=0.0,
            false_positive_rate=0.0,
            detection_evasion_rate=0.0,
            sample_size=0,
            measurement_period="N/A",
            confidence_interval=(0.0, 0.0)
        )
    
    def _get_empty_success_analysis(self) -> SuccessRateAnalysis:
        """Get empty success rate analysis."""
        return SuccessRateAnalysis(
            overall_success_rate=0.0,
            success_by_attack_type={},
            success_by_target_condition={},
            success_by_defense_level={},
            success_trend=[],
            success_confidence_interval=(0.0, 0.0),
            statistical_significance=0.0
        )
    
    def _calculate_success_by_attack_type(self, results: List[UnifiedAttackResult]) -> Dict[str, float]:
        """Calculate success rates by attack type."""
        type_groups = defaultdict(list)
        
        for result in results:
            type_groups[result.attack_type].append(result)
        
        return {
            attack_type: self._calculate_success_rate(group_results)
            for attack_type, group_results in type_groups.items()
        }
    
    def _calculate_success_by_condition(self, results: List[UnifiedAttackResult], conditions: Dict[str, Any]) -> Dict[str, float]:
        """Calculate success rates by target condition."""
        # Placeholder implementation
        return {'normal': 0.7, 'stressed': 0.5, 'defended': 0.3}
    
    def _calculate_success_by_defense_level(self, results: List[UnifiedAttackResult]) -> Dict[str, float]:
        """Calculate success rates by defense level."""
        # Placeholder implementation
        return {'none': 0.9, 'basic': 0.6, 'advanced': 0.3, 'military': 0.1}
    
    def _calculate_success_trend(self, results: List[UnifiedAttackResult]) -> List[float]:
        """Calculate success rate trend over time."""
        if not results:
            return []
        
        # Sort by timestamp
        sorted_results = sorted(results, key=lambda r: r.timestamp)
        
        # Calculate success rate over sliding windows
        window_size = min(10, len(sorted_results) // 5)
        if window_size < 1:
            return [self._calculate_success_rate(sorted_results)]
        
        trend = []
        for i in range(len(sorted_results) - window_size + 1):
            window = sorted_results[i:i + window_size]
            trend.append(self._calculate_success_rate(window))
        
        return trend
    
    def _calculate_success_confidence_interval(self, results: List[UnifiedAttackResult]) -> Tuple[float, float]:
        """Calculate confidence interval for success rate."""
        if not results:
            return (0.0, 0.0)
        
        success_rate = self._calculate_success_rate(results)
        n = len(results)
        
        # Binomial confidence interval
        std_error = np.sqrt(success_rate * (1 - success_rate) / n)
        margin = 1.96 * std_error  # 95% confidence interval
        
        return (
            max(0.0, success_rate - margin),
            min(1.0, success_rate + margin)
        )
    
    def _calculate_statistical_significance(self, results: List[UnifiedAttackResult]) -> float:
        """Calculate statistical significance of success rate."""
        if not results:
            return 0.0
        
        # Test against null hypothesis of 50% success rate
        success_count = sum(1 for r in results if r.success)
        n = len(results)
        
        if n < 2:
            return 0.0
        
        # Binomial test
        p_value = stats.binom_test(success_count, n, 0.5)
        return 1.0 - p_value  # Convert to significance level
    
    def _analyze_trend(self, values: List[float]) -> Dict[str, Any]:
        """Analyze trend in a series of values."""
        if len(values) < 2:
            return {'trend': 'insufficient_data', 'slope': 0.0, 'r_squared': 0.0}
        
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        trend_direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
        
        return {
            'trend': trend_direction,
            'slope': slope,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'is_significant': p_value < self.trend_significance_threshold
        }
    
    def _assess_overall_trend(self, trend_analysis: Dict[str, Any]) -> str:
        """Assess overall trend across all metrics."""
        significant_trends = []
        
        for metric, analysis in trend_analysis.items():
            if analysis.get('is_significant', False):
                significant_trends.append(analysis['trend'])
        
        if not significant_trends:
            return 'no_significant_trends'
        
        # Count trend directions
        increasing = significant_trends.count('increasing')
        decreasing = significant_trends.count('decreasing')
        
        if increasing > decreasing:
            return 'improving'
        elif decreasing > increasing:
            return 'declining'
        else:
            return 'mixed'
    
    def _detect_anomalies(self, metrics_history: List[EffectivenessMetrics]) -> List[Dict[str, Any]]:
        """Detect anomalies in metrics history."""
        anomalies = []
        
        if len(metrics_history) < 10:
            return anomalies
        
        # Check for outliers in success rates
        success_rates = [m.success_rate for m in metrics_history]
        mean_success = np.mean(success_rates)
        std_success = np.std(success_rates)
        
        for i, metric in enumerate(metrics_history):
            if abs(metric.success_rate - mean_success) > 2 * std_success:
                anomalies.append({
                    'type': 'success_rate_outlier',
                    'timestamp': metric.timestamp,
                    'value': metric.success_rate,
                    'expected_range': (mean_success - 2 * std_success, mean_success + 2 * std_success)
                })
        
        return anomalies
    
    def _forecast_effectiveness(self, metrics_history: List[EffectivenessMetrics]) -> Dict[str, Any]:
        """Forecast future effectiveness based on historical trends."""
        if len(metrics_history) < 3:
            return {'forecast': 'insufficient_data'}
        
        # Simple linear forecast for success rate
        success_rates = [m.success_rate for m in metrics_history]
        x = np.arange(len(success_rates))
        
        # Fit linear trend
        coeffs = np.polyfit(x, success_rates, 1)
        
        # Forecast next 5 periods
        forecast_periods = 5
        forecast_x = np.arange(len(success_rates), len(success_rates) + forecast_periods)
        forecast_y = np.polyval(coeffs, forecast_x)
        
        return {
            'forecast_periods': forecast_periods,
            'forecasted_success_rates': forecast_y.tolist(),
            'trend_slope': coeffs[0],
            'confidence': 'low'  # Simple linear model has low confidence
        }
    
    def _calculate_trend_significance(self, values: List[float]) -> float:
        """Calculate statistical significance of trend."""
        if len(values) < 3:
            return 0.0
        
        x = np.arange(len(values))
        _, _, _, p_value, _ = stats.linregress(x, values)
        
        return 1.0 - p_value
    
    def _get_historical_baseline(self) -> List[UnifiedAttackResult]:
        """Get historical baseline for comparison."""
        # Return first half of history as baseline
        all_history = self.orchestrator.unified_history
        if len(all_history) < 20:
            return []
        
        return all_history[:len(all_history) // 2]
    
    def _calculate_basic_metrics(self, results: List[UnifiedAttackResult]) -> Dict[str, float]:
        """Calculate basic metrics for benchmarking."""
        if not results:
            return {}
        
        return {
            'success_rate': self._calculate_success_rate(results),
            'avg_disruption_score': np.mean([r.disruption_score for r in results]),
            'avg_execution_time_ms': np.mean([r.execution_time_ms for r in results]),
            'consistency_score': self._calculate_consistency_score(results),
            'reliability_score': self._calculate_reliability_score(results)
        }
    
    def _rank_attack_performance(self, results: List[UnifiedAttackResult]) -> List[Dict[str, Any]]:
        """Rank attack performance by effectiveness."""
        attack_performance = defaultdict(list)
        
        # Group by attack type
        for result in results:
            attack_performance[result.attack_type].append(result)
        
        # Calculate effectiveness for each attack type
        rankings = []
        for attack_type, attack_results in attack_performance.items():
            success_rate = self._calculate_success_rate(attack_results)
            avg_disruption = np.mean([r.disruption_score for r in attack_results])
            
            # Combined effectiveness score
            effectiveness = (success_rate * 0.6) + (avg_disruption * 0.4)
            
            rankings.append({
                'attack_type': attack_type,
                'effectiveness_score': effectiveness,
                'success_rate': success_rate,
                'avg_disruption': avg_disruption,
                'sample_size': len(attack_results)
            })
        
        # Sort by effectiveness
        rankings.sort(key=lambda x: x['effectiveness_score'], reverse=True)
        
        return rankings
    
    def _analyze_attack_efficiency(self, results: List[UnifiedAttackResult]) -> Dict[str, Any]:
        """Analyze attack efficiency metrics."""
        if not results:
            return {}
        
        # Time efficiency
        execution_times = [r.execution_time_ms for r in results]
        time_efficiency = {
            'avg_execution_time_ms': np.mean(execution_times),
            'median_execution_time_ms': np.median(execution_times),
            'time_consistency': 1.0 / (1.0 + np.std(execution_times))
        }
        
        # Disruption efficiency
        disruption_scores = [r.disruption_score for r in results]
        disruption_efficiency = {
            'avg_disruption_score': np.mean(disruption_scores),
            'disruption_consistency': 1.0 / (1.0 + np.std(disruption_scores)),
            'disruption_per_ms': np.mean(disruption_scores) / max(np.mean(execution_times), 1)
        }
        
        return {
            'time_efficiency': time_efficiency,
            'disruption_efficiency': disruption_efficiency,
            'overall_efficiency': (time_efficiency['time_consistency'] + disruption_efficiency['disruption_consistency']) / 2
        }
    
    def _calculate_overall_benchmark_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall benchmark score."""
        if not metrics:
            return 0.0
        
        # Weighted combination of key metrics
        weights = {
            'success_rate': 0.4,
            'avg_disruption_score': 0.3,
            'consistency_score': 0.15,
            'reliability_score': 0.15
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                score += metrics[metric] * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _generate_benchmark_recommendations(self, comparison: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on benchmark results."""
        recommendations = []
        
        for metric, data in comparison.items():
            improvement = data.get('improvement', 0)
            
            if improvement < -0.1:  # 10% degradation
                recommendations.append(f"Performance degradation detected in {metric}. Consider optimization.")
            elif improvement > 0.1:  # 10% improvement
                recommendations.append(f"Good improvement in {metric}. Continue current approach.")
        
        return recommendations
    
    def _perform_significance_tests(self, before: List[UnifiedAttackResult], after: List[UnifiedAttackResult]) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        before_success = [1 if r.success else 0 for r in before]
        after_success = [1 if r.success else 0 for r in after]
        
        # Chi-square test for success rates
        if len(before_success) > 0 and len(after_success) > 0:
            contingency_table = [
                [sum(before_success), len(before_success) - sum(before_success)],
                [sum(after_success), len(after_success) - sum(after_success)]
            ]
            
            chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
            
            return {
                'chi2_statistic': chi2,
                'p_value': p_value,
                'is_significant': p_value < 0.05,
                'test_type': 'chi_square'
            }
        
        return {'error': 'Insufficient data for significance testing'}
    
    def _rate_countermeasure_effectiveness(self, impact: Dict[str, Any]) -> str:
        """Rate countermeasure effectiveness."""
        if not impact:
            return 'unknown'
        
        # Calculate average reduction
        reductions = [data['reduction'] for data in impact.values() if 'reduction' in data]
        
        if not reductions:
            return 'unknown'
        
        avg_reduction = np.mean(reductions)
        
        if avg_reduction > 0.7:
            return 'highly_effective'
        elif avg_reduction > 0.5:
            return 'effective'
        elif avg_reduction > 0.3:
            return 'moderately_effective'
        elif avg_reduction > 0.1:
            return 'slightly_effective'
        else:
            return 'ineffective'
    
    def _analyze_countermeasure_side_effects(self, before: List[UnifiedAttackResult], after: List[UnifiedAttackResult]) -> Dict[str, Any]:
        """Analyze side effects of countermeasures."""
        # Placeholder implementation
        return {
            'performance_impact': 'low',
            'false_positive_rate': 0.05,
            'operational_overhead': 'medium'
        }
    
    def _generate_countermeasure_recommendations(self, impact: Dict[str, Any], rating: str) -> List[str]:
        """Generate countermeasure recommendations."""
        recommendations = []
        
        if rating == 'ineffective':
            recommendations.append("Countermeasure shows little impact. Consider alternative approaches.")
        elif rating == 'highly_effective':
            recommendations.append("Countermeasure is highly effective. Consider deploying broadly.")
        
        return recommendations
    
    def _analyze_by_module(self, results: List[UnifiedAttackResult]) -> Dict[str, Any]:
        """Analyze results by attack module."""
        module_analysis = {}
        
        for module in AttackModule:
            module_results = [r for r in results if r.attack_module == module.value]
            
            if module_results:
                module_analysis[module.value] = {
                    'total_attacks': len(module_results),
                    'success_rate': self._calculate_success_rate(module_results),
                    'avg_disruption': np.mean([r.disruption_score for r in module_results]),
                    'avg_execution_time': np.mean([r.execution_time_ms for r in module_results])
                }
        
        return module_analysis
    
    def _analyze_by_attack_type(self, results: List[UnifiedAttackResult]) -> Dict[str, Any]:
        """Analyze results by attack type."""
        type_analysis = {}
        
        attack_types = set(r.attack_type for r in results)
        
        for attack_type in attack_types:
            type_results = [r for r in results if r.attack_type == attack_type]
            
            type_analysis[attack_type] = {
                'total_attacks': len(type_results),
                'success_rate': self._calculate_success_rate(type_results),
                'avg_disruption': np.mean([r.disruption_score for r in type_results]),
                'avg_execution_time': np.mean([r.execution_time_ms for r in type_results])
            }
        
        return type_analysis
    
    def _generate_performance_insights(self, metrics: EffectivenessMetrics, success_analysis: SuccessRateAnalysis, benchmark: Dict[str, Any]) -> List[str]:
        """Generate performance insights."""
        insights = []
        
        # Success rate insights
        if success_analysis.overall_success_rate > 0.8:
            insights.append("High attack success rate indicates vulnerable target system.")
        elif success_analysis.overall_success_rate < 0.3:
            insights.append("Low attack success rate suggests strong defensive measures.")
        
        # Consistency insights
        if metrics.consistency_score > 0.8:
            insights.append("High consistency indicates stable attack performance.")
        elif metrics.consistency_score < 0.5:
            insights.append("Low consistency suggests unpredictable attack outcomes.")
        
        # Efficiency insights
        if metrics.resource_efficiency > 1.0:
            insights.append("High resource efficiency indicates optimized attack methods.")
        
        return insights
    
    def _generate_comprehensive_recommendations(self, metrics: EffectivenessMetrics, success_analysis: SuccessRateAnalysis, trend_analysis: Dict[str, Any]) -> List[str]:
        """Generate comprehensive recommendations."""
        recommendations = []
        
        # Performance-based recommendations
        if metrics.success_rate < 0.5:
            recommendations.append("Consider strengthening attack techniques or targeting different vulnerabilities.")
        
        if metrics.consistency_score < 0.6:
            recommendations.append("Improve attack consistency through better parameter tuning.")
        
        if metrics.avg_execution_time_ms > 500:
            recommendations.append("Optimize attack execution time for better stealth.")
        
        # Trend-based recommendations
        if trend_analysis.get('overall_trend') == 'declining':
            recommendations.append("Attack effectiveness is declining. Update techniques to counter evolving defenses.")
        
        return recommendations
    
    def _get_performance_rating(self, score: float) -> str:
        """Get performance rating for a score."""
        if score >= self.performance_thresholds['excellent']:
            return 'excellent'
        elif score >= self.performance_thresholds['good']:
            return 'good'
        elif score >= self.performance_thresholds['fair']:
            return 'fair'
        else:
            return 'poor'
    
    def _assess_attack_risks(self, results: List[UnifiedAttackResult]) -> Dict[str, Any]:
        """Assess risks associated with attack patterns."""
        return {
            'detection_risk': 'medium',
            'escalation_risk': 'low',
            'defensive_adaptation_risk': 'high'
        }
    
    def _prioritize_defensive_measures(self, results: List[UnifiedAttackResult]) -> List[str]:
        """Prioritize defensive measures based on attack patterns."""
        return [
            "Implement coordination attack detection",
            "Strengthen regime detection validation",
            "Add policy gradient protection"
        ]

# Example usage
def run_effectiveness_analysis_demo():
    """Demonstrate attack effectiveness analysis capabilities."""
    print("🚨" * 50)
    print("AGENT GAMMA MISSION - ATTACK EFFECTIVENESS ANALYSIS DEMO")
    print("🚨" * 50)
    
    # Initialize orchestrator and analyzer
    orchestrator = UnifiedAttackOrchestrator()
    analyzer = AttackEffectivenessAnalyzer(orchestrator)
    
    # Generate sample attack results
    sample_results = []
    for i in range(50):
        result = UnifiedAttackResult(
            attack_module=np.random.choice(['coordination', 'temporal', 'policy', 'regime']),
            attack_type=f"attack_type_{i % 5}",
            success=np.random.random() > 0.3,
            confidence=np.random.random(),
            disruption_score=np.random.random(),
            execution_time_ms=np.random.uniform(10, 200),
            original_result=None,
            metadata={},
            timestamp=datetime.now() - timedelta(hours=i)
        )
        sample_results.append(result)
    
    # Add to orchestrator history
    orchestrator.unified_history.extend(sample_results)
    
    print(f"\n📊 Analyzing {len(sample_results)} attack results")
    
    # Calculate effectiveness metrics
    metrics = analyzer.calculate_effectiveness_metrics(sample_results)
    print(f"Effectiveness Metrics:")
    print(f"  Success Rate: {metrics.success_rate:.3f}")
    print(f"  Avg Disruption Score: {metrics.avg_disruption_score:.3f}")
    print(f"  Consistency Score: {metrics.consistency_score:.3f}")
    print(f"  Reliability Score: {metrics.reliability_score:.3f}")
    
    # Success rate analysis
    success_analysis = analyzer.calculate_success_rate_analysis(sample_results)
    print(f"\nSuccess Rate Analysis:")
    print(f"  Overall Success Rate: {success_analysis.overall_success_rate:.3f}")
    print(f"  Success by Attack Type: {success_analysis.success_by_attack_type}")
    
    # Benchmark performance
    benchmark = analyzer.benchmark_attack_performance(sample_results)
    print(f"\nBenchmark Results:")
    print(f"  Overall Score: {benchmark['overall_score']:.3f}")
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report()
    print(f"\nComprehensive Report Generated:")
    print(f"  Performance Rating: {report['executive_summary']['performance_rating']}")
    print(f"  Key Findings: {len(report['insights_and_recommendations']['performance_insights'])}")
    print(f"  Recommendations: {len(report['insights_and_recommendations']['recommendations'])}")
    
    return analyzer

if __name__ == "__main__":
    run_effectiveness_analysis_demo()