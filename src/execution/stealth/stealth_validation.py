"""
Stealth Execution Validation and Impact Analysis
===============================================

Comprehensive validation framework for stealth execution effectiveness.
Analyzes market impact reduction, detection probability, and statistical
indistinguishability of synthetic order patterns.

Key Validation Areas:
1. Statistical indistinguishability tests (KS, AD, chi-square)
2. Market impact reduction measurement
3. Detection probability estimation
4. Timing pattern analysis
5. Performance benchmarking

Mathematical Foundation:
- Indistinguishability: KS test p-value > 0.05
- Impact Reduction: (I_naive - I_stealth) / I_naive
- Detection Prob: P(classifier identifies synthetic orders)
- Statistical Power: Confidence intervals for all metrics

Performance Targets:
- Indistinguishability: >95% confidence
- Impact Reduction: >80% vs naive execution
- Detection Probability: <5%
- Validation Latency: <10ms per test
"""

import logging


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, NamedTuple
import time
import structlog
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.stats import ks_2samp, anderson_ksamp, chi2_contingency
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

from src.execution.stealth.order_fragmentation import (
    FragmentationPlan, 
    ChildOrder, 
    AdaptiveFragmentationEngine,
    FragmentationStrategy
)
from training.imitation_learning_pipeline import TradeRecord, MarketFeatures

logger = structlog.get_logger()


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics for stealth execution"""
    # Statistical indistinguishability
    ks_test_p_value: float = 0.0
    anderson_darling_p_value: float = 0.0
    chi_square_p_value: float = 0.0
    indistinguishability_score: float = 0.0
    
    # Market impact analysis
    naive_impact_bps: float = 0.0
    stealth_impact_bps: float = 0.0
    impact_reduction_pct: float = 0.0
    
    # Detection analysis
    detection_probability: float = 0.0
    classifier_auc: float = 0.0
    false_positive_rate: float = 0.0
    
    # Timing analysis
    timing_naturalness_score: float = 0.0
    autocorr_similarity: float = 0.0
    clustering_similarity: float = 0.0
    
    # Performance metrics
    validation_time_ms: float = 0.0
    sample_size: int = 0
    confidence_level: float = 0.95
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'statistical_indistinguishability': {
                'ks_test_p_value': self.ks_test_p_value,
                'anderson_darling_p_value': self.anderson_darling_p_value,
                'chi_square_p_value': self.chi_square_p_value,
                'indistinguishability_score': self.indistinguishability_score
            },
            'market_impact': {
                'naive_impact_bps': self.naive_impact_bps,
                'stealth_impact_bps': self.stealth_impact_bps,
                'impact_reduction_pct': self.impact_reduction_pct
            },
            'detection_analysis': {
                'detection_probability': self.detection_probability,
                'classifier_auc': self.classifier_auc,
                'false_positive_rate': self.false_positive_rate
            },
            'timing_analysis': {
                'timing_naturalness_score': self.timing_naturalness_score,
                'autocorr_similarity': self.autocorr_similarity,
                'clustering_similarity': self.clustering_similarity
            },
            'performance': {
                'validation_time_ms': self.validation_time_ms,
                'sample_size': self.sample_size,
                'confidence_level': self.confidence_level
            }
        }
    
    def passes_validation(self, 
                         min_indistinguishability: float = 0.05,
                         min_impact_reduction: float = 0.8,
                         max_detection_prob: float = 0.05) -> bool:
        """Check if metrics pass validation thresholds"""
        checks = [
            self.ks_test_p_value > min_indistinguishability,
            self.impact_reduction_pct > min_impact_reduction,
            self.detection_probability < max_detection_prob
        ]
        return all(checks)


class StatisticalIndistinguishabilityTester:
    """
    Advanced statistical tests for order pattern indistinguishability
    
    Implements multiple statistical tests to validate that synthetic
    order patterns are indistinguishable from natural market flow
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
        
    def test_size_distributions(self, 
                              real_sizes: List[float],
                              synthetic_sizes: List[float]) -> Dict[str, float]:
        """Test if size distributions are indistinguishable"""
        results = {}
        
        try:
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = ks_2samp(real_sizes, synthetic_sizes)
            results['ks_statistic'] = ks_stat
            results['ks_p_value'] = ks_p
            results['ks_pass'] = ks_p > self.alpha
            
            # Anderson-Darling test
            ad_stat, ad_crit, ad_p = anderson_ksamp([real_sizes, synthetic_sizes])
            results['anderson_statistic'] = ad_stat
            results['anderson_p_value'] = ad_p
            results['anderson_pass'] = ad_p > self.alpha
            
            # Moment comparison
            real_moments = self._calculate_moments(real_sizes)
            synthetic_moments = self._calculate_moments(synthetic_sizes)
            
            moment_similarity = self._calculate_moment_similarity(real_moments, synthetic_moments)
            results['moment_similarity'] = moment_similarity
            results['moment_pass'] = moment_similarity > 0.9
            
        except Exception as e:
            logger.error("Size distribution test failed", error=str(e))
            results = {'error': str(e)}
        
        return results
    
    def test_timing_distributions(self,
                                real_timings: List[float],
                                synthetic_timings: List[float]) -> Dict[str, float]:
        """Test if timing distributions are indistinguishable"""
        results = {}
        
        try:
            # Convert to inter-arrival times
            real_intervals = np.diff(sorted(real_timings))
            synthetic_intervals = np.diff(sorted(synthetic_timings))
            
            if len(real_intervals) == 0 or len(synthetic_intervals) == 0:
                return {'error': 'Insufficient timing data'}
            
            # KS test on inter-arrival times
            ks_stat, ks_p = ks_2samp(real_intervals, synthetic_intervals)
            results['interval_ks_p_value'] = ks_p
            results['interval_ks_pass'] = ks_p > self.alpha
            
            # Autocorrelation similarity
            real_autocorr = self._calculate_autocorrelation(real_intervals)
            synthetic_autocorr = self._calculate_autocorrelation(synthetic_intervals)
            
            autocorr_similarity = 1.0 - abs(real_autocorr - synthetic_autocorr)
            results['autocorr_similarity'] = max(0.0, autocorr_similarity)
            
            # Clustering analysis (Hawkes process characteristics)
            real_clustering = self._estimate_clustering_coefficient(real_intervals)
            synthetic_clustering = self._estimate_clustering_coefficient(synthetic_intervals)
            
            clustering_similarity = 1.0 - abs(real_clustering - synthetic_clustering)
            results['clustering_similarity'] = max(0.0, clustering_similarity)
            
        except Exception as e:
            logger.error("Timing distribution test failed", error=str(e))
            results = {'error': str(e)}
        
        return results
    
    def test_joint_distributions(self,
                               real_orders: List[Tuple[float, float]],  # (size, timing)
                               synthetic_orders: List[Tuple[float, float]]) -> Dict[str, float]:
        """Test joint size-timing distribution indistinguishability"""
        results = {}
        
        try:
            # Convert to arrays
            real_data = np.array(real_orders)
            synthetic_data = np.array(synthetic_orders)
            
            if len(real_data) == 0 or len(synthetic_data) == 0:
                return {'error': 'Insufficient joint data'}
            
            # Normalize data for comparison
            real_normalized = self._normalize_joint_data(real_data)
            synthetic_normalized = self._normalize_joint_data(synthetic_data)
            
            # 2D KS test approximation using binned data
            bins = 20
            real_hist, x_edges, y_edges = np.histogram2d(
                real_normalized[:, 0], real_normalized[:, 1], bins=bins
            )
            synthetic_hist, _, _ = np.histogram2d(
                synthetic_normalized[:, 0], synthetic_normalized[:, 1], 
                bins=(x_edges, y_edges)
            )
            
            # Normalize histograms
            real_hist = real_hist / real_hist.sum()
            synthetic_hist = synthetic_hist / synthetic_hist.sum()
            
            # Chi-square test
            chi2_stat, chi2_p = self._chi_square_2d_test(real_hist, synthetic_hist)
            results['chi2_statistic'] = chi2_stat
            results['chi2_p_value'] = chi2_p
            results['chi2_pass'] = chi2_p > self.alpha
            
            # Correlation analysis
            real_corr = np.corrcoef(real_data[:, 0], real_data[:, 1])[0, 1]
            synthetic_corr = np.corrcoef(synthetic_data[:, 0], synthetic_data[:, 1])[0, 1]
            
            if not (np.isnan(real_corr) or np.isnan(synthetic_corr)):
                corr_similarity = 1.0 - abs(real_corr - synthetic_corr)
                results['correlation_similarity'] = max(0.0, corr_similarity)
            
        except Exception as e:
            logger.error("Joint distribution test failed", error=str(e))
            results = {'error': str(e)}
        
        return results
    
    def _calculate_moments(self, data: List[float]) -> Dict[str, float]:
        """Calculate statistical moments"""
        data_array = np.array(data)
        return {
            'mean': np.mean(data_array),
            'std': np.std(data_array),
            'skewness': stats.skew(data_array),
            'kurtosis': stats.kurtosis(data_array)
        }
    
    def _calculate_moment_similarity(self, 
                                   moments1: Dict[str, float],
                                   moments2: Dict[str, float]) -> float:
        """Calculate similarity between moment sets"""
        similarities = []
        
        for key in moments1:
            if key in moments2:
                val1, val2 = moments1[key], moments2[key]
                if val1 != 0:
                    similarity = 1.0 - abs(val1 - val2) / abs(val1)
                    similarities.append(max(0.0, similarity))
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_autocorrelation(self, data: List[float], lag: int = 1) -> float:
        """Calculate autocorrelation at given lag"""
        if len(data) <= lag:
            return 0.0
        
        try:
            correlation = np.corrcoef(data[:-lag], data[lag:])[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except (ValueError, TypeError, AttributeError) as e:
            return 0.0
    
    def _estimate_clustering_coefficient(self, intervals: List[float]) -> float:
        """Estimate clustering coefficient for Hawkes-like processes"""
        if len(intervals) < 3:
            return 0.0
        
        # Simple clustering measure: variance of local averages
        window_size = min(5, len(intervals) // 2)
        local_averages = []
        
        for i in range(len(intervals) - window_size + 1):
            local_avg = np.mean(intervals[i:i + window_size])
            local_averages.append(local_avg)
        
        if len(local_averages) < 2:
            return 0.0
        
        global_avg = np.mean(intervals)
        clustering = np.std(local_averages) / max(global_avg, 1e-10)
        
        return min(clustering, 1.0)  # Cap at 1.0
    
    def _normalize_joint_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize joint data for comparison"""
        normalized = data.copy()
        
        for i in range(data.shape[1]):
            col_data = data[:, i]
            col_min, col_max = col_data.min(), col_data.max()
            
            if col_max > col_min:
                normalized[:, i] = (col_data - col_min) / (col_max - col_min)
        
        return normalized
    
    def _chi_square_2d_test(self, 
                          observed: np.ndarray,
                          expected: np.ndarray) -> Tuple[float, float]:
        """2D chi-square test for histograms"""
        # Flatten histograms
        obs_flat = observed.flatten()
        exp_flat = expected.flatten()
        
        # Remove zero bins
        mask = (obs_flat > 0) | (exp_flat > 0)
        obs_flat = obs_flat[mask]
        exp_flat = exp_flat[mask]
        
        if len(obs_flat) == 0:
            return 0.0, 1.0
        
        # Calculate chi-square statistic
        chi2_stat = np.sum((obs_flat - exp_flat) ** 2 / (exp_flat + 1e-10))
        
        # Degrees of freedom
        df = len(obs_flat) - 1
        
        # P-value
        p_value = 1.0 - stats.chi2.cdf(chi2_stat, df)
        
        return chi2_stat, p_value


class MarketImpactAnalyzer:
    """
    Analyzes market impact reduction from stealth execution
    
    Compares impact between naive execution and stealth fragmentation
    """
    
    def __init__(self):
        self.impact_history = []
        
    def calculate_naive_impact(self, 
                             order_size: float,
                             market_volume: float,
                             volatility: float) -> float:
        """Calculate impact of naive (immediate) execution"""
        if market_volume <= 0:
            return float('inf')
        
        # Simple square-root impact model
        participation_rate = order_size / market_volume
        impact_bps = 10.0 * volatility * np.sqrt(participation_rate) * 10000
        
        return impact_bps
    
    def calculate_stealth_impact(self,
                               fragmentation_plan: FragmentationPlan,
                               market_volume: float,
                               volatility: float) -> float:
        """Calculate impact of stealth execution"""
        total_impact = 0.0
        
        for child_order in fragmentation_plan.child_orders:
            # Each fragment has reduced impact
            fragment_participation = child_order.size / market_volume
            fragment_impact = 10.0 * volatility * np.sqrt(fragment_participation) * 10000
            
            # Additional timing benefits (simplified)
            timing_reduction = 0.9  # 10% reduction from timing
            fragment_impact *= timing_reduction
            
            total_impact += fragment_impact
        
        return total_impact
    
    def analyze_impact_reduction(self,
                               order_size: float,
                               fragmentation_plan: FragmentationPlan,
                               market_volume: float,
                               volatility: float) -> Dict[str, float]:
        """Complete impact analysis comparing naive vs stealth"""
        results = {}
        
        try:
            # Calculate impacts
            naive_impact = self.calculate_naive_impact(order_size, market_volume, volatility)
            stealth_impact = self.calculate_stealth_impact(
                fragmentation_plan, market_volume, volatility
            )
            
            # Impact reduction
            if naive_impact > 0:
                impact_reduction = (naive_impact - stealth_impact) / naive_impact
            else:
                impact_reduction = 0.0
            
            results = {
                'naive_impact_bps': naive_impact,
                'stealth_impact_bps': stealth_impact,
                'impact_reduction_pct': impact_reduction,
                'absolute_savings_bps': naive_impact - stealth_impact,
                'num_fragments': fragmentation_plan.get_total_fragments(),
                'execution_window': fragmentation_plan.execution_window
            }
            
            # Track for analytics
            self.impact_history.append(results)
            
        except Exception as e:
            logger.error("Impact analysis failed", error=str(e))
            results = {'error': str(e)}
        
        return results


class DetectionProbabilityEstimator:
    """
    Estimates probability of stealth execution detection
    
    Uses machine learning classifiers to estimate how likely
    synthetic patterns are to be detected as artificial
    """
    
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_names = [
            'size', 'interval', 'size_ratio', 'interval_ratio',
            'cumulative_size', 'time_since_start', 'autocorr_local'
        ]
        self.is_trained = False
    
    def extract_features(self, orders: List[Tuple[float, float, str]]) -> np.ndarray:
        """Extract features from order sequence"""
        if not orders:
            return np.array([]).reshape(0, len(self.feature_names))
        
        features = []
        sizes = [order[0] for order in orders]
        timings = [order[1] for order in orders]
        
        # Sort by timing
        sorted_orders = sorted(zip(sizes, timings), key=lambda x: x[1])
        sorted_sizes = [x[0] for x in sorted_orders]
        sorted_timings = [x[1] for x in sorted_orders]
        
        intervals = np.diff(sorted_timings) if len(sorted_timings) > 1 else [0.0]
        
        for i, (size, timing) in enumerate(zip(sorted_sizes, sorted_timings)):
            feature_vector = []
            
            # Basic features
            feature_vector.append(size)
            feature_vector.append(intervals[min(i, len(intervals) - 1)])
            
            # Relative features
            if i > 0:
                feature_vector.append(size / sorted_sizes[i-1])
                feature_vector.append(intervals[i-1] / max(np.mean(intervals), 1e-10))
            else:
                feature_vector.append(1.0)
                feature_vector.append(1.0)
            
            # Cumulative features
            cumulative_size = sum(sorted_sizes[:i+1])
            feature_vector.append(cumulative_size)
            
            # Timing features
            time_since_start = timing - sorted_timings[0] if sorted_timings else 0.0
            feature_vector.append(time_since_start)
            
            # Local autocorrelation
            if i >= 2:
                recent_intervals = intervals[max(0, i-2):i]
                if len(recent_intervals) >= 2:
                    local_autocorr = np.corrcoef(recent_intervals[:-1], recent_intervals[1:])[0, 1]
                    if np.isnan(local_autocorr):
                        local_autocorr = 0.0
                else:
                    local_autocorr = 0.0
            else:
                local_autocorr = 0.0
            
            feature_vector.append(local_autocorr)
            features.append(feature_vector)
        
        return np.array(features)
    
    def train_detector(self,
                      real_order_sequences: List[List[Tuple[float, float, str]]],
                      synthetic_order_sequences: List[List[Tuple[float, float, str]]]) -> Dict[str, float]:
        """Train detection classifier"""
        try:
            # Extract features
            real_features_list = []
            synthetic_features_list = []
            
            for sequence in real_order_sequences:
                features = self.extract_features(sequence)
                if len(features) > 0:
                    real_features_list.append(features)
            
            for sequence in synthetic_order_sequences:
                features = self.extract_features(sequence)
                if len(features) > 0:
                    synthetic_features_list.append(features)
            
            if not real_features_list or not synthetic_features_list:
                return {'error': 'Insufficient training data'}
            
            # Combine features
            all_features = []
            all_labels = []
            
            for features in real_features_list:
                all_features.extend(features)
                all_labels.extend([0] * len(features))  # 0 = real
            
            for features in synthetic_features_list:
                all_features.extend(features)
                all_labels.extend([1] * len(features))  # 1 = synthetic
            
            X = np.array(all_features)
            y = np.array(all_labels)
            
            if len(X) == 0 or len(np.unique(y)) < 2:
                return {'error': 'Insufficient training samples'}
            
            # Train classifier
            self.classifier.fit(X, y)
            self.is_trained = True
            
            # Cross-validation
            cv_scores = cross_val_score(self.classifier, X, y, cv=5, scoring='roc_auc')
            
            results = {
                'training_samples': len(X),
                'cv_auc_mean': np.mean(cv_scores),
                'cv_auc_std': np.std(cv_scores),
                'feature_importances': dict(zip(self.feature_names, self.classifier.feature_importances_))
            }
            
            return results
            
        except Exception as e:
            logger.error("Detector training failed", error=str(e))
            return {'error': str(e)}
    
    def estimate_detection_probability(self,
                                     synthetic_sequences: List[List[Tuple[float, float, str]]]) -> Dict[str, float]:
        """Estimate detection probability for synthetic sequences"""
        if not self.is_trained:
            return {'error': 'Detector not trained'}
        
        try:
            all_features = []
            sequence_probs = []
            
            for sequence in synthetic_sequences:
                features = self.extract_features(sequence)
                if len(features) > 0:
                    # Predict probabilities
                    probs = self.classifier.predict_proba(features)[:, 1]  # Probability of synthetic
                    sequence_prob = np.mean(probs)
                    sequence_probs.append(sequence_prob)
                    all_features.extend(features)
            
            if not sequence_probs:
                return {'error': 'No valid sequences to analyze'}
            
            results = {
                'mean_detection_prob': np.mean(sequence_probs),
                'std_detection_prob': np.std(sequence_probs),
                'max_detection_prob': np.max(sequence_probs),
                'min_detection_prob': np.min(sequence_probs),
                'sequences_analyzed': len(sequence_probs)
            }
            
            return results
            
        except Exception as e:
            logger.error("Detection probability estimation failed", error=str(e))
            return {'error': str(e)}


class StealthExecutionValidator:
    """
    Main validation framework orchestrating all stealth execution tests
    
    Provides comprehensive validation of stealth execution effectiveness
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        
        # Initialize components
        self.indistinguishability_tester = StatisticalIndistinguishabilityTester(confidence_level)
        self.impact_analyzer = MarketImpactAnalyzer()
        self.detection_estimator = DetectionProbabilityEstimator()
        
        # Results storage
        self.validation_history = []
        
        logger.info("Stealth execution validator initialized", confidence_level=confidence_level)
    
    def validate_fragmentation_plan(self,
                                  fragmentation_plan: FragmentationPlan,
                                  real_trades: List[TradeRecord],
                                  market_volume: float,
                                  volatility: float) -> ValidationMetrics:
        """
        Comprehensive validation of a fragmentation plan
        
        Args:
            fragmentation_plan: Plan to validate
            real_trades: Historical real trade data for comparison
            market_volume: Current market volume
            volatility: Market volatility
            
        Returns:
            Complete validation metrics
        """
        start_time = time.perf_counter()
        
        try:
            # Extract synthetic order data
            synthetic_sizes = [order.size for order in fragmentation_plan.child_orders]
            synthetic_timings = [order.target_time for order in fragmentation_plan.child_orders]
            synthetic_orders = list(zip(synthetic_sizes, synthetic_timings))
            
            # Extract real trade data for comparison
            real_sizes = [trade.size for trade in real_trades[-len(synthetic_sizes):]]
            real_timings = [trade.timestamp for trade in real_trades[-len(synthetic_sizes):]]
            real_orders = list(zip(real_sizes, real_timings))
            
            # Statistical indistinguishability tests
            size_test_results = self.indistinguishability_tester.test_size_distributions(
                real_sizes, synthetic_sizes
            )
            
            timing_test_results = self.indistinguishability_tester.test_timing_distributions(
                real_timings, synthetic_timings
            )
            
            joint_test_results = self.indistinguishability_tester.test_joint_distributions(
                real_orders, synthetic_orders
            )
            
            # Market impact analysis
            impact_results = self.impact_analyzer.analyze_impact_reduction(
                fragmentation_plan.total_size,
                fragmentation_plan,
                market_volume,
                volatility
            )
            
            # Compile validation metrics
            metrics = ValidationMetrics(
                # Statistical tests
                ks_test_p_value=size_test_results.get('ks_p_value', 0.0),
                anderson_darling_p_value=size_test_results.get('anderson_p_value', 0.0),
                chi_square_p_value=joint_test_results.get('chi2_p_value', 0.0),
                
                # Market impact
                naive_impact_bps=impact_results.get('naive_impact_bps', 0.0),
                stealth_impact_bps=impact_results.get('stealth_impact_bps', 0.0),
                impact_reduction_pct=impact_results.get('impact_reduction_pct', 0.0),
                
                # Timing analysis
                autocorr_similarity=timing_test_results.get('autocorr_similarity', 0.0),
                clustering_similarity=timing_test_results.get('clustering_similarity', 0.0),
                
                # Performance
                validation_time_ms=(time.perf_counter() - start_time) * 1000,
                sample_size=len(synthetic_orders),
                confidence_level=self.confidence_level
            )
            
            # Calculate composite scores
            indist_scores = [
                metrics.ks_test_p_value / 0.05,  # Normalize by threshold
                metrics.anderson_darling_p_value / 0.05,
                metrics.chi_square_p_value / 0.05
            ]
            metrics.indistinguishability_score = min(np.mean([s for s in indist_scores if s > 0]), 1.0)
            
            timing_scores = [metrics.autocorr_similarity, metrics.clustering_similarity]
            metrics.timing_naturalness_score = np.mean([s for s in timing_scores if s > 0])
            
            # Store for analytics
            self.validation_history.append(metrics)
            
            logger.info("Fragmentation plan validated",
                       plan_id=fragmentation_plan.parent_order_id,
                       indistinguishability_score=metrics.indistinguishability_score,
                       impact_reduction_pct=metrics.impact_reduction_pct,
                       validation_time_ms=metrics.validation_time_ms)
            
            return metrics
            
        except Exception as e:
            logger.error("Validation failed", error=str(e))
            # Return default metrics on error
            return ValidationMetrics(
                validation_time_ms=(time.perf_counter() - start_time) * 1000,
                sample_size=0
            )
    
    def batch_validate_strategies(self,
                                plans: List[FragmentationPlan],
                                real_trades: List[TradeRecord],
                                market_volume: float,
                                volatility: float) -> Dict[str, ValidationMetrics]:
        """Validate multiple fragmentation plans"""
        results = {}
        
        for plan in plans:
            try:
                metrics = self.validate_fragmentation_plan(
                    plan, real_trades, market_volume, volatility
                )
                results[plan.parent_order_id] = metrics
            except Exception as e:
                logger.error("Batch validation failed for plan", 
                           plan_id=plan.parent_order_id,
                           error=str(e))
        
        return results
    
    def generate_validation_report(self, 
                                 output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        if not self.validation_history:
            return {'error': 'No validation data available'}
        
        # Aggregate statistics
        metrics_list = self.validation_history
        
        report = {
            'summary': {
                'total_validations': len(metrics_list),
                'average_indistinguishability_score': np.mean([m.indistinguishability_score for m in metrics_list]),
                'average_impact_reduction_pct': np.mean([m.impact_reduction_pct for m in metrics_list]),
                'average_validation_time_ms': np.mean([m.validation_time_ms for m in metrics_list]),
                'validation_pass_rate': np.mean([m.passes_validation() for m in metrics_list])
            },
            'detailed_metrics': [m.to_dict() for m in metrics_list],
            'timestamp': datetime.now().isoformat()
        }
        
        # Save report if path provided
        if output_path:
            try:
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info("Validation report saved", path=output_path)
            except Exception as e:
                logger.error("Failed to save validation report", error=str(e))
        
        return report
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get performance analytics for the validation system"""
        if not self.validation_history:
            return {}
        
        metrics_list = self.validation_history
        
        analytics = {
            'validation_performance': {
                'total_validations': len(metrics_list),
                'average_time_ms': np.mean([m.validation_time_ms for m in metrics_list]),
                'max_time_ms': max([m.validation_time_ms for m in metrics_list]),
                'min_time_ms': min([m.validation_time_ms for m in metrics_list])
            },
            'effectiveness_metrics': {
                'indistinguishability_distribution': {
                    'mean': np.mean([m.indistinguishability_score for m in metrics_list]),
                    'std': np.std([m.indistinguishability_score for m in metrics_list]),
                    'min': min([m.indistinguishability_score for m in metrics_list]),
                    'max': max([m.indistinguishability_score for m in metrics_list])
                },
                'impact_reduction_distribution': {
                    'mean': np.mean([m.impact_reduction_pct for m in metrics_list]),
                    'std': np.std([m.impact_reduction_pct for m in metrics_list]),
                    'min': min([m.impact_reduction_pct for m in metrics_list]),
                    'max': max([m.impact_reduction_pct for m in metrics_list])
                }
            },
            'validation_thresholds': {
                'pass_rate_overall': np.mean([m.passes_validation() for m in metrics_list]),
                'indistinguishability_pass_rate': np.mean([m.indistinguishability_score > 1.0 for m in metrics_list]),
                'impact_reduction_pass_rate': np.mean([m.impact_reduction_pct > 0.8 for m in metrics_list])
            }
        }
        
        return analytics


# Export classes and functions
__all__ = [
    'StealthExecutionValidator',
    'ValidationMetrics',
    'StatisticalIndistinguishabilityTester',
    'MarketImpactAnalyzer',
    'DetectionProbabilityEstimator'
]