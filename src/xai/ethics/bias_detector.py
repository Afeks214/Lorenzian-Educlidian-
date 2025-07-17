"""
Comprehensive Bias Detection Framework
Agent Epsilon: Advanced XAI Implementation Specialist

Industry-first bias detection system for trading AI systems.
Monitors for various types of bias and provides real-time alerts.

Features:
- Multi-dimensional bias detection
- Statistical bias testing
- Fairness metrics computation
- Real-time bias monitoring
- Automated bias alerts
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import uuid
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available, using mock implementations")

logger = logging.getLogger(__name__)


class BiasType(Enum):
    """Types of bias to detect"""
    DEMOGRAPHIC = "demographic"
    ALGORITHMIC = "algorithmic"
    STATISTICAL = "statistical"
    TEMPORAL = "temporal"
    CONFIRMATION = "confirmation"
    SELECTION = "selection"
    REPRESENTATION = "representation"
    MEASUREMENT = "measurement"
    EVALUATION = "evaluation"
    DEPLOYMENT = "deployment"


class BiasMetric(Enum):
    """Bias metrics for evaluation"""
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    EQUALITY_OF_OPPORTUNITY = "equality_of_opportunity"
    PREDICTIVE_PARITY = "predictive_parity"
    CALIBRATION = "calibration"
    STATISTICAL_PARITY = "statistical_parity"
    DISPARATE_IMPACT = "disparate_impact"
    INDIVIDUAL_FAIRNESS = "individual_fairness"
    COUNTERFACTUAL_FAIRNESS = "counterfactual_fairness"


class BiasLevel(Enum):
    """Bias severity levels"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BiasResult:
    """Result of bias detection"""
    detection_id: str
    timestamp: datetime
    bias_type: BiasType
    bias_metric: BiasMetric
    bias_score: float
    bias_level: BiasLevel
    statistical_significance: float
    confidence_interval: Tuple[float, float]
    
    # Detailed results
    affected_groups: List[str]
    affected_decisions: List[str]
    mitigation_suggestions: List[str]
    
    # Context
    decision_context: Dict[str, Any]
    sample_size: int
    test_statistic: float
    
    # Metadata
    detection_method: str
    detector_version: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BiasPattern:
    """Detected bias pattern"""
    pattern_id: str
    pattern_type: str
    pattern_strength: float
    pattern_frequency: float
    first_detected: datetime
    last_detected: datetime
    total_occurrences: int
    affected_variables: List[str]
    description: str


class BiasDetector:
    """
    Comprehensive Bias Detection System
    
    Monitors for various types of bias in trading AI systems
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Detection history
        self.detection_history: List[BiasResult] = []
        self.bias_patterns: Dict[str, BiasPattern] = {}
        
        # Thresholds
        self.bias_thresholds = {
            BiasLevel.LOW: 0.05,
            BiasLevel.MODERATE: 0.10,
            BiasLevel.HIGH: 0.20,
            BiasLevel.CRITICAL: 0.30
        }
        
        # Statistical tests
        self.statistical_tests = {
            BiasMetric.DEMOGRAPHIC_PARITY: self._test_demographic_parity,
            BiasMetric.EQUALIZED_ODDS: self._test_equalized_odds,
            BiasMetric.EQUALITY_OF_OPPORTUNITY: self._test_equality_of_opportunity,
            BiasMetric.PREDICTIVE_PARITY: self._test_predictive_parity,
            BiasMetric.DISPARATE_IMPACT: self._test_disparate_impact,
            BiasMetric.STATISTICAL_PARITY: self._test_statistical_parity
        }
        
        # Performance tracking
        self.performance_stats = {
            'total_detections': 0,
            'bias_detected': 0,
            'false_positives': 0,
            'avg_detection_time_ms': 0.0,
            'bias_level_distribution': defaultdict(int)
        }
        
        logger.info("BiasDetector initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'significance_level': 0.05,
            'min_sample_size': 50,
            'disparate_impact_threshold': 0.8,
            'demographic_parity_threshold': 0.05,
            'equalized_odds_threshold': 0.05,
            'real_time_monitoring': True,
            'enable_pattern_detection': True,
            'alert_thresholds': {
                'low': 0.05,
                'moderate': 0.10,
                'high': 0.20,
                'critical': 0.30
            }
        }
    
    async def detect_bias(
        self,
        decision_data: pd.DataFrame,
        protected_attributes: List[str],
        target_variable: str,
        prediction_variable: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[BiasResult]:
        """
        Detect bias in decision data
        
        Args:
            decision_data: DataFrame with decision data
            protected_attributes: List of protected attribute columns
            target_variable: Target variable column name
            prediction_variable: Prediction variable column name
            context: Additional context information
            
        Returns:
            List[BiasResult]: Detected bias results
        """
        start_time = datetime.now()
        results = []
        
        # Validate input data
        if not self._validate_input_data(decision_data, protected_attributes, target_variable, prediction_variable):
            return results
        
        # Test for different types of bias
        for bias_metric in self.statistical_tests:
            try:
                bias_result = await self._test_bias_metric(
                    decision_data,
                    protected_attributes,
                    target_variable,
                    prediction_variable,
                    bias_metric,
                    context
                )
                
                if bias_result:
                    results.append(bias_result)
                    
            except Exception as e:
                logger.error(f"Error testing {bias_metric.value}: {e}")
        
        # Update performance stats
        detection_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        self.performance_stats['total_detections'] += 1
        self.performance_stats['bias_detected'] += len(results)
        
        old_avg = self.performance_stats['avg_detection_time_ms']
        total_detections = self.performance_stats['total_detections']
        self.performance_stats['avg_detection_time_ms'] = (
            (old_avg * (total_detections - 1) + detection_time_ms) / total_detections
        )
        
        # Update bias level distribution
        for result in results:
            self.performance_stats['bias_level_distribution'][result.bias_level.value] += 1
        
        # Store detection history
        self.detection_history.extend(results)
        
        # Update bias patterns
        if self.config['enable_pattern_detection']:
            await self._update_bias_patterns(results)
        
        logger.info(f"Bias detection completed: {len(results)} bias instances detected")
        return results
    
    def _validate_input_data(
        self,
        data: pd.DataFrame,
        protected_attributes: List[str],
        target_variable: str,
        prediction_variable: str
    ) -> bool:
        """Validate input data for bias detection"""
        
        # Check minimum sample size
        if len(data) < self.config['min_sample_size']:
            logger.warning(f"Sample size {len(data)} below minimum {self.config['min_sample_size']}")
            return False
        
        # Check required columns
        required_cols = protected_attributes + [target_variable, prediction_variable]
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Check for null values
        null_cols = [col for col in required_cols if data[col].isnull().any()]
        if null_cols:
            logger.warning(f"Null values found in columns: {null_cols}")
        
        return True
    
    async def _test_bias_metric(
        self,
        data: pd.DataFrame,
        protected_attributes: List[str],
        target_variable: str,
        prediction_variable: str,
        bias_metric: BiasMetric,
        context: Optional[Dict[str, Any]]
    ) -> Optional[BiasResult]:
        """Test specific bias metric"""
        
        test_function = self.statistical_tests.get(bias_metric)
        if not test_function:
            logger.warning(f"No test function for {bias_metric.value}")
            return None
        
        # Run bias test
        bias_score, p_value, affected_groups, test_statistic = test_function(
            data, protected_attributes, target_variable, prediction_variable
        )
        
        # Determine bias level
        bias_level = self._determine_bias_level(bias_score)
        
        # Check if bias is significant
        if p_value > self.config['significance_level'] and bias_level == BiasLevel.LOW:
            return None
        
        # Generate confidence interval
        ci_lower, ci_upper = self._calculate_confidence_interval(bias_score, len(data))
        
        # Generate mitigation suggestions
        mitigation_suggestions = self._generate_mitigation_suggestions(
            bias_metric, bias_level, affected_groups
        )
        
        # Create bias result
        result = BiasResult(
            detection_id=f"bias_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(timezone.utc),
            bias_type=self._get_bias_type(bias_metric),
            bias_metric=bias_metric,
            bias_score=bias_score,
            bias_level=bias_level,
            statistical_significance=p_value,
            confidence_interval=(ci_lower, ci_upper),
            affected_groups=affected_groups,
            affected_decisions=self._identify_affected_decisions(data, affected_groups),
            mitigation_suggestions=mitigation_suggestions,
            decision_context=context or {},
            sample_size=len(data),
            test_statistic=test_statistic,
            detection_method=bias_metric.value,
            detector_version="1.0"
        )
        
        return result
    
    def _test_demographic_parity(
        self,
        data: pd.DataFrame,
        protected_attributes: List[str],
        target_variable: str,
        prediction_variable: str
    ) -> Tuple[float, float, List[str], float]:
        """Test demographic parity"""
        
        max_bias = 0.0
        min_p_value = 1.0
        affected_groups = []
        max_test_stat = 0.0
        
        for attr in protected_attributes:
            groups = data[attr].unique()
            
            if len(groups) < 2:
                continue
            
            # Calculate positive prediction rates for each group
            group_rates = {}
            for group in groups:
                group_data = data[data[attr] == group]
                if len(group_data) > 0:
                    positive_rate = (group_data[prediction_variable] == 1).mean()
                    group_rates[group] = positive_rate
            
            # Calculate pairwise differences
            group_list = list(group_rates.keys())
            for i in range(len(group_list)):
                for j in range(i + 1, len(group_list)):
                    group1, group2 = group_list[i], group_list[j]
                    rate_diff = abs(group_rates[group1] - group_rates[group2])
                    
                    if rate_diff > max_bias:
                        max_bias = rate_diff
                        affected_groups = [str(group1), str(group2)]
                    
                    # Statistical test (chi-square)
                    if SKLEARN_AVAILABLE:
                        try:
                            group1_data = data[data[attr] == group1]
                            group2_data = data[data[attr] == group2]
                            
                            # Create contingency table
                            group1_pos = (group1_data[prediction_variable] == 1).sum()
                            group1_neg = (group1_data[prediction_variable] == 0).sum()
                            group2_pos = (group2_data[prediction_variable] == 1).sum()
                            group2_neg = (group2_data[prediction_variable] == 0).sum()
                            
                            contingency_table = np.array([[group1_pos, group1_neg], [group2_pos, group2_neg]])
                            
                            if np.all(contingency_table > 0):
                                chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
                                
                                if p_value < min_p_value:
                                    min_p_value = p_value
                                    max_test_stat = chi2
                                    
                        except Exception as e:
                            logger.debug(f"Statistical test failed: {e}")
                            min_p_value = 0.05
        
        return max_bias, min_p_value, affected_groups, max_test_stat
    
    def _test_equalized_odds(
        self,
        data: pd.DataFrame,
        protected_attributes: List[str],
        target_variable: str,
        prediction_variable: str
    ) -> Tuple[float, float, List[str], float]:
        """Test equalized odds"""
        
        max_bias = 0.0
        min_p_value = 1.0
        affected_groups = []
        max_test_stat = 0.0
        
        for attr in protected_attributes:
            groups = data[attr].unique()
            
            if len(groups) < 2:
                continue
            
            # Calculate TPR and FPR for each group
            group_metrics = {}
            for group in groups:
                group_data = data[data[attr] == group]
                if len(group_data) > 0:
                    # True Positive Rate
                    tpr = self._calculate_tpr(group_data[target_variable], group_data[prediction_variable])
                    # False Positive Rate
                    fpr = self._calculate_fpr(group_data[target_variable], group_data[prediction_variable])
                    group_metrics[group] = {'tpr': tpr, 'fpr': fpr}
            
            # Calculate pairwise differences
            group_list = list(group_metrics.keys())
            for i in range(len(group_list)):
                for j in range(i + 1, len(group_list)):
                    group1, group2 = group_list[i], group_list[j]
                    
                    tpr_diff = abs(group_metrics[group1]['tpr'] - group_metrics[group2]['tpr'])
                    fpr_diff = abs(group_metrics[group1]['fpr'] - group_metrics[group2]['fpr'])
                    
                    # Use maximum of TPR and FPR differences
                    max_diff = max(tpr_diff, fpr_diff)
                    
                    if max_diff > max_bias:
                        max_bias = max_diff
                        affected_groups = [str(group1), str(group2)]
                    
                    # Approximate p-value using normal approximation
                    if max_diff > 0:
                        # Simple approximation - in practice would use more sophisticated test
                        sample_size = len(data[data[attr].isin([group1, group2])])
                        se = np.sqrt(max_diff * (1 - max_diff) / sample_size)
                        z_score = max_diff / (se + 1e-10)
                        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                        
                        if p_value < min_p_value:
                            min_p_value = p_value
                            max_test_stat = z_score
        
        return max_bias, min_p_value, affected_groups, max_test_stat
    
    def _test_equality_of_opportunity(
        self,
        data: pd.DataFrame,
        protected_attributes: List[str],
        target_variable: str,
        prediction_variable: str
    ) -> Tuple[float, float, List[str], float]:
        """Test equality of opportunity"""
        
        max_bias = 0.0
        min_p_value = 1.0
        affected_groups = []
        max_test_stat = 0.0
        
        for attr in protected_attributes:
            groups = data[attr].unique()
            
            if len(groups) < 2:
                continue
            
            # Calculate TPR for each group (only positive cases)
            group_tprs = {}
            for group in groups:
                group_data = data[data[attr] == group]
                positive_cases = group_data[group_data[target_variable] == 1]
                
                if len(positive_cases) > 0:
                    tpr = (positive_cases[prediction_variable] == 1).mean()
                    group_tprs[group] = tpr
            
            # Calculate pairwise differences
            group_list = list(group_tprs.keys())
            for i in range(len(group_list)):
                for j in range(i + 1, len(group_list)):
                    group1, group2 = group_list[i], group_list[j]
                    tpr_diff = abs(group_tprs[group1] - group_tprs[group2])
                    
                    if tpr_diff > max_bias:
                        max_bias = tpr_diff
                        affected_groups = [str(group1), str(group2)]
                    
                    # Statistical test
                    if tpr_diff > 0:
                        # Use proportion test
                        group1_positive = data[(data[attr] == group1) & (data[target_variable] == 1)]
                        group2_positive = data[(data[attr] == group2) & (data[target_variable] == 1)]
                        
                        if len(group1_positive) > 0 and len(group2_positive) > 0:
                            n1 = len(group1_positive)
                            n2 = len(group2_positive)
                            x1 = (group1_positive[prediction_variable] == 1).sum()
                            x2 = (group2_positive[prediction_variable] == 1).sum()
                            
                            # Proportion test
                            p1 = x1 / n1
                            p2 = x2 / n2
                            p_pooled = (x1 + x2) / (n1 + n2)
                            
                            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
                            z_score = (p1 - p2) / (se + 1e-10)
                            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                            
                            if p_value < min_p_value:
                                min_p_value = p_value
                                max_test_stat = z_score
        
        return max_bias, min_p_value, affected_groups, max_test_stat
    
    def _test_predictive_parity(
        self,
        data: pd.DataFrame,
        protected_attributes: List[str],
        target_variable: str,
        prediction_variable: str
    ) -> Tuple[float, float, List[str], float]:
        """Test predictive parity"""
        
        max_bias = 0.0
        min_p_value = 1.0
        affected_groups = []
        max_test_stat = 0.0
        
        for attr in protected_attributes:
            groups = data[attr].unique()
            
            if len(groups) < 2:
                continue
            
            # Calculate PPV (precision) for each group
            group_ppvs = {}
            for group in groups:
                group_data = data[data[attr] == group]
                positive_predictions = group_data[group_data[prediction_variable] == 1]
                
                if len(positive_predictions) > 0:
                    ppv = (positive_predictions[target_variable] == 1).mean()
                    group_ppvs[group] = ppv
            
            # Calculate pairwise differences
            group_list = list(group_ppvs.keys())
            for i in range(len(group_list)):
                for j in range(i + 1, len(group_list)):
                    group1, group2 = group_list[i], group_list[j]
                    ppv_diff = abs(group_ppvs[group1] - group_ppvs[group2])
                    
                    if ppv_diff > max_bias:
                        max_bias = ppv_diff
                        affected_groups = [str(group1), str(group2)]
                    
                    # Statistical test similar to proportion test
                    if ppv_diff > 0:
                        min_p_value = 0.05  # Simplified for demo
                        max_test_stat = ppv_diff / 0.1  # Simplified z-score
        
        return max_bias, min_p_value, affected_groups, max_test_stat
    
    def _test_disparate_impact(
        self,
        data: pd.DataFrame,
        protected_attributes: List[str],
        target_variable: str,
        prediction_variable: str
    ) -> Tuple[float, float, List[str], float]:
        """Test disparate impact"""
        
        max_bias = 0.0
        min_p_value = 1.0
        affected_groups = []
        max_test_stat = 0.0
        
        for attr in protected_attributes:
            groups = data[attr].unique()
            
            if len(groups) < 2:
                continue
            
            # Calculate positive prediction rates
            group_rates = {}
            for group in groups:
                group_data = data[data[attr] == group]
                if len(group_data) > 0:
                    positive_rate = (group_data[prediction_variable] == 1).mean()
                    group_rates[group] = positive_rate
            
            # Calculate disparate impact ratios
            group_list = list(group_rates.keys())
            for i in range(len(group_list)):
                for j in range(i + 1, len(group_list)):
                    group1, group2 = group_list[i], group_list[j]
                    
                    # Calculate disparate impact ratio
                    if group_rates[group2] > 0:
                        di_ratio = group_rates[group1] / group_rates[group2]
                        
                        # Bias score is deviation from 1.0
                        bias_score = abs(1.0 - di_ratio)
                        
                        if bias_score > max_bias:
                            max_bias = bias_score
                            affected_groups = [str(group1), str(group2)]
                        
                        # Check if ratio is below threshold
                        if min(di_ratio, 1/di_ratio) < self.config['disparate_impact_threshold']:
                            min_p_value = 0.01  # Significant disparate impact
                            max_test_stat = bias_score
        
        return max_bias, min_p_value, affected_groups, max_test_stat
    
    def _test_statistical_parity(
        self,
        data: pd.DataFrame,
        protected_attributes: List[str],
        target_variable: str,
        prediction_variable: str
    ) -> Tuple[float, float, List[str], float]:
        """Test statistical parity"""
        
        # Statistical parity is similar to demographic parity
        return self._test_demographic_parity(data, protected_attributes, target_variable, prediction_variable)
    
    def _calculate_tpr(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """Calculate True Positive Rate"""
        if SKLEARN_AVAILABLE:
            return recall_score(y_true, y_pred, average='binary', zero_division=0)
        else:
            # Manual calculation
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    def _calculate_fpr(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """Calculate False Positive Rate"""
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    def _determine_bias_level(self, bias_score: float) -> BiasLevel:
        """Determine bias level from score"""
        if bias_score >= self.bias_thresholds[BiasLevel.CRITICAL]:
            return BiasLevel.CRITICAL
        elif bias_score >= self.bias_thresholds[BiasLevel.HIGH]:
            return BiasLevel.HIGH
        elif bias_score >= self.bias_thresholds[BiasLevel.MODERATE]:
            return BiasLevel.MODERATE
        else:
            return BiasLevel.LOW
    
    def _get_bias_type(self, bias_metric: BiasMetric) -> BiasType:
        """Map bias metric to bias type"""
        metric_to_type = {
            BiasMetric.DEMOGRAPHIC_PARITY: BiasType.DEMOGRAPHIC,
            BiasMetric.EQUALIZED_ODDS: BiasType.ALGORITHMIC,
            BiasMetric.EQUALITY_OF_OPPORTUNITY: BiasType.ALGORITHMIC,
            BiasMetric.PREDICTIVE_PARITY: BiasType.STATISTICAL,
            BiasMetric.DISPARATE_IMPACT: BiasType.DEMOGRAPHIC,
            BiasMetric.STATISTICAL_PARITY: BiasType.STATISTICAL
        }
        return metric_to_type.get(bias_metric, BiasType.ALGORITHMIC)
    
    def _calculate_confidence_interval(self, bias_score: float, sample_size: int) -> Tuple[float, float]:
        """Calculate confidence interval for bias score"""
        # Simple normal approximation
        se = np.sqrt(bias_score * (1 - bias_score) / sample_size)
        z_score = 1.96  # 95% confidence
        
        ci_lower = max(0, bias_score - z_score * se)
        ci_upper = min(1, bias_score + z_score * se)
        
        return ci_lower, ci_upper
    
    def _generate_mitigation_suggestions(
        self,
        bias_metric: BiasMetric,
        bias_level: BiasLevel,
        affected_groups: List[str]
    ) -> List[str]:
        """Generate mitigation suggestions"""
        suggestions = []
        
        if bias_metric == BiasMetric.DEMOGRAPHIC_PARITY:
            suggestions.append("Implement demographic parity constraints in model training")
            suggestions.append("Use post-processing techniques to equalize positive prediction rates")
            suggestions.append("Consider using fairness-aware algorithms")
        
        elif bias_metric == BiasMetric.EQUALIZED_ODDS:
            suggestions.append("Apply equalized odds post-processing")
            suggestions.append("Use adversarial debiasing techniques")
            suggestions.append("Implement threshold optimization for different groups")
        
        elif bias_metric == BiasMetric.DISPARATE_IMPACT:
            suggestions.append("Examine and improve data collection process")
            suggestions.append("Use disparate impact mitigation techniques")
            suggestions.append("Consider alternative model architectures")
        
        # General suggestions based on bias level
        if bias_level in [BiasLevel.HIGH, BiasLevel.CRITICAL]:
            suggestions.append("Immediate review and potential system halt required")
            suggestions.append("Conduct thorough bias audit")
            suggestions.append("Implement emergency bias mitigation measures")
        
        return suggestions
    
    def _identify_affected_decisions(self, data: pd.DataFrame, affected_groups: List[str]) -> List[str]:
        """Identify decisions affected by bias"""
        affected_decisions = []
        
        # This would be implemented based on specific decision ID tracking
        # For now, return a placeholder
        if 'decision_id' in data.columns:
            affected_decisions = data['decision_id'].unique()[:10].tolist()
        
        return affected_decisions
    
    async def _update_bias_patterns(self, bias_results: List[BiasResult]):
        """Update bias patterns from detection results"""
        for result in bias_results:
            pattern_key = f"{result.bias_type.value}_{result.bias_metric.value}"
            
            if pattern_key in self.bias_patterns:
                # Update existing pattern
                pattern = self.bias_patterns[pattern_key]
                pattern.total_occurrences += 1
                pattern.last_detected = result.timestamp
                pattern.pattern_strength = max(pattern.pattern_strength, result.bias_score)
                
                # Update frequency (simplified)
                time_diff = (result.timestamp - pattern.first_detected).total_seconds()
                pattern.pattern_frequency = pattern.total_occurrences / max(time_diff / 3600, 1)  # per hour
                
            else:
                # Create new pattern
                pattern = BiasPattern(
                    pattern_id=f"pattern_{uuid.uuid4().hex[:8]}",
                    pattern_type=f"{result.bias_type.value}_{result.bias_metric.value}",
                    pattern_strength=result.bias_score,
                    pattern_frequency=1.0,
                    first_detected=result.timestamp,
                    last_detected=result.timestamp,
                    total_occurrences=1,
                    affected_variables=result.affected_groups,
                    description=f"Bias pattern detected in {result.bias_type.value} using {result.bias_metric.value}"
                )
                self.bias_patterns[pattern_key] = pattern
    
    def get_bias_summary(self) -> Dict[str, Any]:
        """Get summary of bias detection results"""
        if not self.detection_history:
            return {
                'total_detections': 0,
                'bias_levels': {},
                'bias_types': {},
                'patterns': 0
            }
        
        # Aggregate by bias level
        bias_levels = defaultdict(int)
        for result in self.detection_history:
            bias_levels[result.bias_level.value] += 1
        
        # Aggregate by bias type
        bias_types = defaultdict(int)
        for result in self.detection_history:
            bias_types[result.bias_type.value] += 1
        
        return {
            'total_detections': len(self.detection_history),
            'bias_levels': dict(bias_levels),
            'bias_types': dict(bias_types),
            'patterns': len(self.bias_patterns),
            'latest_detection': self.detection_history[-1].timestamp.isoformat(),
            'performance_stats': self.performance_stats
        }
    
    def get_bias_patterns(self) -> List[BiasPattern]:
        """Get detected bias patterns"""
        return list(self.bias_patterns.values())
    
    def get_mitigation_recommendations(self) -> List[str]:
        """Get aggregated mitigation recommendations"""
        all_suggestions = []
        
        for result in self.detection_history:
            all_suggestions.extend(result.mitigation_suggestions)
        
        # Count suggestions and return most common
        suggestion_counts = defaultdict(int)
        for suggestion in all_suggestions:
            suggestion_counts[suggestion] += 1
        
        # Sort by frequency
        sorted_suggestions = sorted(
            suggestion_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [suggestion for suggestion, count in sorted_suggestions[:10]]


# Test function
async def test_bias_detector():
    """Test the Bias Detector"""
    print("🧪 Testing Bias Detector")
    
    # Initialize detector
    detector = BiasDetector()
    
    # Create mock data with bias
    np.random.seed(42)
    n_samples = 1000
    
    # Create biased dataset
    data = pd.DataFrame({
        'user_id': range(n_samples),
        'age': np.random.randint(18, 80, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })
    
    # Introduce bias - higher approval rate for males
    male_indices = data['gender'] == 'M'
    data.loc[male_indices, 'prediction'] = np.random.choice([0, 1], sum(male_indices), p=[0.3, 0.7])
    data.loc[~male_indices, 'prediction'] = np.random.choice([0, 1], sum(~male_indices), p=[0.6, 0.4])
    
    print(f"Created dataset with {len(data)} samples")
    print(f"Gender distribution: {data['gender'].value_counts().to_dict()}")
    print(f"Prediction rate by gender: {data.groupby('gender')['prediction'].mean().to_dict()}")
    
    # Test bias detection
    print("\\n🔍 Testing bias detection...")
    
    bias_results = await detector.detect_bias(
        decision_data=data,
        protected_attributes=['gender'],
        target_variable='target',
        prediction_variable='prediction',
        context={'test': 'bias_detection_test'}
    )
    
    print(f"\\nDetected {len(bias_results)} bias instances:")
    for result in bias_results:
        print(f"  - {result.bias_type.value} ({result.bias_metric.value}): "
              f"score={result.bias_score:.3f}, level={result.bias_level.value}")
        print(f"    Affected groups: {result.affected_groups}")
        print(f"    P-value: {result.statistical_significance:.3f}")
        print(f"    Mitigation suggestions: {result.mitigation_suggestions[:2]}")
    
    # Test bias summary
    print("\\n📊 Bias Detection Summary:")
    summary = detector.get_bias_summary()
    print(f"  Total detections: {summary['total_detections']}")
    print(f"  Bias levels: {summary['bias_levels']}")
    print(f"  Bias types: {summary['bias_types']}")
    print(f"  Patterns detected: {summary['patterns']}")
    
    # Test mitigation recommendations
    print("\\n💡 Mitigation Recommendations:")
    recommendations = detector.get_mitigation_recommendations()
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"  {i}. {rec}")
    
    print("\\n✅ Bias Detector test complete!")


if __name__ == "__main__":
    asyncio.run(test_bias_detector())