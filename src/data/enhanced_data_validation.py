"""
Enhanced Data Validation Pipeline
Agent 5: Data Quality & Bias Elimination

Comprehensive data validation pipeline with temporal consistency checks,
bias detection, and quality scoring. Integrates with existing validators
to provide end-to-end data quality assurance.

Key Features:
- Comprehensive data validation pipeline
- Temporal consistency validation
- Look-ahead bias detection integration
- Data quality scoring and monitoring
- Missing data handling and interpolation
- Multi-timeframe validation
- Real-time quality alerts
"""

import asyncio
import threading
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import uuid
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import structlog

from .validators import ValidationResult, BaseValidator, TickValidator, BarValidator, DataQualityMonitor
from .temporal_bias_detector import TemporalBiasDetector, BiasDetectionResult, DataAvailabilityRecord
from .data_handler import TickData
from .bar_generator import BarData

logger = structlog.get_logger(__name__)

# =============================================================================
# ENUMERATIONS AND CONSTANTS
# =============================================================================

class ValidationLevel(str, Enum):
    """Data validation levels"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    STRICT = "strict"

class DataQualityGrade(str, Enum):
    """Data quality grades"""
    EXCELLENT = "excellent"  # >= 0.95
    GOOD = "good"           # >= 0.85
    FAIR = "fair"           # >= 0.70
    POOR = "poor"           # >= 0.50
    CRITICAL = "critical"   # < 0.50

class InterpolationMethod(str, Enum):
    """Data interpolation methods"""
    LINEAR = "linear"
    CUBIC = "cubic"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    SPLINE = "spline"
    KALMAN = "kalman"

class ValidationCategory(str, Enum):
    """Validation categories"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    TEMPORAL_INTEGRITY = "temporal_integrity"

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DataQualityScore:
    """Comprehensive data quality score"""
    overall_score: float
    component_scores: Dict[str, float] = field(default_factory=dict)
    quality_grade: DataQualityGrade = DataQualityGrade.FAIR
    
    # Detailed metrics
    completeness_score: float = 0.0
    accuracy_score: float = 0.0
    consistency_score: float = 0.0
    timeliness_score: float = 0.0
    validity_score: float = 0.0
    temporal_integrity_score: float = 0.0
    
    # Metadata
    calculated_at: datetime = field(default_factory=datetime.utcnow)
    data_points_evaluated: int = 0
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    
    # Quality factors
    bias_penalty: float = 0.0
    missing_data_penalty: float = 0.0
    anomaly_penalty: float = 0.0
    
    # Recommendations
    improvement_recommendations: List[str] = field(default_factory=list)
    critical_issues: List[str] = field(default_factory=list)

@dataclass
class InterpolationResult:
    """Result of data interpolation"""
    interpolated_data: Any
    method_used: InterpolationMethod
    interpolated_points: int
    confidence_score: float
    
    # Quality metrics
    rmse: float = 0.0
    mae: float = 0.0
    r_squared: float = 0.0
    
    # Metadata
    interpolated_at: datetime = field(default_factory=datetime.utcnow)
    source_data_points: int = 0
    time_range: Tuple[datetime, datetime] = field(default_factory=lambda: (datetime.min, datetime.max))

@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    report_id: str
    validation_level: ValidationLevel
    data_quality_score: DataQualityScore
    
    # Validation results
    validation_results: List[ValidationResult] = field(default_factory=list)
    bias_detection_results: List[BiasDetectionResult] = field(default_factory=list)
    
    # Data statistics
    total_data_points: int = 0
    missing_data_points: int = 0
    interpolated_points: int = 0
    anomalous_points: int = 0
    
    # Time range
    validation_period: Tuple[datetime, datetime] = field(default_factory=lambda: (datetime.min, datetime.max))
    
    # Quality metrics by category
    category_scores: Dict[ValidationCategory, float] = field(default_factory=dict)
    
    # Recommendations
    immediate_actions: List[str] = field(default_factory=list)
    long_term_improvements: List[str] = field(default_factory=list)
    
    # Metadata
    generated_at: datetime = field(default_factory=datetime.utcnow)
    validator_version: str = "1.0"

# =============================================================================
# ENHANCED DATA VALIDATION PIPELINE
# =============================================================================

class EnhancedDataValidator:
    """Enhanced data validation pipeline with bias detection and quality scoring"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # Core validators
        self.tick_validator = TickValidator(self.config.get('tick_validation'))
        self.bar_validator = BarValidator(self.config.get('bar_validation'))
        self.quality_monitor = DataQualityMonitor()
        
        # Bias detection
        self.bias_detector = TemporalBiasDetector(self.config.get('bias_detection'))
        
        # Validation pipeline
        self.validation_pipeline = self._setup_validation_pipeline()
        
        # Data tracking
        self.validation_history: deque = deque(maxlen=1000)
        self.quality_scores: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Interpolation engines
        self.interpolation_methods = {
            InterpolationMethod.LINEAR: self._linear_interpolation,
            InterpolationMethod.CUBIC: self._cubic_interpolation,
            InterpolationMethod.FORWARD_FILL: self._forward_fill_interpolation,
            InterpolationMethod.BACKWARD_FILL: self._backward_fill_interpolation,
            InterpolationMethod.SPLINE: self._spline_interpolation,
            InterpolationMethod.KALMAN: self._kalman_interpolation,
        }
        
        # Background processing
        self.processing_executor = ThreadPoolExecutor(max_workers=4)
        
        # Statistics
        self.stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'interpolations_performed': 0,
            'biases_detected': 0,
            'quality_improvements': 0
        }
        
        logger.info("Enhanced data validation pipeline initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'validation_level': ValidationLevel.STANDARD,
            'enable_bias_detection': True,
            'enable_interpolation': True,
            'quality_threshold': 0.7,
            'interpolation_max_gap_minutes': 30,
            'temporal_consistency_checks': True,
            'real_time_monitoring': True,
            'auto_remediation': True,
            'quality_scoring': {
                'completeness_weight': 0.25,
                'accuracy_weight': 0.25,
                'consistency_weight': 0.20,
                'timeliness_weight': 0.15,
                'validity_weight': 0.10,
                'temporal_integrity_weight': 0.05
            }
        }
    
    def _setup_validation_pipeline(self) -> List[Callable]:
        """Setup validation pipeline stages"""
        
        pipeline = []
        
        # Stage 1: Basic validation
        pipeline.append(self._validate_basic_data)
        
        # Stage 2: Temporal consistency
        if self.config.get('temporal_consistency_checks', True):
            pipeline.append(self._validate_temporal_consistency)
        
        # Stage 3: Bias detection
        if self.config.get('enable_bias_detection', True):
            pipeline.append(self._detect_temporal_bias)
        
        # Stage 4: Quality scoring
        pipeline.append(self._calculate_quality_score)
        
        # Stage 5: Interpolation (if needed)
        if self.config.get('enable_interpolation', True):
            pipeline.append(self._handle_missing_data)
        
        # Stage 6: Final validation
        pipeline.append(self._final_validation)
        
        return pipeline
    
    async def validate_data(self, data: Union[TickData, BarData, List[Union[TickData, BarData]]], 
                          context: Optional[Dict[str, Any]] = None) -> ValidationReport:
        """Validate data through comprehensive pipeline"""
        
        context = context or {}
        report_id = str(uuid.uuid4())
        
        # Initialize report
        report = ValidationReport(
            report_id=report_id,
            validation_level=self.config['validation_level'],
            data_quality_score=DataQualityScore(overall_score=0.0)
        )
        
        # Normalize data to list
        if not isinstance(data, list):
            data = [data]
        
        report.total_data_points = len(data)
        
        # Set validation period
        if data:
            timestamps = [self._extract_timestamp(d) for d in data]
            report.validation_period = (min(timestamps), max(timestamps))
        
        # Run validation pipeline
        validation_context = {
            'report': report,
            'data': data,
            'context': context,
            'validator': self
        }
        
        try:
            for stage in self.validation_pipeline:
                await stage(validation_context)
            
            # Finalize report
            self._finalize_report(report)
            
            # Store in history
            self.validation_history.append(report)
            
            # Update statistics
            self._update_statistics(report)
            
            logger.info(f"Validation completed: {report_id}, score: {report.data_quality_score.overall_score:.3f}")
            
            return report
            
        except Exception as e:
            logger.error(f"Validation pipeline error: {e}")
            report.critical_issues.append(f"Pipeline error: {str(e)}")
            return report
    
    async def _validate_basic_data(self, context: Dict[str, Any]):
        """Stage 1: Basic data validation"""
        
        report = context['report']
        data = context['data']
        
        validation_results = []
        
        for data_point in data:
            if isinstance(data_point, TickData):
                result = self.tick_validator.validate(data_point)
                validation_results.append(result)
                
                # Update quality monitor
                self.quality_monitor.validate_tick(data_point)
                
            elif isinstance(data_point, BarData):
                result = self.bar_validator.validate(data_point)
                validation_results.append(result)
                
                # Update quality monitor
                self.quality_monitor.validate_bar(data_point)
            
            else:
                # Custom validation for other data types
                result = self._validate_custom_data(data_point)
                validation_results.append(result)
        
        report.validation_results = validation_results
        
        # Calculate basic metrics
        passed_validations = sum(1 for r in validation_results if r.is_valid)
        report.data_quality_score.validity_score = passed_validations / len(validation_results) if validation_results else 0.0
        
        logger.debug(f"Basic validation: {passed_validations}/{len(validation_results)} passed")
    
    async def _validate_temporal_consistency(self, context: Dict[str, Any]):
        """Stage 2: Temporal consistency validation"""
        
        report = context['report']
        data = context['data']
        
        if len(data) < 2:
            return
        
        # Extract timestamps
        timestamps = [self._extract_timestamp(d) for d in data]
        
        # Check temporal ordering
        temporal_issues = []
        
        for i in range(1, len(timestamps)):
            if timestamps[i] <= timestamps[i-1]:
                temporal_issues.append(f"Timestamp ordering violation at index {i}")
        
        # Check for gaps
        for i in range(1, len(timestamps)):
            gap = (timestamps[i] - timestamps[i-1]).total_seconds()
            max_gap = self.config.get('max_gap_seconds', 300)  # 5 minutes
            
            if gap > max_gap:
                temporal_issues.append(f"Large gap detected: {gap:.0f} seconds")
        
        # Update report
        if temporal_issues:
            report.critical_issues.extend(temporal_issues)
            report.data_quality_score.temporal_integrity_score = 0.5
        else:
            report.data_quality_score.temporal_integrity_score = 1.0
        
        logger.debug(f"Temporal consistency: {len(temporal_issues)} issues found")
    
    async def _detect_temporal_bias(self, context: Dict[str, Any]):
        """Stage 3: Temporal bias detection"""
        
        report = context['report']
        data = context['data']
        validator_context = context['context']
        
        # Prepare data for bias detection
        bias_data = {
            'timestamps': [self._extract_timestamp(d) for d in data],
            'values': [self._extract_value(d) for d in data],
            'data_requests': validator_context.get('data_requests', [])
        }
        
        # Detect bias
        bias_results = self.bias_detector.detect_bias(bias_data, validator_context)
        
        report.bias_detection_results = bias_results
        
        # Calculate bias penalty
        if bias_results:
            critical_biases = sum(1 for b in bias_results if b.severity_level.value == 'critical')
            high_biases = sum(1 for b in bias_results if b.severity_level.value == 'high')
            
            bias_penalty = (critical_biases * 0.5) + (high_biases * 0.2)
            report.data_quality_score.bias_penalty = min(bias_penalty, 1.0)
        
        logger.debug(f"Bias detection: {len(bias_results)} biases detected")
    
    async def _calculate_quality_score(self, context: Dict[str, Any]):
        """Stage 4: Calculate comprehensive quality score"""
        
        report = context['report']
        data = context['data']
        
        # Calculate component scores
        completeness_score = self._calculate_completeness_score(data)
        accuracy_score = self._calculate_accuracy_score(data, report.validation_results)
        consistency_score = self._calculate_consistency_score(data)
        timeliness_score = self._calculate_timeliness_score(data)
        
        # Update report
        report.data_quality_score.completeness_score = completeness_score
        report.data_quality_score.accuracy_score = accuracy_score
        report.data_quality_score.consistency_score = consistency_score
        report.data_quality_score.timeliness_score = timeliness_score
        
        # Calculate weighted overall score
        weights = self.config['quality_scoring']
        overall_score = (
            completeness_score * weights['completeness_weight'] +
            accuracy_score * weights['accuracy_weight'] +
            consistency_score * weights['consistency_weight'] +
            timeliness_score * weights['timeliness_weight'] +
            report.data_quality_score.validity_score * weights['validity_weight'] +
            report.data_quality_score.temporal_integrity_score * weights['temporal_integrity_weight']
        )
        
        # Apply penalties
        overall_score = max(0.0, overall_score - report.data_quality_score.bias_penalty)
        
        report.data_quality_score.overall_score = overall_score
        report.data_quality_score.quality_grade = self._get_quality_grade(overall_score)
        
        logger.debug(f"Quality score calculated: {overall_score:.3f}")
    
    async def _handle_missing_data(self, context: Dict[str, Any]):
        """Stage 5: Handle missing data through interpolation"""
        
        report = context['report']
        data = context['data']
        
        # Identify missing data
        missing_points = self._identify_missing_data(data)
        
        if missing_points:
            report.missing_data_points = len(missing_points)
            
            # Attempt interpolation
            interpolation_results = []
            
            for missing_point in missing_points:
                try:
                    result = await self._interpolate_missing_point(missing_point, data)
                    interpolation_results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Interpolation failed for point {missing_point}: {e}")
            
            # Update report
            successful_interpolations = sum(1 for r in interpolation_results if r.confidence_score > 0.7)
            report.interpolated_points = successful_interpolations
            
            # Calculate missing data penalty
            missing_ratio = len(missing_points) / len(data) if data else 0
            report.data_quality_score.missing_data_penalty = missing_ratio * 0.3
            
            logger.debug(f"Missing data handling: {successful_interpolations}/{len(missing_points)} interpolated")
    
    async def _final_validation(self, context: Dict[str, Any]):
        """Stage 6: Final validation and recommendations"""
        
        report = context['report']
        
        # Generate recommendations
        report.improvement_recommendations = self._generate_recommendations(report)
        
        # Identify immediate actions
        if report.data_quality_score.overall_score < 0.5:
            report.immediate_actions.append("Critical quality issues detected - immediate review required")
        
        if report.bias_detection_results:
            critical_biases = [b for b in report.bias_detection_results if b.severity_level.value == 'critical']
            if critical_biases:
                report.immediate_actions.append(f"Critical temporal biases detected: {len(critical_biases)}")
        
        # Calculate category scores
        for category in ValidationCategory:
            score = self._calculate_category_score(category, report)
            report.category_scores[category] = score
        
        logger.debug("Final validation completed")
    
    def _calculate_completeness_score(self, data: List[Any]) -> float:
        """Calculate data completeness score"""
        
        if not data:
            return 0.0
        
        # Count non-null values
        non_null_count = sum(1 for d in data if self._is_valid_data_point(d))
        
        return non_null_count / len(data)
    
    def _calculate_accuracy_score(self, data: List[Any], validation_results: List[ValidationResult]) -> float:
        """Calculate data accuracy score"""
        
        if not validation_results:
            return 1.0
        
        # Count valid results
        valid_count = sum(1 for r in validation_results if r.is_valid)
        
        return valid_count / len(validation_results)
    
    def _calculate_consistency_score(self, data: List[Any]) -> float:
        """Calculate data consistency score"""
        
        if len(data) < 2:
            return 1.0
        
        # Check for consistency patterns
        values = [self._extract_value(d) for d in data]
        
        # Calculate variation coefficient
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if mean_val > 0:
                cv = std_val / mean_val
                # Lower variation = higher consistency
                return max(0.0, 1.0 - cv)
        
        return 0.5
    
    def _calculate_timeliness_score(self, data: List[Any]) -> float:
        """Calculate data timeliness score"""
        
        if not data:
            return 1.0
        
        current_time = datetime.utcnow()
        
        # Calculate average age
        ages = []
        for d in data:
            timestamp = self._extract_timestamp(d)
            age = (current_time - timestamp).total_seconds()
            ages.append(age)
        
        if ages:
            avg_age = np.mean(ages)
            max_acceptable_age = 3600  # 1 hour
            
            if avg_age <= max_acceptable_age:
                return 1.0
            else:
                # Exponential decay
                return max(0.0, np.exp(-avg_age / (2 * max_acceptable_age)))
        
        return 1.0
    
    def _identify_missing_data(self, data: List[Any]) -> List[Dict[str, Any]]:
        """Identify missing data points"""
        
        if len(data) < 2:
            return []
        
        missing_points = []
        timestamps = [self._extract_timestamp(d) for d in data]
        
        # Find gaps
        for i in range(1, len(timestamps)):
            gap = (timestamps[i] - timestamps[i-1]).total_seconds()
            expected_interval = 60  # 1 minute default
            
            if gap > expected_interval * 2:  # Gap larger than 2 intervals
                missing_points.append({
                    'start_time': timestamps[i-1],
                    'end_time': timestamps[i],
                    'gap_seconds': gap,
                    'expected_points': int(gap / expected_interval) - 1
                })
        
        return missing_points
    
    async def _interpolate_missing_point(self, missing_point: Dict[str, Any], data: List[Any]) -> InterpolationResult:
        """Interpolate missing data point"""
        
        # Select interpolation method based on gap size
        gap_seconds = missing_point['gap_seconds']
        
        if gap_seconds <= 300:  # 5 minutes
            method = InterpolationMethod.LINEAR
        elif gap_seconds <= 1800:  # 30 minutes
            method = InterpolationMethod.CUBIC
        else:
            method = InterpolationMethod.FORWARD_FILL
        
        # Perform interpolation
        interpolation_func = self.interpolation_methods[method]
        
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                self.processing_executor,
                interpolation_func,
                missing_point,
                data
            )
            
            self.stats['interpolations_performed'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Interpolation error: {e}")
            return InterpolationResult(
                interpolated_data=None,
                method_used=method,
                interpolated_points=0,
                confidence_score=0.0
            )
    
    def _linear_interpolation(self, missing_point: Dict[str, Any], data: List[Any]) -> InterpolationResult:
        """Linear interpolation implementation"""
        
        # Simple linear interpolation
        start_time = missing_point['start_time']
        end_time = missing_point['end_time']
        
        # Find surrounding data points
        before_data = [d for d in data if self._extract_timestamp(d) <= start_time]
        after_data = [d for d in data if self._extract_timestamp(d) >= end_time]
        
        if before_data and after_data:
            start_value = self._extract_value(before_data[-1])
            end_value = self._extract_value(after_data[0])
            
            # Create interpolated value
            interpolated_value = (start_value + end_value) / 2
            
            return InterpolationResult(
                interpolated_data=interpolated_value,
                method_used=InterpolationMethod.LINEAR,
                interpolated_points=1,
                confidence_score=0.8
            )
        
        return InterpolationResult(
            interpolated_data=None,
            method_used=InterpolationMethod.LINEAR,
            interpolated_points=0,
            confidence_score=0.0
        )
    
    def _cubic_interpolation(self, missing_point: Dict[str, Any], data: List[Any]) -> InterpolationResult:
        """Cubic interpolation implementation"""
        # Placeholder for cubic interpolation
        return self._linear_interpolation(missing_point, data)
    
    def _forward_fill_interpolation(self, missing_point: Dict[str, Any], data: List[Any]) -> InterpolationResult:
        """Forward fill interpolation implementation"""
        
        start_time = missing_point['start_time']
        
        # Find last valid data point
        before_data = [d for d in data if self._extract_timestamp(d) <= start_time]
        
        if before_data:
            last_value = self._extract_value(before_data[-1])
            
            return InterpolationResult(
                interpolated_data=last_value,
                method_used=InterpolationMethod.FORWARD_FILL,
                interpolated_points=1,
                confidence_score=0.6
            )
        
        return InterpolationResult(
            interpolated_data=None,
            method_used=InterpolationMethod.FORWARD_FILL,
            interpolated_points=0,
            confidence_score=0.0
        )
    
    def _backward_fill_interpolation(self, missing_point: Dict[str, Any], data: List[Any]) -> InterpolationResult:
        """Backward fill interpolation implementation"""
        
        end_time = missing_point['end_time']
        
        # Find next valid data point
        after_data = [d for d in data if self._extract_timestamp(d) >= end_time]
        
        if after_data:
            next_value = self._extract_value(after_data[0])
            
            return InterpolationResult(
                interpolated_data=next_value,
                method_used=InterpolationMethod.BACKWARD_FILL,
                interpolated_points=1,
                confidence_score=0.6
            )
        
        return InterpolationResult(
            interpolated_data=None,
            method_used=InterpolationMethod.BACKWARD_FILL,
            interpolated_points=0,
            confidence_score=0.0
        )
    
    def _spline_interpolation(self, missing_point: Dict[str, Any], data: List[Any]) -> InterpolationResult:
        """Spline interpolation implementation"""
        # Placeholder for spline interpolation
        return self._linear_interpolation(missing_point, data)
    
    def _kalman_interpolation(self, missing_point: Dict[str, Any], data: List[Any]) -> InterpolationResult:
        """Kalman filter interpolation implementation"""
        # Placeholder for Kalman filter interpolation
        return self._linear_interpolation(missing_point, data)
    
    def _validate_custom_data(self, data_point: Any) -> ValidationResult:
        """Validate custom data types"""
        
        result = ValidationResult()
        
        # Basic validation
        if data_point is None:
            result.add_error("Data point is None")
            return result
        
        # Add custom validation logic here
        result.add_metric('custom_validation', True)
        
        return result
    
    def _extract_timestamp(self, data_point: Any) -> datetime:
        """Extract timestamp from data point"""
        
        if isinstance(data_point, (TickData, BarData)):
            return data_point.timestamp
        elif hasattr(data_point, 'timestamp'):
            return data_point.timestamp
        else:
            return datetime.utcnow()
    
    def _extract_value(self, data_point: Any) -> float:
        """Extract value from data point"""
        
        if isinstance(data_point, TickData):
            return data_point.price
        elif isinstance(data_point, BarData):
            return data_point.close
        elif hasattr(data_point, 'value'):
            return data_point.value
        else:
            return 0.0
    
    def _is_valid_data_point(self, data_point: Any) -> bool:
        """Check if data point is valid"""
        
        if data_point is None:
            return False
        
        # Check for required attributes
        if isinstance(data_point, TickData):
            return data_point.price is not None and data_point.price > 0
        elif isinstance(data_point, BarData):
            return all([
                data_point.open is not None,
                data_point.high is not None,
                data_point.low is not None,
                data_point.close is not None,
                data_point.open > 0
            ])
        
        return True
    
    def _get_quality_grade(self, score: float) -> DataQualityGrade:
        """Get quality grade from score"""
        
        if score >= 0.95:
            return DataQualityGrade.EXCELLENT
        elif score >= 0.85:
            return DataQualityGrade.GOOD
        elif score >= 0.70:
            return DataQualityGrade.FAIR
        elif score >= 0.50:
            return DataQualityGrade.POOR
        else:
            return DataQualityGrade.CRITICAL
    
    def _generate_recommendations(self, report: ValidationReport) -> List[str]:
        """Generate improvement recommendations"""
        
        recommendations = []
        
        # Quality-based recommendations
        if report.data_quality_score.overall_score < 0.7:
            recommendations.append("Overall quality below threshold - comprehensive review needed")
        
        if report.data_quality_score.completeness_score < 0.9:
            recommendations.append("Address data completeness issues")
        
        if report.data_quality_score.accuracy_score < 0.9:
            recommendations.append("Improve data accuracy through better validation")
        
        if report.data_quality_score.temporal_integrity_score < 0.9:
            recommendations.append("Fix temporal consistency issues")
        
        # Bias-based recommendations
        if report.bias_detection_results:
            recommendations.append("Address detected temporal biases")
        
        # Missing data recommendations
        if report.missing_data_points > 0:
            recommendations.append("Implement better missing data handling")
        
        return recommendations
    
    def _calculate_category_score(self, category: ValidationCategory, report: ValidationReport) -> float:
        """Calculate score for validation category"""
        
        if category == ValidationCategory.COMPLETENESS:
            return report.data_quality_score.completeness_score
        elif category == ValidationCategory.ACCURACY:
            return report.data_quality_score.accuracy_score
        elif category == ValidationCategory.CONSISTENCY:
            return report.data_quality_score.consistency_score
        elif category == ValidationCategory.TIMELINESS:
            return report.data_quality_score.timeliness_score
        elif category == ValidationCategory.VALIDITY:
            return report.data_quality_score.validity_score
        elif category == ValidationCategory.TEMPORAL_INTEGRITY:
            return report.data_quality_score.temporal_integrity_score
        else:
            return 0.0
    
    def _finalize_report(self, report: ValidationReport):
        """Finalize validation report"""
        
        # Set data points evaluated
        report.data_quality_score.data_points_evaluated = report.total_data_points
        
        # Set validation level
        report.data_quality_score.validation_level = self.config['validation_level']
        
        # Generate final recommendations
        if report.data_quality_score.overall_score < 0.5:
            report.data_quality_score.critical_issues.append("Critical quality score - immediate action required")
    
    def _update_statistics(self, report: ValidationReport):
        """Update validation statistics"""
        
        self.stats['total_validations'] += 1
        
        if report.data_quality_score.overall_score >= 0.7:
            self.stats['passed_validations'] += 1
        else:
            self.stats['failed_validations'] += 1
        
        if report.bias_detection_results:
            self.stats['biases_detected'] += len(report.bias_detection_results)
        
        # Store quality score
        component_key = f"overall_quality"
        self.quality_scores[component_key].append(report.data_quality_score.overall_score)
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary"""
        
        return {
            'statistics': self.stats.copy(),
            'recent_validations': len(self.validation_history),
            'average_quality_score': np.mean([
                r.data_quality_score.overall_score for r in self.validation_history
            ]) if self.validation_history else 0.0,
            'quality_trend': self._calculate_quality_trend(),
            'configuration': {
                'validation_level': self.config['validation_level'],
                'bias_detection_enabled': self.config['enable_bias_detection'],
                'interpolation_enabled': self.config['enable_interpolation']
            }
        }
    
    def _calculate_quality_trend(self) -> str:
        """Calculate quality trend"""
        
        if len(self.validation_history) < 5:
            return "insufficient_data"
        
        recent_scores = [r.data_quality_score.overall_score for r in list(self.validation_history)[-10:]]
        
        # Simple trend calculation
        if len(recent_scores) >= 5:
            first_half = np.mean(recent_scores[:len(recent_scores)//2])
            second_half = np.mean(recent_scores[len(recent_scores)//2:])
            
            if second_half > first_half + 0.05:
                return "improving"
            elif second_half < first_half - 0.05:
                return "declining"
            else:
                return "stable"
        
        return "stable"

# Global instance
enhanced_data_validator = EnhancedDataValidator()

# Export key components
__all__ = [
    'ValidationLevel',
    'DataQualityGrade',
    'InterpolationMethod',
    'ValidationCategory',
    'DataQualityScore',
    'InterpolationResult',
    'ValidationReport',
    'EnhancedDataValidator',
    'enhanced_data_validator'
]