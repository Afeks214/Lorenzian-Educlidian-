"""
Comprehensive Temporal Consistency Test Suite
Agent 5: Data Quality & Bias Elimination

Comprehensive test suite for validating temporal consistency across
the entire data pipeline, including bias detection, boundary enforcement,
and quality validation.

Test Coverage:
- Temporal bias detection
- Boundary enforcement
- Multi-timeframe synchronization
- Data quality validation
- Missing data handling
- Pipeline integration tests
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, AsyncMock

# Import components under test
from src.data.temporal_bias_detector import (
    TemporalBiasDetector, BiasType, SeverityLevel, 
    DataAvailabilityRecord, BiasDetectionResult
)
from src.data.temporal_boundary_enforcer import (
    TemporalBoundaryEnforcer, DataAccessRequest, AccessType,
    BoundaryViolationType, EnforcementAction
)
from src.data.multi_timeframe_synchronizer import (
    MultiTimeframeSynchronizer, TimeframeDefinition, TimeframeType,
    SynchronizationMode, TimeframeDataPoint
)
from src.data.enhanced_data_validation import (
    EnhancedDataValidator, ValidationLevel, DataQualityGrade
)
from src.data.missing_data_handler import (
    MissingDataHandler, InterpolationMethod, MissingDataType
)
from src.data.comprehensive_quality_monitor import (
    ComprehensiveDataQualityMonitor, QualityDimension, AlertPriority
)
from src.data.data_handler import TickData
from src.data.bar_generator import BarData

# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_tick_data():
    """Generate sample tick data for testing"""
    base_time = datetime(2023, 1, 1, 9, 0, 0)
    ticks = []
    
    for i in range(100):
        tick = TickData(
            timestamp=base_time + timedelta(seconds=i),
            symbol="TESTSYM",
            price=100.0 + np.random.normal(0, 0.1),
            volume=1000 + np.random.randint(-100, 100),
            source="test"
        )
        ticks.append(tick)
    
    return ticks

@pytest.fixture
def sample_bar_data():
    """Generate sample bar data for testing"""
    base_time = datetime(2023, 1, 1, 9, 0, 0)
    bars = []
    
    for i in range(50):
        base_price = 100.0 + np.random.normal(0, 0.5)
        bar = BarData(
            timestamp=base_time + timedelta(minutes=i*5),
            open=base_price,
            high=base_price + 0.5,
            low=base_price - 0.5,
            close=base_price + np.random.normal(0, 0.2),
            volume=10000 + np.random.randint(-1000, 1000),
            tick_count=100,
            vwap=base_price,
            timeframe=5
        )
        bars.append(bar)
    
    return bars

@pytest.fixture
def temporal_bias_detector():
    """Create temporal bias detector for testing"""
    return TemporalBiasDetector()

@pytest.fixture
def boundary_enforcer():
    """Create boundary enforcer for testing"""
    return TemporalBoundaryEnforcer()

@pytest.fixture
def timeframe_synchronizer():
    """Create timeframe synchronizer for testing"""
    return MultiTimeframeSynchronizer()

@pytest.fixture
def data_validator():
    """Create data validator for testing"""
    return EnhancedDataValidator()

@pytest.fixture
def missing_data_handler():
    """Create missing data handler for testing"""
    return MissingDataHandler()

@pytest.fixture
def quality_monitor():
    """Create quality monitor for testing"""
    return ComprehensiveDataQualityMonitor()

# =============================================================================
# TEMPORAL BIAS DETECTION TESTS
# =============================================================================

class TestTemporalBiasDetection:
    """Test suite for temporal bias detection"""
    
    def test_look_ahead_bias_detection(self, temporal_bias_detector):
        """Test detection of look-ahead bias"""
        
        # Create data with future timestamps
        current_time = datetime.utcnow()
        future_time = current_time + timedelta(minutes=30)
        
        data = {
            'timestamps': [current_time, future_time],
            'values': [100.0, 101.0]
        }
        
        context = {
            'current_time': current_time,
            'component': 'test_component'
        }
        
        # Detect bias
        bias_results = temporal_bias_detector.detect_bias(data, context)
        
        # Verify bias detection
        assert len(bias_results) > 0
        assert bias_results[0].bias_type == BiasType.LOOK_AHEAD_BIAS
        assert bias_results[0].severity_level == SeverityLevel.CRITICAL
    
    def test_no_bias_detection(self, temporal_bias_detector):
        """Test that no bias is detected for valid data"""
        
        current_time = datetime.utcnow()
        past_time = current_time - timedelta(minutes=30)
        
        data = {
            'timestamps': [past_time, current_time],
            'values': [100.0, 101.0]
        }
        
        context = {
            'current_time': current_time,
            'component': 'test_component'
        }
        
        # Detect bias
        bias_results = temporal_bias_detector.detect_bias(data, context)
        
        # Verify no bias detected
        assert len(bias_results) == 0
    
    def test_temporal_leak_detection(self, temporal_bias_detector):
        """Test detection of temporal information leakage"""
        
        # Create events with dependency violations
        event1_time = datetime.utcnow()
        event2_time = event1_time - timedelta(minutes=10)  # Depends on future event
        
        data = {
            'events': [
                {
                    'id': 'event1',
                    'timestamp': event1_time,
                    'dependencies': ['event2']
                },
                {
                    'id': 'event2',
                    'timestamp': event2_time,
                    'dependencies': []
                }
            ]
        }
        
        context = {
            'component': 'test_component'
        }
        
        # Detect bias
        bias_results = temporal_bias_detector.detect_bias(data, context)
        
        # Verify temporal leak detection
        assert len(bias_results) > 0
        assert bias_results[0].bias_type == BiasType.TEMPORAL_LEAK
    
    def test_data_availability_validation(self, temporal_bias_detector):
        """Test data availability validation"""
        
        # Register data availability record
        availability_record = DataAvailabilityRecord(
            record_id="test_data",
            data_identifier="test_data",
            timestamp=datetime.utcnow(),
            availability_status="available",
            earliest_available_time=datetime.utcnow() - timedelta(hours=1),
            latest_available_time=datetime.utcnow() + timedelta(hours=1),
            data_source_latency=timedelta(seconds=30),
            source_system="test_system"
        )
        
        temporal_bias_detector.register_data_availability(availability_record)
        
        # Test valid access
        current_time = datetime.utcnow()
        data_time = current_time - timedelta(minutes=30)
        
        is_valid, error_msg = temporal_bias_detector.validate_temporal_access(
            "test_data", current_time, data_time
        )
        
        assert is_valid
        assert error_msg is None
        
        # Test invalid access (too early)
        early_time = datetime.utcnow() - timedelta(hours=2)
        
        is_valid, error_msg = temporal_bias_detector.validate_temporal_access(
            "test_data", early_time, data_time
        )
        
        assert not is_valid
        assert error_msg is not None

# =============================================================================
# BOUNDARY ENFORCEMENT TESTS
# =============================================================================

class TestBoundaryEnforcement:
    """Test suite for boundary enforcement"""
    
    @pytest.mark.asyncio
    async def test_future_access_blocking(self, boundary_enforcer):
        """Test blocking of future data access"""
        
        current_time = datetime.utcnow()
        future_time = current_time + timedelta(minutes=30)
        
        # Create access request for future data
        request = DataAccessRequest(
            request_id="test_request",
            requester_id="test_requester",
            access_type=AccessType.READ,
            request_time=current_time,
            data_timestamp=future_time,
            latest_allowable_time=current_time,
            data_identifier="test_data",
            data_type="test_type",
            component="test_component"
        )
        
        # Request access
        is_granted, violation_reason, violation = await boundary_enforcer.request_data_access(request)
        
        # Verify access is blocked
        assert not is_granted
        assert violation_reason is not None
        assert violation is not None
        assert violation.violation_type == BoundaryViolationType.FUTURE_ACCESS
    
    @pytest.mark.asyncio
    async def test_valid_access_granting(self, boundary_enforcer):
        """Test granting of valid data access"""
        
        current_time = datetime.utcnow()
        past_time = current_time - timedelta(minutes=30)
        
        # Create access request for past data
        request = DataAccessRequest(
            request_id="test_request",
            requester_id="test_requester",
            access_type=AccessType.READ,
            request_time=current_time,
            data_timestamp=past_time,
            latest_allowable_time=current_time,
            data_identifier="test_data",
            data_type="test_type",
            component="test_component"
        )
        
        # Request access
        is_granted, violation_reason, violation = await boundary_enforcer.request_data_access(request)
        
        # Verify access is granted
        assert is_granted
        assert violation_reason is None
        assert violation is None
    
    @pytest.mark.asyncio
    async def test_enforcement_statistics(self, boundary_enforcer):
        """Test enforcement statistics tracking"""
        
        # Get initial statistics
        initial_stats = boundary_enforcer.get_enforcement_summary()
        initial_requests = initial_stats['statistics']['total_requests']
        
        current_time = datetime.utcnow()
        past_time = current_time - timedelta(minutes=30)
        
        # Make valid request
        request = DataAccessRequest(
            request_id="test_request",
            requester_id="test_requester",
            access_type=AccessType.READ,
            request_time=current_time,
            data_timestamp=past_time,
            latest_allowable_time=current_time,
            data_identifier="test_data",
            data_type="test_type",
            component="test_component"
        )
        
        await boundary_enforcer.request_data_access(request)
        
        # Check statistics updated
        updated_stats = boundary_enforcer.get_enforcement_summary()
        assert updated_stats['statistics']['total_requests'] == initial_requests + 1

# =============================================================================
# MULTI-TIMEFRAME SYNCHRONIZATION TESTS
# =============================================================================

class TestMultiTimeframeSynchronization:
    """Test suite for multi-timeframe synchronization"""
    
    def test_timeframe_registration(self, timeframe_synchronizer):
        """Test timeframe registration"""
        
        # Create timeframe definition
        timeframe_def = TimeframeDefinition(
            timeframe_id="test_1m",
            timeframe_type=TimeframeType.MINUTE,
            duration_seconds=60,
            sync_mode=SynchronizationMode.STRICT
        )
        
        # Register timeframe
        timeframe_synchronizer.register_timeframe(timeframe_def)
        
        # Verify registration
        assert "test_1m" in timeframe_synchronizer.timeframes
        assert timeframe_synchronizer.timeframes["test_1m"].duration_seconds == 60
    
    @pytest.mark.asyncio
    async def test_data_point_addition(self, timeframe_synchronizer):
        """Test adding data points to timeframes"""
        
        # Register timeframe
        timeframe_def = TimeframeDefinition(
            timeframe_id="test_1m",
            timeframe_type=TimeframeType.MINUTE,
            duration_seconds=60
        )
        
        timeframe_synchronizer.register_timeframe(timeframe_def)
        
        # Create data point
        data_point = TimeframeDataPoint(
            data_id="test_data",
            timeframe_id="test_1m",
            timestamp=datetime.utcnow(),
            data=100.0,
            quality_score=0.95
        )
        
        # Add data point
        result = await timeframe_synchronizer.add_data_point("test_1m", data_point)
        
        # Verify addition
        assert result is True
        assert len(timeframe_synchronizer.data_buffers["test_1m"]) == 1
    
    @pytest.mark.asyncio
    async def test_synchronization_validation(self, timeframe_synchronizer):
        """Test synchronization validation"""
        
        # Register multiple timeframes
        timeframes = ["1m", "5m", "30m"]
        
        for tf in timeframes:
            timeframe_def = TimeframeDefinition(
                timeframe_id=tf,
                timeframe_type=TimeframeType.MINUTE,
                duration_seconds=60 if tf == "1m" else (300 if tf == "5m" else 1800)
            )
            timeframe_synchronizer.register_timeframe(timeframe_def)
        
        # Add data points to each timeframe
        base_time = datetime.utcnow()
        
        for i, tf in enumerate(timeframes):
            data_point = TimeframeDataPoint(
                data_id=f"test_data_{tf}",
                timeframe_id=tf,
                timestamp=base_time + timedelta(seconds=i),
                data=100.0 + i,
                quality_score=0.95
            )
            
            await timeframe_synchronizer.add_data_point(tf, data_point)
        
        # Validate synchronization
        is_valid, errors = timeframe_synchronizer.validate_synchronization()
        
        # Should be valid with proper data
        assert is_valid
        assert len(errors) == 0

# =============================================================================
# DATA VALIDATION TESTS
# =============================================================================

class TestDataValidation:
    """Test suite for data validation"""
    
    @pytest.mark.asyncio
    async def test_tick_data_validation(self, data_validator, sample_tick_data):
        """Test tick data validation"""
        
        # Validate sample tick data
        validation_report = await data_validator.validate_data(sample_tick_data)
        
        # Verify validation results
        assert validation_report.data_quality_score.overall_score > 0.0
        assert validation_report.total_data_points == len(sample_tick_data)
        assert len(validation_report.validation_results) > 0
    
    @pytest.mark.asyncio
    async def test_bar_data_validation(self, data_validator, sample_bar_data):
        """Test bar data validation"""
        
        # Validate sample bar data
        validation_report = await data_validator.validate_data(sample_bar_data)
        
        # Verify validation results
        assert validation_report.data_quality_score.overall_score > 0.0
        assert validation_report.total_data_points == len(sample_bar_data)
        assert len(validation_report.validation_results) > 0
    
    @pytest.mark.asyncio
    async def test_quality_scoring(self, data_validator, sample_tick_data):
        """Test quality scoring"""
        
        # Validate data
        validation_report = await data_validator.validate_data(sample_tick_data)
        
        # Check quality score components
        quality_score = validation_report.data_quality_score
        
        assert 0.0 <= quality_score.completeness_score <= 1.0
        assert 0.0 <= quality_score.accuracy_score <= 1.0
        assert 0.0 <= quality_score.consistency_score <= 1.0
        assert 0.0 <= quality_score.timeliness_score <= 1.0
        assert 0.0 <= quality_score.validity_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_bias_detection_integration(self, data_validator):
        """Test bias detection integration"""
        
        # Create data with future timestamps (bias)
        current_time = datetime.utcnow()
        future_time = current_time + timedelta(minutes=30)
        
        biased_tick = TickData(
            timestamp=future_time,
            symbol="TESTSYM",
            price=100.0,
            volume=1000,
            source="test"
        )
        
        # Validate biased data
        validation_report = await data_validator.validate_data([biased_tick])
        
        # Should detect bias issues
        assert len(validation_report.bias_detection_results) > 0
        assert validation_report.bias_detection_results[0].bias_type == BiasType.LOOK_AHEAD_BIAS

# =============================================================================
# MISSING DATA HANDLING TESTS
# =============================================================================

class TestMissingDataHandling:
    """Test suite for missing data handling"""
    
    @pytest.mark.asyncio
    async def test_gap_detection(self, missing_data_handler):
        """Test gap detection in data"""
        
        # Start processing
        missing_data_handler.start_processing()
        
        try:
            # Add data points with a gap
            base_time = datetime.utcnow()
            
            # First data point
            await missing_data_handler.add_data_point(
                "test_data", "1m", base_time, 100.0, "test_component"
            )
            
            # Second data point with large gap
            gap_time = base_time + timedelta(minutes=10)
            await missing_data_handler.add_data_point(
                "test_data", "1m", gap_time, 101.0, "test_component"
            )
            
            # Allow processing
            await asyncio.sleep(0.1)
            
            # Check gap detection
            assert missing_data_handler.stats['total_gaps_detected'] > 0
            assert len(missing_data_handler.missing_data_points) > 0
            
        finally:
            missing_data_handler.stop_processing()
    
    def test_interpolation_method_selection(self, missing_data_handler):
        """Test interpolation method selection"""
        
        # Test linear interpolation
        assert InterpolationMethod.LINEAR in missing_data_handler.interpolation_methods
        
        # Test cubic interpolation
        assert InterpolationMethod.CUBIC in missing_data_handler.interpolation_methods
        
        # Test forward fill
        assert InterpolationMethod.FORWARD_FILL in missing_data_handler.interpolation_methods
    
    @pytest.mark.asyncio
    async def test_interpolation_quality_scoring(self, missing_data_handler):
        """Test interpolation quality scoring"""
        
        # This would test the quality scoring for interpolated data
        # For now, just verify the method exists
        assert hasattr(missing_data_handler, '_calculate_quality_score')
        assert hasattr(missing_data_handler, '_calculate_confidence_score')

# =============================================================================
# QUALITY MONITORING TESTS
# =============================================================================

class TestQualityMonitoring:
    """Test suite for quality monitoring"""
    
    @pytest.mark.asyncio
    async def test_quality_evaluation(self, quality_monitor, sample_tick_data):
        """Test quality evaluation"""
        
        # Evaluate data quality
        scorecard = await quality_monitor.evaluate_data_quality(
            sample_tick_data, "test_component", "1m"
        )
        
        # Verify scorecard
        assert scorecard.overall_quality_score >= 0.0
        assert scorecard.data_points_evaluated == len(sample_tick_data)
        assert len(scorecard.dimension_scores) > 0
    
    def test_threshold_registration(self, quality_monitor):
        """Test quality threshold registration"""
        
        from src.data.comprehensive_quality_monitor import QualityThreshold
        
        # Create threshold
        threshold = QualityThreshold(
            threshold_id="test_threshold",
            dimension=QualityDimension.ACCURACY,
            component="test_component",
            critical_threshold=0.5,
            warning_threshold=0.7,
            target_threshold=0.9
        )
        
        # Register threshold
        quality_monitor.register_quality_threshold(threshold)
        
        # Verify registration
        assert "test_threshold" in quality_monitor.quality_thresholds
        assert quality_monitor.quality_thresholds["test_threshold"].dimension == QualityDimension.ACCURACY
    
    def test_quality_dashboard(self, quality_monitor):
        """Test quality dashboard generation"""
        
        # Get dashboard
        dashboard = quality_monitor.get_quality_dashboard()
        
        # Verify dashboard structure
        assert 'overall_status' in dashboard
        assert 'dimensional_quality' in dashboard
        assert 'alert_summary' in dashboard
        assert 'trend_analysis' in dashboard
        assert 'statistics' in dashboard

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestPipelineIntegration:
    """Integration tests for the complete pipeline"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(self, sample_tick_data):
        """Test complete end-to-end pipeline"""
        
        # Initialize all components
        bias_detector = TemporalBiasDetector()
        boundary_enforcer = TemporalBoundaryEnforcer()
        timeframe_synchronizer = MultiTimeframeSynchronizer()
        data_validator = EnhancedDataValidator()
        missing_data_handler = MissingDataHandler()
        quality_monitor = ComprehensiveDataQualityMonitor()
        
        # Start monitoring
        bias_detector.start_monitoring()
        boundary_enforcer.start_monitoring()
        timeframe_synchronizer.start_processing()
        missing_data_handler.start_processing()
        quality_monitor.start_monitoring()
        
        try:
            # Process data through pipeline
            validation_report = await data_validator.validate_data(sample_tick_data)
            
            # Evaluate quality
            scorecard = await quality_monitor.evaluate_data_quality(
                sample_tick_data, "test_component", "1m"
            )
            
            # Verify pipeline results
            assert validation_report.data_quality_score.overall_score > 0.0
            assert scorecard.overall_quality_score > 0.0
            
            # Check no critical bias issues
            critical_biases = [
                bias for bias in validation_report.bias_detection_results
                if bias.severity_level == SeverityLevel.CRITICAL
            ]
            assert len(critical_biases) == 0
            
        finally:
            # Stop monitoring
            bias_detector.stop_monitoring()
            boundary_enforcer.stop_monitoring()
            timeframe_synchronizer.stop_processing()
            missing_data_handler.stop_processing()
            quality_monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_temporal_consistency_preservation(self, sample_tick_data):
        """Test that temporal consistency is preserved throughout pipeline"""
        
        # Initialize validator
        data_validator = EnhancedDataValidator()
        
        # Validate data
        validation_report = await data_validator.validate_data(sample_tick_data)
        
        # Check temporal integrity
        assert validation_report.data_quality_score.temporal_integrity_score > 0.8
        
        # Verify no temporal violations
        temporal_errors = [
            error for result in validation_report.validation_results
            for error in result.errors
            if 'timestamp' in error.lower() or 'temporal' in error.lower()
        ]
        assert len(temporal_errors) == 0
    
    @pytest.mark.asyncio
    async def test_bias_free_operation(self, sample_tick_data):
        """Test that pipeline operates without introducing bias"""
        
        # Initialize components
        bias_detector = TemporalBiasDetector()
        data_validator = EnhancedDataValidator()
        
        # Start monitoring
        bias_detector.start_monitoring()
        
        try:
            # Process data
            validation_report = await data_validator.validate_data(sample_tick_data)
            
            # Check for bias issues
            bias_issues = validation_report.bias_detection_results
            
            # Should have no critical bias issues
            critical_biases = [
                bias for bias in bias_issues
                if bias.severity_level == SeverityLevel.CRITICAL
            ]
            assert len(critical_biases) == 0
            
        finally:
            bias_detector.stop_monitoring()

# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance tests for the pipeline"""
    
    @pytest.mark.asyncio
    async def test_validation_performance(self, data_validator):
        """Test validation performance with large datasets"""
        
        # Generate large dataset
        large_dataset = []
        base_time = datetime.utcnow()
        
        for i in range(1000):
            tick = TickData(
                timestamp=base_time + timedelta(seconds=i),
                symbol="TESTSYM",
                price=100.0 + np.random.normal(0, 0.1),
                volume=1000,
                source="test"
            )
            large_dataset.append(tick)
        
        # Measure validation time
        import time
        start_time = time.time()
        
        validation_report = await data_validator.validate_data(large_dataset)
        
        end_time = time.time()
        validation_time = end_time - start_time
        
        # Verify performance
        assert validation_time < 10.0  # Should complete within 10 seconds
        assert validation_report.data_quality_score.overall_score > 0.0
    
    @pytest.mark.asyncio
    async def test_bias_detection_performance(self, temporal_bias_detector):
        """Test bias detection performance"""
        
        # Generate large dataset
        timestamps = []
        values = []
        base_time = datetime.utcnow()
        
        for i in range(10000):
            timestamps.append(base_time + timedelta(seconds=i))
            values.append(100.0 + np.random.normal(0, 0.1))
        
        data = {
            'timestamps': timestamps,
            'values': values
        }
        
        context = {
            'current_time': base_time + timedelta(seconds=10000),
            'component': 'test_component'
        }
        
        # Measure bias detection time
        import time
        start_time = time.time()
        
        bias_results = temporal_bias_detector.detect_bias(data, context)
        
        end_time = time.time()
        detection_time = end_time - start_time
        
        # Verify performance
        assert detection_time < 5.0  # Should complete within 5 seconds

# =============================================================================
# STRESS TESTS
# =============================================================================

class TestStressScenarios:
    """Stress tests for edge cases and failure scenarios"""
    
    @pytest.mark.asyncio
    async def test_empty_data_handling(self, data_validator):
        """Test handling of empty data"""
        
        # Test with empty list
        validation_report = await data_validator.validate_data([])
        
        # Should handle gracefully
        assert validation_report.total_data_points == 0
        assert validation_report.data_quality_score.overall_score >= 0.0
    
    @pytest.mark.asyncio
    async def test_malformed_data_handling(self, data_validator):
        """Test handling of malformed data"""
        
        # Create malformed tick data
        malformed_tick = TickData(
            timestamp=datetime.utcnow(),
            symbol="TESTSYM",
            price=-100.0,  # Invalid negative price
            volume=-1000,  # Invalid negative volume
            source="test"
        )
        
        # Validate malformed data
        validation_report = await data_validator.validate_data([malformed_tick])
        
        # Should detect issues
        assert len(validation_report.validation_results) > 0
        assert not validation_report.validation_results[0].is_valid
    
    @pytest.mark.asyncio
    async def test_extreme_gap_handling(self, missing_data_handler):
        """Test handling of extreme data gaps"""
        
        # Start processing
        missing_data_handler.start_processing()
        
        try:
            # Add data points with extreme gap
            base_time = datetime.utcnow()
            
            # First data point
            await missing_data_handler.add_data_point(
                "test_data", "1m", base_time, 100.0, "test_component"
            )
            
            # Second data point with extreme gap (24 hours)
            gap_time = base_time + timedelta(hours=24)
            await missing_data_handler.add_data_point(
                "test_data", "1m", gap_time, 101.0, "test_component"
            )
            
            # Allow processing
            await asyncio.sleep(0.1)
            
            # Should detect gap but not attempt interpolation
            assert missing_data_handler.stats['total_gaps_detected'] > 0
            
        finally:
            missing_data_handler.stop_processing()

# =============================================================================
# TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    """Run all tests"""
    pytest.main([__file__, "-v", "--tb=short"])