"""
Test Suite for Predictive Monitoring System

Comprehensive testing for the predictive monitoring and alert system.
"""

import unittest
import time
import threading
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from collections import deque

# Import modules to test
from .performance_monitor import PerformanceMetric, AlertSeverity, AlertStatus
from .predictive_monitoring import IntelligentAlertManager, PredictiveDashboard
from .enhanced_monitor import EnhancedPerformanceMonitor

class TestIntelligentAlertManager(unittest.TestCase):
    """Test cases for IntelligentAlertManager"""
    
    def setUp(self):
        """Set up test environment"""
        self.alert_manager = IntelligentAlertManager()
        
        # Create test alerts
        self.test_alerts = [
            self._create_test_alert('cpu_usage', 85.0, AlertSeverity.HIGH),
            self._create_test_alert('memory_usage', 90.0, AlertSeverity.CRITICAL),
            self._create_test_alert('disk_usage', 70.0, AlertSeverity.MEDIUM)
        ]
    
    def _create_test_alert(self, metric_name, value, severity):
        """Create test alert"""
        from .performance_monitor import Alert
        import uuid
        
        return Alert(
            id=str(uuid.uuid4()),
            metric_name=metric_name,
            severity=severity,
            status=AlertStatus.ACTIVE,
            message=f"Test alert for {metric_name}",
            value=value,
            threshold=80.0,
            timestamp=time.time()
        )
    
    def test_add_alert(self):
        """Test adding alerts"""
        alert = self.test_alerts[0]
        alert_id = self.alert_manager.add_alert(alert)
        
        self.assertEqual(alert_id, alert.id)
        self.assertIn(alert.id, self.alert_manager.alerts)
        self.assertEqual(len(self.alert_manager.alert_history), 1)
    
    def test_alert_correlation(self):
        """Test alert correlation"""
        # Add related alerts
        alert1 = self.test_alerts[0]  # cpu_usage
        alert2 = self._create_test_alert('memory_usage', 88.0, AlertSeverity.HIGH)
        
        # Add alerts within correlation window
        self.alert_manager.add_alert(alert1)
        time.sleep(0.1)  # Small delay
        self.alert_manager.add_alert(alert2)
        
        # Check if alerts are correlated
        correlated_alerts = self.alert_manager.get_correlated_alerts(alert1.id)
        self.assertTrue(len(correlated_alerts) > 0)
    
    def test_alert_prioritization(self):
        """Test alert prioritization"""
        # Add alerts with different severities
        for alert in self.test_alerts:
            self.alert_manager.add_alert(alert)
        
        priority_alerts = self.alert_manager.get_priority_alerts(2)
        
        # Should return highest priority alerts first
        self.assertEqual(len(priority_alerts), 2)
        self.assertEqual(priority_alerts[0].severity, AlertSeverity.CRITICAL)
    
    def test_alert_suppression(self):
        """Test alert suppression"""
        alert = self.test_alerts[0]
        alert_id = self.alert_manager.add_alert(alert)
        
        # Suppress alert
        success = self.alert_manager.suppress_alert(alert_id, "Testing suppression", 3600)
        
        self.assertTrue(success)
        self.assertEqual(self.alert_manager.alerts[alert_id].status, AlertStatus.SUPPRESSED)
    
    def test_alert_analytics(self):
        """Test alert analytics"""
        # Add multiple alerts
        for alert in self.test_alerts:
            self.alert_manager.add_alert(alert)
        
        analytics = self.alert_manager.get_alert_analytics()
        
        self.assertIn('total_alerts', analytics)
        self.assertIn('last_hour_count', analytics)
        self.assertIn('top_metrics', analytics)
        self.assertEqual(analytics['total_alerts'], 3)

class TestPredictiveDashboard(unittest.TestCase):
    """Test cases for PredictiveDashboard"""
    
    def setUp(self):
        """Set up test environment"""
        self.mock_metrics_collector = Mock()
        self.mock_alert_manager = Mock()
        
        # Configure mock metrics collector
        self.mock_metrics_collector.get_all_metrics_summary.return_value = {
            'cpu_usage': {'latest': 45.2, 'mean': 42.1, 'std': 5.2, 'status': 'ok'},
            'memory_usage': {'latest': 68.5, 'mean': 65.0, 'std': 8.1, 'status': 'ok'}
        }
        
        self.mock_metrics_collector.get_metric_history.return_value = [
            PerformanceMetric('cpu_usage', 40.0, '%', time.time() - 3600, 'system'),
            PerformanceMetric('cpu_usage', 45.2, '%', time.time(), 'system')
        ]
        
        self.dashboard = PredictiveDashboard(
            self.mock_metrics_collector,
            self.mock_alert_manager
        )
    
    def test_generate_dashboard(self):
        """Test dashboard generation"""
        dashboard_data = self.dashboard.generate_predictive_dashboard()
        
        self.assertIn('timestamp', dashboard_data)
        self.assertIn('metrics_overview', dashboard_data)
        self.assertIn('predictive_charts', dashboard_data)
        self.assertIn('alert_summary', dashboard_data)
        self.assertIn('system_health', dashboard_data)
    
    def test_metrics_overview(self):
        """Test metrics overview generation"""
        overview = self.dashboard._generate_metrics_overview()
        
        self.assertIn('total_metrics', overview)
        self.assertIn('healthy_metrics', overview)
        self.assertIn('data_quality_score', overview)
        self.assertGreaterEqual(overview['total_metrics'], 0)
    
    def test_predictive_charts(self):
        """Test predictive charts generation"""
        charts = self.dashboard._generate_predictive_charts()
        
        # Should have charts for key metrics
        self.assertIsInstance(charts, dict)
        
        # Check if charts have required components
        for metric_name, chart_data in charts.items():
            self.assertIn('historical_data', chart_data)
            self.assertIn('predictions', chart_data)
            self.assertIn('thresholds', chart_data)
    
    def test_real_time_data(self):
        """Test real-time data retrieval"""
        real_time_data = self.dashboard.get_real_time_data()
        
        self.assertIsInstance(real_time_data, dict)
        self.assertIn('timestamp', real_time_data)

class TestEnhancedPerformanceMonitor(unittest.TestCase):
    """Test cases for EnhancedPerformanceMonitor"""
    
    def setUp(self):
        """Set up test environment"""
        self.monitor = EnhancedPerformanceMonitor(
            enable_dashboard=True,
            enable_predictions=True
        )
    
    def test_monitor_initialization(self):
        """Test monitor initialization"""
        self.assertIsNotNone(self.monitor.base_monitor)
        self.assertIsNotNone(self.monitor.predictive_monitor)
        self.assertTrue(self.monitor.enable_predictions)
    
    def test_record_metric(self):
        """Test metric recording"""
        # Record a metric
        self.monitor.record_metric('cpu_usage', 45.2, {'source': 'test'})
        
        # Check if metric was recorded
        summary = self.monitor.base_monitor.get_performance_summary()
        self.assertIn('cpu_usage', summary)
    
    def test_system_health_calculation(self):
        """Test system health calculation"""
        # Record some metrics
        self.monitor.record_metric('cpu_usage', 45.2)
        self.monitor.record_metric('memory_usage', 68.5)
        
        health = self.monitor._calculate_system_health()
        
        self.assertIn('score', health)
        self.assertIn('status', health)
        self.assertIn('issues', health)
        self.assertGreaterEqual(health['score'], 0)
        self.assertLessEqual(health['score'], 100)
    
    def test_comprehensive_report(self):
        """Test comprehensive report generation"""
        report = self.monitor.get_comprehensive_report()
        
        self.assertIn('timestamp', report)
        self.assertIn('basic_metrics', report)
        self.assertIn('system_health', report)
        self.assertIn('predictions', report)
        self.assertIn('recommendations', report)
    
    def test_monitoring_lifecycle(self):
        """Test monitoring start/stop lifecycle"""
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Check status
        status = self.monitor.get_monitoring_status()
        self.assertIn('base_monitor', status)
        self.assertIn('predictive_monitoring', status)
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        # Verify cleanup
        self.assertIsNotNone(self.monitor.base_monitor)

class TestPredictiveAlgorithms(unittest.TestCase):
    """Test cases for predictive algorithms"""
    
    def setUp(self):
        """Set up test environment"""
        self.sample_metrics = self._generate_sample_metrics()
    
    def _generate_sample_metrics(self):
        """Generate sample metrics for testing"""
        metrics = []
        base_time = time.time() - 3600  # 1 hour ago
        
        for i in range(100):
            timestamp = base_time + (i * 36)  # Every 36 seconds
            # Generate trending data
            value = 50 + (i * 0.1) + np.random.normal(0, 2)
            
            metrics.append(PerformanceMetric(
                name='cpu_usage',
                value=max(0, min(100, value)),
                unit='%',
                timestamp=timestamp,
                category='system'
            ))
        
        return metrics
    
    def test_trend_detection(self):
        """Test trend detection algorithm"""
        # Simple trend detection test
        values = [m.value for m in self.sample_metrics]
        
        # Calculate linear trend
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        # Should detect increasing trend
        self.assertGreater(slope, 0)
    
    def test_anomaly_detection(self):
        """Test anomaly detection"""
        values = [m.value for m in self.sample_metrics]
        
        # Add anomalous values
        anomalous_values = values + [120, 150, 200]  # Clear outliers
        
        # Simple anomaly detection using z-score
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        anomalies = []
        for val in anomalous_values:
            z_score = abs(val - mean_val) / std_val
            if z_score > 2:  # 2 standard deviations
                anomalies.append(val)
        
        self.assertGreater(len(anomalies), 0)
    
    def test_capacity_prediction(self):
        """Test capacity prediction"""
        values = [m.value for m in self.sample_metrics]
        
        # Simple capacity prediction
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        # Predict future values
        future_x = len(values) + 10
        predicted_value = slope * future_x + intercept
        
        # Should predict reasonable value
        self.assertGreater(predicted_value, 0)
        self.assertLess(predicted_value, 200)

class TestIntegrationScenarios(unittest.TestCase):
    """Integration test scenarios"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.monitor = EnhancedPerformanceMonitor(
            enable_dashboard=True,
            enable_predictions=True
        )
    
    def test_end_to_end_monitoring(self):
        """Test end-to-end monitoring scenario"""
        # Start monitoring
        self.monitor.start_monitoring()
        
        try:
            # Simulate metric collection
            metrics_to_record = [
                ('cpu_usage', 45.2),
                ('memory_usage', 68.5),
                ('disk_usage', 75.1),
                ('data_load_time', 1.2),
                ('data_throughput', 850.0)
            ]
            
            for metric_name, value in metrics_to_record:
                self.monitor.record_metric(metric_name, value, {'source': 'test'})
                time.sleep(0.1)  # Small delay
            
            # Get comprehensive report
            report = self.monitor.get_comprehensive_report()
            
            # Verify report structure
            self.assertIn('system_health', report)
            self.assertIn('basic_metrics', report)
            
            # Check system health
            health = report['system_health']
            self.assertGreaterEqual(health['score'], 0)
            
            # Get dashboard data
            dashboard_data = self.monitor.get_dashboard_data()
            self.assertIn('timestamp', dashboard_data)
            
        finally:
            # Stop monitoring
            self.monitor.stop_monitoring()
    
    def test_alert_workflow(self):
        """Test complete alert workflow"""
        # Record high CPU usage to trigger alert
        self.monitor.record_metric('cpu_usage', 95.0, {'source': 'stress_test'})
        
        # Give time for alert processing
        time.sleep(0.5)
        
        # Check if alert was created
        if self.monitor.predictive_monitor:
            active_alerts = self.monitor.predictive_monitor.alert_manager.get_priority_alerts()
            
            if active_alerts:
                alert = active_alerts[0]
                
                # Test alert acknowledgment
                ack_result = self.monitor.acknowledge_alert(alert.id, 'test_user')
                self.assertTrue(ack_result)
                
                # Test alert resolution
                resolve_result = self.monitor.resolve_alert(alert.id, 'test_user')
                self.assertTrue(resolve_result)

def run_performance_tests():
    """Run performance tests"""
    print("Running performance tests...")
    
    # Test metric recording performance
    monitor = EnhancedPerformanceMonitor(enable_dashboard=False, enable_predictions=False)
    
    start_time = time.time()
    for i in range(1000):
        monitor.record_metric('test_metric', float(i), {'iteration': i})
    
    duration = time.time() - start_time
    print(f"Recorded 1000 metrics in {duration:.3f} seconds")
    print(f"Rate: {1000/duration:.1f} metrics/second")
    
    # Test dashboard generation performance
    monitor_with_dashboard = EnhancedPerformanceMonitor(enable_dashboard=True, enable_predictions=True)
    
    # Record some metrics first
    for i in range(100):
        monitor_with_dashboard.record_metric('cpu_usage', 50 + (i % 20), {'test': True})
    
    start_time = time.time()
    dashboard_data = monitor_with_dashboard.get_dashboard_data()
    duration = time.time() - start_time
    
    print(f"Generated dashboard in {duration:.3f} seconds")
    
    monitor.stop_monitoring()
    monitor_with_dashboard.stop_monitoring()

if __name__ == '__main__':
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance tests
    print("\n" + "="*50)
    run_performance_tests()
    
    print("\n" + "="*50)
    print("All tests completed!")