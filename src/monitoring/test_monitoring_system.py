#!/usr/bin/env python3
"""
AGENT 6: Monitoring System Validation Script
Comprehensive testing and validation of the monitoring system components.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.monitoring.monitoring_integration import MonitoringSystem, MonitoringConfig
from src.monitoring.health_monitor import HealthMonitor
from src.monitoring.prometheus_metrics import MetricsCollector, MetricsConfig
from src.monitoring.enhanced_alerting import EnhancedAlertingSystem, EnhancedAlert, AlertPriority, AlertStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonitoringSystemValidator:
    """Comprehensive validation of the monitoring system."""
    
    def __init__(self):
        self.test_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'PASS',
            'tests': {},
            'performance_metrics': {},
            'components_tested': []
        }
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all monitoring system tests."""
        logger.info("🧪 Starting comprehensive monitoring system validation...")
        
        # Test individual components
        await self._test_health_monitor()
        await self._test_prometheus_metrics()
        await self._test_alerting_system()
        await self._test_monitoring_integration()
        
        # Test system integration
        await self._test_end_to_end_monitoring()
        
        # Performance tests
        await self._test_performance_under_load()
        
        # Generate final report
        self._generate_test_report()
        
        return self.test_results
        
    async def _test_health_monitor(self):
        """Test health monitoring component."""
        logger.info("Testing Health Monitor...")
        
        try:
            # Test health monitor initialization
            health_monitor = HealthMonitor("redis://localhost:6379")
            
            # Test health check execution
            system_health = await health_monitor.check_all_components()
            
            # Validate health check results
            assert system_health is not None
            assert hasattr(system_health, 'status')
            assert hasattr(system_health, 'components')
            
            # Test detailed health check
            detailed_health = await health_monitor.get_detailed_health()
            assert 'recommendations' in detailed_health
            assert 'check_intervals' in detailed_health
            
            self.test_results['tests']['health_monitor'] = 'PASS'
            self.test_results['components_tested'].append('health_monitor')
            
            logger.info("✅ Health Monitor test passed")
            
        except Exception as e:
            logger.error(f"❌ Health Monitor test failed: {e}")
            self.test_results['tests']['health_monitor'] = f'FAIL: {e}'
            self.test_results['overall_status'] = 'FAIL'
            
    async def _test_prometheus_metrics(self):
        """Test Prometheus metrics component."""
        logger.info("Testing Prometheus Metrics...")
        
        try:
            # Test metrics collector initialization
            config = MetricsConfig(
                enable_system_metrics=True,
                enable_business_metrics=True,
                enable_sla_metrics=True,
                metrics_port=8001  # Use different port for testing
            )
            
            metrics_collector = MetricsCollector(config)
            
            # Test metrics collection
            metrics_collector.start_collection()
            
            # Test individual metric collectors
            metrics_collector.trading_metrics.record_signal('test_strategy', 'buy', 'BTCUSD')
            metrics_collector.trading_metrics.update_pnl('test_strategy', 'BTCUSD', 100.0)
            metrics_collector.risk_metrics.update_var('test_portfolio', '1d', 0.02, 0.03)
            metrics_collector.system_metrics.collect_system_metrics()
            
            # Test metrics export
            metrics_data = metrics_collector.get_metrics()
            assert metrics_data is not None
            
            metrics_collector.stop_collection()
            
            self.test_results['tests']['prometheus_metrics'] = 'PASS'
            self.test_results['components_tested'].append('prometheus_metrics')
            
            logger.info("✅ Prometheus Metrics test passed")
            
        except Exception as e:
            logger.error(f"❌ Prometheus Metrics test failed: {e}")
            self.test_results['tests']['prometheus_metrics'] = f'FAIL: {e}'
            self.test_results['overall_status'] = 'FAIL'
            
    async def _test_alerting_system(self):
        """Test alerting system component."""
        logger.info("Testing Alerting System...")
        
        try:
            # Mock Redis client for testing
            class MockRedis:
                def __init__(self):
                    self.data = {}
                    
                async def setex(self, key, ttl, value):
                    self.data[key] = value
                    
                async def get(self, key):
                    return self.data.get(key)
                    
                async def ping(self):
                    return True
                    
            mock_redis = MockRedis()
            
            # Test alerting system initialization
            alerting_system = EnhancedAlertingSystem(mock_redis)
            
            # Test alert creation and processing
            test_alert = EnhancedAlert(
                id="test_alert_001",
                timestamp=datetime.utcnow(),
                priority=AlertPriority.HIGH,
                status=AlertStatus.ACTIVE,
                source="test_system",
                alert_type="test_alert",
                title="Test Alert",
                message="This is a test alert",
                metrics={'test_metric': 100},
                tags={'test_tag'}
            )
            
            # Test alert processing
            result = await alerting_system.process_alert(test_alert)
            assert result is True
            
            # Test alert status
            alert_status = await alerting_system.get_alert_status()
            assert 'timestamp' in alert_status
            assert 'active_alerts' in alert_status
            
            self.test_results['tests']['alerting_system'] = 'PASS'
            self.test_results['components_tested'].append('alerting_system')
            
            logger.info("✅ Alerting System test passed")
            
        except Exception as e:
            logger.error(f"❌ Alerting System test failed: {e}")
            self.test_results['tests']['alerting_system'] = f'FAIL: {e}'
            self.test_results['overall_status'] = 'FAIL'
            
    async def _test_monitoring_integration(self):
        """Test monitoring integration component."""
        logger.info("Testing Monitoring Integration...")
        
        try:
            # Test monitoring system initialization
            config = MonitoringConfig(
                redis_host="localhost",
                redis_port=6379,
                enable_metrics_server=False,  # Disable for testing
                metrics_port=8002
            )
            
            monitoring_system = MonitoringSystem(config)
            
            # Test initialization
            await monitoring_system.initialize()
            assert monitoring_system.initialized is True
            
            # Test system test
            test_results = await monitoring_system.test_system()
            assert test_results['overall_status'] in ['PASS', 'FAIL']
            
            # Test status collection
            status = await monitoring_system.get_system_status()
            assert 'timestamp' in status
            
            self.test_results['tests']['monitoring_integration'] = 'PASS'
            self.test_results['components_tested'].append('monitoring_integration')
            
            logger.info("✅ Monitoring Integration test passed")
            
        except Exception as e:
            logger.error(f"❌ Monitoring Integration test failed: {e}")
            self.test_results['tests']['monitoring_integration'] = f'FAIL: {e}'
            self.test_results['overall_status'] = 'FAIL'
            
    async def _test_end_to_end_monitoring(self):
        """Test end-to-end monitoring workflow."""
        logger.info("Testing End-to-End Monitoring...")
        
        try:
            # Create full monitoring system
            config = MonitoringConfig(
                enable_metrics_server=False,
                metrics_port=8003
            )
            
            monitoring_system = MonitoringSystem(config)
            
            # Test full lifecycle
            await monitoring_system.initialize()
            
            # Start monitoring (but don't wait for full execution)
            start_task = asyncio.create_task(monitoring_system.start())
            
            # Wait a bit for startup
            await asyncio.sleep(1)
            
            # Test status while running
            status = await monitoring_system.get_system_status()
            assert status is not None
            
            # Test health summary
            health_summary = await monitoring_system.get_health_summary()
            assert health_summary is not None
            
            # Stop monitoring
            await monitoring_system.stop()
            
            # Cancel the start task
            start_task.cancel()
            
            self.test_results['tests']['end_to_end_monitoring'] = 'PASS'
            
            logger.info("✅ End-to-End Monitoring test passed")
            
        except Exception as e:
            logger.error(f"❌ End-to-End Monitoring test failed: {e}")
            self.test_results['tests']['end_to_end_monitoring'] = f'FAIL: {e}'
            self.test_results['overall_status'] = 'FAIL'
            
    async def _test_performance_under_load(self):
        """Test performance under load."""
        logger.info("Testing Performance Under Load...")
        
        try:
            # Create metrics collector for performance testing
            config = MetricsConfig(
                enable_system_metrics=True,
                metrics_port=8004
            )
            
            metrics_collector = MetricsCollector(config)
            metrics_collector.start_collection()
            
            # Generate load
            start_time = time.time()
            num_metrics = 1000
            
            for i in range(num_metrics):
                metrics_collector.trading_metrics.record_signal('test_strategy', 'buy', 'BTCUSD')
                metrics_collector.trading_metrics.update_pnl('test_strategy', 'BTCUSD', float(i))
                
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Calculate performance metrics
            metrics_per_second = num_metrics / processing_time
            
            # Performance assertions
            assert metrics_per_second > 100  # Should process at least 100 metrics/second
            assert processing_time < 10  # Should complete in under 10 seconds
            
            metrics_collector.stop_collection()
            
            self.test_results['performance_metrics'] = {
                'metrics_per_second': metrics_per_second,
                'processing_time': processing_time,
                'total_metrics': num_metrics
            }
            
            self.test_results['tests']['performance_under_load'] = 'PASS'
            
            logger.info(f"✅ Performance test passed: {metrics_per_second:.1f} metrics/second")
            
        except Exception as e:
            logger.error(f"❌ Performance test failed: {e}")
            self.test_results['tests']['performance_under_load'] = f'FAIL: {e}'
            self.test_results['overall_status'] = 'FAIL'
            
    def _generate_test_report(self):
        """Generate comprehensive test report."""
        logger.info("Generating test report...")
        
        # Calculate test statistics
        total_tests = len(self.test_results['tests'])
        passed_tests = len([t for t in self.test_results['tests'].values() if t == 'PASS'])
        failed_tests = total_tests - passed_tests
        
        # Create summary
        self.test_results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            'components_tested': len(self.test_results['components_tested']),
            'test_duration': datetime.utcnow().isoformat()
        }
        
        # Determine overall status
        if failed_tests == 0:
            self.test_results['overall_status'] = 'PASS'
        else:
            self.test_results['overall_status'] = 'FAIL'
            
        logger.info(f"Test Summary: {passed_tests}/{total_tests} tests passed ({self.test_results['summary']['success_rate']:.1f}%)")

async def main():
    """Run monitoring system validation."""
    print("🔍 GrandModel Monitoring System Validation")
    print("=" * 50)
    
    validator = MonitoringSystemValidator()
    
    try:
        # Run all tests
        test_results = await validator.run_all_tests()
        
        # Print results
        print("\n📊 Test Results:")
        print(json.dumps(test_results, indent=2))
        
        # Print summary
        summary = test_results['summary']
        print(f"\n🎯 Summary:")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Components Tested: {summary['components_tested']}")
        
        # Print performance metrics
        if test_results['performance_metrics']:
            perf = test_results['performance_metrics']
            print(f"\n⚡ Performance Metrics:")
            print(f"Metrics/Second: {perf['metrics_per_second']:.1f}")
            print(f"Processing Time: {perf['processing_time']:.2f}s")
            print(f"Total Metrics: {perf['total_metrics']}")
            
        # Final status
        if test_results['overall_status'] == 'PASS':
            print("\n✅ ALL TESTS PASSED - MONITORING SYSTEM VALIDATED")
        else:
            print("\n❌ SOME TESTS FAILED - REVIEW REQUIRED")
            
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        logger.error(f"Validation failed: {e}")
        
    print("\n🏁 Validation Complete")

if __name__ == "__main__":
    asyncio.run(main())