#!/usr/bin/env python3
"""
Test script for the enhanced performance monitor with advanced anomaly detection
"""

import sys
import os
import time
import numpy as np
import logging
from pathlib import Path

# Add the data pipeline to the path
sys.path.append(str(Path(__file__).parent))

from performance_monitor import PerformanceMonitor, AnomalySeverity

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_enhanced_performance_monitor():
    """Test the enhanced performance monitor functionality"""
    print("🚀 Testing Enhanced Performance Monitor with Anomaly Detection")
    print("=" * 60)
    
    # Configuration for anomaly detection
    anomaly_config = {
        'statistical': {
            'z_threshold': 2.5,
            'iqr_multiplier': 1.5,
            'window_size': 50
        },
        'ml': {
            'method': 'isolation_forest',
            'contamination': 0.1
        },
        'pattern': {
            'correlation_threshold': 0.8,
            'peak_threshold': 2.0
        },
        'threshold_adaptation_rate': 0.05,
        'alert_cooldown': 60,
        'email': {
            'enabled': False  # Disable email for testing
        },
        'webhook': {
            'enabled': False  # Disable webhook for testing
        },
        'log': {
            'enabled': True,
            'level': 'INFO'
        }
    }
    
    # Create enhanced performance monitor
    monitor = PerformanceMonitor(enable_dashboard=False, anomaly_config=anomaly_config)
    
    print("✅ Enhanced Performance Monitor initialized")
    
    # Generate normal data
    print("\n📊 Generating normal performance data...")
    normal_data_points = 100
    
    for i in range(normal_data_points):
        # Generate normal metrics
        load_time = np.random.normal(0.5, 0.1)  # Normal loading time
        throughput = np.random.normal(1000, 100)  # Normal throughput
        memory_usage = np.random.normal(512, 50)  # Normal memory usage
        
        monitor.record_metric('data_load_time', max(0.1, load_time))
        monitor.record_metric('data_throughput', max(100, throughput))
        monitor.record_metric('memory_usage', max(100, memory_usage))
        
        if i % 20 == 0:
            print(f"   Generated {i+1}/{normal_data_points} normal data points")
        
        time.sleep(0.01)  # Small delay
    
    print("✅ Normal data generation completed")
    
    # Train anomaly detectors
    print("\n🤖 Training anomaly detectors...")
    monitor.train_anomaly_detectors()
    time.sleep(2)  # Allow training to complete
    print("✅ Anomaly detectors trained")
    
    # Generate anomalous data
    print("\n🚨 Generating anomalous data...")
    anomaly_data_points = 20
    
    for i in range(anomaly_data_points):
        # Generate anomalous metrics
        if i % 5 == 0:
            # Simulate loading time spike
            load_time = np.random.normal(2.0, 0.3)  # Anomalous loading time
            print(f"   🔴 Injecting loading time anomaly: {load_time:.3f}s")
        else:
            load_time = np.random.normal(0.5, 0.1)
        
        if i % 7 == 0:
            # Simulate throughput drop
            throughput = np.random.normal(300, 50)  # Anomalous throughput
            print(f"   🔴 Injecting throughput anomaly: {throughput:.1f} items/sec")
        else:
            throughput = np.random.normal(1000, 100)
        
        if i % 6 == 0:
            # Simulate memory spike
            memory_usage = np.random.normal(1024, 100)  # Anomalous memory usage
            print(f"   🔴 Injecting memory anomaly: {memory_usage:.1f} MB")
        else:
            memory_usage = np.random.normal(512, 50)
        
        monitor.record_metric('data_load_time', max(0.1, load_time))
        monitor.record_metric('data_throughput', max(100, throughput))
        monitor.record_metric('memory_usage', max(100, memory_usage))
        
        time.sleep(0.05)  # Small delay
    
    print("✅ Anomalous data generation completed")
    
    # Wait for anomaly detection to process
    print("\n⏳ Waiting for anomaly detection processing...")
    time.sleep(3)
    
    # Get comprehensive system health report
    print("\n📋 Generating System Health Report...")
    health_summary = monitor.get_performance_summary()
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 ENHANCED PERFORMANCE MONITOR RESULTS")
    print("=" * 60)
    
    # System Status
    system_status = health_summary.get('system_status', {})
    print(f"\n🏥 System Health: {system_status.get('overall_health', 'UNKNOWN')}")
    print(f"📈 Quality Score: {system_status.get('quality_score', 0):.1f}%")
    print(f"🔍 Active Detectors: {system_status.get('active_detectors', 0)}")
    print(f"⚠️  Anomalies (24h): {system_status.get('anomaly_count_24h', 0)}")
    print(f"🚨 Alerts (24h): {system_status.get('alert_count_24h', 0)}")
    
    # Quality Report
    quality_report = monitor.get_quality_report()
    if quality_report.get('status') != 'no_data':
        current_quality = quality_report.get('current_quality', {})
        print(f"\n📊 Data Quality Breakdown:")
        print(f"   Completeness: {current_quality.get('completeness', 0):.1f}%")
        print(f"   Consistency: {current_quality.get('consistency', 0):.1f}%")
        print(f"   Accuracy: {current_quality.get('accuracy', 0):.1f}%")
        print(f"   Timeliness: {current_quality.get('timeliness', 0):.1f}%")
        print(f"   Validity: {current_quality.get('validity', 0):.1f}%")
    
    # Anomaly Statistics
    anomaly_stats = monitor.get_anomaly_statistics()
    if anomaly_stats.get('status') != 'no_anomalies':
        print(f"\n🔍 Anomaly Detection Results:")
        print(f"   Total Anomalies: {anomaly_stats.get('total_anomalies', 0)}")
        
        by_severity = anomaly_stats.get('anomalies_by_severity', {})
        if by_severity:
            print(f"   By Severity:")
            for severity, count in by_severity.items():
                print(f"     {severity}: {count}")
        
        by_type = anomaly_stats.get('anomalies_by_type', {})
        if by_type:
            print(f"   By Type:")
            for anomaly_type, count in by_type.items():
                print(f"     {anomaly_type}: {count}")
        
        most_problematic = anomaly_stats.get('most_problematic_metrics', [])
        if most_problematic:
            print(f"   Most Problematic Metrics:")
            for metric, anomalies in most_problematic[:3]:
                print(f"     {metric}: {len(anomalies)} anomalies")
    
    # Alert Statistics
    alert_stats = monitor.get_alert_statistics()
    if alert_stats.get('status') != 'no_alerts':
        print(f"\n🚨 Alert Statistics:")
        print(f"   Total Alerts (24h): {alert_stats.get('total_alerts_24h', 0)}")
        print(f"   Total Alerts (All Time): {alert_stats.get('total_alerts_all_time', 0)}")
    
    print("\n" + "=" * 60)
    print("✅ Enhanced Performance Monitor Test Completed Successfully!")
    print("=" * 60)
    
    # Export results
    print("\n💾 Exporting results...")
    monitor.export_metrics('/tmp/enhanced_performance_metrics.json', 'json')
    print("✅ Results exported to /tmp/enhanced_performance_metrics.json")
    
    # Cleanup
    monitor.cleanup()
    print("🧹 Cleanup completed")

def main():
    """Main test function"""
    setup_logging()
    
    try:
        test_enhanced_performance_monitor()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())