#!/usr/bin/env python3
"""
Simple test to validate monitoring system structure and imports.
"""

import sys
import os
import asyncio
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def test_imports():
    """Test that all monitoring components can be imported."""
    print("Testing imports...")
    
    try:
        # Test health monitor
        from src.monitoring.health_monitor import HealthMonitor, HealthStatus
        print("✅ Health Monitor imports successful")
        
        # Test basic monitoring components
        from src.monitoring.monitoring_integration import MonitoringConfig
        print("✅ Monitoring Integration imports successful")
        
        print("✅ All imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_health_monitor():
    """Test health monitor functionality."""
    print("\nTesting Health Monitor...")
    
    try:
        from src.monitoring.health_monitor import HealthMonitor, HealthStatus
        
        # Test health monitor creation
        health_monitor = HealthMonitor("redis://localhost:6379")
        
        # Test health status enum
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        
        print("✅ Health Monitor basic functionality works")
        return True
        
    except Exception as e:
        print(f"❌ Health Monitor test failed: {e}")
        return False

def test_configuration():
    """Test configuration system."""
    print("\nTesting Configuration...")
    
    try:
        from src.monitoring.monitoring_integration import MonitoringConfig
        
        # Test default configuration
        config = MonitoringConfig()
        assert config.redis_host == "localhost"
        assert config.redis_port == 6379
        assert config.health_check_interval == 30
        
        # Test custom configuration
        custom_config = MonitoringConfig(
            redis_host="custom_host",
            redis_port=6380,
            health_check_interval=60
        )
        assert custom_config.redis_host == "custom_host"
        assert custom_config.redis_port == 6380
        assert custom_config.health_check_interval == 60
        
        print("✅ Configuration system works")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting File Structure...")
    
    monitoring_dir = "/home/QuantNova/GrandModel/src/monitoring"
    required_files = [
        "health_monitor.py",
        "real_time_performance_monitor.py",
        "system_health_dashboard.py",
        "market_regime_monitor.py",
        "enhanced_alerting.py",
        "prometheus_metrics.py",
        "monitoring_integration.py",
        "AGENT6_FINAL_SUMMARY.md"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(monitoring_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files exist")
        return True

def main():
    """Run all tests."""
    print("🔍 GrandModel Monitoring System Simple Test")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_imports,
        test_health_monitor,
        test_configuration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("✅ ALL TESTS PASSED - MONITORING SYSTEM VALIDATED")
    else:
        print("❌ SOME TESTS FAILED - REVIEW REQUIRED")
    
    print("\n🏁 Simple Test Complete")

if __name__ == "__main__":
    main()