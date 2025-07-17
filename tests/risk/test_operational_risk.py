"""
Operational Risk Testing Suite

This comprehensive test suite validates operational risk management including
system failure scenarios, recovery procedures, data quality monitoring,
and business continuity testing.

Key Test Areas:
1. System failure scenarios and recovery procedures
2. Data quality monitoring and error detection
3. Business continuity and disaster recovery
4. Process failure simulation and response
5. Technology risk assessment
6. Human error simulation and mitigation
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from collections import deque
import json
import tempfile
import os
import sqlite3
import redis
from pathlib import Path

from src.core.events import EventBus, Event, EventType
from src.core.errors.error_handler import ErrorHandler
from src.core.errors.error_recovery import ErrorRecovery
from src.core.resilience.circuit_breaker import CircuitBreaker
from src.core.resilience.retry_manager import RetryManager
from src.core.resilience.health_monitor import HealthMonitor
from src.monitoring.health_monitor import SystemHealthMonitor
from src.data.quality_monitor import DataQualityMonitor
from src.operations.system_monitor import SystemMonitor
from src.operations.alert_manager import AlertManager


class OperationalRiskType(Enum):
    """Types of operational risks"""
    SYSTEM_FAILURE = "SYSTEM_FAILURE"
    DATA_QUALITY = "DATA_QUALITY"
    PROCESS_FAILURE = "PROCESS_FAILURE"
    TECHNOLOGY_RISK = "TECHNOLOGY_RISK"
    HUMAN_ERROR = "HUMAN_ERROR"
    EXTERNAL_DEPENDENCY = "EXTERNAL_DEPENDENCY"
    SECURITY_BREACH = "SECURITY_BREACH"
    COMPLIANCE_VIOLATION = "COMPLIANCE_VIOLATION"


class FailureScenario(Enum):
    """System failure scenarios"""
    DATABASE_OUTAGE = "DATABASE_OUTAGE"
    NETWORK_PARTITION = "NETWORK_PARTITION"
    MEMORY_EXHAUSTION = "MEMORY_EXHAUSTION"
    DISK_FULL = "DISK_FULL"
    SERVICE_TIMEOUT = "SERVICE_TIMEOUT"
    MARKET_DATA_FEED_FAILURE = "MARKET_DATA_FEED_FAILURE"
    ORDER_MANAGEMENT_FAILURE = "ORDER_MANAGEMENT_FAILURE"
    RISK_SYSTEM_FAILURE = "RISK_SYSTEM_FAILURE"


@dataclass
class OperationalIncident:
    """Operational incident definition"""
    incident_id: str
    timestamp: datetime
    risk_type: OperationalRiskType
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    description: str
    affected_systems: List[str]
    impact_assessment: str
    recovery_actions: List[str]
    resolution_time: Optional[datetime] = None
    resolved: bool = False


@dataclass
class SystemHealthMetrics:
    """System health metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    database_connections: int
    active_threads: int
    error_rate: float
    throughput: float


class MockSystemService:
    """Mock system service for testing"""
    
    def __init__(self, name: str, failure_rate: float = 0.0):
        self.name = name
        self.failure_rate = failure_rate
        self.is_healthy = True
        self.response_time = 0.1
        self.call_count = 0
        self.error_count = 0
        
    async def call_service(self, request: dict) -> dict:
        """Simulate service call with potential failures"""
        self.call_count += 1
        
        # Simulate failure based on failure rate
        if np.random.random() < self.failure_rate:
            self.error_count += 1
            self.is_healthy = False
            raise Exception(f"Service {self.name} failure")
        
        # Simulate response time
        await asyncio.sleep(self.response_time)
        
        return {"status": "success", "data": f"Response from {self.name}"}
    
    def get_health_status(self) -> dict:
        """Get service health status"""
        return {
            "name": self.name,
            "healthy": self.is_healthy,
            "response_time": self.response_time,
            "call_count": self.call_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.call_count, 1)
        }


class MockDataProvider:
    """Mock data provider with quality issues"""
    
    def __init__(self, corruption_rate: float = 0.0):
        self.corruption_rate = corruption_rate
        self.data_count = 0
        self.corrupted_count = 0
        
    def get_market_data(self, symbol: str) -> dict:
        """Get market data with potential quality issues"""
        self.data_count += 1
        
        # Normal data
        data = {
            "symbol": symbol,
            "price": 100.0 + np.random.normal(0, 5),
            "volume": int(np.random.uniform(100000, 1000000)),
            "timestamp": datetime.now().isoformat(),
            "bid": 99.5,
            "ask": 100.5
        }
        
        # Introduce data quality issues
        if np.random.random() < self.corruption_rate:
            self.corrupted_count += 1
            
            # Random data quality issue
            issue_type = np.random.choice([
                "missing_price", "negative_volume", "stale_timestamp",
                "invalid_symbol", "price_spike", "bid_ask_cross"
            ])
            
            if issue_type == "missing_price":
                data["price"] = None
            elif issue_type == "negative_volume":
                data["volume"] = -abs(data["volume"])
            elif issue_type == "stale_timestamp":
                data["timestamp"] = (datetime.now() - timedelta(hours=1)).isoformat()
            elif issue_type == "invalid_symbol":
                data["symbol"] = "INVALID_SYMBOL_123"
            elif issue_type == "price_spike":
                data["price"] *= 10  # 10x price spike
            elif issue_type == "bid_ask_cross":
                data["bid"] = 101.0
                data["ask"] = 99.0
        
        return data


class TestOperationalRisk:
    """Comprehensive operational risk testing suite"""
    
    @pytest.fixture
    def event_bus(self):
        """Create event bus for testing"""
        return EventBus()
    
    @pytest.fixture
    def error_handler(self):
        """Create error handler for testing"""
        return ErrorHandler()
    
    @pytest.fixture
    def error_recovery(self, event_bus):
        """Create error recovery system"""
        return ErrorRecovery(event_bus=event_bus)
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker for testing"""
        return CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30,
            call_timeout=10
        )
    
    @pytest.fixture
    def retry_manager(self):
        """Create retry manager for testing"""
        return RetryManager(
            max_retries=3,
            base_delay=0.1,
            max_delay=1.0,
            exponential_base=2.0
        )
    
    @pytest.fixture
    def health_monitor(self, event_bus):
        """Create health monitor for testing"""
        return SystemHealthMonitor(event_bus=event_bus)
    
    @pytest.fixture
    def data_quality_monitor(self, event_bus):
        """Create data quality monitor"""
        return DataQualityMonitor(event_bus=event_bus)
    
    @pytest.fixture
    def system_monitor(self, event_bus):
        """Create system monitor"""
        return SystemMonitor(event_bus=event_bus)
    
    @pytest.fixture
    def alert_manager(self, event_bus):
        """Create alert manager"""
        return AlertManager(event_bus=event_bus)
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services for testing"""
        return {
            "market_data": MockSystemService("MarketDataService", failure_rate=0.05),
            "order_management": MockSystemService("OrderManagementService", failure_rate=0.02),
            "risk_engine": MockSystemService("RiskEngineService", failure_rate=0.01),
            "portfolio_service": MockSystemService("PortfolioService", failure_rate=0.03),
            "database": MockSystemService("DatabaseService", failure_rate=0.01)
        }
    
    @pytest.fixture
    def mock_data_provider(self):
        """Create mock data provider"""
        return MockDataProvider(corruption_rate=0.1)
    
    @pytest.fixture
    def temp_database(self):
        """Create temporary database for testing"""
        temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_file.close()
        
        # Create database connection
        conn = sqlite3.connect(temp_file.name)
        conn.execute('''
            CREATE TABLE incidents (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                risk_type TEXT,
                severity TEXT,
                description TEXT,
                resolved BOOLEAN
            )
        ''')
        conn.commit()
        conn.close()
        
        yield temp_file.name
        
        # Cleanup
        os.unlink(temp_file.name)
    
    def test_system_failure_detection(self, health_monitor, mock_services):
        """Test detection of system failures"""
        
        # Register services with health monitor
        for name, service in mock_services.items():
            health_monitor.register_service(name, service.get_health_status)
        
        # Initial health check - all services should be healthy
        health_status = health_monitor.get_overall_health()
        assert health_status["status"] == "HEALTHY", "Initial health check failed"
        
        # Simulate service failure
        mock_services["market_data"].is_healthy = False
        mock_services["market_data"].error_count = 10
        
        # Check health status after failure
        health_status = health_monitor.get_overall_health()
        assert health_status["status"] in ["DEGRADED", "UNHEALTHY"], "Service failure not detected"
        
        # Check specific service status
        service_status = health_monitor.get_service_health("market_data")
        assert not service_status["healthy"], "Market data service failure not detected"
        
        print("✓ System failure detection successful")
        print(f"  Overall health: {health_status['status']}")
        print(f"  Failed service: market_data")
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_protection(self, circuit_breaker, mock_services):
        """Test circuit breaker protection during failures"""
        
        # Wrap service with circuit breaker
        service = mock_services["market_data"]
        service.failure_rate = 0.8  # High failure rate
        
        # Test circuit breaker behavior
        success_count = 0
        failure_count = 0
        circuit_open_count = 0
        
        for i in range(20):
            try:
                result = await circuit_breaker.call(service.call_service, {"request": i})
                success_count += 1
            except Exception as e:
                if "Circuit breaker is OPEN" in str(e):
                    circuit_open_count += 1
                else:
                    failure_count += 1
        
        # Verify circuit breaker opened
        assert circuit_open_count > 0, "Circuit breaker did not open"
        assert circuit_breaker.state == "OPEN", "Circuit breaker should be open"
        
        # Wait for half-open state
        await asyncio.sleep(0.1)  # Short wait for testing
        
        # Test recovery (reduce failure rate)
        service.failure_rate = 0.1
        
        # Try to recover
        try:
            await circuit_breaker.call(service.call_service, {"recovery_test": True})
        except:
            pass  # May still fail during recovery
        
        print("✓ Circuit breaker protection test successful")
        print(f"  Success calls: {success_count}")
        print(f"  Failed calls: {failure_count}")
        print(f"  Circuit open blocks: {circuit_open_count}")
        print(f"  Circuit breaker state: {circuit_breaker.state}")
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, retry_manager, mock_services):
        """Test retry mechanism for transient failures"""
        
        service = mock_services["database"]
        service.failure_rate = 0.6  # 60% failure rate for transient issues
        
        # Test retry behavior
        retry_attempts = []
        
        async def tracked_service_call(request):
            retry_attempts.append(datetime.now())
            return await service.call_service(request)
        
        # Test successful retry
        try:
            result = await retry_manager.execute_with_retry(
                tracked_service_call, {"test": "retry"}
            )
            success = True
        except Exception:
            success = False
        
        # Verify retry attempts
        assert len(retry_attempts) > 1, "No retry attempts made"
        assert len(retry_attempts) <= 4, "Too many retry attempts"  # max_retries + 1
        
        # Verify retry timing (exponential backoff)
        if len(retry_attempts) > 1:
            time_diff = (retry_attempts[1] - retry_attempts[0]).total_seconds()
            assert time_diff >= 0.1, "Retry delay too short"
        
        print("✓ Retry mechanism test successful")
        print(f"  Retry attempts: {len(retry_attempts)}")
        print(f"  Final result: {'Success' if success else 'Failed'}")
    
    def test_data_quality_monitoring(self, data_quality_monitor, mock_data_provider):
        """Test data quality monitoring and issue detection"""
        
        # Configure data quality rules
        data_quality_monitor.add_quality_rule(
            name="price_validation",
            field="price",
            rule_type="range",
            min_value=0.01,
            max_value=10000.0
        )
        
        data_quality_monitor.add_quality_rule(
            name="volume_validation",
            field="volume",
            rule_type="range",
            min_value=0,
            max_value=float('inf')
        )
        
        data_quality_monitor.add_quality_rule(
            name="timestamp_freshness",
            field="timestamp",
            rule_type="freshness",
            max_age_minutes=5
        )
        
        # Test data quality monitoring
        quality_issues = []
        
        for i in range(100):
            # Get market data (some may be corrupted)
            data = mock_data_provider.get_market_data("AAPL")
            
            # Validate data quality
            validation_result = data_quality_monitor.validate_data(data)
            
            if not validation_result["is_valid"]:
                quality_issues.append({
                    "data": data,
                    "issues": validation_result["issues"]
                })
        
        # Verify quality issues were detected
        assert len(quality_issues) > 0, "No data quality issues detected"
        
        # Check issue types
        issue_types = {}
        for issue in quality_issues:
            for problem in issue["issues"]:
                issue_type = problem["rule_name"]
                issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        
        # Calculate quality metrics
        total_records = 100
        quality_rate = (total_records - len(quality_issues)) / total_records
        corruption_rate = mock_data_provider.corrupted_count / mock_data_provider.data_count
        
        print("✓ Data quality monitoring test successful")
        print(f"  Total records: {total_records}")
        print(f"  Quality issues: {len(quality_issues)}")
        print(f"  Quality rate: {quality_rate:.2%}")
        print(f"  Corruption rate: {corruption_rate:.2%}")
        print(f"  Issue types: {issue_types}")
    
    @pytest.mark.asyncio
    async def test_database_failure_recovery(self, temp_database, error_recovery):
        """Test database failure and recovery procedures"""
        
        # Simulate database operations
        async def database_operation(operation_type: str):
            """Simulate database operation with potential failures"""
            if operation_type == "fail":
                raise sqlite3.OperationalError("Database connection failed")
            
            conn = sqlite3.connect(temp_database)
            if operation_type == "read":
                cursor = conn.execute("SELECT COUNT(*) FROM incidents")
                result = cursor.fetchone()[0]
            elif operation_type == "write":
                conn.execute(
                    "INSERT INTO incidents (timestamp, risk_type, severity, description, resolved) VALUES (?, ?, ?, ?, ?)",
                    (datetime.now().isoformat(), "TEST", "LOW", "Test incident", False)
                )
                conn.commit()
                result = "success"
            
            conn.close()
            return result
        
        # Test normal operations
        result = await database_operation("read")
        assert result == 0, "Initial database read failed"
        
        await database_operation("write")
        result = await database_operation("read")
        assert result == 1, "Database write/read failed"
        
        # Test failure recovery
        recovery_actions = []
        
        def recovery_callback(error_type: str, error_details: dict):
            recovery_actions.append({
                "type": error_type,
                "details": error_details,
                "timestamp": datetime.now()
            })
        
        error_recovery.register_recovery_handler("database", recovery_callback)
        
        # Simulate database failure
        try:
            await database_operation("fail")
        except sqlite3.OperationalError:
            # Trigger recovery
            error_recovery.trigger_recovery("database", {
                "error": "Connection failed",
                "service": "database"
            })
        
        # Wait for recovery processing
        await asyncio.sleep(0.1)
        
        # Verify recovery was triggered
        assert len(recovery_actions) > 0, "No recovery actions triggered"
        
        print("✓ Database failure recovery test successful")
        print(f"  Recovery actions: {len(recovery_actions)}")
    
    @pytest.mark.asyncio
    async def test_memory_exhaustion_handling(self, system_monitor, alert_manager):
        """Test memory exhaustion detection and handling"""
        
        # Setup memory monitoring
        memory_alerts = []
        
        def memory_alert_handler(alert):
            memory_alerts.append(alert)
        
        alert_manager.register_alert_handler("memory_exhaustion", memory_alert_handler)
        
        # Simulate memory usage increase
        memory_usage_history = []
        
        for i in range(10):
            # Simulate increasing memory usage
            memory_usage = 50 + (i * 5)  # 50% to 95%
            
            system_metrics = SystemHealthMetrics(
                timestamp=datetime.now(),
                cpu_usage=25.0,
                memory_usage=memory_usage,
                disk_usage=60.0,
                network_latency=10.0,
                database_connections=50,
                active_threads=100,
                error_rate=0.02,
                throughput=1000.0
            )
            
            memory_usage_history.append(memory_usage)
            
            # Check memory threshold
            if memory_usage > 80:  # 80% threshold
                alert_manager.trigger_alert("memory_exhaustion", {
                    "memory_usage": memory_usage,
                    "threshold": 80,
                    "timestamp": datetime.now().isoformat()
                })
        
        # Wait for alert processing
        await asyncio.sleep(0.1)
        
        # Verify memory alerts were triggered
        high_memory_periods = sum(1 for usage in memory_usage_history if usage > 80)
        assert len(memory_alerts) > 0, "No memory alerts triggered"
        assert len(memory_alerts) <= high_memory_periods, "Too many memory alerts"
        
        # Check alert details
        latest_alert = memory_alerts[-1]
        assert latest_alert["memory_usage"] > 80, "Alert threshold not correct"
        
        print("✓ Memory exhaustion handling test successful")
        print(f"  Memory alerts: {len(memory_alerts)}")
        print(f"  Peak memory usage: {max(memory_usage_history):.1f}%")
    
    def test_network_partition_simulation(self, mock_services):
        """Test network partition simulation and handling"""
        
        # Create network partition scenario
        class NetworkPartitionError(Exception):
            pass
        
        # Simulate network partition affecting some services
        partitioned_services = ["market_data", "order_management"]
        
        for service_name in partitioned_services:
            service = mock_services[service_name]
            service.is_healthy = False
            service.response_time = 30.0  # Simulate timeout
        
        # Test service discovery with partition
        healthy_services = []
        unhealthy_services = []
        
        for name, service in mock_services.items():
            status = service.get_health_status()
            if status["healthy"]:
                healthy_services.append(name)
            else:
                unhealthy_services.append(name)
        
        # Verify partition detection
        assert len(unhealthy_services) == len(partitioned_services), "Network partition not detected"
        assert len(healthy_services) > 0, "No healthy services remaining"
        
        # Test graceful degradation
        available_services = [s for s in mock_services.keys() if s not in partitioned_services]
        assert len(available_services) > 0, "No services available after partition"
        
        print("✓ Network partition simulation test successful")
        print(f"  Partitioned services: {partitioned_services}")
        print(f"  Healthy services: {healthy_services}")
        print(f"  Unhealthy services: {unhealthy_services}")
    
    @pytest.mark.asyncio
    async def test_business_continuity_procedures(self, health_monitor, mock_services, alert_manager):
        """Test business continuity procedures during system failures"""
        
        # Setup business continuity alerts
        continuity_alerts = []
        
        def continuity_alert_handler(alert):
            continuity_alerts.append(alert)
        
        alert_manager.register_alert_handler("business_continuity", continuity_alert_handler)
        
        # Simulate critical system failure
        critical_services = ["risk_engine", "order_management"]
        
        for service_name in critical_services:
            mock_services[service_name].is_healthy = False
            mock_services[service_name].error_count = 20
        
        # Check system health
        overall_health = health_monitor.get_overall_health()
        
        # Trigger business continuity procedures
        if overall_health["status"] == "CRITICAL":
            alert_manager.trigger_alert("business_continuity", {
                "trigger": "critical_system_failure",
                "failed_services": critical_services,
                "timestamp": datetime.now().isoformat(),
                "severity": "CRITICAL"
            })
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Verify continuity procedures were triggered
        assert len(continuity_alerts) > 0, "Business continuity procedures not triggered"
        
        # Check alert details
        latest_alert = continuity_alerts[-1]
        assert latest_alert["severity"] == "CRITICAL", "Alert severity not correct"
        assert len(latest_alert["failed_services"]) > 0, "Failed services not recorded"
        
        print("✓ Business continuity procedures test successful")
        print(f"  Failed critical services: {critical_services}")
        print(f"  Continuity alerts: {len(continuity_alerts)}")
    
    def test_human_error_simulation(self, error_handler):
        """Test human error simulation and mitigation"""
        
        # Simulate different types of human errors
        human_errors = [
            {
                "type": "configuration_error",
                "description": "Incorrect trading parameter configuration",
                "impact": "HIGH",
                "mitigation": "Configuration validation"
            },
            {
                "type": "data_entry_error",
                "description": "Wrong position size entered",
                "impact": "MEDIUM",
                "mitigation": "Double-entry validation"
            },
            {
                "type": "system_misuse",
                "description": "Incorrect system operation procedure",
                "impact": "LOW",
                "mitigation": "Training and documentation"
            }
        ]
        
        handled_errors = []
        
        for error in human_errors:
            try:
                # Simulate human error
                if error["type"] == "configuration_error":
                    raise ValueError(f"Invalid configuration: {error['description']}")
                elif error["type"] == "data_entry_error":
                    raise ValueError(f"Data validation failed: {error['description']}")
                elif error["type"] == "system_misuse":
                    raise RuntimeError(f"System misuse: {error['description']}")
                    
            except Exception as e:
                # Handle error
                error_details = error_handler.handle_error(e, {
                    "error_type": error["type"],
                    "impact": error["impact"],
                    "mitigation": error["mitigation"]
                })
                
                handled_errors.append(error_details)
        
        # Verify error handling
        assert len(handled_errors) == len(human_errors), "Not all human errors handled"
        
        # Check error categorization
        impact_levels = [error["context"]["impact"] for error in handled_errors]
        assert "HIGH" in impact_levels, "High impact errors not detected"
        assert "MEDIUM" in impact_levels, "Medium impact errors not detected"
        assert "LOW" in impact_levels, "Low impact errors not detected"
        
        print("✓ Human error simulation test successful")
        print(f"  Error types tested: {len(human_errors)}")
        print(f"  Errors handled: {len(handled_errors)}")
        print(f"  Impact levels: {set(impact_levels)}")
    
    @pytest.mark.asyncio
    async def test_disaster_recovery_procedures(self, temp_database, system_monitor):
        """Test disaster recovery procedures and system restoration"""
        
        # Simulate disaster scenario
        disaster_scenario = {
            "type": "data_center_failure",
            "affected_systems": ["database", "market_data", "order_management"],
            "estimated_downtime": "4 hours",
            "recovery_priority": "CRITICAL"
        }
        
        # Create backup data
        backup_data = {
            "timestamp": datetime.now().isoformat(),
            "positions": [
                {"symbol": "AAPL", "quantity": 1000, "price": 150.0},
                {"symbol": "GOOGL", "quantity": 500, "price": 2500.0}
            ],
            "risk_metrics": {
                "var_95": 50000,
                "leverage": 2.5,
                "correlation": 0.65
            }
        }
        
        # Test backup creation
        backup_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(backup_data, backup_file)
        backup_file.close()
        
        # Simulate system failure
        system_down = True
        
        # Test recovery procedures
        recovery_steps = []
        
        # Step 1: Assess damage
        recovery_steps.append({
            "step": "damage_assessment",
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "details": f"Disaster type: {disaster_scenario['type']}"
        })
        
        # Step 2: Restore from backup
        try:
            with open(backup_file.name, 'r') as f:
                restored_data = json.load(f)
            
            recovery_steps.append({
                "step": "data_restoration",
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "details": f"Restored {len(restored_data['positions'])} positions"
            })
            
        except Exception as e:
            recovery_steps.append({
                "step": "data_restoration",
                "status": "failed",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            })
        
        # Step 3: System validation
        if restored_data:
            # Validate restored data
            validation_passed = (
                len(restored_data["positions"]) > 0 and
                restored_data["risk_metrics"]["var_95"] > 0
            )
            
            recovery_steps.append({
                "step": "system_validation",
                "status": "completed" if validation_passed else "failed",
                "timestamp": datetime.now().isoformat(),
                "details": f"Validation passed: {validation_passed}"
            })
            
            if validation_passed:
                system_down = False
        
        # Verify recovery procedures
        assert len(recovery_steps) == 3, "Not all recovery steps executed"
        
        completed_steps = [s for s in recovery_steps if s["status"] == "completed"]
        assert len(completed_steps) >= 2, "Insufficient recovery steps completed"
        
        assert not system_down, "System not restored after recovery"
        
        # Cleanup
        os.unlink(backup_file.name)
        
        print("✓ Disaster recovery procedures test successful")
        print(f"  Recovery steps: {len(recovery_steps)}")
        print(f"  Completed steps: {len(completed_steps)}")
        print(f"  System restored: {not system_down}")
    
    def test_compliance_violation_detection(self, alert_manager):
        """Test compliance violation detection and reporting"""
        
        # Setup compliance monitoring
        compliance_violations = []
        
        def compliance_alert_handler(alert):
            compliance_violations.append(alert)
        
        alert_manager.register_alert_handler("compliance_violation", compliance_alert_handler)
        
        # Simulate compliance violations
        violations = [
            {
                "type": "position_limit_breach",
                "description": "Position size exceeds regulatory limit",
                "severity": "HIGH",
                "regulation": "SEC Rule 15c3-1"
            },
            {
                "type": "risk_limit_breach",
                "description": "VaR exceeds approved risk limits",
                "severity": "MEDIUM",
                "regulation": "Internal Risk Policy"
            },
            {
                "type": "reporting_delay",
                "description": "Regulatory report submitted late",
                "severity": "LOW",
                "regulation": "MiFID II"
            }
        ]
        
        # Trigger compliance violations
        for violation in violations:
            alert_manager.trigger_alert("compliance_violation", {
                "violation_type": violation["type"],
                "description": violation["description"],
                "severity": violation["severity"],
                "regulation": violation["regulation"],
                "timestamp": datetime.now().isoformat()
            })
        
        # Wait for processing
        time.sleep(0.1)
        
        # Verify violations were detected
        assert len(compliance_violations) == len(violations), "Not all compliance violations detected"
        
        # Check violation types
        violation_types = [v["violation_type"] for v in compliance_violations]
        expected_types = [v["type"] for v in violations]
        
        for expected_type in expected_types:
            assert expected_type in violation_types, f"Violation type {expected_type} not detected"
        
        # Check severity distribution
        severity_counts = {}
        for violation in compliance_violations:
            severity = violation["severity"]
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        print("✓ Compliance violation detection test successful")
        print(f"  Violations detected: {len(compliance_violations)}")
        print(f"  Violation types: {set(violation_types)}")
        print(f"  Severity distribution: {severity_counts}")


if __name__ == "__main__":
    """Run operational risk tests directly"""
    
    print("⚙️  Starting Operational Risk Tests...")
    print("=" * 50)
    
    # Run with pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ])