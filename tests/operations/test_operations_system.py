"""
Comprehensive Operations System Test Suite

This test suite validates the operations system including workflow management,
system monitoring, alerting, and operational controls.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any

from src.operations.workflow_manager import (
    WorkflowManager, WorkflowDefinition, WorkflowExecution,
    TaskDefinition, WorkflowStatus, TaskStatus, WorkflowTrigger
)
from src.operations.system_monitor import (
    SystemMonitor, SystemMetrics, AlertRule, AlertSeverity, HealthCheck
)
from src.operations.alert_manager import (
    AlertManager, Alert, AlertChannel, AlertPriority, AlertStatus, ChannelType
)
from src.operations.operational_controls import (
    OperationalControls, ControlAction, ActionType, CircuitBreakerState, RateLimiterState
)
from src.core.event_bus import EventBus, Event, EventType


class TestWorkflowManager:
    """Test suite for WorkflowManager"""
    
    @pytest.fixture
    def event_bus(self):
        return EventBus()
    
    @pytest.fixture
    def workflow_manager(self, event_bus):
        return WorkflowManager(event_bus)
    
    @pytest.fixture
    def sample_workflow(self):
        """Sample workflow definition"""
        tasks = [
            TaskDefinition(
                task_id="task1",
                task_name="Health Check",
                task_type="system",
                parameters={"action": "health_check", "service": "api"},
                timeout=30
            ),
            TaskDefinition(
                task_id="task2",
                task_name="Data Validation",
                task_type="data",
                parameters={"action": "validate_data", "data_source": "market_data"},
                dependencies=["task1"],
                timeout=60
            ),
            TaskDefinition(
                task_id="task3",
                task_name="Send Notification",
                task_type="notification",
                parameters={"action": "send_email", "recipient": "admin@test.com"},
                dependencies=["task2"],
                timeout=15
            )
        ]
        
        return WorkflowDefinition(
            workflow_id="test_workflow",
            workflow_name="Test Workflow",
            description="A test workflow",
            tasks=tasks,
            trigger=WorkflowTrigger.MANUAL,
            max_execution_time=300
        )
    
    def test_workflow_manager_initialization(self, workflow_manager):
        """Test workflow manager initialization"""
        assert workflow_manager.event_bus is not None
        assert len(workflow_manager.task_executors) >= 3  # system, data, notification
        assert workflow_manager.total_executions == 0
        assert workflow_manager.successful_executions == 0
        assert workflow_manager.failed_executions == 0
    
    def test_create_workflow_success(self, workflow_manager, sample_workflow):
        """Test successful workflow creation"""
        result = workflow_manager.create_workflow(sample_workflow)
        assert result is True
        assert sample_workflow.workflow_id in workflow_manager.workflows
        
        stored_workflow = workflow_manager.get_workflow(sample_workflow.workflow_id)
        assert stored_workflow.workflow_name == sample_workflow.workflow_name
        assert len(stored_workflow.tasks) == 3
    
    def test_create_workflow_duplicate_task_ids(self, workflow_manager, sample_workflow):
        """Test workflow creation with duplicate task IDs"""
        # Add duplicate task ID
        sample_workflow.tasks.append(TaskDefinition(
            task_id="task1",  # Duplicate ID
            task_name="Duplicate Task",
            task_type="system",
            parameters={"action": "health_check"}
        ))
        
        result = workflow_manager.create_workflow(sample_workflow)
        assert result is False
        assert sample_workflow.workflow_id not in workflow_manager.workflows
    
    def test_create_workflow_invalid_dependencies(self, workflow_manager, sample_workflow):
        """Test workflow creation with invalid dependencies"""
        # Add task with non-existent dependency
        sample_workflow.tasks.append(TaskDefinition(
            task_id="task4",
            task_name="Invalid Dependency Task",
            task_type="system",
            parameters={"action": "health_check"},
            dependencies=["non_existent_task"]
        ))
        
        result = workflow_manager.create_workflow(sample_workflow)
        assert result is False
        assert sample_workflow.workflow_id not in workflow_manager.workflows
    
    def test_create_workflow_circular_dependencies(self, workflow_manager):
        """Test workflow creation with circular dependencies"""
        tasks = [
            TaskDefinition(
                task_id="task1",
                task_name="Task 1",
                task_type="system",
                parameters={"action": "health_check"},
                dependencies=["task2"]
            ),
            TaskDefinition(
                task_id="task2",
                task_name="Task 2",
                task_type="system",
                parameters={"action": "health_check"},
                dependencies=["task1"]  # Circular dependency
            )
        ]
        
        circular_workflow = WorkflowDefinition(
            workflow_id="circular_workflow",
            workflow_name="Circular Workflow",
            description="A workflow with circular dependencies",
            tasks=tasks,
            trigger=WorkflowTrigger.MANUAL
        )
        
        result = workflow_manager.create_workflow(circular_workflow)
        assert result is False
        assert circular_workflow.workflow_id not in workflow_manager.workflows
    
    @pytest.mark.asyncio
    async def test_execute_workflow_success(self, workflow_manager, sample_workflow):
        """Test successful workflow execution"""
        # Create workflow
        workflow_manager.create_workflow(sample_workflow)
        
        # Execute workflow
        execution = await workflow_manager.execute_workflow(sample_workflow.workflow_id)
        
        assert execution.workflow_id == sample_workflow.workflow_id
        assert execution.status == WorkflowStatus.COMPLETED
        assert execution.end_time is not None
        assert execution.completed_tasks == 3
        assert execution.failed_tasks == 0
        assert execution.success_rate == 1.0
        
        # Check statistics
        assert workflow_manager.total_executions == 1
        assert workflow_manager.successful_executions == 1
        assert workflow_manager.failed_executions == 0
    
    @pytest.mark.asyncio
    async def test_execute_workflow_not_found(self, workflow_manager):
        """Test workflow execution with non-existent workflow"""
        with pytest.raises(ValueError, match="Workflow not found"):
            await workflow_manager.execute_workflow("non_existent_workflow")
    
    @pytest.mark.asyncio
    async def test_execute_workflow_disabled(self, workflow_manager, sample_workflow):
        """Test workflow execution when workflow is disabled"""
        # Create and disable workflow
        sample_workflow.enabled = False
        workflow_manager.create_workflow(sample_workflow)
        
        with pytest.raises(ValueError, match="Workflow is disabled"):
            await workflow_manager.execute_workflow(sample_workflow.workflow_id)
    
    @pytest.mark.asyncio
    async def test_execute_workflow_concurrent_limit(self, workflow_manager, sample_workflow):
        """Test workflow execution with concurrent limit"""
        # Set concurrent limit to 1
        sample_workflow.max_concurrent_executions = 1
        workflow_manager.create_workflow(sample_workflow)
        
        # Start first execution (will be running)
        execution1_task = asyncio.create_task(
            workflow_manager.execute_workflow(sample_workflow.workflow_id)
        )
        
        # Wait a bit to ensure first execution starts
        await asyncio.sleep(0.1)
        
        # Try to start second execution (should fail due to limit)
        with pytest.raises(ValueError, match="Maximum concurrent executions reached"):
            await workflow_manager.execute_workflow(sample_workflow.workflow_id)
        
        # Wait for first execution to complete
        await execution1_task
    
    @pytest.mark.asyncio
    async def test_workflow_task_timeout(self, workflow_manager):
        """Test workflow task timeout handling"""
        # Create workflow with very short timeout
        tasks = [
            TaskDefinition(
                task_id="timeout_task",
                task_name="Timeout Task",
                task_type="system",
                parameters={"action": "health_check"},
                timeout=1  # Very short timeout
            )
        ]
        
        timeout_workflow = WorkflowDefinition(
            workflow_id="timeout_workflow",
            workflow_name="Timeout Workflow",
            description="A workflow with timeout",
            tasks=tasks,
            trigger=WorkflowTrigger.MANUAL
        )
        
        workflow_manager.create_workflow(timeout_workflow)
        
        # Mock task executor to simulate long-running task
        original_execute = workflow_manager.task_executors["system"].execute
        
        async def slow_execute(task, context):
            await asyncio.sleep(2)  # Longer than timeout
            return original_execute(task, context)
        
        workflow_manager.task_executors["system"].execute = slow_execute
        
        # Execute workflow
        execution = await workflow_manager.execute_workflow(timeout_workflow.workflow_id)
        
        assert execution.status == WorkflowStatus.FAILED
        assert execution.tasks["timeout_task"].status == TaskStatus.FAILED
        assert "timeout" in execution.tasks["timeout_task"].error.lower()
    
    @pytest.mark.asyncio
    async def test_workflow_task_retry(self, workflow_manager):
        """Test workflow task retry functionality"""
        # Create workflow with retry
        tasks = [
            TaskDefinition(
                task_id="retry_task",
                task_name="Retry Task",
                task_type="system",
                parameters={"action": "health_check"},
                retry_count=2,
                retry_delay=1
            )
        ]
        
        retry_workflow = WorkflowDefinition(
            workflow_id="retry_workflow",
            workflow_name="Retry Workflow",
            description="A workflow with retry",
            tasks=tasks,
            trigger=WorkflowTrigger.MANUAL
        )
        
        workflow_manager.create_workflow(retry_workflow)
        
        # Mock task executor to fail first attempts
        call_count = 0
        original_execute = workflow_manager.task_executors["system"].execute
        
        async def failing_execute(task, context):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail first 2 attempts
                raise Exception("Mock failure")
            return await original_execute(task, context)
        
        workflow_manager.task_executors["system"].execute = failing_execute
        
        # Execute workflow
        execution = await workflow_manager.execute_workflow(retry_workflow.workflow_id)
        
        assert execution.status == WorkflowStatus.COMPLETED
        assert execution.tasks["retry_task"].status == TaskStatus.COMPLETED
        assert execution.tasks["retry_task"].retry_count == 2
        assert call_count == 3  # Original + 2 retries
    
    def test_get_executions_filtering(self, workflow_manager, sample_workflow):
        """Test execution filtering"""
        # Create multiple executions (mock)
        executions = []
        for i in range(5):
            execution = WorkflowExecution(
                execution_id=f"exec_{i}",
                workflow_id=sample_workflow.workflow_id if i < 3 else "other_workflow",
                workflow_name=sample_workflow.workflow_name,
                status=WorkflowStatus.COMPLETED if i % 2 == 0 else WorkflowStatus.FAILED,
                start_time=datetime.now() - timedelta(hours=i)
            )
            executions.append(execution)
            workflow_manager.executions[execution.execution_id] = execution
        
        # Test filtering by workflow_id
        filtered = workflow_manager.get_executions(workflow_id=sample_workflow.workflow_id)
        assert len(filtered) == 3
        
        # Test filtering by status
        filtered = workflow_manager.get_executions(status=WorkflowStatus.COMPLETED)
        assert len(filtered) == 3
        
        # Test limiting results
        filtered = workflow_manager.get_executions(limit=2)
        assert len(filtered) == 2
    
    def test_cleanup_old_executions(self, workflow_manager):
        """Test cleanup of old executions"""
        # Create old executions
        for i in range(5):
            execution = WorkflowExecution(
                execution_id=f"old_exec_{i}",
                workflow_id="test_workflow",
                workflow_name="Test Workflow",
                status=WorkflowStatus.COMPLETED,
                start_time=datetime.now() - timedelta(days=35)  # Older than 30 days
            )
            workflow_manager.executions[execution.execution_id] = execution
        
        # Create recent executions
        for i in range(3):
            execution = WorkflowExecution(
                execution_id=f"recent_exec_{i}",
                workflow_id="test_workflow",
                workflow_name="Test Workflow",
                status=WorkflowStatus.COMPLETED,
                start_time=datetime.now() - timedelta(days=10)  # Recent
            )
            workflow_manager.executions[execution.execution_id] = execution
        
        # Cleanup old executions
        removed_count = workflow_manager.cleanup_old_executions(days_old=30)
        
        assert removed_count == 5
        assert len(workflow_manager.executions) == 3
    
    def test_get_manager_status(self, workflow_manager, sample_workflow):
        """Test workflow manager status"""
        # Create workflow
        workflow_manager.create_workflow(sample_workflow)
        
        # Set some statistics
        workflow_manager.total_executions = 10
        workflow_manager.successful_executions = 8
        workflow_manager.failed_executions = 2
        
        status = workflow_manager.get_manager_status()
        
        assert status["total_workflows"] == 1
        assert status["active_workflows"] == 1
        assert status["total_executions"] == 10
        assert status["successful_executions"] == 8
        assert status["failed_executions"] == 2
        assert status["success_rate"] == 0.8
        assert status["running_executions"] == 0
        assert "system" in status["registered_executors"]
        assert "data" in status["registered_executors"]
        assert "notification" in status["registered_executors"]


class TestSystemMonitor:
    """Test suite for SystemMonitor"""
    
    @pytest.fixture
    def event_bus(self):
        return EventBus()
    
    @pytest.fixture
    def system_monitor(self, event_bus):
        return SystemMonitor(event_bus)
    
    def test_system_monitor_initialization(self, system_monitor):
        """Test system monitor initialization"""
        assert system_monitor.event_bus is not None
        assert system_monitor.is_running is False
        assert len(system_monitor.alert_rules) > 0
        assert len(system_monitor.health_checks) > 0
        assert system_monitor.monitoring_interval == 10
        assert system_monitor.retention_hours == 24
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, system_monitor):
        """Test start and stop monitoring"""
        # Start monitoring
        await system_monitor.start_monitoring()
        assert system_monitor.is_running is True
        assert system_monitor.monitoring_task is not None
        
        # Stop monitoring
        await system_monitor.stop_monitoring()
        assert system_monitor.is_running is False
    
    @pytest.mark.asyncio
    async def test_collect_metrics(self, system_monitor):
        """Test metrics collection"""
        metrics = await system_monitor._collect_metrics()
        
        assert isinstance(metrics, SystemMetrics)
        assert metrics.timestamp is not None
        assert metrics.cpu_usage >= 0
        assert metrics.memory_usage >= 0
        assert metrics.disk_usage >= 0
        assert metrics.process_count > 0
        assert len(metrics.load_average) == 3
    
    @pytest.mark.asyncio
    async def test_alert_rule_triggering(self, system_monitor):
        """Test alert rule triggering"""
        # Create mock metrics with high CPU usage
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=85.0,  # Above threshold
            memory_usage=50.0,
            memory_total=16.0,
            memory_available=8.0,
            disk_usage=60.0,
            disk_total=500.0,
            disk_available=200.0,
            network_io_sent=100.0,
            network_io_recv=200.0,
            load_average=[1.0, 1.5, 2.0],
            process_count=150,
            open_files=100,
            connections=50
        )
        
        # Mock event bus publish
        published_events = []
        
        async def mock_publish(event):
            published_events.append(event)
        
        system_monitor.event_bus.publish = mock_publish
        
        # Check alert rules
        await system_monitor._check_alert_rules(metrics)
        
        # Should trigger CPU high alert
        assert len(published_events) > 0
        alert_event = published_events[0]
        assert alert_event.type == EventType.ALERT
        assert "cpu_usage" in alert_event.payload["metric_name"]
        assert alert_event.payload["metric_value"] == 85.0
    
    @pytest.mark.asyncio
    async def test_health_check_execution(self, system_monitor):
        """Test health check execution"""
        # Mock health check function
        async def mock_health_check():
            return {"status": "healthy", "score": 95.0}
        
        health_check = HealthCheck(
            check_id="test_check",
            name="Test Health Check",
            description="A test health check",
            check_function=mock_health_check,
            interval=1
        )
        
        system_monitor.health_checks["test_check"] = health_check
        
        # Mock event bus publish
        published_events = []
        
        async def mock_publish(event):
            published_events.append(event)
        
        system_monitor.event_bus.publish = mock_publish
        
        # Run health check
        await system_monitor._run_health_check(health_check)
        
        # Verify health check result
        assert health_check.last_result["status"] == "healthy"
        assert health_check.last_result["score"] == 95.0
        assert health_check.failure_count == 0
        
        # Verify event published
        assert len(published_events) > 0
        health_event = published_events[0]
        assert health_event.type == EventType.HEALTH_CHECK
        assert health_event.payload["status"] == "passed"
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, system_monitor):
        """Test health check failure handling"""
        # Mock failing health check function
        async def failing_health_check():
            raise Exception("Health check failed")
        
        health_check = HealthCheck(
            check_id="failing_check",
            name="Failing Health Check",
            description="A failing health check",
            check_function=failing_health_check,
            interval=1
        )
        
        system_monitor.health_checks["failing_check"] = health_check
        
        # Mock event bus publish
        published_events = []
        
        async def mock_publish(event):
            published_events.append(event)
        
        system_monitor.event_bus.publish = mock_publish
        
        # Run health check
        await system_monitor._run_health_check(health_check)
        
        # Verify health check failure
        assert health_check.failure_count == 1
        assert health_check.last_result["status"] == "failed"
        assert "Health check failed" in health_check.last_result["error"]
        
        # Verify event published
        assert len(published_events) > 0
        health_event = published_events[0]
        assert health_event.type == EventType.HEALTH_CHECK
        assert health_event.payload["status"] == "failed"
    
    def test_add_remove_alert_rule(self, system_monitor):
        """Test adding and removing alert rules"""
        # Add alert rule
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            description="A test rule",
            metric_name="test_metric",
            condition="test_metric > 50",
            threshold=50.0,
            severity=AlertSeverity.WARNING
        )
        
        result = system_monitor.add_alert_rule(rule)
        assert result is True
        assert rule.rule_id in system_monitor.alert_rules
        
        # Remove alert rule
        result = system_monitor.remove_alert_rule(rule.rule_id)
        assert result is True
        assert rule.rule_id not in system_monitor.alert_rules
        
        # Try to remove non-existent rule
        result = system_monitor.remove_alert_rule("non_existent")
        assert result is False
    
    def test_custom_metrics(self, system_monitor):
        """Test custom metrics functionality"""
        # Set custom metrics
        system_monitor.set_custom_metric("test_metric", 42.0)
        system_monitor.set_custom_metric("another_metric", "test_value")
        
        # Verify custom metrics
        assert system_monitor.custom_metrics["test_metric"] == 42.0
        assert system_monitor.custom_metrics["another_metric"] == "test_value"
    
    def test_get_monitor_status(self, system_monitor):
        """Test monitor status"""
        # Set some statistics
        system_monitor.alerts_triggered = 5
        system_monitor.health_checks_run = 100
        system_monitor.health_checks_failed = 3
        
        status = system_monitor.get_monitor_status()
        
        assert status["is_running"] is False
        assert status["monitoring_interval"] == 10
        assert status["retention_hours"] == 24
        assert status["alerts_triggered"] == 5
        assert status["health_checks_run"] == 100
        assert status["health_checks_failed"] == 3
        assert status["alert_rules_count"] > 0
        assert status["health_checks_count"] > 0


class TestAlertManager:
    """Test suite for AlertManager"""
    
    @pytest.fixture
    def event_bus(self):
        return EventBus()
    
    @pytest.fixture
    def alert_manager(self, event_bus):
        return AlertManager(event_bus)
    
    @pytest.fixture
    def sample_alert(self):
        return Alert(
            alert_id="test_alert",
            title="Test Alert",
            description="A test alert",
            severity="warning",
            priority=AlertPriority.MEDIUM,
            source="test_source",
            tags={"component": "test", "environment": "test"}
        )
    
    def test_alert_manager_initialization(self, alert_manager):
        """Test alert manager initialization"""
        assert alert_manager.event_bus is not None
        assert alert_manager.is_running is False
        assert len(alert_manager.channels) >= 3  # email, slack, webhook
        assert alert_manager.total_alerts == 0
        assert alert_manager.alerts_processed == 0
    
    @pytest.mark.asyncio
    async def test_start_stop_processing(self, alert_manager):
        """Test start and stop alert processing"""
        # Start processing
        await alert_manager.start_processing()
        assert alert_manager.is_running is True
        assert alert_manager.processing_task is not None
        
        # Stop processing
        await alert_manager.stop_processing()
        assert alert_manager.is_running is False
    
    @pytest.mark.asyncio
    async def test_process_alert(self, alert_manager, sample_alert):
        """Test alert processing"""
        # Mock notification sending
        sent_notifications = []
        
        async def mock_send_notification(alert, channel, template=None):
            sent_notifications.append((alert.alert_id, channel.channel_id))
        
        alert_manager._send_notification = mock_send_notification
        
        # Process alert
        await alert_manager._process_alert(sample_alert)
        
        # Verify alert was processed
        assert sample_alert.alert_id in alert_manager.alerts
        assert alert_manager.alerts_processed == 1
        assert len(sent_notifications) > 0
    
    @pytest.mark.asyncio
    async def test_alert_suppression(self, alert_manager, sample_alert):
        """Test alert suppression"""
        # Add alert to alerts (simulating existing alert)
        alert_manager.alerts[sample_alert.alert_id] = sample_alert
        
        # Create duplicate alert
        duplicate_alert = Alert(
            alert_id="duplicate_alert",
            title=sample_alert.title,  # Same title
            description="Another test alert",
            severity="warning",
            priority=AlertPriority.MEDIUM,
            source=sample_alert.source,  # Same source
            status=AlertStatus.OPEN
        )
        
        # Check if duplicate should be suppressed
        should_suppress = alert_manager._should_suppress_alert(duplicate_alert)
        assert should_suppress is True
    
    @pytest.mark.asyncio
    async def test_alert_escalation(self, alert_manager, sample_alert):
        """Test alert escalation"""
        # Create escalation rule
        from src.operations.alert_manager import EscalationRule
        
        escalation_rule = EscalationRule(
            rule_id="test_escalation",
            name="Test Escalation",
            conditions={"priority": ["high", "critical"]},
            escalation_delay=5,  # 5 seconds
            escalation_channels=["email_default"]
        )
        
        alert_manager.escalation_rules[escalation_rule.rule_id] = escalation_rule
        
        # Create high priority alert
        high_priority_alert = Alert(
            alert_id="high_priority_alert",
            title="High Priority Alert",
            description="A high priority alert",
            severity="error",
            priority=AlertPriority.HIGH,
            source="critical_system",
            created_at=datetime.now() - timedelta(seconds=10)  # Created 10 seconds ago
        )
        
        alert_manager.alerts[high_priority_alert.alert_id] = high_priority_alert
        
        # Mock notification sending
        sent_notifications = []
        
        async def mock_send_notification(alert, channel, template=None):
            sent_notifications.append((alert.alert_id, channel.channel_id))
        
        alert_manager._send_notification = mock_send_notification
        
        # Check escalations
        await alert_manager._check_escalations()
        
        # Verify escalation occurred
        assert high_priority_alert.escalated is True
        assert high_priority_alert.escalation_level == 1
        assert len(sent_notifications) > 0
    
    @pytest.mark.asyncio
    async def test_acknowledge_alert(self, alert_manager, sample_alert):
        """Test alert acknowledgment"""
        # Add alert to manager
        alert_manager.alerts[sample_alert.alert_id] = sample_alert
        
        # Acknowledge alert
        result = await alert_manager.acknowledge_alert(sample_alert.alert_id, "user123")
        
        assert result is True
        assert sample_alert.status == AlertStatus.ACKNOWLEDGED
        assert sample_alert.acknowledged_by == "user123"
        assert sample_alert.acknowledged_at is not None
    
    @pytest.mark.asyncio
    async def test_resolve_alert(self, alert_manager, sample_alert):
        """Test alert resolution"""
        # Add alert to manager
        alert_manager.alerts[sample_alert.alert_id] = sample_alert
        
        # Resolve alert
        result = await alert_manager.resolve_alert(sample_alert.alert_id, "user123")
        
        assert result is True
        assert sample_alert.status == AlertStatus.RESOLVED
        assert sample_alert.resolved_at is not None
    
    def test_suppress_alerts(self, alert_manager):
        """Test alert suppression by source"""
        # Suppress alerts from source
        alert_manager.suppress_alerts("test_source", duration_minutes=30)
        
        # Verify suppression
        assert "test_source" in alert_manager.suppressed_alerts
        
        # Check suppression time
        suppression_time = alert_manager.suppressed_alerts["test_source"]
        expected_time = datetime.now() + timedelta(minutes=30)
        assert abs((suppression_time - expected_time).total_seconds()) < 5
    
    def test_add_remove_alert_channel(self, alert_manager):
        """Test adding and removing alert channels"""
        # Add alert channel
        channel = AlertChannel(
            channel_id="test_channel",
            channel_name="Test Channel",
            channel_type=ChannelType.EMAIL,
            configuration={"recipient": "test@example.com"}
        )
        
        result = alert_manager.add_alert_channel(channel)
        assert result is True
        assert channel.channel_id in alert_manager.channels
        
        # Remove alert channel
        result = alert_manager.remove_alert_channel(channel.channel_id)
        assert result is True
        assert channel.channel_id not in alert_manager.channels
    
    def test_get_active_alerts(self, alert_manager):
        """Test getting active alerts"""
        # Add some alerts
        alerts = [
            Alert(
                alert_id="alert1",
                title="Alert 1",
                description="First alert",
                severity="warning",
                priority=AlertPriority.LOW,
                source="source1",
                status=AlertStatus.OPEN
            ),
            Alert(
                alert_id="alert2",
                title="Alert 2",
                description="Second alert",
                severity="error",
                priority=AlertPriority.HIGH,
                source="source2",
                status=AlertStatus.ACKNOWLEDGED
            ),
            Alert(
                alert_id="alert3",
                title="Alert 3",
                description="Third alert",
                severity="critical",
                priority=AlertPriority.CRITICAL,
                source="source3",
                status=AlertStatus.RESOLVED
            )
        ]
        
        for alert in alerts:
            alert_manager.alerts[alert.alert_id] = alert
        
        # Get active alerts
        active_alerts = alert_manager.get_active_alerts()
        
        # Only one alert should be active (OPEN status)
        assert len(active_alerts) == 1
        assert active_alerts[0].alert_id == "alert1"
    
    def test_get_manager_status(self, alert_manager):
        """Test alert manager status"""
        # Set some statistics
        alert_manager.total_alerts = 50
        alert_manager.alerts_processed = 48
        alert_manager.notifications_sent = 120
        alert_manager.notifications_failed = 3
        
        status = alert_manager.get_manager_status()
        
        assert status["is_running"] is False
        assert status["total_alerts"] == 50
        assert status["alerts_processed"] == 48
        assert status["notifications_sent"] == 120
        assert status["notifications_failed"] == 3
        assert status["channels_count"] >= 3
        assert status["queue_size"] == 0


class TestOperationalControls:
    """Test suite for OperationalControls"""
    
    @pytest.fixture
    def event_bus(self):
        return EventBus()
    
    @pytest.fixture
    def operational_controls(self, event_bus):
        return OperationalControls(event_bus)
    
    def test_operational_controls_initialization(self, operational_controls):
        """Test operational controls initialization"""
        assert operational_controls.event_bus is not None
        assert operational_controls.is_running is False
        assert len(operational_controls.circuit_breakers) > 0
        assert len(operational_controls.rate_limiters) > 0
        assert len(operational_controls.control_actions) > 0
        assert operational_controls.system_status == "normal"
        assert operational_controls.maintenance_mode is False
        assert operational_controls.emergency_stop is False
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, operational_controls):
        """Test start and stop control monitoring"""
        # Start monitoring
        await operational_controls.start_monitoring()
        assert operational_controls.is_running is True
        assert operational_controls.monitoring_task is not None
        
        # Stop monitoring
        await operational_controls.stop_monitoring()
        assert operational_controls.is_running is False
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_success(self, operational_controls):
        """Test circuit breaker success case"""
        async with operational_controls.circuit_breaker("test_breaker"):
            # Simulate successful operation
            await asyncio.sleep(0.01)
        
        # Verify circuit breaker state
        breaker = operational_controls.circuit_breakers["test_breaker"]
        assert breaker.state == "CLOSED"
        assert breaker.successful_calls == 1
        assert breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure(self, operational_controls):
        """Test circuit breaker failure case"""
        breaker_name = "failing_breaker"
        
        # Simulate multiple failures to trip the breaker
        for i in range(6):  # Default threshold is 5
            try:
                async with operational_controls.circuit_breaker(breaker_name):
                    raise Exception(f"Simulated failure {i}")
            except Exception:
                pass  # Expected
        
        # Verify circuit breaker tripped
        breaker = operational_controls.circuit_breakers[breaker_name]
        assert breaker.state == "OPEN"
        assert breaker.failure_count >= 5
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_open_state(self, operational_controls):
        """Test circuit breaker in open state"""
        breaker_name = "open_breaker"
        
        # Manually set breaker to open state
        operational_controls.circuit_breakers[breaker_name] = CircuitBreakerState(
            name=breaker_name,
            state="OPEN",
            failure_count=5,
            last_failure=datetime.now()
        )
        
        # Try to use circuit breaker (should fail immediately)
        with pytest.raises(Exception, match="Circuit breaker .* is OPEN"):
            async with operational_controls.circuit_breaker(breaker_name):
                pass
    
    @pytest.mark.asyncio
    async def test_rate_limiter_allowed(self, operational_controls):
        """Test rate limiter allowing requests"""
        # Check rate limit (should be allowed)
        result = await operational_controls.check_rate_limit("test_limiter")
        assert result is True
        
        # Verify rate limiter state
        limiter = operational_controls.rate_limiters["test_limiter"]
        assert len(limiter.requests) == 1
        assert limiter.blocked_count == 0
    
    @pytest.mark.asyncio
    async def test_rate_limiter_exceeded(self, operational_controls):
        """Test rate limiter blocking requests"""
        limiter_name = "strict_limiter"
        
        # Add strict rate limiter
        operational_controls.add_rate_limiter(limiter_name, max_requests=2, time_window=60)
        
        # Make requests up to limit
        for i in range(2):
            result = await operational_controls.check_rate_limit(limiter_name)
            assert result is True
        
        # Next request should be blocked
        result = await operational_controls.check_rate_limit(limiter_name)
        assert result is False
        
        # Verify rate limiter state
        limiter = operational_controls.rate_limiters[limiter_name]
        assert len(limiter.requests) == 2
        assert limiter.blocked_count == 1
    
    @pytest.mark.asyncio
    async def test_execute_control_action_shutdown(self, operational_controls):
        """Test executing shutdown control action"""
        # Mock event bus publish
        published_events = []
        
        async def mock_publish(event):
            published_events.append(event)
        
        operational_controls.event_bus.publish = mock_publish
        
        # Execute shutdown action
        result = await operational_controls.execute_control_action("emergency_shutdown")
        
        assert result is True
        assert operational_controls.emergency_stop is True
        assert len(published_events) > 0
        
        # Verify shutdown event
        shutdown_event = published_events[0]
        assert shutdown_event.type == EventType.SYSTEM_SHUTDOWN
        assert shutdown_event.payload["graceful"] is True
    
    @pytest.mark.asyncio
    async def test_execute_control_action_throttle(self, operational_controls):
        """Test executing throttle control action"""
        # Execute throttle action
        result = await operational_controls.execute_control_action(
            "throttle_requests",
            context={"throttle_name": "api_throttle"}
        )
        
        assert result is True
        assert operational_controls.is_throttled("api_throttle")
        assert operational_controls.get_throttle_factor("api_throttle") == 0.5
    
    @pytest.mark.asyncio
    async def test_execute_control_action_maintenance(self, operational_controls):
        """Test executing maintenance mode control action"""
        # Execute maintenance mode action
        result = await operational_controls.execute_control_action("maintenance_mode")
        
        assert result is True
        assert operational_controls.maintenance_mode is True
    
    def test_feature_flags(self, operational_controls):
        """Test feature flag functionality"""
        # Set feature flag
        operational_controls.set_feature_flag("test_feature", True)
        assert operational_controls.get_feature_flag("test_feature") is True
        
        # Get non-existent flag with default
        assert operational_controls.get_feature_flag("non_existent", False) is False
        
        # Update existing flag
        operational_controls.set_feature_flag("test_feature", False)
        assert operational_controls.get_feature_flag("test_feature") is False
    
    def test_maintenance_mode_control(self, operational_controls):
        """Test maintenance mode control"""
        # Enable maintenance mode
        operational_controls.enable_maintenance_mode("Scheduled maintenance")
        assert operational_controls.maintenance_mode is True
        
        # Disable maintenance mode
        operational_controls.disable_maintenance_mode()
        assert operational_controls.maintenance_mode is False
    
    def test_emergency_stop_control(self, operational_controls):
        """Test emergency stop control"""
        # Trigger emergency stop
        operational_controls.trigger_emergency_stop()
        assert operational_controls.emergency_stop is True
        assert operational_controls.system_status == "emergency"
        
        # Reset emergency stop
        operational_controls.reset_emergency_stop()
        assert operational_controls.emergency_stop is False
        assert operational_controls.system_status == "normal"
    
    def test_add_circuit_breaker(self, operational_controls):
        """Test adding circuit breaker"""
        # Add circuit breaker
        operational_controls.add_circuit_breaker("custom_breaker", failure_threshold=10, recovery_timeout=120)
        
        # Verify circuit breaker was added
        assert "custom_breaker" in operational_controls.circuit_breakers
        breaker = operational_controls.circuit_breakers["custom_breaker"]
        assert breaker.name == "custom_breaker"
        assert breaker.failure_threshold == 10
        assert breaker.recovery_timeout == 120
    
    def test_add_rate_limiter(self, operational_controls):
        """Test adding rate limiter"""
        # Add rate limiter
        operational_controls.add_rate_limiter("custom_limiter", max_requests=50, time_window=30)
        
        # Verify rate limiter was added
        assert "custom_limiter" in operational_controls.rate_limiters
        limiter = operational_controls.rate_limiters["custom_limiter"]
        assert limiter.name == "custom_limiter"
        assert limiter.max_requests == 50
        assert limiter.time_window == 30
    
    def test_add_control_action(self, operational_controls):
        """Test adding control action"""
        # Add control action
        action = ControlAction(
            action_id="custom_action",
            action_type=ActionType.ALERT,
            description="Custom control action",
            parameters={"severity": "info", "message": "Custom alert"}
        )
        
        operational_controls.add_control_action(action)
        
        # Verify control action was added
        assert "custom_action" in operational_controls.control_actions
        stored_action = operational_controls.control_actions["custom_action"]
        assert stored_action.action_id == "custom_action"
        assert stored_action.action_type == ActionType.ALERT
    
    def test_get_system_status(self, operational_controls):
        """Test getting system status"""
        # Set some state
        operational_controls.maintenance_mode = True
        operational_controls.set_feature_flag("test_flag", True)
        operational_controls.total_control_actions = 10
        
        status = operational_controls.get_system_status()
        
        assert status["system_status"] == "normal"
        assert status["maintenance_mode"] is True
        assert status["emergency_stop"] is False
        assert len(status["circuit_breakers"]) > 0
        assert len(status["rate_limiters"]) > 0
        assert "test_flag" in status["feature_flags"]
        assert status["statistics"]["total_control_actions"] == 10
    
    def test_get_control_events(self, operational_controls):
        """Test getting control events"""
        # Add some control events manually
        events = [
            {
                "timestamp": datetime.now() - timedelta(hours=1),
                "action_id": "action1",
                "action_type": "shutdown",
                "context": {}
            },
            {
                "timestamp": datetime.now() - timedelta(hours=2),
                "action_id": "action2",
                "action_type": "throttle",
                "context": {"throttle_name": "api"}
            },
            {
                "timestamp": datetime.now() - timedelta(hours=30),  # Too old
                "action_id": "action3",
                "action_type": "alert",
                "context": {}
            }
        ]
        
        for event in events:
            operational_controls.control_events.append(event)
        
        # Get control events (last 24 hours)
        recent_events = operational_controls.get_control_events(hours=24)
        
        # Should only return 2 events (the third is older than 24 hours)
        assert len(recent_events) == 2
        assert all(isinstance(event["timestamp"], str) for event in recent_events)


class TestOperationsSystemIntegration:
    """Integration tests for the entire operations system"""
    
    @pytest.fixture
    def event_bus(self):
        return EventBus()
    
    @pytest.fixture
    def operations_system(self, event_bus):
        """Complete operations system setup"""
        return {
            "workflow_manager": WorkflowManager(event_bus),
            "system_monitor": SystemMonitor(event_bus),
            "alert_manager": AlertManager(event_bus),
            "operational_controls": OperationalControls(event_bus),
            "event_bus": event_bus
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow_execution(self, operations_system):
        """Test end-to-end workflow execution with monitoring and alerting"""
        workflow_manager = operations_system["workflow_manager"]
        system_monitor = operations_system["system_monitor"]
        alert_manager = operations_system["alert_manager"]
        
        # Create a workflow
        tasks = [
            TaskDefinition(
                task_id="health_check",
                task_name="System Health Check",
                task_type="system",
                parameters={"action": "health_check", "service": "operations"}
            ),
            TaskDefinition(
                task_id="data_validation",
                task_name="Data Validation",
                task_type="data",
                parameters={"action": "validate_data", "data_source": "operations_data"},
                dependencies=["health_check"]
            ),
            TaskDefinition(
                task_id="notification",
                task_name="Success Notification",
                task_type="notification",
                parameters={"action": "send_email", "recipient": "ops@test.com"},
                dependencies=["data_validation"]
            )
        ]
        
        workflow = WorkflowDefinition(
            workflow_id="ops_workflow",
            workflow_name="Operations Workflow",
            description="End-to-end operations workflow",
            tasks=tasks,
            trigger=WorkflowTrigger.MANUAL
        )
        
        # Create workflow
        assert workflow_manager.create_workflow(workflow) is True
        
        # Start monitoring and alerting
        await system_monitor.start_monitoring()
        await alert_manager.start_processing()
        
        # Execute workflow
        execution = await workflow_manager.execute_workflow(workflow.workflow_id)
        
        # Verify execution
        assert execution.status == WorkflowStatus.COMPLETED
        assert execution.completed_tasks == 3
        assert execution.failed_tasks == 0
        
        # Stop monitoring
        await system_monitor.stop_monitoring()
        await alert_manager.stop_processing()
    
    @pytest.mark.asyncio
    async def test_alert_workflow_integration(self, operations_system):
        """Test integration between alerting and workflow execution"""
        workflow_manager = operations_system["workflow_manager"]
        alert_manager = operations_system["alert_manager"]
        event_bus = operations_system["event_bus"]
        
        # Create alert response workflow
        alert_response_tasks = [
            TaskDefinition(
                task_id="investigate",
                task_name="Investigate Alert",
                task_type="system",
                parameters={"action": "health_check", "service": "investigation"}
            ),
            TaskDefinition(
                task_id="remediate",
                task_name="Remediate Issue",
                task_type="system",
                parameters={"action": "restart_service", "service": "problematic_service"},
                dependencies=["investigate"]
            ),
            TaskDefinition(
                task_id="notify_resolved",
                task_name="Notify Resolution",
                task_type="notification",
                parameters={"action": "send_slack", "channel": "#ops", "message": "Issue resolved"},
                dependencies=["remediate"]
            )
        ]
        
        alert_response_workflow = WorkflowDefinition(
            workflow_id="alert_response",
            workflow_name="Alert Response Workflow",
            description="Automated alert response",
            tasks=alert_response_tasks,
            trigger=WorkflowTrigger.EVENT,
            trigger_event="high_severity_alert"
        )
        
        # Create workflow
        assert workflow_manager.create_workflow(alert_response_workflow) is True
        
        # Start alert processing
        await alert_manager.start_processing()
        
        # Create high severity alert
        alert_event = Event(
            type=EventType.ALERT,
            payload={
                "severity": "critical",
                "rule_name": "High CPU Usage",
                "message": "CPU usage exceeded 95%",
                "source": "system_monitor",
                "priority": "high"
            }
        )
        
        # Publish alert event
        await event_bus.publish(alert_event)
        
        # Wait for alert processing
        await asyncio.sleep(0.5)
        
        # Verify alert was processed
        assert alert_manager.total_alerts > 0
        
        # Stop alert processing
        await alert_manager.stop_processing()
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_workflow_integration(self, operations_system):
        """Test integration between circuit breaker and workflow execution"""
        workflow_manager = operations_system["workflow_manager"]
        operational_controls = operations_system["operational_controls"]
        
        # Create workflow with circuit breaker protection
        protected_tasks = [
            TaskDefinition(
                task_id="protected_task",
                task_name="Protected Task",
                task_type="system",
                parameters={"action": "health_check", "service": "external_api"}
            )
        ]
        
        protected_workflow = WorkflowDefinition(
            workflow_id="protected_workflow",
            workflow_name="Protected Workflow",
            description="Workflow with circuit breaker protection",
            tasks=protected_tasks,
            trigger=WorkflowTrigger.MANUAL
        )
        
        # Create workflow
        assert workflow_manager.create_workflow(protected_workflow) is True
        
        # Add circuit breaker
        operational_controls.add_circuit_breaker("external_api", failure_threshold=2, recovery_timeout=30)
        
        # Mock task executor to use circuit breaker
        original_execute = workflow_manager.task_executors["system"].execute
        
        async def protected_execute(task, context):
            async with operational_controls.circuit_breaker("external_api"):
                return await original_execute(task, context)
        
        workflow_manager.task_executors["system"].execute = protected_execute
        
        # Execute workflow (should succeed)
        execution = await workflow_manager.execute_workflow(protected_workflow.workflow_id)
        assert execution.status == WorkflowStatus.COMPLETED
        
        # Verify circuit breaker state
        breaker = operational_controls.circuit_breakers["external_api"]
        assert breaker.state == "CLOSED"
        assert breaker.successful_calls == 1
    
    @pytest.mark.asyncio
    async def test_rate_limiting_workflow_integration(self, operations_system):
        """Test integration between rate limiting and workflow execution"""
        workflow_manager = operations_system["workflow_manager"]
        operational_controls = operations_system["operational_controls"]
        
        # Add strict rate limiter
        operational_controls.add_rate_limiter("workflow_execution", max_requests=2, time_window=60)
        
        # Create simple workflow
        simple_tasks = [
            TaskDefinition(
                task_id="simple_task",
                task_name="Simple Task",
                task_type="system",
                parameters={"action": "health_check"}
            )
        ]
        
        simple_workflow = WorkflowDefinition(
            workflow_id="simple_workflow",
            workflow_name="Simple Workflow",
            description="Simple workflow for rate limiting test",
            tasks=simple_tasks,
            trigger=WorkflowTrigger.MANUAL
        )
        
        # Create workflow
        assert workflow_manager.create_workflow(simple_workflow) is True
        
        # Execute workflow multiple times
        successful_executions = 0
        rate_limited_executions = 0
        
        for i in range(5):
            # Check rate limit
            if await operational_controls.check_rate_limit("workflow_execution"):
                execution = await workflow_manager.execute_workflow(simple_workflow.workflow_id)
                if execution.status == WorkflowStatus.COMPLETED:
                    successful_executions += 1
            else:
                rate_limited_executions += 1
        
        # Verify rate limiting worked
        assert successful_executions == 2  # Only 2 allowed
        assert rate_limited_executions == 3  # 3 were rate limited
    
    @pytest.mark.asyncio
    async def test_monitoring_alert_escalation_workflow(self, operations_system):
        """Test complete monitoring -> alert -> escalation -> workflow flow"""
        system_monitor = operations_system["system_monitor"]
        alert_manager = operations_system["alert_manager"]
        workflow_manager = operations_system["workflow_manager"]
        
        # Create escalation workflow
        escalation_tasks = [
            TaskDefinition(
                task_id="escalate_to_oncall",
                task_name="Escalate to On-Call",
                task_type="notification",
                parameters={"action": "send_sms", "phone": "+1234567890", "message": "Critical alert escalated"}
            ),
            TaskDefinition(
                task_id="create_incident",
                task_name="Create Incident",
                task_type="system",
                parameters={"action": "health_check", "service": "incident_management"}
            )
        ]
        
        escalation_workflow = WorkflowDefinition(
            workflow_id="escalation_workflow",
            workflow_name="Escalation Workflow",
            description="Critical alert escalation workflow",
            tasks=escalation_tasks,
            trigger=WorkflowTrigger.EVENT,
            trigger_event="critical_alert_escalation"
        )
        
        # Create workflow
        assert workflow_manager.create_workflow(escalation_workflow) is True
        
        # Start monitoring and alerting
        await system_monitor.start_monitoring()
        await alert_manager.start_processing()
        
        # Create critical alert rule
        critical_rule = AlertRule(
            rule_id="critical_cpu",
            name="Critical CPU Usage",
            description="CPU usage exceeds 98%",
            metric_name="cpu_usage",
            condition="cpu_usage > 98",
            threshold=98.0,
            severity=AlertSeverity.CRITICAL
        )
        
        system_monitor.add_alert_rule(critical_rule)
        
        # Simulate critical CPU usage
        critical_metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=99.0,  # Critical level
            memory_usage=50.0,
            memory_total=16.0,
            memory_available=8.0,
            disk_usage=60.0,
            disk_total=500.0,
            disk_available=200.0,
            network_io_sent=100.0,
            network_io_recv=200.0,
            load_average=[1.0, 1.5, 2.0],
            process_count=150,
            open_files=100,
            connections=50
        )
        
        # Trigger alert
        await system_monitor._check_alert_rules(critical_metrics)
        
        # Wait for alert processing
        await asyncio.sleep(0.5)
        
        # Verify alert was created
        assert alert_manager.total_alerts > 0
        
        # Stop monitoring
        await system_monitor.stop_monitoring()
        await alert_manager.stop_processing()
    
    def test_operations_system_status_integration(self, operations_system):
        """Test integrated system status reporting"""
        workflow_manager = operations_system["workflow_manager"]
        system_monitor = operations_system["system_monitor"]
        alert_manager = operations_system["alert_manager"]
        operational_controls = operations_system["operational_controls"]
        
        # Set some state across components
        workflow_manager.total_executions = 25
        workflow_manager.successful_executions = 22
        workflow_manager.failed_executions = 3
        
        system_monitor.alerts_triggered = 8
        system_monitor.health_checks_run = 500
        system_monitor.health_checks_failed = 12
        
        alert_manager.total_alerts = 15
        alert_manager.alerts_processed = 14
        alert_manager.notifications_sent = 42
        
        operational_controls.total_control_actions = 5
        operational_controls.circuit_breaker_trips = 2
        operational_controls.rate_limit_violations = 10
        
        # Get status from all components
        workflow_status = workflow_manager.get_manager_status()
        monitor_status = system_monitor.get_monitor_status()
        alert_status = alert_manager.get_manager_status()
        control_status = operational_controls.get_system_status()
        
        # Verify comprehensive status
        assert workflow_status["success_rate"] == 22/25
        assert monitor_status["alerts_triggered"] == 8
        assert alert_status["total_alerts"] == 15
        assert control_status["statistics"]["total_control_actions"] == 5
        
        # Create integrated status report
        integrated_status = {
            "timestamp": datetime.now().isoformat(),
            "workflow_manager": workflow_status,
            "system_monitor": monitor_status,
            "alert_manager": alert_status,
            "operational_controls": control_status
        }
        
        # Verify integrated status structure
        assert "workflow_manager" in integrated_status
        assert "system_monitor" in integrated_status
        assert "alert_manager" in integrated_status
        assert "operational_controls" in integrated_status
        assert "timestamp" in integrated_status
    
    @pytest.mark.asyncio
    async def test_operations_system_performance_under_load(self, operations_system):
        """Test operations system performance under load"""
        workflow_manager = operations_system["workflow_manager"]
        system_monitor = operations_system["system_monitor"]
        alert_manager = operations_system["alert_manager"]
        operational_controls = operations_system["operational_controls"]
        
        # Create lightweight workflow
        load_test_tasks = [
            TaskDefinition(
                task_id="load_task",
                task_name="Load Test Task",
                task_type="system",
                parameters={"action": "health_check"}
            )
        ]
        
        load_test_workflow = WorkflowDefinition(
            workflow_id="load_test_workflow",
            workflow_name="Load Test Workflow",
            description="Workflow for load testing",
            tasks=load_test_tasks,
            trigger=WorkflowTrigger.MANUAL,
            max_concurrent_executions=10
        )
        
        # Create workflow
        assert workflow_manager.create_workflow(load_test_workflow) is True
        
        # Start all components
        await system_monitor.start_monitoring()
        await alert_manager.start_processing()
        await operational_controls.start_monitoring()
        
        # Execute multiple workflows concurrently
        start_time = time.time()
        
        tasks = []
        for i in range(20):
            if await operational_controls.check_rate_limit("load_test"):
                task = asyncio.create_task(
                    workflow_manager.execute_workflow(load_test_workflow.workflow_id)
                )
                tasks.append(task)
        
        # Wait for all executions to complete
        executions = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Verify performance
        successful_executions = len([e for e in executions if isinstance(e, WorkflowExecution) and e.status == WorkflowStatus.COMPLETED])
        
        assert successful_executions > 0
        assert execution_time < 10  # Should complete within 10 seconds
        
        # Verify system remained stable
        workflow_status = workflow_manager.get_manager_status()
        assert workflow_status["success_rate"] > 0.5  # At least 50% success rate
        
        # Stop all components
        await system_monitor.stop_monitoring()
        await alert_manager.stop_processing()
        await operational_controls.stop_monitoring()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])