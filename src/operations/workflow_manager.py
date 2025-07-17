"""
Workflow Manager for Operations System

This module provides workflow management capabilities including
workflow definition, execution, monitoring, and control.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import structlog
import uuid
import json
from abc import ABC, abstractmethod

from ..core.event_bus import EventBus, Event, EventType

logger = structlog.get_logger()


class WorkflowStatus(Enum):
    """Status of workflow execution"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TaskStatus(Enum):
    """Status of individual tasks"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class WorkflowTrigger(Enum):
    """Workflow trigger types"""
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    EVENT = "event"
    CONDITION = "condition"


@dataclass
class TaskDefinition:
    """Definition of a workflow task"""
    task_id: str
    task_name: str
    task_type: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    timeout: int = 300  # seconds
    retry_count: int = 0
    retry_delay: int = 60  # seconds
    enabled: bool = True
    required: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "task_type": self.task_type,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "retry_delay": self.retry_delay,
            "enabled": self.enabled,
            "required": self.required
        }


@dataclass
class WorkflowDefinition:
    """Definition of a workflow"""
    workflow_id: str
    workflow_name: str
    description: str
    tasks: List[TaskDefinition]
    trigger: WorkflowTrigger
    schedule: Optional[str] = None  # Cron expression
    trigger_condition: Optional[str] = None
    trigger_event: Optional[str] = None
    max_execution_time: int = 3600  # seconds
    max_concurrent_executions: int = 1
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "description": self.description,
            "tasks": [task.to_dict() for task in self.tasks],
            "trigger": self.trigger.value,
            "schedule": self.schedule,
            "trigger_condition": self.trigger_condition,
            "trigger_event": self.trigger_event,
            "max_execution_time": self.max_execution_time,
            "max_concurrent_executions": self.max_concurrent_executions,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class TaskExecution:
    """Execution state of a task"""
    task_id: str
    execution_id: str
    status: TaskStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    logs: List[str] = field(default_factory=list)
    
    @property
    def execution_time(self) -> float:
        """Calculate execution time"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


@dataclass
class WorkflowExecution:
    """Execution state of a workflow"""
    execution_id: str
    workflow_id: str
    workflow_name: str
    status: WorkflowStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    tasks: Dict[str, TaskExecution] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    trigger_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    @property
    def execution_time(self) -> float:
        """Calculate execution time"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def completed_tasks(self) -> int:
        """Count completed tasks"""
        return len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED])
    
    @property
    def failed_tasks(self) -> int:
        """Count failed tasks"""
        return len([t for t in self.tasks.values() if t.status == TaskStatus.FAILED])
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if not self.tasks:
            return 0.0
        return self.completed_tasks / len(self.tasks)


class TaskExecutor(ABC):
    """Abstract base class for task executors"""
    
    @abstractmethod
    async def execute(self, task: TaskDefinition, context: Dict[str, Any]) -> Any:
        """Execute a task"""
        pass
    
    @abstractmethod
    def get_task_type(self) -> str:
        """Get the task type this executor handles"""
        pass


class SystemTaskExecutor(TaskExecutor):
    """Executor for system tasks"""
    
    def get_task_type(self) -> str:
        return "system"
    
    async def execute(self, task: TaskDefinition, context: Dict[str, Any]) -> Any:
        """Execute system task"""
        action = task.parameters.get("action")
        
        if action == "health_check":
            return await self._health_check(task.parameters)
        elif action == "restart_service":
            return await self._restart_service(task.parameters)
        elif action == "backup_data":
            return await self._backup_data(task.parameters)
        elif action == "cleanup":
            return await self._cleanup(task.parameters)
        else:
            raise ValueError(f"Unknown system action: {action}")
    
    async def _health_check(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform health check"""
        service = parameters.get("service", "system")
        
        # Mock health check
        await asyncio.sleep(0.1)
        
        return {
            "service": service,
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {
                "cpu_usage": 25.5,
                "memory_usage": 60.2,
                "disk_usage": 45.8,
                "network_status": "ok"
            }
        }
    
    async def _restart_service(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Restart service"""
        service = parameters.get("service", "unknown")
        
        # Mock service restart
        await asyncio.sleep(2.0)
        
        return {
            "service": service,
            "action": "restart",
            "status": "completed",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _backup_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Backup data"""
        backup_type = parameters.get("backup_type", "full")
        
        # Mock backup
        await asyncio.sleep(1.0)
        
        return {
            "backup_type": backup_type,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "backup_size": "1.2GB",
            "backup_location": "/backups/backup_20231201.tar.gz"
        }
    
    async def _cleanup(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Cleanup operation"""
        cleanup_type = parameters.get("cleanup_type", "logs")
        
        # Mock cleanup
        await asyncio.sleep(0.5)
        
        return {
            "cleanup_type": cleanup_type,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "cleaned_items": 150,
            "space_freed": "500MB"
        }


class DataTaskExecutor(TaskExecutor):
    """Executor for data tasks"""
    
    def get_task_type(self) -> str:
        return "data"
    
    async def execute(self, task: TaskDefinition, context: Dict[str, Any]) -> Any:
        """Execute data task"""
        action = task.parameters.get("action")
        
        if action == "validate_data":
            return await self._validate_data(task.parameters)
        elif action == "process_data":
            return await self._process_data(task.parameters)
        elif action == "export_data":
            return await self._export_data(task.parameters)
        else:
            raise ValueError(f"Unknown data action: {action}")
    
    async def _validate_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data"""
        data_source = parameters.get("data_source", "unknown")
        
        # Mock validation
        await asyncio.sleep(0.3)
        
        return {
            "data_source": data_source,
            "status": "valid",
            "timestamp": datetime.now().isoformat(),
            "records_validated": 10000,
            "errors_found": 5,
            "warnings_found": 12
        }
    
    async def _process_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Process data"""
        process_type = parameters.get("process_type", "transform")
        
        # Mock processing
        await asyncio.sleep(1.5)
        
        return {
            "process_type": process_type,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "records_processed": 10000,
            "output_records": 9995
        }
    
    async def _export_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Export data"""
        export_format = parameters.get("export_format", "csv")
        
        # Mock export
        await asyncio.sleep(0.8)
        
        return {
            "export_format": export_format,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "records_exported": 9995,
            "file_size": "25MB",
            "export_location": f"/exports/data_export.{export_format}"
        }


class NotificationTaskExecutor(TaskExecutor):
    """Executor for notification tasks"""
    
    def get_task_type(self) -> str:
        return "notification"
    
    async def execute(self, task: TaskDefinition, context: Dict[str, Any]) -> Any:
        """Execute notification task"""
        action = task.parameters.get("action")
        
        if action == "send_email":
            return await self._send_email(task.parameters)
        elif action == "send_slack":
            return await self._send_slack(task.parameters)
        elif action == "send_sms":
            return await self._send_sms(task.parameters)
        else:
            raise ValueError(f"Unknown notification action: {action}")
    
    async def _send_email(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Send email notification"""
        recipient = parameters.get("recipient", "admin@company.com")
        subject = parameters.get("subject", "Workflow Notification")
        
        # Mock email sending
        await asyncio.sleep(0.2)
        
        return {
            "recipient": recipient,
            "subject": subject,
            "status": "sent",
            "timestamp": datetime.now().isoformat(),
            "message_id": f"msg_{uuid.uuid4().hex[:8]}"
        }
    
    async def _send_slack(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Send Slack notification"""
        channel = parameters.get("channel", "#operations")
        message = parameters.get("message", "Workflow notification")
        
        # Mock Slack sending
        await asyncio.sleep(0.1)
        
        return {
            "channel": channel,
            "message": message,
            "status": "sent",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _send_sms(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Send SMS notification"""
        phone = parameters.get("phone", "+1234567890")
        message = parameters.get("message", "Workflow notification")
        
        # Mock SMS sending
        await asyncio.sleep(0.1)
        
        return {
            "phone": phone,
            "message": message,
            "status": "sent",
            "timestamp": datetime.now().isoformat()
        }


class WorkflowManager:
    """Main workflow management system"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.running_executions: Dict[str, WorkflowExecution] = {}
        self.task_executors: Dict[str, TaskExecutor] = {}
        
        # Initialize default executors
        self._register_default_executors()
        
        # Statistics
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        
        logger.info("Workflow Manager initialized")
    
    def _register_default_executors(self):
        """Register default task executors"""
        executors = [
            SystemTaskExecutor(),
            DataTaskExecutor(),
            NotificationTaskExecutor()
        ]
        
        for executor in executors:
            self.task_executors[executor.get_task_type()] = executor
    
    def register_task_executor(self, executor: TaskExecutor):
        """Register a task executor"""
        task_type = executor.get_task_type()
        self.task_executors[task_type] = executor
        logger.info("Task executor registered", task_type=task_type)
    
    def create_workflow(self, workflow_def: WorkflowDefinition) -> bool:
        """Create a new workflow"""
        try:
            # Validate workflow
            self._validate_workflow(workflow_def)
            
            # Store workflow
            self.workflows[workflow_def.workflow_id] = workflow_def
            
            logger.info("Workflow created", workflow_id=workflow_def.workflow_id)
            return True
            
        except Exception as e:
            logger.error("Failed to create workflow", workflow_id=workflow_def.workflow_id, error=str(e))
            return False
    
    def _validate_workflow(self, workflow_def: WorkflowDefinition):
        """Validate workflow definition"""
        # Check for duplicate task IDs
        task_ids = [task.task_id for task in workflow_def.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Duplicate task IDs found")
        
        # Check task dependencies
        for task in workflow_def.tasks:
            for dep in task.dependencies:
                if dep not in task_ids:
                    raise ValueError(f"Task {task.task_id} depends on non-existent task {dep}")
        
        # Check for circular dependencies
        self._check_circular_dependencies(workflow_def.tasks)
        
        # Check task executors exist
        for task in workflow_def.tasks:
            if task.task_type not in self.task_executors:
                raise ValueError(f"No executor found for task type: {task.task_type}")
    
    def _check_circular_dependencies(self, tasks: List[TaskDefinition]):
        """Check for circular dependencies"""
        task_deps = {task.task_id: task.dependencies for task in tasks}
        
        def has_cycle(task_id: str, visited: set, rec_stack: set) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)
            
            for dep in task_deps.get(task_id, []):
                if dep not in visited:
                    if has_cycle(dep, visited, rec_stack):
                        return True
                elif dep in rec_stack:
                    return True
            
            rec_stack.remove(task_id)
            return False
        
        visited = set()
        rec_stack = set()
        
        for task in tasks:
            if task.task_id not in visited:
                if has_cycle(task.task_id, visited, rec_stack):
                    raise ValueError("Circular dependency detected")
    
    async def execute_workflow(
        self,
        workflow_id: str,
        trigger_data: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> WorkflowExecution:
        """Execute a workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        workflow_def = self.workflows[workflow_id]
        
        if not workflow_def.enabled:
            raise ValueError(f"Workflow is disabled: {workflow_id}")
        
        # Check concurrent execution limit
        running_count = len([
            e for e in self.running_executions.values()
            if e.workflow_id == workflow_id
        ])
        
        if running_count >= workflow_def.max_concurrent_executions:
            raise ValueError(f"Maximum concurrent executions reached for workflow: {workflow_id}")
        
        # Create execution
        execution = WorkflowExecution(
            execution_id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            workflow_name=workflow_def.workflow_name,
            status=WorkflowStatus.RUNNING,
            start_time=datetime.now(),
            trigger_data=trigger_data,
            context=context or {}
        )
        
        # Initialize task executions
        for task in workflow_def.tasks:
            execution.tasks[task.task_id] = TaskExecution(
                task_id=task.task_id,
                execution_id=execution.execution_id,
                status=TaskStatus.PENDING
            )
        
        # Store execution
        self.executions[execution.execution_id] = execution
        self.running_executions[execution.execution_id] = execution
        self.total_executions += 1
        
        # Execute workflow
        try:
            await self._execute_workflow_tasks(workflow_def, execution)
            
            execution.status = WorkflowStatus.COMPLETED
            execution.end_time = datetime.now()
            self.successful_executions += 1
            
            logger.info(
                "Workflow execution completed",
                execution_id=execution.execution_id,
                workflow_id=workflow_id,
                execution_time=execution.execution_time,
                success_rate=execution.success_rate
            )
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.end_time = datetime.now()
            execution.error = str(e)
            self.failed_executions += 1
            
            logger.error(
                "Workflow execution failed",
                execution_id=execution.execution_id,
                workflow_id=workflow_id,
                error=str(e)
            )
        
        finally:
            # Remove from running executions
            if execution.execution_id in self.running_executions:
                del self.running_executions[execution.execution_id]
        
        return execution
    
    async def _execute_workflow_tasks(self, workflow_def: WorkflowDefinition, execution: WorkflowExecution):
        """Execute workflow tasks"""
        # Build dependency graph
        task_deps = {task.task_id: task.dependencies for task in workflow_def.tasks}
        task_map = {task.task_id: task for task in workflow_def.tasks}
        
        # Execute tasks in dependency order
        completed_tasks = set()
        
        while len(completed_tasks) < len(workflow_def.tasks):
            # Find tasks ready to execute
            ready_tasks = []
            for task in workflow_def.tasks:
                if (task.task_id not in completed_tasks and
                    task.enabled and
                    all(dep in completed_tasks for dep in task.dependencies)):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # Check for failed required tasks
                failed_required = [
                    task_id for task_id, task_exec in execution.tasks.items()
                    if task_exec.status == TaskStatus.FAILED and task_map[task_id].required
                ]
                
                if failed_required:
                    raise Exception(f"Required tasks failed: {failed_required}")
                
                # Skip non-required failed tasks
                failed_tasks = [
                    task_id for task_id, task_exec in execution.tasks.items()
                    if task_exec.status == TaskStatus.FAILED
                ]
                completed_tasks.update(failed_tasks)
                continue
            
            # Execute ready tasks in parallel
            tasks_to_run = []
            for task in ready_tasks:
                if task.task_id not in completed_tasks:
                    tasks_to_run.append(self._execute_task(task, execution))
            
            if tasks_to_run:
                await asyncio.gather(*tasks_to_run, return_exceptions=True)
            
            # Update completed tasks
            for task in ready_tasks:
                if execution.tasks[task.task_id].status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED]:
                    completed_tasks.add(task.task_id)
    
    async def _execute_task(self, task: TaskDefinition, execution: WorkflowExecution):
        """Execute a single task"""
        task_exec = execution.tasks[task.task_id]
        task_exec.status = TaskStatus.RUNNING
        task_exec.start_time = datetime.now()
        
        try:
            # Get executor
            executor = self.task_executors.get(task.task_type)
            if not executor:
                raise ValueError(f"No executor found for task type: {task.task_type}")
            
            # Execute with timeout
            result = await asyncio.wait_for(
                executor.execute(task, execution.context),
                timeout=task.timeout
            )
            
            task_exec.result = result
            task_exec.status = TaskStatus.COMPLETED
            task_exec.end_time = datetime.now()
            
            logger.info(
                "Task completed",
                task_id=task.task_id,
                execution_id=execution.execution_id,
                execution_time=task_exec.execution_time
            )
            
        except asyncio.TimeoutError:
            task_exec.status = TaskStatus.FAILED
            task_exec.error = "Task timeout"
            task_exec.end_time = datetime.now()
            
            logger.error("Task timeout", task_id=task.task_id, execution_id=execution.execution_id)
            
        except Exception as e:
            task_exec.status = TaskStatus.FAILED
            task_exec.error = str(e)
            task_exec.end_time = datetime.now()
            
            logger.error("Task failed", task_id=task.task_id, execution_id=execution.execution_id, error=str(e))
            
            # Retry if configured
            if task_exec.retry_count < task.retry_count:
                task_exec.retry_count += 1
                task_exec.status = TaskStatus.RETRYING
                
                logger.info(
                    "Retrying task",
                    task_id=task.task_id,
                    execution_id=execution.execution_id,
                    retry_count=task_exec.retry_count
                )
                
                await asyncio.sleep(task.retry_delay)
                await self._execute_task(task, execution)
    
    def get_workflow(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """Get workflow definition"""
        return self.workflows.get(workflow_id)
    
    def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution"""
        return self.executions.get(execution_id)
    
    def get_executions(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[WorkflowStatus] = None,
        limit: Optional[int] = None
    ) -> List[WorkflowExecution]:
        """Get workflow executions"""
        executions = list(self.executions.values())
        
        if workflow_id:
            executions = [e for e in executions if e.workflow_id == workflow_id]
        
        if status:
            executions = [e for e in executions if e.status == status]
        
        # Sort by start time (newest first)
        executions.sort(key=lambda x: x.start_time, reverse=True)
        
        if limit:
            executions = executions[:limit]
        
        return executions
    
    def get_running_executions(self) -> List[WorkflowExecution]:
        """Get currently running executions"""
        return list(self.running_executions.values())
    
    def get_manager_status(self) -> Dict[str, Any]:
        """Get workflow manager status"""
        return {
            "total_workflows": len(self.workflows),
            "active_workflows": len([w for w in self.workflows.values() if w.enabled]),
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": self.successful_executions / self.total_executions if self.total_executions > 0 else 0.0,
            "running_executions": len(self.running_executions),
            "registered_executors": list(self.task_executors.keys())
        }
    
    def cleanup_old_executions(self, days_old: int = 30):
        """Clean up old executions"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        executions_to_remove = []
        for execution_id, execution in self.executions.items():
            if execution.start_time < cutoff_date:
                executions_to_remove.append(execution_id)
        
        for execution_id in executions_to_remove:
            del self.executions[execution_id]
        
        logger.info("Cleaned up old executions", count=len(executions_to_remove))
        return len(executions_to_remove)