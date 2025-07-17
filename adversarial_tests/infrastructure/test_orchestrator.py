"""
Async Test Orchestrator

Provides async test execution framework with parallel testing capabilities
for multiple agents and resource management.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
from datetime import datetime
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import pytest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.event_bus import EventBus
from src.core.events import Event


class TestStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class TestPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TestTask:
    """Represents a test task with metadata and execution info"""
    test_id: str
    test_name: str
    test_function: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict = field(default_factory=dict)
    priority: TestPriority = TestPriority.MEDIUM
    timeout: float = 300.0  # 5 minutes default
    max_retries: int = 3
    resource_requirements: Dict = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    # Execution metadata
    status: TestStatus = TestStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0
    resource_usage: Dict = field(default_factory=dict)


@dataclass
class TestSession:
    """Represents a testing session with multiple test tasks"""
    session_id: str
    name: str
    tasks: List[TestTask] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    max_parallel_tests: int = 5
    total_timeout: float = 3600.0  # 1 hour default
    status: TestStatus = TestStatus.PENDING


class ResourceManager:
    """Manages system resources for test execution"""
    
    def __init__(self, max_cpu_percent: float = 80.0, max_memory_percent: float = 80.0):
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
        self.resource_locks = {}
        self.allocated_resources = {}
        self._lock = threading.Lock()
        
    def check_resource_availability(self, requirements: Dict) -> bool:
        """Check if resources are available for test execution"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        if cpu_percent > self.max_cpu_percent:
            return False
        if memory_percent > self.max_memory_percent:
            return False
            
        return True
    
    def allocate_resources(self, test_id: str, requirements: Dict) -> bool:
        """Allocate resources for a test"""
        with self._lock:
            if not self.check_resource_availability(requirements):
                return False
            
            self.allocated_resources[test_id] = {
                'requirements': requirements,
                'allocated_at': datetime.now(),
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent
            }
            return True
    
    def release_resources(self, test_id: str):
        """Release resources for a test"""
        with self._lock:
            if test_id in self.allocated_resources:
                del self.allocated_resources[test_id]
    
    def get_resource_usage(self) -> Dict:
        """Get current resource usage statistics"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'allocated_tests': len(self.allocated_resources),
            'available_cores': psutil.cpu_count()
        }


class TestOrchestrator:
    """
    Async test orchestration framework for parallel test execution
    with resource management and real-time monitoring.
    """
    
    def __init__(self, max_parallel_tests: int = 5, 
                 max_cpu_percent: float = 80.0,
                 max_memory_percent: float = 80.0):
        self.max_parallel_tests = max_parallel_tests
        self.resource_manager = ResourceManager(max_cpu_percent, max_memory_percent)
        self.event_bus = EventBus()
        self.sessions: Dict[str, TestSession] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_queue = asyncio.Queue()
        self.results_queue = asyncio.Queue()
        self.logger = logging.getLogger(__name__)
        self._shutdown_event = asyncio.Event()
        
        # Performance metrics
        self.metrics = {
            'tests_executed': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'average_execution_time': 0.0,
            'total_execution_time': 0.0,
            'parallel_efficiency': 0.0
        }
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('test_orchestrator.log'),
                logging.StreamHandler()
            ]
        )
    
    async def create_session(self, name: str, max_parallel_tests: int = None) -> str:
        """Create a new test session"""
        session_id = f"session_{int(time.time())}"
        session = TestSession(
            session_id=session_id,
            name=name,
            max_parallel_tests=max_parallel_tests or self.max_parallel_tests
        )
        self.sessions[session_id] = session
        
        # Emit session created event
        await self.event_bus.emit(Event(
            type="session_created",
            data={
                "session_id": session_id,
                "name": name,
                "timestamp": datetime.now().isoformat()
            }
        ))
        
        self.logger.info(f"Created test session: {session_id}")
        return session_id
    
    async def add_test_task(self, session_id: str, test_task: TestTask):
        """Add a test task to a session"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        session.tasks.append(test_task)
        
        # Emit task added event
        await self.event_bus.emit(Event(
            type="task_added",
            data={
                "session_id": session_id,
                "test_id": test_task.test_id,
                "test_name": test_task.test_name,
                "priority": test_task.priority.name,
                "timestamp": datetime.now().isoformat()
            }
        ))
        
        self.logger.info(f"Added test task {test_task.test_id} to session {session_id}")
    
    async def execute_session(self, session_id: str) -> Dict:
        """Execute all tests in a session"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        session.start_time = datetime.now()
        session.status = TestStatus.RUNNING
        
        self.logger.info(f"Starting execution of session {session_id}")
        
        # Sort tasks by priority and dependencies
        sorted_tasks = self._sort_tasks_by_priority_and_dependencies(session.tasks)
        
        # Execute tasks with parallelism
        results = await self._execute_tasks_parallel(session_id, sorted_tasks)
        
        session.end_time = datetime.now()
        session.status = TestStatus.COMPLETED
        
        # Calculate session metrics
        session_metrics = self._calculate_session_metrics(session)
        
        # Emit session completed event
        await self.event_bus.emit(Event(
            type="session_completed",
            data={
                "session_id": session_id,
                "results": results,
                "metrics": session_metrics,
                "timestamp": datetime.now().isoformat()
            }
        ))
        
        self.logger.info(f"Completed execution of session {session_id}")
        return {
            "session_id": session_id,
            "results": results,
            "metrics": session_metrics,
            "execution_time": (session.end_time - session.start_time).total_seconds()
        }
    
    def _sort_tasks_by_priority_and_dependencies(self, tasks: List[TestTask]) -> List[TestTask]:
        """Sort tasks by priority and resolve dependencies"""
        # Simple topological sort for dependencies
        sorted_tasks = []
        remaining_tasks = tasks.copy()
        
        while remaining_tasks:
            # Find tasks with no unresolved dependencies
            ready_tasks = []
            for task in remaining_tasks:
                if not task.dependencies or all(
                    dep in [t.test_id for t in sorted_tasks] 
                    for dep in task.dependencies
                ):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # Break circular dependencies by picking highest priority
                ready_tasks = [max(remaining_tasks, key=lambda t: t.priority.value)]
                self.logger.warning(f"Breaking circular dependency for task {ready_tasks[0].test_id}")
            
            # Sort ready tasks by priority
            ready_tasks.sort(key=lambda t: t.priority.value, reverse=True)
            
            # Add to sorted list and remove from remaining
            sorted_tasks.extend(ready_tasks)
            for task in ready_tasks:
                remaining_tasks.remove(task)
        
        return sorted_tasks
    
    async def _execute_tasks_parallel(self, session_id: str, tasks: List[TestTask]) -> List[Dict]:
        """Execute tasks in parallel with resource management"""
        semaphore = asyncio.Semaphore(self.max_parallel_tests)
        results = []
        
        async def execute_single_task(task: TestTask) -> Dict:
            async with semaphore:
                return await self._execute_task(session_id, task)
        
        # Create tasks for parallel execution
        task_coroutines = [execute_single_task(task) for task in tasks]
        
        # Execute with timeout
        try:
            results = await asyncio.gather(*task_coroutines, return_exceptions=True)
        except asyncio.TimeoutError:
            self.logger.error(f"Session {session_id} timed out")
            raise
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "test_id": tasks[i].test_id,
                    "status": TestStatus.FAILED.value,
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_task(self, session_id: str, task: TestTask) -> Dict:
        """Execute a single test task"""
        task.start_time = datetime.now()
        task.status = TestStatus.RUNNING
        
        # Emit task started event
        await self.event_bus.emit(Event(
            type="task_started",
            data={
                "session_id": session_id,
                "test_id": task.test_id,
                "test_name": task.test_name,
                "timestamp": task.start_time.isoformat()
            }
        ))
        
        # Check and allocate resources
        if not self.resource_manager.allocate_resources(task.test_id, task.resource_requirements):
            task.status = TestStatus.FAILED
            task.error = "Insufficient resources"
            self.logger.error(f"Task {task.test_id} failed due to insufficient resources")
            return self._create_task_result(task)
        
        try:
            # Execute task with timeout
            result = await asyncio.wait_for(
                self._run_test_function(task),
                timeout=task.timeout
            )
            
            task.result = result
            task.status = TestStatus.COMPLETED
            self.logger.info(f"Task {task.test_id} completed successfully")
            
        except asyncio.TimeoutError:
            task.status = TestStatus.TIMEOUT
            task.error = f"Task timed out after {task.timeout} seconds"
            self.logger.error(f"Task {task.test_id} timed out")
            
        except Exception as e:
            task.status = TestStatus.FAILED
            task.error = str(e)
            self.logger.error(f"Task {task.test_id} failed: {e}")
            
        finally:
            task.end_time = datetime.now()
            task.execution_time = (task.end_time - task.start_time).total_seconds()
            
            # Record resource usage
            task.resource_usage = self.resource_manager.get_resource_usage()
            
            # Release resources
            self.resource_manager.release_resources(task.test_id)
            
            # Emit task completed event
            await self.event_bus.emit(Event(
                type="task_completed",
                data={
                    "session_id": session_id,
                    "test_id": task.test_id,
                    "status": task.status.value,
                    "execution_time": task.execution_time,
                    "timestamp": task.end_time.isoformat()
                }
            ))
        
        return self._create_task_result(task)
    
    async def _run_test_function(self, task: TestTask) -> Any:
        """Run the actual test function"""
        if asyncio.iscoroutinefunction(task.test_function):
            return await task.test_function(*task.args, **task.kwargs)
        else:
            # Run synchronous function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                lambda: task.test_function(*task.args, **task.kwargs)
            )
    
    def _create_task_result(self, task: TestTask) -> Dict:
        """Create task result dictionary"""
        return {
            "test_id": task.test_id,
            "test_name": task.test_name,
            "status": task.status.value,
            "start_time": task.start_time.isoformat() if task.start_time else None,
            "end_time": task.end_time.isoformat() if task.end_time else None,
            "execution_time": task.execution_time,
            "result": task.result,
            "error": task.error,
            "retry_count": task.retry_count,
            "resource_usage": task.resource_usage
        }
    
    def _calculate_session_metrics(self, session: TestSession) -> Dict:
        """Calculate metrics for a test session"""
        total_tests = len(session.tasks)
        passed_tests = sum(1 for task in session.tasks if task.status == TestStatus.COMPLETED)
        failed_tests = sum(1 for task in session.tasks if task.status == TestStatus.FAILED)
        
        total_execution_time = sum(task.execution_time for task in session.tasks)
        average_execution_time = total_execution_time / total_tests if total_tests > 0 else 0
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            "total_execution_time": total_execution_time,
            "average_execution_time": average_execution_time,
            "session_duration": (session.end_time - session.start_time).total_seconds() if session.end_time and session.start_time else 0
        }
    
    async def get_session_status(self, session_id: str) -> Dict:
        """Get current status of a session"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        return {
            "session_id": session_id,
            "name": session.name,
            "status": session.status.value,
            "start_time": session.start_time.isoformat() if session.start_time else None,
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "total_tasks": len(session.tasks),
            "completed_tasks": sum(1 for task in session.tasks if task.status == TestStatus.COMPLETED),
            "failed_tasks": sum(1 for task in session.tasks if task.status == TestStatus.FAILED),
            "running_tasks": sum(1 for task in session.tasks if task.status == TestStatus.RUNNING),
            "resource_usage": self.resource_manager.get_resource_usage()
        }
    
    async def cancel_session(self, session_id: str):
        """Cancel a running session"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        session.status = TestStatus.CANCELLED
        
        # Cancel running tasks
        for task in session.tasks:
            if task.status == TestStatus.RUNNING:
                task.status = TestStatus.CANCELLED
                self.resource_manager.release_resources(task.test_id)
        
        self.logger.info(f"Cancelled session {session_id}")
    
    async def get_system_metrics(self) -> Dict:
        """Get system-wide metrics"""
        return {
            "resource_usage": self.resource_manager.get_resource_usage(),
            "active_sessions": len([s for s in self.sessions.values() if s.status == TestStatus.RUNNING]),
            "total_sessions": len(self.sessions),
            "orchestrator_metrics": self.metrics
        }


# Example usage and testing functions
async def example_test_function(duration: float = 1.0, should_fail: bool = False):
    """Example test function for demonstration"""
    await asyncio.sleep(duration)
    if should_fail:
        raise ValueError("Test failure simulation")
    return {"success": True, "duration": duration}


def sync_test_function(value: int) -> Dict:
    """Example synchronous test function"""
    time.sleep(0.1)  # Simulate work
    return {"result": value * 2}


async def demo_orchestrator():
    """Demonstration of the test orchestrator"""
    orchestrator = TestOrchestrator(max_parallel_tests=3)
    
    # Create session
    session_id = await orchestrator.create_session("Demo Session")
    
    # Add test tasks
    tasks = [
        TestTask(
            test_id="test_1",
            test_name="Quick Test",
            test_function=example_test_function,
            args=(0.5, False),
            priority=TestPriority.HIGH
        ),
        TestTask(
            test_id="test_2", 
            test_name="Slow Test",
            test_function=example_test_function,
            args=(2.0, False),
            priority=TestPriority.MEDIUM
        ),
        TestTask(
            test_id="test_3",
            test_name="Failing Test",
            test_function=example_test_function,
            args=(1.0, True),
            priority=TestPriority.LOW
        ),
        TestTask(
            test_id="test_4",
            test_name="Sync Test",
            test_function=sync_test_function,
            args=(42,),
            priority=TestPriority.HIGH
        )
    ]
    
    for task in tasks:
        await orchestrator.add_test_task(session_id, task)
    
    # Execute session
    results = await orchestrator.execute_session(session_id)
    
    print("\n=== TEST ORCHESTRATOR DEMO RESULTS ===")
    print(json.dumps(results, indent=2))
    
    # Get system metrics
    metrics = await orchestrator.get_system_metrics()
    print("\n=== SYSTEM METRICS ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    asyncio.run(demo_orchestrator())