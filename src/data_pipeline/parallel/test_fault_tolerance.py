"""
Test suite for fault tolerance and health monitoring features
"""

import unittest
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import threading
import queue

from parallel_processor import (
    ParallelProcessor,
    ParallelConfig,
    DataChunk,
    WorkerState,
    TaskState,
    AlertLevel,
    WorkerHealthMetrics,
    TaskMetadata,
    Alert,
    WorkerHealthMonitor,
    TaskManager,
    AlertManager,
    CheckpointManager
)

class TestFaultTolerance(unittest.TestCase):
    """Test fault tolerance features"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ParallelConfig(
            max_workers=2,
            enable_fault_tolerance=True,
            enable_health_monitoring=True,
            enable_checkpointing=True,
            enable_alerting=True,
            checkpoint_directory=self.temp_dir,
            max_task_retries=2,
            retry_delay_seconds=0.1,
            timeout_seconds=5.0
        )
        self.processor = ParallelProcessor(self.config)
    
    def tearDown(self):
        """Clean up test environment"""
        self.processor.shutdown()
        shutil.rmtree(self.temp_dir)
    
    def test_worker_health_metrics_creation(self):
        """Test worker health metrics creation"""
        metrics = WorkerHealthMetrics(
            worker_id="test_worker",
            state=WorkerState.HEALTHY,
            last_heartbeat=time.time(),
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_mb=512.0,
            disk_usage_percent=70.0,
            task_queue_size=5,
            tasks_completed=10,
            tasks_failed=1,
            avg_task_duration=2.0,
            error_rate=0.1,
            uptime=3600.0
        )
        
        self.assertEqual(metrics.worker_id, "test_worker")
        self.assertEqual(metrics.state, WorkerState.HEALTHY)
        self.assertTrue(metrics.is_healthy())
        self.assertFalse(metrics.is_degraded())
    
    def test_worker_health_degradation_detection(self):
        """Test worker health degradation detection"""
        metrics = WorkerHealthMetrics(
            worker_id="test_worker",
            state=WorkerState.HEALTHY,
            last_heartbeat=time.time(),
            cpu_percent=85.0,  # High CPU
            memory_percent=80.0,  # High memory
            memory_mb=1024.0,
            disk_usage_percent=50.0,
            task_queue_size=5,
            tasks_completed=10,
            tasks_failed=2,
            avg_task_duration=12.0,  # High duration
            error_rate=0.08,  # High error rate
            uptime=3600.0
        )
        
        self.assertTrue(metrics.is_degraded())
        self.assertFalse(metrics.is_healthy())
    
    def test_task_metadata_retry_logic(self):
        """Test task metadata retry logic"""
        task = TaskMetadata(
            task_id="test_task",
            chunk_id="test_chunk",
            state=TaskState.FAILED,
            worker_id="test_worker",
            created_at=time.time(),
            started_at=None,
            completed_at=None,
            retry_count=1,
            max_retries=3,
            error_message="Test error",
            checkpoint_data=None
        )
        
        self.assertTrue(task.can_retry())
        
        # Test retry limit
        task.retry_count = 3
        self.assertFalse(task.can_retry())
    
    def test_alert_creation_and_management(self):
        """Test alert creation and management"""
        alert_manager = AlertManager(self.config)
        
        alert = Alert(
            id="test_alert",
            level=AlertLevel.WARNING,
            message="Test alert message",
            component="test_component",
            worker_id="test_worker",
            timestamp=time.time()
        )
        
        alert_manager.add_alert(alert)
        active_alerts = alert_manager.get_active_alerts()
        
        self.assertEqual(len(active_alerts), 1)
        self.assertEqual(active_alerts[0].id, "test_alert")
        
        # Test alert resolution
        alert_manager.resolve_alert("test_alert")
        self.assertTrue(alert.resolved)
    
    def test_checkpoint_creation_and_loading(self):
        """Test checkpoint creation and loading"""
        checkpoint_manager = CheckpointManager(self.config)
        
        # Create sample tasks
        tasks = {
            "task1": TaskMetadata(
                task_id="task1",
                chunk_id="chunk1",
                state=TaskState.RUNNING,
                worker_id="worker1",
                created_at=time.time(),
                started_at=time.time(),
                completed_at=None,
                retry_count=0,
                max_retries=3,
                error_message=None,
                checkpoint_data={"key": "value"}
            )
        }
        
        # Create checkpoint
        checkpoint_manager.create_checkpoint(tasks)
        
        # Verify checkpoint file exists
        checkpoint_files = list(Path(self.temp_dir).glob("checkpoint_*.json"))
        self.assertGreater(len(checkpoint_files), 0)
        
        # Load checkpoint
        loaded_tasks = {}
        checkpoint_manager.load_checkpoint(loaded_tasks)
        
        # Verify loaded tasks
        self.assertIn("task1", loaded_tasks)
        self.assertEqual(loaded_tasks["task1"].task_id, "task1")
        self.assertEqual(loaded_tasks["task1"].state, TaskState.PENDING)  # Should be reset
    
    def test_worker_health_monitor(self):
        """Test worker health monitor"""
        monitor = WorkerHealthMonitor(self.config)
        
        # Create test worker metrics
        worker_metrics = {
            "worker1": WorkerHealthMetrics(
                worker_id="worker1",
                state=WorkerState.HEALTHY,
                last_heartbeat=time.time(),
                cpu_percent=50.0,
                memory_percent=60.0,
                memory_mb=512.0,
                disk_usage_percent=70.0,
                task_queue_size=5,
                tasks_completed=10,
                tasks_failed=1,
                avg_task_duration=2.0,
                error_rate=0.05,
                uptime=3600.0
            ),
            "worker2": WorkerHealthMetrics(
                worker_id="worker2",
                state=WorkerState.UNHEALTHY,
                last_heartbeat=time.time(),
                cpu_percent=95.0,  # Critical CPU
                memory_percent=90.0,  # Critical memory
                memory_mb=1024.0,
                disk_usage_percent=70.0,
                task_queue_size=5,
                tasks_completed=10,
                tasks_failed=5,
                avg_task_duration=2.0,
                error_rate=0.5,  # Critical error rate
                uptime=3600.0
            )
        }
        
        # Test health check (should log warnings for unhealthy worker)
        with patch('logging.Logger.warning') as mock_warning:
            monitor.check_worker_health(worker_metrics)
            mock_warning.assert_called()
    
    def test_task_manager_reassignment(self):
        """Test task manager reassignment"""
        task_manager = TaskManager(self.config)
        
        # Create active tasks
        active_tasks = {
            "task1": TaskMetadata(
                task_id="task1",
                chunk_id="chunk1",
                state=TaskState.RUNNING,
                worker_id="failed_worker",
                created_at=time.time(),
                started_at=time.time(),
                completed_at=None,
                retry_count=0,
                max_retries=3,
                error_message=None,
                checkpoint_data=None
            ),
            "task2": TaskMetadata(
                task_id="task2",
                chunk_id="chunk2",
                state=TaskState.COMPLETED,
                worker_id="failed_worker",
                created_at=time.time(),
                started_at=time.time(),
                completed_at=time.time(),
                retry_count=0,
                max_retries=3,
                error_message=None,
                checkpoint_data=None
            )
        }
        
        # Reassign tasks from failed worker
        task_manager.reassign_worker_tasks("failed_worker", active_tasks)
        
        # Verify task1 was reassigned (it was running)
        self.assertEqual(active_tasks["task1"].state, TaskState.PENDING)
        self.assertIsNone(active_tasks["task1"].worker_id)
        
        # Verify task2 was not reassigned (it was completed)
        self.assertEqual(active_tasks["task2"].state, TaskState.COMPLETED)
        self.assertEqual(active_tasks["task2"].worker_id, "failed_worker")
    
    def test_processor_statistics(self):
        """Test processor statistics collection"""
        stats = self.processor.get_stats()
        
        # Check base statistics
        self.assertIn('chunks_processed', stats)
        self.assertIn('avg_processing_time', stats)
        self.assertIn('throughput', stats)
        self.assertIn('max_workers', stats)
        
        # Check health monitoring stats
        if self.config.enable_health_monitoring:
            self.assertIn('worker_health', stats)
        
        # Check fault tolerance stats
        if self.config.enable_fault_tolerance:
            self.assertIn('fault_tolerance', stats)
        
        # Check alerting stats
        if self.config.enable_alerting:
            self.assertIn('alerts', stats)
    
    def test_processor_health_status(self):
        """Test processor health status"""
        health_status = self.processor.get_health_status()
        
        self.assertIn('is_running', health_status)
        self.assertIn('worker_count', health_status)
        self.assertIn('healthy_workers', health_status)
        self.assertIn('active_tasks', health_status)
        self.assertIn('active_alerts', health_status)
        
        # Should initially be running
        self.assertTrue(health_status['is_running'])
    
    def test_processor_shutdown_cleanup(self):
        """Test processor shutdown and cleanup"""
        # Add some active tasks
        task = TaskMetadata(
            task_id="test_task",
            chunk_id="test_chunk",
            state=TaskState.RUNNING,
            worker_id="test_worker",
            created_at=time.time(),
            started_at=time.time(),
            completed_at=None,
            retry_count=0,
            max_retries=3,
            error_message=None,
            checkpoint_data=None
        )
        
        self.processor.active_tasks["test_task"] = task
        
        # Shutdown processor
        self.processor.shutdown()
        
        # Check if final checkpoint was created
        checkpoint_files = list(Path(self.temp_dir).glob("checkpoint_*.json"))
        self.assertGreater(len(checkpoint_files), 0)
        
        # Check if processor is no longer running
        self.assertFalse(self.processor.is_running)


class TestResilienceScenarios(unittest.TestCase):
    """Test resilience scenarios"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ParallelConfig(
            max_workers=2,
            enable_fault_tolerance=True,
            enable_health_monitoring=True,
            checkpoint_directory=self.temp_dir,
            max_task_retries=2,
            retry_delay_seconds=0.1
        )
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_processing_with_failures(self):
        """Test processing continues despite failures"""
        processor = ParallelProcessor(self.config)
        
        def failing_processor(chunk):
            # Simulate 50% failure rate
            if hash(chunk.chunk_id) % 2 == 0:
                raise Exception("Simulated failure")
            return chunk
        
        # Create test chunks
        chunks = [
            DataChunk(chunk_id=f"chunk_{i}", data=f"data_{i}", metadata={})
            for i in range(10)
        ]
        
        # Mock data loader
        with patch.object(processor.data_loader, 'load_chunks') as mock_load:
            mock_load.return_value = chunks
            
            # Process with failures
            results = list(processor.process_files_parallel(
                ["test_file.txt"], 
                failing_processor
            ))
            
            # Should get some results despite failures
            self.assertGreater(len(results), 0)
            self.assertLess(len(results), len(chunks))  # Some should have failed
        
        processor.shutdown()
    
    def test_memory_pressure_handling(self):
        """Test handling of memory pressure"""
        processor = ParallelProcessor(self.config)
        
        def memory_intensive_processor(chunk):
            # Simulate memory usage
            large_data = [0] * 1000000  # 1M integers
            time.sleep(0.1)
            del large_data
            return chunk
        
        # Create test chunks
        chunks = [
            DataChunk(chunk_id=f"chunk_{i}", data=f"data_{i}", metadata={})
            for i in range(5)
        ]
        
        # Mock data loader
        with patch.object(processor.data_loader, 'load_chunks') as mock_load:
            mock_load.return_value = chunks
            
            # Process with memory pressure
            results = list(processor.process_files_parallel(
                ["test_file.txt"], 
                memory_intensive_processor
            ))
            
            # Should complete successfully
            self.assertEqual(len(results), len(chunks))
        
        processor.shutdown()


if __name__ == '__main__':
    unittest.main()