"""
Comprehensive Test Suite for Training Optimization
Tests incremental learning, gradient accumulation, distributed training, and large dataset handling
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
import json
import os
import tempfile
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc

# Import our optimization modules
from incremental_learning_manager import (
    IncrementalLearningManager, 
    IncrementalLearningConfig,
    create_incremental_learning_config
)
from gradient_accumulation_optimizer import (
    GradientAccumulationOptimizer,
    GradientAccumulationConfig,
    create_gradient_accumulation_config
)
from distributed_training_coordinator import (
    DistributedTrainingCoordinator,
    DistributedConfig,
    create_distributed_config_auto
)
from optimized_checkpoint_manager import (
    OptimizedCheckpointManager,
    CheckpointConfig,
    create_checkpoint_config
)
from training_progress_monitor import (
    TrainingProgressMonitor,
    MonitoringConfig,
    create_monitoring_config
)
from early_stopping_convergence import (
    EarlyStoppingConvergenceDetector,
    ConvergenceConfig,
    create_convergence_config
)

logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Test result data structure"""
    test_name: str
    passed: bool
    execution_time: float
    memory_usage_mb: float
    performance_metrics: Dict[str, Any]
    error_message: Optional[str] = None
    details: Dict[str, Any] = None

class MockModel(nn.Module):
    """Mock model for testing"""
    
    def __init__(self, input_size: int = 100, hidden_size: int = 256, output_size: int = 10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        return self.layers(x)
    
    def get_model_info(self):
        """Get model information for checkpoint manager"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_shape': (100,),
            'output_size': 10
        }

class LargeDatasetGenerator:
    """Generate large synthetic datasets for testing"""
    
    def __init__(self, 
                 total_samples: int = 100000,
                 features: int = 100,
                 sequence_length: int = 1000,
                 noise_level: float = 0.1):
        self.total_samples = total_samples
        self.features = features
        self.sequence_length = sequence_length
        self.noise_level = noise_level
        
    def generate_time_series_data(self) -> np.ndarray:
        """Generate synthetic time series data"""
        logger.info(f"Generating time series data: {self.total_samples} samples, {self.features} features")
        
        # Generate base patterns
        time_steps = np.linspace(0, 100, self.sequence_length)
        
        data = np.zeros((self.total_samples, self.features))
        
        for i in range(self.total_samples):
            # Generate different patterns for each sample
            patterns = []
            
            # Sine wave pattern
            freq = np.random.uniform(0.1, 2.0)
            phase = np.random.uniform(0, 2 * np.pi)
            sine_pattern = np.sin(freq * time_steps + phase)
            
            # Trend pattern
            trend_slope = np.random.uniform(-0.1, 0.1)
            trend_pattern = trend_slope * time_steps
            
            # Random walk pattern
            random_walk = np.cumsum(np.random.normal(0, 0.1, self.sequence_length))
            
            # Combine patterns
            base_pattern = sine_pattern + trend_pattern + random_walk
            
            # Add noise
            noise = np.random.normal(0, self.noise_level, self.sequence_length)
            final_pattern = base_pattern + noise
            
            # Take features from the pattern
            if self.features <= len(final_pattern):
                data[i] = final_pattern[:self.features]
            else:
                # Repeat pattern if needed
                repeated = np.tile(final_pattern, (self.features // len(final_pattern)) + 1)
                data[i] = repeated[:self.features]
                
        return data.astype(np.float32)
    
    def generate_classification_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic classification data"""
        logger.info(f"Generating classification data: {self.total_samples} samples")
        
        # Generate features
        X = np.random.randn(self.total_samples, self.features).astype(np.float32)
        
        # Generate labels based on feature combinations
        y = np.zeros(self.total_samples, dtype=np.int64)
        
        for i in range(self.total_samples):
            # Simple decision boundary
            if X[i, 0] + X[i, 1] > 0:
                y[i] = 1 if X[i, 2] > 0 else 2
            else:
                y[i] = 3 if np.sum(X[i, :5]) > 0 else 0
        
        return X, y
    
    def save_to_csv(self, data: np.ndarray, filepath: str):
        """Save data to CSV file"""
        pd.DataFrame(data).to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")

class TrainingOptimizationTestSuite:
    """Comprehensive test suite for training optimization"""
    
    def __init__(self, test_dir: str = "training_optimization_tests"):
        self.test_dir = Path(test_dir)
        self.test_dir.mkdir(exist_ok=True)
        
        # Test configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_results = []
        
        # Performance tracking
        self.performance_history = []
        
        logger.info(f"Training optimization test suite initialized in {self.test_dir}")
        logger.info(f"Using device: {self.device}")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all optimization tests"""
        logger.info("Starting comprehensive training optimization tests")
        
        # Test individual components
        self.test_incremental_learning()
        self.test_gradient_accumulation()
        self.test_distributed_training()
        self.test_checkpoint_manager()
        self.test_progress_monitoring()
        self.test_early_stopping()
        
        # Integration tests
        self.test_large_dataset_training()
        self.test_memory_efficiency()
        self.test_convergence_detection()
        
        # Performance benchmarks
        self.benchmark_training_speed()
        self.benchmark_memory_usage()
        
        # Generate report
        return self.generate_test_report()
    
    def test_incremental_learning(self):
        """Test incremental learning manager"""
        logger.info("Testing incremental learning manager")
        
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024**2
        
        try:
            # Create test data
            dataset_size = 10000
            generator = LargeDatasetGenerator(total_samples=dataset_size)
            test_data = generator.generate_time_series_data()
            
            # Save test data
            test_file = self.test_dir / "incremental_test_data.csv"
            generator.save_to_csv(test_data, str(test_file))
            
            # Create model and optimizer
            model = MockModel(input_size=100, hidden_size=128, output_size=10)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            
            # Create incremental learning config
            config = create_incremental_learning_config(
                dataset_size_gb=0.1,  # Small test dataset
                available_memory_gb=4.0,
                target_epochs=1
            )
            
            # Initialize incremental learning manager
            manager = IncrementalLearningManager(model, optimizer, config, self.device)
            
            # Run incremental training
            results = manager.train_incremental(str(test_file), num_epochs=1)
            
            # Verify results
            assert results['chunks_processed'] > 0, "No chunks processed"
            assert results['total_samples'] == dataset_size, f"Expected {dataset_size} samples, got {results['total_samples']}"
            assert results['final_loss'] < 100, f"Loss too high: {results['final_loss']}"
            
            # Test state saving/loading
            state_file = self.test_dir / "incremental_state.pt"
            manager.save_state(str(state_file))
            
            # Create new manager and load state
            new_manager = IncrementalLearningManager(model, optimizer, config, self.device)
            new_manager.load_state(str(state_file))
            
            execution_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss / 1024**2
            
            result = TestResult(
                test_name="incremental_learning",
                passed=True,
                execution_time=execution_time,
                memory_usage_mb=final_memory - initial_memory,
                performance_metrics={
                    'chunks_processed': results['chunks_processed'],
                    'total_samples': results['total_samples'],
                    'final_loss': results['final_loss'],
                    'avg_memory_usage_gb': results['avg_memory_usage_gb']
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss / 1024**2
            
            result = TestResult(
                test_name="incremental_learning",
                passed=False,
                execution_time=execution_time,
                memory_usage_mb=final_memory - initial_memory,
                performance_metrics={},
                error_message=str(e)
            )
            logger.error(f"Incremental learning test failed: {e}")
        
        self.test_results.append(result)
        logger.info(f"Incremental learning test completed: {'PASSED' if result.passed else 'FAILED'}")
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation optimizer"""
        logger.info("Testing gradient accumulation optimizer")
        
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024**2
        
        try:
            # Create model and optimizer
            model = MockModel(input_size=100, hidden_size=256, output_size=10).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            
            # Create gradient accumulation config
            config = create_gradient_accumulation_config(
                available_memory_gb=4.0,
                target_batch_size=128,
                model_size_mb=10.0
            )
            
            # Initialize gradient accumulation optimizer
            grad_optimizer = GradientAccumulationOptimizer(model, optimizer, config, self.device)
            
            # Generate test data
            batch_size = 32
            num_batches = 20
            
            total_loss = 0.0
            update_count = 0
            
            for i in range(num_batches):
                # Generate batch
                batch_data = torch.randn(batch_size, 100).to(self.device)
                target_data = torch.randn(batch_size, 10).to(self.device)
                
                # Define loss function
                def loss_fn(output, target):
                    return nn.MSELoss()(output, target)
                
                # Perform optimization step
                metrics = grad_optimizer.step(batch_data, loss_fn, target_data)
                
                total_loss += metrics['loss']
                if metrics['updated']:
                    update_count += 1
            
            # Get optimization statistics
            stats = grad_optimizer.get_optimization_stats()
            
            # Verify results
            assert update_count > 0, "No gradient updates performed"
            assert stats['total_samples'] == batch_size * num_batches, f"Expected {batch_size * num_batches} samples"
            assert total_loss / num_batches < 100, f"Average loss too high: {total_loss / num_batches}"
            
            execution_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss / 1024**2
            
            result = TestResult(
                test_name="gradient_accumulation",
                passed=True,
                execution_time=execution_time,
                memory_usage_mb=final_memory - initial_memory,
                performance_metrics={
                    'total_samples': stats['total_samples'],
                    'update_count': update_count,
                    'avg_loss': total_loss / num_batches,
                    'avg_step_time': stats['performance']['avg_step_time'],
                    'samples_per_second': stats['performance']['samples_per_second']
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss / 1024**2
            
            result = TestResult(
                test_name="gradient_accumulation",
                passed=False,
                execution_time=execution_time,
                memory_usage_mb=final_memory - initial_memory,
                performance_metrics={},
                error_message=str(e)
            )
            logger.error(f"Gradient accumulation test failed: {e}")
        
        self.test_results.append(result)
        logger.info(f"Gradient accumulation test completed: {'PASSED' if result.passed else 'FAILED'}")
    
    def test_distributed_training(self):
        """Test distributed training coordinator"""
        logger.info("Testing distributed training coordinator")
        
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024**2
        
        try:
            # Create model
            model = MockModel(input_size=100, hidden_size=128, output_size=10)
            
            # Create distributed config (single process for testing)
            config = create_distributed_config_auto(
                model_size_mb=10.0,
                dataset_size_gb=0.1,
                target_batch_size=64
            )
            config.world_size = 1  # Single process test
            
            # Initialize distributed coordinator
            coordinator = DistributedTrainingCoordinator(model, config, self.device)
            
            # Test training step
            batch_size = 32
            num_steps = 10
            
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            
            total_loss = 0.0
            for i in range(num_steps):
                # Generate batch
                batch_data = torch.randn(batch_size, 100).to(self.device)
                target_data = torch.randn(batch_size, 10).to(self.device)
                
                # Define loss function
                def loss_fn(output, target):
                    return nn.MSELoss()(output, target)
                
                # Perform training step
                metrics = coordinator.train_step(batch_data, loss_fn, optimizer, target_data)
                total_loss += metrics['loss']
            
            # Get training statistics
            stats = coordinator.get_training_stats()
            
            # Test checkpointing
            checkpoint_path = coordinator.save_distributed_checkpoint(
                model.state_dict(),
                optimizer.state_dict(),
                epoch=1,
                step=num_steps
            )
            
            # Verify results
            assert os.path.exists(checkpoint_path), "Checkpoint not created"
            assert total_loss / num_steps < 100, f"Average loss too high: {total_loss / num_steps}"
            
            execution_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss / 1024**2
            
            result = TestResult(
                test_name="distributed_training",
                passed=True,
                execution_time=execution_time,
                memory_usage_mb=final_memory - initial_memory,
                performance_metrics={
                    'num_steps': num_steps,
                    'avg_loss': total_loss / num_steps,
                    'avg_throughput': stats['performance']['avg_throughput'],
                    'checkpoint_created': os.path.exists(checkpoint_path)
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss / 1024**2
            
            result = TestResult(
                test_name="distributed_training",
                passed=False,
                execution_time=execution_time,
                memory_usage_mb=final_memory - initial_memory,
                performance_metrics={},
                error_message=str(e)
            )
            logger.error(f"Distributed training test failed: {e}")
        
        self.test_results.append(result)
        logger.info(f"Distributed training test completed: {'PASSED' if result.passed else 'FAILED'}")
    
    def test_checkpoint_manager(self):
        """Test optimized checkpoint manager"""
        logger.info("Testing optimized checkpoint manager")
        
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024**2
        
        try:
            # Create checkpoint config
            checkpoint_dir = self.test_dir / "checkpoints"
            config = create_checkpoint_config(
                checkpoint_dir=str(checkpoint_dir),
                compression_enabled=True,
                cloud_backup=False
            )
            
            # Initialize checkpoint manager
            manager = OptimizedCheckpointManager(config)
            
            # Create model and optimizer
            model = MockModel(input_size=100, hidden_size=128, output_size=10)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            
            # Save multiple checkpoints
            checkpoint_ids = []
            for epoch in range(5):
                checkpoint_id = manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=epoch * 100,
                    loss=1.0 / (epoch + 1),  # Decreasing loss
                    metrics={'accuracy': 0.8 + epoch * 0.02},
                    is_best=(epoch == 4)  # Last one is best
                )
                checkpoint_ids.append(checkpoint_id)
            
            # Test loading checkpoint
            if checkpoint_ids:
                loaded_data = manager.load_checkpoint(
                    checkpoint_ids[-1],
                    model,
                    optimizer
                )
                
                assert loaded_data['epoch'] == 4, f"Expected epoch 4, got {loaded_data['epoch']}"
                assert loaded_data['loss'] == 0.2, f"Expected loss 0.2, got {loaded_data['loss']}"
            
            # Test best checkpoint retrieval
            best_checkpoint = manager.get_best_checkpoint()
            assert best_checkpoint is not None, "Best checkpoint not found"
            
            # Get statistics
            stats = manager.get_checkpoint_stats()
            
            # Verify results
            assert stats['total_checkpoints'] == 5, f"Expected 5 checkpoints, got {stats['total_checkpoints']}"
            assert stats['total_size_mb'] > 0, "No checkpoint size recorded"
            
            execution_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss / 1024**2
            
            result = TestResult(
                test_name="checkpoint_manager",
                passed=True,
                execution_time=execution_time,
                memory_usage_mb=final_memory - initial_memory,
                performance_metrics={
                    'total_checkpoints': stats['total_checkpoints'],
                    'total_size_mb': stats['total_size_mb'],
                    'avg_save_time': stats['avg_save_time'],
                    'avg_compression_ratio': stats['avg_compression_ratio']
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss / 1024**2
            
            result = TestResult(
                test_name="checkpoint_manager",
                passed=False,
                execution_time=execution_time,
                memory_usage_mb=final_memory - initial_memory,
                performance_metrics={},
                error_message=str(e)
            )
            logger.error(f"Checkpoint manager test failed: {e}")
        
        self.test_results.append(result)
        logger.info(f"Checkpoint manager test completed: {'PASSED' if result.passed else 'FAILED'}")
    
    def test_progress_monitoring(self):
        """Test training progress monitor"""
        logger.info("Testing training progress monitor")
        
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024**2
        
        try:
            # Create monitoring config
            config = create_monitoring_config(
                enable_live_plots=False,  # Disable for testing
                enable_alerts=True,
                save_to_disk=True,
                track_system_metrics=True
            )
            
            # Initialize monitor
            monitor = TrainingProgressMonitor(config)
            monitor.start_monitoring()
            
            # Log some metrics
            for epoch in range(10):
                for step in range(5):
                    loss = 1.0 / (epoch * 5 + step + 1)  # Decreasing loss
                    accuracy = 0.5 + (epoch * 5 + step) * 0.01  # Increasing accuracy
                    
                    monitor.log_multiple_metrics({
                        'train_loss': loss,
                        'train_accuracy': accuracy,
                        'learning_rate': 1e-3,
                        'gradient_norm': 0.1 / (step + 1)
                    }, epoch, step)
            
            # Test phase tracking
            monitor.start_phase("training", "Main training phase")
            time.sleep(0.1)  # Brief pause
            monitor.end_phase()
            
            # Test alerts
            monitor.add_alert_rule('train_loss', 0.5, 'greater')
            monitor.log_metric('train_loss', 0.6, 5, 25)  # Should trigger alert
            
            # Wait for monitoring to process
            time.sleep(1.0)
            
            # Get training summary
            summary = monitor.get_training_summary()
            
            # Stop monitoring
            monitor.stop_monitoring()
            
            # Verify results
            assert 'train_loss' in summary['metrics_summary'], "Train loss not tracked"
            assert 'train_accuracy' in summary['metrics_summary'], "Train accuracy not tracked"
            assert summary['metrics_summary']['train_loss']['current'] < 0.1, "Loss not decreasing"
            assert len(summary['training_phases']) == 1, "Phase not tracked"
            
            execution_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss / 1024**2
            
            result = TestResult(
                test_name="progress_monitoring",
                passed=True,
                execution_time=execution_time,
                memory_usage_mb=final_memory - initial_memory,
                performance_metrics={
                    'metrics_tracked': len(summary['metrics_summary']),
                    'phases_tracked': len(summary['training_phases']),
                    'total_duration': summary['total_duration_seconds'],
                    'alerts_triggered': summary['alerts']['total_alerts']
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss / 1024**2
            
            result = TestResult(
                test_name="progress_monitoring",
                passed=False,
                execution_time=execution_time,
                memory_usage_mb=final_memory - initial_memory,
                performance_metrics={},
                error_message=str(e)
            )
            logger.error(f"Progress monitoring test failed: {e}")
        
        self.test_results.append(result)
        logger.info(f"Progress monitoring test completed: {'PASSED' if result.passed else 'FAILED'}")
    
    def test_early_stopping(self):
        """Test early stopping and convergence detection"""
        logger.info("Testing early stopping and convergence detection")
        
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024**2
        
        try:
            # Create convergence config
            config = create_convergence_config(
                patience=5,
                min_delta=1e-4,
                enable_all_methods=True
            )
            
            # Initialize detector
            detector = EarlyStoppingConvergenceDetector(config)
            
            # Create mock model
            model = MockModel(input_size=100, hidden_size=64, output_size=10)
            
            # Simulate training with converging loss
            should_stop = False
            convergence_result = None
            
            for epoch in range(20):
                # Simulate decreasing loss with some noise
                base_loss = 1.0 / (epoch + 1)
                noise = np.random.normal(0, 0.01)
                train_loss = base_loss + noise
                
                validation_loss = train_loss + np.random.normal(0, 0.005)
                
                # Update detector
                should_stop, convergence_result = detector.update(
                    train_loss=train_loss,
                    validation_loss=validation_loss,
                    model=model,
                    epoch=epoch,
                    step=epoch * 100
                )
                
                if should_stop:
                    break
            
            # Verify results
            assert should_stop, "Early stopping not triggered"
            assert convergence_result is not None, "No convergence result"
            assert convergence_result.stopping_reason is not None, "No stopping reason"
            
            # Get convergence summary
            summary = detector.get_convergence_summary()
            
            execution_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss / 1024**2
            
            result = TestResult(
                test_name="early_stopping",
                passed=True,
                execution_time=execution_time,
                memory_usage_mb=final_memory - initial_memory,
                performance_metrics={
                    'converged': convergence_result.converged,
                    'stopping_reason': convergence_result.stopping_reason.value,
                    'convergence_epoch': convergence_result.convergence_epoch,
                    'final_loss': convergence_result.final_value,
                    'confidence': convergence_result.confidence
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss / 1024**2
            
            result = TestResult(
                test_name="early_stopping",
                passed=False,
                execution_time=execution_time,
                memory_usage_mb=final_memory - initial_memory,
                performance_metrics={},
                error_message=str(e)
            )
            logger.error(f"Early stopping test failed: {e}")
        
        self.test_results.append(result)
        logger.info(f"Early stopping test completed: {'PASSED' if result.passed else 'FAILED'}")
    
    def test_large_dataset_training(self):
        """Test integrated training with large dataset"""
        logger.info("Testing integrated training with large dataset")
        
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024**2
        
        try:
            # Create large dataset
            dataset_size = 50000
            generator = LargeDatasetGenerator(total_samples=dataset_size, features=100)
            test_data = generator.generate_time_series_data()
            
            # Save test data
            test_file = self.test_dir / "large_dataset_test.csv"
            generator.save_to_csv(test_data, str(test_file))
            
            # Create integrated training setup
            model = MockModel(input_size=100, hidden_size=256, output_size=10)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            
            # Initialize all components
            incremental_config = create_incremental_learning_config(
                dataset_size_gb=0.5,
                available_memory_gb=4.0,
                target_epochs=1
            )
            
            checkpoint_config = create_checkpoint_config(
                checkpoint_dir=str(self.test_dir / "large_checkpoints"),
                compression_enabled=True
            )
            
            monitoring_config = create_monitoring_config(
                enable_live_plots=False,
                enable_alerts=True
            )
            
            # Initialize managers
            incremental_manager = IncrementalLearningManager(model, optimizer, incremental_config, self.device)
            checkpoint_manager = OptimizedCheckpointManager(checkpoint_config)
            monitor = TrainingProgressMonitor(monitoring_config)
            
            # Start monitoring
            monitor.start_monitoring()
            
            # Train with incremental learning
            training_results = incremental_manager.train_incremental(str(test_file), num_epochs=1)
            
            # Save checkpoint
            checkpoint_id = checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=1,
                step=training_results['chunks_processed'],
                loss=training_results['final_loss'],
                metrics=training_results
            )
            
            # Log final metrics
            monitor.log_multiple_metrics({
                'final_loss': training_results['final_loss'],
                'chunks_processed': training_results['chunks_processed'],
                'total_samples': training_results['total_samples']
            }, 1, training_results['chunks_processed'])
            
            # Stop monitoring
            monitor.stop_monitoring()
            
            # Verify results
            assert training_results['chunks_processed'] > 0, "No chunks processed"
            assert training_results['total_samples'] == dataset_size, "Sample count mismatch"
            assert checkpoint_id is not None, "Checkpoint not saved"
            
            execution_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss / 1024**2
            
            result = TestResult(
                test_name="large_dataset_training",
                passed=True,
                execution_time=execution_time,
                memory_usage_mb=final_memory - initial_memory,
                performance_metrics={
                    'dataset_size': dataset_size,
                    'chunks_processed': training_results['chunks_processed'],
                    'final_loss': training_results['final_loss'],
                    'avg_memory_usage_gb': training_results['avg_memory_usage_gb'],
                    'checkpoint_saved': checkpoint_id is not None
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss / 1024**2
            
            result = TestResult(
                test_name="large_dataset_training",
                passed=False,
                execution_time=execution_time,
                memory_usage_mb=final_memory - initial_memory,
                performance_metrics={},
                error_message=str(e)
            )
            logger.error(f"Large dataset training test failed: {e}")
        
        self.test_results.append(result)
        logger.info(f"Large dataset training test completed: {'PASSED' if result.passed else 'FAILED'}")
    
    def test_memory_efficiency(self):
        """Test memory efficiency optimizations"""
        logger.info("Testing memory efficiency optimizations")
        
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024**2
        
        try:
            # Test memory usage with different configurations
            model = MockModel(input_size=100, hidden_size=512, output_size=10)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            
            # Test without optimizations
            baseline_memory = []
            for i in range(10):
                batch = torch.randn(128, 100).to(self.device)
                output = model(batch)
                loss = nn.MSELoss()(output, torch.randn(128, 10).to(self.device))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                baseline_memory.append(psutil.Process().memory_info().rss / 1024**2)
            
            # Clean up
            del batch, output, loss
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Test with gradient accumulation
            config = create_gradient_accumulation_config(
                available_memory_gb=4.0,
                target_batch_size=512,
                model_size_mb=50.0
            )
            
            grad_optimizer = GradientAccumulationOptimizer(model, optimizer, config, self.device)
            
            optimized_memory = []
            for i in range(10):
                batch = torch.randn(32, 100).to(self.device)  # Smaller batch
                target = torch.randn(32, 10).to(self.device)
                
                metrics = grad_optimizer.step(batch, nn.MSELoss(), target)
                optimized_memory.append(psutil.Process().memory_info().rss / 1024**2)
            
            # Calculate memory efficiency
            avg_baseline = np.mean(baseline_memory)
            avg_optimized = np.mean(optimized_memory)
            memory_improvement = (avg_baseline - avg_optimized) / avg_baseline * 100
            
            execution_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss / 1024**2
            
            result = TestResult(
                test_name="memory_efficiency",
                passed=True,
                execution_time=execution_time,
                memory_usage_mb=final_memory - initial_memory,
                performance_metrics={
                    'baseline_memory_mb': avg_baseline,
                    'optimized_memory_mb': avg_optimized,
                    'memory_improvement_percent': memory_improvement,
                    'peak_memory_mb': max(baseline_memory + optimized_memory)
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss / 1024**2
            
            result = TestResult(
                test_name="memory_efficiency",
                passed=False,
                execution_time=execution_time,
                memory_usage_mb=final_memory - initial_memory,
                performance_metrics={},
                error_message=str(e)
            )
            logger.error(f"Memory efficiency test failed: {e}")
        
        self.test_results.append(result)
        logger.info(f"Memory efficiency test completed: {'PASSED' if result.passed else 'FAILED'}")
    
    def test_convergence_detection(self):
        """Test convergence detection with various scenarios"""
        logger.info("Testing convergence detection scenarios")
        
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024**2
        
        try:
            # Test different convergence scenarios
            scenarios = {
                'fast_convergence': lambda epoch: 1.0 / (epoch + 1)**2,
                'slow_convergence': lambda epoch: 1.0 / (epoch + 1)**0.5,
                'plateau': lambda epoch: 0.1 + 0.01 * np.random.normal(),
                'oscillating': lambda epoch: 0.5 + 0.1 * np.sin(epoch * 0.5)
            }
            
            results = {}
            
            for scenario_name, loss_func in scenarios.items():
                config = create_convergence_config(patience=10, min_delta=1e-4)
                detector = EarlyStoppingConvergenceDetector(config)
                
                should_stop = False
                convergence_result = None
                
                for epoch in range(50):
                    loss = loss_func(epoch)
                    should_stop, convergence_result = detector.update(
                        train_loss=loss,
                        epoch=epoch,
                        step=epoch * 100
                    )
                    
                    if should_stop:
                        break
                
                results[scenario_name] = {
                    'stopped': should_stop,
                    'epoch': epoch,
                    'final_loss': loss,
                    'convergence_result': convergence_result
                }
            
            # Verify results
            assert results['fast_convergence']['stopped'], "Fast convergence not detected"
            assert results['plateau']['stopped'], "Plateau not detected"
            
            execution_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss / 1024**2
            
            result = TestResult(
                test_name="convergence_detection",
                passed=True,
                execution_time=execution_time,
                memory_usage_mb=final_memory - initial_memory,
                performance_metrics={
                    'scenarios_tested': len(scenarios),
                    'fast_convergence_epoch': results['fast_convergence']['epoch'],
                    'plateau_detected': results['plateau']['stopped'],
                    'all_scenarios_handled': all(r['stopped'] for r in results.values())
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss / 1024**2
            
            result = TestResult(
                test_name="convergence_detection",
                passed=False,
                execution_time=execution_time,
                memory_usage_mb=final_memory - initial_memory,
                performance_metrics={},
                error_message=str(e)
            )
            logger.error(f"Convergence detection test failed: {e}")
        
        self.test_results.append(result)
        logger.info(f"Convergence detection test completed: {'PASSED' if result.passed else 'FAILED'}")
    
    def benchmark_training_speed(self):
        """Benchmark training speed optimizations"""
        logger.info("Benchmarking training speed optimizations")
        
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024**2
        
        try:
            # Test different configurations
            configurations = [
                {'name': 'baseline', 'mixed_precision': False, 'gradient_accumulation': False},
                {'name': 'mixed_precision', 'mixed_precision': True, 'gradient_accumulation': False},
                {'name': 'gradient_accumulation', 'mixed_precision': False, 'gradient_accumulation': True},
                {'name': 'optimized', 'mixed_precision': True, 'gradient_accumulation': True}
            ]
            
            benchmark_results = {}
            
            for config in configurations:
                model = MockModel(input_size=100, hidden_size=256, output_size=10).to(self.device)
                optimizer = optim.Adam(model.parameters(), lr=1e-3)
                
                # Setup based on configuration
                if config['gradient_accumulation']:
                    grad_config = create_gradient_accumulation_config(
                        available_memory_gb=4.0,
                        target_batch_size=128,
                        model_size_mb=20.0
                    )
                    grad_config.use_mixed_precision = config['mixed_precision']
                    grad_optimizer = GradientAccumulationOptimizer(model, optimizer, grad_config, self.device)
                
                # Benchmark training
                num_batches = 50
                batch_times = []
                
                for i in range(num_batches):
                    batch_start = time.time()
                    
                    batch_data = torch.randn(32, 100).to(self.device)
                    target_data = torch.randn(32, 10).to(self.device)
                    
                    if config['gradient_accumulation']:
                        metrics = grad_optimizer.step(batch_data, nn.MSELoss(), target_data)
                    else:
                        optimizer.zero_grad()
                        
                        if config['mixed_precision']:
                            with torch.cuda.amp.autocast():
                                output = model(batch_data)
                                loss = nn.MSELoss()(output, target_data)
                        else:
                            output = model(batch_data)
                            loss = nn.MSELoss()(output, target_data)
                        
                        loss.backward()
                        optimizer.step()
                    
                    batch_times.append(time.time() - batch_start)
                
                benchmark_results[config['name']] = {
                    'avg_batch_time': np.mean(batch_times),
                    'total_time': np.sum(batch_times),
                    'throughput': num_batches / np.sum(batch_times)
                }
            
            # Calculate improvements
            baseline_time = benchmark_results['baseline']['avg_batch_time']
            optimized_time = benchmark_results['optimized']['avg_batch_time']
            speed_improvement = (baseline_time - optimized_time) / baseline_time * 100
            
            execution_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss / 1024**2
            
            result = TestResult(
                test_name="training_speed_benchmark",
                passed=True,
                execution_time=execution_time,
                memory_usage_mb=final_memory - initial_memory,
                performance_metrics={
                    'baseline_batch_time': baseline_time,
                    'optimized_batch_time': optimized_time,
                    'speed_improvement_percent': speed_improvement,
                    'benchmark_results': benchmark_results
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss / 1024**2
            
            result = TestResult(
                test_name="training_speed_benchmark",
                passed=False,
                execution_time=execution_time,
                memory_usage_mb=final_memory - initial_memory,
                performance_metrics={},
                error_message=str(e)
            )
            logger.error(f"Training speed benchmark failed: {e}")
        
        self.test_results.append(result)
        logger.info(f"Training speed benchmark completed: {'PASSED' if result.passed else 'FAILED'}")
    
    def benchmark_memory_usage(self):
        """Benchmark memory usage optimizations"""
        logger.info("Benchmarking memory usage optimizations")
        
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024**2
        
        try:
            # Test memory usage with different model sizes
            model_sizes = [
                {'hidden_size': 128, 'name': 'small'},
                {'hidden_size': 256, 'name': 'medium'},
                {'hidden_size': 512, 'name': 'large'}
            ]
            
            memory_results = {}
            
            for model_config in model_sizes:
                model = MockModel(
                    input_size=100,
                    hidden_size=model_config['hidden_size'],
                    output_size=10
                ).to(self.device)
                
                optimizer = optim.Adam(model.parameters(), lr=1e-3)
                
                # Test with gradient accumulation
                grad_config = create_gradient_accumulation_config(
                    available_memory_gb=4.0,
                    target_batch_size=128,
                    model_size_mb=model_config['hidden_size'] / 10
                )
                
                grad_optimizer = GradientAccumulationOptimizer(model, optimizer, grad_config, self.device)
                
                # Measure memory usage
                memory_before = psutil.Process().memory_info().rss / 1024**2
                
                for i in range(10):
                    batch_data = torch.randn(32, 100).to(self.device)
                    target_data = torch.randn(32, 10).to(self.device)
                    
                    metrics = grad_optimizer.step(batch_data, nn.MSELoss(), target_data)
                
                memory_after = psutil.Process().memory_info().rss / 1024**2
                
                memory_results[model_config['name']] = {
                    'memory_usage_mb': memory_after - memory_before,
                    'model_parameters': sum(p.numel() for p in model.parameters()),
                    'hidden_size': model_config['hidden_size']
                }
                
                # Clean up
                del model, optimizer, grad_optimizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            execution_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss / 1024**2
            
            result = TestResult(
                test_name="memory_usage_benchmark",
                passed=True,
                execution_time=execution_time,
                memory_usage_mb=final_memory - initial_memory,
                performance_metrics={
                    'memory_results': memory_results,
                    'memory_scaling': 'efficient' if memory_results['large']['memory_usage_mb'] < memory_results['small']['memory_usage_mb'] * 4 else 'linear'
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss / 1024**2
            
            result = TestResult(
                test_name="memory_usage_benchmark",
                passed=False,
                execution_time=execution_time,
                memory_usage_mb=final_memory - initial_memory,
                performance_metrics={},
                error_message=str(e)
            )
            logger.error(f"Memory usage benchmark failed: {e}")
        
        self.test_results.append(result)
        logger.info(f"Memory usage benchmark completed: {'PASSED' if result.passed else 'FAILED'}")
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        logger.info("Generating comprehensive test report")
        
        # Calculate overall statistics
        passed_tests = [r for r in self.test_results if r.passed]
        failed_tests = [r for r in self.test_results if not r.passed]
        
        total_time = sum(r.execution_time for r in self.test_results)
        total_memory = sum(r.memory_usage_mb for r in self.test_results)
        
        # Generate report
        report = {
            'summary': {
                'total_tests': len(self.test_results),
                'passed_tests': len(passed_tests),
                'failed_tests': len(failed_tests),
                'success_rate': len(passed_tests) / len(self.test_results) * 100,
                'total_execution_time': total_time,
                'total_memory_usage_mb': total_memory
            },
            'test_results': {
                result.test_name: {
                    'passed': result.passed,
                    'execution_time': result.execution_time,
                    'memory_usage_mb': result.memory_usage_mb,
                    'performance_metrics': result.performance_metrics,
                    'error_message': result.error_message
                }
                for result in self.test_results
            },
            'performance_analysis': {
                'fastest_test': min(self.test_results, key=lambda r: r.execution_time).test_name,
                'slowest_test': max(self.test_results, key=lambda r: r.execution_time).test_name,
                'most_memory_intensive': max(self.test_results, key=lambda r: r.memory_usage_mb).test_name,
                'least_memory_intensive': min(self.test_results, key=lambda r: r.memory_usage_mb).test_name
            },
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        report_file = self.test_dir / "training_optimization_test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate plots
        self._generate_plots()
        
        logger.info(f"Test report generated: {report_file}")
        logger.info(f"Overall success rate: {report['summary']['success_rate']:.1f}%")
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on test results"""
        recommendations = []
        
        # Check for failed tests
        failed_tests = [r for r in self.test_results if not r.passed]
        if failed_tests:
            recommendations.append(f"Address {len(failed_tests)} failed tests: {', '.join(r.test_name for r in failed_tests)}")
        
        # Check memory usage
        memory_intensive_tests = [r for r in self.test_results if r.memory_usage_mb > 100]
        if memory_intensive_tests:
            recommendations.append("Consider memory optimization for memory-intensive tests")
        
        # Check execution time
        slow_tests = [r for r in self.test_results if r.execution_time > 10]
        if slow_tests:
            recommendations.append("Optimize execution time for slow tests")
        
        # Check specific optimizations
        if any(r.test_name == 'memory_efficiency' and r.passed for r in self.test_results):
            memory_test = next(r for r in self.test_results if r.test_name == 'memory_efficiency')
            if memory_test.performance_metrics.get('memory_improvement_percent', 0) > 20:
                recommendations.append("Memory efficiency optimizations are working well")
        
        return recommendations
    
    def _generate_plots(self):
        """Generate performance plots"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Test execution times
            test_names = [r.test_name for r in self.test_results]
            execution_times = [r.execution_time for r in self.test_results]
            
            axes[0, 0].bar(test_names, execution_times)
            axes[0, 0].set_title('Test Execution Times')
            axes[0, 0].set_xlabel('Test')
            axes[0, 0].set_ylabel('Time (seconds)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Memory usage
            memory_usage = [r.memory_usage_mb for r in self.test_results]
            
            axes[0, 1].bar(test_names, memory_usage)
            axes[0, 1].set_title('Memory Usage')
            axes[0, 1].set_xlabel('Test')
            axes[0, 1].set_ylabel('Memory (MB)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Test results
            passed_counts = [1 if r.passed else 0 for r in self.test_results]
            failed_counts = [0 if r.passed else 1 for r in self.test_results]
            
            axes[1, 0].bar(test_names, passed_counts, label='Passed', color='green', alpha=0.7)
            axes[1, 0].bar(test_names, failed_counts, bottom=passed_counts, label='Failed', color='red', alpha=0.7)
            axes[1, 0].set_title('Test Results')
            axes[1, 0].set_xlabel('Test')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].legend()
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Performance metrics (if available)
            if any('speed_improvement_percent' in r.performance_metrics for r in self.test_results):
                speed_test = next(r for r in self.test_results if 'speed_improvement_percent' in r.performance_metrics)
                improvements = [
                    speed_test.performance_metrics.get('speed_improvement_percent', 0),
                    speed_test.performance_metrics.get('memory_improvement_percent', 0)
                ]
                
                axes[1, 1].bar(['Speed', 'Memory'], improvements)
                axes[1, 1].set_title('Performance Improvements')
                axes[1, 1].set_xlabel('Metric')
                axes[1, 1].set_ylabel('Improvement (%)')
            
            plt.tight_layout()
            plt.savefig(self.test_dir / "training_optimization_results.png", dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")


def run_training_optimization_tests(test_dir: str = "training_optimization_tests") -> Dict[str, Any]:
    """Run comprehensive training optimization tests"""
    
    # Create test suite
    test_suite = TrainingOptimizationTestSuite(test_dir)
    
    # Run all tests
    report = test_suite.run_all_tests()
    
    return report


if __name__ == "__main__":
    # Run tests
    logging.basicConfig(level=logging.INFO)
    report = run_training_optimization_tests()
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING OPTIMIZATION TEST SUMMARY")
    print("="*50)
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Passed: {report['summary']['passed_tests']}")
    print(f"Failed: {report['summary']['failed_tests']}")
    print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
    print(f"Total Time: {report['summary']['total_execution_time']:.2f}s")
    print(f"Total Memory: {report['summary']['total_memory_usage_mb']:.2f}MB")
    print("="*50)
    
    # Print recommendations
    if report['recommendations']:
        print("\nRECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"- {rec}")
    
    print("\nTest completed successfully!")