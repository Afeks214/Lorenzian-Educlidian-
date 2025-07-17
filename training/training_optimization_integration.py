"""
Training Optimization Integration Module
Combines all training optimization components into a unified system
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import time
import json
from pathlib import Path
from dataclasses import dataclass
import warnings

# Import all optimization components
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
from performance_analysis_framework import (
    PerformanceAnalysisFramework,
    ModelComplexityAnalyzer
)

logger = logging.getLogger(__name__)

@dataclass
class OptimizedTrainingConfig:
    """Unified configuration for optimized training"""
    # Model and data settings
    model_factory: Callable
    optimizer_factory: Callable
    data_source: str
    input_shape: Tuple[int, ...]
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    
    # Optimization settings
    use_incremental_learning: bool = True
    use_gradient_accumulation: bool = True
    use_distributed_training: bool = False
    use_mixed_precision: bool = True
    
    # Memory and performance
    available_memory_gb: float = 8.0
    target_batch_size: int = 128
    
    # Monitoring and checkpointing
    enable_monitoring: bool = True
    enable_checkpointing: bool = True
    checkpoint_frequency: int = 100
    
    # Early stopping and convergence
    enable_early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    
    # Directories
    checkpoint_dir: str = "optimized_checkpoints"
    monitoring_dir: str = "monitoring_logs"
    results_dir: str = "training_results"
    
    # Device settings
    device: str = "auto"  # "auto", "cpu", "cuda"


class OptimizedTrainingSystem:
    """
    Unified optimized training system that integrates all components
    """
    
    def __init__(self, config: OptimizedTrainingConfig):
        self.config = config
        
        # Setup device
        self.device = self._setup_device()
        
        # Initialize model and optimizer
        self.model = config.model_factory().to(self.device)
        self.optimizer = config.optimizer_factory(self.model.parameters())
        
        # Initialize optimization components
        self.components = {}
        self._initialize_components()
        
        # Training state
        self.training_state = {
            'epoch': 0,
            'step': 0,
            'best_loss': float('inf'),
            'training_time': 0.0,
            'converged': False
        }
        
        # Results storage
        self.training_results = {
            'loss_history': [],
            'performance_metrics': [],
            'convergence_info': {},
            'optimization_stats': {}
        }
        
        logger.info(f"Optimized training system initialized on {self.device}")
        self._log_system_info()
    
    def _setup_device(self) -> torch.device:
        """Setup compute device"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU")
        else:
            device = torch.device(self.config.device)
        
        return device
    
    def _initialize_components(self):
        """Initialize all optimization components"""
        
        # Incremental learning manager
        if self.config.use_incremental_learning:
            incremental_config = create_incremental_learning_config(
                dataset_size_gb=1.0,  # Estimated
                available_memory_gb=self.config.available_memory_gb,
                target_epochs=self.config.epochs
            )
            self.components['incremental_learning'] = IncrementalLearningManager(
                self.model, self.optimizer, incremental_config, self.device
            )
        
        # Gradient accumulation optimizer
        if self.config.use_gradient_accumulation:
            model_complexity = ModelComplexityAnalyzer(self.model)
            model_size_mb = model_complexity.get_complexity_report(self.config.input_shape)['model_size_mb']
            
            grad_config = create_gradient_accumulation_config(
                available_memory_gb=self.config.available_memory_gb,
                target_batch_size=self.config.target_batch_size,
                model_size_mb=model_size_mb
            )
            grad_config.use_mixed_precision = self.config.use_mixed_precision
            
            self.components['gradient_accumulation'] = GradientAccumulationOptimizer(
                self.model, self.optimizer, grad_config, self.device
            )
        
        # Distributed training coordinator
        if self.config.use_distributed_training:
            distributed_config = create_distributed_config_auto(
                model_size_mb=10.0,  # Estimated
                dataset_size_gb=1.0,  # Estimated
                target_batch_size=self.config.target_batch_size
            )
            self.components['distributed_training'] = DistributedTrainingCoordinator(
                self.model, distributed_config, self.device
            )
        
        # Checkpoint manager
        if self.config.enable_checkpointing:
            checkpoint_config = create_checkpoint_config(
                checkpoint_dir=self.config.checkpoint_dir,
                compression_enabled=True,
                cloud_backup=False
            )
            self.components['checkpoint_manager'] = OptimizedCheckpointManager(checkpoint_config)
        
        # Progress monitor
        if self.config.enable_monitoring:
            monitoring_config = create_monitoring_config(
                enable_live_plots=True,
                enable_alerts=True,
                save_to_disk=True,
                track_system_metrics=True
            )
            self.components['progress_monitor'] = TrainingProgressMonitor(monitoring_config)
        
        # Early stopping detector
        if self.config.enable_early_stopping:
            convergence_config = create_convergence_config(
                patience=self.config.patience,
                min_delta=self.config.min_delta,
                enable_all_methods=True
            )
            self.components['early_stopping'] = EarlyStoppingConvergenceDetector(convergence_config)
        
        logger.info(f"Initialized {len(self.components)} optimization components")
    
    def _log_system_info(self):
        """Log system information"""
        if 'progress_monitor' in self.components:
            monitor = self.components['progress_monitor']
            monitor.start_monitoring()
            
            # Log system metrics
            import psutil
            logger.info(f"System Memory: {psutil.virtual_memory().total / 1024**3:.2f} GB")
            logger.info(f"Available Memory: {psutil.virtual_memory().available / 1024**3:.2f} GB")
            logger.info(f"CPU Count: {psutil.cpu_count()}")
            
            if torch.cuda.is_available():
                logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    def train(self) -> Dict[str, Any]:
        """
        Run optimized training with all components
        """
        logger.info("Starting optimized training")
        
        # Start progress monitoring
        if 'progress_monitor' in self.components:
            monitor = self.components['progress_monitor']
            monitor.start_phase("training", "Main training phase")
        
        training_start_time = time.time()
        
        try:
            # Use incremental learning if enabled
            if self.config.use_incremental_learning:
                results = self._train_with_incremental_learning()
            else:
                results = self._train_standard()
            
            # Calculate total training time
            self.training_state['training_time'] = time.time() - training_start_time
            
            # Finalize results
            return self._finalize_training_results(results)
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Stop progress monitoring
            if 'progress_monitor' in self.components:
                monitor = self.components['progress_monitor']
                monitor.end_phase()
                monitor.stop_monitoring()
    
    def _train_with_incremental_learning(self) -> Dict[str, Any]:
        """Train using incremental learning"""
        logger.info("Training with incremental learning")
        
        incremental_manager = self.components['incremental_learning']
        
        # Setup progress monitoring
        monitor = self.components.get('progress_monitor')
        
        # Custom training loop with incremental learning
        results = incremental_manager.train_incremental(
            self.config.data_source,
            num_epochs=self.config.epochs
        )
        
        # Log final metrics
        if monitor:
            monitor.log_multiple_metrics({
                'final_loss': results['final_loss'],
                'total_samples': results['total_samples'],
                'chunks_processed': results['chunks_processed']
            }, self.config.epochs, results['chunks_processed'])
        
        return results
    
    def _train_standard(self) -> Dict[str, Any]:
        """Standard training loop with optimizations"""
        logger.info("Training with standard loop + optimizations")
        
        # Get components
        grad_optimizer = self.components.get('gradient_accumulation')
        distributed_coordinator = self.components.get('distributed_training')
        checkpoint_manager = self.components.get('checkpoint_manager')
        monitor = self.components.get('progress_monitor')
        early_stopping = self.components.get('early_stopping')
        
        # Training loop
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.config.epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            batch_count = 0
            
            # Generate dummy data (replace with actual data loader)
            num_batches_per_epoch = 100
            
            for batch_idx in range(num_batches_per_epoch):
                # Generate batch (replace with actual data)
                batch_data = torch.randn(self.config.batch_size, *self.config.input_shape).to(self.device)
                target_data = torch.randn(self.config.batch_size, 10).to(self.device)  # Assuming 10 classes
                
                # Training step
                if grad_optimizer:
                    # Use gradient accumulation
                    metrics = grad_optimizer.step(batch_data, nn.MSELoss(), target_data)
                    loss = metrics['loss']
                elif distributed_coordinator:
                    # Use distributed training
                    metrics = distributed_coordinator.train_step(batch_data, nn.MSELoss(), self.optimizer, target_data)
                    loss = metrics['loss']
                else:
                    # Standard training step
                    self.optimizer.zero_grad()
                    output = self.model(batch_data)
                    loss = nn.MSELoss()(output, target_data)
                    loss.backward()
                    self.optimizer.step()
                    loss = loss.item()
                
                epoch_loss += loss
                batch_count += 1
                
                # Log metrics
                if monitor:
                    monitor.log_multiple_metrics({
                        'train_loss': loss,
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'batch_time': time.time() - epoch_start_time
                    }, epoch, batch_idx)
            
            # Calculate epoch metrics
            avg_epoch_loss = epoch_loss / batch_count
            total_loss += avg_epoch_loss
            
            # Update training state
            self.training_state['epoch'] = epoch
            self.training_state['step'] = epoch * num_batches_per_epoch
            
            # Check for improvement
            if avg_epoch_loss < self.training_state['best_loss']:
                self.training_state['best_loss'] = avg_epoch_loss
                is_best = True
            else:
                is_best = False
            
            # Save checkpoint
            if checkpoint_manager and (epoch % self.config.checkpoint_frequency == 0 or is_best):
                checkpoint_id = checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    step=self.training_state['step'],
                    loss=avg_epoch_loss,
                    metrics={'epoch_loss': avg_epoch_loss},
                    is_best=is_best
                )
                logger.info(f"Saved checkpoint: {checkpoint_id}")
            
            # Early stopping check
            if early_stopping:
                should_stop, convergence_result = early_stopping.update(
                    train_loss=avg_epoch_loss,
                    model=self.model,
                    epoch=epoch,
                    step=self.training_state['step']
                )
                
                if should_stop:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    self.training_state['converged'] = True
                    self.training_results['convergence_info'] = {
                        'converged': convergence_result.converged,
                        'method': convergence_result.method.value if convergence_result.method else None,
                        'confidence': convergence_result.confidence,
                        'stopping_reason': convergence_result.stopping_reason.value if convergence_result.stopping_reason else None
                    }
                    break
            
            # Log epoch completion
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs} completed in {epoch_time:.2f}s, Loss: {avg_epoch_loss:.6f}")
        
        return {
            'final_loss': self.training_state['best_loss'],
            'total_epochs': self.training_state['epoch'] + 1,
            'total_steps': self.training_state['step'],
            'converged': self.training_state['converged']
        }
    
    def _finalize_training_results(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize and compile training results"""
        logger.info("Finalizing training results")
        
        # Compile comprehensive results
        final_results = {
            'training_config': {
                'epochs': self.config.epochs,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'use_incremental_learning': self.config.use_incremental_learning,
                'use_gradient_accumulation': self.config.use_gradient_accumulation,
                'use_distributed_training': self.config.use_distributed_training,
                'use_mixed_precision': self.config.use_mixed_precision
            },
            'training_results': training_results,
            'training_state': self.training_state,
            'optimization_stats': {},
            'system_performance': {}
        }
        
        # Get optimization statistics
        if 'gradient_accumulation' in self.components:
            grad_optimizer = self.components['gradient_accumulation']
            final_results['optimization_stats']['gradient_accumulation'] = grad_optimizer.get_optimization_stats()
        
        if 'distributed_training' in self.components:
            distributed_coordinator = self.components['distributed_training']
            final_results['optimization_stats']['distributed_training'] = distributed_coordinator.get_training_stats()
        
        if 'checkpoint_manager' in self.components:
            checkpoint_manager = self.components['checkpoint_manager']
            final_results['optimization_stats']['checkpoint_manager'] = checkpoint_manager.get_checkpoint_stats()
        
        # Get progress monitoring summary
        if 'progress_monitor' in self.components:
            monitor = self.components['progress_monitor']
            final_results['system_performance'] = monitor.get_training_summary()
        
        # Get early stopping summary
        if 'early_stopping' in self.components:
            early_stopping = self.components['early_stopping']
            final_results['convergence_summary'] = early_stopping.get_convergence_summary()
        
        # Save results
        self._save_results(final_results)
        
        return final_results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save training results"""
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(exist_ok=True)
        
        # Save JSON results
        results_file = results_dir / f"training_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Training results saved to {results_file}")
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze training performance"""
        logger.info("Analyzing training performance")
        
        # Initialize performance analysis framework
        analyzer = PerformanceAnalysisFramework(self.config.results_dir)
        
        # Analyze model complexity
        complexity_report = analyzer.analyze_model_complexity(self.model, self.config.input_shape)
        
        # Run benchmarks
        benchmark_results = analyzer.run_comprehensive_benchmark(
            self.config.model_factory,
            self.config.optimizer_factory,
            self.config.input_shape,
            device=str(self.device)
        )
        
        # Generate optimization report
        optimization_report = analyzer.generate_optimization_report()
        
        return {
            'complexity_report': complexity_report,
            'benchmark_results': benchmark_results,
            'optimization_report': optimization_report
        }
    
    def get_recommendations(self) -> List[str]:
        """Get optimization recommendations"""
        recommendations = []
        
        # Model complexity recommendations
        complexity_analyzer = ModelComplexityAnalyzer(self.model)
        complexity_report = complexity_analyzer.get_complexity_report(self.config.input_shape)
        
        total_params = complexity_report['parameters']['total_parameters']
        model_size_mb = complexity_report['model_size_mb']
        
        if total_params > 10_000_000:
            recommendations.append("Large model detected - consider model pruning or quantization")
        
        if model_size_mb > 100:
            recommendations.append("Model is large - consider gradient checkpointing to save memory")
        
        # Training optimization recommendations
        if not self.config.use_mixed_precision and torch.cuda.is_available():
            recommendations.append("Enable mixed precision training for better performance")
        
        if not self.config.use_gradient_accumulation and self.config.target_batch_size > self.config.batch_size:
            recommendations.append("Enable gradient accumulation for effective larger batch sizes")
        
        # Memory recommendations
        if self.config.available_memory_gb < 8:
            recommendations.append("Limited memory available - consider reducing batch size or model size")
        
        return recommendations
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up training system")
        
        # Clean up components
        if 'progress_monitor' in self.components:
            self.components['progress_monitor'].cleanup()
        
        if 'checkpoint_manager' in self.components:
            self.components['checkpoint_manager'].cleanup()
        
        if 'distributed_training' in self.components:
            self.components['distributed_training'].cleanup()
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Cleanup completed")


# Convenience functions for easy usage
def create_optimized_training_config(
    model_factory: Callable,
    optimizer_factory: Callable,
    data_source: str,
    input_shape: Tuple[int, ...],
    **kwargs
) -> OptimizedTrainingConfig:
    """Create optimized training configuration with sensible defaults"""
    
    config = OptimizedTrainingConfig(
        model_factory=model_factory,
        optimizer_factory=optimizer_factory,
        data_source=data_source,
        input_shape=input_shape,
        **kwargs
    )
    
    return config


def run_optimized_training(
    model_factory: Callable,
    optimizer_factory: Callable,
    data_source: str,
    input_shape: Tuple[int, ...],
    **kwargs
) -> Dict[str, Any]:
    """Run optimized training with all components"""
    
    # Create configuration
    config = create_optimized_training_config(
        model_factory,
        optimizer_factory,
        data_source,
        input_shape,
        **kwargs
    )
    
    # Initialize training system
    training_system = OptimizedTrainingSystem(config)
    
    try:
        # Run training
        results = training_system.train()
        
        # Analyze performance
        performance_analysis = training_system.analyze_performance()
        
        # Get recommendations
        recommendations = training_system.get_recommendations()
        
        # Compile final results
        final_results = {
            'training_results': results,
            'performance_analysis': performance_analysis,
            'recommendations': recommendations
        }
        
        return final_results
        
    finally:
        # Clean up
        training_system.cleanup()


# Example usage
def example_usage():
    """Example of how to use the optimized training system"""
    
    # Define model factory
    def create_model():
        return nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    # Define optimizer factory
    def create_optimizer(params):
        return optim.Adam(params, lr=0.001)
    
    # Run optimized training
    results = run_optimized_training(
        model_factory=create_model,
        optimizer_factory=create_optimizer,
        data_source="dummy_data.csv",  # Replace with actual data source
        input_shape=(100,),
        epochs=50,
        batch_size=32,
        use_incremental_learning=True,
        use_gradient_accumulation=True,
        use_mixed_precision=True,
        available_memory_gb=8.0
    )
    
    print("Training completed successfully!")
    print(f"Final loss: {results['training_results']['final_loss']:.6f}")
    print(f"Total epochs: {results['training_results']['total_epochs']}")
    print(f"Converged: {results['training_results']['converged']}")
    
    print("\nRecommendations:")
    for rec in results['recommendations']:
        print(f"- {rec}")
    
    return results


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run example
    results = example_usage()
    print("Example completed successfully!")