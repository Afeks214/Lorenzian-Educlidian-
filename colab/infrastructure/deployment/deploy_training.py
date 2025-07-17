#!/usr/bin/env python3
"""
Comprehensive Training Deployment Script
Orchestrates the entire training infrastructure with all optimizations
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import torch
import torch.nn as nn
from dataclasses import dataclass

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import our infrastructure components
from infrastructure.monitoring.training_monitor import TrainingMonitor, TrainingMetrics
from infrastructure.monitoring.logging_config import setup_training_logging
from infrastructure.optimization.gpu_optimizer import create_gpu_optimizer
from infrastructure.optimization.memory_optimizer import create_memory_optimizer
from infrastructure.backup.backup_system import create_backup_system
from infrastructure.testing.test_pipeline import create_test_pipeline

@dataclass
class DeploymentConfig:
    """Training deployment configuration"""
    model_name: str = "tactical_mappo"
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    checkpoint_interval: int = 10
    validation_interval: int = 5
    enable_mixed_precision: bool = True
    enable_gpu_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_monitoring: bool = True
    enable_backups: bool = True
    enable_testing: bool = True
    max_memory_percent: float = 85.0
    log_level: str = "INFO"
    output_dir: str = "/home/QuantNova/GrandModel/colab/exports"
    data_path: str = "/home/QuantNova/GrandModel/colab/data"

class TrainingDeployment:
    """Complete training deployment orchestrator"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.start_time = time.time()
        self.deployment_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Setup logging first
        self.setup_logging()
        
        # Initialize components
        self.monitor = None
        self.gpu_optimizer = None
        self.memory_optimizer = None
        self.backup_system = None
        self.test_pipeline = None
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scaler = None
        self.training_state = {
            'epoch': 0,
            'step': 0,
            'best_loss': float('inf'),
            'training_time': 0.0
        }
        
        self.logger.info(f"Training deployment initialized: {self.deployment_id}")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_level = getattr(logging, self.config.log_level.upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    f"/home/QuantNova/GrandModel/colab/logs/training/deployment_{self.deployment_id}.log"
                )
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup training logger
        setup_training_logging()
    
    def initialize_infrastructure(self):
        """Initialize all infrastructure components"""
        self.logger.info("Initializing training infrastructure...")
        
        # GPU Optimization
        if self.config.enable_gpu_optimization and torch.cuda.is_available():
            self.gpu_optimizer = create_gpu_optimizer(
                mixed_precision=self.config.enable_mixed_precision,
                compile_model=True
            )
            self.logger.info("GPU optimization enabled")
        
        # Memory Optimization
        if self.config.enable_memory_optimization:
            self.memory_optimizer = create_memory_optimizer(
                max_memory_percent=self.config.max_memory_percent
            )
            self.logger.info("Memory optimization enabled")
        
        # Monitoring
        if self.config.enable_monitoring:
            self.monitor = TrainingMonitor()
            self.monitor.start_monitoring()
            self.logger.info("Performance monitoring enabled")
        
        # Backup System
        if self.config.enable_backups:
            self.backup_system = create_backup_system()
            self.logger.info("Backup system enabled")
        
        # Testing Pipeline
        if self.config.enable_testing:
            self.test_pipeline = create_test_pipeline()
            self.logger.info("Testing pipeline enabled")
    
    def create_model(self) -> nn.Module:
        """Create and optimize model"""
        self.logger.info(f"Creating model: {self.config.model_name}")
        
        # Create a sample model (replace with your actual model)
        if self.config.model_name == "tactical_mappo":
            model = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        else:
            # Default model
            model = nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
        
        # Optimize model
        if self.gpu_optimizer:
            model = self.gpu_optimizer.optimize_model(model, self.config.model_name)
        
        if self.memory_optimizer:
            model = self.memory_optimizer.optimize_for_training(model)
        
        self.model = model
        return model
    
    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer"""
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-4
        )
        
        self.optimizer = optimizer
        return optimizer
    
    def create_data_loader(self):
        """Create data loader (mock implementation)"""
        # This is a mock implementation - replace with your actual data loading
        from torch.utils.data import DataLoader, TensorDataset
        
        # Generate mock data
        X = torch.randn(1000, 256)
        y = torch.randn(1000, 1)
        
        dataset = TensorDataset(X, y)
        
        # Create optimized data loader
        if self.memory_optimizer:
            dataloader = self.memory_optimizer.create_memory_efficient_dataloader(
                dataset, 
                batch_size=self.config.batch_size,
                shuffle=True
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
        
        return dataloader
    
    def run_pre_training_tests(self):
        """Run pre-training tests"""
        if not self.test_pipeline:
            return
        
        self.logger.info("Running pre-training tests...")
        
        # Create basic tests
        self.test_pipeline.create_default_tests()
        
        # Run smoke tests
        smoke_results = self.test_pipeline.run_test_suite(
            next(suite for suite in self.test_pipeline.test_suites if suite.name == "smoke_tests")
        )
        
        # Check if tests passed
        failed_tests = [r for r in smoke_results if r.status.value in ['failed', 'error']]
        if failed_tests:
            self.logger.error(f"Pre-training tests failed: {[t.test_name for t in failed_tests]}")
            raise RuntimeError("Pre-training tests failed")
        
        self.logger.info("Pre-training tests passed")
    
    def training_loop(self):
        """Main training loop"""
        self.logger.info("Starting training loop...")
        
        # Create components
        model = self.create_model()
        optimizer = self.create_optimizer()
        dataloader = self.create_data_loader()
        
        # Create scaler for mixed precision
        if self.gpu_optimizer:
            self.scaler = self.gpu_optimizer.create_scaler()
        
        criterion = nn.MSELoss()
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            
            model.train()
            
            for step, (batch_x, batch_y) in enumerate(dataloader):
                step_start_time = time.time()
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    batch_x = batch_x.cuda(non_blocking=True)
                    batch_y = batch_y.cuda(non_blocking=True)
                
                # Forward pass with mixed precision
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                    
                    # Backward pass
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                
                optimizer.zero_grad()
                
                # Update tracking
                epoch_loss += loss.item()
                self.training_state['step'] += 1
                
                # Log training metrics
                if self.monitor:
                    training_metrics = TrainingMetrics(
                        timestamp=time.time(),
                        epoch=epoch,
                        step=self.training_state['step'],
                        loss=loss.item(),
                        learning_rate=optimizer.param_groups[0]['lr'],
                        batch_size=self.config.batch_size,
                        training_time_per_step=time.time() - step_start_time
                    )
                    self.monitor.log_training_metrics(training_metrics)
                
                # Checkpoint saving
                if self.training_state['step'] % self.config.checkpoint_interval == 0:
                    self.save_checkpoint(epoch, self.training_state['step'], loss.item())
            
            # Epoch completed
            epoch_duration = time.time() - epoch_start_time
            avg_loss = epoch_loss / len(dataloader)
            
            self.training_state['epoch'] = epoch
            self.training_state['training_time'] += epoch_duration
            
            if avg_loss < self.training_state['best_loss']:
                self.training_state['best_loss'] = avg_loss
                self.save_best_model(epoch, avg_loss)
            
            self.logger.info(f"Epoch {epoch:03d} completed - Loss: {avg_loss:.6f}, Time: {epoch_duration:.2f}s")
            
            # Memory optimization
            if self.memory_optimizer and epoch % 10 == 0:
                self.memory_optimizer.optimize_memory()
            
            # Validation
            if epoch % self.config.validation_interval == 0:
                self.run_validation(model, criterion)
    
    def run_validation(self, model: nn.Module, criterion: nn.Module):
        """Run validation"""
        self.logger.info("Running validation...")
        
        # Create validation data (mock)
        val_x = torch.randn(100, 256)
        val_y = torch.randn(100, 1)
        
        if torch.cuda.is_available():
            val_x = val_x.cuda()
            val_y = val_y.cuda()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_x)
            val_loss = criterion(val_outputs, val_y)
        
        self.logger.info(f"Validation Loss: {val_loss.item():.6f}")
        
        # Log validation metrics
        if self.monitor:
            training_metrics = TrainingMetrics(
                timestamp=time.time(),
                epoch=self.training_state['epoch'],
                step=self.training_state['step'],
                loss=0.0,  # Not used for validation
                learning_rate=0.0,  # Not used for validation
                batch_size=self.config.batch_size,
                validation_loss=val_loss.item()
            )
            self.monitor.log_training_metrics(training_metrics)
    
    def save_checkpoint(self, epoch: int, step: int, loss: float):
        """Save training checkpoint"""
        if not self.backup_system:
            return
        
        try:
            checkpoint_path = self.backup_system.create_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                step,
                loss,
                training_time=self.training_state['training_time']
            )
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def save_best_model(self, epoch: int, loss: float):
        """Save best model"""
        output_dir = Path(self.config.output_dir) / self.deployment_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        best_model_path = output_dir / "best_model.pth"
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'training_time': self.training_state['training_time'],
            'config': self.config
        }, best_model_path)
        
        self.logger.info(f"Best model saved: {best_model_path}")
    
    def run_post_training_tests(self):
        """Run post-training tests"""
        if not self.test_pipeline:
            return
        
        self.logger.info("Running post-training tests...")
        
        # Run performance tests
        performance_results = self.test_pipeline.run_test_suite(
            next(suite for suite in self.test_pipeline.test_suites if suite.name == "performance_tests")
        )
        
        # Save test results
        self.test_pipeline.save_test_report(f"post_training_tests_{self.deployment_id}.json")
        
        self.logger.info("Post-training tests completed")
    
    def generate_deployment_report(self):
        """Generate comprehensive deployment report"""
        total_duration = time.time() - self.start_time
        
        report = {
            'deployment_id': self.deployment_id,
            'config': self.config.__dict__,
            'training_state': self.training_state,
            'total_duration': total_duration,
            'infrastructure': {
                'gpu_optimization': self.gpu_optimizer is not None,
                'memory_optimization': self.memory_optimizer is not None,
                'monitoring': self.monitor is not None,
                'backups': self.backup_system is not None,
                'testing': self.test_pipeline is not None
            },
            'system_info': {
                'cuda_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'python_version': sys.version,
                'pytorch_version': torch.__version__
            },
            'timestamp': time.time()
        }
        
        # Add monitoring data
        if self.monitor:
            report['performance_summary'] = self.monitor.get_performance_summary()
        
        # Add memory data
        if self.memory_optimizer:
            report['memory_recommendations'] = self.memory_optimizer.get_memory_recommendations()
        
        # Add backup data
        if self.backup_system:
            report['backup_status'] = self.backup_system.get_backup_status()
        
        return report
    
    def save_deployment_report(self):
        """Save deployment report"""
        report = self.generate_deployment_report()
        
        output_dir = Path(self.config.output_dir) / self.deployment_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / "deployment_report.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Deployment report saved: {report_path}")
        return report_path
    
    def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up resources...")
        
        if self.monitor:
            self.monitor.stop_monitoring()
            self.monitor.save_metrics(f"training_metrics_{self.deployment_id}.json")
        
        if self.memory_optimizer:
            self.memory_optimizer.cleanup()
        
        if self.gpu_optimizer:
            self.gpu_optimizer.cleanup()
        
        if self.backup_system:
            self.backup_system.cleanup()
        
        self.logger.info("Cleanup completed")
    
    def run(self):
        """Run complete training deployment"""
        try:
            self.logger.info(f"Starting training deployment: {self.deployment_id}")
            
            # Initialize infrastructure
            self.initialize_infrastructure()
            
            # Run pre-training tests
            self.run_pre_training_tests()
            
            # Run training
            self.training_loop()
            
            # Run post-training tests
            self.run_post_training_tests()
            
            # Generate final report
            self.save_deployment_report()
            
            self.logger.info(f"Training deployment completed successfully: {self.deployment_id}")
            
        except Exception as e:
            self.logger.error(f"Training deployment failed: {e}")
            
            # Emergency backup
            if self.backup_system and self.model and self.optimizer:
                self.backup_system.emergency_backup(
                    self.model, 
                    self.optimizer, 
                    self.training_state['epoch'],
                    self.training_state['step'],
                    self.training_state['best_loss']
                )
            
            raise
        
        finally:
            self.cleanup()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Deploy training infrastructure")
    parser.add_argument("--model-name", default="tactical_mappo", help="Model name")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--output-dir", default="/home/QuantNova/GrandModel/colab/exports", help="Output directory")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--disable-gpu", action="store_true", help="Disable GPU optimization")
    parser.add_argument("--disable-monitoring", action="store_true", help="Disable monitoring")
    parser.add_argument("--disable-backups", action="store_true", help="Disable backups")
    parser.add_argument("--disable-testing", action="store_true", help="Disable testing")
    
    args = parser.parse_args()
    
    # Create configuration
    config = DeploymentConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir,
        log_level=args.log_level,
        enable_gpu_optimization=not args.disable_gpu,
        enable_monitoring=not args.disable_monitoring,
        enable_backups=not args.disable_backups,
        enable_testing=not args.disable_testing
    )
    
    # Create and run deployment
    deployment = TrainingDeployment(config)
    deployment.run()

if __name__ == "__main__":
    main()