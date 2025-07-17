#!/usr/bin/env python3
"""
Memory Optimization Integration Script for GrandModel

This script demonstrates how to integrate and use the comprehensive memory 
optimization system with the GrandModel agent system.

Usage:
    python run_memory_optimization.py [--config config.json] [--report output.json]

Author: Claude Code Assistant
Date: July 2025
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Import optimization systems
from memory_optimization_system import (
    MemoryOptimizationSystem,
    MemoryOptimizationConfig,
    create_memory_optimized_config,
    quick_memory_optimization
)
from training_memory_optimizer import (
    TrainingMemoryOptimizer,
    TrainingMemoryConfig,
    create_memory_efficient_config,
    memory_efficient_training
)

# Import existing model components
try:
    from models.tactical_architectures import TacticalMARLSystem
except ImportError:
    TacticalMARLSystem = None
    logging.warning("TacticalMARLSystem not found, using mock model")

import torch
import torch.nn as nn
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockTacticalMARLSystem(nn.Module):
    """Mock model for testing when real model is not available"""
    
    def __init__(self):
        super().__init__()
        self.agents = nn.ModuleDict({
            'fvg': nn.Sequential(
                nn.Linear(420, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 3)
            ),
            'momentum': nn.Sequential(
                nn.Linear(420, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 3)
            ),
            'entry': nn.Sequential(
                nn.Linear(420, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 3)
            )
        })
        self.critic = nn.Sequential(
            nn.Linear(420 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        
        agent_outputs = {}
        for agent_name, agent in self.agents.items():
            logits = agent(x_flat)
            agent_outputs[agent_name] = {
                'action': torch.argmax(logits, dim=-1),
                'action_probs': torch.softmax(logits, dim=-1),
                'log_prob': torch.log_softmax(logits, dim=-1).gather(1, torch.argmax(logits, dim=-1).unsqueeze(-1)).squeeze(-1),
                'logits': logits,
                'temperature': 1.0
            }
        
        # Critic forward
        combined_state = x_flat.repeat(1, 3)
        critic_output = {
            'value': self.critic(combined_state).squeeze(-1)
        }
        
        return {
            'agents': agent_outputs,
            'critic': critic_output
        }
    
    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'total_parameters': total_params,
            'model_size_mb': total_params * 4 / (1024 ** 2),
            'architecture': 'MockTacticalMARLSystem'
        }


class MockDataset:
    """Mock dataset for testing"""
    
    def __init__(self, num_samples: int = 1000):
        self.num_samples = num_samples
        self.data = torch.randn(num_samples, 60, 7)  # (batch, sequence, features)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


def create_default_configs() -> tuple[MemoryOptimizationConfig, TrainingMemoryConfig]:
    """Create default configurations for memory optimization"""
    
    # Memory optimization config
    memory_config = create_memory_optimized_config(
        max_memory_usage_gb=8.0,
        memory_warning_threshold=0.75,
        memory_critical_threshold=0.90,
        
        # Garbage collection
        gc_threshold_0=700,
        gc_threshold_1=10,
        gc_threshold_2=10,
        auto_gc_interval=60.0,
        
        # Buffer management
        buffer_size_limit_mb=1024,
        enable_buffer_compression=True,
        
        # Training optimization
        batch_size_auto_scaling=True,
        gradient_accumulation_steps=4,
        
        # Monitoring
        monitoring_interval=5.0,
        enable_memory_profiling=True
    )
    
    # Training memory config
    training_config = create_memory_efficient_config(
        initial_batch_size=32,
        max_batch_size=256,
        min_batch_size=8,
        memory_threshold=0.85,
        
        # Mixed precision
        enable_mixed_precision=True,
        
        # Gradient checkpointing
        enable_gradient_checkpointing=True,
        checkpoint_segments=4,
        
        # Data loading
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        
        # Distributed training
        use_distributed=False,
        
        # Advanced optimizations
        enable_activation_checkpointing=True,
        enable_parameter_offloading=False
    )
    
    return memory_config, training_config


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        return {}


def demonstrate_memory_optimization():
    """Demonstrate memory optimization system"""
    logger.info("=== Memory Optimization Demonstration ===")
    
    # Get default configs
    memory_config, training_config = create_default_configs()
    
    # Create model
    if TacticalMARLSystem:
        model = TacticalMARLSystem()
    else:
        model = MockTacticalMARLSystem()
    
    logger.info(f"Model info: {model.get_model_info()}")
    
    # Create sample data
    sample_input = torch.randn(32, 60, 7)  # (batch, sequence, features)
    
    # Initialize memory optimization system
    with MemoryOptimizationSystem(memory_config) as optimizer:
        logger.info("Memory optimization system initialized")
        
        # Get initial memory report
        initial_report = optimizer.get_comprehensive_report()
        logger.info(f"Initial memory usage: {initial_report['current_memory_usage']['memory_percent']:.1%}")
        
        # Optimize model architecture
        logger.info("Optimizing model architecture...")
        model_optimization_results = optimizer.optimize_model_architecture(model)
        logger.info(f"Model optimization results: {model_optimization_results}")
        
        # Optimize training pipeline
        logger.info("Optimizing training pipeline...")
        training_optimization_results = optimizer.optimize_training_pipeline(model, sample_input)
        logger.info(f"Training optimization results: {training_optimization_results}")
        
        # Run system optimization
        logger.info("Running system memory optimization...")
        system_optimization_results = optimizer.optimize_memory_usage()
        logger.info(f"System optimization results: {system_optimization_results}")
        
        # Generate recommendations
        logger.info("Generating optimization recommendations...")
        recommendations = optimizer.generate_optimization_recommendations()
        logger.info(f"Found {len(recommendations)} optimization recommendations")
        
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"Recommendation {i}: [{rec['priority']}] {rec['recommendation']}")
        
        # Get final report
        final_report = optimizer.get_comprehensive_report()
        logger.info(f"Final memory usage: {final_report['current_memory_usage']['memory_percent']:.1%}")
        
        # Export report
        timestamp = int(time.time())
        report_path = f"memory_optimization_report_{timestamp}.json"
        optimizer.export_optimization_report(report_path)
        logger.info(f"Report exported to {report_path}")
        
        return final_report


def demonstrate_training_optimization():
    """Demonstrate training memory optimization"""
    logger.info("=== Training Memory Optimization Demonstration ===")
    
    # Get configs
    memory_config, training_config = create_default_configs()
    
    # Create model and dataset
    if TacticalMARLSystem:
        model = TacticalMARLSystem()
    else:
        model = MockTacticalMARLSystem()
    
    dataset = MockDataset(num_samples=1000)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Loss function
    def loss_fn(output, target):
        # Simple MSE loss for demonstration
        agent_losses = []
        for agent_name, agent_output in output['agents'].items():
            # Target is the same as input for this demo
            target_flat = target.view(target.size(0), -1)
            logits = agent_output['logits']
            loss = torch.nn.functional.mse_loss(logits, target_flat[:, :logits.size(1)])
            agent_losses.append(loss)
        return sum(agent_losses) / len(agent_losses)
    
    # Initialize training memory optimizer
    with TrainingMemoryOptimizer(training_config) as training_optimizer:
        logger.info("Training memory optimizer initialized")
        
        # Run optimized training loop (just a few steps for demo)
        logger.info("Running optimized training...")
        
        try:
            training_results = training_optimizer.optimize_training_loop(
                model=model,
                dataset=dataset,
                optimizer=optimizer,
                num_epochs=2,  # Just 2 epochs for demo
                loss_fn=loss_fn
            )
            
            logger.info(f"Training completed successfully")
            logger.info(f"Final memory usage: {training_results['final_memory_usage']:.2f} GB")
            logger.info(f"Training metrics: {training_results['training_metrics']}")
            
            # Export training report
            timestamp = int(time.time())
            training_report_path = f"training_optimization_report_{timestamp}.json"
            training_optimizer.export_report(training_report_path)
            logger.info(f"Training report exported to {training_report_path}")
            
            return training_results
            
        except Exception as e:
            logger.error(f"Training optimization failed: {e}")
            return None


def run_quick_optimization():
    """Run quick memory optimization"""
    logger.info("=== Quick Memory Optimization ===")
    
    results = quick_memory_optimization()
    logger.info(f"Quick optimization results: {results}")
    
    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='GrandModel Memory Optimization')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--report', type=str, help='Output report file path')
    parser.add_argument('--mode', type=str, choices=['full', 'training', 'quick'], 
                       default='full', help='Optimization mode')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting GrandModel Memory Optimization")
    logger.info(f"Mode: {args.mode}")
    
    # Load custom config if provided
    custom_config = {}
    if args.config:
        custom_config = load_config_from_file(args.config)
        logger.info(f"Loaded custom configuration from {args.config}")
    
    # Run optimization based on mode
    results = None
    
    if args.mode == 'quick':
        results = run_quick_optimization()
    
    elif args.mode == 'training':
        results = demonstrate_training_optimization()
    
    elif args.mode == 'full':
        # Run both memory and training optimization
        memory_results = demonstrate_memory_optimization()
        training_results = demonstrate_training_optimization()
        
        results = {
            'memory_optimization': memory_results,
            'training_optimization': training_results,
            'timestamp': time.time()
        }
    
    # Export results if requested
    if args.report and results:
        with open(args.report, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results exported to {args.report}")
    
    logger.info("Memory optimization completed successfully!")
    
    # Print summary
    if results:
        logger.info("\n=== OPTIMIZATION SUMMARY ===")
        if args.mode == 'full' and isinstance(results, dict):
            if 'memory_optimization' in results:
                mem_usage = results['memory_optimization']['current_memory_usage']['memory_percent']
                logger.info(f"Memory Usage: {mem_usage:.1%}")
            if 'training_optimization' in results:
                training_mem = results['training_optimization']['final_memory_usage']
                logger.info(f"Training Memory: {training_mem:.2f} GB")
        logger.info("All optimizations completed successfully!")


if __name__ == "__main__":
    main()
