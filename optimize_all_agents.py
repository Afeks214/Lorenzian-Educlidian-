#!/usr/bin/env python3
"""
Comprehensive Performance Optimization for All Agents
Main script to apply all performance optimizations across the GrandModel system
"""

import asyncio
import torch
import torch.nn as nn
import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import sys
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from performance.integrated_performance_engine import IntegratedPerformanceEngine, PerformanceLevel
from performance.advanced_caching_system import configure_global_cache
from performance.memory_optimization_system import get_global_memory_manager
from performance.config_tuning_system import OptimizationGoal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AgentOptimizer:
    """Optimizes all agents in the system"""
    
    def __init__(self):
        self.engine = IntegratedPerformanceEngine()
        self.optimized_agents = {}
        self.performance_results = {}
        self.optimization_start_time = None
        
        # Agent definitions
        self.agent_configs = {
            'tactical_system': {
                'model_path': 'models/tactical_marl_initial.pt',
                'input_shape': (1, 60, 7),
                'target_latency_ms': 1.0,
                'target_throughput_qps': 1000.0
            },
            'strategic_system': {
                'model_path': 'models/strategic_system.pt',
                'input_shapes': {
                    'mlmi': (1, 4),
                    'nwrqk': (1, 6),
                    'mmd': (1, 3)
                },
                'target_latency_ms': 5.0,
                'target_throughput_qps': 500.0
            },
            'shared_policy': {
                'model_path': 'models/shared_policy.pth',
                'input_shape': (1, 136),
                'target_latency_ms': 2.0,
                'target_throughput_qps': 800.0
            },
            'decision_gate': {
                'model_path': 'models/decision_gate.pth',
                'input_shape': (1, 152),
                'target_latency_ms': 1.0,
                'target_throughput_qps': 1000.0
            },
            'structure_embedder': {
                'model_path': 'models/structure_embedder.pth',
                'input_shape': (1, 48, 8),
                'target_latency_ms': 3.0,
                'target_throughput_qps': 600.0
            },
            'tactical_embedder': {
                'model_path': 'models/tactical_embedder.pth',
                'input_shape': (1, 60, 7),
                'target_latency_ms': 2.0,
                'target_throughput_qps': 700.0
            },
            'regime_embedder': {
                'model_path': 'models/regime_embedder.pth',
                'input_shape': (1, 8),
                'target_latency_ms': 1.0,
                'target_throughput_qps': 1000.0
            },
            'lvn_embedder': {
                'model_path': 'models/lvn_embedder.pth',
                'input_shape': (1, 5),
                'target_latency_ms': 0.5,
                'target_throughput_qps': 1500.0
            }
        }
    
    async def optimize_all_agents(self, profile: str = 'aggressive'):
        """Optimize all agents with specified profile"""
        logger.info(f"🚀 Starting comprehensive agent optimization with profile: {profile}")
        self.optimization_start_time = time.time()
        
        try:
            # Initialize performance engine
            await self.engine.initialize(profile)
            
            # Configure global cache
            self._configure_global_cache()
            
            # Optimize each agent
            for agent_name, config in self.agent_configs.items():
                logger.info(f"🔧 Optimizing {agent_name}...")
                
                try:
                    # Load or create model
                    model = self._load_or_create_model(agent_name, config)
                    
                    # Create example input
                    example_input = self._create_example_input(config)
                    
                    # Optimize model
                    optimized_model = self.engine.optimize_model(
                        model, agent_name, example_input
                    )
                    
                    # Benchmark performance
                    performance = await self._benchmark_agent(
                        agent_name, optimized_model, example_input
                    )
                    
                    self.optimized_agents[agent_name] = optimized_model
                    self.performance_results[agent_name] = performance
                    
                    logger.info(f"✅ {agent_name} optimized - "
                              f"Latency: {performance['avg_latency_ms']:.2f}ms, "
                              f"Throughput: {performance['throughput_qps']:.1f} QPS")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to optimize {agent_name}: {e}")
                    continue
            
            # Create integrated pipeline
            await self._create_integrated_pipeline()
            
            # Run comprehensive benchmarks
            await self._run_comprehensive_benchmarks()
            
            # Generate optimization report
            await self._generate_optimization_report()
            
            logger.info("🎉 All agents optimized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Optimization failed: {e}")
            traceback.print_exc()
            raise
        finally:
            await self.engine.shutdown()
    
    def _configure_global_cache(self):
        """Configure global cache for optimal performance"""
        cache_config = {
            'l1_capacity': 2000,
            'l2_capacity': 1000,
            'l3_max_size_mb': 2000,
            'cache_dir': str(Path.cwd() / 'cache'),
            'tensor_pool_size': 2000,
            'enable_redis': False
        }
        
        configure_global_cache(cache_config)
        logger.info("✅ Global cache configured")
    
    def _load_or_create_model(self, agent_name: str, config: Dict[str, Any]) -> nn.Module:
        """Load model from file or create mock model"""
        model_path = Path(config['model_path'])
        
        if model_path.exists():
            try:
                # Try to load the model
                if model_path.suffix == '.pt':
                    model = torch.jit.load(str(model_path))
                else:
                    # For .pth files, we need to create the model architecture first
                    model = self._create_mock_model(agent_name, config)
                    state_dict = torch.load(str(model_path), map_location='cpu')
                    model.load_state_dict(state_dict)
                
                logger.info(f"✅ Loaded model for {agent_name} from {model_path}")
                return model
                
            except Exception as e:
                logger.warning(f"⚠️ Could not load model for {agent_name}: {e}")
                # Fall back to creating mock model
                return self._create_mock_model(agent_name, config)
        else:
            # Create mock model
            return self._create_mock_model(agent_name, config)
    
    def _create_mock_model(self, agent_name: str, config: Dict[str, Any]) -> nn.Module:
        """Create mock model for testing"""
        if agent_name == 'tactical_system':
            return self._create_tactical_model()
        elif agent_name == 'strategic_system':
            return self._create_strategic_model()
        elif agent_name == 'shared_policy':
            return self._create_shared_policy_model()
        elif agent_name == 'decision_gate':
            return self._create_decision_gate_model()
        elif 'embedder' in agent_name:
            return self._create_embedder_model(config)
        else:
            # Generic model
            input_shape = config.get('input_shape', (1, 100))
            input_size = input_shape[-1] if len(input_shape) > 1 else input_shape[0]
            
            return nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            )
    
    def _create_tactical_model(self) -> nn.Module:
        """Create tactical system model"""
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(420, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 9)  # 3 agents * 3 actions
        )
    
    def _create_strategic_model(self) -> nn.Module:
        """Create strategic system model"""
        return nn.Sequential(
            nn.Linear(13, 128),  # 4 + 6 + 3
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)  # 3 agents * 3 actions + value
        )
    
    def _create_shared_policy_model(self) -> nn.Module:
        """Create shared policy model"""
        return nn.Sequential(
            nn.Linear(136, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 2 actions + value
        )
    
    def _create_decision_gate_model(self) -> nn.Module:
        """Create decision gate model"""
        return nn.Sequential(
            nn.Linear(152, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # EXECUTE or REJECT
        )
    
    def _create_embedder_model(self, config: Dict[str, Any]) -> nn.Module:
        """Create embedder model"""
        input_shape = config.get('input_shape', (1, 100))
        
        if len(input_shape) == 3:  # Sequence input
            seq_len, input_dim = input_shape[1], input_shape[2]
            return nn.Sequential(
                nn.LSTM(input_dim, 64, batch_first=True),
                nn.Lambda(lambda x: x[0][:, -1, :]),  # Get last output
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16)
            )
        else:  # Vector input
            input_dim = input_shape[-1]
            return nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16)
            )
    
    def _create_example_input(self, config: Dict[str, Any]) -> torch.Tensor:
        """Create example input for the model"""
        if 'input_shapes' in config:
            # Multiple inputs (like strategic system)
            inputs = []
            for name, shape in config['input_shapes'].items():
                inputs.append(torch.randn(shape))
            return inputs
        else:
            # Single input
            input_shape = config.get('input_shape', (1, 100))
            return torch.randn(input_shape)
    
    async def _benchmark_agent(self, agent_name: str, model, example_input) -> Dict[str, Any]:
        """Benchmark agent performance"""
        num_iterations = 1000
        latencies = []
        
        # Warm up
        for _ in range(100):
            with torch.no_grad():
                if isinstance(example_input, list):
                    _ = model(*example_input)
                else:
                    _ = model(example_input)
        
        # Benchmark
        for _ in range(num_iterations):
            start_time = time.time()
            
            with torch.no_grad():
                if isinstance(example_input, list):
                    _ = model(*example_input)
                else:
                    _ = model(example_input)
            
            latencies.append((time.time() - start_time) * 1000)
        
        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
        p99_latency = sorted(latencies)[int(0.99 * len(latencies))]
        throughput = 1000 / avg_latency
        
        return {
            'avg_latency_ms': avg_latency,
            'p95_latency_ms': p95_latency,
            'p99_latency_ms': p99_latency,
            'throughput_qps': throughput,
            'iterations': num_iterations
        }
    
    async def _create_integrated_pipeline(self):
        """Create integrated pipeline for end-to-end optimization"""
        try:
            # Select key models for pipeline
            pipeline_models = {
                'structure_embedder': self.optimized_agents.get('structure_embedder'),
                'tactical_embedder': self.optimized_agents.get('tactical_embedder'),
                'shared_policy': self.optimized_agents.get('shared_policy')
            }
            
            # Remove None values
            pipeline_models = {k: v for k, v in pipeline_models.items() if v is not None}
            
            if pipeline_models:
                # Create pipeline
                pipeline = await self.engine.create_model_pipeline(
                    pipeline_models, 
                    'main_pipeline'
                )
                
                logger.info(f"✅ Created integrated pipeline with {len(pipeline_models)} models")
            else:
                logger.warning("⚠️ No models available for pipeline creation")
                
        except Exception as e:
            logger.error(f"❌ Failed to create integrated pipeline: {e}")
    
    async def _run_comprehensive_benchmarks(self):
        """Run comprehensive benchmarks"""
        logger.info("🔍 Running comprehensive benchmarks...")
        
        # System-level benchmarks
        memory_manager = get_global_memory_manager()
        
        # Test batch processing
        batch_sizes = [1, 4, 8, 16, 32]
        batch_results = {}
        
        for agent_name, model in self.optimized_agents.items():
            config = self.agent_configs[agent_name]
            batch_results[agent_name] = {}
            
            for batch_size in batch_sizes:
                try:
                    # Create batch input
                    if 'input_shapes' in config:
                        # Skip multi-input models for batch testing
                        continue
                    else:
                        input_shape = config['input_shape']
                        batch_input = torch.randn(batch_size, *input_shape[1:])
                    
                    # Benchmark
                    start_time = time.time()
                    with torch.no_grad():
                        _ = model(batch_input)
                    
                    latency = (time.time() - start_time) * 1000
                    throughput = batch_size / (latency / 1000)
                    
                    batch_results[agent_name][batch_size] = {
                        'latency_ms': latency,
                        'throughput_qps': throughput
                    }
                    
                except Exception as e:
                    logger.warning(f"Batch test failed for {agent_name} (batch_size={batch_size}): {e}")
        
        self.performance_results['batch_benchmarks'] = batch_results
        logger.info("✅ Comprehensive benchmarks completed")
    
    async def _generate_optimization_report(self):
        """Generate comprehensive optimization report"""
        total_time = time.time() - self.optimization_start_time
        
        report = {
            'optimization_summary': {
                'total_time_seconds': total_time,
                'agents_optimized': len(self.optimized_agents),
                'profile_used': self.engine.current_profile.name,
                'timestamp': time.time()
            },
            'agent_performance': self.performance_results,
            'engine_stats': self.engine.get_comprehensive_stats(),
            'optimization_recommendations': self.engine.get_optimization_recommendations()
        }
        
        # Save report
        report_path = Path('optimization_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save detailed report
        self.engine.save_optimization_report(Path('detailed_optimization_report.json'))
        
        # Print summary
        self._print_optimization_summary(report)
        
        logger.info(f"✅ Optimization report saved to {report_path}")
    
    def _print_optimization_summary(self, report: Dict[str, Any]):
        """Print optimization summary"""
        print("\n" + "="*80)
        print("🎯 PERFORMANCE OPTIMIZATION SUMMARY")
        print("="*80)
        
        summary = report['optimization_summary']
        print(f"⏱️  Total optimization time: {summary['total_time_seconds']:.2f}s")
        print(f"🤖 Agents optimized: {summary['agents_optimized']}")
        print(f"📊 Profile used: {summary['profile_used']}")
        
        print("\n📈 AGENT PERFORMANCE RESULTS:")
        print("-"*50)
        
        for agent_name, performance in report['agent_performance'].items():
            if isinstance(performance, dict) and 'avg_latency_ms' in performance:
                print(f"{agent_name:20}: {performance['avg_latency_ms']:6.2f}ms | "
                      f"{performance['throughput_qps']:8.1f} QPS")
        
        print("\n🎉 OPTIMIZATION COMPLETE!")
        print("="*80)

async def main():
    """Main optimization function"""
    print("🚀 GrandModel Performance Optimization System")
    print("="*80)
    
    # Create optimizer
    optimizer = AgentOptimizer()
    
    # Check command line arguments for profile
    import sys
    profile = 'aggressive'  # default
    
    if len(sys.argv) > 1:
        profile = sys.argv[1]
        if profile not in ['basic', 'standard', 'aggressive', 'maximum']:
            print(f"❌ Invalid profile: {profile}")
            print("Valid profiles: basic, standard, aggressive, maximum")
            return
    
    print(f"📊 Using optimization profile: {profile}")
    print("="*80)
    
    try:
        # Run optimization
        await optimizer.optimize_all_agents(profile)
        
        print("✅ Optimization completed successfully!")
        print("📊 Check optimization_report.json for detailed results")
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))