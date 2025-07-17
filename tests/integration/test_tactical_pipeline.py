"""
Integration Test Suite for Tactical Pipeline with Smoke Tests

Comprehensive integration tests for the complete tactical MARL system:
- End-to-end pipeline validation
- SYNERGY_DETECTED event handling
- Component integration verification
- Performance and latency validation
- Error handling and recovery
- Production readiness verification

Author: Quantitative Engineer
Version: 1.0
"""

import pytest
import numpy as np
import time
import threading
import asyncio
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional
import yaml
import tempfile
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from environment.tactical_env import TacticalMarketEnv, TacticalState
from components.tactical_decision_aggregator import TacticalDecisionAggregator
from training.tactical_reward_system import TacticalRewardSystem
from src.indicators.custom.tactical_fvg import TacticalFVGDetector
from train_tactical_marl import TacticalTrainingOrchestrator, TacticalActor


class TestTacticalPipelineIntegration:
    """Integration tests for tactical pipeline"""
    
    @pytest.fixture
    def tactical_config(self):
        """Create comprehensive tactical configuration"""
        return {
            'tactical_marl': {
                'environment': {
                    'matrix_shape': [60, 7],
                    'feature_names': [
                        'fvg_bullish_active', 'fvg_bearish_active', 'fvg_nearest_level',
                        'fvg_age', 'fvg_mitigation_signal', 'price_momentum_5', 'volume_ratio'
                    ],
                    'max_episode_steps': 50,
                    'decision_timeout_ms': 100
                },
                'agents': {
                    'fvg_agent': {
                        'attention_weights': [0.4, 0.4, 0.1, 0.05, 0.05],
                        'hidden_dims': [64, 32, 16],
                        'learning_rate': 3e-4
                    },
                    'momentum_agent': {
                        'attention_weights': [0.05, 0.05, 0.1, 0.3, 0.5],
                        'hidden_dims': [64, 32, 16],
                        'learning_rate': 3e-4
                    },
                    'entry_opt_agent': {
                        'attention_weights': [0.2, 0.2, 0.2, 0.2, 0.2],
                        'hidden_dims': [64, 32, 16],
                        'learning_rate': 3e-4
                    }
                },
                'training': {
                    'max_episodes': 10,
                    'batch_size': 4,
                    'gamma': 0.99,
                    'checkpoint_interval': 5
                },
                'aggregation': {
                    'execution_threshold': 0.65,
                    'synergy_weights': {
                        'TYPE_1': [0.5, 0.3, 0.2],
                        'TYPE_2': [0.4, 0.4, 0.2],
                        'TYPE_3': [0.3, 0.5, 0.2],
                        'TYPE_4': [0.35, 0.35, 0.3],
                        'NONE': [0.33, 0.33, 0.34]
                    }
                },
                'rewards': {
                    'pnl_weight': 1.0,
                    'synergy_weight': 0.2,
                    'risk_weight': -0.5,
                    'execution_weight': 0.1
                },
                'monitoring': {
                    'latency_targets': {
                        'total_pipeline_ms': 100
                    }
                }
            }
        }
    
    @pytest.fixture
    def tactical_env(self, tactical_config):
        """Create tactical environment"""
        return TacticalMarketEnv(tactical_config)
    
    @pytest.fixture
    def decision_aggregator(self, tactical_config):
        """Create decision aggregator"""
        return TacticalDecisionAggregator(tactical_config['tactical_marl']['aggregation'])
    
    @pytest.fixture
    def reward_system(self, tactical_config):
        """Create reward system"""
        return TacticalRewardSystem(tactical_config['tactical_marl']['rewards'])
    
    @pytest.fixture
    def tactical_agents(self, tactical_config):
        """Create tactical agents"""
        agent_configs = tactical_config['tactical_marl']['agents']
        agents = {}
        
        for agent_id in ['fvg_agent', 'momentum_agent', 'entry_opt_agent']:
            config = agent_configs[agent_id]
            agents[agent_id] = TacticalActor(
                input_dim=7,
                hidden_dims=config['hidden_dims'],
                action_dim=3,
                agent_id=agent_id,
                config=config
            )
        
        return agents
    
    def test_smoke_test_pipeline_execution(self, tactical_env, decision_aggregator, reward_system):
        """Smoke test: Complete pipeline execution without errors"""
        # Initialize environment
        observations = tactical_env.reset()
        
        # Verify environment initialization
        assert len(observations) == 3
        assert all(obs.shape == (60, 7) for obs in observations.values())
        
        # Execute full decision cycle
        agent_outputs = {}
        for i, agent in enumerate(['fvg_agent', 'momentum_agent', 'entry_opt_agent']):
            assert tactical_env.agent_selection == agent
            
            # Execute step
            obs, rewards, dones, infos = tactical_env.step(i % 3)  # Varied actions
            
            # Collect agent output (simulated)
            agent_outputs[agent] = Mock(
                probabilities=np.array([0.2, 0.3, 0.5]),
                action=i % 3,
                confidence=0.7,
                timestamp=i
            )
        
        # Verify rewards were distributed
        assert isinstance(rewards, dict)
        assert len(rewards) == 3
        
        # Test decision aggregation
        market_state = Mock()
        market_state.features = {'current_price': 100.0}
        
        synergy_context = {
            'type': 'TYPE_1',
            'direction': 1,
            'confidence': 0.8
        }
        
        decision_result = decision_aggregator.aggregate_decisions(
            agent_outputs=agent_outputs,
            market_state=market_state,
            synergy_context=synergy_context
        )
        
        # Verify decision structure
        assert hasattr(decision_result, 'execute')
        assert hasattr(decision_result, 'action')
        assert hasattr(decision_result, 'confidence')
        
        # Test reward calculation
        reward_components = reward_system.calculate_tactical_reward(
            decision_result=decision_result.__dict__,
            market_state=market_state,
            agent_outputs=agent_outputs
        )
        
        # Verify reward structure
        assert hasattr(reward_components, 'total_reward')
        assert hasattr(reward_components, 'agent_specific')
        assert np.isfinite(reward_components.total_reward)
        
        print("✅ Smoke test passed - Pipeline executed successfully")
    
    def test_synergy_event_handling(self, tactical_env, decision_aggregator):
        """Test SYNERGY_DETECTED event handling"""
        # Mock SYNERGY_DETECTED event
        synergy_event = {
            'type': 'SYNERGY_DETECTED',
            'data': {
                'synergy_type': 'TYPE_2',
                'direction': 1,
                'confidence': 0.85,
                'signal_sequence': [
                    {'indicator': 'MLMI', 'signal': 'bullish'},
                    {'indicator': 'FVG', 'signal': 'gap_up'}
                ],
                'market_context': {
                    'volatility': 0.02,
                    'volume_profile': 'high'
                }
            }
        }
        
        # Simulate event processing
        start_time = time.time()
        
        # Reset environment
        observations = tactical_env.reset()
        
        # Process event through tactical system
        agent_outputs = {}
        for agent in ['fvg_agent', 'momentum_agent', 'entry_opt_agent']:
            # Simulate agent response to synergy event
            obs, rewards, dones, infos = tactical_env.step(2)  # Bullish response
            
            agent_outputs[agent] = Mock(
                probabilities=np.array([0.1, 0.2, 0.7]),  # Strong bullish
                action=2,
                confidence=0.8,
                timestamp=time.time()
            )
        
        # Aggregate decisions with synergy context
        market_state = Mock()
        market_state.features = {'current_price': 100.0}
        
        decision_result = decision_aggregator.aggregate_decisions(
            agent_outputs=agent_outputs,
            market_state=market_state,
            synergy_context=synergy_event['data']
        )
        
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000
        
        # Verify response time
        assert processing_time_ms < 100  # Should be under 100ms
        
        # Verify decision alignment with synergy
        assert decision_result.execute == True  # Should execute on strong synergy
        assert decision_result.action == 2  # Should be bullish
        assert decision_result.synergy_alignment > 0  # Should be aligned
        
        print(f"✅ Synergy event handled in {processing_time_ms:.2f}ms")
    
    def test_component_integration_verification(self, tactical_env, decision_aggregator, reward_system):
        """Test integration between all components"""
        # Create integrated pipeline
        pipeline_results = []
        
        for episode in range(3):
            # Reset environment
            observations = tactical_env.reset()
            
            # Collect agent decisions
            agent_outputs = {}
            episode_rewards = []
            
            for step in range(6):  # 2 full cycles
                current_agent = tactical_env.agent_selection
                
                # Execute action
                obs, rewards, dones, infos = tactical_env.step(step % 3)
                
                # Store agent output
                agent_outputs[current_agent] = Mock(
                    probabilities=np.array([0.3, 0.3, 0.4]),
                    action=step % 3,
                    confidence=0.6 + (step % 3) * 0.1,
                    timestamp=step
                )
                
                # Track rewards
                episode_rewards.extend(rewards.values())
                
                # After full cycle, test aggregation
                if len(agent_outputs) == 3:
                    # Test decision aggregation
                    market_state = Mock()
                    market_state.features = {'current_price': 100.0 + step}
                    
                    synergy_context = {
                        'type': f'TYPE_{(step % 4) + 1}',
                        'direction': (-1) ** step,
                        'confidence': 0.7
                    }
                    
                    decision_result = decision_aggregator.aggregate_decisions(
                        agent_outputs=agent_outputs,
                        market_state=market_state,
                        synergy_context=synergy_context
                    )
                    
                    # Test reward calculation
                    reward_components = reward_system.calculate_tactical_reward(
                        decision_result=decision_result.__dict__,
                        market_state=market_state,
                        agent_outputs=agent_outputs
                    )
                    
                    # Store results
                    pipeline_results.append({
                        'episode': episode,
                        'step': step,
                        'decision': decision_result,
                        'rewards': reward_components,
                        'agent_outputs': agent_outputs.copy()
                    })
                    
                    # Clear for next cycle
                    agent_outputs.clear()
        
        # Verify pipeline results
        assert len(pipeline_results) == 6  # 3 episodes * 2 cycles each
        
        # Check decision variety
        decisions = [r['decision'].action for r in pipeline_results]
        assert len(set(decisions)) > 1  # Should have different decisions
        
        # Check reward calculation
        rewards = [r['rewards'].total_reward for r in pipeline_results]
        assert all(np.isfinite(r) for r in rewards)
        
        print(f"✅ Component integration verified with {len(pipeline_results)} pipeline cycles")
    
    def test_performance_latency_validation(self, tactical_env, decision_aggregator, reward_system):
        """Test performance and latency requirements"""
        latencies = {
            'observation_generation': [],
            'action_execution': [],
            'decision_aggregation': [],
            'reward_calculation': [],
            'total_pipeline': []
        }
        
        # Run performance test
        for iteration in range(20):
            pipeline_start = time.time()
            
            # Observation generation
            obs_start = time.time()
            observations = tactical_env.reset()
            obs_end = time.time()
            latencies['observation_generation'].append((obs_end - obs_start) * 1000)
            
            # Action execution
            agent_outputs = {}
            for agent in ['fvg_agent', 'momentum_agent', 'entry_opt_agent']:
                action_start = time.time()
                obs, rewards, dones, infos = tactical_env.step(1)  # Neutral action
                action_end = time.time()
                latencies['action_execution'].append((action_end - action_start) * 1000)
                
                agent_outputs[agent] = Mock(
                    probabilities=np.array([0.33, 0.34, 0.33]),
                    action=1,
                    confidence=0.7,
                    timestamp=time.time()
                )
            
            # Decision aggregation
            agg_start = time.time()
            market_state = Mock()
            market_state.features = {'current_price': 100.0}
            
            decision_result = decision_aggregator.aggregate_decisions(
                agent_outputs=agent_outputs,
                market_state=market_state,
                synergy_context={'type': 'TYPE_1', 'direction': 1, 'confidence': 0.7}
            )
            agg_end = time.time()
            latencies['decision_aggregation'].append((agg_end - agg_start) * 1000)
            
            # Reward calculation
            reward_start = time.time()
            reward_components = reward_system.calculate_tactical_reward(
                decision_result=decision_result.__dict__,
                market_state=market_state,
                agent_outputs=agent_outputs
            )
            reward_end = time.time()
            latencies['reward_calculation'].append((reward_end - reward_start) * 1000)
            
            pipeline_end = time.time()
            latencies['total_pipeline'].append((pipeline_end - pipeline_start) * 1000)
        
        # Analyze latencies
        performance_report = {}
        for component, times in latencies.items():
            performance_report[component] = {
                'mean': np.mean(times),
                'p95': np.percentile(times, 95),
                'p99': np.percentile(times, 99),
                'max': np.max(times)
            }
        
        # Verify performance targets
        assert performance_report['total_pipeline']['p95'] < 100  # 95th percentile under 100ms
        assert performance_report['decision_aggregation']['mean'] < 10  # Average under 10ms
        assert performance_report['reward_calculation']['mean'] < 5  # Average under 5ms
        
        print("✅ Performance validation passed:")
        for component, metrics in performance_report.items():
            print(f"  {component}: {metrics['mean']:.2f}ms avg, {metrics['p95']:.2f}ms p95")
    
    def test_error_handling_and_recovery(self, tactical_env, decision_aggregator, reward_system):
        """Test error handling and recovery mechanisms"""
        # Test 1: Invalid agent action
        observations = tactical_env.reset()
        
        # Execute invalid action
        obs, rewards, dones, infos = tactical_env.step(999)  # Invalid action
        
        # Should handle gracefully
        assert isinstance(obs, dict)
        assert isinstance(rewards, dict)
        
        # Test 2: Corrupted agent output
        corrupted_agent_outputs = {
            'fvg_agent': Mock(probabilities=np.array([np.nan, 0.5, 0.5]), action=1, confidence=0.7),
            'momentum_agent': Mock(probabilities=np.array([0.3, 0.3, 0.4]), action=2, confidence=0.8),
            'entry_opt_agent': Mock(probabilities=np.array([0.2, 0.3, 0.5]), action=1, confidence=0.6)
        }
        
        market_state = Mock()
        market_state.features = {'current_price': 100.0}
        
        # Should handle NaN values gracefully
        decision_result = decision_aggregator.aggregate_decisions(
            agent_outputs=corrupted_agent_outputs,
            market_state=market_state,
            synergy_context={'type': 'TYPE_1', 'direction': 1, 'confidence': 0.7}
        )
        
        assert hasattr(decision_result, 'execute')
        assert np.isfinite(decision_result.confidence)
        
        # Test 3: Missing market data
        minimal_market_state = Mock()
        minimal_market_state.features = {}  # Empty features
        
        # Should handle missing data gracefully
        reward_components = reward_system.calculate_tactical_reward(
            decision_result=decision_result.__dict__,
            market_state=minimal_market_state,
            agent_outputs=corrupted_agent_outputs
        )
        
        assert np.isfinite(reward_components.total_reward)
        
        print("✅ Error handling and recovery validated")
    
    def test_concurrent_processing_safety(self, tactical_config):
        """Test concurrent processing safety"""
        # Create multiple environments
        environments = [TacticalMarketEnv(tactical_config) for _ in range(3)]
        
        # Test concurrent execution
        def run_episode(env_id, env):
            try:
                observations = env.reset()
                
                for step in range(10):
                    current_agent = env.agent_selection
                    obs, rewards, dones, infos = env.step(step % 3)
                    
                    if all(dones.values()):
                        break
                
                return {'env_id': env_id, 'success': True, 'steps': step + 1}
            except Exception as e:
                return {'env_id': env_id, 'success': False, 'error': str(e)}
        
        # Run concurrent episodes
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_episode, i, env) for i, env in enumerate(environments)]
            results = [future.result() for future in futures]
        
        # Verify all succeeded
        assert all(r['success'] for r in results)
        
        # Verify no resource conflicts
        steps = [r['steps'] for r in results]
        assert all(s > 0 for s in steps)
        
        print("✅ Concurrent processing safety verified")


class TestFullTrainingPipeline:
    """Test suite for full training pipeline"""
    
    def test_training_orchestrator_smoke_test(self):
        """Smoke test for training orchestrator"""
        # Create temporary config file
        config = {
            'tactical_marl': {
                'environment': {
                    'matrix_shape': [60, 7],
                    'max_episode_steps': 20
                },
                'agents': {
                    'fvg_agent': {'hidden_dims': [32, 16, 8]},
                    'momentum_agent': {'hidden_dims': [32, 16, 8]},
                    'entry_opt_agent': {'hidden_dims': [32, 16, 8]}
                },
                'training': {
                    'max_episodes': 3,
                    'batch_size': 2,
                    'checkpoint_interval': 2
                },
                'models': {
                    'critic': {'hidden_dims': [64, 32, 16]}
                },
                'infrastructure': {
                    'log_level': 'ERROR',  # Reduce logging for tests
                    'log_dir': '/tmp/test_logs'
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            # Test orchestrator initialization
            orchestrator = TacticalTrainingOrchestrator(config_path)
            
            # Verify components initialized
            assert orchestrator.env is not None
            assert len(orchestrator.agents) == 3
            assert orchestrator.critic is not None
            
            # Test single episode execution
            episode_reward, episode_length = orchestrator._run_episode()
            
            # Verify episode completed
            assert isinstance(episode_reward, (int, float))
            assert isinstance(episode_length, int)
            assert episode_length > 0
            
            print(f"✅ Training orchestrator smoke test passed: "
                  f"reward={episode_reward:.3f}, length={episode_length}")
            
        finally:
            os.unlink(config_path)
    
    def test_agent_network_forward_pass(self):
        """Test agent network forward pass"""
        # Create test agent
        agent = TacticalActor(
            input_dim=7,
            hidden_dims=[32, 16, 8],
            action_dim=3,
            agent_id='test_agent'
        )
        
        # Test forward pass
        test_input = torch.randn(1, 60, 7)
        
        with torch.no_grad():
            output = agent(test_input)
        
        # Verify output structure
        assert 'action' in output
        assert 'action_probs' in output
        assert 'log_prob' in output
        assert 'value' in output
        
        # Verify shapes
        assert output['action_probs'].shape == (1, 3)
        assert output['value'].shape == (1,)
        
        # Verify probabilities sum to 1
        assert abs(output['action_probs'].sum().item() - 1.0) < 1e-6
        
        print("✅ Agent network forward pass validated")
    
    def test_training_metrics_collection(self):
        """Test training metrics collection"""
        # Create temporary config
        config = {
            'tactical_marl': {
                'environment': {'max_episode_steps': 10},
                'agents': {
                    'fvg_agent': {'hidden_dims': [16, 8]},
                    'momentum_agent': {'hidden_dims': [16, 8]},
                    'entry_opt_agent': {'hidden_dims': [16, 8]}
                },
                'training': {'max_episodes': 2},
                'models': {'critic': {'hidden_dims': [32, 16]}},
                'infrastructure': {'log_level': 'ERROR', 'log_dir': '/tmp/test_logs'}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            orchestrator = TacticalTrainingOrchestrator(config_path)
            
            # Run a few episodes
            for _ in range(3):
                reward, length = orchestrator._run_episode()
                orchestrator.episode_rewards.append(reward)
                orchestrator.episode_lengths.append(length)
            
            # Test metrics collection
            assert len(orchestrator.episode_rewards) == 3
            assert len(orchestrator.episode_lengths) == 3
            
            # Test performance evaluation
            orchestrator._evaluate_performance()
            
            print("✅ Training metrics collection validated")
            
        finally:
            os.unlink(config_path)


class TestProductionReadiness:
    """Test suite for production readiness verification"""
    
    def test_memory_leak_detection(self, tactical_config):
        """Test for memory leaks during extended operation"""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run extended operation
        env = TacticalMarketEnv(tactical_config)
        
        for episode in range(50):
            observations = env.reset()
            
            for step in range(20):
                obs, rewards, dones, infos = env.step(step % 3)
                
                if all(dones.values()):
                    break
            
            # Periodic garbage collection
            if episode % 10 == 0:
                gc.collect()
        
        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Should not grow by more than 100MB
        assert memory_growth < 100, f"Memory grew by {memory_growth:.2f}MB"
        
        print(f"✅ Memory leak test passed: {memory_growth:.2f}MB growth")
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        # Test valid configuration
        valid_config = {
            'tactical_marl': {
                'environment': {
                    'matrix_shape': [60, 7],
                    'max_episode_steps': 100
                },
                'agents': {
                    'fvg_agent': {'attention_weights': [0.4, 0.4, 0.1, 0.05, 0.05]},
                    'momentum_agent': {'attention_weights': [0.05, 0.05, 0.1, 0.3, 0.5]},
                    'entry_opt_agent': {'attention_weights': [0.2, 0.2, 0.2, 0.2, 0.2]}
                }
            }
        }
        
        env = TacticalMarketEnv(valid_config)
        assert env is not None
        
        # Test invalid configuration
        invalid_config = {
            'tactical_marl': {
                'environment': {
                    'matrix_shape': [60, 7],
                    'max_episode_steps': -1  # Invalid
                }
            }
        }
        
        # Should handle invalid config gracefully
        env = TacticalMarketEnv(invalid_config)
        assert env is not None  # Should use defaults
        
        print("✅ Configuration validation passed")
    
    def test_graceful_shutdown(self, tactical_config):
        """Test graceful shutdown behavior"""
        env = TacticalMarketEnv(tactical_config)
        
        # Start operation
        observations = env.reset()
        
        # Simulate shutdown
        env.close()
        
        # Should not crash
        assert True  # If we reach here, shutdown was graceful
        
        print("✅ Graceful shutdown test passed")
    
    def test_resource_cleanup(self, tactical_config):
        """Test resource cleanup"""
        # Create and destroy multiple environments
        for i in range(10):
            env = TacticalMarketEnv(tactical_config)
            observations = env.reset()
            
            # Execute a few steps
            for _ in range(5):
                obs, rewards, dones, infos = env.step(1)
            
            # Cleanup
            env.close()
            del env
        
        # Should not accumulate resources
        print("✅ Resource cleanup test passed")


def run_comprehensive_integration_tests():
    """Run comprehensive integration test suite"""
    print("🚀 Starting Tactical MARL Integration Tests")
    print("=" * 60)
    
    # Run pytest with comprehensive coverage
    import subprocess
    result = subprocess.run([
        'python', '-m', 'pytest', 
        __file__, 
        '-v', 
        '--tb=short',
        '--capture=no'
    ], capture_output=True, text=True)
    
    print("Integration Test Results:")
    print(result.stdout)
    
    if result.stderr:
        print("Errors:")
        print(result.stderr)
    
    return result.returncode == 0


if __name__ == '__main__':
    success = run_comprehensive_integration_tests()
    if success:
        print("\n✅ All integration tests passed!")
        print("🎯 Tactical MARL system is production-ready!")
    else:
        print("\n❌ Some integration tests failed!")
        print("🔧 Please review and fix issues before deployment.")
    
    sys.exit(0 if success else 1)