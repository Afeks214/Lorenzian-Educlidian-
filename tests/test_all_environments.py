"""
Comprehensive Test Suite for All PettingZoo Environments

This test suite runs comprehensive tests for all four PettingZoo environments:
- Strategic Environment (3 agents)
- Tactical Environment (3 agents)  
- Risk Environment (4 agents)
- Execution Environment (5 agents)

Features:
- PettingZoo API compliance testing
- Cross-environment compatibility
- Performance benchmarking
- Integration testing
- Error handling validation

Author: Claude Code
Version: 1.0
"""

import pytest
import numpy as np
import time
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.environment.strategic_env import StrategicMARLEnvironment, raw_env as strategic_raw_env
from src.environment.tactical_env import TacticalMarketEnv, make_tactical_env
from src.environment.risk_env import RiskManagementEnv, create_risk_environment
from src.environment.execution_env import ExecutionEnvironment, raw_env as execution_raw_env
from pettingzoo.test import api_test


class TestAllEnvironments:
    """Test suite for all PettingZoo environments"""
    
    @pytest.fixture
    def environments(self):
        """Create all environment instances for testing"""
        return {
            'strategic': strategic_raw_env(),
            'tactical': make_tactical_env(),
            'risk': create_risk_environment(),
            'execution': execution_raw_env()
        }
    
    def test_all_environments_initialization(self, environments):
        """Test that all environments initialize correctly"""
        expected_agent_counts = {
            'strategic': 3,
            'tactical': 3,
            'risk': 4,
            'execution': 5
        }
        
        for env_name, env in environments.items():
            assert env is not None
            assert len(env.possible_agents) == expected_agent_counts[env_name]
            assert hasattr(env, 'reset')
            assert hasattr(env, 'step')
            assert hasattr(env, 'observe')
            assert hasattr(env, 'close')
    
    def test_all_environments_pettingzoo_compliance(self, environments):
        """Test PettingZoo API compliance for all environments"""
        for env_name, env in environments.items():
            print(f"Testing PettingZoo compliance for {env_name} environment...")
            
            try:
                # Run PettingZoo API test with reduced cycles for speed
                api_test(env, num_cycles=3, verbose_progress=False)
                print(f"✓ {env_name} environment passes PettingZoo API test")
            except Exception as e:
                pytest.fail(f"❌ {env_name} environment failed PettingZoo API test: {e}")
    
    def test_all_environments_action_spaces(self, environments):
        """Test action spaces for all environments"""
        for env_name, env in environments.items():
            env.reset()
            
            for agent in env.possible_agents:
                action_space = env.action_spaces[agent]
                
                # Test sampling
                for _ in range(3):
                    action = action_space.sample()
                    assert action_space.contains(action), f"{env_name} {agent} action space violation"
    
    def test_all_environments_observation_spaces(self, environments):
        """Test observation spaces for all environments"""
        for env_name, env in environments.items():
            env.reset()
            
            for agent in env.possible_agents:
                obs_space = env.observation_spaces[agent]
                obs = env.observe(agent)
                
                assert obs_space.contains(obs), f"{env_name} {agent} observation space violation"
                assert np.all(np.isfinite(obs)), f"{env_name} {agent} observation contains invalid values"
    
    def test_all_environments_episode_execution(self, environments):
        """Test episode execution for all environments"""
        for env_name, env in environments.items():
            env.reset()
            
            steps = 0
            max_steps = 20
            
            while env.agents and steps < max_steps:
                agent = env.agent_selection
                action = env.action_spaces[agent].sample()
                
                # Execute step
                env.step(action)
                steps += 1
                
                # Check environment state consistency
                if env.agents:
                    assert env.agent_selection in env.agents
                    assert agent in env.rewards if hasattr(env, 'rewards') else True
                    assert agent in env.terminations if hasattr(env, 'terminations') else True
                    assert agent in env.truncations if hasattr(env, 'truncations') else True
                    assert agent in env.infos if hasattr(env, 'infos') else True
            
            print(f"✓ {env_name} environment executed {steps} steps successfully")
    
    def test_all_environments_performance(self, environments):
        """Test performance benchmarks for all environments"""
        performance_results = {}
        
        for env_name, env in environments.items():
            # Measure reset time
            start_time = time.time()
            env.reset()
            reset_time = time.time() - start_time
            
            # Measure step times
            step_times = []
            for _ in range(10):
                start_time = time.time()
                action = env.action_spaces[env.agent_selection].sample()
                env.step(action)
                step_time = time.time() - start_time
                step_times.append(step_time)
            
            avg_step_time = np.mean(step_times)
            
            performance_results[env_name] = {
                'reset_time': reset_time,
                'avg_step_time': avg_step_time,
                'max_step_time': np.max(step_times),
                'min_step_time': np.min(step_times)
            }
            
            # Performance assertions
            assert reset_time < 2.0, f"{env_name} reset time too slow: {reset_time:.3f}s"
            assert avg_step_time < 0.5, f"{env_name} average step time too slow: {avg_step_time:.3f}s"
        
        # Print performance summary
        print("\n=== Performance Summary ===")
        for env_name, results in performance_results.items():
            print(f"{env_name.upper()}:")
            print(f"  Reset time: {results['reset_time']:.3f}s")
            print(f"  Avg step time: {results['avg_step_time']:.3f}s")
            print(f"  Min/Max step time: {results['min_step_time']:.3f}s / {results['max_step_time']:.3f}s")
    
    def test_all_environments_memory_stability(self, environments):
        """Test memory stability across multiple episodes"""
        for env_name, env in environments.items():
            initial_memory = {}
            
            for episode in range(3):
                env.reset()
                steps = 0
                
                while env.agents and steps < 15:
                    action = env.action_spaces[env.agent_selection].sample()
                    env.step(action)
                    steps += 1
                
                # Check memory usage doesn't grow unbounded
                # This is a simplified check - in practice you'd monitor actual memory usage
                if hasattr(env, 'performance_metrics'):
                    current_memory = len(str(env.performance_metrics))
                    if episode == 0:
                        initial_memory[env_name] = current_memory
                    else:
                        # Memory shouldn't grow more than 2x
                        assert current_memory < initial_memory[env_name] * 2, \
                            f"{env_name} memory growth detected"
    
    def test_environments_error_handling(self, environments):
        """Test error handling and recovery"""
        for env_name, env in environments.items():
            env.reset()
            
            # Test invalid action handling
            try:
                if hasattr(env.action_spaces[env.agent_selection], 'n'):
                    # Discrete action space
                    env.step(999)  # Invalid action
                else:
                    # Continuous action space
                    env.step(np.array([999.0] * env.action_spaces[env.agent_selection].shape[0]))
            except Exception:
                pass  # Expected to fail
            
            # Environment should still be functional
            if env.agents:
                valid_action = env.action_spaces[env.agent_selection].sample()
                env.step(valid_action)
                assert env.agent_selection in env.agents
    
    def test_environments_deterministic_behavior(self, environments):
        """Test deterministic behavior with same seed"""
        for env_name, env in environments.items():
            # Skip execution environment due to async complexity
            if env_name == 'execution':
                continue
            
            # Run with same seed twice
            results1 = []
            env.reset(seed=42)
            
            for _ in range(6):
                if not env.agents:
                    break
                action = env.action_spaces[env.agent_selection].sample()
                env.step(action)
                results1.append(env.agent_selection)
            
            results2 = []
            env.reset(seed=42)
            
            for _ in range(6):
                if not env.agents:
                    break
                action = env.action_spaces[env.agent_selection].sample()  
                env.step(action)
                results2.append(env.agent_selection)
            
            # Agent selection sequence should be identical
            assert results1 == results2, f"{env_name} not deterministic with same seed"
    
    def test_environments_rendering(self, environments):
        """Test rendering capabilities"""
        for env_name, env in environments.items():
            env.reset()
            
            # Test human rendering
            try:
                env.render(mode='human')
            except Exception as e:
                pytest.fail(f"{env_name} human rendering failed: {e}")
            
            # Test rgb_array rendering if supported
            if 'rgb_array' in env.metadata.get('render_modes', []):
                try:
                    result = env.render(mode='rgb_array')
                    if result is not None:
                        assert isinstance(result, np.ndarray)
                        assert result.ndim == 3  # Height x Width x Channels
                except Exception as e:
                    pytest.fail(f"{env_name} rgb_array rendering failed: {e}")
    
    def test_environments_cleanup(self, environments):
        """Test environment cleanup"""
        for env_name, env in environments.items():
            env.reset()
            
            # Execute a few steps
            for _ in range(5):
                if not env.agents:
                    break
                action = env.action_spaces[env.agent_selection].sample()
                env.step(action)
            
            # Close environment
            try:
                env.close()
            except Exception as e:
                pytest.fail(f"{env_name} cleanup failed: {e}")


class TestEnvironmentComparison:
    """Test suite for comparing environments"""
    
    def test_environment_characteristics(self):
        """Test and compare environment characteristics"""
        environments = {
            'strategic': strategic_raw_env(),
            'tactical': make_tactical_env(),
            'risk': create_risk_environment(),
            'execution': execution_raw_env()
        }
        
        characteristics = {}
        
        for env_name, env in environments.items():
            env.reset()
            
            # Gather characteristics
            char = {
                'num_agents': len(env.possible_agents),
                'agent_names': env.possible_agents,
                'action_spaces': {},
                'observation_spaces': {},
                'metadata': env.metadata
            }
            
            for agent in env.possible_agents:
                char['action_spaces'][agent] = {
                    'type': type(env.action_spaces[agent]).__name__,
                    'shape': getattr(env.action_spaces[agent], 'shape', None),
                    'n': getattr(env.action_spaces[agent], 'n', None)
                }
                
                char['observation_spaces'][agent] = {
                    'type': type(env.observation_spaces[agent]).__name__,
                    'shape': getattr(env.observation_spaces[agent], 'shape', None)
                }
            
            characteristics[env_name] = char
            env.close()
        
        # Print comparison
        print("\n=== Environment Characteristics ===")
        for env_name, char in characteristics.items():
            print(f"{env_name.upper()}:")
            print(f"  Agents: {char['num_agents']} - {char['agent_names']}")
            print(f"  Metadata: {char['metadata']}")
            
            for agent in char['agent_names']:
                action_info = char['action_spaces'][agent]
                obs_info = char['observation_spaces'][agent]
                print(f"  {agent}:")
                print(f"    Action: {action_info['type']} {action_info.get('shape', action_info.get('n'))}")
                print(f"    Observation: {obs_info['type']} {obs_info.get('shape')}")
        
        # Verify environment diversity
        agent_counts = [char['num_agents'] for char in characteristics.values()]
        assert len(set(agent_counts)) == 4, "All environments should have different agent counts"
    
    def test_environment_scalability(self):
        """Test environment scalability"""
        environments = {
            'strategic': strategic_raw_env(),
            'tactical': make_tactical_env(),
            'risk': create_risk_environment(),
            'execution': execution_raw_env()
        }
        
        scalability_results = {}
        
        for env_name, env in environments.items():
            # Test multiple episodes
            episode_times = []
            
            for episode in range(3):
                start_time = time.time()
                env.reset()
                
                steps = 0
                while env.agents and steps < 10:
                    action = env.action_spaces[env.agent_selection].sample()
                    env.step(action)
                    steps += 1
                
                episode_time = time.time() - start_time
                episode_times.append(episode_time)
            
            scalability_results[env_name] = {
                'avg_episode_time': np.mean(episode_times),
                'episode_consistency': np.std(episode_times) < 0.1  # Low variance
            }
            
            env.close()
        
        # Print scalability results
        print("\n=== Scalability Results ===")
        for env_name, results in scalability_results.items():
            print(f"{env_name}: {results['avg_episode_time']:.3f}s avg, consistent: {results['episode_consistency']}")
            
            # Assert reasonable performance
            assert results['avg_episode_time'] < 1.0, f"{env_name} episode time too slow"
            assert results['episode_consistency'], f"{env_name} episode times inconsistent"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])