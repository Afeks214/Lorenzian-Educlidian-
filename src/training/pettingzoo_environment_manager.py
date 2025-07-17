"""
PettingZoo Environment Manager for MARL Training

This module provides a unified interface for managing PettingZoo environments
across different training scenarios (tactical, strategic, execution, risk).
It handles environment initialization, validation, and lifecycle management.

Key Features:
- Environment factory pattern for different MARL systems
- PettingZoo API compliance validation
- Environment wrappers for enhanced functionality
- Multi-environment support for parallel training
- Performance monitoring and debugging
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Any, Callable, Type, Union, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass, field
from enum import Enum
import yaml
import json
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor

# PettingZoo imports
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.test import api_test
from gymnasium import spaces

# Import existing environments
from src.environment.strategic_env import StrategicMARLEnvironment
from src.environment.tactical_env import TacticalMarketEnv
from src.environment.execution_env import ExecutionEnvironment  
from src.environment.risk_env import RiskEnvironment

logger = logging.getLogger(__name__)


class EnvironmentType(Enum):
    """Supported environment types"""
    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    EXECUTION = "execution"
    RISK = "risk"
    UNIFIED = "unified"


@dataclass
class EnvironmentConfig:
    """Configuration for PettingZoo environments"""
    env_type: EnvironmentType
    env_params: Dict[str, Any] = field(default_factory=dict)
    
    # PettingZoo specific settings
    max_cycles: int = 1000
    render_mode: str = "human"
    
    # Wrappers to apply
    apply_assert_wrapper: bool = True
    apply_order_wrapper: bool = True
    apply_clip_actions_wrapper: bool = False
    apply_clip_reward_wrapper: bool = False
    
    # Validation settings
    run_api_test: bool = False
    api_test_cycles: int = 10
    
    # Performance settings
    enable_performance_monitoring: bool = True
    log_environment_stats: bool = True
    
    # Multi-environment settings
    num_parallel_envs: int = 1
    shared_seed: bool = False
    
    # Data collection
    collect_observations: bool = False
    collect_actions: bool = False
    collect_rewards: bool = False
    observation_buffer_size: int = 10000


class EnvironmentValidator:
    """Validates PettingZoo environment compliance"""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_environment(self, env: AECEnv) -> Dict[str, Any]:
        """Comprehensive environment validation"""
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'api_compliance': True,
            'performance_metrics': {}
        }
        
        try:
            # Basic attribute checks
            self._check_basic_attributes(env, results)
            
            # API compliance check
            self._check_api_compliance(env, results)
            
            # Space validation
            self._check_spaces(env, results)
            
            # Reset/step validation
            self._check_reset_step_cycle(env, results)
            
            # Performance benchmarks
            self._benchmark_performance(env, results)
            
        except Exception as e:
            results['is_valid'] = False
            results['errors'].append(f"Validation failed: {str(e)}")
            logger.error(f"Environment validation failed: {e}")
        
        return results
    
    def _check_basic_attributes(self, env: AECEnv, results: Dict[str, Any]):
        """Check basic PettingZoo attributes"""
        required_attrs = [
            'agent_selection', 'agents', 'possible_agents',
            'observation_spaces', 'action_spaces',
            'rewards', 'dones', 'truncations', 'infos'
        ]
        
        for attr in required_attrs:
            if not hasattr(env, attr):
                results['errors'].append(f"Missing required attribute: {attr}")
                results['is_valid'] = False
    
    def _check_api_compliance(self, env: AECEnv, results: Dict[str, Any]):
        """Check PettingZoo API compliance"""
        try:
            # Run official API test
            api_test(env, num_cycles=5, verbose_progress=False)
            results['api_compliance'] = True
        except Exception as e:
            results['api_compliance'] = False
            results['errors'].append(f"API test failed: {str(e)}")
    
    def _check_spaces(self, env: AECEnv, results: Dict[str, Any]):
        """Validate observation and action spaces"""
        try:
            for agent in env.possible_agents:
                # Check observation space
                obs_space = env.observation_space(agent)
                if not isinstance(obs_space, spaces.Space):
                    results['errors'].append(f"Invalid observation space for {agent}")
                
                # Check action space
                action_space = env.action_space(agent)
                if not isinstance(action_space, spaces.Space):
                    results['errors'].append(f"Invalid action space for {agent}")
        except Exception as e:
            results['errors'].append(f"Space validation failed: {str(e)}")
    
    def _check_reset_step_cycle(self, env: AECEnv, results: Dict[str, Any]):
        """Test basic reset/step cycle"""
        try:
            # Reset environment
            env.reset()
            
            # Check initial state
            if not env.agents:
                results['errors'].append("No agents after reset")
                return
            
            # Run a few steps
            for _ in range(5):
                if not env.agents:
                    break
                
                current_agent = env.agent_selection
                obs = env.observe(current_agent)
                action = env.action_space(current_agent).sample()
                
                env.step(action)
                
                # Check state consistency
                if current_agent in env.agents:
                    if current_agent not in env.rewards:
                        results['warnings'].append(f"No reward for active agent {current_agent}")
                    if current_agent not in env.dones:
                        results['warnings'].append(f"No done flag for active agent {current_agent}")
        
        except Exception as e:
            results['errors'].append(f"Reset/step cycle failed: {str(e)}")
    
    def _benchmark_performance(self, env: AECEnv, results: Dict[str, Any]):
        """Benchmark environment performance"""
        import time
        
        try:
            # Reset timing
            start_time = time.time()
            env.reset()
            reset_time = time.time() - start_time
            
            # Step timing
            step_times = []
            for _ in range(10):
                if not env.agents:
                    break
                
                current_agent = env.agent_selection
                obs = env.observe(current_agent)
                action = env.action_space(current_agent).sample()
                
                start_time = time.time()
                env.step(action)
                step_time = time.time() - start_time
                step_times.append(step_time)
            
            # Observation timing
            obs_times = []
            for _ in range(10):
                if not env.agents:
                    break
                
                current_agent = env.agent_selection
                start_time = time.time()
                obs = env.observe(current_agent)
                obs_time = time.time() - start_time
                obs_times.append(obs_time)
            
            results['performance_metrics'] = {
                'reset_time': reset_time,
                'avg_step_time': np.mean(step_times) if step_times else 0,
                'avg_observation_time': np.mean(obs_times) if obs_times else 0,
                'total_benchmark_steps': len(step_times)
            }
        
        except Exception as e:
            results['warnings'].append(f"Performance benchmark failed: {str(e)}")


class EnvironmentFactory:
    """Factory for creating PettingZoo environments"""
    
    def __init__(self):
        self.registered_environments = {}
        self._register_default_environments()
    
    def _register_default_environments(self):
        """Register default environment types"""
        self.registered_environments = {
            EnvironmentType.STRATEGIC: self._create_strategic_env,
            EnvironmentType.TACTICAL: self._create_tactical_env,
            EnvironmentType.EXECUTION: self._create_execution_env,
            EnvironmentType.RISK: self._create_risk_env,
            EnvironmentType.UNIFIED: self._create_unified_env
        }
    
    def create_environment(self, config: EnvironmentConfig) -> AECEnv:
        """Create environment based on configuration"""
        if config.env_type not in self.registered_environments:
            raise ValueError(f"Unknown environment type: {config.env_type}")
        
        # Create base environment
        env_factory = self.registered_environments[config.env_type]
        env = env_factory(config.env_params)
        
        # Apply wrappers
        env = self._apply_wrappers(env, config)
        
        # Validate environment
        if config.run_api_test:
            validator = EnvironmentValidator()
            validation_results = validator.validate_environment(env)
            
            if not validation_results['is_valid']:
                raise ValueError(f"Environment validation failed: {validation_results['errors']}")
            
            logger.info(f"Environment validation passed: {validation_results['performance_metrics']}")
        
        return env
    
    def _create_strategic_env(self, params: Dict[str, Any]) -> AECEnv:
        """Create strategic MARL environment"""
        return StrategicMARLEnvironment(config=params)
    
    def _create_tactical_env(self, params: Dict[str, Any]) -> AECEnv:
        """Create tactical MARL environment"""
        return TacticalMarketEnv(config=params)
    
    def _create_execution_env(self, params: Dict[str, Any]) -> AECEnv:
        """Create execution environment"""
        return ExecutionEnvironment(config=params)
    
    def _create_risk_env(self, params: Dict[str, Any]) -> AECEnv:
        """Create risk management environment"""
        return RiskEnvironment(config=params)
    
    def _create_unified_env(self, params: Dict[str, Any]) -> AECEnv:
        """Create unified environment with all agents"""
        # This would create a unified environment with all agent types
        # For now, default to strategic
        return self._create_strategic_env(params)
    
    def _apply_wrappers(self, env: AECEnv, config: EnvironmentConfig) -> AECEnv:
        """Apply PettingZoo wrappers"""
        if config.apply_assert_wrapper:
            env = wrappers.AssertOutOfBoundsWrapper(env)
        
        if config.apply_order_wrapper:
            env = wrappers.OrderEnforcingWrapper(env)
        
        if config.apply_clip_actions_wrapper:
            env = wrappers.ClipOutOfBoundsWrapper(env)
        
        if config.apply_clip_reward_wrapper:
            env = wrappers.ClipRewardWrapper(env, min_reward=-1.0, max_reward=1.0)
        
        return env
    
    def register_environment(self, env_type: EnvironmentType, factory_func: Callable):
        """Register custom environment type"""
        self.registered_environments[env_type] = factory_func


class MultiEnvironmentManager:
    """Manages multiple parallel environments"""
    
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.environments = []
        self.factory = EnvironmentFactory()
        self.thread_pool = ThreadPoolExecutor(max_workers=config.num_parallel_envs)
        
        # Initialize environments
        self._initialize_environments()
    
    def _initialize_environments(self):
        """Initialize parallel environments"""
        for i in range(self.config.num_parallel_envs):
            # Create environment config for this instance
            env_config = EnvironmentConfig(
                env_type=self.config.env_type,
                env_params=self.config.env_params.copy(),
                max_cycles=self.config.max_cycles,
                render_mode=self.config.render_mode
            )
            
            # Set unique seed if not shared
            if not self.config.shared_seed:
                env_config.env_params['seed'] = i
            
            # Create environment
            env = self.factory.create_environment(env_config)
            self.environments.append(env)
        
        logger.info(f"Initialized {len(self.environments)} parallel environments")
    
    def reset_all(self) -> List[Dict[str, Any]]:
        """Reset all environments"""
        def reset_env(env):
            env.reset()
            return {agent: env.observe(agent) for agent in env.agents}
        
        futures = [self.thread_pool.submit(reset_env, env) for env in self.environments]
        return [future.result() for future in futures]
    
    def step_all(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Step all environments"""
        def step_env(env, action_dict):
            results = {}
            for agent, action in action_dict.items():
                if agent == env.agent_selection:
                    env.step(action)
                    results[agent] = {
                        'observation': env.observe(agent),
                        'reward': env.rewards.get(agent, 0.0),
                        'done': env.dones.get(agent, False),
                        'truncated': env.truncations.get(agent, False),
                        'info': env.infos.get(agent, {})
                    }
            return results
        
        futures = [
            self.thread_pool.submit(step_env, env, actions[i]) 
            for i, env in enumerate(self.environments)
        ]
        return [future.result() for future in futures]
    
    def get_environment(self, index: int) -> AECEnv:
        """Get specific environment by index"""
        return self.environments[index]
    
    def close_all(self):
        """Close all environments"""
        for env in self.environments:
            if hasattr(env, 'close'):
                env.close()
        
        self.thread_pool.shutdown(wait=True)


class EnvironmentDataCollector:
    """Collects data from environment interactions"""
    
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.data = {
            'observations': deque(maxlen=config.observation_buffer_size),
            'actions': deque(maxlen=config.observation_buffer_size),
            'rewards': deque(maxlen=config.observation_buffer_size),
            'dones': deque(maxlen=config.observation_buffer_size),
            'infos': deque(maxlen=config.observation_buffer_size),
            'agent_data': {}
        }
        
        self.episode_count = 0
        self.step_count = 0
        self.collection_active = False
    
    def start_collection(self):
        """Start data collection"""
        self.collection_active = True
        logger.info("Environment data collection started")
    
    def stop_collection(self):
        """Stop data collection"""
        self.collection_active = False
        logger.info("Environment data collection stopped")
    
    def record_step(self, agent: str, observation: Any, action: Any, 
                   reward: float, done: bool, info: Dict[str, Any]):
        """Record single step data"""
        if not self.collection_active:
            return
        
        if self.config.collect_observations:
            self.data['observations'].append({
                'agent': agent,
                'observation': observation,
                'step': self.step_count,
                'episode': self.episode_count
            })
        
        if self.config.collect_actions:
            self.data['actions'].append({
                'agent': agent,
                'action': action,
                'step': self.step_count,
                'episode': self.episode_count
            })
        
        if self.config.collect_rewards:
            self.data['rewards'].append({
                'agent': agent,
                'reward': reward,
                'step': self.step_count,
                'episode': self.episode_count
            })
        
        self.data['dones'].append({
            'agent': agent,
            'done': done,
            'step': self.step_count,
            'episode': self.episode_count
        })
        
        self.data['infos'].append({
            'agent': agent,
            'info': info,
            'step': self.step_count,
            'episode': self.episode_count
        })
        
        # Agent-specific data
        if agent not in self.data['agent_data']:
            self.data['agent_data'][agent] = {
                'total_reward': 0.0,
                'step_count': 0,
                'episode_count': 0
            }
        
        self.data['agent_data'][agent]['total_reward'] += reward
        self.data['agent_data'][agent]['step_count'] += 1
        
        self.step_count += 1
    
    def record_episode_end(self):
        """Record episode end"""
        if not self.collection_active:
            return
        
        self.episode_count += 1
        
        # Update agent episode counts
        for agent_data in self.data['agent_data'].values():
            agent_data['episode_count'] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return {
            'total_episodes': self.episode_count,
            'total_steps': self.step_count,
            'buffer_sizes': {
                'observations': len(self.data['observations']),
                'actions': len(self.data['actions']),
                'rewards': len(self.data['rewards']),
                'dones': len(self.data['dones']),
                'infos': len(self.data['infos'])
            },
            'agent_statistics': self.data['agent_data'].copy(),
            'collection_active': self.collection_active
        }
    
    def save_data(self, filepath: str):
        """Save collected data to file"""
        data_to_save = {
            'statistics': self.get_statistics(),
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        # Convert deques to lists for JSON serialization
        for key, value in self.data.items():
            if isinstance(value, deque):
                data_to_save[key] = list(value)
            else:
                data_to_save[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(data_to_save, f, indent=2, default=str)
        
        logger.info(f"Environment data saved to {filepath}")


def create_environment_config(env_type: EnvironmentType, **kwargs) -> EnvironmentConfig:
    """Create environment configuration"""
    return EnvironmentConfig(env_type=env_type, **kwargs)


def create_strategic_environment(config: Dict[str, Any] = None) -> AECEnv:
    """Create strategic MARL environment"""
    env_config = create_environment_config(
        EnvironmentType.STRATEGIC,
        env_params=config or {},
        run_api_test=True
    )
    
    factory = EnvironmentFactory()
    return factory.create_environment(env_config)


def create_tactical_environment(config: Dict[str, Any] = None) -> AECEnv:
    """Create tactical MARL environment"""
    env_config = create_environment_config(
        EnvironmentType.TACTICAL,
        env_params=config or {},
        run_api_test=True
    )
    
    factory = EnvironmentFactory()
    return factory.create_environment(env_config)


def create_execution_environment(config: Dict[str, Any] = None) -> AECEnv:
    """Create execution environment"""
    env_config = create_environment_config(
        EnvironmentType.EXECUTION,
        env_params=config or {},
        run_api_test=True
    )
    
    factory = EnvironmentFactory()
    return factory.create_environment(env_config)


def create_risk_environment(config: Dict[str, Any] = None) -> AECEnv:
    """Create risk management environment"""
    env_config = create_environment_config(
        EnvironmentType.RISK,
        env_params=config or {},
        run_api_test=True
    )
    
    factory = EnvironmentFactory()
    return factory.create_environment(env_config)


def validate_environment_setup(env: AECEnv) -> Dict[str, Any]:
    """Validate environment setup for training"""
    validator = EnvironmentValidator()
    return validator.validate_environment(env)


# Example usage and testing
if __name__ == "__main__":
    # Create strategic environment
    strategic_env = create_strategic_environment({
        'max_episode_steps': 1000,
        'confidence_threshold': 0.6
    })
    
    # Validate environment
    validation_results = validate_environment_setup(strategic_env)
    print("Strategic Environment Validation:", validation_results)
    
    # Create tactical environment
    tactical_env = create_tactical_environment({
        'max_episode_steps': 500,
        'decision_timeout_ms': 100
    })
    
    # Validate environment
    validation_results = validate_environment_setup(tactical_env)
    print("Tactical Environment Validation:", validation_results)
    
    # Test multi-environment setup
    multi_env_config = create_environment_config(
        EnvironmentType.TACTICAL,
        env_params={'max_episode_steps': 100},
        num_parallel_envs=2
    )
    
    multi_manager = MultiEnvironmentManager(multi_env_config)
    
    # Test parallel reset
    initial_obs = multi_manager.reset_all()
    print(f"Initialized {len(initial_obs)} parallel environments")
    
    # Cleanup
    multi_manager.close_all()
    strategic_env.close()
    tactical_env.close()
    
    print("Environment manager testing completed!")