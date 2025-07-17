# PettingZoo Troubleshooting Guide

## Overview

This guide provides solutions to common issues encountered when working with GrandModel's PettingZoo environments. It covers installation problems, runtime errors, performance issues, and integration challenges.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Environment Initialization](#environment-initialization)
- [Runtime Errors](#runtime-errors)
- [Performance Issues](#performance-issues)
- [Training Problems](#training-problems)
- [Integration Issues](#integration-issues)
- [API Compliance](#api-compliance)
- [Debugging Tools](#debugging-tools)

## Installation Issues

### Issue 1: PettingZoo Installation Fails

**Symptoms:**
```bash
ERROR: Could not find a version that satisfies the requirement pettingzoo
```

**Solutions:**

1. **Update pip and try again:**
```bash
pip install --upgrade pip
pip install pettingzoo[classic]
```

2. **Use specific version:**
```bash
pip install pettingzoo==1.24.0
```

3. **Install from source:**
```bash
git clone https://github.com/Farama-Foundation/PettingZoo.git
cd PettingZoo
pip install -e .
```

4. **Check Python version compatibility:**
```bash
python --version  # Should be 3.8+
```

### Issue 2: Gymnasium Version Conflicts

**Symptoms:**
```bash
ImportError: cannot import name 'Space' from 'gymnasium.spaces'
```

**Solutions:**

1. **Install compatible versions:**
```bash
pip install gymnasium==0.29.1
pip install pettingzoo[classic]==1.24.0
```

2. **Create clean virtual environment:**
```bash
python -m venv fresh_env
source fresh_env/bin/activate
pip install pettingzoo[classic] gymnasium numpy torch
```

3. **Check for conflicts:**
```bash
pip list | grep -E "gymnasium|pettingzoo|gym"
```

### Issue 3: Missing Dependencies

**Symptoms:**
```bash
ModuleNotFoundError: No module named 'torch'
```

**Solutions:**

1. **Install all required dependencies:**
```bash
pip install torch numpy pandas matplotlib
pip install pettingzoo[classic] gymnasium
```

2. **Use requirements file:**
```bash
pip install -r requirements.txt
```

3. **Check installation:**
```bash
python -c "import pettingzoo, gymnasium, torch, numpy; print('All imports successful')"
```

## Environment Initialization

### Issue 1: Configuration File Not Found

**Symptoms:**
```bash
FileNotFoundError: [Errno 2] No such file or directory: 'config.yaml'
```

**Solutions:**

1. **Use absolute paths:**
```python
import os
config_path = os.path.abspath('configs/strategic_marl.yaml')
env = StrategicMarketEnv({'config_path': config_path})
```

2. **Create default configuration:**
```python
default_config = {
    'strategic_marl': {
        'environment': {
            'matrix_shape': [48, 13],
            'max_episode_steps': 1000
        }
    }
}
env = StrategicMarketEnv(default_config)
```

3. **Verify file exists:**
```bash
ls -la configs/
find . -name "*.yaml" -type f
```

### Issue 2: Invalid Configuration Format

**Symptoms:**
```bash
ValueError: Invalid configuration format
```

**Solutions:**

1. **Validate YAML syntax:**
```bash
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

2. **Check required fields:**
```python
def validate_config(config):
    required_fields = ['strategic_marl', 'environment', 'matrix_shape']
    for field in required_fields:
        if field not in str(config):
            raise ValueError(f"Missing required field: {field}")
    return True
```

3. **Use configuration schema:**
```python
import cerberus

schema = {
    'strategic_marl': {
        'type': 'dict',
        'required': True,
        'schema': {
            'environment': {
                'type': 'dict',
                'required': True,
                'schema': {
                    'matrix_shape': {'type': 'list', 'required': True},
                    'max_episode_steps': {'type': 'integer', 'required': True}
                }
            }
        }
    }
}

validator = cerberus.Validator(schema)
if not validator.validate(config):
    print(f"Configuration errors: {validator.errors}")
```

### Issue 3: Memory Allocation Errors

**Symptoms:**
```bash
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size:**
```python
config['training']['batch_size'] = 16  # Reduce from 64
config['training']['parallel_envs'] = 2  # Reduce from 4
```

2. **Use CPU-only mode:**
```python
import torch
torch.cuda.set_device(-1)  # Force CPU
```

3. **Clear GPU memory:**
```python
import torch
torch.cuda.empty_cache()
```

## Runtime Errors

### Issue 1: Agent Iteration Errors

**Symptoms:**
```bash
StopIteration: No more agents to iterate
```

**Solutions:**

1. **Proper agent iteration:**
```python
env.reset()
for agent in env.agent_iter():
    try:
        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            action = None
        else:
            action = agent_policy(observation)
        
        env.step(action)
    except StopIteration:
        break
```

2. **Check environment state:**
```python
if env.agents:  # Check if agents exist
    for agent in env.agent_iter():
        # Process agent
        pass
else:
    print("No agents available")
```

3. **Handle environment termination:**
```python
env.reset()
while env.agents:  # Continue while agents exist
    for agent in env.agent_iter():
        if agent in env.agents:  # Check agent still active
            # Process agent
            pass
```

### Issue 2: Observation Shape Mismatch

**Symptoms:**
```bash
ValueError: Expected observation shape (48, 13), got (47, 13)
```

**Solutions:**

1. **Validate observation shape:**
```python
def validate_observation(observation, expected_shape):
    if observation.shape != expected_shape:
        raise ValueError(f"Shape mismatch: expected {expected_shape}, got {observation.shape}")
    return observation
```

2. **Reshape observation:**
```python
import numpy as np

def fix_observation_shape(observation, target_shape):
    if observation.shape[0] < target_shape[0]:
        # Pad with zeros
        padding = np.zeros((target_shape[0] - observation.shape[0], observation.shape[1]))
        observation = np.vstack([observation, padding])
    elif observation.shape[0] > target_shape[0]:
        # Truncate
        observation = observation[:target_shape[0]]
    return observation
```

3. **Debug observation creation:**
```python
class DebugEnvironment(StrategicMarketEnv):
    def observe(self, agent):
        observation = super().observe(agent)
        print(f"Agent {agent} observation shape: {observation.shape}")
        return observation
```

### Issue 3: Action Space Violations

**Symptoms:**
```bash
AssertionError: Action 3 not in action space Discrete(3)
```

**Solutions:**

1. **Validate actions:**
```python
def validate_action(action, action_space):
    if not action_space.contains(action):
        raise ValueError(f"Action {action} not valid for space {action_space}")
    return action
```

2. **Clip actions to valid range:**
```python
def clip_action(action, action_space):
    if hasattr(action_space, 'n'):  # Discrete space
        return max(0, min(action, action_space.n - 1))
    else:  # Box space
        return np.clip(action, action_space.low, action_space.high)
```

3. **Use safe action selection:**
```python
def safe_action_selection(agent_output, action_space):
    if hasattr(action_space, 'n'):  # Discrete
        probabilities = torch.softmax(agent_output, dim=-1)
        action = torch.multinomial(probabilities, 1).item()
        return min(action, action_space.n - 1)
    else:  # Continuous
        return np.clip(agent_output, action_space.low, action_space.high)
```

## Performance Issues

### Issue 1: Slow Environment Steps

**Symptoms:**
- Environment steps taking >10ms
- Training significantly slower than expected

**Solutions:**

1. **Profile environment performance:**
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run environment
env.reset()
for _ in range(1000):
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        action = env.action_space(agent).sample()
        env.step(action)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

2. **Optimize observation computation:**
```python
from functools import lru_cache
import numpy as np

class OptimizedEnvironment(StrategicMarketEnv):
    @lru_cache(maxsize=128)
    def _compute_technical_indicators(self, price_tuple):
        # Cache expensive calculations
        return self._calculate_indicators(np.array(price_tuple))
```

3. **Use vectorized operations:**
```python
# Instead of loops
rewards = []
for agent in agents:
    reward = calculate_reward(agent)
    rewards.append(reward)

# Use vectorized computation
rewards = np.array([calculate_reward(agent) for agent in agents])
```

### Issue 2: Memory Leaks

**Symptoms:**
- Memory usage increasing over time
- Out of memory errors during long training

**Solutions:**

1. **Clear environment history:**
```python
class MemoryEfficientEnvironment(StrategicMarketEnv):
    def reset(self):
        # Clear observation history
        if hasattr(self, '_observation_history'):
            self._observation_history.clear()
        
        # Clear reward history
        if hasattr(self, '_reward_history'):
            self._reward_history.clear()
        
        return super().reset()
```

2. **Use memory monitoring:**
```python
import psutil
import gc

def monitor_memory():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.2f}MB")
    
    # Force garbage collection
    gc.collect()
```

3. **Limit buffer sizes:**
```python
from collections import deque

class BoundedEnvironment(StrategicMarketEnv):
    def __init__(self, config):
        super().__init__(config)
        self.observation_buffer = deque(maxlen=1000)
        self.reward_buffer = deque(maxlen=1000)
```

### Issue 3: GPU Memory Issues

**Symptoms:**
```bash
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Monitor GPU memory:**
```python
import torch

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB allocated")
        print(f"GPU Memory: {torch.cuda.memory_reserved()/1024**2:.2f}MB reserved")
```

2. **Use gradient checkpointing:**
```python
import torch

class MemoryEfficientAgent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.use_checkpoint = True
    
    def forward(self, x):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x)
        return self._forward(x)
```

3. **Clear GPU cache:**
```python
import torch

def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
```

## Training Problems

### Issue 1: Agents Not Learning

**Symptoms:**
- Rewards not improving over time
- Agent actions remain random

**Solutions:**

1. **Check reward scaling:**
```python
def analyze_rewards(rewards):
    import numpy as np
    print(f"Reward statistics:")
    print(f"  Mean: {np.mean(rewards):.4f}")
    print(f"  Std: {np.std(rewards):.4f}")
    print(f"  Min: {np.min(rewards):.4f}")
    print(f"  Max: {np.max(rewards):.4f}")
    
    # Check for sparse rewards
    nonzero_rewards = np.count_nonzero(rewards)
    print(f"  Non-zero rewards: {nonzero_rewards}/{len(rewards)} ({nonzero_rewards/len(rewards)*100:.1f}%)")
```

2. **Verify gradient flow:**
```python
def check_gradients(model):
    total_norm = 0
    param_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            print(f"{name}: {param_norm:.6f}")
    
    total_norm = total_norm ** (1. / 2)
    print(f"Total gradient norm: {total_norm:.6f}")
    print(f"Parameters with gradients: {param_count}")
```

3. **Debug action distribution:**
```python
from collections import Counter

def analyze_actions(actions):
    action_counts = Counter(actions)
    total_actions = len(actions)
    
    print("Action distribution:")
    for action, count in sorted(action_counts.items()):
        percentage = count / total_actions * 100
        print(f"  Action {action}: {count}/{total_actions} ({percentage:.1f}%)")
```

### Issue 2: Training Instability

**Symptoms:**
- Loss values fluctuating wildly
- Training crashes with NaN values

**Solutions:**

1. **Add gradient clipping:**
```python
import torch.nn.utils as utils

def train_step(model, optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    
    # Clip gradients
    utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
```

2. **Use stable loss functions:**
```python
import torch
import torch.nn.functional as F

def stable_cross_entropy(logits, targets):
    # Use log_softmax for numerical stability
    log_probs = F.log_softmax(logits, dim=-1)
    return F.nll_loss(log_probs, targets)
```

3. **Monitor for NaN values:**
```python
def check_for_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        return True
    return False
```

### Issue 3: Slow Training Convergence

**Symptoms:**
- Training takes much longer than expected
- Minimal improvement after many episodes

**Solutions:**

1. **Adjust learning rate:**
```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=100, factor=0.5)

# In training loop
scheduler.step(mean_reward)
```

2. **Use experience replay:**
```python
from collections import deque
import random

class ExperienceReplay:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
```

3. **Implement curriculum learning:**
```python
class CurriculumScheduler:
    def __init__(self, initial_difficulty=0.1, max_difficulty=1.0):
        self.current_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
    
    def step(self, performance):
        if performance > 0.8:  # Good performance
            self.current_difficulty = min(
                self.current_difficulty * 1.1,
                self.max_difficulty
            )
```

## Integration Issues

### Issue 1: Ray RLlib Integration Problems

**Symptoms:**
```bash
TypeError: 'StrategicMarketEnv' object is not callable
```

**Solutions:**

1. **Proper environment creation:**
```python
from ray.rllib.env import PettingZooEnv

def env_creator(config):
    return PettingZooEnv(StrategicMarketEnv(config))

# Register environment
from ray import tune
tune.register_env("strategic_marl", env_creator)
```

2. **Fix observation space issues:**
```python
class RLlibCompatibleEnv(StrategicMarketEnv):
    def __init__(self, config):
        super().__init__(config)
        # Ensure observation spaces are properly defined
        for agent in self.possible_agents:
            if agent not in self.observation_spaces:
                self.observation_spaces[agent] = self.observation_space(agent)
```

3. **Handle multi-agent policies:**
```python
config = {
    "multiagent": {
        "policies": {
            agent_id: (None, obs_space, act_space, {})
            for agent_id, (obs_space, act_space) in zip(
                env.possible_agents,
                [(env.observation_space(agent), env.action_space(agent)) 
                 for agent in env.possible_agents]
            )
        },
        "policy_mapping_fn": lambda agent_id: agent_id,
    }
}
```

### Issue 2: Stable Baselines3 Integration

**Symptoms:**
```bash
AttributeError: 'StrategicMarketEnv' object has no attribute 'num_envs'
```

**Solutions:**

1. **Use proper wrappers:**
```python
from stable_baselines3.common.env_util import make_vec_env
from pettingzoo.utils.conversions import parallel_wrapper_fn

def make_env():
    return StrategicMarketEnv(config)

env_fn = parallel_wrapper_fn(make_env)
vec_env = make_vec_env(env_fn, n_envs=4)
```

2. **Single-agent wrapper:**
```python
from pettingzoo.utils.conversions import aec_to_parallel
from stable_baselines3.common.env_util import make_vec_env

def create_single_agent_env():
    env = StrategicMarketEnv(config)
    # Convert to single-agent if needed
    return env

vec_env = make_vec_env(create_single_agent_env, n_envs=1)
```

### Issue 3: Custom Training Loop Issues

**Symptoms:**
- Environment not resetting properly
- Agent states not updating correctly

**Solutions:**

1. **Proper reset handling:**
```python
def training_loop(env, agents, num_episodes):
    for episode in range(num_episodes):
        env.reset()
        
        # Clear agent states
        for agent in agents.values():
            agent.reset_episode()
        
        # Episode loop
        for agent_name in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                action = None
            else:
                action = agents[agent_name].select_action(observation)
            
            env.step(action)
```

2. **State synchronization:**
```python
class StatefulAgent:
    def __init__(self):
        self.episode_rewards = []
        self.episode_actions = []
    
    def reset_episode(self):
        self.episode_rewards.clear()
        self.episode_actions.clear()
    
    def update_state(self, reward, action):
        self.episode_rewards.append(reward)
        self.episode_actions.append(action)
```

## API Compliance

### Issue 1: PettingZoo API Test Failures

**Symptoms:**
```bash
AssertionError: Environment failed API test
```

**Solutions:**

1. **Run API test with debugging:**
```python
from pettingzoo.test import api_test

try:
    api_test(env, num_cycles=10)
    print("API test passed")
except AssertionError as e:
    print(f"API test failed: {e}")
    # Debug the specific issue
```

2. **Check required methods:**
```python
def check_required_methods(env):
    required_methods = ['reset', 'step', 'observe', 'render', 'close']
    missing_methods = []
    
    for method in required_methods:
        if not hasattr(env, method):
            missing_methods.append(method)
    
    if missing_methods:
        print(f"Missing methods: {missing_methods}")
        return False
    return True
```

3. **Validate properties:**
```python
def validate_properties(env):
    required_properties = ['agents', 'possible_agents', 'observation_spaces', 'action_spaces']
    
    for prop in required_properties:
        if not hasattr(env, prop):
            print(f"Missing property: {prop}")
            return False
        
        value = getattr(env, prop)
        if value is None:
            print(f"Property {prop} is None")
            return False
    
    return True
```

### Issue 2: Observation Space Inconsistencies

**Symptoms:**
```bash
AssertionError: Observation space mismatch
```

**Solutions:**

1. **Standardize observation spaces:**
```python
from gymnasium.spaces import Box
import numpy as np

def create_observation_space(shape):
    return Box(
        low=-np.inf,
        high=np.inf,
        shape=shape,
        dtype=np.float32
    )
```

2. **Validate observations:**
```python
def validate_observation(observation, observation_space):
    if not observation_space.contains(observation):
        print(f"Observation {observation} not in space {observation_space}")
        return False
    return True
```

## Debugging Tools

### Environment Debugger

```python
class EnvironmentDebugger:
    def __init__(self, env):
        self.env = env
        self.step_count = 0
        self.episode_count = 0
        self.debug_log = []
    
    def debug_step(self, agent, action):
        self.step_count += 1
        
        # Log step information
        observation, reward, termination, truncation, info = self.env.last()
        
        debug_info = {
            'step': self.step_count,
            'episode': self.episode_count,
            'agent': agent,
            'action': action,
            'observation_shape': observation.shape,
            'reward': reward,
            'termination': termination,
            'truncation': truncation,
            'info': info
        }
        
        self.debug_log.append(debug_info)
        
        # Print debug info
        print(f"Step {self.step_count}: Agent {agent}, Action {action}, Reward {reward}")
        
        return debug_info
    
    def debug_episode(self):
        self.episode_count += 1
        self.step_count = 0
        print(f"Starting Episode {self.episode_count}")
    
    def save_debug_log(self, filename):
        import json
        with open(filename, 'w') as f:
            json.dump(self.debug_log, f, indent=2)
```

### Performance Profiler

```python
import time
from contextlib import contextmanager

class PerformanceProfiler:
    def __init__(self):
        self.timings = {}
    
    @contextmanager
    def profile(self, name):
        start_time = time.time()
        yield
        end_time = time.time()
        
        if name not in self.timings:
            self.timings[name] = []
        
        self.timings[name].append(end_time - start_time)
    
    def get_stats(self):
        stats = {}
        for name, times in self.timings.items():
            stats[name] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'count': len(times)
            }
        return stats
    
    def print_stats(self):
        stats = self.get_stats()
        print("Performance Statistics:")
        print("-" * 50)
        for name, stat in stats.items():
            print(f"{name}:")
            print(f"  Mean: {stat['mean']*1000:.2f}ms")
            print(f"  Std:  {stat['std']*1000:.2f}ms")
            print(f"  Min:  {stat['min']*1000:.2f}ms")
            print(f"  Max:  {stat['max']*1000:.2f}ms")
            print(f"  Count: {stat['count']}")
            print()

# Usage
profiler = PerformanceProfiler()

with profiler.profile("environment_step"):
    env.step(action)

with profiler.profile("agent_decision"):
    action = agent.select_action(observation)

profiler.print_stats()
```

### Memory Monitor

```python
import psutil
import gc
import torch

class MemoryMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.memory_log = []
    
    def log_memory(self, stage):
        memory_info = self.process.memory_info()
        
        log_entry = {
            'stage': stage,
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': self.process.memory_percent()
        }
        
        if torch.cuda.is_available():
            log_entry['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            log_entry['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        
        self.memory_log.append(log_entry)
        
        print(f"Memory at {stage}: {log_entry['rss_mb']:.2f}MB RSS, {log_entry['percent']:.1f}%")
    
    def check_memory_leak(self, threshold_mb=100):
        if len(self.memory_log) < 2:
            return False
        
        initial_memory = self.memory_log[0]['rss_mb']
        current_memory = self.memory_log[-1]['rss_mb']
        
        if current_memory - initial_memory > threshold_mb:
            print(f"Potential memory leak detected: {current_memory - initial_memory:.2f}MB increase")
            return True
        
        return False
    
    def cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

## Getting Help

### Community Resources

- **PettingZoo Documentation**: https://pettingzoo.farama.org/
- **PettingZoo GitHub**: https://github.com/Farama-Foundation/PettingZoo
- **Gymnasium Documentation**: https://gymnasium.farama.org/
- **GrandModel Issues**: https://github.com/Afeks214/GrandModel/issues

### Debug Information to Include

When reporting issues, include:

1. **Environment Information**:
```bash
python -c "
import pettingzoo, gymnasium, torch, numpy
print(f'PettingZoo: {pettingzoo.__version__}')
print(f'Gymnasium: {gymnasium.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'NumPy: {numpy.__version__}')
"
```

2. **Error Traceback**:
```python
import traceback
try:
    # Your code that fails
    pass
except Exception as e:
    traceback.print_exc()
```

3. **Minimal Reproducible Example**:
```python
# Simplest possible code that reproduces the issue
from src.environment.strategic_env import StrategicMarketEnv

config = {
    'strategic_marl': {
        'environment': {
            'matrix_shape': [48, 13],
            'max_episode_steps': 10
        }
    }
}

env = StrategicMarketEnv(config)
env.reset()
# ... code that fails
```

4. **System Information**:
```bash
uname -a
python --version
pip list | grep -E "pettingzoo|gymnasium|torch"
```

This troubleshooting guide should help resolve most common issues with PettingZoo environments in GrandModel. For issues not covered here, please consult the community resources or open a GitHub issue with detailed information about the problem.