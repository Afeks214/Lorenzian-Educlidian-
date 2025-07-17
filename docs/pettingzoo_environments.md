# PettingZoo Environments Documentation

## Overview

GrandModel implements a comprehensive suite of PettingZoo environments designed specifically for multi-agent reinforcement learning in financial markets. These environments provide standardized, high-performance platforms for training and deploying MARL agents with proper API compliance and production-ready features.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Environment Specifications](#environment-specifications)
- [Installation and Setup](#installation-and-setup)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [Training Integration](#training-integration)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

## Architecture Overview

### PettingZoo Integration

All GrandModel environments inherit from `pettingzoo.AECEnv` and follow the **Agent-Environment-Cycle (AEC)** pattern:

```
┌─────────────────────────────────────────────────────────────┐
│                    PettingZoo AEC Pattern                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────┐    ┌─────────────┐    ┌───────────────┐    │
│  │   Agent   │ -> │ Environment │ -> │   Observation │    │
│  │  Action   │    │    Step     │    │   + Reward    │    │
│  └───────────┘    └─────────────┘    └───────────────┘    │
│       ^                                       |           │
│       |                                       v           │
│  ┌───────────┐    ┌─────────────┐    ┌───────────────┐    │
│  │   Next    │ <- │    Agent    │ <- │   Decision    │    │
│  │   Agent   │    │ Selection   │    │   Process     │    │
│  └───────────┘    └─────────────┘    └───────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Environment Hierarchy

```
GrandModel MARL Environments
├── Strategic Environment (Long-term Strategy)
│   ├── MLMI Expert Agent
│   ├── NWRQK Expert Agent
│   └── Regime Detection Agent
├── Tactical Environment (Short-term Execution)
│   ├── FVG Agent
│   ├── Momentum Agent
│   └── Entry Optimization Agent
├── Risk Management Environment
│   ├── Position Sizing Agent
│   ├── Stop/Target Agent
│   ├── Risk Monitor Agent
│   └── Portfolio Optimizer Agent
└── Execution Environment
    ├── Position Sizing Agent
    ├── Stop/Target Agent
    ├── Risk Monitor Agent
    ├── Portfolio Optimizer Agent
    └── Routing Agent
```

## Environment Specifications

### 1. Strategic Market Environment

**File**: `src/environment/strategic_env.py`

**Purpose**: Long-term strategic decision making with multi-expert coordination

**Key Features**:
- 48×13 matrix observations (48 time steps, 13 features)
- 3 parallel expert agents with specialized knowledge
- Synergy detection and regime-aware decision making
- Centralized critic with 83-dimensional state space

**Agents**:
- `mlmi_expert`: Market microstructure and liquidity intelligence
- `nwrqk_expert`: News, weather, and quantitative signals
- `regime_expert`: Market regime detection and adaptation

**Configuration**:
```yaml
strategic_marl:
  environment:
    matrix_shape: [48, 13]
    max_episode_steps: 2000
    reward_scaling: 1.0
    observation_noise: 0.01
    
  agents:
    mlmi_expert:
      observation_columns: [0, 1, 2, 3]
      action_space_size: 3
      expertise_weight: 0.35
      
    nwrqk_expert:
      observation_columns: [4, 5, 6, 7]
      action_space_size: 3
      expertise_weight: 0.35
      
    regime_expert:
      observation_columns: [8, 9, 10, 11, 12]
      action_space_size: 3
      expertise_weight: 0.30
```

**Observation Space**: 
- Type: `Box(low=-np.inf, high=np.inf, shape=(48, 13), dtype=np.float32)`
- Features: MLMI values, NWRQK signals, regime indicators, time features

**Action Space**:
- Type: `Discrete(3)`
- Actions: `[0: SHORT, 1: NEUTRAL, 2: LONG]`

**Reward Structure**:
- Base reward: Profit/Loss from strategic decisions
- Synergy bonus: +0.2 for aligned agent decisions
- Risk penalty: -0.1 × max_drawdown
- Sharpe bonus: +0.1 × sharpe_ratio

### 2. Tactical Market Environment

**File**: `src/environment/tactical_env.py`

**Purpose**: High-frequency tactical execution with state machine coordination

**Key Features**:
- 60×7 matrix observations (60 time steps, 7 features)
- Sequential agent coordination with state machine
- Fair Value Gap (FVG) pattern detection
- Microsecond precision timing optimization

**Agents**:
- `fvg_agent`: Fair Value Gap detection and analysis
- `momentum_agent`: Price momentum and trend evaluation
- `entry_opt_agent`: Entry timing optimization

**State Machine**:
```
AWAITING_FVG -> AWAITING_MOMENTUM -> AWAITING_ENTRY_OPT -> READY_FOR_AGGREGATION
```

**Configuration**:
```yaml
tactical_marl:
  environment:
    matrix_shape: [60, 7]
    max_episode_steps: 1000
    state_machine: true
    byzantine_tolerance: true
    
  agents:
    fvg_agent:
      observation_columns: [0, 1, 2]
      detection_threshold: 0.7
      confidence_threshold: 0.8
      
    momentum_agent:
      observation_columns: [3, 4]
      momentum_window: 14
      trend_threshold: 0.6
      
    entry_opt_agent:
      observation_columns: [5, 6]
      optimization_steps: 10
      timing_precision: 0.001  # 1ms
```

**Observation Space**:
- Type: `Box(low=-np.inf, high=np.inf, shape=(60, 7), dtype=np.float32)`
- Features: FVG patterns, momentum indicators, volume analysis, timing signals

**Action Space**:
- Type: `Discrete(3)`
- Actions: `[0: SHORT, 1: NEUTRAL, 2: LONG]`

**Reward Structure**:
- Execution quality: Based on fill price vs. optimal price
- Timing bonus: +0.15 for optimal entry timing
- State coordination: +0.1 for proper state machine flow
- Risk penalty: -0.2 × execution slippage

### 3. Risk Management Environment

**File**: `src/environment/risk_env.py`

**Purpose**: Comprehensive risk monitoring and portfolio protection

**Key Features**:
- Real-time VaR calculation and correlation tracking
- Emergency risk protocols and position sizing
- Black swan event simulation and response
- Portfolio optimization with risk constraints

**Agents**:
- `position_sizing`: Dynamic position sizing based on Kelly Criterion
- `stop_target`: Stop-loss and take-profit optimization
- `risk_monitor`: Real-time risk assessment and alerts
- `portfolio_optimizer`: Portfolio allocation optimization

**Configuration**:
```yaml
risk_management:
  environment:
    observation_space: 10  # Portfolio risk metrics
    max_episode_steps: 500
    var_confidence: 0.95
    correlation_threshold: 0.5
    
  agents:
    position_sizing:
      kelly_fraction: 0.25
      max_position: 0.1
      risk_free_rate: 0.02
      
    stop_target:
      atr_multiplier: 2.0
      profit_target_ratio: 2.0
      dynamic_adjustment: true
      
    risk_monitor:
      var_threshold: 0.05
      correlation_shock_threshold: 0.3
      emergency_protocols: true
      
    portfolio_optimizer:
      max_weight: 0.4
      correlation_limit: 0.7
      rebalance_threshold: 0.05
```

**Observation Space**:
- Type: `Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)`
- Features: Portfolio metrics, VaR, correlation matrix, drawdown, exposure

**Action Space**:
- Type: `Discrete(5)`
- Actions: `[0: REDUCE_RISK, 1: SLIGHT_REDUCE, 2: MAINTAIN, 3: SLIGHT_INCREASE, 4: INCREASE_RISK]`

**Reward Structure**:
- Risk-adjusted returns: Sharpe ratio maximization
- VaR compliance: +0.1 for staying within VaR limits
- Drawdown penalty: -0.5 × max_drawdown
- Emergency response: +0.3 for proper crisis management

### 4. Execution Environment

**File**: `src/environment/execution_env.py`

**Purpose**: Unified execution system with intelligent order routing

**Key Features**:
- Multi-venue order routing optimization
- Market impact minimization
- Sub-millisecond execution targeting
- Broker integration and performance monitoring

**Agents**:
- `position_sizing`: Position size optimization for execution
- `stop_target`: Stop-loss and take-profit management
- `risk_monitor`: Execution risk monitoring
- `portfolio_optimizer`: Portfolio-level execution optimization
- `routing`: Intelligent order routing

**Configuration**:
```yaml
execution:
  environment:
    max_episode_steps: 200
    execution_venues: ["venue_a", "venue_b", "venue_c"]
    latency_target: 0.001  # 1ms
    
  agents:
    position_sizing:
      min_size: 100
      max_size: 10000
      size_increment: 100
      
    stop_target:
      execution_slippage_tolerance: 0.001
      
    risk_monitor:
      execution_risk_limit: 0.02
      
    portfolio_optimizer:
      execution_cost_weight: 0.3
      
    routing:
      venue_selection_algorithm: "smart_routing"
      market_impact_model: "almgren_chriss"
```

**Observation Space**:
- Type: Variable (depends on market data and venue information)
- Features: Order book depth, venue latency, market impact estimates

**Action Space**:
- Type: `Box` (continuous execution parameters)
- Actions: Execution timing, venue selection, order size, order type

**Reward Structure**:
- Execution quality: Based on implementation shortfall
- Speed bonus: +0.2 for sub-millisecond execution
- Market impact penalty: -0.1 × market_impact_cost
- Venue optimization: +0.1 for optimal venue selection

## Installation and Setup

### Prerequisites

```bash
# Install core dependencies
pip install pettingzoo[classic] gymnasium numpy pandas torch

# Install GrandModel-specific dependencies
pip install -r requirements.txt
```

### Environment Setup

1. **Configure Environment Variables**:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export GRANDMODEL_ENV=development
export PETTINGZOO_ENV_DIR="src/environment"
```

2. **Verify Installation**:
```bash
python -c "
import pettingzoo
from src.environment.strategic_env import StrategicMarketEnv
from src.environment.tactical_env import TacticalMarketEnv
print('✅ PettingZoo environments ready')
"
```

3. **Run Environment Tests**:
```bash
# Basic environment testing
python test_pettingzoo_minimal.py

# Comprehensive validation
python verify_pettingzoo_comprehensive.py

# API compliance check
python scripts/verify_pettingzoo_envs.py
```

## Usage Guide

### Basic Usage Pattern

```python
from src.environment.strategic_env import StrategicMarketEnv
import numpy as np

# 1. Initialize environment
config = {
    'strategic_marl': {
        'environment': {
            'matrix_shape': [48, 13],
            'max_episode_steps': 2000
        }
    }
}

env = StrategicMarketEnv(config)

# 2. Reset environment
env.reset()

# 3. Training loop
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    
    if termination or truncation:
        action = None
    else:
        # Your agent policy here
        action = your_agent_policy(observation)
    
    env.step(action)
```

### Advanced Usage with Custom Agents

```python
from src.environment.tactical_env import TacticalMarketEnv
import torch
import torch.nn as nn

class CustomTacticalAgent(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_space.shape[0] * observation_space.shape[1], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_space.n)
        )
    
    def forward(self, observation):
        x = observation.flatten()
        return self.network(x)

# Initialize environment and agents
tactical_env = TacticalMarketEnv(tactical_config)
agents = {
    'fvg_agent': CustomTacticalAgent(tactical_env.observation_space, tactical_env.action_space),
    'momentum_agent': CustomTacticalAgent(tactical_env.observation_space, tactical_env.action_space),
    'entry_opt_agent': CustomTacticalAgent(tactical_env.observation_space, tactical_env.action_space)
}

# Training loop with custom agents
tactical_env.reset()
for agent_name in tactical_env.agent_iter():
    observation, reward, termination, truncation, info = tactical_env.last()
    
    if termination or truncation:
        action = None
    else:
        with torch.no_grad():
            agent_output = agents[agent_name](torch.FloatTensor(observation))
            action = torch.argmax(agent_output).item()
    
    tactical_env.step(action)
```

### Parallel Environment Usage

```python
from src.environment.strategic_env import StrategicMarketEnv
from pettingzoo.utils import parallel_to_aec

# Create parallel environment
parallel_env = parallel_to_aec(StrategicMarketEnv(config))

# Use with vectorized training
observations = parallel_env.reset()

for step in range(1000):
    actions = {}
    for agent_name, observation in observations.items():
        actions[agent_name] = agent_policies[agent_name](observation)
    
    observations, rewards, dones, infos = parallel_env.step(actions)
    
    if all(dones.values()):
        observations = parallel_env.reset()
```

## API Reference

### Common Environment Methods

All GrandModel environments implement the standard PettingZoo AEC API:

#### Core Methods

```python
def reset(self, seed=None, options=None) -> None:
    """Reset environment to initial state"""
    pass

def step(self, action) -> None:
    """Execute action for current agent"""
    pass

def observe(self, agent: str) -> np.ndarray:
    """Get observation for specified agent"""
    pass

def last(self) -> Tuple[np.ndarray, float, bool, bool, Dict]:
    """Get last observation, reward, termination, truncation, info"""
    pass

def agent_iter(self) -> Iterator[str]:
    """Iterator over agents in turn-based order"""
    pass

def render(self, mode='human') -> None:
    """Render environment state"""
    pass

def close(self) -> None:
    """Clean up environment resources"""
    pass
```

#### Properties

```python
@property
def agents(self) -> List[str]:
    """List of all agents in environment"""
    pass

@property
def possible_agents(self) -> List[str]:
    """List of all possible agents"""
    pass

@property
def observation_spaces(self) -> Dict[str, gym.Space]:
    """Observation space for each agent"""
    pass

@property
def action_spaces(self) -> Dict[str, gym.Space]:
    """Action space for each agent"""
    pass

@property
def agent_selection(self) -> str:
    """Currently selected agent"""
    pass
```

### Environment-Specific Methods

#### Strategic Environment

```python
def get_synergy_score(self) -> float:
    """Get current synergy score between agents"""
    pass

def get_regime_state(self) -> Dict:
    """Get current market regime information"""
    pass

def get_agent_contributions(self) -> Dict[str, float]:
    """Get individual agent contributions to decisions"""
    pass
```

#### Tactical Environment

```python
def get_state_machine_status(self) -> str:
    """Get current state machine status"""
    pass

def get_fvg_patterns(self) -> List[Dict]:
    """Get detected FVG patterns"""
    pass

def get_execution_metrics(self) -> Dict:
    """Get execution quality metrics"""
    pass
```

#### Risk Environment

```python
def get_var_metrics(self) -> Dict:
    """Get current VaR calculations"""
    pass

def get_correlation_matrix(self) -> np.ndarray:
    """Get current correlation matrix"""
    pass

def get_risk_alerts(self) -> List[Dict]:
    """Get active risk alerts"""
    pass
```

#### Execution Environment

```python
def get_execution_venues(self) -> List[str]:
    """Get available execution venues"""
    pass

def get_market_impact_estimate(self, order_size: float) -> float:
    """Estimate market impact for order size"""
    pass

def get_venue_latency(self, venue: str) -> float:
    """Get latency for specific venue"""
    pass
```

## Training Integration

### Integration with Popular MARL Libraries

#### Ray RLlib

```python
from ray.rllib.env import PettingZooEnv
from ray.rllib.agents.ppo import PPOTrainer

# Create RLlib-compatible environment
env = PettingZooEnv(StrategicMarketEnv(config))

# Configure trainer
trainer_config = {
    "env": env,
    "framework": "torch",
    "multiagent": {
        "policies": {
            "mlmi_expert": (None, env.observation_space, env.action_space, {}),
            "nwrqk_expert": (None, env.observation_space, env.action_space, {}),
            "regime_expert": (None, env.observation_space, env.action_space, {})
        },
        "policy_mapping_fn": lambda agent_id: agent_id
    }
}

trainer = PPOTrainer(config=trainer_config)
```

#### Stable Baselines3

```python
from stable_baselines3 import PPO
from pettingzoo.utils import ss_to_sb3

# Convert to Stable Baselines3 format
env = ss_to_sb3(TacticalMarketEnv(config))

# Train with PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

#### CleanRL

```python
from cleanrl.ppo_pettingzoo import train_ppo

# Train with CleanRL
train_ppo(
    env_fn=lambda: StrategicMarketEnv(config),
    total_timesteps=1000000,
    learning_rate=3e-4,
    num_envs=4
)
```

### Custom Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

class MARLTrainer:
    def __init__(self, env, agents, config):
        self.env = env
        self.agents = agents
        self.config = config
        self.optimizers = {
            name: optim.Adam(agent.parameters(), lr=config['learning_rate'])
            for name, agent in agents.items()
        }
        self.loss_fn = nn.CrossEntropyLoss()
    
    def train_episode(self):
        self.env.reset()
        episode_rewards = defaultdict(float)
        episode_losses = defaultdict(float)
        
        for agent_name in self.env.agent_iter():
            observation, reward, termination, truncation, info = self.env.last()
            
            if termination or truncation:
                action = None
            else:
                # Get agent prediction
                agent_output = self.agents[agent_name](torch.FloatTensor(observation))
                action = torch.argmax(agent_output).item()
                
                # Store for training
                if reward != 0:  # Only train on non-zero rewards
                    loss = self.loss_fn(agent_output.unsqueeze(0), torch.tensor([action]))
                    
                    self.optimizers[agent_name].zero_grad()
                    loss.backward()
                    self.optimizers[agent_name].step()
                    
                    episode_losses[agent_name] += loss.item()
                
                episode_rewards[agent_name] += reward
            
            self.env.step(action)
        
        return dict(episode_rewards), dict(episode_losses)
    
    def train(self, num_episodes):
        for episode in range(num_episodes):
            rewards, losses = self.train_episode()
            
            if episode % 100 == 0:
                print(f"Episode {episode}")
                print(f"Rewards: {rewards}")
                print(f"Losses: {losses}")
```

## Performance Optimization

### Memory Optimization

```python
# Use shared memory for observations
import torch.multiprocessing as mp

class OptimizedEnvironment:
    def __init__(self, config):
        self.config = config
        self.shared_memory = mp.Manager().dict()
        self.observation_buffer = mp.Queue(maxsize=1000)
    
    def reset(self):
        # Clear shared memory
        self.shared_memory.clear()
        # Pre-allocate observation arrays
        for agent in self.agents:
            self.shared_memory[f"{agent}_obs"] = np.zeros(self.observation_shape)
```

### Vectorized Operations

```python
# Vectorized reward calculation
def calculate_rewards_vectorized(self, actions, observations):
    """Calculate rewards for all agents simultaneously"""
    actions_tensor = torch.tensor(actions)
    observations_tensor = torch.tensor(observations)
    
    # Vectorized reward computation
    rewards = self.reward_network(
        torch.cat([actions_tensor, observations_tensor], dim=1)
    )
    
    return rewards.numpy()
```

### Caching and Memoization

```python
from functools import lru_cache
import numpy as np

class CachedEnvironment:
    @lru_cache(maxsize=10000)
    def _calculate_technical_indicators(self, price_tuple):
        """Cache expensive technical indicator calculations"""
        prices = np.array(price_tuple)
        # Expensive calculations here
        return indicators
    
    def _get_cached_observation(self, state_hash):
        """Use state hash for observation caching"""
        if state_hash in self.observation_cache:
            return self.observation_cache[state_hash]
        
        observation = self._compute_observation()
        self.observation_cache[state_hash] = observation
        return observation
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ImportError: No module named 'pettingzoo'`

**Solution**:
```bash
pip install pettingzoo[classic] gymnasium
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 2. Environment Initialization Errors

**Problem**: `ValueError: Invalid configuration`

**Solution**:
```python
# Validate configuration before initialization
def validate_config(config):
    required_keys = ['strategic_marl', 'environment', 'matrix_shape']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    return True

# Use validated config
validate_config(config)
env = StrategicMarketEnv(config)
```

#### 3. Agent Iteration Issues

**Problem**: `StopIteration` in agent iterator

**Solution**:
```python
# Proper agent iteration handling
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

#### 4. Memory Issues

**Problem**: `OutOfMemoryError` during training

**Solution**:
```python
# Use gradient checkpointing and memory optimization
import torch

# Enable memory efficient attention
torch.backends.cuda.enable_flash_sdp(True)

# Use gradient checkpointing
class MemoryEfficientAgent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_checkpoint = config.get('use_checkpoint', True)
    
    def forward(self, x):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x)
        return self._forward(x)
```

#### 5. Performance Issues

**Problem**: Slow environment step times

**Solution**:
```python
# Profile and optimize bottlenecks
import cProfile
import pstats

def profile_environment():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run environment steps
    env.reset()
    for _ in range(1000):
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            action = random.choice([0, 1, 2]) if not (termination or truncation) else None
            env.step(action)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
```

### Debug Mode

```python
# Enable debug mode for detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Environment debug configuration
debug_config = {
    'debug': True,
    'verbose_logging': True,
    'performance_monitoring': True,
    'state_validation': True
}

env = StrategicMarketEnv({**config, **debug_config})
```

### Validation Tools

```bash
# Run comprehensive validation
python scripts/validate_pettingzoo_envs.py

# Check API compliance
python -c "
from pettingzoo.test import api_test
from src.environment.strategic_env import StrategicMarketEnv
api_test(StrategicMarketEnv(config), num_cycles=1000)
"

# Performance benchmarking
python scripts/benchmark_pettingzoo_performance.py
```

## Related Documentation

- [Main README](../README.md) - Project overview and quick start
- [Training Guide](guides/training_guide.md) - MARL training instructions
- [Getting Started](guides/getting_started.md) - Installation and setup
- [API Documentation](api/) - Detailed API reference
- [Architecture Overview](architecture/system_overview.md) - System design

## Support

For issues with PettingZoo environments:

1. Check the [troubleshooting section](#troubleshooting) above
2. Run validation scripts to identify issues
3. Review environment logs for error details
4. Consult the PettingZoo documentation for API questions
5. Open a GitHub issue with detailed error information

## Contributing

When contributing to PettingZoo environments:

1. Follow the PettingZoo API standards
2. Add comprehensive tests for new features
3. Update documentation for any changes
4. Ensure performance benchmarks are maintained
5. Test with multiple MARL libraries