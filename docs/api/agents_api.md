# MARL Agents API Documentation

## Overview

The GrandModel Multi-Agent Reinforcement Learning (MARL) system consists of specialized agents that collaborate to make intelligent trading decisions. Each agent has specific responsibilities and operates within a coordinated framework to optimize trading performance.

## Table of Contents

- [Agent Architecture](#agent-architecture)
- [Strategic MARL Component](#strategic-marl-component)
- [Agent Types](#agent-types)
- [Risk Management Integration](#risk-management-integration)
- [Training and Learning](#training-and-learning)
- [API Reference](#api-reference)
- [Examples](#examples)

## Agent Architecture

### Multi-Agent Framework

```
┌─────────────────────────────────────────────────────┐
│                MARL System                          │
├─────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ Strategic   │  │ Tactical    │  │ Risk        │ │
│  │ Agent       │  │ Agent       │  │ Agent       │ │
│  │             │  │             │  │             │ │
│  │ 30min       │  │ 5min        │  │ Portfolio   │ │
│  │ decisions   │  │ execution   │  │ protection  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────┤
│              Coordination Layer                     │
│  ┌─────────────────────────────────────────────────┐ │
│  │         Main MARL Core                          │ │
│  │   • Agent coordination                          │ │
│  │   • Reward distribution                         │ │
│  │   • Experience sharing                          │ │
│  └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

### Agent Communication

Agents communicate through:
1. **Shared Environment State**: Common market observations
2. **Action Coordination**: Synchronized decision making
3. **Experience Sharing**: Learning from collective outcomes
4. **Reward Propagation**: Aligned incentive structures

## Strategic MARL Component

### Class: StrategicMARLComponent

The main orchestrator for all MARL agents in the system.

```python
class StrategicMARLComponent:
    def __init__(self, name: str, kernel: AlgoSpaceKernel)
    def initialize(self) -> bool
    def _handle_synergy_detected(self, event: Event) -> None
    def _handle_new_30min_bar(self, event: Event) -> None
    def get_strategic_decision(self, market_state: Dict) -> Dict
    def update_learning(self, outcome: Dict) -> None
```

### Initialization

```python
def initialize(self) -> bool:
    """
    Initialize the MARL environment and agents.
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
```

**Initialization Process:**
1. Load pre-trained models if available
2. Initialize PettingZoo MARL environment
3. Set up agent observation/action spaces
4. Configure reward functions
5. Establish inter-agent communication

**Example:**
```python
strategic_marl = StrategicMARLComponent("StrategicMARL", kernel)
if strategic_marl.initialize():
    print("MARL system ready")
else:
    print("MARL initialization failed")
```

### Event Handling

#### Synergy Detection Handler

```python
def _handle_synergy_detected(self, event: Event) -> None:
    """
    Process synergy detection events and make strategic decisions.
    
    Args:
        event: Event containing synergy detection results
    """
```

**Process:**
1. Extract synergy pattern information
2. Update agent observations
3. Query agents for coordinated action
4. Publish strategic decision if consensus reached

**Example Event Payload:**
```python
{
    "pattern": "TYPE_1",
    "confidence": 0.85,
    "timeframe": "30m",
    "features": {
        "mlmi_signal": 1.0,
        "nwrqk_slope": 0.75,
        "market_regime": "trending"
    }
}
```

#### Market Data Handler

```python
def _handle_new_30min_bar(self, event: Event) -> None:
    """
    Process new 30-minute bar data for strategic updates.
    
    Args:
        event: Event containing new bar data
    """
```

**Process:**
1. Update market state representations
2. Recalculate agent observations
3. Trigger periodic strategy review
4. Update risk assessments

### Strategic Decision Making

```python
def get_strategic_decision(self, market_state: Dict) -> Dict:
    """
    Get coordinated decision from all strategic agents.
    
    Args:
        market_state: Current market observations
        
    Returns:
        Dict: Strategic decision with confidence levels
    """
```

**Decision Process:**
1. **Observation Formation**: Convert market state to agent observations
2. **Individual Decisions**: Each agent processes observations independently
3. **Coordination Phase**: Agents negotiate and align decisions
4. **Consensus Building**: Reach agreement or default to conservative action
5. **Decision Output**: Unified strategic direction

**Return Format:**
```python
{
    "action": "long",  # or "short", "hold"
    "confidence": 0.78,
    "agents": {
        "strategic_agent": {"action": "long", "confidence": 0.82},
        "risk_agent": {"action": "long", "confidence": 0.75},
        "tactical_agent": {"action": "long", "confidence": 0.77}
    },
    "reasoning": {
        "primary_signal": "synergy_pattern_TYPE_1",
        "supporting_signals": ["momentum_confirmation", "regime_alignment"],
        "risk_factors": ["moderate_volatility"]
    }
}
```

## Agent Types

### Strategic Agent

**Responsibility**: Long-term market direction and position sizing
**Timeframe**: 30-minute bars
**Observations**: Market regime, trend strength, volatility patterns
**Actions**: Direction (long/short/neutral), conviction level

```python
class StrategicAgent:
    observation_space: gym.Space  # Market regime features
    action_space: gym.Space       # Direction + conviction
    
    def step(self, observation: np.ndarray) -> Tuple[int, float]:
        """
        Args:
            observation: Market state features (regime, trends, etc.)
        Returns:
            action: Direction decision (0=short, 1=hold, 2=long)
            conviction: Confidence level (0.0 to 1.0)
        """
```

**Observation Components:**
- Market regime classification (trending/ranging/volatile)
- Long-term momentum indicators
- Volatility regime assessment
- Economic cycle position
- Sector rotation signals

### Tactical Agent

**Responsibility**: Entry/exit timing and execution optimization
**Timeframe**: 5-minute bars
**Observations**: Price action, volume patterns, order flow
**Actions**: Entry timing, stop placement, position scaling

```python
class TacticalAgent:
    observation_space: gym.Space  # Price action features
    action_space: gym.Space       # Timing + scaling
    
    def step(self, observation: np.ndarray) -> Tuple[int, float, float]:
        """
        Args:
            observation: Price action and microstructure features
        Returns:
            timing: Entry timing signal (0=wait, 1=enter_now, 2=scale_in)
            stop_distance: Stop loss distance as % of price
            scale_factor: Position scaling (0.5=half size, 1.0=full, 1.5=1.5x)
        """
```

**Observation Components:**
- Price action patterns (flags, breakouts, reversals)
- Volume profile analysis
- Fair value gap detection
- Order flow imbalance
- Support/resistance levels

### Risk Agent

**Responsibility**: Portfolio protection and risk management
**Timeframe**: Real-time
**Observations**: Portfolio metrics, correlation, drawdown
**Actions**: Risk adjustments, emergency stops, correlation limits

```python
class RiskAgent:
    observation_space: gym.Space  # Portfolio risk metrics
    action_space: gym.Space       # Risk adjustments
    
    def step(self, observation: np.ndarray) -> Tuple[float, float, bool]:
        """
        Args:
            observation: Portfolio and market risk features
        Returns:
            position_sizing: Kelly-adjusted position size multiplier
            max_drawdown: Maximum acceptable drawdown threshold
            emergency_stop: Emergency stop trigger (True/False)
        """
```

**Observation Components:**
- Current portfolio exposure
- Correlation matrix analysis
- VaR and expected shortfall
- Drawdown progression
- Market stress indicators

## Risk Management Integration

### Kelly Criterion Integration

The MARL system integrates with Kelly Criterion calculations for optimal position sizing:

```python
def calculate_marl_kelly_size(
    self, 
    strategic_decision: Dict, 
    market_conditions: Dict
) -> float:
    """
    Calculate Kelly-optimal position size adjusted by MARL confidence.
    
    Args:
        strategic_decision: MARL agent decision output
        market_conditions: Current market state
        
    Returns:
        float: Optimal position size as fraction of capital
    """
    
    # Base Kelly calculation
    win_prob = strategic_decision["confidence"]
    win_loss_ratio = self._estimate_win_loss_ratio(market_conditions)
    kelly_fraction = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
    
    # MARL risk adjustment
    risk_agent_confidence = strategic_decision["agents"]["risk_agent"]["confidence"]
    marl_risk_multiplier = min(risk_agent_confidence, 0.25)  # Cap at 25% of capital
    
    return kelly_fraction * marl_risk_multiplier
```

### VaR Integration

```python
def update_var_with_marl_decision(
    self, 
    current_var: float, 
    strategic_decision: Dict
) -> float:
    """
    Update VaR calculation considering MARL decision impact.
    
    Args:
        current_var: Current portfolio VaR
        strategic_decision: MARL strategic decision
        
    Returns:
        float: Updated VaR estimate
    """
    
    # Estimate decision impact on portfolio risk
    decision_direction = strategic_decision["action"]
    decision_confidence = strategic_decision["confidence"]
    
    # Adjust VaR based on new position
    if decision_direction in ["long", "short"]:
        # Increase VaR due to additional exposure
        exposure_increase = decision_confidence * self.max_position_size
        var_adjustment = exposure_increase * self.market_volatility
        return current_var + var_adjustment
    
    return current_var
```

## Training and Learning

### Experience Replay

```python
class MARLExperienceBuffer:
    """
    Stores experiences for multi-agent learning.
    """
    
    def add_experience(
        self,
        state: Dict,
        actions: Dict[str, Any],
        rewards: Dict[str, float],
        next_state: Dict,
        done: bool
    ) -> None:
        """
        Add experience tuple for all agents.
        
        Args:
            state: Environment state observed by all agents
            actions: Actions taken by each agent
            rewards: Rewards received by each agent
            next_state: Resulting environment state
            done: Whether episode terminated
        """
```

### Reward Function

```python
def calculate_agent_rewards(
    self,
    trade_outcome: Dict,
    market_state: Dict
) -> Dict[str, float]:
    """
    Calculate individual agent rewards based on trade outcome.
    
    Args:
        trade_outcome: Trade result information
        market_state: Market conditions during trade
        
    Returns:
        Dict[str, float]: Reward for each agent
    """
    
    base_reward = trade_outcome["pnl"] / trade_outcome["risk_amount"]
    
    return {
        "strategic_agent": base_reward * self._strategic_reward_weight(trade_outcome),
        "tactical_agent": base_reward * self._tactical_reward_weight(trade_outcome),
        "risk_agent": base_reward * self._risk_reward_weight(trade_outcome)
    }
```

### Model Updates

```python
def update_agent_models(self, batch_size: int = 32) -> Dict[str, float]:
    """
    Update agent neural networks using experience replay.
    
    Args:
        batch_size: Size of experience batch for training
        
    Returns:
        Dict[str, float]: Training losses for each agent
    """
    
    if len(self.experience_buffer) < batch_size:
        return {}
    
    # Sample experience batch
    experiences = self.experience_buffer.sample(batch_size)
    
    losses = {}
    for agent_name, agent in self.agents.items():
        # Extract agent-specific data
        states = [exp["state"][agent_name] for exp in experiences]
        actions = [exp["actions"][agent_name] for exp in experiences]
        rewards = [exp["rewards"][agent_name] for exp in experiences]
        next_states = [exp["next_state"][agent_name] for exp in experiences]
        
        # Update agent
        loss = agent.update(states, actions, rewards, next_states)
        losses[agent_name] = loss
    
    return losses
```

## API Reference

### Environment Interface

```python
class MARLTradingEnvironment:
    """PettingZoo-compatible MARL trading environment"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize trading environment with configuration"""
    
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment to initial state"""
    
    def step(self, actions: Dict[str, Any]) -> Tuple[
        Dict[str, np.ndarray],  # observations
        Dict[str, float],       # rewards
        Dict[str, bool],        # done flags
        Dict[str, Dict]         # info
    ]:
        """Execute one step of the environment"""
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render environment state"""
    
    def close(self) -> None:
        """Clean up environment resources"""
```

### Agent Interface

```python
class MARLAgent:
    """Base class for MARL agents"""
    
    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        """Initialize agent with observation and action spaces"""
    
    def select_action(self, observation: np.ndarray, training: bool = False) -> Any:
        """Select action given observation"""
    
    def update(
        self, 
        states: List[np.ndarray], 
        actions: List[Any], 
        rewards: List[float], 
        next_states: List[np.ndarray]
    ) -> float:
        """Update agent parameters and return training loss"""
    
    def save_model(self, path: str) -> None:
        """Save agent model to disk"""
    
    def load_model(self, path: str) -> None:
        """Load agent model from disk"""
```

## Examples

### Basic MARL Usage

```python
from src.agents.strategic_marl_component import StrategicMARLComponent
from src.core.kernel import AlgoSpaceKernel

# Initialize kernel and MARL component
kernel = AlgoSpaceKernel("config/marl_config.yaml")
kernel.initialize()

strategic_marl = kernel.get_component("strategic_marl")

# Process market data
market_state = {
    "price": 4250.75,
    "volume": 1500,
    "indicators": {
        "rsi": 65.5,
        "macd": 0.25
    },
    "regime": "trending"
}

# Get strategic decision
decision = strategic_marl.get_strategic_decision(market_state)
print(f"MARL Decision: {decision['action']} with {decision['confidence']:.2f} confidence")
```

### Custom Agent Implementation

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Any, List

class CustomStrategicAgent(MARLAgent):
    """Custom strategic agent implementation"""
    
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        
        # Neural network architecture
        self.network = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_space.n)
        )
        
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
    
    def select_action(self, observation: np.ndarray, training: bool = False) -> int:
        """Select action using epsilon-greedy strategy"""
        if training and np.random.random() < self.epsilon:
            return np.random.choice(self.action_space.n)
        
        with torch.no_grad():
            q_values = self.network(torch.FloatTensor(observation))
            return q_values.argmax().item()
    
    def update(
        self, 
        states: List[np.ndarray], 
        actions: List[int], 
        rewards: List[float], 
        next_states: List[np.ndarray]
    ) -> float:
        """Update network using Q-learning"""
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)
        
        # Current Q values
        current_q_values = self.network(states_tensor).gather(1, actions_tensor.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.network(next_states_tensor).max(1)[0]
            target_q_values = rewards_tensor + self.gamma * next_q_values
        
        # Calculate loss
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

### Training Loop

```python
def train_marl_agents(
    strategic_marl: StrategicMARLComponent,
    training_data: List[Dict],
    episodes: int = 1000
) -> Dict[str, List[float]]:
    """
    Train MARL agents using historical data.
    
    Args:
        strategic_marl: MARL component to train
        training_data: Historical market data
        episodes: Number of training episodes
        
    Returns:
        Dict[str, List[float]]: Training losses for each agent
    """
    
    training_losses = {agent: [] for agent in strategic_marl.agents.keys()}
    
    for episode in range(episodes):
        # Reset environment
        state = strategic_marl.env.reset()
        episode_reward = 0
        
        for step, market_data in enumerate(training_data):
            # Get actions from all agents
            actions = {}
            for agent_name, agent in strategic_marl.agents.items():
                actions[agent_name] = agent.select_action(
                    state[agent_name], 
                    training=True
                )
            
            # Step environment
            next_state, rewards, done, info = strategic_marl.env.step(actions)
            
            # Store experience
            strategic_marl.experience_buffer.add_experience(
                state, actions, rewards, next_state, done
            )
            
            # Update agents
            if step % 32 == 0:  # Update every 32 steps
                losses = strategic_marl.update_agent_models(batch_size=32)
                for agent_name, loss in losses.items():
                    training_losses[agent_name].append(loss)
            
            state = next_state
            episode_reward += sum(rewards.values())
            
            if done:
                break
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {episode_reward:.2f}")
    
    return training_losses
```

### Performance Monitoring

```python
class MARLPerformanceMonitor:
    """Monitor MARL agent performance"""
    
    def __init__(self):
        self.metrics = {
            "decision_accuracy": [],
            "average_reward": [],
            "convergence_speed": [],
            "coordination_efficiency": []
        }
    
    def update_metrics(self, decisions: Dict, outcomes: Dict) -> None:
        """Update performance metrics"""
        
        # Calculate decision accuracy
        correct_predictions = sum(
            1 for decision, outcome in zip(decisions.values(), outcomes.values())
            if self._is_correct_prediction(decision, outcome)
        )
        accuracy = correct_predictions / len(decisions)
        self.metrics["decision_accuracy"].append(accuracy)
        
        # Calculate average reward
        avg_reward = np.mean(list(outcomes.values()))
        self.metrics["average_reward"].append(avg_reward)
    
    def get_performance_report(self) -> Dict[str, float]:
        """Generate performance report"""
        return {
            metric: np.mean(values[-100:])  # Last 100 observations
            for metric, values in self.metrics.items()
            if values
        }
```

## Related Documentation

- [System Architecture](../architecture/system_overview.md)
- [Training Guide](../guides/training_guide.md)
- [Risk Management API](risk_management_api.md)
- [Performance Optimization](../guides/performance_guide.md)