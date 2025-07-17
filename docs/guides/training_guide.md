# MARL Agent Training Guide

## Overview

This comprehensive guide covers training Multi-Agent Reinforcement Learning (MARL) agents in the GrandModel system. You'll learn how to train strategic, tactical, and risk agents, optimize their performance, and deploy trained models to production.

## Table of Contents

- [Training Architecture](#training-architecture)
- [Environment Setup](#environment-setup)
- [Training Configuration](#training-configuration)
- [Training Process](#training-process)
- [Model Evaluation](#model-evaluation)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Production Deployment](#production-deployment)
- [Advanced Topics](#advanced-topics)

## Training Architecture

### MARL Training Framework

```
┌─────────────────────────────────────────────────────────┐
│                Training Environment                     │
├─────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────┐ │
│  │   Historical  │  │   Synthetic   │  │  Live Paper │ │
│  │   Market Data │  │  Market Data  │  │   Trading   │ │
│  └───────────────┘  └───────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────┤
│                Training Orchestrator                    │
│  ┌───────────────────────────────────────────────────┐ │
│  │              MARL Environment                     │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │ │
│  │  │ Strategic   │ │ Tactical    │ │ Risk        │ │ │
│  │  │ Agent       │ │ Agent       │ │ Agent       │ │ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ │ │
│  └───────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│               Experience & Learning                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │ Experience  │  │ Model       │  │ Performance │   │
│  │ Replay      │  │ Updates     │  │ Metrics     │   │
│  └─────────────┘  └─────────────┘  └─────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Training Data Sources

1. **Historical Market Data**
   - High-frequency tick data
   - OHLCV bars (1min, 5min, 30min)
   - Order book snapshots
   - Economic indicators

2. **Synthetic Data**
   - Monte Carlo simulations
   - GAN-generated market scenarios
   - Stress test scenarios
   - Edge case situations

3. **Live Paper Trading**
   - Real-time market conditions
   - Zero-risk validation
   - Production environment testing
   - Model adaptation

## Environment Setup

### Training Environment Configuration

Create `configs/training/marl_training.yaml`:

```yaml
training:
  mode: marl_training
  environment: pettingzoo
  parallel_envs: 4
  
data:
  sources:
    - type: historical
      path: data/historical/ES_2023_2024.csv
      timeframe: ["1min", "5min", "30min"]
    - type: synthetic
      generator: monte_carlo
      scenarios: 1000
  
  preprocessing:
    normalize: true
    feature_engineering: true
    regime_detection: true
    
agents:
  strategic_agent:
    algorithm: PPO
    network_architecture:
      hidden_layers: [256, 128, 64]
      activation: relu
      dropout: 0.1
    learning_rate: 0.0003
    batch_size: 64
    epochs_per_update: 10
    
  tactical_agent:
    algorithm: SAC
    network_architecture:
      hidden_layers: [128, 64, 32]
      activation: tanh
      dropout: 0.05
    learning_rate: 0.001
    batch_size: 32
    target_update_frequency: 100
    
  risk_agent:
    algorithm: DQN
    network_architecture:
      hidden_layers: [64, 32]
      activation: relu
      dropout: 0.0
    learning_rate: 0.0005
    batch_size: 16
    epsilon_decay: 0.995

training_params:
  total_episodes: 10000
  episode_length: 2000  # steps
  evaluation_frequency: 100
  checkpoint_frequency: 500
  early_stopping:
    patience: 1000
    min_improvement: 0.01
    
experience_replay:
  buffer_size: 100000
  min_size_to_train: 1000
  prioritized_replay: true
  alpha: 0.6
  beta: 0.4
  
reward_shaping:
  profit_weight: 1.0
  risk_penalty: 0.5
  transaction_cost: 0.001
  sharpe_bonus: 0.2
  max_drawdown_penalty: 1.0
  
logging:
  tensorboard: true
  wandb: false
  checkpoint_dir: models/checkpoints/
  log_frequency: 10
```

### Training Data Preparation

```python
# scripts/prepare_training_data.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

class TrainingDataPreparator:
    """Prepare market data for MARL training"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_sources = config['data']['sources']
        
    def prepare_historical_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess historical market data"""
        
        # Load raw data
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Feature engineering
        df = self._add_technical_indicators(df)
        df = self._add_regime_features(df)
        df = self._add_microstructure_features(df)
        
        # Normalize features
        if self.config['data']['preprocessing']['normalize']:
            df = self._normalize_features(df)
            
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        
        # Momentum indicators
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        df['macd'], df['macd_signal'] = self._calculate_macd(df['close'])
        
        # Volatility indicators
        df['atr_14'] = self._calculate_atr(df, 14)
        df['bb_upper'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime classification features"""
        
        # Trend regime
        df['trend_regime'] = self._classify_trend_regime(df)
        
        # Volatility regime
        df['vol_regime'] = self._classify_volatility_regime(df)
        
        # Market session
        df['market_session'] = self._classify_market_session(df.index)
        
        return df
    
    def create_training_episodes(
        self, 
        df: pd.DataFrame, 
        episode_length: int = 2000
    ) -> List[pd.DataFrame]:
        """Split data into training episodes"""
        
        episodes = []
        total_length = len(df)
        
        for start_idx in range(0, total_length - episode_length, episode_length // 2):
            end_idx = start_idx + episode_length
            if end_idx <= total_length:
                episode_data = df.iloc[start_idx:end_idx].copy()
                episodes.append(episode_data)
                
        return episodes

# Usage
preparator = TrainingDataPreparator(config)
historical_data = preparator.prepare_historical_data("data/ES_2023_2024.csv")
episodes = preparator.create_training_episodes(historical_data)
```

## Training Configuration

### Agent-Specific Configurations

#### Strategic Agent (PPO)

```python
# configs/agents/strategic_agent.py
STRATEGIC_AGENT_CONFIG = {
    "algorithm": "PPO",
    "observation_space": {
        "type": "Box",
        "shape": (48, 13),  # 48 bars, 13 features
        "dtype": "float32"
    },
    "action_space": {
        "type": "Discrete",
        "n": 3  # [short, neutral, long]
    },
    "network": {
        "shared_layers": [256, 128],
        "policy_layers": [64, 32],
        "value_layers": [64, 32],
        "activation": "relu",
        "dropout": 0.1
    },
    "training": {
        "learning_rate": 3e-4,
        "batch_size": 64,
        "epochs_per_update": 10,
        "clip_range": 0.2,
        "entropy_coeff": 0.01,
        "value_loss_coeff": 0.5,
        "max_grad_norm": 0.5
    }
}
```

#### Tactical Agent (SAC)

```python
# configs/agents/tactical_agent.py
TACTICAL_AGENT_CONFIG = {
    "algorithm": "SAC",
    "observation_space": {
        "type": "Box", 
        "shape": (60, 7),  # 60 bars, 7 features
        "dtype": "float32"
    },
    "action_space": {
        "type": "Box",
        "shape": (3,),  # [entry_timing, stop_distance, position_scale]
        "low": [0.0, 0.001, 0.1],
        "high": [1.0, 0.05, 2.0]
    },
    "network": {
        "hidden_layers": [128, 64, 32],
        "activation": "tanh",
        "dropout": 0.05
    },
    "training": {
        "learning_rate": 1e-3,
        "batch_size": 32,
        "tau": 0.005,
        "alpha": 0.2,
        "target_update_frequency": 100,
        "gradient_steps": 1
    }
}
```

#### Risk Agent (DQN)

```python
# configs/agents/risk_agent.py
RISK_AGENT_CONFIG = {
    "algorithm": "DQN",
    "observation_space": {
        "type": "Box",
        "shape": (10,),  # Portfolio risk metrics
        "dtype": "float32"
    },
    "action_space": {
        "type": "Discrete",
        "n": 5  # Risk adjustment levels
    },
    "network": {
        "hidden_layers": [64, 32],
        "activation": "relu",
        "dueling": True,
        "double_dqn": True
    },
    "training": {
        "learning_rate": 5e-4,
        "batch_size": 16,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.995,
        "target_update_frequency": 1000
    }
}
```

## Training Process

### Training Script

```python
# scripts/train_marl_agents.py
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import wandb
from tensorboard import SummaryWriter

from src.training.marl_trainer import MARLTrainer
from src.training.environments import TradingEnvironment
from src.training.agents import StrategicAgent, TacticalAgent, RiskAgent

class MARLTrainingOrchestrator:
    """Orchestrates the training of multiple MARL agents"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Initialize components
        self.environment = self._create_environment()
        self.agents = self._create_agents()
        self.trainer = MARLTrainer(self.config, self.environment, self.agents)
        
        # Monitoring
        self.tensorboard_writer = SummaryWriter(log_dir="logs/tensorboard")
        if self.config['logging']['wandb']:
            wandb.init(project="grandmodel-marl")
    
    def _create_environment(self) -> TradingEnvironment:
        """Create the MARL trading environment"""
        return TradingEnvironment(
            data_sources=self.config['data']['sources'],
            episode_length=self.config['training_params']['episode_length'],
            transaction_costs=self.config['reward_shaping']['transaction_cost']
        )
    
    def _create_agents(self) -> Dict[str, object]:
        """Create all MARL agents"""
        agents = {}
        
        # Strategic Agent
        agents['strategic'] = StrategicAgent(
            config=self.config['agents']['strategic_agent'],
            observation_space=self.environment.observation_spaces['strategic'],
            action_space=self.environment.action_spaces['strategic']
        )
        
        # Tactical Agent  
        agents['tactical'] = TacticalAgent(
            config=self.config['agents']['tactical_agent'],
            observation_space=self.environment.observation_spaces['tactical'],
            action_space=self.environment.action_spaces['tactical']
        )
        
        # Risk Agent
        agents['risk'] = RiskAgent(
            config=self.config['agents']['risk_agent'],
            observation_space=self.environment.observation_spaces['risk'],
            action_space=self.environment.action_spaces['risk']
        )
        
        return agents
    
    def train(self) -> Dict[str, List[float]]:
        """Execute the complete training process"""
        
        self.logger.info("🚀 Starting MARL agent training")
        
        training_metrics = {
            'episode_rewards': [],
            'agent_losses': {agent: [] for agent in self.agents.keys()},
            'sharpe_ratios': [],
            'max_drawdowns': [],
            'win_rates': []
        }
        
        total_episodes = self.config['training_params']['total_episodes']
        eval_frequency = self.config['training_params']['evaluation_frequency']
        checkpoint_frequency = self.config['training_params']['checkpoint_frequency']
        
        for episode in range(total_episodes):
            
            # Training episode
            episode_metrics = self._run_training_episode(episode)
            self._update_training_metrics(training_metrics, episode_metrics)
            
            # Logging
            if episode % self.config['logging']['log_frequency'] == 0:
                self._log_training_progress(episode, episode_metrics)
            
            # Evaluation
            if episode % eval_frequency == 0:
                eval_metrics = self._run_evaluation(episode)
                self._log_evaluation_results(episode, eval_metrics)
                
                # Early stopping check
                if self._check_early_stopping(training_metrics):
                    self.logger.info(f"Early stopping at episode {episode}")
                    break
            
            # Checkpointing
            if episode % checkpoint_frequency == 0:
                self._save_checkpoint(episode)
        
        # Final evaluation and model saving
        final_metrics = self._run_final_evaluation()
        self._save_final_models()
        
        self.logger.info("✅ MARL agent training completed")
        return training_metrics
    
    def _run_training_episode(self, episode: int) -> Dict:
        """Run a single training episode"""
        
        # Reset environment
        observations = self.environment.reset()
        episode_rewards = {agent: 0.0 for agent in self.agents.keys()}
        episode_steps = 0
        
        done = False
        while not done and episode_steps < self.config['training_params']['episode_length']:
            
            # Get actions from all agents
            actions = {}
            for agent_name, agent in self.agents.items():
                action = agent.select_action(
                    observations[agent_name], 
                    training=True
                )
                actions[agent_name] = action
            
            # Environment step
            next_observations, rewards, done, info = self.environment.step(actions)
            
            # Store experience for each agent
            for agent_name, agent in self.agents.items():
                agent.store_experience(
                    state=observations[agent_name],
                    action=actions[agent_name],
                    reward=rewards[agent_name],
                    next_state=next_observations[agent_name],
                    done=done
                )
                episode_rewards[agent_name] += rewards[agent_name]
            
            observations = next_observations
            episode_steps += 1
        
        # Update agent models
        agent_losses = {}
        for agent_name, agent in self.agents.items():
            if agent.can_update():
                loss = agent.update()
                agent_losses[agent_name] = loss
        
        return {
            'episode_rewards': episode_rewards,
            'agent_losses': agent_losses,
            'episode_steps': episode_steps,
            'episode_info': info
        }
    
    def _run_evaluation(self, episode: int) -> Dict:
        """Run evaluation on validation data"""
        
        self.logger.info(f"Running evaluation at episode {episode}")
        
        # Switch to evaluation mode
        for agent in self.agents.values():
            agent.set_eval_mode()
        
        eval_episodes = 10
        eval_rewards = []
        eval_sharpe_ratios = []
        eval_max_drawdowns = []
        
        for eval_ep in range(eval_episodes):
            observations = self.environment.reset(evaluation=True)
            episode_reward = 0.0
            episode_returns = []
            
            done = False
            while not done:
                # Get actions (no exploration)
                actions = {}
                for agent_name, agent in self.agents.items():
                    action = agent.select_action(
                        observations[agent_name], 
                        training=False
                    )
                    actions[agent_name] = action
                
                observations, rewards, done, info = self.environment.step(actions)
                step_reward = sum(rewards.values())
                episode_reward += step_reward
                episode_returns.append(step_reward)
            
            # Calculate metrics
            eval_rewards.append(episode_reward)
            if len(episode_returns) > 1:
                sharpe = self._calculate_sharpe_ratio(episode_returns)
                max_dd = self._calculate_max_drawdown(episode_returns)
                eval_sharpe_ratios.append(sharpe)
                eval_max_drawdowns.append(max_dd)
        
        # Switch back to training mode
        for agent in self.agents.values():
            agent.set_train_mode()
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_sharpe': np.mean(eval_sharpe_ratios) if eval_sharpe_ratios else 0.0,
            'mean_max_drawdown': np.mean(eval_max_drawdowns) if eval_max_drawdowns else 0.0
        }
    
    def _log_training_progress(self, episode: int, metrics: Dict) -> None:
        """Log training progress to various outputs"""
        
        # Console logging
        total_reward = sum(metrics['episode_rewards'].values())
        self.logger.info(
            f"Episode {episode}: Total Reward = {total_reward:.4f}, "
            f"Steps = {metrics['episode_steps']}"
        )
        
        # TensorBoard logging
        self.tensorboard_writer.add_scalar('Training/TotalReward', total_reward, episode)
        for agent_name, reward in metrics['episode_rewards'].items():
            self.tensorboard_writer.add_scalar(f'Training/Reward_{agent_name}', reward, episode)
        
        for agent_name, loss in metrics.get('agent_losses', {}).items():
            self.tensorboard_writer.add_scalar(f'Training/Loss_{agent_name}', loss, episode)
        
        # Weights & Biases logging
        if self.config['logging']['wandb']:
            wandb.log({
                'episode': episode,
                'total_reward': total_reward,
                **{f'reward_{k}': v for k, v in metrics['episode_rewards'].items()},
                **{f'loss_{k}': v for k, v in metrics.get('agent_losses', {}).items()}
            })
    
    def _save_checkpoint(self, episode: int) -> None:
        """Save training checkpoint"""
        
        checkpoint_dir = Path(self.config['logging']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'episode': episode,
            'config': self.config,
            'agents': {
                name: agent.get_state_dict() 
                for name, agent in self.agents.items()
            },
            'optimizer_states': {
                name: agent.get_optimizer_state()
                for name, agent in self.agents.items()
            }
        }
        
        checkpoint_path = checkpoint_dir / f"checkpoint_episode_{episode}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

# Training execution
def main():
    trainer = MARLTrainingOrchestrator("configs/training/marl_training.yaml")
    metrics = trainer.train()
    
    print("Training completed!")
    print(f"Final average reward: {np.mean(metrics['episode_rewards'][-100:]):.4f}")

if __name__ == "__main__":
    main()
```

### Running Training

```bash
# Basic training
python scripts/train_marl_agents.py

# Training with custom config
python scripts/train_marl_agents.py --config configs/training/advanced_marl.yaml

# Training with GPU acceleration
CUDA_VISIBLE_DEVICES=0 python scripts/train_marl_agents.py

# Distributed training across multiple GPUs
python -m torch.distributed.launch --nproc_per_node=4 scripts/train_marl_agents.py

# Resume from checkpoint
python scripts/train_marl_agents.py --resume checkpoints/checkpoint_episode_5000.pth

# Training with wandb logging
python scripts/train_marl_agents.py --wandb --project grandmodel-experiment-1
```

## Model Evaluation

### Evaluation Metrics

```python
# src/training/evaluation.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class MARLEvaluator:
    """Comprehensive evaluation of MARL agents"""
    
    def __init__(self, agents: Dict, environment):
        self.agents = agents
        self.environment = environment
    
    def evaluate_performance(
        self, 
        episodes: int = 100, 
        save_results: bool = True
    ) -> Dict:
        """Comprehensive performance evaluation"""
        
        results = {
            'episode_rewards': [],
            'episode_returns': [],
            'episode_actions': [],
            'sharpe_ratios': [],
            'max_drawdowns': [],
            'win_rates': [],
            'agent_contributions': {agent: [] for agent in self.agents.keys()}
        }
        
        for episode in range(episodes):
            episode_data = self._run_evaluation_episode()
            self._update_results(results, episode_data)
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(results)
        
        if save_results:
            self._save_evaluation_results(results, summary_stats)
        
        return {
            'detailed_results': results,
            'summary_statistics': summary_stats
        }
    
    def _run_evaluation_episode(self) -> Dict:
        """Run single evaluation episode"""
        
        observations = self.environment.reset(evaluation=True)
        episode_data = {
            'rewards': [],
            'actions': [],
            'states': [],
            'agent_rewards': {agent: [] for agent in self.agents.keys()}
        }
        
        done = False
        while not done:
            # Get actions from all agents
            actions = {}
            for agent_name, agent in self.agents.items():
                action = agent.select_action(observations[agent_name], training=False)
                actions[agent_name] = action
            
            # Environment step
            next_observations, rewards, done, info = self.environment.step(actions)
            
            # Store episode data
            episode_data['actions'].append(actions.copy())
            episode_data['rewards'].append(sum(rewards.values()))
            episode_data['states'].append(observations.copy())
            
            for agent_name, reward in rewards.items():
                episode_data['agent_rewards'][agent_name].append(reward)
            
            observations = next_observations
        
        return episode_data
    
    def _calculate_summary_statistics(self, results: Dict) -> Dict:
        """Calculate comprehensive summary statistics"""
        
        episode_rewards = np.array(results['episode_rewards'])
        
        return {
            'performance_metrics': {
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'median_reward': np.median(episode_rewards),
                'min_reward': np.min(episode_rewards),
                'max_reward': np.max(episode_rewards),
                'reward_skewness': self._calculate_skewness(episode_rewards),
                'reward_kurtosis': self._calculate_kurtosis(episode_rewards)
            },
            'risk_metrics': {
                'mean_sharpe_ratio': np.mean(results['sharpe_ratios']),
                'mean_max_drawdown': np.mean(results['max_drawdowns']),
                'value_at_risk_95': np.percentile(episode_rewards, 5),
                'conditional_var_95': np.mean(episode_rewards[episode_rewards <= np.percentile(episode_rewards, 5)]),
                'volatility': np.std(episode_rewards) * np.sqrt(252)  # Annualized
            },
            'trading_metrics': {
                'win_rate': np.mean(results['win_rates']),
                'profit_factor': self._calculate_profit_factor(episode_rewards),
                'average_win': np.mean(episode_rewards[episode_rewards > 0]) if np.any(episode_rewards > 0) else 0,
                'average_loss': np.mean(episode_rewards[episode_rewards < 0]) if np.any(episode_rewards < 0) else 0,
                'largest_win': np.max(episode_rewards) if len(episode_rewards) > 0 else 0,
                'largest_loss': np.min(episode_rewards) if len(episode_rewards) > 0 else 0
            },
            'agent_analysis': {
                agent: {
                    'mean_contribution': np.mean(contributions),
                    'std_contribution': np.std(contributions),
                    'correlation_with_total': np.corrcoef(
                        contributions, 
                        results['episode_rewards']
                    )[0, 1] if len(contributions) > 1 else 0
                }
                for agent, contributions in results['agent_contributions'].items()
            }
        }
    
    def generate_evaluation_report(self, results: Dict) -> str:
        """Generate comprehensive evaluation report"""
        
        summary = results['summary_statistics']
        
        report = f"""
# MARL Agent Evaluation Report

## Performance Summary
- **Mean Episode Reward**: {summary['performance_metrics']['mean_reward']:.4f}
- **Reward Standard Deviation**: {summary['performance_metrics']['std_reward']:.4f}
- **Sharpe Ratio**: {summary['risk_metrics']['mean_sharpe_ratio']:.4f}
- **Maximum Drawdown**: {summary['risk_metrics']['mean_max_drawdown']:.4f}
- **Win Rate**: {summary['trading_metrics']['win_rate']:.2%}

## Risk Analysis
- **Value at Risk (95%)**: {summary['risk_metrics']['value_at_risk_95']:.4f}
- **Conditional VaR (95%)**: {summary['risk_metrics']['conditional_var_95']:.4f}
- **Annualized Volatility**: {summary['risk_metrics']['volatility']:.2%}

## Trading Performance
- **Profit Factor**: {summary['trading_metrics']['profit_factor']:.4f}
- **Average Win**: {summary['trading_metrics']['average_win']:.4f}
- **Average Loss**: {summary['trading_metrics']['average_loss']:.4f}
- **Largest Win**: {summary['trading_metrics']['largest_win']:.4f}
- **Largest Loss**: {summary['trading_metrics']['largest_loss']:.4f}

## Agent Contributions
"""
        
        for agent, metrics in summary['agent_analysis'].items():
            report += f"""
### {agent.title()} Agent
- **Mean Contribution**: {metrics['mean_contribution']:.4f}
- **Contribution Volatility**: {metrics['std_contribution']:.4f}
- **Correlation with Total**: {metrics['correlation_with_total']:.4f}
"""
        
        return report
```

### Backtesting Framework

```python
# src/training/backtesting.py
class MARLBacktester:
    """Comprehensive backtesting framework for MARL agents"""
    
    def __init__(self, agents: Dict, data_path: str):
        self.agents = agents
        self.data = self._load_historical_data(data_path)
        
    def run_backtest(
        self, 
        start_date: str, 
        end_date: str,
        initial_capital: float = 100000.0,
        transaction_costs: float = 0.001
    ) -> Dict:
        """Run comprehensive backtest"""
        
        # Filter data for date range
        backtest_data = self._filter_data_by_date(start_date, end_date)
        
        # Initialize portfolio
        portfolio = Portfolio(
            initial_capital=initial_capital,
            transaction_costs=transaction_costs
        )
        
        backtest_results = {
            'equity_curve': [],
            'trade_log': [],
            'daily_returns': [],
            'positions': [],
            'agent_decisions': []
        }
        
        # Run simulation
        for timestamp, market_data in backtest_data.iterrows():
            
            # Get agent observations
            observations = self._prepare_observations(timestamp, market_data)
            
            # Get agent actions
            actions = {}
            for agent_name, agent in self.agents.items():
                action = agent.select_action(observations[agent_name], training=False)
                actions[agent_name] = action
            
            # Execute trading logic
            trade_signal = self._combine_agent_actions(actions)
            if trade_signal:
                trade_result = portfolio.execute_trade(trade_signal, market_data)
                backtest_results['trade_log'].append(trade_result)
            
            # Update portfolio
            portfolio.update_positions(market_data)
            
            # Record results
            backtest_results['equity_curve'].append(portfolio.total_value)
            backtest_results['positions'].append(portfolio.current_positions.copy())
            backtest_results['agent_decisions'].append(actions.copy())
        
        # Calculate performance metrics
        performance_metrics = self._calculate_backtest_metrics(backtest_results)
        
        return {
            'results': backtest_results,
            'metrics': performance_metrics,
            'portfolio_summary': portfolio.get_summary()
        }
```

## Hyperparameter Optimization

### Automated Hyperparameter Tuning

```python
# scripts/hyperparameter_optimization.py
import optuna
from typing import Dict, Any
import numpy as np

class MARLHyperparameterOptimizer:
    """Automated hyperparameter optimization for MARL agents"""
    
    def __init__(self, base_config: Dict):
        self.base_config = base_config
        
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for hyperparameter optimization"""
        
        # Sample hyperparameters
        config = self._sample_hyperparameters(trial)
        
        # Train agents with sampled hyperparameters
        trainer = MARLTrainingOrchestrator(config)
        
        # Run shorter training for optimization
        config['training_params']['total_episodes'] = 1000
        metrics = trainer.train()
        
        # Return negative mean reward (Optuna minimizes)
        return -np.mean(metrics['episode_rewards'][-100:])
    
    def _sample_hyperparameters(self, trial: optuna.Trial) -> Dict:
        """Sample hyperparameters using Optuna"""
        
        config = self.base_config.copy()
        
        # Strategic agent hyperparameters
        config['agents']['strategic_agent']['learning_rate'] = trial.suggest_float(
            'strategic_lr', 1e-5, 1e-2, log=True
        )
        config['agents']['strategic_agent']['batch_size'] = trial.suggest_int(
            'strategic_batch_size', 16, 128, step=16
        )
        config['agents']['strategic_agent']['epochs_per_update'] = trial.suggest_int(
            'strategic_epochs', 5, 20
        )
        
        # Tactical agent hyperparameters
        config['agents']['tactical_agent']['learning_rate'] = trial.suggest_float(
            'tactical_lr', 1e-5, 1e-2, log=True
        )
        config['agents']['tactical_agent']['batch_size'] = trial.suggest_int(
            'tactical_batch_size', 8, 64, step=8
        )
        
        # Risk agent hyperparameters
        config['agents']['risk_agent']['learning_rate'] = trial.suggest_float(
            'risk_lr', 1e-5, 1e-2, log=True
        )
        config['agents']['risk_agent']['epsilon_decay'] = trial.suggest_float(
            'risk_epsilon_decay', 0.99, 0.999
        )
        
        # Network architecture
        hidden_size = trial.suggest_int('hidden_size', 64, 512, step=64)
        config['agents']['strategic_agent']['network']['hidden_layers'] = [
            hidden_size, hidden_size // 2
        ]
        
        # Reward shaping
        config['reward_shaping']['risk_penalty'] = trial.suggest_float(
            'risk_penalty', 0.1, 2.0
        )
        config['reward_shaping']['sharpe_bonus'] = trial.suggest_float(
            'sharpe_bonus', 0.0, 1.0
        )
        
        return config
    
    def optimize(self, n_trials: int = 100) -> Dict:
        """Run hyperparameter optimization"""
        
        # Create Optuna study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Optimize
        study.optimize(self.objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        # Create optimized configuration
        optimized_config = self._apply_best_params(best_params)
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'optimized_config': optimized_config,
            'study': study
        }

# Usage
optimizer = MARLHyperparameterOptimizer(base_config)
optimization_results = optimizer.optimize(n_trials=50)
```

## Production Deployment

### Model Export and Deployment

```python
# scripts/export_trained_models.py
import torch
import onnx
from pathlib import Path

class ModelExporter:
    """Export trained MARL models for production deployment"""
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint = torch.load(checkpoint_path)
        self.agents = self._load_agents_from_checkpoint()
    
    def export_to_production(self, output_dir: str) -> None:
        """Export models in production-ready format"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for agent_name, agent in self.agents.items():
            
            # Export PyTorch model
            torch_path = output_path / f"{agent_name}_model.pth"
            torch.save(agent.state_dict(), torch_path)
            
            # Export to ONNX for inference optimization
            onnx_path = output_path / f"{agent_name}_model.onnx"
            self._export_to_onnx(agent, onnx_path)
            
            # Export model metadata
            metadata_path = output_path / f"{agent_name}_metadata.json"
            self._export_metadata(agent, metadata_path)
    
    def _export_to_onnx(self, agent, output_path: Path) -> None:
        """Export model to ONNX format"""
        
        # Create dummy input
        dummy_input = torch.randn(1, *agent.observation_space.shape)
        
        # Export to ONNX
        torch.onnx.export(
            agent.policy_network,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['observation'],
            output_names=['action'],
            dynamic_axes={
                'observation': {0: 'batch_size'},
                'action': {0: 'batch_size'}
            }
        )

# Production deployment script
def deploy_models_to_production():
    """Deploy trained models to production environment"""
    
    # Export models
    exporter = ModelExporter("models/checkpoints/best_model.pth")
    exporter.export_to_production("models/production/")
    
    # Update production configuration
    production_config = {
        "strategic_marl": {
            "enabled": True,
            "model_path": "models/production/strategic_model.pth",
            "onnx_path": "models/production/strategic_model.onnx"
        },
        "agents": {
            "strategic": {
                "model_path": "models/production/strategic_model.pth"
            },
            "tactical": {
                "model_path": "models/production/tactical_model.pth"
            },
            "risk": {
                "model_path": "models/production/risk_model.pth"
            }
        }
    }
    
    # Save production config
    with open("configs/production/marl_models.yaml", 'w') as f:
        yaml.dump(production_config, f)
    
    print("✅ Models deployed to production")
```

### Model Monitoring and Updates

```python
# src/training/model_monitoring.py
class ProductionModelMonitor:
    """Monitor MARL model performance in production"""
    
    def __init__(self, model_config: Dict):
        self.model_config = model_config
        self.performance_metrics = {}
        
    def monitor_model_performance(self) -> Dict:
        """Monitor real-time model performance"""
        
        # Collect recent performance data
        recent_trades = self._get_recent_trades()
        recent_decisions = self._get_recent_decisions()
        
        # Calculate performance metrics
        performance = {
            'win_rate': self._calculate_win_rate(recent_trades),
            'sharpe_ratio': self._calculate_sharpe_ratio(recent_trades),
            'decision_consistency': self._calculate_decision_consistency(recent_decisions),
            'model_confidence': self._calculate_avg_confidence(recent_decisions)
        }
        
        # Check for performance degradation
        alerts = self._check_performance_alerts(performance)
        
        return {
            'performance': performance,
            'alerts': alerts,
            'recommendation': self._get_retraining_recommendation(performance)
        }
    
    def _check_performance_alerts(self, performance: Dict) -> List[str]:
        """Check for performance degradation alerts"""
        
        alerts = []
        
        if performance['win_rate'] < 0.45:  # Below 45% win rate
            alerts.append("LOW_WIN_RATE")
        
        if performance['sharpe_ratio'] < 0.5:  # Below 0.5 Sharpe ratio
            alerts.append("LOW_SHARPE_RATIO")
            
        if performance['decision_consistency'] < 0.7:  # Low consistency
            alerts.append("INCONSISTENT_DECISIONS")
            
        if performance['model_confidence'] < 0.6:  # Low confidence
            alerts.append("LOW_MODEL_CONFIDENCE")
        
        return alerts
    
    def _get_retraining_recommendation(self, performance: Dict) -> str:
        """Recommend whether models need retraining"""
        
        if len(self._check_performance_alerts(performance)) >= 2:
            return "IMMEDIATE_RETRAINING_REQUIRED"
        elif performance['win_rate'] < 0.5:
            return "RETRAINING_RECOMMENDED"
        else:
            return "PERFORMANCE_ACCEPTABLE"
```

## Advanced Topics

### Curriculum Learning

```python
# src/training/curriculum_learning.py
class CurriculumLearningScheduler:
    """Implement curriculum learning for MARL agents"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.current_difficulty = 0
        self.difficulty_levels = self._define_difficulty_levels()
    
    def _define_difficulty_levels(self) -> List[Dict]:
        """Define curriculum difficulty levels"""
        
        return [
            # Level 0: Simple trending markets
            {
                "market_conditions": ["strong_trend"],
                "volatility_range": [0.1, 0.3],
                "noise_level": 0.1,
                "episode_length": 500
            },
            # Level 1: Mixed market conditions
            {
                "market_conditions": ["trend", "sideways"],
                "volatility_range": [0.2, 0.5],
                "noise_level": 0.2,
                "episode_length": 1000
            },
            # Level 2: Complex market scenarios
            {
                "market_conditions": ["trend", "sideways", "volatile"],
                "volatility_range": [0.1, 0.8],
                "noise_level": 0.3,
                "episode_length": 1500
            },
            # Level 3: Real market complexity
            {
                "market_conditions": ["all"],
                "volatility_range": [0.05, 1.0],
                "noise_level": 0.4,
                "episode_length": 2000
            }
        ]
    
    def should_advance_difficulty(self, recent_performance: List[float]) -> bool:
        """Determine if agents are ready for next difficulty level"""
        
        if len(recent_performance) < 100:
            return False
        
        # Check if performance is consistently good
        recent_avg = np.mean(recent_performance[-100:])
        recent_std = np.std(recent_performance[-100:])
        
        # Advancement criteria
        threshold = self.config.get('advancement_threshold', 0.7)
        stability_threshold = self.config.get('stability_threshold', 0.2)
        
        return recent_avg > threshold and recent_std < stability_threshold
    
    def get_current_difficulty_config(self) -> Dict:
        """Get configuration for current difficulty level"""
        return self.difficulty_levels[self.current_difficulty]
    
    def advance_difficulty(self) -> bool:
        """Advance to next difficulty level"""
        if self.current_difficulty < len(self.difficulty_levels) - 1:
            self.current_difficulty += 1
            return True
        return False
```

### Multi-Objective Optimization

```python
# src/training/multi_objective.py
class MultiObjectiveTrainer:
    """Train MARL agents with multiple objectives"""
    
    def __init__(self, objectives: List[str], weights: List[float]):
        self.objectives = objectives  # ['profit', 'risk', 'sharpe', 'drawdown']
        self.weights = weights
        self.pareto_solutions = []
    
    def calculate_multi_objective_reward(self, trade_results: Dict) -> float:
        """Calculate reward considering multiple objectives"""
        
        objective_values = {
            'profit': trade_results['pnl'],
            'risk': -abs(trade_results['max_drawdown']),
            'sharpe': trade_results['sharpe_ratio'],
            'drawdown': -trade_results['max_drawdown']
        }
        
        # Weighted sum of objectives
        total_reward = sum(
            self.weights[i] * objective_values[obj] 
            for i, obj in enumerate(self.objectives)
        )
        
        return total_reward
    
    def find_pareto_optimal_solutions(self, population: List[Dict]) -> List[Dict]:
        """Find Pareto optimal solutions in multi-objective space"""
        
        pareto_solutions = []
        
        for candidate in population:
            is_dominated = False
            
            for other in population:
                if self._dominates(other, candidate):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_solutions.append(candidate)
        
        return pareto_solutions
```

### Meta-Learning for Agent Adaptation

```python
# src/training/meta_learning.py
class MetaLearningMARL:
    """Meta-learning for rapid adaptation to new market conditions"""
    
    def __init__(self, base_agents: Dict):
        self.base_agents = base_agents
        self.meta_learner = self._create_meta_learner()
    
    def adapt_to_new_market(self, adaptation_data: pd.DataFrame, steps: int = 100) -> Dict:
        """Quickly adapt agents to new market conditions"""
        
        # Extract market characteristics
        market_features = self._extract_market_features(adaptation_data)
        
        # Use meta-learner to predict optimal hyperparameters
        optimal_params = self.meta_learner.predict_optimal_params(market_features)
        
        # Fine-tune agents with predicted parameters
        adapted_agents = {}
        for agent_name, agent in self.base_agents.items():
            adapted_agent = self._fine_tune_agent(
                agent, 
                adaptation_data, 
                optimal_params[agent_name],
                steps
            )
            adapted_agents[agent_name] = adapted_agent
        
        return adapted_agents
```

## Related Documentation

- [Getting Started Guide](getting_started.md)
- [API Documentation](../api/)
- [Architecture Overview](../architecture/system_overview.md)
- [Deployment Guide](deployment_guide.md)
- [Performance Optimization](performance_guide.md)