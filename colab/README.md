# 🎯 GrandModel Colab Training Environment

Welcome to the **GrandModel MARL (Multi-Agent Reinforcement Learning) Training Environment** - a comprehensive system for training tactical and strategic trading agents using Google Colab's GPU infrastructure.

## 🚀 Quick Start

### 1. **Tactical Training** (5-minute data)
```python
# Open and run: notebooks/tactical_mappo_training.ipynb
# Trains: tactical_agent, risk_agent, execution_agent
# Duration: ~30-60 minutes
# Output: Production-ready tactical models
```

### 2. **Strategic Training** (30-minute data) 
```python
# Open and run: notebooks/strategic_mappo_training.ipynb  
# Trains: strategic_agent, portfolio_manager, regime_detector
# Duration: ~60-120 minutes
# Output: Production-ready strategic models
```

## 📁 Directory Structure

```
colab/
├── notebooks/           # Training notebooks
│   ├── tactical_mappo_training.ipynb
│   └── strategic_mappo_training.ipynb
├── trainers/           # Core training modules
│   ├── tactical_mappo_trainer.py
│   └── strategic_mappo_trainer.py
├── utils/              # Utilities and optimizations
│   ├── gpu_optimizer.py
│   └── __init__.py
├── configs/            # Configuration files
│   └── training_config.yaml
├── data/               # Market data files
│   ├── NQ - 5 min - ETH.csv
│   └── NQ - 30 min - ETH.csv
├── exports/            # Training outputs
│   └── [timestamp]/    # Organized by training session
└── README.md           # This file
```

## 🤖 Agent Architecture

### Tactical Agents (5-minute timeframe)
- **Tactical Agent**: Short-term trading decisions and entry/exit timing
- **Risk Agent**: Position sizing and risk management
- **Execution Agent**: Order execution optimization and slippage minimization

### Strategic Agents (30-minute timeframe)
- **Strategic Agent**: Long-term market positioning and trend following
- **Portfolio Manager**: Portfolio optimization and risk-adjusted returns
- **Regime Detector**: Market condition identification and strategy adaptation

## 🎯 Training Features

### Advanced MAPPO Implementation
- **Multi-Agent Coordination**: Agents learn to cooperate and specialize
- **Proximal Policy Optimization**: Stable policy gradient learning
- **Generalized Advantage Estimation**: Improved value function learning
- **Automatic Mixed Precision**: GPU memory optimization

### Market Intelligence
- **Regime Detection**: Automatic identification of market conditions
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Volume Analysis**: Volume-price relationship modeling
- **Risk Metrics**: Sharpe ratio, maximum drawdown, VaR calculation

### GPU Optimization
- **Automatic Batch Size Detection**: Optimal memory utilization
- **Memory Monitoring**: Real-time memory usage tracking
- **Performance Profiling**: Model complexity analysis
- **Cache Management**: Automatic memory cleanup

## 📊 Data Requirements

### Market Data Format
```csv
Date,Open,High,Low,Close,Volume
2024-01-01 00:00:00,16850.25,16875.75,16832.50,16861.25,12450
2024-01-01 00:05:00,16861.25,16890.00,16847.75,16878.50,15620
...
```

### Data Specifications
- **Tactical Data**: 5-minute OHLCV bars (minimum 1000 bars)
- **Strategic Data**: 30-minute OHLCV bars (minimum 500 bars)
- **Format**: CSV with DateTime index
- **Instrument**: NQ Futures (NASDAQ-100 E-mini)

## ⚙️ Configuration

### Training Parameters
```yaml
# Tactical Configuration
tactical:
  num_episodes: 200
  episode_length: 1000
  lr_actor: 3e-4
  lr_critic: 1e-3
  
# Strategic Configuration  
strategic:
  num_episodes: 300
  episode_length: 500
  lr_actor: 1e-4
  lr_critic: 3e-4
```

### Performance Targets
- **Tactical**: Target reward > 100, Training episodes < 200
- **Strategic**: Target Sharpe > 2.0, Max drawdown < 15%

## 🎯 Action Spaces

### Tactical Actions
- `0`: **HOLD** - Maintain current position
- `1`: **BUY_SMALL** - Small long position increase
- `2`: **BUY_LARGE** - Large long position increase  
- `3`: **SELL_SMALL** - Small position reduction
- `4`: **SELL_LARGE** - Large position reduction

### Strategic Actions
- `0`: **HOLD** - Maintain current positions
- `1`: **BUY_CONSERVATIVE** - Conservative long increase
- `2`: **BUY_AGGRESSIVE** - Aggressive long increase
- `3`: **SELL_CONSERVATIVE** - Conservative position reduction
- `4`: **SELL_AGGRESSIVE** - Aggressive position reduction
- `5`: **REDUCE_RISK** - Lower portfolio risk exposure
- `6`: **INCREASE_RISK** - Higher portfolio risk exposure

## 📈 Performance Metrics

### Training Metrics
- **Episode Rewards**: Cumulative reward per episode
- **Sharpe Ratio**: Risk-adjusted returns
- **Portfolio Value**: Simulated portfolio growth
- **Maximum Drawdown**: Peak-to-trough portfolio decline
- **Win Rate**: Percentage of profitable episodes

### Model Metrics
- **Actor Loss**: Policy network optimization
- **Critic Loss**: Value function accuracy
- **Training Speed**: Episodes per minute
- **Memory Usage**: GPU and system memory utilization

## 🚀 Training Workflow

### 1. Environment Setup
```python
# Install dependencies
!pip install torch pandas matplotlib pettingzoo

# Clone repository
!git clone https://github.com/Afeks214/GrandModel.git

# Mount Google Drive (optional)
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Data Preparation
```python
# Load market data
df = pd.read_csv('colab/data/NQ - 5 min - ETH.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Verify data quality
print(f"Data shape: {df.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
```

### 3. Model Training
```python
# Initialize trainer
trainer = TacticalMAPPOTrainer(
    state_dim=7,
    action_dim=5,
    n_agents=3,
    device='cuda'
)

# Train model
for episode in range(num_episodes):
    reward, steps = trainer.train_episode(data=df)
    if episode % 50 == 0:
        trainer.plot_training_progress()
```

### 4. Model Export
```python
# Save trained models
trainer.save_checkpoint('best_tactical_model.pth')

# Export for production
export_models(trainer, save_dir='exports/tactical_training')
```

## 📦 Export Package

### Model Files
- `best_*_model.pth`: Best performing checkpoints
- `final_*_model.pth`: Final training checkpoints
- `*_actor.pth`: Individual agent actor networks
- `*_critic.pth`: Individual agent critic networks

### Configuration
- `model_config.json`: Complete model configuration
- `training_statistics.json`: Performance metrics
- `validation_results.json`: Backtesting results

### Documentation
- `README.md`: Comprehensive usage guide
- `deploy_model.py`: Production deployment script
- `plots/`: Training progress visualizations

## 🔧 Troubleshooting

### Common Issues

#### GPU Memory Error
```python
# Reduce batch size
trainer = TacticalMAPPOTrainer(batch_size=16)

# Clear GPU cache
import torch
torch.cuda.empty_cache()
```

#### Data Loading Error
```python
# Check file path
import os
assert os.path.exists('colab/data/NQ - 5 min - ETH.csv')

# Verify data format
df = pd.read_csv('colab/data/NQ - 5 min - ETH.csv')
print(df.columns)  # Should be: Date, Open, High, Low, Close, Volume
```

#### Training Convergence Issues
```python
# Reduce learning rate
trainer = TacticalMAPPOTrainer(lr_actor=1e-4, lr_critic=3e-4)

# Increase episode length
trainer.train_episode(episode_length=2000)
```

### Performance Optimization

#### GPU Optimization
```python
# Enable mixed precision
scaler = torch.cuda.amp.GradScaler()

# Optimize batch size
optimal_batch = gpu_optimizer.optimize_batch_size(model, input_shape)
```

#### Memory Management
```python
# Monitor memory usage
memory_info = gpu_optimizer.monitor_memory()

# Clear cache periodically
if episode % 20 == 0:
    gpu_optimizer.clear_cache()
```

## 📚 Advanced Usage

### Custom Market Data
```python
# Load your own data
custom_data = pd.read_csv('your_market_data.csv')

# Ensure proper format
required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
assert all(col in custom_data.columns for col in required_columns)

# Train on custom data
trainer.train_episode(data=custom_data)
```

### Hyperparameter Tuning
```python
# Grid search example
learning_rates = [1e-4, 3e-4, 1e-3]
batch_sizes = [16, 32, 64]

best_performance = 0
for lr in learning_rates:
    for bs in batch_sizes:
        trainer = TacticalMAPPOTrainer(lr_actor=lr, batch_size=bs)
        performance = trainer.train_episode(data=df)
        if performance > best_performance:
            best_performance = performance
            trainer.save_checkpoint(f'best_lr{lr}_bs{bs}.pth')
```

### Multi-Agent Analysis
```python
# Analyze individual agent performance
for i, agent_name in enumerate(['tactical', 'risk', 'execution']):
    agent_rewards = trainer.get_agent_rewards(agent_idx=i)
    print(f"{agent_name} agent average reward: {np.mean(agent_rewards):.3f}")
    
# Visualize agent coordination
trainer.plot_agent_coordination()
```

## 🏆 Production Deployment

### Model Loading
```python
# Load trained models
from colab.trainers.tactical_mappo_trainer import TacticalMAPPOTrainer

trainer = TacticalMAPPOTrainer()
trainer.load_checkpoint('best_tactical_model.pth')

# Make predictions
actions, _, _ = trainer.get_action(states, deterministic=True)
```

### Integration with GrandModel
```python
# Integration example
from src.agents.trading_env import TradingEnvironment

# Initialize environment with trained agents
env = TradingEnvironment()
env.load_tactical_agents('exports/tactical_training/best_tactical_model.pth')
env.load_strategic_agents('exports/strategic_training/best_strategic_model.pth')

# Execute trading decisions
tactical_actions = env.get_tactical_actions(market_data_5min)
strategic_actions = env.get_strategic_actions(market_data_30min)
```

## 📝 Citation

If you use this training environment in your research or trading systems, please cite:

```bibtex
@software{grandmodel_colab_training,
  title={GrandModel MARL Training Environment},
  author={QuantNova},
  year={2024},
  url={https://github.com/Afeks214/GrandModel}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: See notebook examples and inline comments
- **Issues**: Open GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas

---

**🎯 Ready to train world-class MARL trading agents? Start with the tactical training notebook!**

*GrandModel Colab Training Environment - Empowering the future of algorithmic trading*