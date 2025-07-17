# Strategic MARL Models - Integration Guide

This directory contains the core neural network models for the Strategic MARL 30m system.

## Quick Start

### Loading Pre-trained Models

```python
import torch
from models.architectures import MLMIActor, NWRQKActor, MMDActor, CentralizedCritic

# Load checkpoint
checkpoint = torch.load('models/checkpoints/strategic_marl_initial.pt')

# Initialize models
agents = {
    'mlmi': MLMIActor(input_dim=4, hidden_dims=[256, 128, 64], dropout_rate=0.1),
    'nwrqk': NWRQKActor(input_dim=6, hidden_dims=[256, 128, 64], dropout_rate=0.1),
    'mmd': MMDActor(input_dim=3, hidden_dims=[128, 64, 32], dropout_rate=0.1)
}

critic = CentralizedCritic(
    state_dim=13,  # 4 + 6 + 3
    n_agents=3,
    hidden_dims=[512, 256, 128],
    dropout_rate=0.1
)

# Load state dicts
for name, agent in agents.items():
    agent.load_state_dict(checkpoint['model_state_dicts']['agents'][name])
    
critic.load_state_dict(checkpoint['model_state_dicts']['critic'])
```

### Using Models for Inference

```python
# Prepare input data
# Each agent expects: [batch_size, input_dim, sequence_length]
mlmi_state = torch.randn(1, 4, 48)    # MLMI features
nwrqk_state = torch.randn(1, 6, 48)   # NWRQK features
mmd_state = torch.randn(1, 3, 48)     # MMD features

# Get actions from each agent
mlmi_output = agents['mlmi'](mlmi_state)
nwrqk_output = agents['nwrqk'](nwrqk_state)
mmd_output = agents['mmd'](mmd_state)

# Extract actions
actions = {
    'mlmi': mlmi_output['action'].item(),      # 0, 1, or 2
    'nwrqk': nwrqk_output['action'].item(),
    'mmd': mmd_output['action'].item()
}

# Get value estimate from critic
combined_state = torch.cat([
    mlmi_state.mean(dim=-1),    # Average over sequence
    nwrqk_state.mean(dim=-1),
    mmd_state.mean(dim=-1)
], dim=-1)  # Shape: [1, 13]

critic_output = critic(combined_state)
state_value = critic_output['value'].item()
```

## Model Architecture

### Actors

Each actor follows the pattern: **Conv1D → Temporal (LSTM/GRU) → Attention → Output**

- **MLMIActor**: Momentum specialist
  - Input: 4D features `[mlmi_value, mlmi_signal, momentum_20, momentum_50]`
  - Kernels: [3, 5] (small for short-term patterns)
  - Temporal: LSTM (2 layers)
  
- **NWRQKActor**: Support/Resistance levels
  - Input: 6D features `[nwrqk_pred, lvn_strength, fvg_size, price_distance, volume_imbalance, level_touches]`
  - Kernels: [7, 9] (large for broader patterns)
  - Temporal: Bidirectional LSTM (3 layers)
  
- **MMDActor**: Market regime detection
  - Input: 3D features `[mmd_score, volatility_30, volume_profile_skew]`
  - Kernels: [5, 7]
  - Temporal: GRU (2 layers, efficient)
  - Special: Volatility adjustment layer

### Critic

- **CentralizedCritic**: Sees all agent observations
  - Input: 13D combined state
  - Architecture: Deep MLP with residual blocks
  - Features: Layer normalization for stability

## Configuration

Load configuration from YAML:

```python
import yaml

with open('configs/strategic_marl_v1.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Access agent configs
mlmi_config = config['agents']['mlmi']
nwrqk_config = config['agents']['nwrqk']
mmd_config = config['agents']['mmd']
```

## Training Integration

For training with MAPPO:

```python
from training.mappo_trainer import MAPPOTrainer

# Initialize trainer with config
trainer = MAPPOTrainer(config['training'])

# Training loop
for episode in range(n_episodes):
    experiences = trainer.collect_experience(env, n_steps=2048)
    metrics = trainer.update_agents(experiences)
    
    if episode % 100 == 0:
        trainer.save_checkpoint(metrics)
```

## Checkpointing

Using the CheckpointManager:

```python
from utils.checkpoint_manager import CheckpointManager

checkpoint_manager = CheckpointManager(
    checkpoint_dir='models/checkpoints',
    max_checkpoints=5,
    best_metric='average_reward'
)

# Save checkpoint
checkpoint_manager.save_checkpoint(
    agents=agents,
    critic=critic,
    metrics={'reward': 0.85},
    is_best=True
)

# Load best checkpoint
checkpoint = checkpoint_manager.load_best_checkpoint(
    agents=agents,
    critic=critic
)
```

## Input Data Format

Each agent expects specific features in the following format:

### MLMI Agent
```python
# Shape: [batch_size, 4, 48]
# Features:
# - mlmi_value: MLMI indicator value
# - mlmi_signal: Binary signal (0/1)
# - momentum_20: 20-period momentum
# - momentum_50: 50-period momentum
```

### NWRQK Agent
```python
# Shape: [batch_size, 6, 48]
# Features:
# - nwrqk_pred: Level prediction (-1 to 1)
# - lvn_strength: Low volume node strength
# - fvg_size: Fair value gap size
# - price_distance: Distance to nearest level
# - volume_imbalance: Buy/sell volume imbalance
# - level_touches: Number of level touches
```

### MMD Agent
```python
# Shape: [batch_size, 3, 48]
# Features:
# - mmd_score: Market regime score
# - volatility_30: 30-period volatility
# - volume_profile_skew: Volume distribution skew
```

## Action Space

All agents output discrete actions:
- **0**: Bearish (reduce position/sell)
- **1**: Neutral (hold/no action)
- **2**: Bullish (increase position/buy)

## Performance Considerations

- Models are optimized for batch processing
- Use GPU when available: `model.to('cuda')`
- For inference, use `model.eval()` and `torch.no_grad()`
- Sequence length is fixed at 48 timesteps

## Troubleshooting

### Common Issues

1. **Dimension Mismatch**: Ensure input shape is `[batch, features, sequence_length]`
2. **Device Mismatch**: Move all tensors to same device before forward pass
3. **Missing Features**: Each agent requires exact number of input features

### Verification Script

```python
# Verify model loading
def verify_models(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    print(f"Checkpoint keys: {checkpoint.keys()}")
    print(f"Config version: {checkpoint['metadata']['config_version']}")
    print(f"Created: {checkpoint['metadata']['created']}")
    
verify_models('models/checkpoints/strategic_marl_initial.pt')
```

## Contact

For integration support or questions about the models, please refer to the main project documentation or contact the ML team.