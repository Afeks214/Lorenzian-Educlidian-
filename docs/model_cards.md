# Strategic MARL Model Cards

## Overview

This document provides detailed model cards for each specialized actor in the Strategic MARL 30m system. Each card describes the model's purpose, architecture, inputs, outputs, and intended use within the multi-agent ensemble.

---

## MLMIActor - Momentum Specialist

### Purpose
The MLMIActor specializes in detecting and responding to momentum patterns in market data. It focuses on short-term price accelerations and momentum shifts that indicate potential trading opportunities.

### Architecture
- **Type**: Convolutional-Temporal Neural Network
- **Key Components**:
  - Conv1D layers with small kernels [3, 5] for capturing short-term patterns
  - 2-layer LSTM for temporal sequence modeling
  - 8-head attention mechanism for focusing on momentum shifts
  - Residual blocks with GELU activation
  - Learnable temperature parameter for exploration control

### Input Features
- **Dimension**: 4D vector over 48 timesteps
- **Features**:
  1. `mlmi_value`: MLMI indicator value (normalized)
  2. `mlmi_signal`: Binary momentum signal (0/1)
  3. `momentum_20`: 20-period price momentum
  4. `momentum_50`: 50-period price momentum

### Output
- **Action Space**: 3 discrete actions
  - 0: Bearish momentum detected (reduce position)
  - 1: Neutral momentum (hold position)
  - 2: Bullish momentum detected (increase position)
- **Additional Outputs**:
  - Action probabilities (softmax distribution)
  - State value estimate
  - Attention weights for interpretability

### Intended Use
- Detects momentum regime changes
- Provides early signals for trend beginnings
- Works best in trending market conditions
- Should be combined with other agents for complete strategy

### Performance Characteristics
- **Strengths**: Fast reaction to momentum changes, good at trend detection
- **Limitations**: May generate false signals in choppy markets
- **Computational**: ~1.45M parameters, <5ms inference time on GPU

---

## NWRQKActor - Support/Resistance Level Specialist

### Purpose
The NWRQKActor specializes in identifying and trading around key support and resistance levels. It uses larger receptive fields to understand price behavior around significant market levels.

### Architecture
- **Type**: Bidirectional Convolutional-Temporal Network
- **Key Components**:
  - Conv1D layers with large kernels [7, 9] for broader pattern recognition
  - 3-layer bidirectional LSTM for past/future context
  - 8-head attention mechanism for level relationships
  - Layer normalization for training stability
  - Residual blocks with Swish activation

### Input Features
- **Dimension**: 6D vector over 48 timesteps
- **Features**:
  1. `nwrqk_pred`: NWRQK level prediction (-1 to 1)
  2. `lvn_strength`: Low volume node strength (0 to 1)
  3. `fvg_size`: Fair value gap size (normalized)
  4. `price_distance`: Distance to nearest significant level
  5. `volume_imbalance`: Buy/sell volume imbalance ratio
  6. `level_touches`: Count of recent level touches

### Output
- **Action Space**: 3 discrete actions
  - 0: Bearish (expecting level rejection)
  - 1: Neutral (no clear level interaction)
  - 2: Bullish (expecting level break/bounce)
- **Additional Outputs**:
  - Action probabilities with level confidence
  - State value estimate
  - Attention focus on key levels

### Intended Use
- Trading around support/resistance levels
- Identifying breakout or reversal opportunities
- Risk management near key levels
- Provides structural market context

### Performance Characteristics
- **Strengths**: Excellent at level-based trading, good risk/reward identification
- **Limitations**: Less effective in trending markets without clear levels
- **Computational**: ~4.81M parameters, <8ms inference time on GPU

---

## MMDActor - Market Regime Detector

### Purpose
The MMDActor specializes in identifying market regimes and adjusting trading behavior based on volatility and market conditions. It provides adaptive behavior for different market environments.

### Architecture
- **Type**: GRU-based Temporal Network with Volatility Adjustment
- **Key Components**:
  - Conv1D layers with medium kernels [5, 7]
  - 2-layer GRU for efficient sequence modeling
  - 4-head attention mechanism (fewer heads for simpler features)
  - Volatility adjustment layer (learnable parameter)
  - Instance normalization for regime stability

### Input Features
- **Dimension**: 3D vector over 48 timesteps
- **Features**:
  1. `mmd_score`: Maximum mean discrepancy regime score
  2. `volatility_30`: 30-period realized volatility
  3. `volume_profile_skew`: Volume distribution skewness

### Output
- **Action Space**: 3 discrete actions
  - 0: Risk-off regime (reduce exposure)
  - 1: Normal regime (standard positioning)
  - 2: Risk-on regime (increase exposure)
- **Additional Outputs**:
  - Volatility-adjusted action probabilities
  - Regime confidence score
  - State value estimate

### Intended Use
- Market regime classification
- Position sizing based on market conditions
- Risk adjustment for other agents' signals
- Portfolio-level risk management

### Performance Characteristics
- **Strengths**: Adapts to changing market conditions, provides stability
- **Limitations**: May lag in regime transitions
- **Computational**: ~285K parameters (lightweight), <3ms inference time on GPU

---

## CentralizedCritic - Value Estimator

### Purpose
The CentralizedCritic provides value estimates for the combined state of all agents, enabling coordinated learning through centralized training with decentralized execution.

### Architecture
- **Type**: Deep Residual MLP
- **Key Components**:
  - Input layer for combined 13D state
  - Residual blocks with layer normalization
  - 4-head attention pooling over agent contributions
  - Auxiliary value heads for each agent
  - Dropout for regularization

### Input
- **Dimension**: 13D combined state vector
- **Composition**: Concatenated features from all agents
  - MLMI features: 4D
  - NWRQK features: 6D
  - MMD features: 3D

### Output
- **Primary**: Scalar state value estimate
- **Auxiliary**: Individual agent value estimates (optional)

### Intended Use
- Provides baseline for advantage estimation
- Enables coordinated multi-agent learning
- Reduces variance in policy gradient estimates

### Performance Characteristics
- **Parameters**: ~495K
- **Inference**: <2ms on GPU
- **Stability**: Layer normalization ensures stable training

---

## Ensemble Behavior

### Coordination
The three specialized actors work together through:
1. **Complementary Specializations**: Each focuses on different market aspects
2. **Shared Objective**: Coordinated through centralized critic
3. **Action Synthesis**: Combined actions create nuanced trading decisions

### Integration Example
```python
# Momentum suggests bullish (2), Levels suggest neutral (1), Regime suggests risk-off (0)
# Ensemble decision: Moderate bullish with reduced size
ensemble_action = weighted_average([
    mlmi_action * 0.4,   # Momentum weight
    nwrqk_action * 0.4,  # Levels weight
    mmd_action * 0.2     # Regime weight
])
```

### Best Practices
1. **Never rely on single agent**: Always consider ensemble output
2. **Monitor attention patterns**: Indicates what each agent focuses on
3. **Adjust weights by market**: Different conditions favor different agents
4. **Regular retraining**: Markets evolve, models should too

---

## Model Limitations and Ethical Considerations

### Limitations
- Models trained on historical data may not adapt to unprecedented events
- Fixed 48-timestep window may miss longer-term patterns
- Discrete action space simplifies continuous position sizing

### Ethical Considerations
- Models should not be used for market manipulation
- Risk management systems must be in place
- Human oversight required for production deployment
- Regular monitoring for drift and bias

### Responsible Use
- Test thoroughly in paper trading before live deployment
- Implement position limits and risk controls
- Monitor for unusual behavior or degraded performance
- Maintain audit trails of model decisions

---

## Version History

- **v1.0.0** (Current): Initial release with three specialized agents
- Architecture validated through unit tests and overfitting verification
- Attention mechanisms provide interpretability
- Optimized hyperparameters from simulated study

---

## References

1. Strategic MARL 30m PRD - Complete Mathematical & Production Specification
2. Multi-Agent Proximal Policy Optimization (MAPPO) paper
3. Attention Is All You Need - Transformer architecture
4. Domain-specific indicators: MLMI, NWRQK, MMD documentation