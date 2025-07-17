# Agent-Specific Mathematical Models - Detailed Analysis

**Document**: Complete Mathematical Model Specifications for Strategic MARL Agents  
**Date**: 2025-07-13  
**Component**: Strategic MARL 30m Individual Agent Mathematics  

## Overview

This document provides detailed mathematical specifications for each of the three strategic agents (MLMI, NWRQK, Regime) extracted from the Strategic MARL 30m PRD, with complete formulations ready for implementation.

---

## 1. MLMI Strategic Agent - Complete Mathematical Model

### 1.1 Input Feature Specification

**Feature Vector (4D):**
```
s_mlmi = [mlmi_value, mlmi_signal, momentum_20, momentum_50]
```

**Feature Descriptions:**
- `mlmi_value`: Core MLMI indicator value from kernel regression
- `mlmi_signal`: Binary/continuous signal strength [-1, 1]
- `momentum_20`: 20-period momentum indicator
- `momentum_50`: 50-period momentum indicator

### 1.2 Feature Engineering Pipeline

**Normalization:**
```
s_norm = (s_mlmi - μ_mlmi) / σ_mlmi
```

**Running Statistics Update:**
```
μ_mlmi ← α · μ_mlmi + (1-α) · s_mlmi    # α = 0.99
σ_mlmi ← α · σ_mlmi + (1-α) · |s_mlmi - μ_mlmi|
```

### 1.3 Policy Network Architecture

**Forward Pass:**
```
h_1 = ReLU(W_1 · s_norm + b_1)         # [4] → [256]
h_2 = ReLU(W_2 · h_1 + b_2)            # [256] → [128]  
h_3 = ReLU(W_3 · h_2 + b_3)            # [128] → [64]
logits = W_out · h_3 + b_out           # [64] → [3]
```

**Weight Initialization:**
```
W_i ~ Xavier_normal(gain=√2)           # ReLU-optimized
b_i ~ Uniform(-1/√fan_in, 1/√fan_in)
```

### 1.4 Action Distribution & Temperature Scaling

**Temperature-Scaled Softmax:**
```
π_mlmi(a|s) = Softmax(logits / τ)
```

**Adaptive Temperature:**
```
τ = τ_base · (1 + uncertainty_factor)
uncertainty_factor = |momentum_20 - momentum_50| / volatility_measure
```

### 1.5 MLMI-Specific Reward Function

**Multi-Component Reward:**
```
R_mlmi = w_base · R_base + w_synergy · R_synergy + w_momentum · R_momentum
```

**Component Calculations:**
```
R_base = tanh(PnL_normalized)
R_synergy = I_synergy · synergy_strength
R_momentum = |momentum_change| · direction_alignment
```

**Weight Learning (via MAPPO):**
```
∇w_base = ∂L_policy/∂w_base
w_base ← w_base - lr · ∇w_base
```

---

## 2. NWRQK Strategic Agent - Complete Mathematical Model

### 2.1 Input Feature Specification

**Feature Vector (4D):**
```
s_nwrqk = [nwrqk_value, nwrqk_slope, lvn_distance, lvn_strength]
```

**Feature Descriptions:**
- `nwrqk_value`: Rational quadratic kernel regression value
- `nwrqk_slope`: Local slope/gradient of NWRQK surface
- `lvn_distance`: Distance to nearest Low Volume Node
- `lvn_strength`: Strength/confidence of LVN level

### 2.2 Kernel Regression Mathematics

**NWRQK Value Computation:**
```
ŷ_t = Σ_(i=1)^n K_h(x_t, x_i) · y_i / Σ_(i=1)^n K_h(x_t, x_i)
```

**Rational Quadratic Kernel:**
```
K_h(x_t, x_i) = (1 + ||x_t - x_i||^2 / (2αh^2))^(-α)
```

**Hyperparameters:**
- `α = 2.0` (shape parameter)
- `h = adaptive_bandwidth(data)` (automatically determined)

**Slope Calculation:**
```
nwrqk_slope = ∂ŷ_t/∂x_t ≈ (ŷ_t - ŷ_(t-1)) / Δt
```

### 2.3 Support/Resistance Logic

**Distance Penalty:**
```
distance_penalty = min(1, lvn_distance / max_distance)
max_distance = 0.01 · current_price    # 1% of price
```

**Support Strength:**
```
support_strength = max(0, lvn_strength - distance_penalty)
```

**Resistance Logic:**
```
resistance_strength = max(0, lvn_strength + distance_penalty)
```

### 2.4 Action Probability Calculation

**Exponential Probability Model:**
```
P_bullish ∝ exp(β_1 · nwrqk_slope + β_2 · support_strength)
P_bearish ∝ exp(-β_1 · nwrqk_slope - β_2 · support_strength)  
P_neutral ∝ exp(β_3 · uncertainty_measure)
```

**Uncertainty Measure:**
```
uncertainty_measure = 1 - max(support_strength, resistance_strength)
```

**β Parameter Learning:**
```
β = [β_1, β_2, β_3]^T
∇β = ∂L_policy/∂β    # Learned via MAPPO
β ← β - lr · ∇β
```

### 2.5 NWRQK-Specific Reward Function

**Support/Resistance Alignment:**
```
R_nwrqk = w_sr · R_support_resistance + w_trend · R_trend + w_base · R_base
```

**Support/Resistance Reward:**
```
R_support_resistance = {
    +support_strength,  if action = bullish & near_support
    +resistance_strength, if action = bearish & near_resistance  
    -penalty,          otherwise
}
```

---

## 3. Regime Detection Agent - Complete Mathematical Model

### 3.1 Input Feature Specification

**Feature Vector (3D):**
```
s_regime = [mmd_score, volatility_30, volume_profile_skew]
```

**Feature Descriptions:**
- `mmd_score`: Maximum Mean Discrepancy between distributions
- `volatility_30`: 30-period rolling volatility
- `volume_profile_skew`: Asymmetry in volume distribution

### 3.2 Maximum Mean Discrepancy (MMD) Computation

**MMD Formulation:**
```
MMD²(P, Q) = E_P[k(x,x')] + E_Q[k(y,y')] - 2E_{P,Q}[k(x,y)]
```

**Gaussian Kernel:**
```
k(x,y) = exp(-||x-y||²/(2σ²))
```

**Empirical MMD Estimate:**
```
MMD²_emp = (1/m²)Σ_i,j k(x_i,x_j) + (1/n²)Σ_i,j k(y_i,y_j) - (2/mn)Σ_i,j k(x_i,y_j)
```

**Bandwidth Selection:**
```
σ² = median_heuristic(data) = median(||x_i - x_j||²)
```

### 3.3 Volatility Regime Classification

**Volatility Buckets:**
```
regime_class = {
    0: low_vol,     if volatility_30 < 0.1
    1: medium_vol,  if 0.1 ≤ volatility_30 < 0.3
    2: high_vol,    if volatility_30 ≥ 0.3
}
```

**MLP Regime Classification:**
```
h_1 = ReLU(W_1 · [mmd_score, volatility_30, volume_skew] + b_1)
h_2 = ReLU(W_2 · h_1 + b_2)
regime_logits = W_out · h_2 + b_out
regime_probs = Softmax(regime_logits)
```

### 3.4 Volatility-Adjusted Policy

**Policy Adjustment:**
```
π_regime(a|s) = Softmax(logits · volatility_adjustment)
```

**Volatility Adjustment Factor:**
```
volatility_adjustment = max(0.5, min(2.0, 1/volatility_30))
```

**Interpretation:**
- High volatility → Lower adjustment → More conservative
- Low volatility → Higher adjustment → More aggressive

### 3.5 Regime-Specific Reward Function

**Regime Consistency Reward:**
```
R_regime = w_consistency · R_consistency + w_transition · R_transition
```

**Consistency Reward:**
```
R_consistency = regime_confidence · action_regime_alignment
```

**Transition Detection Reward:**
```
R_transition = {
    +bonus,  if regime_change_detected & action_appropriate
    -penalty, if regime_change_missed
    0,       otherwise
}
```

---

## 4. Agent Coordination Mathematics

### 4.1 Superposition Ensemble

**Individual Outputs:**
```
P_mlmi = Softmax(logits_mlmi / τ_mlmi)
P_nwrqk = Softmax(logits_nwrqk / τ_nwrqk)
P_regime = Softmax(logits_regime / τ_regime)
```

**Weighted Combination:**
```
P_ensemble = w_mlmi · P_mlmi + w_nwrqk · P_nwrqk + w_regime · P_regime
```

**Weight Normalization:**
```
w = Softmax([w_mlmi_raw, w_nwrqk_raw, w_regime_raw])
```

### 4.2 Confidence Estimation

**Ensemble Confidence:**
```
confidence = max(P_ensemble) - entropy(P_ensemble)
```

**Entropy Calculation:**
```
entropy(P) = -Σ_i p_i log(p_i)
```

**Per-Agent Confidence:**
```
conf_i = max(P_i) - entropy(P_i)
```

### 4.3 Dynamic Weight Learning

**Performance-Based Weighting:**
```
performance_score_i = recent_accuracy_i · confidence_i
weight_adjustment_i = performance_score_i / Σ_j performance_score_j
```

**Exponential Moving Average:**
```
w_i ← α · w_i + (1-α) · weight_adjustment_i
```

---

## 5. Implementation Considerations

### 5.1 Numerical Stability

**Softmax Numerical Stability:**
```
logits_stable = logits - max(logits)
P = exp(logits_stable) / sum(exp(logits_stable))
```

**Division by Zero Protection:**
```
denominator = max(ε, actual_denominator)    # ε = 1e-8
```

### 5.2 Gradient Flow Optimization

**Gradient Clipping:**
```
grad_norm = ||∇θ||_2
if grad_norm > max_grad_norm:
    ∇θ ← ∇θ · (max_grad_norm / grad_norm)
```

**Learning Rate Scheduling:**
```
lr_t = lr_0 · decay_factor^(step / decay_steps)
```

### 5.3 Memory Efficiency

**Feature Standardization:**
```
# Use running statistics instead of batch statistics
μ_running ← α · μ_running + (1-α) · μ_batch
σ_running ← α · σ_running + (1-α) · σ_batch
```

---

## 6. Validation & Testing

### 6.1 Mathematical Unit Tests

**GAE Validation:**
```python
def test_gae_computation():
    # Verify GAE matches analytical solution for simple cases
    assert abs(computed_gae - analytical_gae) < 1e-6
```

**Kernel Computation:**
```python
def test_rational_quadratic_kernel():
    # Verify kernel properties: symmetry, positive definiteness
    assert K(x,y) == K(y,x)
    assert K(x,x) == 1.0
```

### 6.2 Numerical Consistency

**Probability Sum Validation:**
```python
def test_probability_consistency():
    assert abs(sum(probabilities) - 1.0) < 1e-8
    assert all(p >= 0 for p in probabilities)
```

---

## 7. Summary

This document provides the complete mathematical foundation for implementing all three strategic agents with:

✅ **Complete Formulations**: Every mathematical operation specified  
✅ **Implementation Ready**: All formulas in computational form  
✅ **Numerically Stable**: Stability considerations included  
✅ **Validation Ready**: Test specifications provided  

**Next Step**: Use these specifications for exact implementation alignment with the PRD mathematical requirements.

---

*This serves as the definitive mathematical reference for agent implementation.*