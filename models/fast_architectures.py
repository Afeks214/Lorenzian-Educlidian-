"""
AGENT 4: Ultra-Fast Neural Network Architectures for <5ms Inference
Lightweight replacements for existing models optimized for speed over complexity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import time

class FastTacticalActor(nn.Module):
    """
    Ultra-fast tactical actor using shallow, wide architecture.
    Target: <1.5ms inference per agent
    """
    
    def __init__(
        self,
        agent_id: str,
        input_shape: Tuple[int, int] = (60, 7),
        action_dim: int = 3,
        hidden_dim: int = 128,  # Reduced from 256
    ):
        super().__init__()
        
        self.agent_id = agent_id
        self.input_size = input_shape[0] * input_shape[1]  # 420
        self.action_dim = action_dim
        
        # Agent-specific feature weights (fixed, not learnable for speed)
        if agent_id == "fvg":
            # Focus on FVG features: [fvg_bullish, fvg_bearish, fvg_level, fvg_age, fvg_mitigation, momentum, volume]
            weights = torch.tensor([2.0, 2.0, 1.0, 0.5, 0.5, 0.3, 0.3])
        elif agent_id == "momentum":
            # Focus on momentum and volume
            weights = torch.tensor([0.3, 0.3, 0.5, 0.2, 0.2, 2.0, 2.0])
        elif agent_id == "entry":
            # Balanced approach
            weights = torch.tensor([1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8])
        else:
            weights = torch.ones(7)
        
        # Expand weights to full input size (60 time steps * 7 features)
        expanded_weights = weights.repeat(60)
        self.register_buffer('feature_weights', expanded_weights)
        
        # Ultra-simple network: Input -> Wide Hidden -> Output
        # No attention, no LSTM, no convolutions
        self.network = nn.Sequential(
            nn.Linear(self.input_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Initialize with Xavier uniform for fast convergence
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Ultra-fast forward pass."""
        # Flatten input: (batch_size, 60, 7) -> (batch_size, 420)
        if state.dim() == 3:
            x = state.view(state.size(0), -1)
        else:
            x = state.view(-1)
            x = x.unsqueeze(0)
        
        # Apply feature weights (in-place for speed)
        x = x * self.feature_weights
        
        # Network forward
        action_probs = self.network(x)
        
        # Sample action
        action = torch.argmax(action_probs, dim=-1)
        
        # Calculate log probabilities for training
        log_probs = torch.log(action_probs + 1e-8)
        action_log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
        
        return {
            'action': action,
            'action_probs': action_probs,
            'log_prob': action_log_prob,
            'logits': action_probs  # For compatibility
        }


class FastMLMIActor(nn.Module):
    """
    Ultra-fast MLMI actor for momentum analysis.
    Target: <1ms inference
    """
    
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,  # Much smaller
        action_dim: int = 3
    ):
        super().__init__()
        
        # Simple 2-layer network
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """Fast forward pass."""
        action_probs = self.network(state)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            # Sample from categorical distribution
            action = torch.multinomial(action_probs, 1).squeeze(-1)
        
        log_probs = torch.log(action_probs + 1e-8)
        action_log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
        
        return {
            'action': action,
            'action_probs': action_probs,
            'log_prob': action_log_prob,
            'logits': action_probs
        }


class FastNWRQKActor(nn.Module):
    """
    Ultra-fast NWRQK actor for support/resistance analysis.
    Target: <1ms inference
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        action_dim: int = 3
    ):
        super().__init__()
        
        # Simple 2-layer network
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """Fast forward pass."""
        action_probs = self.network(state)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            action = torch.multinomial(action_probs, 1).squeeze(-1)
        
        log_probs = torch.log(action_probs + 1e-8)
        action_log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
        
        return {
            'action': action,
            'action_probs': action_probs,
            'log_prob': action_log_prob,
            'logits': action_probs
        }


class FastMMDActor(nn.Module):
    """
    Ultra-fast MMD actor for regime detection.
    Target: <1ms inference
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 32,  # Very small for 3D input
        action_dim: int = 3
    ):
        super().__init__()
        
        # Minimal network for regime detection
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """Fast forward pass."""
        action_probs = self.network(state)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            action = torch.multinomial(action_probs, 1).squeeze(-1)
        
        log_probs = torch.log(action_probs + 1e-8)
        action_log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
        
        return {
            'action': action,
            'action_probs': action_probs,
            'log_prob': action_log_prob,
            'logits': action_probs
        }


class FastCentralizedCritic(nn.Module):
    """
    Ultra-fast centralized critic.
    Target: <0.5ms inference
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64  # Much smaller
    ):
        super().__init__()
        
        # Simple 2-layer MLP
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Fast forward pass."""
        value = self.network(states).squeeze(-1)
        return {'value': value}


class FastTacticalMARLSystem(nn.Module):
    """
    Ultra-fast tactical MARL system combining all fast agents.
    Target: <5ms total inference (all 3 agents + decision aggregation)
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int] = (60, 7),
        action_dim: int = 3,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.input_shape = input_shape
        self.action_dim = action_dim
        self.state_dim = input_shape[0] * input_shape[1]  # 420
        
        # Fast tactical agents
        self.agents = nn.ModuleDict({
            'fvg': FastTacticalActor(
                agent_id='fvg',
                input_shape=input_shape,
                action_dim=action_dim,
                hidden_dim=hidden_dim
            ),
            'momentum': FastTacticalActor(
                agent_id='momentum',
                input_shape=input_shape,
                action_dim=action_dim,
                hidden_dim=hidden_dim
            ),
            'entry': FastTacticalActor(
                agent_id='entry',
                input_shape=input_shape,
                action_dim=action_dim,
                hidden_dim=hidden_dim
            )
        })
        
        # Fast centralized critic
        self.critic = FastCentralizedCritic(
            state_dim=self.state_dim * 3,  # Combined state
            hidden_dim=64
        )
    
    def forward(self, state: torch.Tensor, deterministic: bool = False) -> Dict[str, Any]:
        """Ultra-fast forward pass through all agents."""
        batch_size = state.size(0) if state.dim() == 3 else 1
        
        # Get actions from all agents (parallel inference)
        agent_outputs = {}
        for agent_name, agent in self.agents.items():
            agent_outputs[agent_name] = agent(state)
        
        # Prepare combined state for critic
        combined_state = state.view(batch_size, -1).repeat(1, 3)
        
        # Get value from critic
        critic_output = self.critic(combined_state)
        
        return {
            'agents': agent_outputs,
            'critic': critic_output,
            'combined_state': combined_state
        }
    
    @torch.no_grad()
    def fast_inference(self, state: torch.Tensor) -> Dict[str, Any]:
        """Optimized inference mode with no gradient computation."""
        self.eval()
        
        # Single forward pass
        result = self.forward(state, deterministic=True)
        
        # Extract actions quickly
        actions = {}
        for agent_name, agent_output in result['agents'].items():
            actions[agent_name] = agent_output['action'].item() if agent_output['action'].numel() == 1 else agent_output['action'][0].item()
        
        return {
            'actions': actions,
            'value': result['critic']['value'].item() if result['critic']['value'].numel() == 1 else result['critic']['value'][0].item()
        }


class FastStrategicMARLSystem(nn.Module):
    """
    Ultra-fast strategic MARL system combining all strategic agents.
    Target: <2ms total inference
    """
    
    def __init__(self):
        super().__init__()
        
        # Fast strategic agents
        self.agents = nn.ModuleDict({
            'mlmi': FastMLMIActor(input_dim=4),
            'nwrqk': FastNWRQKActor(input_dim=6),
            'mmd': FastMMDActor(input_dim=3)
        })
        
        # Fast critic for combined state (4+6+3=13)
        self.critic = FastCentralizedCritic(state_dim=13)
    
    def forward(self, states: Dict[str, torch.Tensor], deterministic: bool = False) -> Dict[str, Any]:
        """Ultra-fast forward pass through strategic agents."""
        # Get actions from all agents
        agent_outputs = {}
        combined_states = []
        
        for agent_name, agent in self.agents.items():
            if agent_name in states:
                agent_output = agent(states[agent_name], deterministic=deterministic)
                agent_outputs[agent_name] = agent_output
                combined_states.append(states[agent_name])
        
        # Combine states for critic
        if combined_states:
            combined_state = torch.cat(combined_states, dim=-1)
            critic_output = self.critic(combined_state)
        else:
            critic_output = {'value': torch.zeros(1)}
        
        return {
            'agents': agent_outputs,
            'critic': critic_output
        }
    
    @torch.no_grad()
    def fast_inference(self, states: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Optimized inference mode."""
        self.eval()
        
        result = self.forward(states, deterministic=True)
        
        # Extract actions quickly
        actions = {}
        for agent_name, agent_output in result['agents'].items():
            actions[agent_name] = agent_output['action'].item() if agent_output['action'].numel() == 1 else agent_output['action'][0].item()
        
        return {
            'actions': actions,
            'value': result['critic']['value'].item() if result['critic']['value'].numel() == 1 else result['critic']['value'][0].item()
        }


def create_optimized_models() -> Dict[str, nn.Module]:
    """Create all optimized models for production deployment."""
    models = {
        'fast_tactical_marl': FastTacticalMARLSystem(),
        'fast_strategic_marl': FastStrategicMARLSystem(),
        'fast_mlmi_actor': FastMLMIActor(),
        'fast_nwrqk_actor': FastNWRQKActor(),
        'fast_mmd_actor': FastMMDActor(),
        'fast_critic': FastCentralizedCritic(state_dim=13)
    }
    
    # Set all models to eval mode
    for model in models.values():
        model.eval()
    
    return models


def benchmark_fast_models() -> Dict[str, Dict[str, float]]:
    """Benchmark the fast models to ensure <5ms performance."""
    models = create_optimized_models()
    
    # Test inputs
    tactical_input = torch.randn(1, 60, 7)
    strategic_inputs = {
        'mlmi': torch.randn(1, 4),
        'nwrqk': torch.randn(1, 6),
        'mmd': torch.randn(1, 3)
    }
    
    results = {}
    
    # Benchmark each model
    for model_name, model in models.items():
        latencies = []
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                if 'tactical' in model_name:
                    _ = model.fast_inference(tactical_input)
                elif 'strategic' in model_name:
                    _ = model.fast_inference(strategic_inputs)
                elif 'mlmi' in model_name:
                    _ = model(strategic_inputs['mlmi'])
                elif 'nwrqk' in model_name:
                    _ = model(strategic_inputs['nwrqk'])
                elif 'mmd' in model_name:
                    _ = model(strategic_inputs['mmd'])
                else:  # critic
                    _ = model(torch.randn(1, 13))
        
        # Benchmark
        with torch.no_grad():
            for _ in range(1000):
                start_time = time.perf_counter()
                
                if 'tactical' in model_name:
                    _ = model.fast_inference(tactical_input)
                elif 'strategic' in model_name:
                    _ = model.fast_inference(strategic_inputs)
                elif 'mlmi' in model_name:
                    _ = model(strategic_inputs['mlmi'])
                elif 'nwrqk' in model_name:
                    _ = model(strategic_inputs['nwrqk'])
                elif 'mmd' in model_name:
                    _ = model(strategic_inputs['mmd'])
                else:  # critic
                    _ = model(torch.randn(1, 13))
                
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        results[model_name] = {
            'avg_latency_ms': np.mean(latencies),
            'p99_latency_ms': np.percentile(latencies, 99),
            'meets_target': np.percentile(latencies, 99) < 5.0,
            'params': sum(p.numel() for p in model.parameters())
        }
    
    return results


if __name__ == "__main__":
    print("🚀 Benchmarking Fast Architecture Models")
    results = benchmark_fast_models()
    
    print("\n" + "="*60)
    print("📊 FAST MODEL PERFORMANCE RESULTS")
    print("="*60)
    
    total_tactical_time = 0
    total_strategic_time = 0
    
    for model_name, stats in results.items():
        status = "✅ PASS" if stats['meets_target'] else "❌ FAIL"
        print(f"{model_name:25}: {stats['p99_latency_ms']:6.2f}ms p99 | {stats['params']:8,} params {status}")
        
        if 'tactical' in model_name:
            total_tactical_time = stats['p99_latency_ms']
        elif 'strategic' in model_name:
            total_strategic_time = stats['p99_latency_ms']
    
    print("\n" + "="*60)
    print(f"🎯 TOTAL TACTICAL INFERENCE: {total_tactical_time:.2f}ms")
    print(f"🎯 TOTAL STRATEGIC INFERENCE: {total_strategic_time:.2f}ms")
    
    if total_tactical_time < 5.0 and total_strategic_time < 5.0:
        print("✅ SUCCESS: All models meet <5ms inference requirement!")
    else:
        print("❌ OPTIMIZATION NEEDED: Models still exceed 5ms target")
    
    print("="*60)