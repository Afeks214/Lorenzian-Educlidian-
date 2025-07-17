"""
AGENT 4: JIT-Compatible Ultra-Fast Models for Production
Models designed specifically for TorchScript compilation with tensor-only outputs
"""

import logging


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import time
import numpy as np
import gc
from contextlib import contextmanager

# Import performance optimizations
try:
    from src.performance.memory_manager import get_memory_manager, temporary_tensor, cleanup_memory
except ImportError:
    # Fallback implementations
    def get_memory_manager():
        return None
    
    @contextmanager
    def temporary_tensor(shape, dtype, device):
        tensor = torch.empty(shape, dtype=dtype, device=device)
        try:
            yield tensor
        finally:
            del tensor
    
    def cleanup_memory(force=False):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

class JITFastTacticalActor(nn.Module):
    """JIT-compatible tactical actor with tensor-only outputs and memory optimization."""
    
    def __init__(self, agent_id: str, input_size: int = 420, hidden_dim: int = 128, action_dim: int = 3):
        super().__init__()
        
        self.agent_id = agent_id
        self.input_size = input_size
        
        # Agent-specific feature weights (fixed buffers for JIT compatibility)
        if agent_id == "fvg":
            weights = torch.tensor([2.0, 2.0, 1.0, 0.5, 0.5, 0.3, 0.3]).repeat(60)
        elif agent_id == "momentum":
            weights = torch.tensor([0.3, 0.3, 0.5, 0.2, 0.2, 2.0, 2.0]).repeat(60)
        elif agent_id == "entry":
            weights = torch.tensor([1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8]).repeat(60)
        else:
            weights = torch.ones(420)
        
        self.register_buffer('feature_weights', weights)
        
        # Ultra-simple network
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Memory optimization - track tensor usage
        self._memory_manager = get_memory_manager()
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
        # Register cleanup hook
        self.register_full_backward_hook(self._cleanup_hook)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return action probabilities only (JIT compatible) with memory optimization."""
        # Track tensor usage
        if self._memory_manager:
            self._memory_manager.track_tensor(state, f"input_{self.agent_id}")
        
        # Flatten input: (batch_size, 60, 7) -> (batch_size, 420)
        if state.dim() == 3:
            x = state.view(state.size(0), -1)
        else:
            x = state.view(1, -1)
        
        # Apply feature weights with memory optimization
        with torch.no_grad():
            # Use temporary tensor for intermediate computation
            x = x * self.feature_weights
        
        # Network forward
        logits = self.network(x)
        action_probs = F.softmax(logits, dim=-1)
        
        # Track output tensor
        if self._memory_manager:
            self._memory_manager.track_tensor(action_probs, f"output_{self.agent_id}")
        
        return action_probs
    
    def _cleanup_hook(self, module, grad_input, grad_output):
        """Cleanup hook for memory management."""
        # Cleanup gradients immediately after backward pass
        if grad_input is not None:
            for grad in grad_input:
                if grad is not None:
                    del grad
        
        # Force garbage collection periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            cleanup_memory(force=True)
        except (ImportError, ModuleNotFoundError) as e:
            logger.error(f'Error occurred: {e}')


class JITFastStrategicActor(nn.Module):
    """JIT-compatible strategic actor with tensor-only outputs."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, action_dim: int = 3):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return action probabilities only."""
        logits = self.network(state)
        action_probs = F.softmax(logits, dim=-1)
        return action_probs


class JITFastCritic(nn.Module):
    """JIT-compatible critic with tensor-only outputs."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        
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
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return state value only."""
        value = self.network(state)
        return value.squeeze(-1)


class JITFastTacticalSystem(nn.Module):
    """JIT-compatible tactical system returning concatenated outputs with memory optimization."""
    
    def __init__(self, input_size: int = 420, hidden_dim: int = 128):
        super().__init__()
        
        # Three tactical agents
        self.fvg_agent = JITFastTacticalActor("fvg", input_size, hidden_dim)
        self.momentum_agent = JITFastTacticalActor("momentum", input_size, hidden_dim)
        self.entry_agent = JITFastTacticalActor("entry", input_size, hidden_dim)
        
        # Critic for combined state
        self.critic = JITFastCritic(input_size * 3, 64)
        
        # Memory optimization
        self._memory_manager = get_memory_manager()
        self._inference_count = 0
        self._cleanup_frequency = 100  # Cleanup every 100 inferences
        
        # Register cleanup hook
        self.register_full_backward_hook(self._system_cleanup_hook)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return tuple of (fvg_probs, momentum_probs, entry_probs, value).
        JIT-compatible with tensor-only outputs and memory optimization.
        """
        # Increment inference counter
        self._inference_count += 1
        
        # Track input tensor
        if self._memory_manager:
            self._memory_manager.track_tensor(state, "tactical_system_input")
        
        # Get action probabilities from each agent with memory optimization
        with torch.no_grad():
            # Use temporary tensors for intermediate state processing
            fvg_probs = self.fvg_agent(state)
            momentum_probs = self.momentum_agent(state)
            entry_probs = self.entry_agent(state)
        
        # Prepare combined state for critic with memory optimization
        batch_size = state.size(0) if state.dim() == 3 else 1
        
        # Use temporary tensor for combined state
        with temporary_tensor(torch.Size([batch_size, state.numel() * 3]), state.dtype, state.device) as combined_state:
            state_flat = state.view(batch_size, -1)
            combined_state[:, :state_flat.size(1)] = state_flat
            combined_state[:, state_flat.size(1):state_flat.size(1)*2] = state_flat
            combined_state[:, state_flat.size(1)*2:] = state_flat
            
            # Get value estimate
            value = self.critic(combined_state)
        
        # Periodic cleanup
        if self._inference_count % self._cleanup_frequency == 0:
            self._periodic_cleanup()
        
        return fvg_probs, momentum_probs, entry_probs, value
    
    def _system_cleanup_hook(self, module, grad_input, grad_output):
        """System-wide cleanup hook."""
        # Cleanup gradients
        if grad_input is not None:
            for grad in grad_input:
                if grad is not None:
                    del grad
        
        # Force GPU cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _periodic_cleanup(self):
        """Periodic memory cleanup."""
        try:
            cleanup_memory(force=False)
            
            # Reset inference counter
            self._inference_count = 0
            
        except Exception as e:
            # Log error but don't fail inference
            print(f"Warning: Memory cleanup failed: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            cleanup_memory(force=True)
        except (ImportError, ModuleNotFoundError) as e:
            logger.error(f'Error occurred: {e}')


class JITFastStrategicSystem(nn.Module):
    """JIT-compatible strategic system with separate agent inference."""
    
    def __init__(self):
        super().__init__()
        
        self.mlmi_agent = JITFastStrategicActor(input_dim=4, hidden_dim=64)
        self.nwrqk_agent = JITFastStrategicActor(input_dim=6, hidden_dim=64)
        self.mmd_agent = JITFastStrategicActor(input_dim=3, hidden_dim=32)
        
        # Critic for combined state (4+6+3=13)
        self.critic = JITFastCritic(state_dim=13, hidden_dim=64)
    
    def forward(self, mlmi_state: torch.Tensor, nwrqk_state: torch.Tensor, mmd_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return tuple of (mlmi_probs, nwrqk_probs, mmd_probs, value).
        JIT-compatible with tensor-only outputs.
        """
        mlmi_probs = self.mlmi_agent(mlmi_state)
        nwrqk_probs = self.nwrqk_agent(nwrqk_state)
        mmd_probs = self.mmd_agent(mmd_state)
        
        # Combine states for critic
        combined_state = torch.cat([mlmi_state, nwrqk_state, mmd_state], dim=-1)
        value = self.critic(combined_state)
        
        return mlmi_probs, nwrqk_probs, mmd_probs, value


def create_jit_models() -> dict:
    """Create and compile JIT models."""
    
    # Create models
    tactical_system = JITFastTacticalSystem()
    strategic_system = JITFastStrategicSystem()
    
    # Individual agents for separate deployment
    fvg_agent = JITFastTacticalActor("fvg")
    momentum_agent = JITFastTacticalActor("momentum")
    entry_agent = JITFastTacticalActor("entry")
    
    mlmi_agent = JITFastStrategicActor(input_dim=4)
    nwrqk_agent = JITFastStrategicActor(input_dim=6)
    mmd_agent = JITFastStrategicActor(input_dim=3)
    
    critic = JITFastCritic(state_dim=13)
    
    # Set all to eval mode
    models = {
        'tactical_system': tactical_system,
        'strategic_system': strategic_system,
        'fvg_agent': fvg_agent,
        'momentum_agent': momentum_agent,
        'entry_agent': entry_agent,
        'mlmi_agent': mlmi_agent,
        'nwrqk_agent': nwrqk_agent,
        'mmd_agent': mmd_agent,
        'critic': critic
    }
    
    for model in models.values():
        model.eval()
    
    return models


def compile_models_jit(models: dict) -> dict:
    """Compile models using TorchScript with fallback mechanisms."""
    compiled_models = {}
    
    # Example inputs for tracing
    tactical_input = torch.randn(1, 60, 7)
    mlmi_input = torch.randn(1, 4)
    nwrqk_input = torch.randn(1, 6)
    mmd_input = torch.randn(1, 3)
    critic_input = torch.randn(1, 13)
    
    # Compile tactical system with fallback
    try:
        with torch.no_grad():
            traced_tactical = torch.jit.trace(models['tactical_system'], tactical_input)
            traced_tactical = torch.jit.optimize_for_inference(traced_tactical)
            compiled_models['tactical_system'] = traced_tactical
            print("✅ Compiled tactical system")
    except Exception as e:
        print(f"❌ Failed to compile tactical system: {e}")
        print("🔄 Using original model as fallback")
        compiled_models['tactical_system'] = models['tactical_system']
    
    # Compile strategic system with fallback  
    try:
        with torch.no_grad():
            traced_strategic = torch.jit.trace(models['strategic_system'], (mlmi_input, nwrqk_input, mmd_input))
            traced_strategic = torch.jit.optimize_for_inference(traced_strategic)
            compiled_models['strategic_system'] = traced_strategic
            print("✅ Compiled strategic system")
    except Exception as e:
        print(f"❌ Failed to compile strategic system: {e}")
        print("🔄 Using original model as fallback")
        compiled_models['strategic_system'] = models['strategic_system']
    
    # Compile individual agents with fallback
    agent_inputs = {
        'fvg_agent': tactical_input,
        'momentum_agent': tactical_input,
        'entry_agent': tactical_input,
        'mlmi_agent': mlmi_input,
        'nwrqk_agent': nwrqk_input,
        'mmd_agent': mmd_input,
        'critic': critic_input
    }
    
    for agent_name, agent_model in models.items():
        if agent_name in ['tactical_system', 'strategic_system']:
            continue
            
        try:
            with torch.no_grad():
                traced_agent = torch.jit.trace(agent_model, agent_inputs[agent_name])
                traced_agent = torch.jit.optimize_for_inference(traced_agent)
                compiled_models[agent_name] = traced_agent
                print(f"✅ Compiled {agent_name}")
        except Exception as e:
            print(f"❌ Failed to compile {agent_name}: {e}")
            print(f"🔄 Using original {agent_name} as fallback")
            compiled_models[agent_name] = agent_model
    
    return compiled_models


def benchmark_jit_models(compiled_models: dict) -> dict:
    """Benchmark JIT compiled models with memory optimization."""
    
    # Test inputs
    tactical_input = torch.randn(1, 60, 7)
    mlmi_input = torch.randn(1, 4)
    nwrqk_input = torch.randn(1, 6)
    mmd_input = torch.randn(1, 3)
    critic_input = torch.randn(1, 13)
    
    results = {}
    
    for model_name, model in compiled_models.items():
        # Determine input
        if 'tactical' in model_name or model_name in ['fvg_agent', 'momentum_agent', 'entry_agent']:
            test_input = tactical_input
        elif model_name == 'strategic_system':
            test_input = (mlmi_input, nwrqk_input, mmd_input)
        elif model_name == 'mlmi_agent':
            test_input = mlmi_input
        elif model_name == 'nwrqk_agent':
            test_input = nwrqk_input
        elif model_name == 'mmd_agent':
            test_input = mmd_input
        else:  # critic
            test_input = critic_input
        
        # Memory cleanup before benchmark
        cleanup_memory(force=True)
        
        # Warm up with memory tracking
        with torch.no_grad():
            for _ in range(20):
                if isinstance(test_input, tuple):
                    _ = model(*test_input)
                else:
                    _ = model(test_input)
                
                # Periodic cleanup during warmup
                if _ % 10 == 0:
                    cleanup_memory(force=False)
        
        # Final cleanup before benchmark
        cleanup_memory(force=True)
        
        # Benchmark with memory monitoring
        latencies = []
        memory_usage = []
        
        with torch.no_grad():
            for i in range(1000):
                # Memory cleanup every 100 iterations
                if i % 100 == 0:
                    cleanup_memory(force=False)
                
                # Track memory before inference
                if torch.cuda.is_available():
                    mem_before = torch.cuda.memory_allocated()
                else:
                    mem_before = 0
                
                start_time = time.perf_counter()
                if isinstance(test_input, tuple):
                    _ = model(*test_input)
                else:
                    _ = model(test_input)
                end_time = time.perf_counter()
                
                # Track memory after inference
                if torch.cuda.is_available():
                    mem_after = torch.cuda.memory_allocated()
                else:
                    mem_after = 0
                
                latencies.append((end_time - start_time) * 1000)
                memory_usage.append(mem_after - mem_before)
        
        # Final cleanup
        cleanup_memory(force=True)
        
        results[model_name] = {
            'avg_latency_ms': np.mean(latencies),
            'p99_latency_ms': np.percentile(latencies, 99),
            'meets_target': np.percentile(latencies, 99) < 5.0,
            'throughput_qps': 1000 / np.mean(latencies),
            'avg_memory_usage_mb': np.mean(memory_usage) / (1024 * 1024),
            'max_memory_usage_mb': np.max(memory_usage) / (1024 * 1024),
            'memory_stable': np.std(memory_usage) < (1024 * 1024)  # <1MB std dev
        }
    
    return results


def save_jit_models(compiled_models: dict, save_dir: str = "models/jit_optimized"):
    """Save JIT compiled models."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    for model_name, model in compiled_models.items():
        model_path = os.path.join(save_dir, f"{model_name}_jit.pt")
        torch.jit.save(model, model_path)
        print(f"💾 Saved {model_name} to {model_path}")


if __name__ == "__main__":
    print("🚀 Creating JIT-Compatible Ultra-Fast Models")
    
    # Create models
    models = create_jit_models()
    
    # Compile with JIT
    compiled_models = compile_models_jit(models)
    
    # Benchmark performance
    results = benchmark_jit_models(compiled_models)
    
    # Save models
    save_jit_models(compiled_models)
    
    # Display results
    print("\n" + "="*80)
    print("📊 JIT-COMPILED MODEL PERFORMANCE RESULTS")
    print("="*80)
    
    tactical_p99 = results.get('tactical_system', {}).get('p99_latency_ms', 999)
    strategic_p99 = results.get('strategic_system', {}).get('p99_latency_ms', 999)
    
    print(f"\n🎯 SYSTEM PERFORMANCE:")
    print(f"  Tactical System:   {tactical_p99:6.2f}ms p99 {'✅ PASS' if tactical_p99 < 5.0 else '❌ FAIL'}")
    print(f"  Strategic System:  {strategic_p99:6.2f}ms p99 {'✅ PASS' if strategic_p99 < 5.0 else '❌ FAIL'}")
    
    print(f"\n🔧 INDIVIDUAL AGENT PERFORMANCE:")
    for model_name, stats in results.items():
        if model_name not in ['tactical_system', 'strategic_system']:
            status = "✅ PASS" if stats['meets_target'] else "❌ FAIL"
            print(f"  {model_name:15}: {stats['p99_latency_ms']:6.2f}ms p99 | {stats['throughput_qps']:8.1f} QPS {status}")
    
    # Final assessment
    systems_ready = (tactical_p99 < 5.0) + (strategic_p99 < 5.0)
    
    print("\n" + "="*80)
    print("🏆 FINAL PERFORMANCE ASSESSMENT")
    print("="*80)
    
    if systems_ready == 2:
        print("✅ SUCCESS: ALL SYSTEMS MEET <5MS REQUIREMENT!")
        print("🚀 Models are ready for production deployment")
        print("📦 JIT-compiled models saved for immediate use")
    else:
        print(f"⚠️ WARNING: {2-systems_ready} system(s) need further optimization")
        
    print("="*80)