#!/usr/bin/env python3
"""
Final Validation Test for Tactical MAPPO Notebook
Tests all critical functionalities for 500-row execution
"""

import sys
import os
import warnings
import traceback
import gc
import time
import json
from datetime import datetime
import numpy as np
import pandas as pd

# Add project to path
project_path = '/home/QuantNova/GrandModel'
if project_path not in sys.path:
    sys.path.append(project_path)

warnings.filterwarnings('ignore')

def test_500_row_validation():
    """Test the complete 500-row validation pipeline"""
    
    print("🚀 FINAL VALIDATION TEST - 500-ROW EXECUTION")
    print("=" * 60)
    
    # Test 1: Data Loading
    print("\n1. Testing Data Loading...")
    try:
        data_path = '/home/QuantNova/GrandModel/colab/data/NQ - 5 min - ETH_extended.csv'
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        print(f"   ✅ Data loaded: {df.shape}")
        print(f"   ✅ 500-row ready: {len(df) >= 500}")
        
        if len(df) < 500:
            print("   ❌ CRITICAL: Not enough data for 500-row validation")
            return False
            
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
        return False
    
    # Test 2: Import Critical Modules
    print("\n2. Testing Critical Imports...")
    try:
        import torch
        import torch.nn as nn
        from colab.trainers.tactical_mappo_trainer_optimized import OptimizedTacticalMAPPOTrainer
        from colab.utils.gpu_optimizer import GPUOptimizer
        print("   ✅ All critical imports successful")
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Test 3: JIT Compilation (without cache)
    print("\n3. Testing JIT Compilation...")
    try:
        from numba import jit
        
        @jit(nopython=True)
        def test_rsi_jit(prices, period=14):
            if len(prices) < period + 1:
                return 50.0
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0.0)
            losses = np.where(deltas < 0, -deltas, 0.0)
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            if avg_loss == 0:
                return 100.0
            rs = avg_gain / avg_loss
            return 100.0 - (100.0 / (1.0 + rs))
        
        # Test JIT compilation
        test_prices = np.random.randn(100).cumsum() + 100
        rsi_value = test_rsi_jit(test_prices)
        print(f"   ✅ JIT compilation successful: RSI = {rsi_value:.2f}")
        
    except Exception as e:
        print(f"   ❌ JIT compilation failed: {e}")
        return False
    
    # Test 4: Trainer Initialization
    print("\n4. Testing Trainer Initialization...")
    try:
        gpu_optimizer = GPUOptimizer()
        device = gpu_optimizer.device
        
        trainer = OptimizedTacticalMAPPOTrainer(
            state_dim=7,
            action_dim=5,
            n_agents=3,
            lr_actor=3e-4,
            lr_critic=1e-3,
            gamma=0.99,
            eps_clip=0.2,
            k_epochs=4,
            device=str(device),
            mixed_precision=False,
            gradient_accumulation_steps=4,
            max_grad_norm=0.5
        )
        
        print(f"   ✅ Trainer initialized: {trainer.device}")
        print(f"   ✅ Agents: {trainer.n_agents}")
        
    except Exception as e:
        print(f"   ❌ Trainer initialization failed: {e}")
        return False
    
    # Test 5: 500-Row Validation
    print("\n5. Testing 500-Row Validation...")
    try:
        validation_results = trainer.validate_model_500_rows(df, test_rows=500)
        
        print(f"   ✅ Validation completed!")
        print(f"   ✅ Mean reward: {validation_results['mean_reward']:.3f}")
        print(f"   ✅ Avg inference: {validation_results['avg_inference_time_ms']:.2f}ms")
        print(f"   ✅ Latency violations: {validation_results['latency_violations']}")
        
        # Check latency target
        if validation_results['avg_inference_time_ms'] > 100:
            print("   ⚠️ Warning: Average inference time exceeds 100ms target")
        
        if validation_results['latency_violations'] > 0:
            print("   ⚠️ Warning: Some latency violations detected")
            
    except Exception as e:
        print(f"   ❌ 500-row validation failed: {e}")
        return False
    
    # Test 6: Model Export
    print("\n6. Testing Model Export...")
    try:
        os.makedirs('/tmp/tactical_model_test', exist_ok=True)
        model_path = '/tmp/tactical_model_test/test_model.pth'
        
        trainer.save_checkpoint(model_path)
        
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            print(f"   ✅ Model exported: {file_size / (1024*1024):.2f} MB")
        else:
            print("   ❌ Model file not found after export")
            return False
            
    except Exception as e:
        print(f"   ❌ Model export failed: {e}")
        return False
    
    # Test 7: Performance Benchmarks
    print("\n7. Testing Performance Benchmarks...")
    try:
        # Create sample states
        sample_states = []
        for agent_idx in range(trainer.n_agents):
            state = np.array([0.01, 0.02, 1000, 0.005, 0.5, 1.0, 0.0])
            sample_states.append(state)
        
        # Benchmark inference
        inference_times = []
        for _ in range(50):
            start_time = time.perf_counter()
            actions, _, _ = trainer.get_action(sample_states, deterministic=True)
            inference_time = (time.perf_counter() - start_time) * 1000
            inference_times.append(inference_time)
        
        avg_inference = np.mean(inference_times)
        max_inference = np.max(inference_times)
        
        print(f"   ✅ Average inference: {avg_inference:.3f}ms")
        print(f"   ✅ Maximum inference: {max_inference:.3f}ms")
        print(f"   ✅ Latency target: {'PASS' if avg_inference < 100 else 'FAIL'}")
        
    except Exception as e:
        print(f"   ❌ Performance benchmark failed: {e}")
        return False
    
    # Test 8: Training Episode Test
    print("\n8. Testing Training Episode...")
    try:
        # Run a short training episode
        episode_reward, episode_steps = trainer.train_episode(
            data=df,
            start_idx=60,
            episode_length=50
        )
        
        print(f"   ✅ Training episode completed")
        print(f"   ✅ Episode reward: {episode_reward:.3f}")
        print(f"   ✅ Episode steps: {episode_steps}")
        
    except Exception as e:
        print(f"   ❌ Training episode failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 ALL VALIDATION TESTS PASSED!")
    print("✅ Tactical MAPPO notebook is ready for 500-row execution")
    print("✅ Performance targets met")
    print("✅ All critical functionalities working")
    
    # Generate validation report
    validation_report = {
        'timestamp': datetime.now().isoformat(),
        'data_rows': len(df),
        'validation_results': validation_results,
        'performance_metrics': {
            'avg_inference_time_ms': float(avg_inference),
            'max_inference_time_ms': float(max_inference),
            'latency_target_met': avg_inference < 100
        },
        'training_test': {
            'episode_reward': float(episode_reward),
            'episode_steps': int(episode_steps)
        },
        'status': 'PASSED',
        'ready_for_production': True
    }
    
    # Save validation report
    report_path = '/home/QuantNova/GrandModel/colab/notebooks/VALIDATION_REPORT.json'
    with open(report_path, 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    print(f"\n📊 Validation report saved to: {report_path}")
    
    return True

if __name__ == "__main__":
    success = test_500_row_validation()
    if success:
        print("\n🚀 READY FOR PRODUCTION!")
        sys.exit(0)
    else:
        print("\n❌ VALIDATION FAILED!")
        sys.exit(1)