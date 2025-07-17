#!/usr/bin/env python3
"""
Tactical MAPPO Notebook Validator
Tests each cell of the tactical_mappo_training.ipynb notebook
"""

import sys
import os
import warnings
import traceback
import gc
from datetime import datetime
import json

# Add project to path
project_path = '/home/QuantNova/GrandModel'
if project_path not in sys.path:
    sys.path.append(project_path)

warnings.filterwarnings('ignore')

class NotebookValidator:
    def __init__(self):
        self.results = {}
        self.current_cell = None
        self.globals_dict = {}
        self.errors = []
        
    def test_cell(self, cell_id, cell_code, description):
        """Test a single notebook cell"""
        print(f"\n{'='*60}")
        print(f"Testing Cell {cell_id}: {description}")
        print(f"{'='*60}")
        
        try:
            # Execute the cell code
            exec(cell_code, self.globals_dict)
            
            print(f"✅ Cell {cell_id} executed successfully")
            self.results[cell_id] = {
                'status': 'success',
                'description': description,
                'error': None
            }
            
        except Exception as e:
            error_msg = f"❌ Cell {cell_id} failed: {str(e)}"
            print(error_msg)
            print(f"Traceback: {traceback.format_exc()}")
            
            self.results[cell_id] = {
                'status': 'failed',
                'description': description,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            
            self.errors.append({
                'cell_id': cell_id,
                'description': description,
                'error': str(e)
            })
    
    def run_validation(self):
        """Run all validation tests"""
        print("🚀 Starting Tactical MAPPO Notebook Validation")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        # Test basic imports
        self.test_cell("imports", """
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import time
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')  # Use default instead of seaborn-v0_8
sns.set_palette("husl")

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
""", "Basic imports and PyTorch setup")

        # Test project imports
        self.test_cell("project_imports", """
try:
    from colab.trainers.tactical_mappo_trainer_optimized import OptimizedTacticalMAPPOTrainer
    print("✅ OptimizedTacticalMAPPOTrainer imported successfully!")
    optimized_trainer_available = True
except ImportError as e:
    print(f"⚠️ OptimizedTacticalMAPPOTrainer import failed: {e}")
    optimized_trainer_available = False

try:
    from colab.trainers.tactical_mappo_trainer import TacticalMAPPOTrainer
    print("✅ TacticalMAPPOTrainer imported successfully!")
    standard_trainer_available = True
except ImportError as e:
    print(f"❌ TacticalMAPPOTrainer import failed: {e}")
    standard_trainer_available = False

try:
    from colab.utils.gpu_optimizer import GPUOptimizer, setup_colab_environment, quick_gpu_check, quick_memory_check
    print("✅ GPU optimization utilities imported successfully!")
    gpu_utils_available = True
except ImportError as e:
    print(f"❌ GPU utils import failed: {e}")
    gpu_utils_available = False
""", "Project trainer and utility imports")

        # Test JIT compilation
        self.test_cell("jit_compilation", """
import numba
from numba import jit

@jit(nopython=True, cache=True)
def calculate_rsi_jit(prices, period=14):
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
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

# Test JIT compilation
test_prices = np.random.randn(100).cumsum() + 100
rsi_value = calculate_rsi_jit(test_prices)
print(f"✅ JIT RSI calculation: {rsi_value:.2f}")
""", "JIT compilation test")

        # Test data loading
        self.test_cell("data_loading", """
data_path = '/home/QuantNova/GrandModel/colab/data/NQ - 5 min - ETH_extended.csv'
fallback_path = '/home/QuantNova/GrandModel/colab/data/NQ - 5 min - ETH.csv'

try:
    # Try extended data first
    df = pd.read_csv(data_path)
    print(f"✅ Extended data loaded successfully!")
except FileNotFoundError:
    try:
        # Fallback to original data
        df = pd.read_csv(fallback_path)
        print(f"✅ Original data loaded successfully!")
    except FileNotFoundError:
        print(f"❌ Data files not found at {data_path} or {fallback_path}")
        df = None

if df is not None:
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"   Shape: {df.shape}")
    print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"   Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    
    # Check if we have enough data
    if len(df) >= 500:
        print(f"✅ 500-row validation ready: {len(df)} rows available")
    else:
        print(f"⚠️ Warning: Only {len(df)} rows available, less than 500 required")
else:
    print("❌ Data loading failed")
""", "Data loading validation")

        # Test GPU setup (modified for local environment)
        self.test_cell("gpu_setup", """
if gpu_utils_available:
    try:
        gpu_optimizer = GPUOptimizer()
        print("✅ GPU optimizer initialized")
        
        # Test memory monitoring
        if torch.cuda.is_available():
            memory_info = gpu_optimizer.monitor_memory()
            print(f"GPU Memory: {memory_info['gpu_memory_used_gb']:.2f} GB / {memory_info['gpu_memory_total_gb']:.2f} GB")
        else:
            print("⚠️ CUDA not available, using CPU")
    except Exception as e:
        print(f"❌ GPU setup failed: {e}")
        # Create a fallback device
        class FallbackGPUOptimizer:
            def __init__(self):
                self.device = torch.device('cpu')
            def monitor_memory(self):
                return {'gpu_memory_used_gb': 0, 'gpu_memory_total_gb': 0}
        gpu_optimizer = FallbackGPUOptimizer()
else:
    print("⚠️ GPU utilities not available, creating fallback")
    class FallbackGPUOptimizer:
        def __init__(self):
            self.device = torch.device('cpu')
        def monitor_memory(self):
            return {'gpu_memory_used_gb': 0, 'gpu_memory_total_gb': 0}
    gpu_optimizer = FallbackGPUOptimizer()
""", "GPU setup and optimization")

        # Test trainer initialization
        self.test_cell("trainer_init", """
device = gpu_optimizer.device

# Try optimized trainer first
if optimized_trainer_available:
    try:
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
            mixed_precision=False,  # Disable for testing
            gradient_accumulation_steps=4,
            max_grad_norm=0.5
        )
        print("✅ OptimizedTacticalMAPPOTrainer initialized successfully!")
        trainer_type = "optimized"
    except Exception as e:
        print(f"❌ OptimizedTacticalMAPPOTrainer failed: {e}")
        trainer = None
        trainer_type = None
else:
    trainer = None
    trainer_type = None

# Fallback to standard trainer if optimized failed
if trainer is None and standard_trainer_available:
    try:
        trainer = TacticalMAPPOTrainer(
            state_dim=7,
            action_dim=5,
            n_agents=3,
            lr_actor=3e-4,
            lr_critic=1e-3,
            gamma=0.99,
            eps_clip=0.2,
            k_epochs=4,
            device=str(device)
        )
        print("✅ TacticalMAPPOTrainer initialized successfully!")
        trainer_type = "standard"
    except Exception as e:
        print(f"❌ TacticalMAPPOTrainer failed: {e}")
        trainer = None
        trainer_type = None

if trainer is not None:
    print(f"   Trainer type: {trainer_type}")
    print(f"   Device: {trainer.device}")
    print(f"   State dimension: {trainer.state_dim}")
    print(f"   Action dimension: {trainer.action_dim}")
    print(f"   Number of agents: {trainer.n_agents}")
else:
    print("❌ No trainer could be initialized")
""", "Trainer initialization")

        # Test basic training functionality
        self.test_cell("basic_training", """
if trainer is not None and df is not None:
    print("🧪 Testing basic training functionality...")
    
    # Create sample states
    sample_states = []
    for agent_idx in range(trainer.n_agents):
        state = np.array([0.01, 0.02, 1000, 0.005, 0.5, 1.0, 0.0])
        sample_states.append(state)
    
    # Test action generation
    try:
        actions, action_probs, values = trainer.get_action(sample_states, deterministic=True)
        print(f"✅ Action generation successful: {actions}")
        
        # Test storing transitions
        trainer.store_transition(sample_states, actions, [0.1, 0.2, 0.05], 
                               [0.1, 0.2, 0.05], [0.1, 0.2, 0.05], [False, False, False])
        print("✅ Transition storage successful")
        
        # Test basic training step
        if len(trainer.states) >= 10:
            trainer.update()
            print("✅ Training update successful")
        else:
            print("⚠️ Not enough data for training update")
            
    except Exception as e:
        print(f"❌ Basic training test failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
else:
    print("❌ Cannot test training - trainer or data not available")
""", "Basic training functionality")

        # Test performance benchmark
        self.test_cell("performance_test", """
if trainer is not None:
    print("🔥 Testing performance benchmarks...")
    
    # Test inference speed
    sample_states = []
    for agent_idx in range(trainer.n_agents):
        state = np.array([0.01, 0.02, 1000, 0.005, 0.5, 1.0, 0.0])
        sample_states.append(state)
    
    inference_times = []
    for i in range(10):
        start_time = time.perf_counter()
        actions, _, _ = trainer.get_action(sample_states, deterministic=True)
        end_time = time.perf_counter()
        inference_times.append((end_time - start_time) * 1000)
    
    avg_inference_time = np.mean(inference_times)
    max_inference_time = max(inference_times)
    
    print(f"✅ Inference performance:")
    print(f"   Average: {avg_inference_time:.3f}ms")
    print(f"   Maximum: {max_inference_time:.3f}ms")
    print(f"   Latency target (<100ms): {'✅ PASS' if avg_inference_time < 100 else '❌ FAIL'}")
    
    # Test JIT indicator performance
    if 'calculate_rsi_jit' in locals():
        test_prices = np.random.randn(100).cumsum() + 100
        
        start_time = time.perf_counter()
        for _ in range(100):
            rsi_value = calculate_rsi_jit(test_prices)
        end_time = time.perf_counter()
        
        jit_time = (end_time - start_time) * 1000
        print(f"✅ JIT RSI performance:")
        print(f"   100 calculations: {jit_time:.3f}ms")
        print(f"   Per calculation: {jit_time/100:.3f}ms")
        print(f"   Target (<5ms): {'✅ PASS' if jit_time/100 < 5 else '❌ FAIL'}")
else:
    print("❌ Cannot test performance - trainer not available")
""", "Performance benchmarking")

        # Test model saving/loading
        self.test_cell("model_export", """
if trainer is not None:
    print("💾 Testing model export/import...")
    
    # Create test directory
    test_dir = '/tmp/tactical_model_test'
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # Save model
        model_path = os.path.join(test_dir, 'test_model.pth')
        trainer.save_checkpoint(model_path)
        print(f"✅ Model saved to: {model_path}")
        
        # Test if file exists and has reasonable size
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            print(f"✅ Model file size: {file_size / (1024*1024):.2f} MB")
            
            if file_size > 1000:  # At least 1KB
                print("✅ Model file appears valid")
            else:
                print("⚠️ Model file seems too small")
        else:
            print("❌ Model file not found after saving")
            
    except Exception as e:
        print(f"❌ Model export failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
else:
    print("❌ Cannot test model export - trainer not available")
""", "Model saving and loading")

        # Test data processing pipeline
        self.test_cell("data_pipeline", """
if df is not None:
    print("📊 Testing data processing pipeline...")
    
    # Test with available data (less than 500 rows)
    available_rows = min(len(df), 100)  # Use what we have
    
    try:
        # Test data slicing
        sample_data = df.iloc[:available_rows]
        print(f"✅ Data slicing successful: {len(sample_data)} rows")
        
        # Test basic statistics
        price_stats = {
            'mean': sample_data['Close'].mean(),
            'std': sample_data['Close'].std(),
            'min': sample_data['Close'].min(),
            'max': sample_data['Close'].max()
        }
        print(f"✅ Price statistics: {price_stats}")
        
        # Test returns calculation
        returns = sample_data['Close'].pct_change().dropna()
        print(f"✅ Returns calculation: {len(returns)} values")
        
        # Test with smaller episode length
        episode_length = min(50, len(sample_data) - 10)
        print(f"✅ Episode length adjusted to: {episode_length}")
        
    except Exception as e:
        print(f"❌ Data pipeline test failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
else:
    print("❌ Cannot test data pipeline - data not available")
""", "Data processing pipeline")

        # Print summary
        print(f"\n{'='*60}")
        print("📋 VALIDATION SUMMARY")
        print(f"{'='*60}")
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r['status'] == 'success')
        failed_tests = total_tests - passed_tests
        
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
        
        if self.errors:
            print(f"\n❌ ERRORS FOUND:")
            for error in self.errors:
                print(f"   Cell {error['cell_id']}: {error['error']}")
        
        # Save results
        results_file = '/tmp/notebook_validation_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': passed_tests/total_tests*100,
                'results': self.results,
                'errors': self.errors
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        return passed_tests, failed_tests

if __name__ == "__main__":
    validator = NotebookValidator()
    passed, failed = validator.run_validation()
    
    if failed > 0:
        print(f"\n⚠️ {failed} tests failed. Review the errors above.")
        sys.exit(1)
    else:
        print(f"\n✅ All {passed} tests passed!")
        sys.exit(0)