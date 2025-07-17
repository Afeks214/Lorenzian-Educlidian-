#!/usr/bin/env python3
"""
GrandModel Colab Environment Verification Script
Comprehensive testing of the rebuilt training environment
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"🔍 {title}")
    print("="*60)

def print_success(message):
    """Print success message"""
    print(f"✅ {message}")

def print_error(message):
    """Print error message"""
    print(f"❌ {message}")

def print_warning(message):
    """Print warning message"""
    print(f"⚠️ {message}")

def print_info(message):
    """Print info message"""
    print(f"ℹ️ {message}")

def verify_directory_structure():
    """Verify the colab directory structure"""
    print_header("Directory Structure Verification")
    
    base_dir = Path("/home/QuantNova/GrandModel/colab")
    
    required_dirs = [
        "notebooks",
        "trainers", 
        "utils",
        "configs",
        "data",
        "exports"
    ]
    
    required_files = [
        "notebooks/tactical_mappo_training.ipynb",
        "notebooks/strategic_mappo_training.ipynb",
        "trainers/__init__.py",
        "trainers/tactical_mappo_trainer.py",
        "trainers/strategic_mappo_trainer.py",
        "utils/__init__.py",
        "utils/gpu_optimizer.py",
        "configs/training_config.yaml",
        "data/NQ - 5 min - ETH.csv",
        "data/NQ - 30 min - ETH.csv",
        "README.md"
    ]
    
    # Check directories
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print_success(f"Directory exists: {dir_name}/")
        else:
            print_error(f"Missing directory: {dir_name}/")
    
    # Check files
    for file_path in required_files:
        full_path = base_dir / file_path
        if full_path.exists():
            print_success(f"File exists: {file_path}")
        else:
            print_error(f"Missing file: {file_path}")
    
    return True

def verify_data_files():
    """Verify data files are properly formatted"""
    print_header("Data Files Verification")
    
    base_dir = Path("/home/QuantNova/GrandModel/colab/data")
    
    data_files = [
        "NQ - 5 min - ETH.csv",
        "NQ - 30 min - ETH.csv"
    ]
    
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    for file_name in data_files:
        file_path = base_dir / file_name
        
        try:
            df = pd.read_csv(file_path)
            
            # Check columns
            if all(col in df.columns for col in required_columns):
                print_success(f"{file_name}: Columns correct")
            else:
                print_error(f"{file_name}: Missing required columns")
                print_info(f"  Required: {required_columns}")
                print_info(f"  Found: {list(df.columns)}")
            
            # Check data shape
            print_info(f"{file_name}: Shape {df.shape}")
            
            # Check date format
            try:
                pd.to_datetime(df['Date'])
                print_success(f"{file_name}: Date format valid")
            except (FileNotFoundError, IOError, OSError) as e:
                print_error(f"{file_name}: Invalid date format")
            
            # Check for missing values
            missing_count = df.isnull().sum().sum()
            if missing_count == 0:
                print_success(f"{file_name}: No missing values")
            else:
                print_warning(f"{file_name}: {missing_count} missing values")
            
            # Check data range
            if len(df) >= 100:
                print_success(f"{file_name}: Sufficient data ({len(df)} rows)")
            else:
                print_warning(f"{file_name}: Limited data ({len(df)} rows)")
                
        except Exception as e:
            print_error(f"{file_name}: Failed to load - {e}")

def verify_imports():
    """Verify all imports work correctly"""
    print_header("Import Verification")
    
    # Test basic imports
    try:
        import torch
        print_success(f"PyTorch: {torch.__version__}")
        print_info(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print_info(f"  GPU: {torch.cuda.get_device_name()}")
    except ImportError as e:
        print_error(f"PyTorch import failed: {e}")
    
    try:
        import pandas as pd
        print_success(f"Pandas: {pd.__version__}")
    except ImportError as e:
        print_error(f"Pandas import failed: {e}")
    
    try:
        import numpy as np
        print_success(f"NumPy: {np.__version__}")
    except ImportError as e:
        print_error(f"NumPy import failed: {e}")
    
    try:
        import matplotlib.pyplot as plt
        print_success("Matplotlib imported successfully")
    except ImportError as e:
        print_error(f"Matplotlib import failed: {e}")
    
    # Test project imports
    sys.path.append("/home/QuantNova/GrandModel")
    
    try:
        from colab.trainers.tactical_mappo_trainer import TacticalMAPPOTrainer
        print_success("TacticalMAPPOTrainer imported successfully")
    except ImportError as e:
        print_error(f"TacticalMAPPOTrainer import failed: {e}")
    
    try:
        from colab.trainers.strategic_mappo_trainer import StrategicMAPPOTrainer
        print_success("StrategicMAPPOTrainer imported successfully")
    except ImportError as e:
        print_error(f"StrategicMAPPOTrainer import failed: {e}")
    
    try:
        from colab.utils.gpu_optimizer import GPUOptimizer, setup_colab_environment
        print_success("GPU utilities imported successfully")
    except ImportError as e:
        print_error(f"GPU utilities import failed: {e}")

def verify_trainer_initialization():
    """Verify trainers can be initialized"""
    print_header("Trainer Initialization Verification")
    
    sys.path.append("/home/QuantNova/GrandModel")
    
    try:
        from colab.trainers.tactical_mappo_trainer import TacticalMAPPOTrainer
        
        trainer = TacticalMAPPOTrainer(
            state_dim=7,
            action_dim=5,
            n_agents=3,
            device='cpu'  # Use CPU for testing
        )
        
        print_success("TacticalMAPPOTrainer initialized successfully")
        print_info(f"  Device: {trainer.device}")
        print_info(f"  Agents: {trainer.n_agents}")
        print_info(f"  State dim: {trainer.state_dim}")
        print_info(f"  Action dim: {trainer.action_dim}")
        
    except Exception as e:
        print_error(f"TacticalMAPPOTrainer initialization failed: {e}")
    
    try:
        from colab.trainers.strategic_mappo_trainer import StrategicMAPPOTrainer
        
        strategic_trainer = StrategicMAPPOTrainer(
            state_dim=13,
            action_dim=7,
            n_agents=3,
            device='cpu'  # Use CPU for testing
        )
        
        print_success("StrategicMAPPOTrainer initialized successfully")
        print_info(f"  Device: {strategic_trainer.device}")
        print_info(f"  Agents: {strategic_trainer.n_agents}")
        print_info(f"  State dim: {strategic_trainer.state_dim}")
        print_info(f"  Action dim: {strategic_trainer.action_dim}")
        
    except Exception as e:
        print_error(f"StrategicMAPPOTrainer initialization failed: {e}")

def verify_gpu_optimizer():
    """Verify GPU optimizer functionality"""
    print_header("GPU Optimizer Verification")
    
    sys.path.append("/home/QuantNova/GrandModel")
    
    try:
        from colab.utils.gpu_optimizer import GPUOptimizer, setup_colab_environment
        
        # Test GPU optimizer initialization
        gpu_optimizer = GPUOptimizer()
        print_success("GPUOptimizer initialized successfully")
        print_info(f"  Device: {gpu_optimizer.device}")
        
        # Test memory monitoring
        memory_info = gpu_optimizer.monitor_memory()
        print_success("Memory monitoring working")
        print_info(f"  System memory: {memory_info['system_memory_percent']:.1f}%")
        
        # Test setup function
        setup_optimizer = setup_colab_environment()
        print_success("Setup colab environment working")
        
        # Test recommendations
        recommendations = gpu_optimizer.get_optimization_recommendations()
        print_success(f"Optimization recommendations: {len(recommendations)} items")
        
    except Exception as e:
        print_error(f"GPU optimizer verification failed: {e}")

def verify_configuration():
    """Verify configuration files"""
    print_header("Configuration Verification")
    
    config_path = Path("/home/QuantNova/GrandModel/colab/configs/training_config.yaml")
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print_success("Configuration file loaded successfully")
        
        # Check main sections
        required_sections = ['global', 'data', 'tactical', 'strategic', 'market', 'risk']
        for section in required_sections:
            if section in config:
                print_success(f"Config section '{section}' found")
            else:
                print_error(f"Config section '{section}' missing")
                
    except ImportError:
        print_warning("PyYAML not installed - configuration verification skipped")
    except Exception as e:
        print_error(f"Configuration verification failed: {e}")

def verify_notebook_structure():
    """Verify Jupyter notebook structure"""
    print_header("Notebook Structure Verification")
    
    notebooks = [
        "/home/QuantNova/GrandModel/colab/notebooks/tactical_mappo_training.ipynb",
        "/home/QuantNova/GrandModel/colab/notebooks/strategic_mappo_training.ipynb"
    ]
    
    for notebook_path in notebooks:
        try:
            with open(notebook_path, 'r') as f:
                notebook_data = json.load(f)
            
            notebook_name = Path(notebook_path).name
            
            # Check notebook structure
            if 'cells' in notebook_data:
                num_cells = len(notebook_data['cells'])
                print_success(f"{notebook_name}: {num_cells} cells")
            else:
                print_error(f"{notebook_name}: Invalid notebook structure")
            
            # Check for key sections
            cell_sources = []
            for cell in notebook_data.get('cells', []):
                if 'source' in cell:
                    cell_sources.extend(cell['source'])
            
            full_source = '\n'.join(cell_sources)
            
            key_sections = [
                'Setup and Installation',
                'Import Libraries',
                'Training Configuration',
                'Model Export'
            ]
            
            for section in key_sections:
                if section.lower() in full_source.lower():
                    print_success(f"{notebook_name}: '{section}' section found")
                else:
                    print_warning(f"{notebook_name}: '{section}' section not found")
                    
        except Exception as e:
            print_error(f"Notebook verification failed for {notebook_path}: {e}")

def verify_sample_training():
    """Verify a minimal training run works"""
    print_header("Sample Training Verification")
    
    sys.path.append("/home/QuantNova/GrandModel")
    
    try:
        # Load sample data
        data_path = "/home/QuantNova/GrandModel/colab/data/NQ - 5 min - ETH.csv"
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        print_success("Sample data loaded successfully")
        
        # Initialize trainer
        from colab.trainers.tactical_mappo_trainer import TacticalMAPPOTrainer
        
        trainer = TacticalMAPPOTrainer(
            state_dim=7,
            action_dim=5,
            n_agents=3,
            device='cpu'
        )
        
        print_success("Trainer initialized for sample run")
        
        # Test single episode (very short)
        if len(df) >= 200:
            episode_reward, episode_steps = trainer.train_episode(
                data=df,
                start_idx=60,
                episode_length=100  # Very short for testing
            )
            
            print_success(f"Sample training episode completed")
            print_info(f"  Episode reward: {episode_reward:.3f}")
            print_info(f"  Episode steps: {episode_steps}")
            print_info(f"  Training stats: {trainer.get_training_stats()}")
        else:
            print_warning("Insufficient data for sample training run")
            
    except Exception as e:
        print_error(f"Sample training verification failed: {e}")

def run_verification():
    """Run complete verification suite"""
    print_header("GrandModel Colab Environment Verification")
    print_info(f"Verification started at: {datetime.now()}")
    print_info(f"Python version: {sys.version}")
    
    # Run all verification checks
    verify_directory_structure()
    verify_data_files()
    verify_imports()
    verify_trainer_initialization()
    verify_gpu_optimizer()
    verify_configuration()
    verify_notebook_structure()
    verify_sample_training()
    
    print_header("Verification Complete")
    print_success("GrandModel Colab Environment verification finished!")
    print_info("If all checks passed, the environment is ready for training.")
    print_info("To start training:")
    print_info("  1. Open tactical_mappo_training.ipynb for 5-minute tactical training")
    print_info("  2. Open strategic_mappo_training.ipynb for 30-minute strategic training")

if __name__ == "__main__":
    run_verification()