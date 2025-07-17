# Getting Started with GrandModel

## Overview

This guide will walk you through setting up and running the GrandModel MARL trading system for the first time. By the end of this guide, you'll have a fully functional trading system running on your local machine.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [First Run](#first-run)
- [Basic Operations](#basic-operations)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 4 cores, 2.5GHz+
- RAM: 8GB (16GB recommended)
- Storage: 10GB free space
- OS: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10/11 with WSL2

**Recommended Requirements:**
- CPU: 8 cores, 3.0GHz+
- RAM: 16GB+
- Storage: 50GB SSD free space
- OS: Linux (Ubuntu 22.04+)

### Software Dependencies

**Required:**
- Python 3.12+
- Git
- Docker & Docker Compose
- pip (Python package manager)

**Optional but Recommended:**
- CUDA-compatible GPU (for accelerated training)
- Visual Studio Code with Python extension
- PyCharm Professional

### Installation Prerequisites

1. **Install Python 3.12:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-pip

# macOS (using Homebrew)
brew install python@3.12

# Windows (using chocolatey)
choco install python --version=3.12.0
```

2. **Install Docker:**
```bash
# Ubuntu/Debian
sudo apt install docker.io docker-compose
sudo usermod -aG docker $USER

# macOS
brew install docker docker-compose

# Windows
# Download Docker Desktop from https://docker.com
```

3. **Install Git:**
```bash
# Ubuntu/Debian
sudo apt install git

# macOS
brew install git

# Windows
# Download from https://git-scm.com
```

## Installation

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/Afeks214/GrandModel.git
cd GrandModel

# Verify repository structure
ls -la
```

Expected output:
```
total 128
drwxr-xr-x  15 user user  4096 Jul 13 10:00 .
drwxr-xr-x   3 user user  4096 Jul 13 10:00 ..
drwxr-xr-x   8 user user  4096 Jul 13 10:00 .git
-rw-r--r--   1 user user  1234 Jul 13 10:00 CLAUDE.md
drwxr-xr-x   3 user user  4096 Jul 13 10:00 docker
drwxr-xr-x   5 user user  4096 Jul 13 10:00 docs
-rw-r--r--   1 user user  2048 Jul 13 10:00 production_config.yaml
-rw-r--r--   1 user user  5678 Jul 13 10:00 README.md
-rw-r--r--   1 user user   512 Jul 13 10:00 requirements.txt
drwxr-xr-x   2 user user  4096 Jul 13 10:00 scripts
drwxr-xr-x  15 user user  4096 Jul 13 10:00 src
drwxr-xr-x   3 user user  4096 Jul 13 10:00 tests
```

### Step 2: Create Python Virtual Environment

```bash
# Create virtual environment
python3.12 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate  # Windows

# Verify activation (should show venv path)
which python
```

### Step 3: Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PettingZoo and core MARL dependencies
pip install pettingzoo[classic] gymnasium numpy torch pandas

# Install production dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt

# Verify installation including PettingZoo
python -c "
import torch, pettingzoo, gymnasium, pandas, numpy
print('✅ All core dependencies installed')
print(f'PettingZoo version: {pettingzoo.__version__}')
print(f'Gymnasium version: {gymnasium.__version__}')
print(f'PyTorch version: {torch.__version__}')
"
```

### Step 4: Set Environment Variables

```bash
# Add to your shell profile (.bashrc, .zshrc, etc.)
export PYTHONPATH="${PYTHONPATH}:/path/to/GrandModel"
export GRANDMODEL_ENV=development
export LOG_LEVEL=INFO

# For current session
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export GRANDMODEL_ENV=development
export LOG_LEVEL=INFO
```

### Step 5: Start Supporting Services

```bash
# Start Redis and Ollama services
docker-compose up -d redis ollama

# Verify services are running
docker-compose ps

# Expected output:
#     Name                   Command               State           Ports
# ----------------------------------------------------------------------------
# grandmodel_redis_1     docker-entrypoint.sh redis ...   Up      6379/tcp
# grandmodel_ollama_1    /bin/ollama serve               Up      11434/tcp

# Test Redis connection
docker-compose exec redis redis-cli ping
# Expected: PONG
```

### Step 6: Initialize Ollama Model

```bash
# Pull the Phi model for LLM integration
docker-compose exec ollama ollama pull phi

# Verify model is available
docker-compose exec ollama ollama list
# Expected: phi:latest
```

### Step 7: Verify Installation

```bash
# Run installation verification
python -c "
import sys
print(f'Python version: {sys.version}')

import torch
print(f'PyTorch version: {torch.__version__}')

import pettingzoo
print(f'PettingZoo version: {pettingzoo.__version__}')

import gymnasium
print(f'Gymnasium version: {gymnasium.__version__}')

import pandas as pd
print(f'Pandas version: {pd.__version__}')

print('✅ Installation verification complete')
"

# Verify PettingZoo environments
python -c "
from src.environment.strategic_env import StrategicMarketEnv
from src.environment.tactical_env import TacticalMarketEnv

# Test strategic environment
strategic_config = {
    'strategic_marl': {
        'environment': {
            'matrix_shape': [48, 13],
            'max_episode_steps': 10
        }
    }
}
strategic_env = StrategicMarketEnv(strategic_config)
strategic_env.reset()
print('✅ Strategic environment initialized')

# Test tactical environment
tactical_config = {
    'tactical_marl': {
        'environment': {
            'matrix_shape': [60, 7],
            'max_episode_steps': 10
        }
    }
}
tactical_env = TacticalMarketEnv(tactical_config)
tactical_env.reset()
print('✅ Tactical environment initialized')

print('✅ PettingZoo environments verified')
"

# Run PettingZoo API tests
python -c "
from pettingzoo.test import api_test
from src.environment.strategic_env import StrategicMarketEnv

config = {
    'strategic_marl': {
        'environment': {
            'matrix_shape': [48, 13],
            'max_episode_steps': 5
        }
    }
}

env = StrategicMarketEnv(config)
api_test(env, num_cycles=3)
print('✅ PettingZoo API test passed')
"

# Run system tests
pytest tests/test_production_ready.py::TestProductionReady::test_pytorch_installation -v
```

## Configuration

### Step 1: Create Configuration Directory

```bash
# Create configuration directories
mkdir -p configs/{system,models,data,environments}

# Copy example configuration
cp production_config.yaml configs/system/development.yaml

# Create PettingZoo environment configurations
mkdir -p configs/environments/pettingzoo
```

### Step 2: Edit Development Configuration

Edit `configs/system/development.yaml`:

```yaml
# Development Configuration
system:
  environment: development
  log_level: DEBUG
  
data_handler:
  type: backtest  # Use backtest for development
  connection:
    host: localhost
    port: 3001
  symbols:
    - ES  # S&P 500 E-mini futures
    - NQ  # NASDAQ E-mini futures
  
matrix_assemblers:
  30m:
    window_size: 48
    features:
      - mlmi_value
      - mlmi_signal
      - nwrqk_value
      - nwrqk_slope
      - time_hour_sin
      - time_hour_cos
  5m:
    window_size: 60
    features:
      - fvg_bullish_active
      - fvg_bearish_active
      - price_momentum_5
      - volume_ratio

# PettingZoo Configuration
pettingzoo:
  environments:
    strategic:
      enabled: true
      config_path: configs/environments/pettingzoo/strategic_marl.yaml
      
    tactical:
      enabled: true
      config_path: configs/environments/pettingzoo/tactical_marl.yaml
      
    risk:
      enabled: false
      config_path: configs/environments/pettingzoo/risk_management.yaml
      
    execution:
      enabled: false
      config_path: configs/environments/pettingzoo/execution.yaml

strategic_marl:
  enabled: true
  model_path: models/strategic_agent.pth
  learning_rate: 0.0001
  batch_size: 32
  
risk_management:
  max_position_size: 0.02  # 2% of capital per trade
  max_daily_loss: 0.05     # 5% daily loss limit
  kelly_fraction: 0.25     # Kelly Criterion multiplier
  
logging:
  level: DEBUG
  file: logs/grandmodel_dev.log
  max_size: 10MB
  backup_count: 5
```

### Step 3: Create PettingZoo Environment Configurations

Create `configs/environments/pettingzoo/strategic_marl.yaml`:

```yaml
# Strategic MARL Environment Configuration
strategic_marl:
  environment:
    matrix_shape: [48, 13]
    max_episode_steps: 2000
    reward_scaling: 1.0
    observation_noise: 0.01
    synergy_weight: 0.2
    regime_detection: true
    
  agents:
    mlmi_expert:
      observation_columns: [0, 1, 2, 3]  # MLMI features
      action_space_size: 3
      expertise_weight: 0.35
      
    nwrqk_expert:
      observation_columns: [4, 5, 6, 7]  # NWRQK features
      action_space_size: 3
      expertise_weight: 0.35
      
    regime_expert:
      observation_columns: [8, 9, 10, 11, 12]  # Regime features
      action_space_size: 3
      expertise_weight: 0.30
      
  reward_shaping:
    profit_weight: 1.0
    synergy_bonus: 0.2
    risk_penalty: 0.1
    sharpe_bonus: 0.1
```

Create `configs/environments/pettingzoo/tactical_marl.yaml`:

```yaml
# Tactical MARL Environment Configuration
tactical_marl:
  environment:
    matrix_shape: [60, 7]
    max_episode_steps: 1000
    reward_scaling: 1.0
    state_machine: true
    byzantine_tolerance: true
    
  agents:
    fvg_agent:
      observation_columns: [0, 1, 2]
      detection_threshold: 0.7
      confidence_threshold: 0.8
      
    momentum_agent:
      observation_columns: [3, 4]
      momentum_window: 14
      trend_threshold: 0.6
      
    entry_opt_agent:
      observation_columns: [5, 6]
      optimization_steps: 10
      timing_precision: 0.001  # 1ms
      
  reward_shaping:
    execution_quality_weight: 1.0
    timing_bonus: 0.15
    coordination_bonus: 0.1
    slippage_penalty: 0.2
```

### Step 4: Create Logging Directory

```bash
# Create logs directory
mkdir -p logs

# Set permissions
chmod 755 logs
```

### Step 5: Validate Configuration

```bash
# Validate main configuration syntax
python -c "
import yaml
with open('configs/system/development.yaml', 'r') as f:
    config = yaml.safe_load(f)
print('✅ Main configuration file is valid YAML')
print(f'Environment: {config[\"system\"][\"environment\"]}')
print(f'Data handler: {config[\"data_handler\"][\"type\"]}')
print(f'PettingZoo environments: {list(config[\"pettingzoo\"][\"environments\"].keys())}')
"

# Validate PettingZoo environment configurations
python -c "
import yaml
from pathlib import Path

# Validate strategic environment config
strategic_path = Path('configs/environments/pettingzoo/strategic_marl.yaml')
if strategic_path.exists():
    with open(strategic_path, 'r') as f:
        strategic_config = yaml.safe_load(f)
    print('✅ Strategic MARL configuration is valid')
    print(f'  Matrix shape: {strategic_config[\"strategic_marl\"][\"environment\"][\"matrix_shape\"]}')
    print(f'  Agents: {list(strategic_config[\"strategic_marl\"][\"agents\"].keys())}')
else:
    print('⚠️  Strategic MARL configuration not found')

# Validate tactical environment config
tactical_path = Path('configs/environments/pettingzoo/tactical_marl.yaml')
if tactical_path.exists():
    with open(tactical_path, 'r') as f:
        tactical_config = yaml.safe_load(f)
    print('✅ Tactical MARL configuration is valid')
    print(f'  Matrix shape: {tactical_config[\"tactical_marl\"][\"environment\"][\"matrix_shape\"]}')
    print(f'  Agents: {list(tactical_config[\"tactical_marl\"][\"agents\"].keys())}')
else:
    print('⚠️  Tactical MARL configuration not found')
"
```

## First Run

### Step 1: Run System Check

```bash
# Run comprehensive system check
python -c "
from src.core.kernel import AlgoSpaceKernel
from src.core.minimal_config import load_config

try:
    # Test configuration loading
    config = load_config('configs/system/development.yaml')
    print('✅ Configuration loaded successfully')
    
    # Test kernel initialization
    kernel = AlgoSpaceKernel('configs/system/development.yaml')
    print('✅ Kernel initialized successfully')
    
    print('🎉 System check passed!')
    
except Exception as e:
    print(f'❌ System check failed: {e}')
    raise
"
```

### Step 2: Start the System

```bash
# Start GrandModel system
python src/main.py

# Expected initial output:
# 2025-07-13 10:00:00,000 - src.main - INFO - 🚀 GrandModel starting...
# 2025-07-13 10:00:00,001 - src.core.kernel - INFO - AlgoSpace Kernel initialized with config path: configs/system/development.yaml
# 2025-07-13 10:00:00,002 - src.core.kernel - INFO - === AlgoSpace System Initialization Starting ===
# ...
```

### Step 3: Monitor System Status

In a new terminal window:

```bash
# Monitor logs in real-time
tail -f logs/grandmodel_dev.log

# Check system health (if HTTP endpoint is available)
curl http://localhost:8000/health

# Monitor Docker services
docker-compose logs -f
```

### Step 4: Graceful Shutdown

```bash
# In the main terminal, press Ctrl+C to shutdown
# Expected output:
# 2025-07-13 10:05:00,000 - src.main - INFO - 🛑 Shutdown requested by user
# 2025-07-13 10:05:00,001 - src.core.kernel - INFO - === Graceful Shutdown Initiated ===
# ...
# 2025-07-13 10:05:00,010 - src.main - INFO - 🏁 GrandModel shutdown complete
```

## Basic Operations

### Starting and Stopping

```bash
# Start system
python src/main.py

# Start with custom config
python src/main.py --config configs/system/production.yaml

# Start in debug mode
LOG_LEVEL=DEBUG python src/main.py

# Stop system (graceful)
# Press Ctrl+C in the terminal running the system
```

### Monitoring

```bash
# View real-time logs
tail -f logs/grandmodel_dev.log

# View system status
python -c "
from src.core.kernel import AlgoSpaceKernel
kernel = AlgoSpaceKernel('configs/system/development.yaml')
kernel.initialize()
status = kernel.get_status()
print(f'System Status: {status}')
"

# Monitor resource usage
htop  # or top on macOS
docker stats
```

### Configuration Changes

```bash
# Validate new configuration
python -c "
import yaml
with open('configs/system/new_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print('Configuration is valid')
"

# Test configuration with dry run
python src/main.py --config configs/system/new_config.yaml --dry-run
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_production_ready.py -v  # Production readiness
pytest tests/risk/ -v                     # Risk management
pytest tests/integration/ -v              # Integration tests

# Run with coverage
pytest --cov=src --cov-report=html
open htmlcov/index.html  # View coverage report
```

### Accessing Components

```python
# Interactive Python session
python -i -c "
from src.core.kernel import AlgoSpaceKernel
kernel = AlgoSpaceKernel('configs/system/development.yaml')
kernel.initialize()

# Access specific components
data_handler = kernel.get_component('data_handler')
strategic_marl = kernel.get_component('strategic_marl')
event_bus = kernel.get_event_bus()

print('Components available for inspection')
"

# In the Python session:
# >>> status = kernel.get_status()
# >>> print(status)
# >>> exit()
```

## Troubleshooting

### Common Issues

#### Issue 1: Import Errors

**Symptoms:**
```
ImportError: No module named 'src'
```

**Solution:**
```bash
# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or add to your shell profile
echo 'export PYTHONPATH="${PYTHONPATH}:/path/to/GrandModel"' >> ~/.bashrc
source ~/.bashrc
```

#### Issue 2: Docker Services Not Starting

**Symptoms:**
```
docker-compose up -d redis
ERROR: Could not find docker-compose.yml
```

**Solution:**
```bash
# Verify you're in the project root
pwd
ls docker-compose.yml

# If missing, check if using different filename
ls docker-compose*.yml

# Start with specific file
docker-compose -f docker-compose.dev.yml up -d redis
```

#### Issue 3: Configuration File Not Found

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'config/settings.yaml'
```

**Solution:**
```bash
# Use absolute path
python src/main.py --config $(pwd)/configs/system/development.yaml

# Or create default config directory
mkdir -p config
cp configs/system/development.yaml config/settings.yaml
```

#### Issue 4: Permission Errors

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied: 'logs/grandmodel.log'
```

**Solution:**
```bash
# Create logs directory with proper permissions
mkdir -p logs
chmod 755 logs

# If running with Docker, fix ownership
sudo chown -R $USER:$USER logs/
```

#### Issue 5: PyTorch Installation Issues

**Symptoms:**
```
ImportError: No module named 'torch'
```

**Solution:**
```bash
# Reinstall PyTorch
pip uninstall torch
pip install torch==2.7.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# For CUDA (if available)
pip install torch==2.7.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

### Debug Mode

```bash
# Enable maximum debugging
export LOG_LEVEL=DEBUG
export GRANDMODEL_ENV=development

# Run with Python debugging
python -u src/main.py 2>&1 | tee debug.log

# Use pdb for interactive debugging
python -m pdb src/main.py
```

### Getting Help

```bash
# Check system requirements
python scripts/check_requirements.py

# Run diagnostic script
python scripts/diagnose_system.py

# Check all dependencies
pip list | grep -E "torch|pettingzoo|pandas|numpy"
```

### Log Analysis

```bash
# Find errors in logs
grep ERROR logs/grandmodel_dev.log

# Find warnings
grep WARN logs/grandmodel_dev.log

# Show startup sequence
grep "Initialization\|Starting\|Ready" logs/grandmodel_dev.log

# Monitor real-time with filtering
tail -f logs/grandmodel_dev.log | grep -E "ERROR|WARN|CRITICAL"
```

## Next Steps

### 1. Explore the System

```bash
# Read core documentation
ls docs/

# Examine source code structure
tree src/ -d

# Look at example configurations
ls configs/
```

### 2. Run Example Strategies

```bash
# Try different configuration files
python src/main.py --config configs/system/backtest.yaml
python src/main.py --config configs/system/paper_trading.yaml
```

### 3. Customize Configuration

```bash
# Create your own configuration
cp configs/system/development.yaml configs/system/my_config.yaml
# Edit my_config.yaml with your preferences
```

### 4. Learn the API

- Read [Kernel API Documentation](../api/kernel_api.md)
- Read [Event System Documentation](../api/events_api.md)
- Read [MARL Agents Documentation](../api/agents_api.md)

### 5. Development Setup

```bash
# Install development tools
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Configure your IDE for the project
```

### 6. Training Models

- Follow the [Training Guide](training_guide.md)
- Read about [Model Development](../development/model_development.md)
- Explore [Performance Optimization](performance_guide.md)

### 7. Production Deployment

- Review [Deployment Guide](deployment_guide.md)
- Understand [Security Considerations](../architecture/security.md)
- Set up [Monitoring and Alerting](../deployment/monitoring.md)

## Quick Reference

### Essential Commands

```bash
# Start system
python src/main.py

# Start with custom config
python src/main.py --config path/to/config.yaml

# Run tests
pytest tests/ -v

# Check system health
python -c "from src.core.kernel import AlgoSpaceKernel; k=AlgoSpaceKernel(); print(k.get_status())"

# View logs
tail -f logs/grandmodel_dev.log

# Start Docker services
docker-compose up -d redis ollama

# Stop Docker services
docker-compose down
```

### Important File Locations

- **Main entry point**: `src/main.py`
- **Configuration**: `configs/system/`
- **Logs**: `logs/`
- **Tests**: `tests/`
- **Documentation**: `docs/`
- **Docker configs**: `docker/`

### Key Configuration Sections

- `data_handler`: Market data configuration
- `strategic_marl`: MARL agent settings
- `risk_management`: Risk limits and controls
- `logging`: Log configuration
- `matrix_assemblers`: Feature engineering settings

Congratulations! You now have GrandModel up and running. The system is ready for further exploration, customization, and development.

## Related Documentation

- [Training Guide](training_guide.md)
- [Configuration Reference](configuration_guide.md)
- [Deployment Guide](deployment_guide.md)
- [API Documentation](../api/)
- [Architecture Overview](../architecture/system_overview.md)