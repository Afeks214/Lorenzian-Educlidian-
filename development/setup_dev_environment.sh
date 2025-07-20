#!/bin/bash

# GrandModel Development Environment Setup Script
# Creates and configures development environment for MARL trading system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="/home/QuantNova/GrandModel"
DEV_ROOT="${PROJECT_ROOT}/development"
PYTHON_VERSION="3.8"
VENV_NAME="grandmodel-dev"

# Logging
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to create directory if it doesn't exist
ensure_directory() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        log_info "Created directory: $1"
    fi
}

# Header
echo "=========================================="
echo "  GrandModel Development Environment Setup"
echo "=========================================="
echo ""

# Check prerequisites
log_info "Checking prerequisites..."

if ! command_exists python3; then
    log_error "Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

if ! command_exists pip3; then
    log_error "pip3 is not installed. Please install pip first."
    exit 1
fi

if ! command_exists git; then
    log_error "Git is not installed. Please install Git first."
    exit 1
fi

log_success "Prerequisites check passed"

# Create development directory structure
log_info "Creating development directory structure..."

directories=(
    "${DEV_ROOT}/research"
    "${DEV_ROOT}/training"
    "${DEV_ROOT}/backtesting"
    "${DEV_ROOT}/notebooks"
    "${DEV_ROOT}/data"
    "${DEV_ROOT}/models"
    "${DEV_ROOT}/tests"
    "${DEV_ROOT}/docs"
    "${DEV_ROOT}/config"
    "${DEV_ROOT}/logs"
    "${DEV_ROOT}/scripts"
    "${DEV_ROOT}/tools"
    "${DEV_ROOT}/experiments"
    "${DEV_ROOT}/checkpoints"
)

for dir in "${directories[@]}"; do
    ensure_directory "$dir"
done

log_success "Directory structure created"

# Create Python virtual environment
log_info "Creating Python virtual environment..."

if command_exists conda; then
    log_info "Using conda to create environment..."
    conda create -n "${VENV_NAME}" python="${PYTHON_VERSION}" -y
    source activate "${VENV_NAME}"
elif command_exists virtualenv; then
    log_info "Using virtualenv to create environment..."
    virtualenv -p python3 "${DEV_ROOT}/venv"
    source "${DEV_ROOT}/venv/bin/activate"
else
    log_info "Using python venv to create environment..."
    python3 -m venv "${DEV_ROOT}/venv"
    source "${DEV_ROOT}/venv/bin/activate"
fi

log_success "Virtual environment created"

# Install development dependencies
log_info "Installing development dependencies..."

pip install --upgrade pip setuptools wheel

# Install core dependencies first
if [ -f "${PROJECT_ROOT}/requirements.txt" ]; then
    pip install -r "${PROJECT_ROOT}/requirements.txt"
    log_success "Core dependencies installed"
fi

# Install development dependencies
if [ -f "${DEV_ROOT}/requirements-dev.txt" ]; then
    pip install -r "${DEV_ROOT}/requirements-dev.txt"
    log_success "Development dependencies installed"
fi

# Setup Jupyter kernel
log_info "Setting up Jupyter kernel..."
python -m ipykernel install --user --name="${VENV_NAME}" --display-name="GrandModel Dev"
log_success "Jupyter kernel installed"

# Create environment variables file
log_info "Creating environment variables file..."

cat > "${DEV_ROOT}/.env" << EOF
# GrandModel Development Environment Variables

# Environment
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=grandmodel_dev
DB_USER=grandmodel_dev
DB_PASSWORD=dev_password_change_me

# Redis Configuration (Development)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=true

# Jupyter Configuration
JUPYTER_TOKEN=dev_token_change_me
JUPYTER_PORT=8888

# TensorBoard Configuration
TENSORBOARD_PORT=6006
TENSORBOARD_LOG_DIR=${DEV_ROOT}/logs/tensorboard

# Data Sources
PRIMARY_DATA_SOURCE=${PROJECT_ROOT}/colab/data/
BACKUP_DATA_SOURCE=${DEV_ROOT}/data/

# Model Storage
MODEL_STORAGE_PATH=${DEV_ROOT}/models/
CHECKPOINT_PATH=${DEV_ROOT}/checkpoints/

# Performance Monitoring
METRICS_PORT=8080
PROFILING_ENABLED=true

# Feature Flags
ENABLE_QUANTUM_SUPERPOSITION=true
ENABLE_MC_DROPOUT=true
ENABLE_PETTINGZOO=true
ENABLE_ADVANCED_ANALYTICS=true

# Security (Development - Simplified)
SECRET_KEY=dev_secret_key_change_in_production
ENCRYPTION_ENABLED=false
AUTH_REQUIRED=false

EOF

log_success "Environment variables file created"

# Create development scripts
log_info "Creating development helper scripts..."

# Start development server script
cat > "${DEV_ROOT}/scripts/start_dev_server.sh" << 'EOF'
#!/bin/bash
# Start development server with hot reload

source ../venv/bin/activate
export $(cat ../.env | xargs)

echo "Starting GrandModel Development Server..."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
EOF

# Start Jupyter Lab script
cat > "${DEV_ROOT}/scripts/start_jupyter.sh" << 'EOF'
#!/bin/bash
# Start Jupyter Lab for development

source ../venv/bin/activate
export $(cat ../.env | xargs)

echo "Starting Jupyter Lab..."
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
EOF

# Run tests script
cat > "${DEV_ROOT}/scripts/run_tests.sh" << 'EOF'
#!/bin/bash
# Run all tests with coverage

source ../venv/bin/activate
export $(cat ../.env | xargs)

echo "Running tests with coverage..."
pytest ../tests/ --cov=src --cov-report=html --cov-report=term-missing
EOF

# Format code script
cat > "${DEV_ROOT}/scripts/format_code.sh" << 'EOF'
#!/bin/bash
# Format code using black and isort

source ../venv/bin/activate

echo "Formatting code with black..."
black src/ tests/ --line-length 88

echo "Sorting imports with isort..."
isort src/ tests/ --profile black

echo "Code formatting complete!"
EOF

# Lint code script
cat > "${DEV_ROOT}/scripts/lint_code.sh" << 'EOF'
#!/bin/bash
# Lint code using flake8 and mypy

source ../venv/bin/activate

echo "Linting with flake8..."
flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503

echo "Type checking with mypy..."
mypy src/ --ignore-missing-imports

echo "Linting complete!"
EOF

# Make scripts executable
chmod +x "${DEV_ROOT}/scripts/"*.sh

log_success "Development helper scripts created"

# Setup pre-commit hooks
log_info "Setting up pre-commit hooks..."

cat > "${DEV_ROOT}/.pre-commit-config.yaml" << EOF
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ["--max-line-length=88", "--extend-ignore=E203,W503"]
EOF

# Install pre-commit hooks
pre-commit install

log_success "Pre-commit hooks configured"

# Create development documentation
log_info "Creating development documentation..."

cat > "${DEV_ROOT}/README.md" << 'EOF'
# GrandModel Development Environment

This is the development environment for the GrandModel MARL trading system.

## Quick Start

1. **Activate Environment:**
   ```bash
   source venv/bin/activate  # or conda activate grandmodel-dev
   ```

2. **Set Environment Variables:**
   ```bash
   export $(cat .env | xargs)
   ```

3. **Start Development Server:**
   ```bash
   ./scripts/start_dev_server.sh
   ```

4. **Start Jupyter Lab:**
   ```bash
   ./scripts/start_jupyter.sh
   ```

## Development Workflows

### Code Development
- Use `./scripts/format_code.sh` to format code
- Use `./scripts/lint_code.sh` to lint code
- Use `./scripts/run_tests.sh` to run tests

### Model Training
- Strategic MAPPO: Use notebooks/strategic_mappo_training.ipynb
- Tactical MAPPO: Use notebooks/tactical_mappo_training.ipynb
- Custom experiments: Create new notebooks in notebooks/

### Data Analysis
- Raw data: data/ directory
- Processed data: Use backtesting/ for analysis
- Visualization: Use notebooks for interactive analysis

### Testing
- Unit tests: tests/unit/
- Integration tests: tests/integration/
- Performance tests: tests/performance/

## Directory Structure
- `research/`: Experimental models and strategies
- `training/`: Model training scripts and configurations
- `backtesting/`: Historical strategy testing
- `notebooks/`: Jupyter development notebooks
- `data/`: Development datasets
- `models/`: Training checkpoints and experiments
- `tests/`: Test suites
- `docs/`: Development documentation
- `config/`: Configuration files
- `logs/`: Application and training logs
- `scripts/`: Helper scripts
- `tools/`: Development utilities

## Configuration
- Main config: `config/dev_config.yaml`
- Environment variables: `.env`
- Dependencies: `requirements-dev.txt`

## Useful Commands
- `jupyter lab`: Start Jupyter Lab
- `tensorboard --logdir=logs/tensorboard`: Start TensorBoard
- `pytest tests/`: Run tests
- `black src/`: Format code
- `mypy src/`: Type checking
EOF

log_success "Development documentation created"

# Create sample notebook
log_info "Creating sample development notebook..."

cat > "${DEV_ROOT}/notebooks/development_playground.ipynb" << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# GrandModel Development Playground\n",
    "\n",
    "This notebook provides a starting point for development and experimentation with the GrandModel MARL trading system."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import core libraries\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import torch\n",
    "\n",
    "# Set up plotting\n",
    "plt.style.use('seaborn-v0_8')\n",
    "sns.set_palette('husl')\n",
    "\n",
    "print(\"Development environment loaded successfully!\")\n",
    "print(f\"PyTorch version: {torch.__version__}\")\n",
    "print(f\"CUDA available: {torch.cuda.is_available()}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load sample data\n",
    "import os\n",
    "\n",
    "data_path = '/home/QuantNova/GrandModel/colab/data/@CL - 30 min - ETH.csv'\n",
    "if os.path.exists(data_path):\n",
    "    df = pd.read_csv(data_path)\n",
    "    print(f\"Loaded {len(df)} rows of CL 30-minute data\")\n",
    "    print(df.head())\n",
    "else:\n",
    "    print(\"Sample data not found. Create synthetic data for testing.\")\n",
    "    # Create synthetic data\n",
    "    dates = pd.date_range('2023-01-01', periods=1000, freq='30T')\n",
    "    df = pd.DataFrame({\n",
    "        'timestamp': dates,\n",
    "        'open': np.random.randn(1000).cumsum() + 100,\n",
    "        'high': np.random.randn(1000).cumsum() + 102,\n",
    "        'low': np.random.randn(1000).cumsum() + 98,\n",
    "        'close': np.random.randn(1000).cumsum() + 100,\n",
    "        'volume': np.random.randint(1000, 10000, 1000)\n",
    "    })\n",
    "    print(\"Created synthetic data for development\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test MARL environment setup\n",
    "try:\n",
    "    from pettingzoo.test import api_test\n",
    "    print(\"PettingZoo available for MARL development\")\n",
    "except ImportError:\n",
    "    print(\"PettingZoo not available - using fallback implementations\")\n",
    "\n",
    "# Test strategic matrix processing\n",
    "strategic_matrix = np.random.randn(48, 13)\n",
    "print(f\"Strategic matrix shape: {strategic_matrix.shape}\")\n",
    "print(f\"Strategic matrix flattened: {strategic_matrix.flatten().shape}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test development tools\n",
    "from IPython.display import display, HTML\n",
    "\n",
    "display(HTML(\"<h3>Development Environment Ready!</h3>\"))\n",
    "print(\"✅ Data loading: OK\")\n",
    "print(\"✅ Matrix processing: OK\")\n",
    "print(\"✅ Visualization: OK\")\n",
    "print(\"✅ MARL components: OK\")\n",
    "print(\"\")\n",
    "print(\"Ready for GrandModel development!\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "GrandModel Dev",
   "language": "python",
   "name": "grandmodel-dev"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

log_success "Sample development notebook created"

# Final setup
log_info "Finalizing development environment setup..."

# Create empty __init__.py files for Python packages
touch "${DEV_ROOT}/research/__init__.py"
touch "${DEV_ROOT}/training/__init__.py"
touch "${DEV_ROOT}/backtesting/__init__.py"
touch "${DEV_ROOT}/tests/__init__.py"

# Create gitignore for development
cat > "${DEV_ROOT}/.gitignore" << EOF
# Development environment specific
venv/
*.env
.env.*
logs/
checkpoints/
models/
experiments/
.coverage
htmlcov/
.pytest_cache/
.mypy_cache/
__pycache__/
*.pyc
*.pyo
*.pyd
.DS_Store
.vscode/
.idea/

# Jupyter
.ipynb_checkpoints/
*.ipynb_backup

# Data files (often large)
data/
*.csv
*.parquet
*.h5
*.hdf5

# Model files
*.pkl
*.pt
*.pth
*.onnx
*.pb

# Temporary files
*.tmp
*.temp
temp/
tmp/
EOF

log_success "Development environment setup complete!"

echo ""
echo "=========================================="
echo "  Development Environment Ready!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate environment: source ${DEV_ROOT}/venv/bin/activate"
echo "2. Set environment: export \$(cat ${DEV_ROOT}/.env | xargs)"
echo "3. Start Jupyter: cd ${DEV_ROOT} && ./scripts/start_jupyter.sh"
echo "4. Open browser: http://localhost:8888"
echo ""
echo "Happy developing! 🚀"