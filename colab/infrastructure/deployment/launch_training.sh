#!/bin/bash

# Training Infrastructure Launcher Script
# Provides easy deployment of training infrastructure with all optimizations

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
PYTHON_CMD="python3"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Function to check dependencies
check_dependencies() {
    log "Checking dependencies..."
    
    # Check Python
    if ! command -v ${PYTHON_CMD} &> /dev/null; then
        error "Python3 not found. Please install Python 3.8 or higher."
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$(${PYTHON_CMD} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ "$(echo "${PYTHON_VERSION} < 3.8" | bc -l)" -eq 1 ]]; then
        error "Python version ${PYTHON_VERSION} is too old. Please install Python 3.8 or higher."
        exit 1
    fi
    
    # Check PyTorch
    if ! ${PYTHON_CMD} -c "import torch" &> /dev/null; then
        error "PyTorch not found. Please install PyTorch."
        exit 1
    fi
    
    # Check CUDA if available
    if ${PYTHON_CMD} -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        log "CUDA is available"
        ${PYTHON_CMD} -c "import torch; print(f'CUDA Version: {torch.version.cuda}')"
        ${PYTHON_CMD} -c "import torch; print(f'GPU Count: {torch.cuda.device_count()}')"
    else
        warn "CUDA is not available. Training will run on CPU."
    fi
    
    log "Dependencies check completed"
}

# Function to setup environment
setup_environment() {
    log "Setting up environment..."
    
    # Create necessary directories
    mkdir -p "${PROJECT_ROOT}/colab/exports"
    mkdir -p "${PROJECT_ROOT}/colab/logs"
    mkdir -p "${PROJECT_ROOT}/colab/infrastructure/backup"
    
    # Set environment variables
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    
    log "Environment setup completed"
}

# Function to run system tests
run_system_tests() {
    log "Running system tests..."
    
    cd "${PROJECT_ROOT}"
    
    # Run basic system tests
    ${PYTHON_CMD} -c "
import sys
import torch
import numpy as np
print(f'Python version: {sys.version}')
print(f'PyTorch version: {torch.__version__}')
print(f'NumPy version: {np.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"
    
    log "System tests completed"
}

# Function to launch training
launch_training() {
    log "Launching training deployment..."
    
    cd "${PROJECT_ROOT}"
    
    # Default parameters
    MODEL_NAME="${MODEL_NAME:-tactical_mappo}"
    BATCH_SIZE="${BATCH_SIZE:-32}"
    LEARNING_RATE="${LEARNING_RATE:-0.001}"
    NUM_EPOCHS="${NUM_EPOCHS:-10}"
    LOG_LEVEL="${LOG_LEVEL:-INFO}"
    OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/colab/exports}"
    
    # Build command
    CMD="${PYTHON_CMD} ${SCRIPT_DIR}/deploy_training.py"
    CMD="${CMD} --model-name ${MODEL_NAME}"
    CMD="${CMD} --batch-size ${BATCH_SIZE}"
    CMD="${CMD} --learning-rate ${LEARNING_RATE}"
    CMD="${CMD} --num-epochs ${NUM_EPOCHS}"
    CMD="${CMD} --log-level ${LOG_LEVEL}"
    CMD="${CMD} --output-dir ${OUTPUT_DIR}"
    
    # Add optional flags
    if [[ "${DISABLE_GPU}" == "true" ]]; then
        CMD="${CMD} --disable-gpu"
    fi
    
    if [[ "${DISABLE_MONITORING}" == "true" ]]; then
        CMD="${CMD} --disable-monitoring"
    fi
    
    if [[ "${DISABLE_BACKUPS}" == "true" ]]; then
        CMD="${CMD} --disable-backups"
    fi
    
    if [[ "${DISABLE_TESTING}" == "true" ]]; then
        CMD="${CMD} --disable-testing"
    fi
    
    log "Executing: ${CMD}"
    
    # Execute training
    if eval "${CMD}"; then
        log "Training completed successfully!"
    else
        error "Training failed!"
        exit 1
    fi
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Training Infrastructure Launcher

OPTIONS:
    -h, --help              Show this help message
    -m, --model-name        Model name (default: tactical_mappo)
    -b, --batch-size        Batch size (default: 32)
    -l, --learning-rate     Learning rate (default: 0.001)
    -e, --epochs            Number of epochs (default: 10)
    -o, --output-dir        Output directory (default: PROJECT_ROOT/colab/exports)
    --log-level             Log level (default: INFO)
    --disable-gpu           Disable GPU optimization
    --disable-monitoring    Disable performance monitoring
    --disable-backups       Disable backup system
    --disable-testing       Disable testing pipeline
    --skip-tests            Skip system tests
    --dry-run              Show commands without executing

ENVIRONMENT VARIABLES:
    MODEL_NAME             Model name
    BATCH_SIZE             Batch size
    LEARNING_RATE          Learning rate
    NUM_EPOCHS             Number of epochs
    LOG_LEVEL              Log level
    OUTPUT_DIR             Output directory
    DISABLE_GPU            Disable GPU optimization
    DISABLE_MONITORING     Disable monitoring
    DISABLE_BACKUPS        Disable backups
    DISABLE_TESTING        Disable testing
    CUDA_VISIBLE_DEVICES   CUDA devices to use

EXAMPLES:
    # Basic training with defaults
    $0
    
    # Training with custom parameters
    $0 --model-name my_model --batch-size 64 --epochs 50
    
    # Training without GPU optimization
    $0 --disable-gpu
    
    # Training with environment variables
    MODEL_NAME=custom_model BATCH_SIZE=128 $0
    
    # Dry run to see commands
    $0 --dry-run

EOF
}

# Main function
main() {
    local skip_tests=false
    local dry_run=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -m|--model-name)
                MODEL_NAME="$2"
                shift 2
                ;;
            -b|--batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            -l|--learning-rate)
                LEARNING_RATE="$2"
                shift 2
                ;;
            -e|--epochs)
                NUM_EPOCHS="$2"
                shift 2
                ;;
            -o|--output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --log-level)
                LOG_LEVEL="$2"
                shift 2
                ;;
            --disable-gpu)
                DISABLE_GPU=true
                shift
                ;;
            --disable-monitoring)
                DISABLE_MONITORING=true
                shift
                ;;
            --disable-backups)
                DISABLE_BACKUPS=true
                shift
                ;;
            --disable-testing)
                DISABLE_TESTING=true
                shift
                ;;
            --skip-tests)
                skip_tests=true
                shift
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            *)
                error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Print banner
    echo -e "${BLUE}"
    echo "======================================"
    echo "   Training Infrastructure Launcher"
    echo "======================================"
    echo -e "${NC}"
    
    # Check dependencies
    check_dependencies
    
    # Setup environment
    setup_environment
    
    # Run system tests
    if [[ "${skip_tests}" != true ]]; then
        run_system_tests
    fi
    
    # Show configuration
    log "Configuration:"
    log "  Model Name: ${MODEL_NAME:-tactical_mappo}"
    log "  Batch Size: ${BATCH_SIZE:-32}"
    log "  Learning Rate: ${LEARNING_RATE:-0.001}"
    log "  Epochs: ${NUM_EPOCHS:-10}"
    log "  Log Level: ${LOG_LEVEL:-INFO}"
    log "  Output Dir: ${OUTPUT_DIR:-${PROJECT_ROOT}/colab/exports}"
    log "  Disable GPU: ${DISABLE_GPU:-false}"
    log "  Disable Monitoring: ${DISABLE_MONITORING:-false}"
    log "  Disable Backups: ${DISABLE_BACKUPS:-false}"
    log "  Disable Testing: ${DISABLE_TESTING:-false}"
    
    # Dry run or execute
    if [[ "${dry_run}" == true ]]; then
        log "Dry run mode - commands would be executed:"
        log "  ${PYTHON_CMD} ${SCRIPT_DIR}/deploy_training.py [options]"
    else
        # Launch training
        launch_training
    fi
    
    log "Script completed"
}

# Run main function
main "$@"