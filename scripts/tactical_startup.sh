#!/bin/bash

# Tactical MARL System Startup Script
# Handles TorchScript JIT compilation and service startup

set -e

echo "🚀 Starting Tactical 5-Minute MARL System"
echo "=========================================="

# Check if models directory exists
if [ ! -d "/app/models/tactical" ]; then
    echo "❌ Models directory not found. Creating..."
    mkdir -p /app/models/tactical
fi

# Set CPU affinity for optimal performance
# Bind to cores 0-15 for consistent performance
if [ -n "$TACTICAL_CPU_AFFINITY" ]; then
    echo "🔧 Setting CPU affinity: $TACTICAL_CPU_AFFINITY"
    taskset -c $TACTICAL_CPU_AFFINITY echo "CPU affinity set"
fi

# JIT compile models if they exist
echo "⚡ Starting TorchScript JIT compilation..."
if [ -f "/app/scripts/jit_compile_models.py" ]; then
    python /app/scripts/jit_compile_models.py
    if [ $? -eq 0 ]; then
        echo "✅ JIT compilation completed successfully"
    else
        echo "⚠️ JIT compilation failed, continuing with standard models"
    fi
else
    echo "⚠️ JIT compilation script not found, skipping optimization"
fi

# Warm up Python imports
echo "🔄 Warming up Python imports..."
python -c "
import torch
import numpy as np
import asyncio
import redis
import fastapi
from datetime import datetime
print('✅ Core imports warmed up')
"

# Pre-allocate tensor memory for consistent performance
echo "🧠 Pre-allocating tensor memory..."
python -c "
import torch
# Pre-allocate common tensor shapes for tactical system
_ = torch.zeros(1, 60, 7, dtype=torch.float32)  # Matrix input
_ = torch.zeros(1, 3, dtype=torch.float32)      # Agent output
_ = torch.zeros(32, 60, 7, dtype=torch.float32) # Batch processing
print('✅ Memory pre-allocation complete')
"

# Verify Redis connectivity
echo "🔍 Verifying Redis connectivity..."
python -c "
import redis
import os
redis_url = os.getenv('REDIS_URL', 'redis://redis:6379/2')
try:
    r = redis.from_url(redis_url)
    r.ping()
    print('✅ Redis connection verified')
except Exception as e:
    print(f'❌ Redis connection failed: {e}')
    exit(1)
"

# Start the tactical service
echo "🎯 Starting Tactical MARL Service..."
echo "Listening on port 8001 for API requests"
echo "Metrics available on port 9091"
echo "=========================================="

# Execute the main tactical service
exec python -m src.api.tactical_main