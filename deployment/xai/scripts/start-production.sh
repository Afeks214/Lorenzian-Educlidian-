#!/bin/bash
# Production startup script for XAI Trading System
# Agent Epsilon - Production Deployment Specialist

set -euo pipefail

# Configuration
export XAI_ENV="production"
export PYTHONPATH="/app/src"
export WORKERS=${XAI_WORKERS:-4}
export MAX_WORKERS=${XAI_MAX_WORKERS:-8}
export WORKER_CLASS=${XAI_WORKER_CLASS:-uvicorn.workers.UvicornWorker}
export HOST=${XAI_API_HOST:-0.0.0.0}
export PORT=${XAI_API_PORT:-8000}
export LOG_LEVEL=${XAI_LOG_LEVEL:-info}

# Logging setup
LOG_DIR="/app/logs"
mkdir -p "$LOG_DIR"

echo "🚀 Starting XAI Trading System in Production Mode"
echo "================================================"
echo "Environment: $XAI_ENV"
echo "Workers: $WORKERS"
echo "Host: $HOST:$PORT"
echo "Log Level: $LOG_LEVEL"
echo "Python Path: $PYTHONPATH"
echo "================================================"

# Pre-flight checks
echo "🔍 Running pre-flight checks..."

# Check required environment variables
required_vars=(
    "XAI_REDIS_URL"
    "XAI_POSTGRES_URL"
    "XAI_OLLAMA_URL"
)

for var in "${required_vars[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        echo "❌ ERROR: Required environment variable $var is not set"
        exit 1
    fi
    echo "✅ $var is set"
done

# Check database connectivity
echo "🔍 Testing database connections..."

# Test Redis
if timeout 10 python3 -c "
import redis
import os
r = redis.from_url(os.environ['XAI_REDIS_URL'])
r.ping()
print('✅ Redis connection successful')
"; then
    echo "Redis: OK"
else
    echo "❌ Redis connection failed"
    exit 1
fi

# Test PostgreSQL
if timeout 10 python3 -c "
import psycopg2
import os
conn = psycopg2.connect(os.environ['XAI_POSTGRES_URL'])
conn.close()
print('✅ PostgreSQL connection successful')
"; then
    echo "PostgreSQL: OK"
else
    echo "❌ PostgreSQL connection failed"
    exit 1
fi

# Test Ollama
if timeout 30 python3 -c "
import requests
import os
url = os.environ['XAI_OLLAMA_URL']
response = requests.get(f'{url}/api/tags', timeout=10)
response.raise_for_status()
print('✅ Ollama connection successful')
"; then
    echo "Ollama: OK"
else
    echo "❌ Ollama connection failed"
    exit 1
fi

# Initialize models and data
echo "🔧 Initializing system components..."

# Ensure model directory exists and has proper permissions
mkdir -p /app/models
chmod 755 /app/models

# Load pre-trained models if available
if [[ -d "/app/models/tactical" ]]; then
    echo "✅ Tactical models found"
else
    echo "⚠️  Tactical models not found, will download on first run"
fi

# Initialize database schema if needed
python3 -c "
from src.api.main import initialize_database
initialize_database()
print('✅ Database initialization complete')
"

# Start monitoring processes
echo "📊 Starting monitoring..."

# Start metrics collection in background
python3 -c "
from src.monitoring.metrics_exporter import start_metrics_server
start_metrics_server()
" &

# Memory monitoring
python3 -c "
from src.monitoring.memory_profiler import start_memory_monitor
start_memory_monitor()
" &

# Performance monitoring
py-spy top --pid $$ --duration 60 > "$LOG_DIR/startup-profile.txt" 2>&1 &

echo "🌟 All pre-flight checks passed!"

# Start the application with Gunicorn
echo "🚀 Starting XAI Trading System..."

exec gunicorn \
    --bind "$HOST:$PORT" \
    --workers "$WORKERS" \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --worker-class "$WORKER_CLASS" \
    --worker-connections 1000 \
    --timeout 300 \
    --keepalive 5 \
    --preload \
    --log-level "$LOG_LEVEL" \
    --access-logfile "$LOG_DIR/access.log" \
    --error-logfile "$LOG_DIR/error.log" \
    --capture-output \
    --enable-stdio-inheritance \
    --log-config configs/logging.yaml \
    --pid /tmp/gunicorn.pid \
    --graceful-timeout 30 \
    --worker-tmp-dir /dev/shm \
    "src.api.main:app"