# Tactical 5-Minute MARL System - High-Performance Docker Image
# Optimized for sub-100ms latency with TorchScript JIT compilation

# Build stage - Contains build tools and dependencies
FROM ubuntu:22.04 AS builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-pip \
    pkg-config \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt requirements-prod.txt ./

# Install Python dependencies with CPU-specific optimizations
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-prod.txt && \
    pip install --no-cache-dir torch==2.7.1+cpu torchvision==0.18.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
        intel-openmp \
        mkl \
        mkl-include

# Production stage - Minimal runtime image
FROM ubuntu:22.04 AS production

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.12 \
    python3.12-venv \
    curl \
    htop \
    iotop \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create tactical user for security
RUN groupadd -r tactical && useradd -r -g tactical tactical

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/
COPY scripts/ ./scripts/

# Copy JIT compilation script
COPY scripts/jit_compile_models.py ./scripts/

# Set CPU optimization environment variables
ENV TORCH_NUM_THREADS=16
ENV OMP_NUM_THREADS=16
ENV MKL_NUM_THREADS=16
ENV OPENBLAS_NUM_THREADS=16
ENV VECLIB_MAXIMUM_THREADS=16
ENV NUMEXPR_NUM_THREADS=16

# Enable CPU optimizations
ENV TORCH_CPU_ALLOCATOR_BACKEND=native
ENV TORCH_SHOW_CPP_STACKTRACES=1

# Memory optimization
ENV MALLOC_TRIM_THRESHOLD_=131072
ENV MALLOC_MMAP_THRESHOLD_=131072
ENV MALLOC_MMAP_MAX_=65536

# Python optimizations
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=random

# Security settings
ENV PYTHONSAFEPATH=1

# Create necessary directories
RUN mkdir -p /app/logs /app/models/tactical /app/data/tactical && \
    chown -R tactical:tactical /app

# Switch to tactical user
USER tactical

# Health check
HEALTHCHECK --interval=15s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expose tactical service port
EXPOSE 8001 9091

# JIT compilation and startup script
COPY --chown=tactical:tactical scripts/tactical_startup.sh /app/scripts/
RUN chmod +x /app/scripts/tactical_startup.sh

# Start tactical service with JIT compilation
CMD ["/app/scripts/tactical_startup.sh"]