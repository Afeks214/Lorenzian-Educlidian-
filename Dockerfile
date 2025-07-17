# GrandModel Production Dockerfile
# AGENT 5 - Configuration Recovery Mission
# Multi-stage build for production deployment

# Base Python image
FROM python:3.12-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    git \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r grandmodel && useradd -r -g grandmodel grandmodel

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install TA-Lib (required for technical indicators)
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Development stage
FROM base as development

# Copy requirements
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install Jupyter and development tools
RUN pip install jupyter jupyterlab ipywidgets

# Copy source code
COPY . .

# Change ownership to non-root user
RUN chown -R grandmodel:grandmodel /app

# Switch to non-root user
USER grandmodel

# Expose ports
EXPOSE 8000 8888

# Development command
CMD ["python", "-m", "src.main"]

# Production stage
FROM base as production

# Copy production requirements
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Copy configuration files
COPY production_config.yaml .

# Create necessary directories
RUN mkdir -p logs data models temp

# Change ownership to non-root user
RUN chown -R grandmodel:grandmodel /app

# Switch to non-root user
USER grandmodel

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 9090

# Production command
CMD ["python", "-m", "src.main", "--config", "production_config.yaml"]

# Strategic Agent stage
FROM production as strategic

# Copy strategic agent specific files
COPY src/agents/strategic/ ./src/agents/strategic/
COPY configs/trading/strategic_config.yaml ./configs/trading/

# Environment
ENV AGENT_TYPE=strategic
ENV MODEL_PATH=/app/models/strategic

# Command
CMD ["python", "-m", "src.agents.strategic.main"]

# Tactical Agent stage
FROM production as tactical

# Copy tactical agent specific files
COPY src/agents/tactical/ ./src/agents/tactical/
COPY configs/trading/tactical_config.yaml ./configs/trading/

# Environment
ENV AGENT_TYPE=tactical
ENV MODEL_PATH=/app/models/tactical

# Command
CMD ["python", "-m", "src.agents.tactical.main"]

# Risk Agent stage
FROM production as risk

# Copy risk agent specific files
COPY src/risk/ ./src/risk/
COPY configs/trading/risk_config.yaml ./configs/trading/

# Environment
ENV AGENT_TYPE=risk
ENV MODEL_PATH=/app/models/risk

# Command
CMD ["python", "-m", "src.risk.main"]

# Testing stage
FROM base as testing

# Copy test requirements
COPY requirements.txt .
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy all source code for testing
COPY . .

# Change ownership
RUN chown -R grandmodel:grandmodel /app

# Switch to non-root user
USER grandmodel

# Test command
CMD ["pytest", "tests/", "-v", "--cov=src"]

# Multi-arch build support
FROM --platform=$BUILDPLATFORM base as builder

ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Copy requirements for the target platform
COPY requirements-prod.txt .

# Install dependencies for target platform
RUN pip install --no-cache-dir -r requirements-prod.txt

# Final production image
FROM python:3.12-slim as final

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN groupadd -r grandmodel && useradd -r -g grandmodel grandmodel

# Set working directory
WORKDIR /app

# Copy from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application
COPY --chown=grandmodel:grandmodel src/ ./src/
COPY --chown=grandmodel:grandmodel configs/ ./configs/
COPY --chown=grandmodel:grandmodel production_config.yaml .

# Create directories
RUN mkdir -p logs data models temp && chown -R grandmodel:grandmodel /app

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER grandmodel

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000

# Default command
CMD ["python", "-m", "src.main"]