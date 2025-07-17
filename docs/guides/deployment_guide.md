# Production Deployment Guide

## Overview

This guide covers deploying the GrandModel MARL trading system to production environments. We'll cover containerization, orchestration, monitoring, security, and operational best practices for running a high-frequency trading system in production.

## Table of Contents

- [Deployment Architecture](#deployment-architecture)
- [Environment Preparation](#environment-preparation)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Orchestration](#kubernetes-orchestration)
- [Configuration Management](#configuration-management)
- [Security Implementation](#security-implementation)
- [Monitoring and Alerting](#monitoring-and-alerting)
- [High Availability Setup](#high-availability-setup)
- [Performance Optimization](#performance-optimization)
- [Operational Procedures](#operational-procedures)

## Deployment Architecture

### Production Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Production Environment                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Load Balancer │  │  API Gateway    │  │   Web UI        │ │
│  │   (HAProxy)     │  │   (Kong/Nginx)  │  │   (React)       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                    Application Layer                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  GrandModel     │  │  GrandModel     │  │  Risk Manager   │ │
│  │  Instance 1     │  │  Instance 2     │  │  Service        │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                     Data Layer                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   PostgreSQL    │  │     Redis       │  │   TimescaleDB   │ │
│  │   (Primary)     │  │   (Cache)       │  │  (Time Series)  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                 Infrastructure Layer                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Prometheus    │  │    Grafana      │  │   ELK Stack     │ │
│  │  (Metrics)      │  │ (Dashboards)    │  │   (Logging)     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Deployment Environments

1. **Development**: Local development and testing
2. **Staging**: Pre-production validation
3. **Production**: Live trading environment
4. **DR (Disaster Recovery)**: Backup production environment

## Environment Preparation

### Infrastructure Requirements

#### Production Server Specifications

**Primary Trading Server:**
- CPU: 32+ cores, 3.5GHz+ (Intel Xeon or AMD EPYC)
- RAM: 128GB+ DDR4-3200
- Storage: 2TB NVMe SSD (primary) + 10TB HDD (backup)
- Network: 10Gbps+ with low latency
- OS: Ubuntu Server 22.04 LTS

**Secondary/DR Server:**
- CPU: 16+ cores, 3.0GHz+
- RAM: 64GB+ DDR4
- Storage: 1TB NVMe SSD + 5TB HDD
- Network: 1Gbps+ 
- OS: Ubuntu Server 22.04 LTS

#### Network Configuration

```bash
# High-performance network tuning
# /etc/sysctl.conf
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.ipv4.tcp_window_scaling = 1
net.ipv4.tcp_timestamps = 1
net.ipv4.tcp_sack = 1
net.core.netdev_max_backlog = 5000

# Apply changes
sudo sysctl -p
```

#### System Optimization

```bash
# CPU performance optimization
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable CPU idle states for low latency
sudo cpupower idle-set -D 0

# Memory huge pages configuration
echo 1024 | sudo tee /proc/sys/vm/nr_hugepages

# Set process limits
echo "grandmodel soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "grandmodel hard nofile 65536" | sudo tee -a /etc/security/limits.conf
```

### Software Prerequisites

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.21.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Kubernetes (if using K8s)
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee -a /etc/apt/sources.list.d/kubernetes.list
sudo apt update
sudo apt install -y kubelet kubeadm kubectl

# Install monitoring tools
sudo apt install -y htop iotop nethogs tcpdump wireshark-common
```

## Docker Deployment

### Production Dockerfile

```dockerfile
# docker/prod/Dockerfile
FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    g++ \
    git \
    libc6-dev \
    make \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Create non-root user
RUN useradd --create-home --shell /bin/bash grandmodel && \
    chown -R grandmodel:grandmodel /app
USER grandmodel

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start application
CMD ["python", "src/main.py"]
```

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  grandmodel:
    build:
      context: .
      dockerfile: docker/prod/Dockerfile
    image: grandmodel:production
    container_name: grandmodel-prod
    restart: unless-stopped
    environment:
      - GRANDMODEL_ENV=production
      - LOG_LEVEL=INFO
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://grandmodel:${DB_PASSWORD}@postgres:5432/grandmodel_prod
    volumes:
      - ./configs/production:/app/configs/production:ro
      - ./logs:/app/logs
      - ./models:/app/models:ro
    ports:
      - "8000:8000"
    depends_on:
      - redis
      - postgres
      - timescaledb
    networks:
      - grandmodel-network
    deploy:
      resources:
        limits:
          cpus: '8.0'
          memory: 16G
        reservations:
          cpus: '4.0'
          memory: 8G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    container_name: grandmodel-redis
    restart: unless-stopped
    command: redis-server --appendonly yes --maxmemory 4gb --maxmemory-policy allkeys-lru
    volumes:
      - redis-data:/data
    ports:
      - "6379:6379"
    networks:
      - grandmodel-network

  postgres:
    image: postgres:15-alpine
    container_name: grandmodel-postgres
    restart: unless-stopped
    environment:
      - POSTGRES_DB=grandmodel_prod
      - POSTGRES_USER=grandmodel
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./scripts/sql/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    ports:
      - "5432:5432"
    networks:
      - grandmodel-network

  timescaledb:
    image: timescale/timescaledb:latest-pg15
    container_name: grandmodel-timescale
    restart: unless-stopped
    environment:
      - POSTGRES_DB=grandmodel_timeseries
      - POSTGRES_USER=grandmodel
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - timescale-data:/var/lib/postgresql/data
    ports:
      - "5433:5432"
    networks:
      - grandmodel-network

  prometheus:
    image: prom/prometheus:latest
    container_name: grandmodel-prometheus
    restart: unless-stopped
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - grandmodel-network

  grafana:
    image: grafana/grafana:latest
    container_name: grandmodel-grafana
    restart: unless-stopped
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    ports:
      - "3000:3000"
    networks:
      - grandmodel-network

  nginx:
    image: nginx:alpine
    container_name: grandmodel-nginx
    restart: unless-stopped
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - grandmodel
    networks:
      - grandmodel-network

volumes:
  redis-data:
  postgres-data:
  timescale-data:
  prometheus-data:
  grafana-data:

networks:
  grandmodel-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### Production Deployment Commands

```bash
# Set environment variables
export DB_PASSWORD="your_secure_db_password"
export GRAFANA_PASSWORD="your_secure_grafana_password"

# Create production environment file
cat > .env.prod << EOF
DB_PASSWORD=${DB_PASSWORD}
GRAFANA_PASSWORD=${GRAFANA_PASSWORD}
GRANDMODEL_ENV=production
LOG_LEVEL=INFO
EOF

# Build and deploy
docker-compose -f docker-compose.prod.yml --env-file .env.prod up -d

# Verify deployment
docker-compose -f docker-compose.prod.yml ps
docker-compose -f docker-compose.prod.yml logs -f grandmodel
```

## Kubernetes Orchestration

### Kubernetes Manifests

#### Namespace and ConfigMap

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: grandmodel
  labels:
    name: grandmodel
    environment: production

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: grandmodel-config
  namespace: grandmodel
data:
  production.yaml: |
    system:
      environment: production
      log_level: INFO
    
    data_handler:
      type: rithmic
      connection:
        host: market-data-service
        port: 3001
    
    strategic_marl:
      enabled: true
      model_path: /app/models/strategic_agent.pth
    
    risk_management:
      max_position_size: 0.02
      max_daily_loss: 0.05
```

#### Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grandmodel
  namespace: grandmodel
  labels:
    app: grandmodel
    version: v1.0.0
spec:
  replicas: 2
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: grandmodel
  template:
    metadata:
      labels:
        app: grandmodel
        version: v1.0.0
    spec:
      serviceAccountName: grandmodel
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: grandmodel
        image: grandmodel:production
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          protocol: TCP
        env:
        - name: GRANDMODEL_ENV
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: grandmodel-secrets
              key: database-url
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        volumeMounts:
        - name: config-volume
          mountPath: /app/configs/production
          readOnly: true
        - name: models-volume
          mountPath: /app/models
          readOnly: true
        - name: logs-volume
          mountPath: /app/logs
        resources:
          requests:
            cpu: 2000m
            memory: 4Gi
          limits:
            cpu: 4000m
            memory: 8Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
      volumes:
      - name: config-volume
        configMap:
          name: grandmodel-config
      - name: models-volume
        persistentVolumeClaim:
          claimName: models-pvc
      - name: logs-volume
        persistentVolumeClaim:
          claimName: logs-pvc
      nodeSelector:
        node-type: trading
      tolerations:
      - key: "trading-workload"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
```

#### Service and Ingress

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: grandmodel-service
  namespace: grandmodel
  labels:
    app: grandmodel
spec:
  type: ClusterIP
  ports:
  - port: 8000
    targetPort: 8000
    protocol: TCP
    name: http
  selector:
    app: grandmodel

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: grandmodel-ingress
  namespace: grandmodel
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - trading.grandmodel.ai
    secretName: grandmodel-tls
  rules:
  - host: trading.grandmodel.ai
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: grandmodel-service
            port:
              number: 8000
```

### Deploy to Kubernetes

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# Verify deployment
kubectl get pods -n grandmodel
kubectl logs -f deployment/grandmodel -n grandmodel

# Check service status
kubectl get svc -n grandmodel
kubectl describe ingress grandmodel-ingress -n grandmodel
```

## Configuration Management

### Environment-Specific Configurations

#### Production Configuration

```yaml
# configs/production/production.yaml
system:
  environment: production
  log_level: INFO
  debug: false
  
security:
  enable_authentication: true
  jwt_secret_key: ${JWT_SECRET_KEY}
  encryption_key: ${ENCRYPTION_KEY}
  rate_limiting:
    enabled: true
    requests_per_minute: 1000
  
data_handler:
  type: rithmic
  connection:
    host: ${MARKET_DATA_HOST}
    port: 3001
    username: ${MARKET_DATA_USERNAME}
    password: ${MARKET_DATA_PASSWORD}
    timeout: 30
  symbols:
    - ES
    - NQ
  
database:
  primary:
    url: ${DATABASE_URL}
    pool_size: 20
    max_overflow: 30
    pool_timeout: 30
  timeseries:
    url: ${TIMESERIES_DATABASE_URL}
    pool_size: 10
    
redis:
  url: ${REDIS_URL}
  max_connections: 100
  socket_timeout: 30
  
strategic_marl:
  enabled: true
  model_path: /app/models/production/strategic_agent.pth
  batch_size: 32
  inference_timeout: 100  # milliseconds
  
risk_management:
  max_position_size: 0.02
  max_daily_loss: 0.05
  var_confidence: 0.95
  kelly_multiplier: 0.25
  emergency_stop_threshold: 0.10
  
monitoring:
  metrics_enabled: true
  metrics_port: 9090
  health_check_port: 8080
  log_sampling_rate: 0.1
  
performance:
  max_memory_usage: 8G
  cpu_affinity: [0, 1, 2, 3]  # Pin to specific CPU cores
  garbage_collection_frequency: 300  # seconds
```

### Secret Management

```bash
# Create Kubernetes secrets
kubectl create secret generic grandmodel-secrets \
  --from-literal=database-url="postgresql://user:pass@host:5432/db" \
  --from-literal=jwt-secret="your-jwt-secret" \
  --from-literal=encryption-key="your-encryption-key" \
  --from-literal=market-data-username="your-username" \
  --from-literal=market-data-password="your-password" \
  -n grandmodel

# Or use environment variables for Docker
cat > .env.production << EOF
DATABASE_URL=postgresql://user:pass@host:5432/db
TIMESERIES_DATABASE_URL=postgresql://user:pass@timescale:5432/timeseries
REDIS_URL=redis://redis:6379/0
JWT_SECRET_KEY=your-jwt-secret-here
ENCRYPTION_KEY=your-encryption-key-here
MARKET_DATA_HOST=market-data-feed.com
MARKET_DATA_USERNAME=your-username
MARKET_DATA_PASSWORD=your-password
EOF
```

### Configuration Validation

```python
# scripts/validate_production_config.py
import yaml
import os
import sys
from typing import Dict, List

class ProductionConfigValidator:
    """Validate production configuration before deployment"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.errors = []
        self.warnings = []
    
    def validate(self) -> bool:
        """Validate entire configuration"""
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            self.errors.append(f"Failed to load config: {e}")
            return False
        
        # Validate sections
        self._validate_system_config(config.get('system', {}))
        self._validate_security_config(config.get('security', {}))
        self._validate_database_config(config.get('database', {}))
        self._validate_marl_config(config.get('strategic_marl', {}))
        self._validate_risk_config(config.get('risk_management', {}))
        
        # Check environment variables
        self._validate_environment_variables()
        
        # Report results
        self._report_validation_results()
        
        return len(self.errors) == 0
    
    def _validate_system_config(self, config: Dict) -> None:
        """Validate system configuration"""
        
        if config.get('environment') != 'production':
            self.errors.append("Environment must be 'production'")
        
        if config.get('debug', False):
            self.warnings.append("Debug mode should be disabled in production")
        
        if config.get('log_level') not in ['INFO', 'WARNING', 'ERROR']:
            self.warnings.append("Log level should be INFO, WARNING, or ERROR in production")
    
    def _validate_security_config(self, config: Dict) -> None:
        """Validate security configuration"""
        
        if not config.get('enable_authentication', False):
            self.errors.append("Authentication must be enabled in production")
        
        if not config.get('jwt_secret_key'):
            self.errors.append("JWT secret key is required")
        
        rate_limiting = config.get('rate_limiting', {})
        if not rate_limiting.get('enabled', False):
            self.warnings.append("Rate limiting should be enabled in production")
    
    def _validate_environment_variables(self) -> None:
        """Validate required environment variables"""
        
        required_vars = [
            'DATABASE_URL',
            'REDIS_URL',
            'JWT_SECRET_KEY',
            'ENCRYPTION_KEY'
        ]
        
        for var in required_vars:
            if not os.getenv(var):
                self.errors.append(f"Required environment variable {var} is not set")
    
    def _report_validation_results(self) -> None:
        """Report validation results"""
        
        if self.errors:
            print("❌ Configuration validation failed:")
            for error in self.errors:
                print(f"  ERROR: {error}")
        
        if self.warnings:
            print("⚠️  Configuration warnings:")
            for warning in self.warnings:
                print(f"  WARNING: {warning}")
        
        if not self.errors and not self.warnings:
            print("✅ Configuration validation passed")

# Usage
if __name__ == "__main__":
    validator = ProductionConfigValidator("configs/production/production.yaml")
    success = validator.validate()
    sys.exit(0 if success else 1)
```

## Security Implementation

### Network Security

```bash
# Configure firewall
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (change port as needed)
sudo ufw allow 22/tcp

# Allow application ports
sudo ufw allow 8000/tcp  # GrandModel API
sudo ufw allow 3000/tcp  # Grafana
sudo ufw allow 9090/tcp  # Prometheus

# Allow database ports (only from application network)
sudo ufw allow from 172.20.0.0/16 to any port 5432
sudo ufw allow from 172.20.0.0/16 to any port 6379
```

### SSL/TLS Configuration

```nginx
# nginx/nginx.conf
server {
    listen 80;
    server_name trading.grandmodel.ai;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name trading.grandmodel.ai;
    
    ssl_certificate /etc/nginx/ssl/grandmodel.crt;
    ssl_certificate_key /etc/nginx/ssl/grandmodel.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    location / {
        proxy_pass http://grandmodel:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Rate limiting
        limit_req zone=api_limit burst=20 nodelay;
    }
}
```

### Application Security

```python
# src/security/authentication.py
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict

class SecurityManager:
    """Handle authentication and authorization"""
    
    def __init__(self, jwt_secret: str, encryption_key: str):
        self.jwt_secret = jwt_secret
        self.encryption_key = encryption_key
    
    def generate_jwt_token(self, user_id: str, permissions: List[str]) -> str:
        """Generate JWT token for authenticated user"""
        
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'exp': datetime.utcnow() + timedelta(hours=24),
            'iat': datetime.utcnow()
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
    
    def verify_jwt_token(self, token: str) -> Optional[Dict]:
        """Verify and decode JWT token"""
        
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def hash_password(self, password: str) -> str:
        """Hash password for storage"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
```

## Monitoring and Alerting

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'grandmodel'
    static_configs:
      - targets: ['grandmodel:9090']
    metrics_path: '/metrics'
    scrape_interval: 5s
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
    
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
```

### Alert Rules

```yaml
# monitoring/alert_rules.yml
groups:
  - name: grandmodel_alerts
    rules:
      - alert: HighMemoryUsage
        expr: (process_resident_memory_bytes / 1024 / 1024) > 8192
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GrandModel memory usage is high"
          description: "Memory usage is {{ $value }}MB, above 8GB threshold"
      
      - alert: HighCPUUsage
        expr: rate(process_cpu_seconds_total[5m]) > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GrandModel CPU usage is high"
          description: "CPU usage is {{ $value | humanizePercentage }}"
      
      - alert: TradingSystemDown
        expr: up{job="grandmodel"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "GrandModel trading system is down"
          description: "The trading system has been down for more than 1 minute"
      
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High request latency"
          description: "95th percentile latency is {{ $value }}s"
      
      - alert: RiskBreach
        expr: risk_var_current > risk_var_limit
        for: 0s
        labels:
          severity: critical
        annotations:
          summary: "Risk limit breached"
          description: "Current VaR {{ $value }} exceeds limit"
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "title": "GrandModel Trading System",
    "panels": [
      {
        "title": "System Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(process_cpu_seconds_total[5m])",
            "legendFormat": "CPU Usage"
          },
          {
            "expr": "process_resident_memory_bytes / 1024 / 1024",
            "legendFormat": "Memory Usage (MB)"
          }
        ]
      },
      {
        "title": "Trading Metrics",
        "type": "singlestat",
        "targets": [
          {
            "expr": "trading_pnl_total",
            "legendFormat": "Total P&L"
          }
        ]
      },
      {
        "title": "Risk Metrics",
        "type": "graph",
        "targets": [
          {
            "expr": "risk_var_current",
            "legendFormat": "Current VaR"
          },
          {
            "expr": "risk_exposure_total",
            "legendFormat": "Total Exposure"
          }
        ]
      }
    ]
  }
}
```

## High Availability Setup

### Multi-Region Deployment

```yaml
# k8s/ha-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grandmodel-primary
  namespace: grandmodel
spec:
  replicas: 3
  selector:
    matchLabels:
      app: grandmodel
      role: primary
  template:
    metadata:
      labels:
        app: grandmodel
        role: primary
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchLabels:
                app: grandmodel
            topologyKey: kubernetes.io/hostname
      containers:
      - name: grandmodel
        image: grandmodel:production
        env:
        - name: ROLE
          value: "primary"
        - name: CLUSTER_PEERS
          value: "grandmodel-0.grandmodel,grandmodel-1.grandmodel,grandmodel-2.grandmodel"

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grandmodel-secondary
  namespace: grandmodel
spec:
  replicas: 2
  selector:
    matchLabels:
      app: grandmodel
      role: secondary
  template:
    metadata:
      labels:
        app: grandmodel
        role: secondary
    spec:
      containers:
      - name: grandmodel
        image: grandmodel:production
        env:
        - name: ROLE
          value: "secondary"
        - name: PRIMARY_ENDPOINT
          value: "http://grandmodel-primary:8000"
```

### Database High Availability

```yaml
# Database failover setup
apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata:
  name: postgres-cluster
  namespace: grandmodel
spec:
  instances: 3
  primaryUpdateStrategy: unsupervised
  
  postgresql:
    parameters:
      max_connections: "200"
      shared_buffers: "256MB"
      effective_cache_size: "1GB"
      
  bootstrap:
    initdb:
      database: grandmodel_prod
      owner: grandmodel
      
  monitoring:
    enabled: true
    
  backup:
    retentionPolicy: "30d"
    barmanObjectStore:
      serverName: "postgres-backup"
      destinationPath: "s3://grandmodel-backups/postgres"
```

## Performance Optimization

### Application Tuning

```python
# src/performance/optimization.py
import asyncio
import uvloop
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

class PerformanceOptimizer:
    """Optimize application performance for production"""
    
    def __init__(self):
        self.cpu_count = mp.cpu_count()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.cpu_count * 2)
    
    def setup_event_loop(self):
        """Setup optimized event loop"""
        # Use uvloop for better performance
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        
        # Configure loop settings
        loop = asyncio.get_event_loop()
        loop.set_default_executor(self.thread_pool)
        
        return loop
    
    def setup_cpu_affinity(self, core_ids: List[int]):
        """Pin process to specific CPU cores"""
        import psutil
        p = psutil.Process()
        p.cpu_affinity(core_ids)
    
    def optimize_memory_usage(self):
        """Optimize memory allocation"""
        import gc
        
        # Tune garbage collection
        gc.set_threshold(700, 10, 10)
        
        # Force garbage collection
        gc.collect()
```

### Database Optimization

```sql
-- Database performance tuning
-- configs/sql/performance_tuning.sql

-- Connection pooling
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET work_mem = '256MB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';

-- WAL settings for performance
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET checkpoint_timeout = '15min';

-- Query optimization
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

-- Reload configuration
SELECT pg_reload_conf();

-- Create performance indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_timestamp 
ON trades(timestamp DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_symbol_timestamp 
ON market_data(symbol, timestamp DESC);

-- Partition large tables
CREATE TABLE market_data_partitioned (
    LIKE market_data INCLUDING ALL
) PARTITION BY RANGE (timestamp);

-- Create monthly partitions
CREATE TABLE market_data_2025_01 PARTITION OF market_data_partitioned
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
```

## Operational Procedures

### Deployment Procedures

```bash
#!/bin/bash
# scripts/deploy_production.sh

set -e

DEPLOYMENT_VERSION=$1
ROLLBACK_VERSION=$2

echo "🚀 Starting production deployment of version $DEPLOYMENT_VERSION"

# Pre-deployment checks
echo "📋 Running pre-deployment checks..."
python scripts/validate_production_config.py
python scripts/check_dependencies.py
python scripts/verify_models.py

# Backup current state
echo "💾 Creating backup..."
kubectl create backup grandmodel-backup-$(date +%Y%m%d-%H%M%S) -n grandmodel

# Deploy new version
echo "🔄 Deploying new version..."
kubectl set image deployment/grandmodel grandmodel=grandmodel:$DEPLOYMENT_VERSION -n grandmodel

# Wait for rollout
echo "⏳ Waiting for deployment to complete..."
kubectl rollout status deployment/grandmodel -n grandmodel --timeout=600s

# Run health checks
echo "🔍 Running health checks..."
sleep 30
./scripts/health_check.sh

# Verify metrics
echo "📊 Verifying metrics..."
python scripts/verify_deployment_metrics.py

echo "✅ Deployment completed successfully"
```

### Monitoring Procedures

```bash
#!/bin/bash
# scripts/health_check.sh

set -e

HEALTH_ENDPOINT="https://trading.grandmodel.ai/health"
METRICS_ENDPOINT="https://trading.grandmodel.ai/metrics"

echo "🔍 Performing health check..."

# Check application health
response=$(curl -s -o /dev/null -w "%{http_code}" $HEALTH_ENDPOINT)
if [ $response -eq 200 ]; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed (HTTP $response)"
    exit 1
fi

# Check metrics endpoint
response=$(curl -s -o /dev/null -w "%{http_code}" $METRICS_ENDPOINT)
if [ $response -eq 200 ]; then
    echo "✅ Metrics endpoint accessible"
else
    echo "⚠️  Metrics endpoint not accessible (HTTP $response)"
fi

# Check database connectivity
kubectl exec -n grandmodel deployment/grandmodel -- python -c "
import psycopg2
import os
try:
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    conn.close()
    print('✅ Database connectivity confirmed')
except Exception as e:
    print(f'❌ Database connectivity failed: {e}')
    exit(1)
"

# Check Redis connectivity
kubectl exec -n grandmodel deployment/grandmodel -- python -c "
import redis
import os
try:
    r = redis.from_url(os.environ['REDIS_URL'])
    r.ping()
    print('✅ Redis connectivity confirmed')
except Exception as e:
    print(f'❌ Redis connectivity failed: {e}')
    exit(1)
"

echo "🎉 All health checks passed"
```

### Backup and Recovery

```bash
#!/bin/bash
# scripts/backup_system.sh

set -e

BACKUP_DATE=$(date +%Y%m%d-%H%M%S)
BACKUP_DIR="/backups/grandmodel-$BACKUP_DATE"

echo "💾 Starting system backup..."

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database
echo "🗃️  Backing up database..."
kubectl exec -n grandmodel deployment/postgres -- pg_dump grandmodel_prod > $BACKUP_DIR/database.sql

# Backup Redis
echo "📦 Backing up Redis..."
kubectl exec -n grandmodel deployment/redis -- redis-cli BGSAVE
kubectl cp grandmodel/redis:/data/dump.rdb $BACKUP_DIR/redis-dump.rdb

# Backup configurations
echo "⚙️  Backing up configurations..."
kubectl get configmaps -n grandmodel -o yaml > $BACKUP_DIR/configmaps.yaml
kubectl get secrets -n grandmodel -o yaml > $BACKUP_DIR/secrets.yaml

# Backup models
echo "🧠 Backing up models..."
kubectl cp grandmodel/grandmodel:/app/models $BACKUP_DIR/models

# Create backup archive
echo "📁 Creating backup archive..."
tar -czf grandmodel-backup-$BACKUP_DATE.tar.gz -C /backups grandmodel-$BACKUP_DATE

# Upload to S3 (if configured)
if [ -n "$AWS_S3_BACKUP_BUCKET" ]; then
    echo "☁️  Uploading to S3..."
    aws s3 cp grandmodel-backup-$BACKUP_DATE.tar.gz s3://$AWS_S3_BACKUP_BUCKET/
fi

echo "✅ Backup completed: grandmodel-backup-$BACKUP_DATE.tar.gz"
```

This comprehensive deployment guide provides the foundation for running GrandModel in production environments. The configuration is designed for high performance, reliability, and security while maintaining operational simplicity.

## Related Documentation

- [Getting Started Guide](getting_started.md)
- [Training Guide](training_guide.md)
- [Architecture Overview](../architecture/system_overview.md)
- [API Documentation](../api/)
- [Security Guide](../architecture/security.md)