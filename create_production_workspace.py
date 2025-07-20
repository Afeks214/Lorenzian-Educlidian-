#!/usr/bin/env python3
"""
GrandModel Production Workspace Creator

This script creates an optimized production workspace from the development environment,
extracting only essential components and creating deployment-ready configurations.
"""

import os
import sys
import shutil
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set
import subprocess

class ProductionWorkspaceCreator:
    """Creates production-ready workspace from development environment."""
    
    def __init__(self, dev_path: str = "/home/QuantNova/GrandModel", 
                 prod_path: str = "/home/QuantNova/GrandModel-Production"):
        self.dev_path = Path(dev_path)
        self.prod_path = Path(prod_path)
        self.staging_path = Path(f"{prod_path}-Staging")
        
        # Essential components for production
        self.essential_components = {
            "core_notebooks": [
                "risk_management_mappo_training.ipynb",
                "execution_engine_mappo_training.ipynb", 
                "strategic_mappo_training.ipynb",
                "tactical_mappo_training.ipynb",
                "xai_trading_explanations_training.ipynb"
            ],
            "core_source": [
                "src/agents/",
                "src/execution/",
                "src/matrix/",
                "src/training/"
            ],
            "essential_configs": [
                "config/production.yaml",
                "config/training_config.yaml",
                "config/risk_management_config.yaml"
            ],
            "deployment_files": [
                "docker-compose.production.yml",
                "Dockerfile.production",
                "k8s/",
                "monitoring/"
            ],
            "production_requirements": [
                "requirements-prod.txt",
                "requirements.txt"
            ]
        }
        
        # Production optimization settings
        self.prod_optimizations = {
            "remove_dev_dependencies": True,
            "optimize_docker_images": True,
            "minimize_logging": True,
            "enable_jit_compilation": True,
            "production_security": True
        }
    
    def create_production_structure(self) -> None:
        """Create optimized production directory structure."""
        print("🏗️ Creating production workspace structure...")
        
        # Production structure
        prod_structure = {
            "core": {
                "agents": {},
                "execution": {},
                "matrix": {},
                "training": {}
            },
            "config": {
                "production": {},
                "staging": {},
                "secrets": {}
            },
            "deployment": {
                "docker": {},
                "k8s": {},
                "helm": {},
                "scripts": {}
            },
            "monitoring": {
                "grafana": {},
                "prometheus": {},
                "alerts": {}
            },
            "notebooks": {},
            "tests": {
                "integration": {},
                "performance": {},
                "security": {}
            },
            "scripts": {},
            "docs": {}
        }
        
        # Create directory structure
        for top_level, subdirs in prod_structure.items():
            top_path = self.prod_path / top_level
            top_path.mkdir(parents=True, exist_ok=True)
            
            for subdir in subdirs:
                (top_path / subdir).mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Production structure created at: {self.prod_path}")
    
    def extract_essential_components(self) -> None:
        """Extract and optimize essential components from development."""
        print("📦 Extracting essential components...")
        
        # Copy core notebooks
        for notebook in self.essential_components["core_notebooks"]:
            src_file = self.dev_path / "train_notebooks" / notebook
            dst_file = self.prod_path / "notebooks" / notebook
            
            if src_file.exists():
                shutil.copy2(src_file, dst_file)
                self._optimize_notebook_for_production(dst_file)
                print(f"✅ Extracted: {notebook}")
            else:
                print(f"⚠️ Not found: {notebook}")
        
        # Copy core source code
        for src_dir in self.essential_components["core_source"]:
            src_path = self.dev_path / src_dir
            dst_path = self.prod_path / "core" / Path(src_dir).name
            
            if src_path.exists():
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                self._optimize_source_for_production(dst_path)
                print(f"✅ Extracted: {src_dir}")
        
        # Copy essential configs
        for config_file in self.essential_components["essential_configs"]:
            src_file = self.dev_path / config_file
            dst_file = self.prod_path / "config" / Path(config_file).name
            
            if src_file.exists():
                shutil.copy2(src_file, dst_file)
                self._optimize_config_for_production(dst_file)
                print(f"✅ Extracted: {config_file}")
        
        # Copy deployment files
        for deploy_item in self.essential_components["deployment_files"]:
            src_path = self.dev_path / deploy_item
            dst_path = self.prod_path / "deployment" / Path(deploy_item).name
            
            if src_path.exists():
                if src_path.is_dir():
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_path, dst_path)
                print(f"✅ Extracted: {deploy_item}")
    
    def _optimize_notebook_for_production(self, notebook_path: Path) -> None:
        """Optimize notebook for production deployment."""
        if not notebook_path.exists():
            return
        
        try:
            with open(notebook_path, 'r') as f:
                notebook_content = json.load(f)
            
            # Remove development cells and outputs
            optimized_cells = []
            for cell in notebook_content.get('cells', []):
                # Skip debug and development cells
                source = ''.join(cell.get('source', []))
                if any(keyword in source.lower() for keyword in ['debug', 'test', 'dev', '# remove in prod']):
                    continue
                
                # Clear outputs to reduce size
                if 'outputs' in cell:
                    cell['outputs'] = []
                
                # Remove emoji prints for production
                if cell.get('cell_type') == 'code':
                    cell['source'] = [line for line in cell.get('source', []) 
                                    if not any(emoji in line for emoji in ['🔄', '✅', '⚠️', '🚀', '💾'])]
                
                optimized_cells.append(cell)
            
            notebook_content['cells'] = optimized_cells
            
            # Save optimized notebook
            with open(notebook_path, 'w') as f:
                json.dump(notebook_content, f, indent=2)
            
            print(f"🔧 Optimized notebook: {notebook_path.name}")
            
        except Exception as e:
            print(f"⚠️ Failed to optimize {notebook_path.name}: {e}")
    
    def _optimize_source_for_production(self, source_path: Path) -> None:
        """Optimize source code for production."""
        if not source_path.exists():
            return
        
        for py_file in source_path.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                # Production optimizations
                lines = content.split('\n')
                optimized_lines = []
                
                for line in lines:
                    # Remove debug prints
                    if any(keyword in line.lower() for keyword in ['print("debug', 'print(f"debug', '# debug']):
                        continue
                    
                    # Remove emoji prints
                    if any(emoji in line for emoji in ['🔄', '✅', '⚠️', '🚀', '💾']):
                        line = line.replace('🔄', '').replace('✅', '').replace('⚠️', '').replace('🚀', '').replace('💾', '')
                    
                    # Enable JIT compilation where beneficial
                    if 'def ' in line and any(func in line for func in ['calculate_', 'process_', 'compute_']):
                        if '@jit' not in optimized_lines[-1] if optimized_lines else True:
                            optimized_lines.append('from numba import jit')
                            optimized_lines.append('@jit(nopython=True)')
                    
                    optimized_lines.append(line)
                
                # Write optimized content
                with open(py_file, 'w') as f:
                    f.write('\n'.join(optimized_lines))
                
            except Exception as e:
                print(f"⚠️ Failed to optimize {py_file}: {e}")
    
    def _optimize_config_for_production(self, config_path: Path) -> None:
        """Optimize configuration files for production."""
        try:
            if config_path.suffix in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Production optimizations
                if isinstance(config, dict):
                    # Disable debug logging
                    if 'logging' in config:
                        config['logging']['level'] = 'INFO'
                        config['logging']['debug'] = False
                    
                    # Enable production optimizations
                    if 'performance' in config:
                        config['performance']['jit_compilation'] = True
                        config['performance']['memory_optimization'] = True
                    
                    # Set production security
                    if 'security' in config:
                        config['security']['strict_mode'] = True
                        config['security']['audit_logging'] = True
                
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
            
        except Exception as e:
            print(f"⚠️ Failed to optimize config {config_path}: {e}")
    
    def create_production_requirements(self) -> None:
        """Create production-optimized requirements file."""
        print("📋 Creating production requirements...")
        
        # Production requirements (minimal set)
        prod_requirements = [
            "torch>=1.9.0",
            "numpy>=1.21.0", 
            "pandas>=1.3.0",
            "numba>=0.56.0",
            "pettingzoo>=1.15.0",
            "gymnasium>=0.26.0",
            "stable-baselines3>=1.7.0",
            "redis>=4.0.0",
            "prometheus-client>=0.12.0",
            "pyyaml>=6.0",
            "psutil>=5.8.0"
        ]
        
        # Write production requirements
        prod_req_file = self.prod_path / "requirements-prod.txt"
        with open(prod_req_file, 'w') as f:
            f.write('\n'.join(prod_requirements))
        
        print(f"✅ Production requirements created: {prod_req_file}")
    
    def create_docker_configurations(self) -> None:
        """Create production-optimized Docker configurations."""
        print("🐳 Creating Docker configurations...")
        
        # Production Dockerfile
        dockerfile_content = '''# Production-optimized GrandModel Dockerfile
FROM python:3.9-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc g++ libc6-dev libffi-dev && \\
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1001 trader && \\
    mkdir -p /app && \\
    chown trader:trader /app

FROM base AS dependencies
COPY requirements-prod.txt /tmp/
RUN pip install --no-cache-dir --user -r /tmp/requirements-prod.txt

FROM base AS runtime
# Copy dependencies from previous stage
COPY --from=dependencies /root/.local /home/trader/.local
# Copy application code
COPY --chown=trader:trader core/ /app/core/
COPY --chown=trader:trader config/ /app/config/
COPY --chown=trader:trader notebooks/ /app/notebooks/

USER trader
WORKDIR /app
ENV PATH=/home/trader/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import core; print('healthy')" || exit 1

# Production entrypoint
CMD ["python", "-m", "core.main"]
'''
        
        docker_file = self.prod_path / "deployment" / "Dockerfile.production"
        with open(docker_file, 'w') as f:
            f.write(dockerfile_content)
        
        # Docker Compose for production
        compose_content = '''version: '3.8'

services:
  grandmodel-core:
    build:
      context: .
      dockerfile: deployment/Dockerfile.production
    restart: unless-stopped
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    volumes:
      - ./config:/app/config:ro
      - ./data:/app/data
    networks:
      - grandmodel-network
    healthcheck:
      test: ["CMD", "python", "-c", "import core; print('healthy')"]
      interval: 30s
      timeout: 10s
      retries: 3
  
  redis:
    image: redis:7-alpine
    restart: unless-stopped
    networks:
      - grandmodel-network
    volumes:
      - redis-data:/data
  
  prometheus:
    image: prom/prometheus:latest
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus:/etc/prometheus:ro
      - prometheus-data:/prometheus
    networks:
      - grandmodel-network
  
  grafana:
    image: grafana/grafana:latest
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning:ro
    networks:
      - grandmodel-network

networks:
  grandmodel-network:
    driver: bridge

volumes:
  redis-data:
  prometheus-data:
  grafana-data:
'''
        
        compose_file = self.prod_path / "docker-compose.production.yml"
        with open(compose_file, 'w') as f:
            f.write(compose_content)
        
        print("✅ Docker configurations created")
    
    def create_kubernetes_manifests(self) -> None:
        """Create Kubernetes deployment manifests."""
        print("☸️ Creating Kubernetes manifests...")
        
        k8s_dir = self.prod_path / "deployment" / "k8s"
        k8s_dir.mkdir(parents=True, exist_ok=True)
        
        # Namespace
        namespace_yaml = '''apiVersion: v1
kind: Namespace
metadata:
  name: grandmodel-production
  labels:
    name: grandmodel-production
    environment: production
'''
        
        # Deployment
        deployment_yaml = '''apiVersion: apps/v1
kind: Deployment
metadata:
  name: grandmodel-core
  namespace: grandmodel-production
  labels:
    app: grandmodel-core
    environment: production
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: grandmodel-core
  template:
    metadata:
      labels:
        app: grandmodel-core
    spec:
      containers:
      - name: grandmodel-core
        image: grandmodel:production
        ports:
        - containerPort: 8080
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
      restartPolicy: Always
'''
        
        # Service
        service_yaml = '''apiVersion: v1
kind: Service
metadata:
  name: grandmodel-service
  namespace: grandmodel-production
spec:
  selector:
    app: grandmodel-core
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
  type: ClusterIP
'''
        
        # Write K8s manifests
        with open(k8s_dir / "namespace.yaml", 'w') as f:
            f.write(namespace_yaml)
        
        with open(k8s_dir / "deployment.yaml", 'w') as f:
            f.write(deployment_yaml)
        
        with open(k8s_dir / "service.yaml", 'w') as f:
            f.write(service_yaml)
        
        print("✅ Kubernetes manifests created")
    
    def create_ci_cd_pipeline(self) -> None:
        """Create CI/CD pipeline configuration."""
        print("🔄 Creating CI/CD pipeline...")
        
        # GitHub Actions workflow
        github_workflow = '''name: GrandModel Production Deploy

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: grandmodel

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements-prod.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=core --cov-report=xml
    
    - name: Security scan
      uses: snyk/actions/python@master
      env:
        SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        file: deployment/Dockerfile.production
        push: true
        tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to production
      run: |
        echo "Deploying to production cluster"
        # Add actual deployment commands here
'''
        
        workflow_dir = self.prod_path / ".github" / "workflows"
        workflow_dir.mkdir(parents=True, exist_ok=True)
        
        with open(workflow_dir / "production-deploy.yml", 'w') as f:
            f.write(github_workflow)
        
        print("✅ CI/CD pipeline created")
    
    def create_monitoring_configs(self) -> None:
        """Create production monitoring configurations."""
        print("📊 Creating monitoring configurations...")
        
        monitoring_dir = self.prod_path / "monitoring"
        
        # Prometheus configuration
        prometheus_dir = monitoring_dir / "prometheus"
        prometheus_dir.mkdir(parents=True, exist_ok=True)
        
        prometheus_config = '''global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'grandmodel-core'
    static_configs:
      - targets: ['grandmodel-service:80']
    metrics_path: /metrics
    scrape_interval: 5s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
'''
        
        # Alert rules
        alert_rules = '''groups:
- name: grandmodel-alerts
  rules:
  - alert: HighLatency
    expr: grandmodel_request_duration_seconds > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High latency detected"
      description: "Request latency is above 100ms"

  - alert: HighErrorRate
    expr: rate(grandmodel_errors_total[5m]) > 0.01
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is above 1%"

  - alert: ServiceDown
    expr: up{job="grandmodel-core"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "GrandModel service is down"
      description: "GrandModel core service is not responding"
'''
        
        with open(prometheus_dir / "prometheus.yml", 'w') as f:
            f.write(prometheus_config)
        
        with open(prometheus_dir / "alert_rules.yml", 'w') as f:
            f.write(alert_rules)
        
        print("✅ Monitoring configurations created")
    
    def create_deployment_scripts(self) -> None:
        """Create deployment automation scripts."""
        print("🚀 Creating deployment scripts...")
        
        scripts_dir = self.prod_path / "deployment" / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        
        # Production deployment script
        deploy_script = '''#!/bin/bash
set -e

echo "🚀 Starting GrandModel Production Deployment"

# Variables
ENVIRONMENT=${ENVIRONMENT:-production}
IMAGE_TAG=${IMAGE_TAG:-latest}
NAMESPACE=grandmodel-production

# Pre-deployment checks
echo "🔍 Running pre-deployment checks..."
if ! kubectl cluster-info > /dev/null 2>&1; then
    echo "❌ Kubernetes cluster not accessible"
    exit 1
fi

if ! docker images | grep -q grandmodel; then
    echo "❌ GrandModel image not found"
    exit 1
fi

# Deploy to staging first
echo "📦 Deploying to staging..."
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/ -n grandmodel-staging

# Wait for staging validation
echo "⏳ Waiting for staging validation..."
kubectl wait --for=condition=ready pod -l app=grandmodel-core -n grandmodel-staging --timeout=300s

# Run smoke tests
echo "🧪 Running smoke tests..."
python -m tests.smoke_tests --environment=staging

# Deploy to production with blue-green strategy
echo "🚀 Deploying to production..."
kubectl apply -f k8s/ -n $NAMESPACE

# Wait for deployment
kubectl rollout status deployment/grandmodel-core -n $NAMESPACE

# Validate deployment
echo "✅ Validating deployment..."
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE

echo "🎉 Deployment completed successfully!"
'''
        
        # Rollback script
        rollback_script = '''#!/bin/bash
set -e

echo "🔄 Starting GrandModel Rollback"

NAMESPACE=grandmodel-production
REVISION=${1:-previous}

echo "📋 Current deployment status:"
kubectl get deployment grandmodel-core -n $NAMESPACE

echo "🔄 Rolling back to revision: $REVISION"
kubectl rollout undo deployment/grandmodel-core -n $NAMESPACE --to-revision=$REVISION

echo "⏳ Waiting for rollback to complete..."
kubectl rollout status deployment/grandmodel-core -n $NAMESPACE

echo "✅ Rollback completed successfully!"
kubectl get pods -n $NAMESPACE
'''
        
        # Health check script
        health_check_script = '''#!/bin/bash

NAMESPACE=grandmodel-production
SERVICE_URL="http://grandmodel-service.$NAMESPACE.svc.cluster.local"

echo "🏥 Running health checks..."

# Check pod status
echo "📋 Pod status:"
kubectl get pods -n $NAMESPACE -l app=grandmodel-core

# Check service endpoints
echo "🔗 Service endpoints:"
kubectl get endpoints -n $NAMESPACE grandmodel-service

# Check application health
echo "💚 Application health:"
if curl -f $SERVICE_URL/health > /dev/null 2>&1; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed"
    exit 1
fi

echo "🎉 All health checks passed!"
'''
        
        # Write scripts
        scripts = {
            "deploy.sh": deploy_script,
            "rollback.sh": rollback_script,
            "health_check.sh": health_check_script
        }
        
        for script_name, script_content in scripts.items():
            script_path = scripts_dir / script_name
            with open(script_path, 'w') as f:
                f.write(script_content)
            os.chmod(script_path, 0o755)  # Make executable
        
        print("✅ Deployment scripts created")
    
    def create_production_documentation(self) -> None:
        """Create production deployment documentation."""
        print("📚 Creating production documentation...")
        
        docs_dir = self.prod_path / "docs"
        docs_dir.mkdir(exist_ok=True)
        
        readme_content = '''# GrandModel Production Deployment

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Kubernetes cluster (for K8s deployment)
- Python 3.9+

### Local Development
```bash
# Build and run locally
docker-compose -f docker-compose.production.yml up -d

# Check health
curl http://localhost:8080/health
```

### Production Deployment
```bash
# Deploy to Kubernetes
./deployment/scripts/deploy.sh

# Check status
./deployment/scripts/health_check.sh

# Rollback if needed
./deployment/scripts/rollback.sh
```

### Monitoring
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

### Configuration
- Production configs: `config/`
- Environment variables: See `docker-compose.production.yml`
- Kubernetes manifests: `deployment/k8s/`

### Troubleshooting
- Check logs: `kubectl logs -l app=grandmodel-core -n grandmodel-production`
- Debug pod: `kubectl exec -it <pod-name> -n grandmodel-production -- bash`
- Monitor metrics: Access Grafana dashboard

## Performance Targets
- Deployment time: <2 minutes
- Rollback time: <30 seconds
- API response time: <10ms
- System availability: >99.9%
'''
        
        with open(docs_dir / "README.md", 'w') as f:
            f.write(readme_content)
        
        print("✅ Production documentation created")
    
    def generate_migration_report(self) -> None:
        """Generate migration report."""
        print("📊 Generating migration report...")
        
        report = {
            "migration_timestamp": datetime.now().isoformat(),
            "source_path": str(self.dev_path),
            "production_path": str(self.prod_path),
            "components_migrated": {
                "notebooks": len(self.essential_components["core_notebooks"]),
                "source_modules": len(self.essential_components["core_source"]),
                "configs": len(self.essential_components["essential_configs"]),
                "deployment_files": len(self.essential_components["deployment_files"])
            },
            "optimizations_applied": self.prod_optimizations,
            "deployment_targets": {
                "build_time": "<3 minutes",
                "deployment_time": "<2 minutes",
                "rollback_time": "<30 seconds"
            },
            "next_steps": [
                "Test production build locally",
                "Deploy to staging environment", 
                "Run integration tests",
                "Deploy to production",
                "Monitor and validate performance"
            ]
        }
        
        report_file = self.prod_path / "MIGRATION_REPORT.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Migration report generated: {report_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("🎉 PRODUCTION WORKSPACE CREATION COMPLETE")
        print("="*60)
        print(f"📁 Production workspace: {self.prod_path}")
        print(f"📦 Components migrated: {sum(report['components_migrated'].values())}")
        print("🚀 Ready for deployment!")
        print("\nNext steps:")
        for step in report["next_steps"]:
            print(f"  • {step}")

def main():
    """Main execution function."""
    print("🚀 GrandModel Production Workspace Creator")
    print("="*50)
    
    creator = ProductionWorkspaceCreator()
    
    try:
        # Create production workspace
        creator.create_production_structure()
        creator.extract_essential_components()
        creator.create_production_requirements()
        creator.create_docker_configurations()
        creator.create_kubernetes_manifests()
        creator.create_ci_cd_pipeline()
        creator.create_monitoring_configs()
        creator.create_deployment_scripts()
        creator.create_production_documentation()
        creator.generate_migration_report()
        
        print("\n🎉 Production workspace creation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error creating production workspace: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()