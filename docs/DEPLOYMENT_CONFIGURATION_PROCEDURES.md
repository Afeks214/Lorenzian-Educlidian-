# 🚀 GRANDMODEL DEPLOYMENT & CONFIGURATION PROCEDURES
**COMPREHENSIVE DEPLOYMENT GUIDE**

---

## 📋 DOCUMENT OVERVIEW

**Document Purpose**: Complete deployment and configuration procedures for GrandModel system  
**Target Audience**: DevOps engineers, system administrators, deployment teams  
**Classification**: DEPLOYMENT CRITICAL  
**Version**: 1.0  
**Last Updated**: July 17, 2025  
**Agent**: Documentation & Training Agent (Agent 9)

---

## 🎯 DEPLOYMENT OVERVIEW

The GrandModel system supports multiple deployment strategies across different environments. This document provides comprehensive procedures for deploying, configuring, and managing the system across development, staging, and production environments.

### Deployment Principles
- **Infrastructure as Code**: All infrastructure defined in version control
- **Automated Deployments**: Fully automated CI/CD pipelines
- **Environment Parity**: Consistent configuration across environments
- **Zero-Downtime Deployments**: Blue-green and rolling deployment strategies
- **Rollback Capability**: Quick rollback procedures for issues

### Supported Deployment Targets
- **Kubernetes**: Primary deployment platform
- **Docker Compose**: Development and testing
- **Cloud Platforms**: AWS, Azure, Google Cloud
- **On-Premises**: Private cloud and bare metal

---

## 🏗️ INFRASTRUCTURE REQUIREMENTS

### 1. Minimum System Requirements

#### Development Environment
```yaml
development:
  nodes: 1
  cpu_per_node: 4 cores
  memory_per_node: 8 GB
  storage_per_node: 50 GB SSD
  network: 1 Gbps
  kubernetes_version: "1.26+"
```

#### Staging Environment
```yaml
staging:
  nodes: 3
  cpu_per_node: 8 cores
  memory_per_node: 16 GB
  storage_per_node: 100 GB SSD
  network: 10 Gbps
  kubernetes_version: "1.26+"
  high_availability: true
```

#### Production Environment
```yaml
production:
  nodes: 5
  cpu_per_node: 16 cores
  memory_per_node: 32 GB
  storage_per_node: 500 GB SSD
  network: 10 Gbps
  kubernetes_version: "1.26+"
  high_availability: true
  disaster_recovery: true
```

### 2. Network Requirements
```yaml
network_requirements:
  internal_network: "10.0.0.0/16"
  service_network: "10.100.0.0/16"
  pod_network: "10.200.0.0/16"
  
  ingress_ports:
    - 80   # HTTP
    - 443  # HTTPS
    - 8080 # API Gateway
    
  egress_requirements:
    - rithmic_api: "api.rithmic.com:443"
    - ib_api: "api.interactivebrokers.com:4001"
    - monitoring: "prometheus.io:443"
    
  firewall_rules:
    - allow_ingress_from_load_balancer
    - allow_egress_to_trading_apis
    - deny_all_other_traffic
```

### 3. Storage Requirements
```yaml
storage_requirements:
  database:
    type: "persistent_volume"
    size: "1TB"
    iops: "3000"
    backup_retention: "90 days"
    
  time_series:
    type: "persistent_volume"
    size: "500GB"
    iops: "1000"
    backup_retention: "30 days"
    
  logs:
    type: "persistent_volume"
    size: "200GB"
    iops: "500"
    backup_retention: "7 days"
    
  models:
    type: "persistent_volume"
    size: "100GB"
    iops: "1000"
    backup_retention: "365 days"
```

---

## 🔧 ENVIRONMENT SETUP

### 1. Kubernetes Cluster Setup

#### Prerequisites Installation
```bash
#!/bin/bash
# Kubernetes Cluster Setup Script
# Location: /home/QuantNova/GrandModel/scripts/setup_kubernetes.sh

echo "=== KUBERNETES CLUSTER SETUP ==="
echo "Environment: $1"
echo "Date: $(date)"
echo

# 1. Install required tools
echo "1. Installing required tools..."
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Install kustomize
curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash

# 2. Create namespace
echo "2. Creating namespace..."
kubectl create namespace grandmodel

# 3. Setup RBAC
echo "3. Setting up RBAC..."
kubectl apply -f - <<EOF
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: grandmodel-cluster-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "endpoints", "persistentvolumeclaims", "events", "configmaps", "secrets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets", "statefulsets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["networking.k8s.io"]
  resources: ["ingresses", "networkpolicies"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: grandmodel-cluster-role-binding
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: grandmodel-cluster-role
subjects:
- kind: ServiceAccount
  name: grandmodel-service-account
  namespace: grandmodel
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: grandmodel-service-account
  namespace: grandmodel
EOF

# 4. Install cert-manager
echo "4. Installing cert-manager..."
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml

# 5. Install ingress controller
echo "5. Installing ingress controller..."
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update
helm install ingress-nginx ingress-nginx/ingress-nginx --namespace ingress-nginx --create-namespace

# 6. Setup monitoring
echo "6. Setting up monitoring..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack --namespace monitoring --create-namespace

echo "=== KUBERNETES CLUSTER SETUP COMPLETE ==="
```

#### Cluster Validation
```bash
#!/bin/bash
# Cluster Validation Script
# Location: /home/QuantNova/GrandModel/scripts/validate_cluster.sh

echo "=== CLUSTER VALIDATION ==="
echo "Date: $(date)"
echo

# 1. Check cluster info
echo "1. Checking cluster info..."
kubectl cluster-info

# 2. Check nodes
echo "2. Checking nodes..."
kubectl get nodes -o wide

# 3. Check system pods
echo "3. Checking system pods..."
kubectl get pods --all-namespaces | grep -E "(kube-system|ingress-nginx|monitoring)"

# 4. Check persistent volumes
echo "4. Checking persistent volumes..."
kubectl get pv
kubectl get pvc --all-namespaces

# 5. Check network connectivity
echo "5. Checking network connectivity..."
kubectl run network-test --image=busybox --rm -it --restart=Never -- ping -c 3 google.com

# 6. Check resource quotas
echo "6. Checking resource quotas..."
kubectl describe quota --all-namespaces

# 7. Validate RBAC
echo "7. Validating RBAC..."
kubectl auth can-i create pods --as=system:serviceaccount:grandmodel:grandmodel-service-account -n grandmodel

echo "=== CLUSTER VALIDATION COMPLETE ==="
```

---

### 2. Database Setup

#### PostgreSQL Deployment
```yaml
# PostgreSQL StatefulSet
# Location: /home/QuantNova/GrandModel/k8s/database/postgres-statefulset.yaml

apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: grandmodel
spec:
  serviceName: postgres-service
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: grandmodel
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: username
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        - name: PGDATA
          value: /var/lib/postgresql/data/pgdata
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        - name: postgres-config
          mountPath: /etc/postgresql/postgresql.conf
          subPath: postgresql.conf
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - $(POSTGRES_USER)
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - $(POSTGRES_USER)
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: postgres-config
        configMap:
          name: postgres-config
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
      storageClassName: fast-ssd
```

#### Database Configuration
```sql
-- Database Initialization Script
-- Location: /home/QuantNova/GrandModel/database/init.sql

-- Create database
CREATE DATABASE grandmodel;

-- Create users
CREATE USER grandmodel_app WITH ENCRYPTED PASSWORD 'secure_password';
CREATE USER grandmodel_readonly WITH ENCRYPTED PASSWORD 'readonly_password';

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE grandmodel TO grandmodel_app;
GRANT CONNECT ON DATABASE grandmodel TO grandmodel_readonly;

-- Use grandmodel database
\c grandmodel;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Create tables
CREATE TABLE trading.trades (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(4) NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    agent VARCHAR(50) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    pnl DECIMAL(10,2),
    commission DECIMAL(10,2),
    slippage DECIMAL(6,4)
);

CREATE TABLE trading.positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    quantity INTEGER NOT NULL,
    average_price DECIMAL(10,2) NOT NULL,
    unrealized_pnl DECIMAL(10,2),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    agent VARCHAR(50) NOT NULL,
    strategy VARCHAR(50) NOT NULL
);

CREATE TABLE analytics.performance_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(50) NOT NULL,
    metric_value DECIMAL(10,6) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    agent VARCHAR(50),
    timeframe VARCHAR(10)
);

CREATE TABLE monitoring.system_health (
    id SERIAL PRIMARY KEY,
    component VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    cpu_usage DECIMAL(5,2),
    memory_usage DECIMAL(5,2),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes
CREATE INDEX idx_trades_symbol_timestamp ON trading.trades(symbol, timestamp);
CREATE INDEX idx_trades_agent ON trading.trades(agent);
CREATE INDEX idx_positions_symbol ON trading.positions(symbol);
CREATE INDEX idx_performance_metrics_timestamp ON analytics.performance_metrics(timestamp);
CREATE INDEX idx_system_health_component ON monitoring.system_health(component);

-- Grant schema permissions
GRANT ALL PRIVILEGES ON SCHEMA trading TO grandmodel_app;
GRANT ALL PRIVILEGES ON SCHEMA analytics TO grandmodel_app;
GRANT ALL PRIVILEGES ON SCHEMA monitoring TO grandmodel_app;

GRANT USAGE ON SCHEMA trading TO grandmodel_readonly;
GRANT USAGE ON SCHEMA analytics TO grandmodel_readonly;
GRANT USAGE ON SCHEMA monitoring TO grandmodel_readonly;

-- Grant table permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading TO grandmodel_app;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO grandmodel_app;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO grandmodel_app;

GRANT SELECT ON ALL TABLES IN SCHEMA trading TO grandmodel_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO grandmodel_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA monitoring TO grandmodel_readonly;

-- Grant sequence permissions
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA trading TO grandmodel_app;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA analytics TO grandmodel_app;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA monitoring TO grandmodel_app;
```

#### Redis Deployment
```yaml
# Redis Deployment
# Location: /home/QuantNova/GrandModel/k8s/database/redis-deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: grandmodel
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        command: ["redis-server"]
        args: ["/etc/redis/redis.conf"]
        volumeMounts:
        - name: redis-config
          mountPath: /etc/redis/redis.conf
          subPath: redis.conf
        - name: redis-data
          mountPath: /data
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "200m"
        livenessProbe:
          exec:
            command:
            - redis-cli
            - ping
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - redis-cli
            - ping
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: redis-config
        configMap:
          name: redis-config
      - name: redis-data
        persistentVolumeClaim:
          claimName: redis-data-pvc
```

---

### 3. Application Configuration

#### ConfigMap for Application Settings
```yaml
# Application Configuration
# Location: /home/QuantNova/GrandModel/k8s/config/application-config.yaml

apiVersion: v1
kind: ConfigMap
metadata:
  name: grandmodel-config
  namespace: grandmodel
data:
  application.yaml: |
    system:
      environment: production
      log_level: INFO
      debug_mode: false
      max_memory_gb: 16
      
    database:
      host: postgres-service.grandmodel.svc.cluster.local
      port: 5432
      name: grandmodel
      max_connections: 100
      connection_timeout: 30
      
    redis:
      host: redis-service.grandmodel.svc.cluster.local
      port: 6379
      db: 0
      max_connections: 50
      
    agents:
      strategic_marl:
        enabled: true
        model_path: /models/strategic_marl.pt
        confidence_threshold: 0.7
        max_position_size: 100
        update_frequency: 30
        
      tactical_marl:
        enabled: true
        model_path: /models/tactical_marl.pt
        confidence_threshold: 0.75
        max_position_size: 50
        update_frequency: 5
        
      risk_management:
        enabled: true
        max_var: 0.03
        max_drawdown: 0.05
        position_limit: 0.1
        
    data_handler:
      type: rithmic
      connection_timeout: 30
      retry_attempts: 3
      heartbeat_interval: 60
      
    monitoring:
      enabled: true
      prometheus_port: 9090
      metrics_interval: 15
      
    security:
      jwt_secret_key: ${JWT_SECRET_KEY}
      token_expiry: 3600
      rate_limit_per_minute: 1000
      
    trading:
      max_orders_per_minute: 100
      max_position_value: 1000000
      slippage_tolerance: 0.005
      commission_rate: 0.0001
```

#### Secrets Management
```yaml
# Secrets Configuration
# Location: /home/QuantNova/GrandModel/k8s/secrets/secrets.yaml

apiVersion: v1
kind: Secret
metadata:
  name: grandmodel-secrets
  namespace: grandmodel
type: Opaque
data:
  # Database credentials (base64 encoded)
  database_username: Z3JhbmRtb2RlbF9hcHA=
  database_password: c2VjdXJlX3Bhc3N3b3Jk
  
  # Trading API credentials
  rithmic_username: dHJhZGluZ191c2Vy
  rithmic_password: dHJhZGluZ19wYXNz
  
  # JWT secret key
  jwt_secret_key: c3VwZXJfc2VjdXJlX2p3dF9rZXk=
  
  # Monitoring credentials
  prometheus_username: cHJvbWV0aGV1cw==
  prometheus_password: bW9uaXRvcmluZw==
```

---

## 🚀 DEPLOYMENT PROCEDURES

### 1. Development Environment Deployment

#### Single-Node Development Setup
```bash
#!/bin/bash
# Development Environment Deployment Script
# Location: /home/QuantNova/GrandModel/scripts/deploy_development.sh

echo "=== DEVELOPMENT ENVIRONMENT DEPLOYMENT ==="
echo "Date: $(date)"
echo "Environment: development"
echo

# 1. Setup environment variables
export ENVIRONMENT=development
export NAMESPACE=grandmodel-dev
export IMAGE_TAG=latest

# 2. Create namespace
echo "1. Creating namespace..."
kubectl create namespace $NAMESPACE

# 3. Apply configurations
echo "2. Applying configurations..."
kubectl apply -f k8s/config/development/ -n $NAMESPACE

# 4. Deploy database
echo "3. Deploying database..."
kubectl apply -f k8s/database/postgres-deployment-dev.yaml -n $NAMESPACE
kubectl apply -f k8s/database/redis-deployment-dev.yaml -n $NAMESPACE

# 5. Wait for database to be ready
echo "4. Waiting for database to be ready..."
kubectl wait --for=condition=ready pod -l app=postgres -n $NAMESPACE --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n $NAMESPACE --timeout=300s

# 6. Initialize database
echo "5. Initializing database..."
kubectl exec -n $NAMESPACE deployment/postgres -- psql -U postgres -f /docker-entrypoint-initdb.d/init.sql

# 7. Deploy applications
echo "6. Deploying applications..."
kubectl apply -f k8s/deployments/development/ -n $NAMESPACE

# 8. Wait for applications to be ready
echo "7. Waiting for applications to be ready..."
kubectl wait --for=condition=ready pod -l app=strategic-marl -n $NAMESPACE --timeout=300s
kubectl wait --for=condition=ready pod -l app=tactical-marl -n $NAMESPACE --timeout=300s
kubectl wait --for=condition=ready pod -l app=api-gateway -n $NAMESPACE --timeout=300s

# 9. Setup port forwarding
echo "8. Setting up port forwarding..."
kubectl port-forward service/api-gateway 8080:8080 -n $NAMESPACE &
kubectl port-forward service/grafana 3000:3000 -n $NAMESPACE &

# 10. Verify deployment
echo "9. Verifying deployment..."
sleep 30
curl -f http://localhost:8080/health || echo "API Gateway not ready"

echo "=== DEVELOPMENT ENVIRONMENT DEPLOYMENT COMPLETE ==="
echo "API Gateway: http://localhost:8080"
echo "Grafana: http://localhost:3000"
```

#### Development Environment Validation
```bash
#!/bin/bash
# Development Environment Validation Script
# Location: /home/QuantNova/GrandModel/scripts/validate_development.sh

echo "=== DEVELOPMENT ENVIRONMENT VALIDATION ==="
echo "Date: $(date)"
echo

# 1. Check pod status
echo "1. Checking pod status..."
kubectl get pods -n grandmodel-dev

# 2. Check service endpoints
echo "2. Checking service endpoints..."
kubectl get services -n grandmodel-dev

# 3. Test API endpoints
echo "3. Testing API endpoints..."
curl -f http://localhost:8080/health || echo "Health check failed"
curl -f http://localhost:8080/api/v1/system/status || echo "System status failed"

# 4. Check logs for errors
echo "4. Checking logs for errors..."
kubectl logs -n grandmodel-dev deployment/strategic-marl --tail=50 | grep -i error
kubectl logs -n grandmodel-dev deployment/tactical-marl --tail=50 | grep -i error

# 5. Test database connectivity
echo "5. Testing database connectivity..."
kubectl exec -n grandmodel-dev deployment/postgres -- psql -U postgres -c "SELECT 1;"

# 6. Test Redis connectivity
echo "6. Testing Redis connectivity..."
kubectl exec -n grandmodel-dev deployment/redis -- redis-cli ping

echo "=== DEVELOPMENT ENVIRONMENT VALIDATION COMPLETE ==="
```

---

### 2. Staging Environment Deployment

#### Staging Deployment with High Availability
```bash
#!/bin/bash
# Staging Environment Deployment Script
# Location: /home/QuantNova/GrandModel/scripts/deploy_staging.sh

echo "=== STAGING ENVIRONMENT DEPLOYMENT ==="
echo "Date: $(date)"
echo "Environment: staging"
echo

# 1. Setup environment variables
export ENVIRONMENT=staging
export NAMESPACE=grandmodel-staging
export IMAGE_TAG=${1:-latest}

# 2. Create namespace
echo "1. Creating namespace..."
kubectl create namespace $NAMESPACE

# 3. Apply secrets
echo "2. Applying secrets..."
kubectl apply -f k8s/secrets/staging-secrets.yaml -n $NAMESPACE

# 4. Apply configurations
echo "3. Applying configurations..."
kubectl apply -f k8s/config/staging/ -n $NAMESPACE

# 5. Deploy persistent volumes
echo "4. Deploying persistent volumes..."
kubectl apply -f k8s/storage/staging-pv.yaml -n $NAMESPACE

# 6. Deploy database cluster
echo "5. Deploying database cluster..."
kubectl apply -f k8s/database/postgres-cluster-staging.yaml -n $NAMESPACE
kubectl apply -f k8s/database/redis-cluster-staging.yaml -n $NAMESPACE

# 7. Wait for database cluster to be ready
echo "6. Waiting for database cluster to be ready..."
kubectl wait --for=condition=ready pod -l app=postgres-cluster -n $NAMESPACE --timeout=600s
kubectl wait --for=condition=ready pod -l app=redis-cluster -n $NAMESPACE --timeout=600s

# 8. Initialize database
echo "7. Initializing database..."
kubectl exec -n $NAMESPACE deployment/postgres-cluster -- psql -U postgres -f /docker-entrypoint-initdb.d/init.sql

# 9. Deploy applications with multiple replicas
echo "8. Deploying applications..."
kubectl apply -f k8s/deployments/staging/ -n $NAMESPACE

# 10. Wait for applications to be ready
echo "9. Waiting for applications to be ready..."
kubectl wait --for=condition=ready pod -l app=strategic-marl -n $NAMESPACE --timeout=600s
kubectl wait --for=condition=ready pod -l app=tactical-marl -n $NAMESPACE --timeout=600s
kubectl wait --for=condition=ready pod -l app=api-gateway -n $NAMESPACE --timeout=600s

# 11. Setup ingress
echo "10. Setting up ingress..."
kubectl apply -f k8s/ingress/staging-ingress.yaml -n $NAMESPACE

# 12. Deploy monitoring
echo "11. Deploying monitoring..."
kubectl apply -f k8s/monitoring/staging-monitoring.yaml -n $NAMESPACE

# 13. Run smoke tests
echo "12. Running smoke tests..."
python scripts/smoke_tests.py --environment staging

echo "=== STAGING ENVIRONMENT DEPLOYMENT COMPLETE ==="
echo "API Gateway: https://staging-api.grandmodel.quantnova.com"
echo "Grafana: https://staging-grafana.grandmodel.quantnova.com"
```

---

### 3. Production Environment Deployment

#### Blue-Green Production Deployment
```bash
#!/bin/bash
# Production Blue-Green Deployment Script
# Location: /home/QuantNova/GrandModel/scripts/deploy_production_blue_green.sh

VERSION=$1
COLOR=${2:-blue}  # blue or green

echo "=== PRODUCTION BLUE-GREEN DEPLOYMENT ==="
echo "Version: $VERSION"
echo "Color: $COLOR"
echo "Date: $(date)"
echo

# 1. Validate inputs
if [ -z "$VERSION" ]; then
    echo "Error: Version is required"
    exit 1
fi

# 2. Setup environment variables
export ENVIRONMENT=production
export NAMESPACE=grandmodel
export IMAGE_TAG=$VERSION
export DEPLOYMENT_COLOR=$COLOR

# 3. Pre-deployment validation
echo "1. Running pre-deployment validation..."
python scripts/pre_deployment_validation.py --version $VERSION

# 4. Deploy to target color environment
echo "2. Deploying to $COLOR environment..."
kubectl apply -f k8s/deployments/production/${COLOR}/ -n $NAMESPACE

# 5. Update image tags
echo "3. Updating image tags..."
kubectl set image deployment/strategic-marl-${COLOR} strategic-marl=grandmodel/strategic-marl:$VERSION -n $NAMESPACE
kubectl set image deployment/tactical-marl-${COLOR} tactical-marl=grandmodel/tactical-marl:$VERSION -n $NAMESPACE
kubectl set image deployment/risk-management-${COLOR} risk-management=grandmodel/risk-management:$VERSION -n $NAMESPACE

# 6. Wait for rollout to complete
echo "4. Waiting for rollout to complete..."
kubectl rollout status deployment/strategic-marl-${COLOR} -n $NAMESPACE --timeout=600s
kubectl rollout status deployment/tactical-marl-${COLOR} -n $NAMESPACE --timeout=600s
kubectl rollout status deployment/risk-management-${COLOR} -n $NAMESPACE --timeout=600s

# 7. Run health checks
echo "5. Running health checks..."
python scripts/health_check.py --environment production --color $COLOR

# 8. Run smoke tests
echo "6. Running smoke tests..."
python scripts/smoke_tests.py --environment production --color $COLOR

# 9. Load testing
echo "7. Running load tests..."
python scripts/load_test.py --environment production --color $COLOR --duration 300

# 10. Switch traffic if tests pass
echo "8. Switching traffic to $COLOR environment..."
kubectl patch service/strategic-marl-service -n $NAMESPACE -p "{\"spec\":{\"selector\":{\"deployment\":\"$COLOR\"}}}"
kubectl patch service/tactical-marl-service -n $NAMESPACE -p "{\"spec\":{\"selector\":{\"deployment\":\"$COLOR\"}}}"
kubectl patch service/risk-management-service -n $NAMESPACE -p "{\"spec\":{\"selector\":{\"deployment\":\"$COLOR\"}}}"

# 11. Monitor new deployment
echo "9. Monitoring new deployment..."
python scripts/monitor_deployment.py --environment production --color $COLOR --duration 1800

# 12. Cleanup old deployment
echo "10. Cleaning up old deployment..."
OLD_COLOR=$([ "$COLOR" == "blue" ] && echo "green" || echo "blue")
kubectl delete deployment/strategic-marl-${OLD_COLOR} -n $NAMESPACE
kubectl delete deployment/tactical-marl-${OLD_COLOR} -n $NAMESPACE
kubectl delete deployment/risk-management-${OLD_COLOR} -n $NAMESPACE

# 13. Update deployment records
echo "11. Updating deployment records..."
python scripts/update_deployment_records.py --version $VERSION --color $COLOR

echo "=== PRODUCTION BLUE-GREEN DEPLOYMENT COMPLETE ==="
echo "Version $VERSION deployed to production"
echo "Active color: $COLOR"
```

#### Production Rollback Procedure
```bash
#!/bin/bash
# Production Rollback Script
# Location: /home/QuantNova/GrandModel/scripts/rollback_production.sh

ROLLBACK_VERSION=$1
REASON=${2:-"emergency_rollback"}

echo "=== PRODUCTION ROLLBACK ==="
echo "Rollback Version: $ROLLBACK_VERSION"
echo "Reason: $REASON"
echo "Date: $(date)"
echo

# 1. Validate rollback version
if [ -z "$ROLLBACK_VERSION" ]; then
    echo "Error: Rollback version is required"
    exit 1
fi

# 2. Create incident ticket
echo "1. Creating incident ticket..."
python scripts/create_incident_ticket.py --type rollback --version $ROLLBACK_VERSION --reason "$REASON"

# 3. Notify stakeholders
echo "2. Notifying stakeholders..."
python scripts/notify_rollback.py --version $ROLLBACK_VERSION --reason "$REASON"

# 4. Stop all trading
echo "3. Stopping all trading..."
kubectl scale deployment/strategic-marl --replicas=0 -n grandmodel
kubectl scale deployment/tactical-marl --replicas=0 -n grandmodel

# 5. Rollback images
echo "4. Rolling back images..."
kubectl set image deployment/strategic-marl strategic-marl=grandmodel/strategic-marl:$ROLLBACK_VERSION -n grandmodel
kubectl set image deployment/tactical-marl tactical-marl=grandmodel/tactical-marl:$ROLLBACK_VERSION -n grandmodel
kubectl set image deployment/risk-management risk-management=grandmodel/risk-management:$ROLLBACK_VERSION -n grandmodel

# 6. Scale back up
echo "5. Scaling back up..."
kubectl scale deployment/strategic-marl --replicas=2 -n grandmodel
kubectl scale deployment/tactical-marl --replicas=3 -n grandmodel

# 7. Wait for rollout
echo "6. Waiting for rollout..."
kubectl rollout status deployment/strategic-marl -n grandmodel --timeout=300s
kubectl rollout status deployment/tactical-marl -n grandmodel --timeout=300s

# 8. Verify rollback
echo "7. Verifying rollback..."
python scripts/verify_rollback.py --version $ROLLBACK_VERSION

# 9. Resume trading
echo "8. Resuming trading..."
python scripts/resume_trading.py

# 10. Monitor system
echo "9. Monitoring system..."
python scripts/monitor_rollback.py --duration 1800

# 11. Update records
echo "10. Updating records..."
python scripts/update_rollback_records.py --version $ROLLBACK_VERSION --reason "$REASON"

echo "=== PRODUCTION ROLLBACK COMPLETE ==="
echo "Rolled back to version: $ROLLBACK_VERSION"
```

---

## 🔧 CONFIGURATION MANAGEMENT

### 1. Environment-Specific Configuration

#### Configuration Templates
```yaml
# Base Configuration Template
# Location: /home/QuantNova/GrandModel/config/templates/base.yaml

system:
  environment: {{ ENVIRONMENT }}
  log_level: {{ LOG_LEVEL | default('INFO') }}
  debug_mode: {{ DEBUG_MODE | default(false) }}
  max_memory_gb: {{ MAX_MEMORY_GB | default(16) }}

database:
  host: {{ DATABASE_HOST }}
  port: {{ DATABASE_PORT | default(5432) }}
  name: {{ DATABASE_NAME }}
  username: {{ DATABASE_USERNAME }}
  password: {{ DATABASE_PASSWORD }}
  max_connections: {{ DATABASE_MAX_CONNECTIONS | default(100) }}
  connection_timeout: {{ DATABASE_CONNECTION_TIMEOUT | default(30) }}

redis:
  host: {{ REDIS_HOST }}
  port: {{ REDIS_PORT | default(6379) }}
  db: {{ REDIS_DB | default(0) }}
  password: {{ REDIS_PASSWORD }}
  max_connections: {{ REDIS_MAX_CONNECTIONS | default(50) }}

agents:
  strategic_marl:
    enabled: {{ STRATEGIC_MARL_ENABLED | default(true) }}
    model_path: {{ STRATEGIC_MARL_MODEL_PATH }}
    confidence_threshold: {{ STRATEGIC_MARL_CONFIDENCE_THRESHOLD | default(0.7) }}
    max_position_size: {{ STRATEGIC_MARL_MAX_POSITION_SIZE | default(100) }}
    update_frequency: {{ STRATEGIC_MARL_UPDATE_FREQUENCY | default(30) }}
    
  tactical_marl:
    enabled: {{ TACTICAL_MARL_ENABLED | default(true) }}
    model_path: {{ TACTICAL_MARL_MODEL_PATH }}
    confidence_threshold: {{ TACTICAL_MARL_CONFIDENCE_THRESHOLD | default(0.75) }}
    max_position_size: {{ TACTICAL_MARL_MAX_POSITION_SIZE | default(50) }}
    update_frequency: {{ TACTICAL_MARL_UPDATE_FREQUENCY | default(5) }}
    
  risk_management:
    enabled: {{ RISK_MANAGEMENT_ENABLED | default(true) }}
    max_var: {{ RISK_MANAGEMENT_MAX_VAR | default(0.03) }}
    max_drawdown: {{ RISK_MANAGEMENT_MAX_DRAWDOWN | default(0.05) }}
    position_limit: {{ RISK_MANAGEMENT_POSITION_LIMIT | default(0.1) }}

monitoring:
  enabled: {{ MONITORING_ENABLED | default(true) }}
  prometheus_port: {{ MONITORING_PROMETHEUS_PORT | default(9090) }}
  metrics_interval: {{ MONITORING_METRICS_INTERVAL | default(15) }}
  
security:
  jwt_secret_key: {{ JWT_SECRET_KEY }}
  token_expiry: {{ JWT_TOKEN_EXPIRY | default(3600) }}
  rate_limit_per_minute: {{ SECURITY_RATE_LIMIT_PER_MINUTE | default(1000) }}
```

#### Environment Variable Files
```bash
# Development Environment Variables
# Location: /home/QuantNova/GrandModel/config/environments/development.env

ENVIRONMENT=development
LOG_LEVEL=DEBUG
DEBUG_MODE=true
MAX_MEMORY_GB=8

DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=grandmodel_dev
DATABASE_USERNAME=dev_user
DATABASE_PASSWORD=dev_password
DATABASE_MAX_CONNECTIONS=50

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
REDIS_MAX_CONNECTIONS=25

STRATEGIC_MARL_ENABLED=true
STRATEGIC_MARL_MODEL_PATH=/models/strategic_marl_dev.pt
STRATEGIC_MARL_CONFIDENCE_THRESHOLD=0.6
STRATEGIC_MARL_MAX_POSITION_SIZE=10
STRATEGIC_MARL_UPDATE_FREQUENCY=30

TACTICAL_MARL_ENABLED=true
TACTICAL_MARL_MODEL_PATH=/models/tactical_marl_dev.pt
TACTICAL_MARL_CONFIDENCE_THRESHOLD=0.65
TACTICAL_MARL_MAX_POSITION_SIZE=5
TACTICAL_MARL_UPDATE_FREQUENCY=5

RISK_MANAGEMENT_ENABLED=true
RISK_MANAGEMENT_MAX_VAR=0.01
RISK_MANAGEMENT_MAX_DRAWDOWN=0.02
RISK_MANAGEMENT_POSITION_LIMIT=0.05

MONITORING_ENABLED=true
MONITORING_PROMETHEUS_PORT=9090
MONITORING_METRICS_INTERVAL=15

JWT_SECRET_KEY=dev_secret_key
JWT_TOKEN_EXPIRY=3600
SECURITY_RATE_LIMIT_PER_MINUTE=100
```

```bash
# Production Environment Variables
# Location: /home/QuantNova/GrandModel/config/environments/production.env

ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG_MODE=false
MAX_MEMORY_GB=32

DATABASE_HOST=postgres-cluster.grandmodel.svc.cluster.local
DATABASE_PORT=5432
DATABASE_NAME=grandmodel
DATABASE_USERNAME=grandmodel_app
DATABASE_PASSWORD=${DATABASE_PASSWORD_SECRET}
DATABASE_MAX_CONNECTIONS=200

REDIS_HOST=redis-cluster.grandmodel.svc.cluster.local
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=${REDIS_PASSWORD_SECRET}
REDIS_MAX_CONNECTIONS=100

STRATEGIC_MARL_ENABLED=true
STRATEGIC_MARL_MODEL_PATH=/models/strategic_marl_v1.2.3.pt
STRATEGIC_MARL_CONFIDENCE_THRESHOLD=0.7
STRATEGIC_MARL_MAX_POSITION_SIZE=100
STRATEGIC_MARL_UPDATE_FREQUENCY=30

TACTICAL_MARL_ENABLED=true
TACTICAL_MARL_MODEL_PATH=/models/tactical_marl_v1.2.3.pt
TACTICAL_MARL_CONFIDENCE_THRESHOLD=0.75
TACTICAL_MARL_MAX_POSITION_SIZE=50
TACTICAL_MARL_UPDATE_FREQUENCY=5

RISK_MANAGEMENT_ENABLED=true
RISK_MANAGEMENT_MAX_VAR=0.03
RISK_MANAGEMENT_MAX_DRAWDOWN=0.05
RISK_MANAGEMENT_POSITION_LIMIT=0.1

MONITORING_ENABLED=true
MONITORING_PROMETHEUS_PORT=9090
MONITORING_METRICS_INTERVAL=15

JWT_SECRET_KEY=${JWT_SECRET_KEY_SECRET}
JWT_TOKEN_EXPIRY=3600
SECURITY_RATE_LIMIT_PER_MINUTE=1000
```

### 2. Configuration Validation

#### Configuration Validator Script
```python
#!/usr/bin/env python3
# Configuration Validator
# Location: /home/QuantNova/GrandModel/scripts/validate_config.py

import yaml
import os
import sys
from typing import Dict, Any, List
from jsonschema import validate, ValidationError

class ConfigValidator:
    def __init__(self):
        self.schema = self.load_schema()
        
    def load_schema(self) -> Dict[str, Any]:
        """Load configuration schema"""
        schema_path = "config/schema/config_schema.yaml"
        with open(schema_path, 'r') as f:
            return yaml.safe_load(f)
    
    def validate_config(self, config_path: str) -> bool:
        """Validate configuration file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate against schema
            validate(config, self.schema)
            
            # Custom validation rules
            self.validate_custom_rules(config)
            
            print(f"✅ Configuration {config_path} is valid")
            return True
            
        except ValidationError as e:
            print(f"❌ Configuration validation failed: {e.message}")
            return False
        except Exception as e:
            print(f"❌ Configuration validation error: {e}")
            return False
    
    def validate_custom_rules(self, config: Dict[str, Any]) -> None:
        """Validate custom business rules"""
        
        # Rule 1: Confidence thresholds must be between 0 and 1
        for agent_name, agent_config in config.get('agents', {}).items():
            threshold = agent_config.get('confidence_threshold', 0.5)
            if not 0 <= threshold <= 1:
                raise ValueError(f"Invalid confidence threshold for {agent_name}: {threshold}")
        
        # Rule 2: Risk limits must be positive
        risk_config = config.get('agents', {}).get('risk_management', {})
        for limit_name, limit_value in risk_config.items():
            if limit_name.startswith('max_') and limit_value <= 0:
                raise ValueError(f"Invalid risk limit {limit_name}: {limit_value}")
        
        # Rule 3: Database connections must be reasonable
        max_connections = config.get('database', {}).get('max_connections', 100)
        if max_connections < 10 or max_connections > 1000:
            raise ValueError(f"Invalid database max_connections: {max_connections}")
        
        # Rule 4: Memory allocation must be reasonable
        max_memory = config.get('system', {}).get('max_memory_gb', 16)
        if max_memory < 4 or max_memory > 128:
            raise ValueError(f"Invalid max_memory_gb: {max_memory}")

    def validate_all_environments(self) -> bool:
        """Validate all environment configurations"""
        environments = ['development', 'staging', 'production']
        all_valid = True
        
        for env in environments:
            config_path = f"config/environments/{env}.yaml"
            if os.path.exists(config_path):
                if not self.validate_config(config_path):
                    all_valid = False
            else:
                print(f"⚠️  Configuration file not found: {config_path}")
                all_valid = False
        
        return all_valid

def main():
    validator = ConfigValidator()
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        if validator.validate_config(config_path):
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        if validator.validate_all_environments():
            print("✅ All configurations are valid")
            sys.exit(0)
        else:
            print("❌ Some configurations are invalid")
            sys.exit(1)

if __name__ == "__main__":
    main()
```

---

### 3. Configuration Deployment

#### Configuration Deployment Script
```bash
#!/bin/bash
# Configuration Deployment Script
# Location: /home/QuantNova/GrandModel/scripts/deploy_config.sh

ENVIRONMENT=$1
CONFIG_VERSION=${2:-latest}

echo "=== CONFIGURATION DEPLOYMENT ==="
echo "Environment: $ENVIRONMENT"
echo "Config Version: $CONFIG_VERSION"
echo "Date: $(date)"
echo

# 1. Validate environment
if [ -z "$ENVIRONMENT" ]; then
    echo "Error: Environment is required"
    exit 1
fi

# 2. Validate configuration
echo "1. Validating configuration..."
python scripts/validate_config.py config/environments/${ENVIRONMENT}.yaml

# 3. Backup current configuration
echo "2. Backing up current configuration..."
kubectl get configmap grandmodel-config -n grandmodel -o yaml > backups/config_backup_$(date +%Y%m%d_%H%M%S).yaml

# 4. Generate configuration from template
echo "3. Generating configuration from template..."
python scripts/generate_config.py \
    --template config/templates/base.yaml \
    --environment $ENVIRONMENT \
    --output config/generated/${ENVIRONMENT}.yaml

# 5. Create ConfigMap
echo "4. Creating ConfigMap..."
kubectl create configmap grandmodel-config \
    --from-file=config/generated/${ENVIRONMENT}.yaml \
    --dry-run=client -o yaml | kubectl apply -f -

# 6. Create Secrets
echo "5. Creating Secrets..."
kubectl apply -f k8s/secrets/${ENVIRONMENT}-secrets.yaml

# 7. Restart applications to pick up new config
echo "6. Restarting applications..."
kubectl rollout restart deployment/strategic-marl -n grandmodel
kubectl rollout restart deployment/tactical-marl -n grandmodel
kubectl rollout restart deployment/risk-management -n grandmodel
kubectl rollout restart deployment/api-gateway -n grandmodel

# 8. Wait for rollout to complete
echo "7. Waiting for rollout to complete..."
kubectl rollout status deployment/strategic-marl -n grandmodel --timeout=300s
kubectl rollout status deployment/tactical-marl -n grandmodel --timeout=300s
kubectl rollout status deployment/risk-management -n grandmodel --timeout=300s
kubectl rollout status deployment/api-gateway -n grandmodel --timeout=300s

# 9. Verify configuration
echo "8. Verifying configuration..."
python scripts/verify_config.py --environment $ENVIRONMENT

# 10. Update configuration version
echo "9. Updating configuration version..."
kubectl label configmap grandmodel-config version=$CONFIG_VERSION -n grandmodel

echo "=== CONFIGURATION DEPLOYMENT COMPLETE ==="
```

---

## 🔍 MONITORING AND VALIDATION

### 1. Deployment Monitoring

#### Deployment Health Check Script
```python
#!/usr/bin/env python3
# Deployment Health Check
# Location: /home/QuantNova/GrandModel/scripts/deployment_health_check.py

import requests
import time
import sys
from typing import Dict, Any, List

class DeploymentHealthChecker:
    def __init__(self, environment: str):
        self.environment = environment
        self.base_url = self.get_base_url()
        self.health_endpoints = [
            "/health",
            "/api/v1/system/status",
            "/api/v1/agents/status",
            "/metrics"
        ]
    
    def get_base_url(self) -> str:
        """Get base URL for environment"""
        urls = {
            'development': 'http://localhost:8080',
            'staging': 'https://staging-api.grandmodel.quantnova.com',
            'production': 'https://api.grandmodel.quantnova.com'
        }
        return urls.get(self.environment, 'http://localhost:8080')
    
    def check_endpoint(self, endpoint: str) -> Dict[str, Any]:
        """Check individual endpoint health"""
        try:
            response = requests.get(f"{self.base_url}{endpoint}", timeout=30)
            return {
                'endpoint': endpoint,
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'error': None
            }
        except Exception as e:
            return {
                'endpoint': endpoint,
                'status': 'unhealthy',
                'status_code': None,
                'response_time': None,
                'error': str(e)
            }
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health"""
        results = []
        
        for endpoint in self.health_endpoints:
            result = self.check_endpoint(endpoint)
            results.append(result)
            
            if result['status'] == 'healthy':
                print(f"✅ {endpoint}: {result['response_time']:.3f}s")
            else:
                print(f"❌ {endpoint}: {result['error']}")
        
        healthy_count = sum(1 for r in results if r['status'] == 'healthy')
        overall_health = 'healthy' if healthy_count == len(results) else 'unhealthy'
        
        return {
            'environment': self.environment,
            'overall_health': overall_health,
            'healthy_endpoints': healthy_count,
            'total_endpoints': len(results),
            'results': results
        }
    
    def continuous_monitoring(self, duration_minutes: int = 30) -> None:
        """Continuously monitor deployment health"""
        end_time = time.time() + (duration_minutes * 60)
        
        while time.time() < end_time:
            health_status = self.check_system_health()
            
            if health_status['overall_health'] == 'healthy':
                print(f"✅ System healthy at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print(f"❌ System unhealthy at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                
            time.sleep(60)  # Check every minute

def main():
    if len(sys.argv) < 2:
        print("Usage: python deployment_health_check.py <environment> [duration_minutes]")
        sys.exit(1)
    
    environment = sys.argv[1]
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    checker = DeploymentHealthChecker(environment)
    
    if len(sys.argv) > 2:
        checker.continuous_monitoring(duration)
    else:
        health_status = checker.check_system_health()
        
        if health_status['overall_health'] == 'healthy':
            print("\n✅ Deployment health check passed")
            sys.exit(0)
        else:
            print("\n❌ Deployment health check failed")
            sys.exit(1)

if __name__ == "__main__":
    main()
```

### 2. Performance Validation

#### Load Testing Script
```python
#!/usr/bin/env python3
# Load Testing Script
# Location: /home/QuantNova/GrandModel/scripts/load_test.py

import asyncio
import aiohttp
import time
import json
import sys
from typing import Dict, Any, List
from statistics import mean, median

class LoadTester:
    def __init__(self, base_url: str, concurrent_users: int = 100, duration: int = 300):
        self.base_url = base_url
        self.concurrent_users = concurrent_users
        self.duration = duration
        self.results = []
    
    async def make_request(self, session: aiohttp.ClientSession, endpoint: str) -> Dict[str, Any]:
        """Make individual request and measure performance"""
        start_time = time.time()
        
        try:
            async with session.get(f"{self.base_url}{endpoint}") as response:
                response_time = time.time() - start_time
                return {
                    'endpoint': endpoint,
                    'status_code': response.status,
                    'response_time': response_time,
                    'success': 200 <= response.status < 300,
                    'error': None
                }
        except Exception as e:
            response_time = time.time() - start_time
            return {
                'endpoint': endpoint,
                'status_code': None,
                'response_time': response_time,
                'success': False,
                'error': str(e)
            }
    
    async def user_session(self, user_id: int) -> List[Dict[str, Any]]:
        """Simulate user session"""
        endpoints = [
            '/api/v1/system/status',
            '/api/v1/agents/status',
            '/api/v1/performance/metrics',
            '/api/v1/market/quotes/NQ'
        ]
        
        results = []
        async with aiohttp.ClientSession() as session:
            end_time = time.time() + self.duration
            
            while time.time() < end_time:
                for endpoint in endpoints:
                    result = await self.make_request(session, endpoint)
                    result['user_id'] = user_id
                    result['timestamp'] = time.time()
                    results.append(result)
                
                await asyncio.sleep(1)  # 1 second between requests
        
        return results
    
    async def run_load_test(self) -> Dict[str, Any]:
        """Run load test with multiple concurrent users"""
        print(f"Starting load test with {self.concurrent_users} users for {self.duration} seconds")
        
        tasks = []
        for user_id in range(self.concurrent_users):
            task = asyncio.create_task(self.user_session(user_id))
            tasks.append(task)
        
        all_results = await asyncio.gather(*tasks)
        
        # Flatten results
        self.results = []
        for user_results in all_results:
            self.results.extend(user_results)
        
        return self.analyze_results()
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze load test results"""
        if not self.results:
            return {'error': 'No results to analyze'}
        
        # Calculate metrics
        response_times = [r['response_time'] for r in self.results]
        success_count = sum(1 for r in self.results if r['success'])
        total_requests = len(self.results)
        
        # Group by endpoint
        endpoint_metrics = {}
        for result in self.results:
            endpoint = result['endpoint']
            if endpoint not in endpoint_metrics:
                endpoint_metrics[endpoint] = {
                    'requests': [],
                    'response_times': [],
                    'success_count': 0
                }
            
            endpoint_metrics[endpoint]['requests'].append(result)
            endpoint_metrics[endpoint]['response_times'].append(result['response_time'])
            if result['success']:
                endpoint_metrics[endpoint]['success_count'] += 1
        
        # Calculate per-endpoint metrics
        for endpoint, metrics in endpoint_metrics.items():
            total_endpoint_requests = len(metrics['requests'])
            metrics['success_rate'] = metrics['success_count'] / total_endpoint_requests
            metrics['avg_response_time'] = mean(metrics['response_times'])
            metrics['median_response_time'] = median(metrics['response_times'])
            metrics['p95_response_time'] = sorted(metrics['response_times'])[int(0.95 * len(metrics['response_times']))]
        
        return {
            'total_requests': total_requests,
            'successful_requests': success_count,
            'success_rate': success_count / total_requests,
            'avg_response_time': mean(response_times),
            'median_response_time': median(response_times),
            'p95_response_time': sorted(response_times)[int(0.95 * len(response_times))],
            'requests_per_second': total_requests / self.duration,
            'endpoint_metrics': endpoint_metrics
        }

async def main():
    if len(sys.argv) < 2:
        print("Usage: python load_test.py <base_url> [concurrent_users] [duration]")
        sys.exit(1)
    
    base_url = sys.argv[1]
    concurrent_users = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    duration = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    
    tester = LoadTester(base_url, concurrent_users, duration)
    results = await tester.run_load_test()
    
    print("\n=== LOAD TEST RESULTS ===")
    print(f"Total Requests: {results['total_requests']}")
    print(f"Success Rate: {results['success_rate']:.2%}")
    print(f"Average Response Time: {results['avg_response_time']:.3f}s")
    print(f"95th Percentile Response Time: {results['p95_response_time']:.3f}s")
    print(f"Requests per Second: {results['requests_per_second']:.2f}")
    
    # Save detailed results
    with open(f"load_test_results_{int(time.time())}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Determine if test passed
    if (results['success_rate'] >= 0.95 and 
        results['p95_response_time'] <= 1.0 and 
        results['requests_per_second'] >= 50):
        print("\n✅ Load test passed")
        sys.exit(0)
    else:
        print("\n❌ Load test failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🎯 CONCLUSION

This comprehensive deployment and configuration documentation provides everything needed to successfully deploy and manage the GrandModel system across all environments. The procedures are designed to ensure:

- **Consistency**: Standardized deployment procedures across environments
- **Reliability**: Robust validation and monitoring processes
- **Scalability**: Support for different deployment scales
- **Security**: Secure configuration management
- **Maintainability**: Clear procedures for updates and rollbacks

Regular updates and improvements ensure these procedures remain current with system evolution and operational requirements.

---

**Document Version**: 1.0  
**Last Updated**: July 17, 2025  
**Next Review**: July 24, 2025  
**Owner**: Documentation & Training Agent (Agent 9)  
**Classification**: DEPLOYMENT CRITICAL  

---

*This document serves as the definitive deployment guide for the GrandModel system, providing essential procedures for all deployment scenarios.*