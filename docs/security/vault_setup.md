# HashiCorp Vault Setup and Secret Management

## Overview

This document provides comprehensive instructions for setting up and using HashiCorp Vault for secure secret management in the GrandModel trading system. As part of **Agent 3's mission to eliminate hardcoded secrets**, all sensitive data is now managed through Vault.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Vault Installation](#vault-installation)
3. [Vault Configuration](#vault-configuration)
4. [Secret Organization](#secret-organization)
5. [Authentication Methods](#authentication-methods)
6. [Development Setup](#development-setup)
7. [Production Deployment](#production-deployment)
8. [Secret Management](#secret-management)
9. [Troubleshooting](#troubleshooting)
10. [Security Best Practices](#security-best-practices)

## Prerequisites

- Docker and Docker Compose (for containerized deployment)
- Kubernetes cluster (for production deployment)
- Network access to Vault server
- Administrative privileges for initial setup

## Vault Installation

### Option 1: Docker Compose (Development)

```yaml
# docker-compose.vault.yml
version: '3.8'
services:
  vault:
    image: vault:1.15.2
    container_name: grandmodel-vault
    ports:
      - "8200:8200"
    environment:
      VAULT_DEV_ROOT_TOKEN_ID: "root-token"
      VAULT_DEV_LISTEN_ADDRESS: "0.0.0.0:8200"
    cap_add:
      - IPC_LOCK
    volumes:
      - vault-data:/vault/data
      - ./vault/config:/vault/config
    command: ["vault", "server", "-dev"]
    
  vault-ui:
    image: djenriquez/vault-ui:latest
    container_name: grandmodel-vault-ui
    ports:
      - "8000:8000"
    environment:
      VAULT_URL_DEFAULT: "http://vault:8200"
    depends_on:
      - vault

volumes:
  vault-data:
```

### Option 2: Kubernetes (Production)

```yaml
# k8s/vault-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vault
  namespace: grandmodel
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vault
  template:
    metadata:
      labels:
        app: vault
    spec:
      containers:
      - name: vault
        image: vault:1.15.2
        ports:
        - containerPort: 8200
        env:
        - name: VAULT_ADDR
          value: "http://0.0.0.0:8200"
        - name: VAULT_API_ADDR
          value: "http://0.0.0.0:8200"
        volumeMounts:
        - name: vault-config
          mountPath: /vault/config
        - name: vault-data
          mountPath: /vault/data
        securityContext:
          capabilities:
            add:
              - IPC_LOCK
      volumes:
      - name: vault-config
        configMap:
          name: vault-config
      - name: vault-data
        persistentVolumeClaim:
          claimName: vault-data-pvc
```

### Option 3: Binary Installation

```bash
# Download and install Vault
wget https://releases.hashicorp.com/vault/1.15.2/vault_1.15.2_linux_amd64.zip
unzip vault_1.15.2_linux_amd64.zip
sudo mv vault /usr/local/bin/
sudo chmod +x /usr/local/bin/vault

# Verify installation
vault version
```

## Vault Configuration

### Server Configuration

Create `/etc/vault/config.hcl`:

```hcl
# Vault server configuration
storage "file" {
  path = "/vault/data"
}

# Production: Use Consul or other HA storage
# storage "consul" {
#   address = "127.0.0.1:8500"
#   path    = "vault/"
# }

listener "tcp" {
  address     = "0.0.0.0:8200"
  tls_disable = 1  # Enable TLS in production
}

# Production TLS configuration
# listener "tcp" {
#   address       = "0.0.0.0:8200"
#   tls_cert_file = "/etc/vault/vault.crt"
#   tls_key_file  = "/etc/vault/vault.key"
# }

api_addr = "http://0.0.0.0:8200"
cluster_addr = "https://0.0.0.0:8201"
ui = true

# Seal configuration (use auto-unseal in production)
# seal "awskms" {
#   region     = "us-east-1"
#   kms_key_id = "alias/vault-seal"
# }
```

### Initialize Vault

```bash
# Start Vault server
vault server -config=/etc/vault/config.hcl

# Initialize Vault (run once)
vault operator init

# Unseal Vault (required after restart)
vault operator unseal <unseal_key_1>
vault operator unseal <unseal_key_2>
vault operator unseal <unseal_key_3>

# Authenticate as root
vault auth <initial_root_token>
```

## Secret Organization

### Secret Hierarchy

```
secret/
├── app/
│   ├── jwt/
│   │   └── secret_key
│   ├── database/
│   │   └── primary/
│   │       ├── host
│   │       ├── port
│   │       ├── database
│   │       ├── username
│   │       └── password
│   └── encryption/
│       └── default/
│           └── key
├── users/
│   ├── admin/
│   │   └── password
│   ├── risk_manager/
│   │   └── password
│   ├── operator/
│   │   └── password
│   └── compliance/
│       └── password
├── api_keys/
│   ├── openai/
│   │   └── key
│   ├── anthropic/
│   │   └── key
│   └── bloomberg/
│       └── key
├── oauth/
│   ├── google/
│   │   ├── client_id
│   │   └── client_secret
│   └── github/
│       ├── client_id
│       └── client_secret
└── infrastructure/
    ├── redis/
    │   └── password
    ├── elasticsearch/
    │   └── password
    └── prometheus/
        └── api_key
```

### Enable KV v2 Engine

```bash
# Enable KV v2 secrets engine
vault secrets enable -version=2 kv

# Or if using different mount point
vault secrets enable -path=grandmodel-secrets -version=2 kv
```

## Authentication Methods

### 1. Token Authentication (Development)

```bash
# Create a token for development
vault token create -policy=grandmodel-policy -period=24h
```

### 2. AppRole Authentication (Production)

```bash
# Enable AppRole auth method
vault auth enable approle

# Create role
vault write auth/approle/role/grandmodel-app \
    token_policies="grandmodel-policy" \
    token_ttl=1h \
    token_max_ttl=4h

# Get role ID
vault read auth/approle/role/grandmodel-app/role-id

# Generate secret ID
vault write -f auth/approle/role/grandmodel-app/secret-id
```

### 3. Kubernetes Authentication (K8s Deployment)

```bash
# Enable Kubernetes auth method
vault auth enable kubernetes

# Configure Kubernetes auth
vault write auth/kubernetes/config \
    token_reviewer_jwt="$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)" \
    kubernetes_host="https://kubernetes.default.svc.cluster.local" \
    kubernetes_ca_cert=@/var/run/secrets/kubernetes.io/serviceaccount/ca.crt

# Create role
vault write auth/kubernetes/role/grandmodel-app \
    bound_service_account_names=grandmodel-app \
    bound_service_account_namespaces=grandmodel \
    policies=grandmodel-policy \
    ttl=1h
```

### Policy Configuration

Create `grandmodel-policy.hcl`:

```hcl
# GrandModel application policy
path "secret/data/app/*" {
  capabilities = ["read"]
}

path "secret/data/users/*" {
  capabilities = ["read"]
}

path "secret/data/api_keys/*" {
  capabilities = ["read"]
}

path "secret/data/oauth/*" {
  capabilities = ["read"]
}

path "secret/data/infrastructure/*" {
  capabilities = ["read"]
}

# Allow token renewal
path "auth/token/renew-self" {
  capabilities = ["update"]
}

# Allow token lookup
path "auth/token/lookup-self" {
  capabilities = ["read"]
}
```

Apply the policy:

```bash
vault policy write grandmodel-policy grandmodel-policy.hcl
```

## Development Setup

### 1. Environment Configuration

Create `.env.vault`:

```bash
# Copy from example
cp .env.vault.example .env.vault

# Edit configuration
VAULT_URL=http://localhost:8200
VAULT_TOKEN=your_dev_token_here
VAULT_MOUNT_POINT=secret
```

### 2. Development Secrets File

Create `.dev_secrets.json`:

```bash
# Copy from example
cp .dev_secrets.json.example .dev_secrets.json

# Edit with development secrets
{
  "jwt_secret_key": "dev_jwt_secret_64_chars_minimum_for_security_compliance",
  "admin_password": "dev_admin123!",
  "database_password": "dev_db_password"
}
```

### 3. Populate Development Secrets

```bash
#!/bin/bash
# scripts/populate_dev_secrets.sh

# JWT secret
vault kv put secret/app/jwt \
    secret_key="dev_jwt_secret_64_chars_minimum_for_security_compliance"

# User passwords
vault kv put secret/users/admin \
    password="dev_admin123!"

vault kv put secret/users/risk_manager \
    password="dev_risk123!"

vault kv put secret/users/operator \
    password="dev_operator123!"

vault kv put secret/users/compliance \
    password="dev_compliance123!"

# Database credentials
vault kv put secret/app/database/primary \
    host="localhost" \
    port="5432" \
    database="grandmodel_dev" \
    username="grandmodel" \
    password="dev_db_password"

# API keys
vault kv put secret/api_keys/openai \
    key="dev_openai_api_key"

vault kv put secret/api_keys/anthropic \
    key="dev_anthropic_api_key"

# OAuth credentials
vault kv put secret/oauth/google \
    client_id="dev_google_client_id" \
    client_secret="dev_google_client_secret"

echo "Development secrets populated successfully"
```

## Production Deployment

### 1. High Availability Setup

```yaml
# k8s/vault-ha.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: vault
spec:
  serviceName: vault
  replicas: 3
  selector:
    matchLabels:
      app: vault
  template:
    metadata:
      labels:
        app: vault
    spec:
      containers:
      - name: vault
        image: vault:1.15.2
        env:
        - name: VAULT_ADDR
          value: "https://0.0.0.0:8200"
        - name: VAULT_CLUSTER_ADDR
          value: "https://0.0.0.0:8201"
        - name: VAULT_REDIRECT_ADDR
          value: "https://vault.grandmodel.com"
        volumeMounts:
        - name: vault-config
          mountPath: /vault/config
        - name: vault-data
          mountPath: /vault/data
        - name: vault-tls
          mountPath: /vault/tls
```

### 2. Auto-Unseal Configuration

```hcl
# AWS KMS auto-unseal
seal "awskms" {
  region     = "us-east-1"
  kms_key_id = "alias/vault-seal-key"
}

# Azure Key Vault auto-unseal
seal "azurekeyvault" {
  tenant_id      = "your-tenant-id"
  client_id      = "your-client-id"
  client_secret  = "your-client-secret"
  vault_name     = "vault-seal-keyvault"
  key_name       = "vault-seal-key"
}
```

### 3. TLS Configuration

```bash
# Generate TLS certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout vault.key -out vault.crt \
    -subj "/C=US/ST=State/L=City/O=GrandModel/CN=vault.grandmodel.com"

# Create Kubernetes secret
kubectl create secret tls vault-tls \
    --cert=vault.crt \
    --key=vault.key \
    -n grandmodel
```

### 4. Production Secrets Management

```bash
#!/bin/bash
# scripts/populate_prod_secrets.sh

# Generate secure JWT secret
JWT_SECRET=$(openssl rand -base64 64)
vault kv put secret/app/jwt secret_key="$JWT_SECRET"

# Set production user passwords
vault kv put secret/users/admin password="$(openssl rand -base64 32)"
vault kv put secret/users/risk_manager password="$(openssl rand -base64 32)"
vault kv put secret/users/operator password="$(openssl rand -base64 32)"
vault kv put secret/users/compliance password="$(openssl rand -base64 32)"

# Database credentials
vault kv put secret/app/database/primary \
    host="prod-db.grandmodel.com" \
    port="5432" \
    database="grandmodel_prod" \
    username="grandmodel_app" \
    password="$(openssl rand -base64 32)"

# API keys (set actual values)
vault kv put secret/api_keys/openai key="$OPENAI_API_KEY"
vault kv put secret/api_keys/anthropic key="$ANTHROPIC_API_KEY"
vault kv put secret/api_keys/bloomberg key="$BLOOMBERG_API_KEY"

# OAuth credentials
vault kv put secret/oauth/google \
    client_id="$GOOGLE_CLIENT_ID" \
    client_secret="$GOOGLE_CLIENT_SECRET"

echo "Production secrets configured securely"
```

## Secret Management

### Reading Secrets

```bash
# Read JWT secret
vault kv get secret/app/jwt

# Read database credentials
vault kv get secret/app/database/primary

# Read specific field
vault kv get -field=password secret/app/database/primary
```

### Updating Secrets

```bash
# Update JWT secret
vault kv put secret/app/jwt secret_key="new_jwt_secret_key"

# Update database password
vault kv patch secret/app/database/primary password="new_db_password"
```

### Secret Rotation

```bash
#!/bin/bash
# scripts/rotate_secrets.sh

# Function to rotate JWT secret
rotate_jwt_secret() {
    NEW_SECRET=$(openssl rand -base64 64)
    vault kv put secret/app/jwt secret_key="$NEW_SECRET"
    echo "JWT secret rotated"
}

# Function to rotate database password
rotate_db_password() {
    NEW_PASSWORD=$(openssl rand -base64 32)
    vault kv patch secret/app/database/primary password="$NEW_PASSWORD"
    echo "Database password rotated"
}

# Function to rotate user passwords
rotate_user_passwords() {
    for user in admin risk_manager operator compliance; do
        NEW_PASSWORD=$(openssl rand -base64 32)
        vault kv put secret/users/$user password="$NEW_PASSWORD"
        echo "Password rotated for user: $user"
    done
}

# Execute rotation
rotate_jwt_secret
rotate_db_password
rotate_user_passwords

echo "Secret rotation completed"
```

### Backup and Recovery

```bash
#!/bin/bash
# scripts/backup_vault.sh

# Create snapshot
vault operator raft snapshot save backup-$(date +%Y%m%d-%H%M%S).snap

# Restore from snapshot
vault operator raft snapshot restore backup-20231215-120000.snap
```

## Application Integration

### Python Code Examples

```python
# Example: Getting JWT secret
from src.security.vault_client import get_jwt_secret

async def authenticate_user():
    jwt_secret = await get_jwt_secret()
    # Use jwt_secret for token validation
    
# Example: Getting database credentials
from src.security.vault_client import get_database_credentials

async def connect_to_database():
    creds = await get_database_credentials()
    connection_string = f"postgresql://{creds['username']}:{creds['password']}@{creds['host']}:{creds['port']}/{creds['database']}"
    
# Example: Getting API key
from src.security.vault_client import get_secret

async def call_external_api():
    api_key = await get_secret("api_keys/openai", "key")
    # Use api_key for API calls
```

### Environment Variables

```bash
# Set in production environment
export VAULT_URL=https://vault.grandmodel.com
export VAULT_ROLE_ID=your_role_id
export VAULT_SECRET_ID=your_secret_id
export VAULT_NAMESPACE=grandmodel
```

## Monitoring and Auditing

### Vault Audit Logs

```bash
# Enable audit logging
vault audit enable file file_path=/vault/logs/vault_audit.log

# Enable syslog audit
vault audit enable syslog tag="vault" facility="LOCAL0"
```

### Monitoring Configuration

```yaml
# prometheus/vault-monitoring.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: vault-monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'vault'
      static_configs:
      - targets: ['vault:8200']
      metrics_path: '/v1/sys/metrics'
      params:
        format: ['prometheus']
```

## Troubleshooting

### Common Issues

1. **Vault is sealed**
   ```bash
   # Check seal status
   vault status
   
   # Unseal vault
   vault operator unseal
   ```

2. **Authentication failed**
   ```bash
   # Check authentication methods
   vault auth list
   
   # Verify token
   vault token lookup
   ```

3. **Permission denied**
   ```bash
   # Check policies
   vault policy list
   vault policy read grandmodel-policy
   
   # Check token policies
   vault token lookup -format=json | jq '.data.policies'
   ```

4. **Secret not found**
   ```bash
   # List secrets
   vault kv list secret/
   
   # Check secret path
   vault kv get secret/app/jwt
   ```

### Health Checks

```bash
# Check Vault health
vault status

# Check authentication
vault auth list

# Check policies
vault policy list

# Check secret engines
vault secrets list
```

### Application Debugging

```python
# Enable debug logging
import logging
logging.getLogger('src.security.vault_client').setLevel(logging.DEBUG)

# Check Vault health from application
from src.security.vault_client import get_vault_client

async def debug_vault():
    client = await get_vault_client()
    health = await client.health_check()
    print(f"Vault health: {health}")
```

## Security Best Practices

### 1. Least Privilege Access

- Create specific policies for different applications
- Use AppRole authentication for applications
- Rotate tokens regularly
- Implement short TTL values

### 2. Network Security

- Use TLS for all Vault communications
- Implement network segmentation
- Use firewall rules to restrict access
- Enable audit logging

### 3. Secret Management

- Never store secrets in code or configuration files
- Use dynamic secrets when possible
- Implement secret rotation
- Monitor secret access

### 4. Backup and Recovery

- Regular backups of Vault data
- Test restore procedures
- Document recovery processes
- Implement disaster recovery plans

### 5. Monitoring and Alerting

- Monitor Vault health and performance
- Set up alerts for seal events
- Track secret access patterns
- Implement audit log analysis

## Migration from Hardcoded Secrets

### Phase 1: Inventory (Completed)
- ✅ Identified 7+ hardcoded secrets
- ✅ Documented secret locations
- ✅ Categorized secret types

### Phase 2: Vault Setup (Completed)
- ✅ Implemented Vault client
- ✅ Created authentication methods
- ✅ Configured secret engines

### Phase 3: Code Migration (Completed)
- ✅ Updated authentication module
- ✅ Replaced hardcoded JWT secrets
- ✅ Migrated user passwords
- ✅ Updated test configurations

### Phase 4: Deployment (Next Steps)
- 🔄 Deploy Vault in production
- 🔄 Configure monitoring
- 🔄 Implement backup procedures
- 🔄 Train operations team

## Conclusion

The HashiCorp Vault integration provides enterprise-grade secret management for the GrandModel trading system. All hardcoded secrets have been eliminated and replaced with secure Vault integration, including fallback mechanisms for development environments.

**Key Benefits:**
- ✅ No more hardcoded secrets in code
- ✅ Centralized secret management
- ✅ Audit trails for all secret access
- ✅ Automatic secret rotation capability
- ✅ Fine-grained access controls
- ✅ Development environment fallbacks

**Next Steps:**
1. Deploy Vault in production environment
2. Configure monitoring and alerting
3. Implement automated secret rotation
4. Train development and operations teams
5. Conduct security audit and penetration testing

For questions or issues, contact the Security Team or refer to the [HashiCorp Vault documentation](https://www.vaultproject.io/docs).