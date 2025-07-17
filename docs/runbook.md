# GrandModel Production Runbook

## Table of Contents
1. [Quick Start](#quick-start)
2. [System Architecture](#system-architecture)
3. [Starting the System](#starting-the-system)
4. [Health Monitoring](#health-monitoring)
5. [Viewing Logs](#viewing-logs)
6. [Common Operations](#common-operations)
7. [Troubleshooting](#troubleshooting)
8. [Emergency Procedures](#emergency-procedures)

## Quick Start

### Prerequisites
- Docker 20.10+ and Docker Compose 2.0+
- 4GB+ RAM available
- Ports 80, 443, 8000, 9090, 3000 available
- Valid secrets configured in `/secrets` directory

### Start Everything
```bash
# Clone repository
git clone https://github.com/Afeks214/GrandModel.git
cd GrandModel

# Configure secrets (see secrets/README.md)
cp secrets.example/* secrets/

# Start all services
docker-compose -f docker-compose.prod.yml up -d

# Verify health
curl http://localhost:8000/health
```

## System Architecture

### Core Services
- **GrandModel API** (port 8000): Main application server
- **Redis** (port 6379): Cache and message broker
- **Prometheus** (port 9090): Metrics collection
- **Grafana** (port 3000): Metrics visualization
- **NGINX** (ports 80/443): Reverse proxy and load balancer
- **Ollama** (port 11434): LLM integration

### Network Architecture
```
Internet → NGINX → GrandModel API → Redis
                 ↓                  ↓
            Prometheus ← ← ← ← ← ← ↓
                 ↓
             Grafana
```

## Starting the System

### Step-by-Step Startup

1. **Prepare Environment**
```bash
# Set environment variables
export REDIS_PASSWORD=$(openssl rand -base64 32)
export JWT_SECRET=$(openssl rand -base64 32)
export GRAFANA_PASSWORD=$(openssl rand -base64 24)

# Save to .env file
cat > .env << EOF
REDIS_PASSWORD=${REDIS_PASSWORD}
JWT_SECRET=${JWT_SECRET}
GRAFANA_PASSWORD=${GRAFANA_PASSWORD}
API_BASE_URL=https://grandmodel.app
EOF
```

2. **Start Infrastructure Services**
```bash
# Start Redis first
docker-compose -f docker-compose.prod.yml up -d redis

# Verify Redis is healthy
docker-compose -f docker-compose.prod.yml exec redis redis-cli ping

# Start monitoring stack
docker-compose -f docker-compose.prod.yml up -d prometheus grafana
```

3. **Start Application**
```bash
# Build and start main application
docker-compose -f docker-compose.prod.yml up -d grandmodel

# Start NGINX
docker-compose -f docker-compose.prod.yml up -d nginx
```

4. **Verify Startup**
```bash
# Check all containers are running
docker-compose -f docker-compose.prod.yml ps

# Check application health
curl -s http://localhost:8000/health | jq .
```

### Expected Output
```json
{
  "status": "healthy",
  "timestamp": "2024-01-11T10:00:00Z",
  "components": [
    {"name": "redis", "status": "healthy", "message": "Connection: OK"},
    {"name": "models", "status": "healthy", "message": "All models loaded"},
    {"name": "api", "status": "healthy", "message": "API responding"},
    {"name": "monitoring", "status": "healthy", "message": "Metrics available"}
  ],
  "version": "1.0.0"
}
```

## Health Monitoring

### Health Check Endpoint
```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health with auth
curl -H "Authorization: Bearer ${API_TOKEN}" \
     http://localhost:8000/health
```

### Component Health Checks

1. **Redis Health**
```bash
docker-compose -f docker-compose.prod.yml exec redis redis-cli \
  --pass ${REDIS_PASSWORD} INFO server
```

2. **API Health**
```bash
# Check response time
time curl -s http://localhost:8000/health > /dev/null

# Check metrics endpoint
curl -s http://localhost:8000/metrics | grep grandmodel_health_status
```

3. **System Resources**
```bash
# Check container resources
docker stats --no-stream

# Check specific container
docker inspect grandmodel-app | jq '.[0].State.Health'
```

## Viewing Logs

### Application Logs
```bash
# Follow main application logs
docker-compose -f docker-compose.prod.yml logs -f grandmodel

# View last 100 lines
docker-compose -f docker-compose.prod.yml logs --tail=100 grandmodel

# Filter by log level
docker-compose -f docker-compose.prod.yml logs grandmodel | \
  jq 'select(.level == "ERROR")'
```

### Service-Specific Logs
```bash
# Redis logs
docker-compose -f docker-compose.prod.yml logs redis

# NGINX access logs
docker-compose -f docker-compose.prod.yml logs nginx

# Prometheus logs
docker-compose -f docker-compose.prod.yml logs prometheus
```

### Structured Log Queries
```bash
# Find logs by correlation ID
CORRELATION_ID="abc-123-def"
docker-compose -f docker-compose.prod.yml logs grandmodel | \
  jq --arg id "$CORRELATION_ID" 'select(.correlation_id == $id)'

# Find slow requests (>100ms)
docker-compose -f docker-compose.prod.yml logs grandmodel | \
  jq 'select(.duration_seconds > 0.1)'

# Count errors by type
docker-compose -f docker-compose.prod.yml logs grandmodel | \
  jq -r 'select(.level == "ERROR") | .error_type' | \
  sort | uniq -c
```

## Common Operations

### Scaling Services
```bash
# Scale API workers
docker-compose -f docker-compose.prod.yml up -d --scale grandmodel=3

# Verify scaling
docker-compose -f docker-compose.prod.yml ps | grep grandmodel
```

### Updating Configuration
```bash
# Edit configuration
vim configs/system/production.yaml

# Restart service to apply
docker-compose -f docker-compose.prod.yml restart grandmodel

# Verify new config loaded
docker-compose -f docker-compose.prod.yml logs --tail=50 grandmodel | \
  grep "Configuration loaded"
```

### Rotating Secrets
```bash
# Generate new secret
NEW_JWT_SECRET=$(openssl rand -base64 32)

# Update secret file
echo $NEW_JWT_SECRET > secrets/jwt_secret.txt

# Recreate containers to load new secret
docker-compose -f docker-compose.prod.yml up -d --force-recreate grandmodel
```

### Database Operations
```bash
# Connect to Redis
docker-compose -f docker-compose.prod.yml exec redis redis-cli \
  --pass ${REDIS_PASSWORD}

# Backup Redis data
docker-compose -f docker-compose.prod.yml exec redis redis-cli \
  --pass ${REDIS_PASSWORD} BGSAVE

# Check background save status
docker-compose -f docker-compose.prod.yml exec redis redis-cli \
  --pass ${REDIS_PASSWORD} LASTSAVE
```

## Troubleshooting

### Service Won't Start

1. **Check logs**
```bash
docker-compose -f docker-compose.prod.yml logs --tail=100 [service_name]
```

2. **Check configuration**
```bash
# Validate Docker Compose file
docker-compose -f docker-compose.prod.yml config

# Check for missing secrets
ls -la secrets/
```

3. **Check resources**
```bash
# Check disk space
df -h

# Check memory
free -h

# Check port conflicts
netstat -tulpn | grep -E '(8000|6379|9090|3000)'
```

### High Latency

1. **Check metrics**
```bash
# Query Prometheus for latency metrics
curl -s http://localhost:9090/api/v1/query?query=grandmodel_inference_latency_seconds
```

2. **Check resource usage**
```bash
# Container CPU/Memory
docker stats grandmodel-app

# Redis performance
docker-compose -f docker-compose.prod.yml exec redis redis-cli \
  --pass ${REDIS_PASSWORD} INFO stats
```

3. **Check connection pool**
```bash
# Active connections
curl -s http://localhost:8000/metrics | \
  grep grandmodel_active_connections
```

### Memory Issues

1. **Identify memory usage**
```bash
# Container memory
docker stats --no-stream --format "table {{.Container}}\t{{.MemUsage}}"

# Process memory inside container
docker-compose -f docker-compose.prod.yml exec grandmodel \
  ps aux --sort=-%mem | head
```

2. **Clear caches**
```bash
# Clear Redis cache
docker-compose -f docker-compose.prod.yml exec redis redis-cli \
  --pass ${REDIS_PASSWORD} FLUSHDB

# Restart to free memory
docker-compose -f docker-compose.prod.yml restart grandmodel
```

## Emergency Procedures

### Service Degradation

1. **Enable circuit breaker**
```bash
# Set feature flag to disable ML inference
curl -X POST http://localhost:8000/admin/feature-flags \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  -d '{"enable_ml_inference": false}'
```

2. **Increase rate limits**
```bash
# Temporarily restrict traffic
docker-compose -f docker-compose.prod.yml exec nginx \
  nginx -s reload
```

### Complete System Restart

```bash
# Stop all services gracefully
docker-compose -f docker-compose.prod.yml down

# Clean up volumes (WARNING: data loss)
docker-compose -f docker-compose.prod.yml down -v

# Start fresh
docker-compose -f docker-compose.prod.yml up -d
```

### Rollback Procedure

```bash
# Tag current version
docker tag grandmodel:latest grandmodel:rollback

# Deploy previous version
docker-compose -f docker-compose.prod.yml \
  pull grandmodel:previous

docker-compose -f docker-compose.prod.yml up -d
```

### Data Recovery

```bash
# Restore Redis from backup
docker-compose -f docker-compose.prod.yml stop redis
docker cp redis_backup.rdb grandmodel-redis:/data/dump.rdb
docker-compose -f docker-compose.prod.yml start redis
```

## Monitoring Dashboard Access

### Grafana
- URL: http://localhost:3000
- Default user: admin
- Password: Set in GRAFANA_PASSWORD env var

### Prometheus
- URL: http://localhost:9090
- No authentication by default

### Key Dashboards
1. **System Overview**: Overall health and performance
2. **API Metrics**: Request rates, latencies, errors
3. **Trading Metrics**: Positions, P&L, model confidence
4. **Infrastructure**: CPU, memory, disk, network

## Contact Information

### Escalation Path
1. On-call engineer (check PagerDuty)
2. Team lead
3. Infrastructure team
4. Security team (for security incidents)

### External Dependencies
- Redis: Internal team
- Ollama: support@ollama.ai
- Infrastructure: Cloud provider support

---

**Last Updated**: 2024-01-11
**Version**: 1.0.0
**Maintained By**: Systems & MLOps Team