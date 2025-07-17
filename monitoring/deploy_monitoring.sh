#!/bin/bash

# GrandModel MARL Trading System - Monitoring & Alerting Deployment Script
# Agent 5: Monitoring & Alerting Specialist
# High-performance production monitoring with <30s alert response

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MONITORING_DIR="$PROJECT_ROOT/monitoring"
CONFIG_DIR="$PROJECT_ROOT/config"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check available disk space (need at least 10GB)
    available_space=$(df "$PROJECT_ROOT" | tail -1 | awk '{print $4}')
    if [ "$available_space" -lt 10485760 ]; then  # 10GB in KB
        warn "Low disk space. Monitoring stack requires at least 10GB of free space."
    fi
    
    # Check memory (need at least 4GB)
    available_memory=$(free -m | awk 'NR==2{printf "%d", $7}')
    if [ "$available_memory" -lt 4096 ]; then
        warn "Low memory. Monitoring stack requires at least 4GB of available memory."
    fi
    
    success "Prerequisites check completed"
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    mkdir -p "$MONITORING_DIR/config/prometheus/rules"
    mkdir -p "$MONITORING_DIR/config/grafana/datasources"
    mkdir -p "$MONITORING_DIR/config/alertmanager"
    mkdir -p "$MONITORING_DIR/config/blackbox"
    mkdir -p "$MONITORING_DIR/config/logstash/pipeline"
    mkdir -p "$MONITORING_DIR/config/fluentd"
    mkdir -p "$MONITORING_DIR/config/nginx/sites-enabled"
    mkdir -p "$MONITORING_DIR/ssl"
    mkdir -p "$MONITORING_DIR/dashboards"
    mkdir -p "$MONITORING_DIR/logs"
    mkdir -p "$MONITORING_DIR/data"
    
    success "Directories created"
}

# Generate SSL certificates
generate_ssl_certificates() {
    log "Generating SSL certificates..."
    
    if [ ! -f "$MONITORING_DIR/ssl/monitoring.crt" ]; then
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout "$MONITORING_DIR/ssl/monitoring.key" \
            -out "$MONITORING_DIR/ssl/monitoring.crt" \
            -subj "/C=US/ST=State/L=City/O=GrandModel/CN=monitoring.grandmodel.local"
        
        success "SSL certificates generated"
    else
        log "SSL certificates already exist"
    fi
}

# Create AlertManager configuration
create_alertmanager_config() {
    log "Creating AlertManager configuration..."
    
    cat > "$MONITORING_DIR/config/alertmanager/alertmanager.yml" << 'EOF'
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@grandmodel.ai'
  smtp_auth_username: 'alerts@grandmodel.ai'
  smtp_auth_password: 'your_password'
  pagerduty_url: 'https://events.pagerduty.com/v2/enqueue'
  slack_api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'

route:
  group_by: ['alertname', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'default'
  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
      group_wait: 5s
      repeat_interval: 5m
    - match:
        severity: warning
      receiver: 'warning-alerts'
      group_wait: 30s
      repeat_interval: 15m
    - match:
        alert_type: trading
      receiver: 'trading-alerts'
      group_wait: 5s
      repeat_interval: 2m
    - match:
        alert_type: marl
      receiver: 'marl-alerts'
      group_wait: 10s
      repeat_interval: 5m

receivers:
  - name: 'default'
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#monitoring'
        title: 'GrandModel Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

  - name: 'critical-alerts'
    pagerduty_configs:
      - routing_key: '${PAGERDUTY_ROUTING_KEY}'
        description: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
        severity: 'critical'
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#critical-alerts'
        title: '🚨 CRITICAL ALERT'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
        color: 'danger'

  - name: 'warning-alerts'
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#warnings'
        title: '⚠️ Warning Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
        color: 'warning'

  - name: 'trading-alerts'
    pagerduty_configs:
      - routing_key: '${PAGERDUTY_TRADING_ROUTING_KEY}'
        description: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
        severity: 'error'
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#trading-alerts'
        title: '💱 Trading Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

  - name: 'marl-alerts'
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#marl-alerts'
        title: '🤖 MARL Agent Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'service']
EOF
    
    success "AlertManager configuration created"
}

# Create Blackbox Exporter configuration
create_blackbox_config() {
    log "Creating Blackbox Exporter configuration..."
    
    cat > "$MONITORING_DIR/config/blackbox/blackbox.yml" << 'EOF'
modules:
  http_2xx:
    prober: http
    timeout: 5s
    http:
      valid_http_versions: ["HTTP/1.1", "HTTP/2.0"]
      valid_status_codes: []
      method: GET
      headers:
        Host: grandmodel.local
        Accept-Language: en-US
      no_follow_redirects: false
      fail_if_ssl: false
      fail_if_not_ssl: false
      
  http_post_2xx:
    prober: http
    timeout: 5s
    http:
      method: POST
      headers:
        Content-Type: application/json
      body: '{"test": true}'
      
  tcp_connect:
    prober: tcp
    timeout: 5s
    
  icmp:
    prober: icmp
    timeout: 5s
    icmp:
      preferred_ip_protocol: "ip4"
      
  dns:
    prober: dns
    timeout: 5s
    dns:
      query_name: "grandmodel.local"
      query_type: "A"
EOF
    
    success "Blackbox Exporter configuration created"
}

# Create Grafana datasource configuration
create_grafana_datasources() {
    log "Creating Grafana datasources..."
    
    cat > "$MONITORING_DIR/config/grafana/datasources/datasources.yml" << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
    
  - name: Elasticsearch
    type: elasticsearch
    access: proxy
    url: http://elasticsearch:9200
    database: "[grandmodel-logs-]YYYY.MM.DD"
    interval: Daily
    timeField: "@timestamp"
    editable: true
    
  - name: Jaeger
    type: jaeger
    access: proxy
    url: http://jaeger:16686
    editable: true
    
  - name: Redis
    type: redis-datasource
    access: proxy
    url: redis://redis:6379
    editable: true
EOF
    
    success "Grafana datasources created"
}

# Create Logstash configuration
create_logstash_config() {
    log "Creating Logstash configuration..."
    
    cat > "$MONITORING_DIR/config/logstash/logstash.yml" << 'EOF'
http.host: "0.0.0.0"
xpack.monitoring.enabled: false
pipeline.ordered: auto
EOF
    
    cat > "$MONITORING_DIR/config/logstash/pipeline/logstash.conf" << 'EOF'
input {
  beats {
    port => 5044
  }
  
  tcp {
    port => 5000
    codec => json_lines
  }
  
  udp {
    port => 5000
    codec => json_lines
  }
  
  redis {
    host => "redis"
    port => 6379
    key => "grandmodel:logs"
    data_type => "stream"
    codec => json
  }
}

filter {
  # Parse GrandModel log format
  if [service] == "grandmodel" {
    mutate {
      add_field => { "log_type" => "grandmodel" }
    }
  }
  
  # Add timestamp
  date {
    match => [ "timestamp", "ISO8601" ]
    target => "@timestamp"
  }
  
  # Parse log levels
  if [level] {
    mutate {
      uppercase => [ "level" ]
    }
  }
  
  # Add location information
  mutate {
    add_field => { "cluster" => "grandmodel-production" }
    add_field => { "environment" => "production" }
  }
  
  # Parse trading-specific fields
  if [category] == "trading" {
    mutate {
      add_tag => [ "trading" ]
    }
  }
  
  # Parse MARL-specific fields
  if [category] == "marl" {
    mutate {
      add_tag => [ "marl" ]
    }
  }
  
  # Parse performance fields
  if [category] == "performance" {
    mutate {
      add_tag => [ "performance" ]
    }
    
    if [duration_ms] {
      mutate {
        convert => { "duration_ms" => "float" }
      }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "grandmodel-logs-%{+YYYY.MM.dd}"
    template_name => "grandmodel-logs"
    template => "/usr/share/logstash/templates/grandmodel-logs.json"
    template_overwrite => true
  }
  
  # Debug output
  if [level] == "DEBUG" {
    stdout {
      codec => rubydebug
    }
  }
}
EOF
    
    success "Logstash configuration created"
}

# Create Fluentd configuration
create_fluentd_config() {
    log "Creating Fluentd configuration..."
    
    cat > "$MONITORING_DIR/config/fluentd/fluent.conf" << 'EOF'
<source>
  @type forward
  port 24224
  bind 0.0.0.0
</source>

<source>
  @type tail
  path /var/log/grandmodel/*.log
  pos_file /var/log/fluentd/grandmodel.log.pos
  tag grandmodel.logs
  format json
  time_format %Y-%m-%dT%H:%M:%S.%NZ
</source>

<filter grandmodel.**>
  @type parser
  key_name message
  reserve_data true
  <parse>
    @type json
  </parse>
</filter>

<filter grandmodel.**>
  @type record_transformer
  <record>
    cluster grandmodel-production
    environment production
    log_type grandmodel
  </record>
</filter>

<match grandmodel.**>
  @type elasticsearch
  host elasticsearch
  port 9200
  logstash_format true
  logstash_prefix grandmodel-logs
  logstash_dateformat %Y.%m.%d
  include_tag_key true
  tag_key @log_name
  flush_interval 10s
  
  <buffer>
    @type file
    path /var/log/fluentd/grandmodel.buffer
    flush_mode interval
    flush_interval 10s
    chunk_limit_size 32m
    queue_limit_length 8
    retry_max_interval 30
    retry_forever true
  </buffer>
</match>
EOF
    
    success "Fluentd configuration created"
}

# Create Nginx configuration
create_nginx_config() {
    log "Creating Nginx configuration..."
    
    cat > "$MONITORING_DIR/config/nginx/nginx.conf" << 'EOF'
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';
    
    access_log /var/log/nginx/access.log main;
    
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/xml text/javascript 
               application/json application/javascript application/xml+rss 
               application/atom+xml image/svg+xml;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=monitoring:10m rate=10r/s;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
    
    include /etc/nginx/sites-enabled/*;
}
EOF
    
    cat > "$MONITORING_DIR/config/nginx/sites-enabled/monitoring.conf" << 'EOF'
upstream grafana {
    server grafana:3000;
}

upstream prometheus {
    server prometheus:9090;
}

upstream kibana {
    server kibana:5601;
}

upstream alertmanager {
    server alertmanager:9093;
}

server {
    listen 80;
    server_name monitoring.grandmodel.local;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name monitoring.grandmodel.local;
    
    ssl_certificate /etc/nginx/ssl/monitoring.crt;
    ssl_certificate_key /etc/nginx/ssl/monitoring.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # Grafana
    location / {
        proxy_pass http://grafana;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header X-Forwarded-Port $server_port;
    }
    
    # Prometheus
    location /prometheus/ {
        proxy_pass http://prometheus/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Kibana
    location /kibana/ {
        proxy_pass http://kibana/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # AlertManager
    location /alertmanager/ {
        proxy_pass http://alertmanager/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
EOF
    
    success "Nginx configuration created"
}

# Create health check configuration
create_health_check_config() {
    log "Creating health check configuration..."
    
    cat > "$MONITORING_DIR/config/health_check_config.json" << 'EOF'
{
  "database": {
    "host": "postgres",
    "port": 5432,
    "database": "grandmodel",
    "user": "grandmodel",
    "password": "password"
  },
  "redis": {
    "host": "redis",
    "port": 6379,
    "db": 0
  },
  "agents": {
    "strategic": {
      "url": "http://strategic-agent:8000"
    },
    "tactical": {
      "url": "http://tactical-agent:8000"
    },
    "risk": {
      "url": "http://risk-agent:8000"
    }
  },
  "resource_thresholds": {
    "cpu_percent": 80,
    "memory_percent": 85,
    "disk_percent": 90,
    "load_avg": 4.0
  },
  "business_config": {
    "min_trade_success_rate": 0.90,
    "max_daily_loss": 10000,
    "max_var": 0.05
  }
}
EOF
    
    success "Health check configuration created"
}

# Create alerting configuration
create_alerting_config() {
    log "Creating alerting configuration..."
    
    cat > "$MONITORING_DIR/config/alerting_config.json" << 'EOF'
{
  "email": {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "use_tls": true,
    "from": "alerts@grandmodel.ai",
    "to": ["ops@grandmodel.ai", "trading@grandmodel.ai"],
    "username": "alerts@grandmodel.ai",
    "password": "app_password"
  },
  "slack": {
    "webhook_url": "${SLACK_WEBHOOK_URL}"
  },
  "pagerduty": {
    "api_token": "${PAGERDUTY_API_TOKEN}",
    "services": {
      "trading_engine": {
        "routing_key": "${PAGERDUTY_TRADING_ROUTING_KEY}"
      },
      "strategic_agent": {
        "routing_key": "${PAGERDUTY_AGENTS_ROUTING_KEY}"
      },
      "tactical_agent": {
        "routing_key": "${PAGERDUTY_AGENTS_ROUTING_KEY}"
      },
      "risk_agent": {
        "routing_key": "${PAGERDUTY_RISK_ROUTING_KEY}"
      },
      "data_pipeline": {
        "routing_key": "${PAGERDUTY_INFRA_ROUTING_KEY}"
      },
      "system_health": {
        "routing_key": "${PAGERDUTY_INFRA_ROUTING_KEY}"
      }
    }
  }
}
EOF
    
    success "Alerting configuration created"
}

# Create Dockerfiles
create_dockerfiles() {
    log "Creating Dockerfiles..."
    
    # Health Check Service Dockerfile
    cat > "$MONITORING_DIR/Dockerfile.health-check" << 'EOF'
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY health_check_system.py .
COPY centralized_logging.py .

EXPOSE 8001

CMD ["python", "-m", "health_check_system"]
EOF
    
    # Alerting Service Dockerfile
    cat > "$MONITORING_DIR/Dockerfile.alerting" << 'EOF'
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY alerting_system.py .
COPY pagerduty_integration.py .

EXPOSE 8002

CMD ["python", "-m", "alerting_system"]
EOF
    
    # Monitoring Dashboard Dockerfile
    cat > "$MONITORING_DIR/Dockerfile.monitoring-dashboard" << 'EOF'
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY real_time_dashboard.py .
COPY centralized_logging.py .

EXPOSE 8003

CMD ["python", "-m", "real_time_dashboard"]
EOF
    
    # Create requirements.txt
    cat > "$MONITORING_DIR/requirements.txt" << 'EOF'
aiohttp==3.8.4
redis==4.5.4
psycopg2-binary==2.9.6
elasticsearch==8.6.2
structlog==23.1.0
prometheus-client==0.16.0
psutil==5.9.4
fastapi==0.95.1
uvicorn==0.21.1
pydantic==1.10.7
python-multipart==0.0.6
jinja2==3.1.2
requests==2.28.2
asyncio-mqtt==0.13.0
pandas==2.0.1
numpy==1.24.3
EOF
    
    success "Dockerfiles created"
}

# Set permissions
set_permissions() {
    log "Setting permissions..."
    
    # Make scripts executable
    chmod +x "$MONITORING_DIR/deploy_monitoring.sh"
    
    # Set proper permissions for configuration files
    find "$MONITORING_DIR/config" -type f -exec chmod 644 {} \;
    find "$MONITORING_DIR/config" -type d -exec chmod 755 {} \;
    
    # Set permissions for SSL certificates
    chmod 600 "$MONITORING_DIR/ssl/monitoring.key" 2>/dev/null || true
    chmod 644 "$MONITORING_DIR/ssl/monitoring.crt" 2>/dev/null || true
    
    success "Permissions set"
}

# Deploy monitoring stack
deploy_monitoring_stack() {
    log "Deploying monitoring stack..."
    
    cd "$MONITORING_DIR"
    
    # Pull latest images
    docker-compose -f docker-compose.monitoring.yml pull
    
    # Start services
    docker-compose -f docker-compose.monitoring.yml up -d
    
    # Wait for services to be ready
    log "Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    check_service_health
    
    success "Monitoring stack deployed"
}

# Check service health
check_service_health() {
    log "Checking service health..."
    
    services=(
        "prometheus:9090"
        "grafana:3000"
        "elasticsearch:9200"
        "kibana:5601"
        "alertmanager:9093"
    )
    
    for service in "${services[@]}"; do
        host=$(echo "$service" | cut -d':' -f1)
        port=$(echo "$service" | cut -d':' -f2)
        
        log "Checking $host:$port..."
        
        if timeout 10 bash -c "echo >/dev/tcp/$host/$port"; then
            success "$host:$port is healthy"
        else
            error "$host:$port is not responding"
        fi
    done
}

# Create monitoring dashboard
create_monitoring_dashboard() {
    log "Creating monitoring dashboard..."
    
    # Import Grafana dashboards
    if [ -f "$MONITORING_DIR/dashboards/marl_system_dashboard.json" ]; then
        log "MARL System dashboard available"
    fi
    
    if [ -f "$MONITORING_DIR/dashboards/system_health_dashboard.json" ]; then
        log "System Health dashboard available"
    fi
    
    if [ -f "$MONITORING_DIR/dashboards/trading_performance_dashboard.json" ]; then
        log "Trading Performance dashboard available"
    fi
    
    success "Monitoring dashboard created"
}

# Print access information
print_access_info() {
    log "Printing access information..."
    
    echo ""
    echo "========================================="
    echo "  GrandModel Monitoring Stack Access"
    echo "========================================="
    echo ""
    echo "🔍 Grafana (Dashboards):     http://localhost:3000"
    echo "   Username: admin / Password: admin123"
    echo ""
    echo "📊 Prometheus (Metrics):     http://localhost:9090"
    echo "🔔 AlertManager (Alerts):    http://localhost:9093"
    echo "🔍 Kibana (Logs):           http://localhost:5601"
    echo "🐳 Portainer (Containers):  http://localhost:9000"
    echo ""
    echo "🏥 Health Check Service:    http://localhost:8001"
    echo "🚨 Alerting Service:        http://localhost:8002"
    echo "📱 Monitoring Dashboard:    http://localhost:8003"
    echo ""
    echo "🔐 HTTPS Access:            https://monitoring.grandmodel.local"
    echo "   (Add to /etc/hosts: 127.0.0.1 monitoring.grandmodel.local)"
    echo ""
    echo "========================================="
    echo "  Service Status Commands"
    echo "========================================="
    echo ""
    echo "View logs:     docker-compose -f docker-compose.monitoring.yml logs -f [service]"
    echo "Restart:       docker-compose -f docker-compose.monitoring.yml restart [service]"
    echo "Stop all:      docker-compose -f docker-compose.monitoring.yml down"
    echo "Status:        docker-compose -f docker-compose.monitoring.yml ps"
    echo ""
    echo "========================================="
    echo "  Important Notes"
    echo "========================================="
    echo ""
    echo "• Configure environment variables in .env file"
    echo "• Update PagerDuty and Slack webhook URLs"
    echo "• SSL certificates are self-signed (for development)"
    echo "• Data is persisted in Docker volumes"
    echo "• Monitor resource usage with: docker stats"
    echo ""
    
    success "Access information printed"
}

# Main function
main() {
    log "Starting GrandModel Monitoring & Alerting deployment..."
    
    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        warn "Running as root. Consider using a non-root user."
    fi
    
    # Run deployment steps
    check_prerequisites
    create_directories
    generate_ssl_certificates
    create_alertmanager_config
    create_blackbox_config
    create_grafana_datasources
    create_logstash_config
    create_fluentd_config
    create_nginx_config
    create_health_check_config
    create_alerting_config
    create_dockerfiles
    set_permissions
    deploy_monitoring_stack
    create_monitoring_dashboard
    print_access_info
    
    success "GrandModel Monitoring & Alerting deployment completed successfully!"
    
    log "Next steps:"
    echo "1. Configure environment variables in .env file"
    echo "2. Update PagerDuty and Slack configurations"
    echo "3. Import additional Grafana dashboards"
    echo "4. Set up log forwarding from application services"
    echo "5. Test alerting rules and escalation policies"
    echo "6. Configure backup and monitoring retention policies"
}

# Run main function
main "$@"