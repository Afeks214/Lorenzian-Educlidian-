#!/bin/bash

# Deployment Script for Patroni RTO Optimization
# Applies optimized configuration to achieve <30s RTO target

set -e

echo "=== Deploying Patroni RTO Optimization ==="
echo "Target: <30s Database RTO"
echo "Date: $(date)"
echo

# Configuration paths
PATRONI_CONFIG="infrastructure/database/patroni-config.yml"
CLUSTER_CONFIG="infrastructure/database/postgresql-cluster.yml"
BACKUP_DIR="backups/rto_optimization_$(date +%Y%m%d_%H%M%S)"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Function to backup current configuration
backup_config() {
    echo "Backing up current configuration..."
    
    if [ -f "$PATRONI_CONFIG" ]; then
        cp "$PATRONI_CONFIG" "$BACKUP_DIR/patroni-config.yml.backup"
        echo "✓ Patroni config backed up"
    fi
    
    if [ -f "$CLUSTER_CONFIG" ]; then
        cp "$CLUSTER_CONFIG" "$BACKUP_DIR/postgresql-cluster.yml.backup"
        echo "✓ Cluster config backed up"
    fi
    
    echo "Backup location: $BACKUP_DIR"
}

# Function to validate configuration
validate_config() {
    echo "Validating optimized configuration..."
    
    # Check required optimization parameters
    local required_params=(
        "loop_wait: 5"
        "retry_timeout: 15"
        "ttl: 15"
        "failover_timeout: 30"
        "switchover_timeout: 30"
    )
    
    for param in "${required_params[@]}"; do
        if ! grep -q "$param" "$PATRONI_CONFIG"; then
            echo "ERROR: Required parameter not found: $param"
            exit 1
        fi
    done
    
    echo "✓ Configuration validated"
}

# Function to restart services
restart_services() {
    echo "Restarting Patroni services to apply optimization..."
    
    # Check if services are running
    if docker ps | grep -q "patroni-primary\|patroni-standby"; then
        echo "Stopping Patroni services..."
        
        # Stop Patroni containers
        docker stop patroni-primary patroni-standby 2>/dev/null || true
        
        # Wait for clean shutdown
        sleep 10
        
        # Start services
        echo "Starting optimized Patroni services..."
        docker start patroni-primary patroni-standby
        
        # Wait for services to initialize
        echo "Waiting for services to initialize..."
        sleep 30
        
        # Verify services are healthy
        local retries=0
        while [ $retries -lt 12 ]; do
            if curl -s -f http://localhost:8008/health > /dev/null && \
               curl -s -f http://localhost:8009/health > /dev/null; then
                echo "✓ Services restarted successfully"
                return 0
            fi
            
            echo "Waiting for services to become healthy..."
            sleep 10
            retries=$((retries + 1))
        done
        
        echo "ERROR: Services did not become healthy within timeout"
        exit 1
    else
        echo "No Patroni services found running. Please start services manually."
    fi
}

# Function to verify optimization
verify_optimization() {
    echo "Verifying RTO optimization..."
    
    # Check cluster status
    local cluster_status=$(curl -s http://localhost:8008/cluster)
    
    if [ -z "$cluster_status" ]; then
        echo "ERROR: Could not retrieve cluster status"
        exit 1
    fi
    
    # Check if we have a leader
    if ! echo "$cluster_status" | grep -q '"role": "Leader"'; then
        echo "ERROR: No leader found in cluster"
        exit 1
    fi
    
    # Test database connectivity
    python3 -c "
import psycopg2
import time

try:
    start = time.time()
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        database='grandmodel',
        user='grandmodel',
        password='your_password_here',
        connect_timeout=5
    )
    cursor = conn.cursor()
    cursor.execute('SELECT 1')
    cursor.fetchone()
    conn.close()
    
    response_time = (time.time() - start) * 1000
    print(f'Database response time: {response_time:.2f}ms')
    
    if response_time > 1000:
        print('WARNING: High response time detected')
    else:
        print('✓ Database connectivity verified')
        
except Exception as e:
    print(f'ERROR: Database connectivity test failed: {e}')
    exit(1)
"
    
    echo "✓ Optimization verification completed"
}

# Function to display optimization summary
display_summary() {
    echo
    echo "=== OPTIMIZATION SUMMARY ==="
    echo "Applied optimizations:"
    echo "  • loop_wait: 10s → 5s (50% faster detection)"
    echo "  • retry_timeout: 30s → 15s (50% faster retries)"
    echo "  • ttl: 30s → 15s (50% faster consensus)"
    echo "  • failover_timeout: added 30s limit"
    echo "  • switchover_timeout: added 30s limit"
    echo "  • master_start_timeout: 300s → 60s (80% faster startup)"
    echo "  • archive_timeout: 60s → 30s (50% faster archiving)"
    echo
    echo "Expected improvements:"
    echo "  • Faster failure detection (5s vs 10s)"
    echo "  • Quicker leader election (15s vs 30s TTL)"
    echo "  • Reduced failover time (<30s target)"
    echo "  • Better performance during network partitions"
    echo
    echo "Backup location: $BACKUP_DIR"
    echo "Configuration: $PATRONI_CONFIG"
    echo
}

# Function to rollback on failure
rollback() {
    echo "ERROR: Deployment failed. Rolling back..."
    
    if [ -f "$BACKUP_DIR/patroni-config.yml.backup" ]; then
        cp "$BACKUP_DIR/patroni-config.yml.backup" "$PATRONI_CONFIG"
        echo "✓ Patroni config restored"
    fi
    
    if [ -f "$BACKUP_DIR/postgresql-cluster.yml.backup" ]; then
        cp "$BACKUP_DIR/postgresql-cluster.yml.backup" "$CLUSTER_CONFIG"
        echo "✓ Cluster config restored"
    fi
    
    echo "Rollback completed. Please restart services manually."
    exit 1
}

# Set up error handling
trap rollback ERR

# Main deployment process
main() {
    echo "Starting deployment of RTO optimization..."
    
    # Pre-deployment checks
    if [ ! -f "$PATRONI_CONFIG" ]; then
        echo "ERROR: Patroni configuration file not found: $PATRONI_CONFIG"
        exit 1
    fi
    
    # Deployment steps
    backup_config
    validate_config
    restart_services
    verify_optimization
    display_summary
    
    echo "=== DEPLOYMENT COMPLETE ==="
    echo "RTO optimization has been successfully deployed."
    echo "Run test_rto_optimization.sh to validate the improvements."
    echo
}

# Execute main function
main "$@"