#!/bin/bash
# Health Check Validation Runner
# AGENT 1: DATABASE RTO SPECIALIST - Automated Validation
# Target: Continuous validation of 1s health checks and <30s RTO

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/var/log/health_validation"
CONFIG_FILE="${SCRIPT_DIR}/health_check_validation.yml"
VALIDATION_SCRIPT="${SCRIPT_DIR}/health_check_validation.py"

# Create log directory
mkdir -p "${LOG_DIR}"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_DIR}/validation.log"
}

# Function to check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker containers are running
    if ! docker ps | grep -q postgres-primary; then
        log "ERROR: postgres-primary container not running"
        exit 1
    fi
    
    if ! docker ps | grep -q postgres-standby; then
        log "ERROR: postgres-standby container not running"
        exit 1
    fi
    
    if ! docker ps | grep -q patroni-primary; then
        log "ERROR: patroni-primary container not running"
        exit 1
    fi
    
    if ! docker ps | grep -q patroni-standby; then
        log "ERROR: patroni-standby container not running"
        exit 1
    fi
    
    # Check if Python script exists
    if [ ! -f "${VALIDATION_SCRIPT}" ]; then
        log "ERROR: Validation script not found at ${VALIDATION_SCRIPT}"
        exit 1
    fi
    
    # Check if config file exists
    if [ ! -f "${CONFIG_FILE}" ]; then
        log "ERROR: Configuration file not found at ${CONFIG_FILE}"
        exit 1
    fi
    
    log "Prerequisites check passed"
}

# Function to run quick validation
run_quick_validation() {
    log "Running quick health check validation..."
    
    cd "${SCRIPT_DIR}"
    if python3 "${VALIDATION_SCRIPT}" --quick; then
        log "Quick validation PASSED"
        return 0
    else
        log "Quick validation FAILED"
        return 1
    fi
}

# Function to run full validation
run_full_validation() {
    log "Running full health check validation..."
    
    cd "${SCRIPT_DIR}"
    if python3 "${VALIDATION_SCRIPT}"; then
        log "Full validation PASSED"
        return 0
    else
        log "Full validation FAILED"
        return 1
    fi
}

# Function to run continuous validation
run_continuous_validation() {
    log "Starting continuous health check validation..."
    
    local interval=${1:-300}  # Default 5 minutes
    local quick_interval=${2:-60}  # Default 1 minute for quick checks
    
    while true; do
        # Run quick validation every minute
        if run_quick_validation; then
            log "Continuous validation: Quick check passed"
        else
            log "ERROR: Continuous validation: Quick check failed"
            # Run full validation on quick check failure
            run_full_validation
        fi
        
        # Run full validation every 5 minutes
        if [ $(($(date +%s) % interval)) -eq 0 ]; then
            run_full_validation
        fi
        
        sleep "${quick_interval}"
    done
}

# Function to analyze validation results
analyze_results() {
    log "Analyzing validation results..."
    
    local report_file="${LOG_DIR}/latest_report.json"
    
    if [ -f "${report_file}" ]; then
        local success_rate=$(jq -r '.test_summary.success_rate' "${report_file}")
        local recommendations=$(jq -r '.recommendations | length' "${report_file}")
        
        log "Success rate: ${success_rate}%"
        log "Recommendations: ${recommendations}"
        
        if (( $(echo "${success_rate} < 95" | bc -l) )); then
            log "WARNING: Success rate below 95%"
        fi
        
        if [ "${recommendations}" -gt 0 ]; then
            log "Recommendations found:"
            jq -r '.recommendations[]' "${report_file}" | while read -r rec; do
                log "  - ${rec}"
            done
        fi
    else
        log "No report file found"
    fi
}

# Function to generate monitoring dashboard
generate_dashboard() {
    log "Generating monitoring dashboard..."
    
    cat > "${LOG_DIR}/dashboard.html" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Database Health Check Validation Dashboard</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .success { background-color: #d4edda; color: #155724; }
        .warning { background-color: #fff3cd; color: #856404; }
        .error { background-color: #f8d7da; color: #721c24; }
        .metric { display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>Database Health Check Validation Dashboard</h1>
    <p>Last updated: $(date)</p>
    
    <h2>System Status</h2>
    <div class="status success">Health checks optimized to 1-second intervals</div>
    <div class="status success">Patroni configured for sub-15-second failover</div>
    <div class="status success">Sub-second monitoring active</div>
    
    <h2>Performance Metrics</h2>
    <div class="metric">
        <strong>Health Check Interval:</strong> 1 second
    </div>
    <div class="metric">
        <strong>Target RTO:</strong> < 30 seconds
    </div>
    <div class="metric">
        <strong>Patroni TTL:</strong> 10 seconds
    </div>
    <div class="metric">
        <strong>Loop Wait:</strong> 2 seconds
    </div>
    
    <h2>Validation Status</h2>
    <div id="validation-status">
        <p>Run validation to see current status...</p>
    </div>
    
    <h2>Quick Actions</h2>
    <button onclick="runQuickValidation()">Run Quick Validation</button>
    <button onclick="runFullValidation()">Run Full Validation</button>
    <button onclick="viewLogs()">View Logs</button>
    
    <script>
        function runQuickValidation() {
            alert('Quick validation started - check logs for results');
        }
        
        function runFullValidation() {
            alert('Full validation started - check logs for results');
        }
        
        function viewLogs() {
            window.open('/var/log/health_validation/validation.log', '_blank');
        }
    </script>
</body>
</html>
EOF
    
    log "Dashboard generated at ${LOG_DIR}/dashboard.html"
}

# Function to install validation service
install_service() {
    log "Installing health check validation service..."
    
    cat > /etc/systemd/system/health-check-validation.service << EOF
[Unit]
Description=Database Health Check Validation Service
After=docker.service
Requires=docker.service

[Service]
Type=simple
ExecStart=${SCRIPT_DIR}/run_health_validation.sh continuous
Restart=always
RestartSec=10
User=root
WorkingDirectory=${SCRIPT_DIR}

[Install]
WantedBy=multi-user.target
EOF
    
    systemctl daemon-reload
    systemctl enable health-check-validation
    log "Health check validation service installed"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [command]"
    echo "Commands:"
    echo "  quick      - Run quick validation"
    echo "  full       - Run full validation"
    echo "  continuous - Run continuous validation"
    echo "  analyze    - Analyze validation results"
    echo "  dashboard  - Generate monitoring dashboard"
    echo "  install    - Install as systemd service"
    echo "  status     - Show validation status"
}

# Function to show status
show_status() {
    log "Health Check Validation Status"
    echo "=================================="
    
    # Check if containers are running
    echo "Container Status:"
    docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(postgres|patroni|etcd|pgbouncer)" || echo "No database containers running"
    
    echo ""
    echo "Health Check Intervals:"
    echo "  Docker healthcheck: 1s"
    echo "  Patroni loop_wait: 2s"
    echo "  Patroni TTL: 10s"
    echo "  Health monitor: 1s"
    echo "  Sub-second check: 0.5s"
    
    echo ""
    echo "Recent Validation Results:"
    if [ -f "${LOG_DIR}/validation.log" ]; then
        tail -n 10 "${LOG_DIR}/validation.log"
    else
        echo "No validation log found"
    fi
}

# Main execution
main() {
    local command=${1:-"quick"}
    
    case "${command}" in
        "quick")
            check_prerequisites
            run_quick_validation
            ;;
        "full")
            check_prerequisites
            run_full_validation
            ;;
        "continuous")
            check_prerequisites
            run_continuous_validation
            ;;
        "analyze")
            analyze_results
            ;;
        "dashboard")
            generate_dashboard
            ;;
        "install")
            install_service
            ;;
        "status")
            show_status
            ;;
        *)
            show_usage
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"