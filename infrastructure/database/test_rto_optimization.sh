#!/bin/bash

# Test Script for Patroni RTO Optimization
# Tests the optimized configuration to ensure <30s RTO target is achieved

set -e

echo "=== Patroni RTO Optimization Test Suite ==="
echo "Target: <30s Database RTO"
echo "Date: $(date)"
echo

# Configuration
TEST_CONFIG="infrastructure/database/rto_config.json"
RESULTS_DIR="test_results/rto_$(date +%Y%m%d_%H%M%S)"
PATRONI_CONFIG="infrastructure/database/patroni-config.yml"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Function to check if services are running
check_services() {
    echo "Checking service status..."
    
    # Check if Docker containers are running
    if ! docker ps | grep -q "postgres-primary"; then
        echo "ERROR: postgres-primary container not running"
        exit 1
    fi
    
    if ! docker ps | grep -q "postgres-standby"; then
        echo "ERROR: postgres-standby container not running"
        exit 1
    fi
    
    if ! docker ps | grep -q "patroni-primary"; then
        echo "ERROR: patroni-primary container not running"
        exit 1
    fi
    
    if ! docker ps | grep -q "patroni-standby"; then
        echo "ERROR: patroni-standby container not running"
        exit 1
    fi
    
    echo "✓ All services are running"
}

# Function to validate Patroni configuration
validate_patroni_config() {
    echo "Validating Patroni configuration..."
    
    # Check if optimized values are present
    if ! grep -q "loop_wait: 5" "$PATRONI_CONFIG"; then
        echo "ERROR: loop_wait not set to 5s"
        exit 1
    fi
    
    if ! grep -q "retry_timeout: 15" "$PATRONI_CONFIG"; then
        echo "ERROR: retry_timeout not set to 15s"
        exit 1
    fi
    
    if ! grep -q "ttl: 15" "$PATRONI_CONFIG"; then
        echo "ERROR: TTL not set to 15s"
        exit 1
    fi
    
    if ! grep -q "failover_timeout: 30" "$PATRONI_CONFIG"; then
        echo "ERROR: failover_timeout not set to 30s"
        exit 1
    fi
    
    echo "✓ Patroni configuration validated"
}

# Function to test cluster health
test_cluster_health() {
    echo "Testing cluster health..."
    
    # Check Patroni API endpoints
    if ! curl -s -f http://localhost:8008/health > /dev/null; then
        echo "ERROR: Primary Patroni API not responding"
        exit 1
    fi
    
    if ! curl -s -f http://localhost:8009/health > /dev/null; then
        echo "ERROR: Standby Patroni API not responding"
        exit 1
    fi
    
    # Check cluster status
    CLUSTER_STATUS=$(curl -s http://localhost:8008/cluster)
    echo "$CLUSTER_STATUS" | python3 -m json.tool > "$RESULTS_DIR/cluster_status.json"
    
    # Verify we have a leader
    if ! echo "$CLUSTER_STATUS" | grep -q '"role": "Leader"'; then
        echo "ERROR: No leader found in cluster"
        exit 1
    fi
    
    echo "✓ Cluster health checks passed"
}

# Function to run baseline performance test
run_baseline_test() {
    echo "Running baseline performance test..."
    
    # Test database connectivity and response times
    python3 -c "
import psycopg2
import time
import json

results = []
for i in range(10):
    start = time.time()
    try:
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='grandmodel',
            user='grandmodel',
            password='your_password_here',
            connect_timeout=5
        )
        cursor = conn.cursor()
        cursor.execute('SELECT 1, current_timestamp')
        cursor.fetchone()
        conn.close()
        
        response_time = (time.time() - start) * 1000
        results.append({'test': i+1, 'response_time_ms': response_time, 'success': True})
    except Exception as e:
        results.append({'test': i+1, 'response_time_ms': -1, 'success': False, 'error': str(e)})
    
    time.sleep(1)

with open('$RESULTS_DIR/baseline_performance.json', 'w') as f:
    json.dump(results, f, indent=2)

# Calculate average response time
successful_tests = [r for r in results if r['success']]
if successful_tests:
    avg_response = sum(r['response_time_ms'] for r in successful_tests) / len(successful_tests)
    print(f'Average response time: {avg_response:.2f}ms')
    print(f'Success rate: {len(successful_tests)/len(results)*100:.1f}%')
"
    
    echo "✓ Baseline performance test completed"
}

# Function to run automated failover tests
run_failover_tests() {
    echo "Running automated failover tests..."
    
    # Run graceful failover test
    echo "  Running graceful failover test..."
    python3 infrastructure/database/test_failover.py \
        --config "$TEST_CONFIG" \
        --single-test \
        --test-type graceful \
        --output "$RESULTS_DIR/graceful_failover.json" \
        > "$RESULTS_DIR/graceful_failover.log" 2>&1
    
    # Wait for cluster to stabilize
    echo "  Waiting for cluster to stabilize..."
    sleep 30
    
    # Run crash failover test
    echo "  Running crash failover test..."
    python3 infrastructure/database/test_failover.py \
        --config "$TEST_CONFIG" \
        --single-test \
        --test-type crash \
        --output "$RESULTS_DIR/crash_failover.json" \
        > "$RESULTS_DIR/crash_failover.log" 2>&1
    
    echo "✓ Failover tests completed"
}

# Function to run continuous monitoring test
run_monitoring_test() {
    echo "Running 5-minute continuous monitoring test..."
    
    # Start RTO monitor in background
    python3 infrastructure/database/rto_monitor.py \
        --config "$TEST_CONFIG" \
        --export "$RESULTS_DIR/monitoring_results.json" &
    
    MONITOR_PID=$!
    
    # Let it run for 5 minutes
    sleep 300
    
    # Stop monitor
    kill $MONITOR_PID 2>/dev/null || true
    
    echo "✓ Continuous monitoring test completed"
}

# Function to analyze results
analyze_results() {
    echo "Analyzing test results..."
    
    python3 -c "
import json
import os
import glob

results_dir = '$RESULTS_DIR'
summary = {
    'test_timestamp': '$(date -Iseconds)',
    'target_rto_seconds': 30,
    'configuration_optimizations': {
        'loop_wait': '5s (reduced from 10s)',
        'retry_timeout': '15s (reduced from 30s)',
        'ttl': '15s (reduced from 30s)',
        'failover_timeout': '30s (new)',
        'switchover_timeout': '30s (new)'
    },
    'test_results': {}
}

# Analyze failover tests
for test_type in ['graceful', 'crash']:
    test_file = f'{results_dir}/{test_type}_failover.json'
    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            data = json.load(f)
            
        rto_stats = data.get('rto_statistics', {})
        test_summary = data.get('test_summary', {})
        
        summary['test_results'][test_type] = {
            'rto_seconds': rto_stats.get('average_rto_seconds', 0),
            'success': test_summary.get('success_rate', 0) == 100,
            'target_achieved': rto_stats.get('average_rto_seconds', 999) < 30
        }

# Analyze monitoring results
monitor_file = f'{results_dir}/monitoring_results.json'
if os.path.exists(monitor_file):
    with open(monitor_file, 'r') as f:
        data = json.load(f)
        
    stats = data.get('statistics', {})
    summary['monitoring_results'] = {
        'uptime_percentage': stats.get('uptime_percentage', 0),
        'average_response_time_ms': stats.get('average_response_time_ms', 0),
        'total_events': stats.get('total_events', 0)
    }

# Calculate overall success
overall_success = True
for test_type, results in summary['test_results'].items():
    if not results.get('target_achieved', False):
        overall_success = False
        break

summary['overall_success'] = overall_success
summary['target_achieved'] = overall_success

# Save summary
with open(f'{results_dir}/test_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Print summary
print('=== TEST SUMMARY ===')
print(f'Target RTO: {summary[\"target_rto_seconds\"]}s')
print(f'Overall Success: {\"YES\" if summary[\"overall_success\"] else \"NO\"}')
print()

for test_type, results in summary['test_results'].items():
    print(f'{test_type.title()} Failover:')
    print(f'  RTO: {results[\"rto_seconds\"]:.2f}s')
    print(f'  Target Achieved: {\"YES\" if results[\"target_achieved\"] else \"NO\"}')
    print()

if 'monitoring_results' in summary:
    mon = summary['monitoring_results']
    print(f'Monitoring Results:')
    print(f'  Uptime: {mon[\"uptime_percentage\"]:.2f}%')
    print(f'  Avg Response: {mon[\"average_response_time_ms\"]:.2f}ms')
    print(f'  Total Events: {mon[\"total_events\"]}')
"
    
    echo "✓ Results analysis completed"
}

# Function to generate recommendations
generate_recommendations() {
    echo "Generating optimization recommendations..."
    
    python3 -c "
import json
import os

results_dir = '$RESULTS_DIR'
summary_file = f'{results_dir}/test_summary.json'

if os.path.exists(summary_file):
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    recommendations = []
    
    # Check if target was achieved
    if not summary.get('target_achieved', False):
        recommendations.append('CRITICAL: RTO target not achieved. Consider further optimization.')
    
    # Check individual test results
    for test_type, results in summary.get('test_results', {}).items():
        rto = results.get('rto_seconds', 0)
        if rto > 30:
            recommendations.append(f'{test_type.title()} failover RTO ({rto:.2f}s) exceeds target.')
        elif rto > 25:
            recommendations.append(f'{test_type.title()} failover RTO ({rto:.2f}s) is close to target.')
        elif rto > 20:
            recommendations.append(f'{test_type.title()} failover RTO ({rto:.2f}s) has room for improvement.')
    
    # Performance recommendations
    if summary.get('test_results', {}).get('graceful', {}).get('rto_seconds', 0) > 20:
        recommendations.append('Consider reducing loop_wait to 3s for faster detection.')
    
    if summary.get('test_results', {}).get('crash', {}).get('rto_seconds', 0) > 25:
        recommendations.append('Consider reducing TTL to 10s for faster consensus.')
    
    # Monitoring recommendations
    mon = summary.get('monitoring_results', {})
    if mon.get('uptime_percentage', 100) < 99.9:
        recommendations.append('Investigate connectivity issues affecting uptime.')
    
    if mon.get('average_response_time_ms', 0) > 100:
        recommendations.append('Database response time is high. Consider performance tuning.')
    
    # General recommendations
    if summary.get('target_achieved', False):
        recommendations.append('Target achieved! Monitor in production and fine-tune as needed.')
    else:
        recommendations.append('Consider additional optimizations: faster storage, network tuning.')
    
    # Save recommendations
    with open(f'{results_dir}/recommendations.txt', 'w') as f:
        f.write('PATRONI RTO OPTIMIZATION RECOMMENDATIONS\n')
        f.write('=' * 50 + '\n\n')
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                f.write(f'{i}. {rec}\n')
        else:
            f.write('No specific recommendations. Configuration appears optimal.\n')
    
    # Print recommendations
    print('RECOMMENDATIONS:')
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f'{i}. {rec}')
    else:
        print('No specific recommendations. Configuration appears optimal.')
"
    
    echo "✓ Recommendations generated"
}

# Main execution
main() {
    echo "Starting RTO optimization test suite..."
    
    # Pre-flight checks
    check_services
    validate_patroni_config
    test_cluster_health
    
    # Run tests
    run_baseline_test
    run_failover_tests
    run_monitoring_test
    
    # Analyze and report
    analyze_results
    generate_recommendations
    
    echo
    echo "=== TEST COMPLETE ==="
    echo "Results saved to: $RESULTS_DIR"
    echo "View summary: cat $RESULTS_DIR/test_summary.json"
    echo "View recommendations: cat $RESULTS_DIR/recommendations.txt"
}

# Execute main function
main "$@"