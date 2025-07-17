#!/bin/bash
# Comprehensive Load Testing Script for XAI Trading System
# Agent Epsilon - Production Performance Validation

set -euo pipefail

# Configuration
XAI_HOST="${XAI_HOST:-http://localhost:443}"
USERS_START="${USERS_START:-10}"
USERS_MAX="${USERS_MAX:-100}"
SPAWN_RATE="${SPAWN_RATE:-5}"
TEST_DURATION="${TEST_DURATION:-300}"  # 5 minutes
RESULTS_DIR="${RESULTS_DIR:-./load_test_results}"

echo "🔥 XAI Trading System Load Testing Suite"
echo "========================================"
echo "Target: $XAI_HOST"
echo "Users: $USERS_START -> $USERS_MAX (spawn rate: $SPAWN_RATE/s)"
echo "Duration: ${TEST_DURATION}s"
echo "Results: $RESULTS_DIR"
echo "========================================"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Get timestamp for this test run
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
TEST_ID="xai_load_test_${TIMESTAMP}"

echo "📋 Test ID: $TEST_ID"

# Function to run a specific load test scenario
run_test_scenario() {
    local scenario_name="$1"
    local users="$2"
    local duration="$3"
    local description="$4"
    
    echo ""
    echo "🧪 Running Scenario: $scenario_name"
    echo "   Description: $description"
    echo "   Users: $users, Duration: ${duration}s"
    echo "----------------------------------------"
    
    local output_file="$RESULTS_DIR/${TEST_ID}_${scenario_name}"
    
    locust \
        -f locustfile.py \
        --host="$XAI_HOST" \
        --users="$users" \
        --spawn-rate="$SPAWN_RATE" \
        --run-time="${duration}s" \
        --headless \
        --html="${output_file}.html" \
        --csv="${output_file}" \
        --logfile="${output_file}.log" \
        --loglevel=INFO \
        --print-stats \
        --only-summary \
        2>&1 | tee "${output_file}_console.log"
    
    echo "✅ Scenario $scenario_name completed"
    echo "   Results: ${output_file}.*"
}

# Function to check system health before testing
check_system_health() {
    echo "🏥 Checking system health before load testing..."
    
    # Check API endpoint
    if curl -f -s "$XAI_HOST/health" > /dev/null; then
        echo "✅ API endpoint is responsive"
    else
        echo "❌ API endpoint is not responsive"
        echo "   Please ensure the XAI system is running before load testing"
        exit 1
    fi
    
    # Check explanation endpoint
    if curl -f -s -X POST "$XAI_HOST/api/v1/explanations/generate" \
        -H "Content-Type: application/json" \
        -d '{"test": true}' > /dev/null 2>&1; then
        echo "✅ Explanation endpoint is accessible"
    else
        echo "⚠️  Explanation endpoint returned error (expected for test payload)"
    fi
    
    echo "✅ System health check completed"
}

# Function to analyze results
analyze_results() {
    echo ""
    echo "📊 Analyzing Load Test Results"
    echo "=============================="
    
    python3 - <<EOF
import pandas as pd
import json
import glob
import os

results_dir = "$RESULTS_DIR"
test_id = "$TEST_ID"

print(f"Analyzing results in {results_dir}")

# Find all CSV result files
csv_files = glob.glob(f"{results_dir}/{test_id}_*_stats.csv")

if not csv_files:
    print("❌ No result files found")
    exit(1)

print(f"Found {len(csv_files)} result files")

for csv_file in csv_files:
    scenario = csv_file.split('_')[-2]  # Extract scenario name
    print(f"\n📈 Scenario: {scenario}")
    print("-" * 40)
    
    try:
        df = pd.read_csv(csv_file)
        
        # Filter for explanation generation requests
        explanation_rows = df[df['Name'].str.contains('explanation', case=False, na=False)]
        
        if not explanation_rows.empty:
            for _, row in explanation_rows.iterrows():
                print(f"  Endpoint: {row['Name']}")
                print(f"    Requests: {row['Request Count']}")
                print(f"    Failures: {row['Failure Count']}")
                print(f"    Avg Response Time: {row['Average Response Time']:.2f}ms")
                print(f"    95th Percentile: {row['95%']:.2f}ms")
                print(f"    99th Percentile: {row['99%']:.2f}ms")
                print(f"    Max Response Time: {row['Max Response Time']:.2f}ms")
                
                # Check requirements
                if row['95%'] <= 100:
                    print(f"    ✅ PASSED: 95th percentile {row['95%']:.2f}ms <= 100ms")
                else:
                    print(f"    ❌ FAILED: 95th percentile {row['95%']:.2f}ms > 100ms")
        
        # Query performance
        query_rows = df[df['Name'].str.contains('query', case=False, na=False)]
        
        if not query_rows.empty:
            for _, row in query_rows.iterrows():
                print(f"  Query Endpoint: {row['Name']}")
                print(f"    Avg Response Time: {row['Average Response Time']:.2f}ms")
                print(f"    95th Percentile: {row['95%']:.2f}ms")
                
                if row['95%'] <= 2000:
                    print(f"    ✅ PASSED: 95th percentile {row['95%']:.2f}ms <= 2000ms")
                else:
                    print(f"    ❌ FAILED: 95th percentile {row['95%']:.2f}ms > 2000ms")
        
    except Exception as e:
        print(f"❌ Error analyzing {csv_file}: {e}")

print("\n🏁 Analysis Complete")
EOF
}

# Function to generate performance report
generate_report() {
    echo ""
    echo "📝 Generating Performance Report"
    echo "==============================="
    
    cat > "$RESULTS_DIR/${TEST_ID}_performance_report.md" <<EOF
# XAI Trading System Load Test Report

**Test ID:** $TEST_ID  
**Timestamp:** $(date)  
**Target System:** $XAI_HOST  
**Test Duration:** ${TEST_DURATION}s per scenario  

## Test Configuration

- **Users:** $USERS_START -> $USERS_MAX
- **Spawn Rate:** $SPAWN_RATE users/second
- **Test Scenarios:** Multiple performance scenarios
- **Requirements:**
  - Explanation latency: <100ms (95th percentile)
  - Query response time: <2 seconds (95th percentile)
  - System availability: >99.9%

## Results Summary

Results files generated:
EOF
    
    # List all generated files
    find "$RESULTS_DIR" -name "${TEST_ID}*" -type f | while read -r file; do
        echo "- $(basename "$file")" >> "$RESULTS_DIR/${TEST_ID}_performance_report.md"
    done
    
    cat >> "$RESULTS_DIR/${TEST_ID}_performance_report.md" <<EOF

## Key Metrics

For detailed metrics, see the HTML reports and CSV files.

### Performance Requirements Validation

- [ ] Explanation latency <100ms (95th percentile)
- [ ] Query response time <2s (95th percentile)  
- [ ] System availability >99.9%

### Recommendations

1. Monitor explanation latency under load
2. Optimize query response times if needed
3. Scale horizontally if throughput requirements increase
4. Implement caching for frequently requested explanations

---
Generated by Agent Epsilon Load Testing Suite
EOF

    echo "✅ Performance report generated: $RESULTS_DIR/${TEST_ID}_performance_report.md"
}

# Main test execution
main() {
    echo "🚀 Starting XAI Trading System Load Test Suite"
    
    # Pre-test health check
    check_system_health
    
    echo ""
    echo "🎯 Running Load Test Scenarios"
    echo "=============================="
    
    # Scenario 1: Baseline performance test
    run_test_scenario \
        "baseline" \
        "$USERS_START" \
        "120" \
        "Baseline performance with low user load"
    
    # Scenario 2: Explanation latency focus
    run_test_scenario \
        "explanation_focus" \
        "50" \
        "180" \
        "Focus on explanation generation latency"
    
    # Scenario 3: Query performance focus  
    run_test_scenario \
        "query_focus" \
        "30" \
        "120" \
        "Focus on complex query performance"
    
    # Scenario 4: Peak load test
    run_test_scenario \
        "peak_load" \
        "$USERS_MAX" \
        "$TEST_DURATION" \
        "Peak load simulation"
    
    # Scenario 5: Sustained load test
    run_test_scenario \
        "sustained_load" \
        "75" \
        "600" \
        "Sustained load over 10 minutes"
    
    # Analyze all results
    analyze_results
    
    # Generate comprehensive report
    generate_report
    
    echo ""
    echo "🏆 Load Testing Complete!"
    echo "========================"
    echo "Results available in: $RESULTS_DIR"
    echo ""
    echo "Key files:"
    echo "- Performance report: ${TEST_ID}_performance_report.md"
    echo "- HTML dashboards: ${TEST_ID}_*.html"
    echo "- Raw data: ${TEST_ID}_*.csv"
    echo ""
    echo "Next steps:"
    echo "1. Review HTML reports for detailed metrics"
    echo "2. Analyze CSV data for trends"
    echo "3. Validate performance requirements"
    echo "4. Optimize system if needed"
}

# Check dependencies
check_dependencies() {
    echo "🔍 Checking dependencies..."
    
    if ! command -v locust &> /dev/null; then
        echo "❌ Locust is not installed"
        echo "   Install with: pip install locust"
        exit 1
    fi
    
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python 3 is not available"
        exit 1
    fi
    
    # Check for required Python packages
    python3 -c "import pandas, numpy, websocket" 2>/dev/null || {
        echo "❌ Required Python packages missing"
        echo "   Install with: pip install pandas numpy websocket-client"
        exit 1
    }
    
    echo "✅ All dependencies available"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    check_dependencies
    main "$@"
fi