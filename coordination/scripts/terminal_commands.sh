#!/bin/bash
# Terminal Coordination Commands
# Quick commands for terminal coordination operations

COORDINATION_DIR="/home/QuantNova/GrandModel/coordination"
SCRIPTS_DIR="$COORDINATION_DIR/scripts"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check if coordination system is running
check_coordination_system() {
    if pgrep -f "coordination_master.py" > /dev/null; then
        print_status $GREEN "✅ Coordination system is running"
        return 0
    else
        print_status $RED "❌ Coordination system is not running"
        return 1
    fi
}

# Start coordination system
start_coordination() {
    print_status $BLUE "🚀 Starting coordination system..."
    python3 "$SCRIPTS_DIR/coordination_master.py" --start
}

# Stop coordination system
stop_coordination() {
    print_status $BLUE "🛑 Stopping coordination system..."
    python3 "$SCRIPTS_DIR/coordination_master.py" --stop
}

# Get system status
system_status() {
    print_status $BLUE "📊 Getting system status..."
    python3 "$SCRIPTS_DIR/coordination_master.py" --status
}

# Terminal 1 specific commands
terminal1_status() {
    print_status $BLUE "📋 Terminal 1 Status:"
    python3 "$SCRIPTS_DIR/update_terminal1_status.py" --report
}

terminal1_update() {
    local component=$1
    local progress=$2
    local status=$3
    
    if [ -z "$component" ]; then
        echo "Usage: terminal1_update <component> [progress] [status]"
        echo "Components: risk_management, execution_engine, xai_explanations"
        return 1
    fi
    
    local args="--component $component"
    [ -n "$progress" ] && args="$args --progress $progress"
    [ -n "$status" ] && args="$args --status $status"
    
    print_status $BLUE "📝 Updating Terminal 1 component: $component"
    python3 "$SCRIPTS_DIR/update_terminal1_status.py" $args
}

# Terminal 2 specific commands
terminal2_status() {
    print_status $BLUE "📋 Terminal 2 Status:"
    python3 "$SCRIPTS_DIR/update_terminal2_status.py" --report
}

terminal2_update() {
    local component=$1
    local progress=$2
    local status=$3
    
    if [ -z "$component" ]; then
        echo "Usage: terminal2_update <component> [progress] [status]"
        echo "Components: strategic_training, tactical_training"
        return 1
    fi
    
    local args="--component $component"
    [ -n "$progress" ] && args="$args --progress $progress"
    [ -n "$status" ] && args="$args --status $status"
    
    print_status $BLUE "📝 Updating Terminal 2 component: $component"
    python3 "$SCRIPTS_DIR/update_terminal2_status.py" $args
}

# Dependency management
check_dependencies() {
    print_status $BLUE "🔗 Checking dependencies..."
    python3 "$SCRIPTS_DIR/check_dependencies.py" --all
}

wait_for_dependency() {
    local dependency=$1
    local timeout=${2:-2}  # Default 2 hours
    
    if [ -z "$dependency" ]; then
        echo "Usage: wait_for_dependency <dependency> [timeout_hours]"
        echo "Dependencies: strategic_models, tactical_models, risk_models, execution_models"
        return 1
    fi
    
    print_status $YELLOW "⏳ Waiting for dependency: $dependency (timeout: ${timeout}h)"
    python3 "$SCRIPTS_DIR/check_dependencies.py" --wait "$dependency" --timeout "$timeout"
}

check_terminal1_ready() {
    print_status $BLUE "🔍 Checking Terminal 1 readiness..."
    python3 "$SCRIPTS_DIR/check_dependencies.py" --terminal1
}

check_terminal2_ready() {
    print_status $BLUE "🔍 Checking Terminal 2 readiness..."
    python3 "$SCRIPTS_DIR/check_dependencies.py" --terminal2
}

# Milestone management
sync_milestones() {
    print_status $BLUE "🎯 Syncing milestones..."
    python3 "$SCRIPTS_DIR/sync_milestones.py" --sync --report
}

check_blocking_milestones() {
    print_status $BLUE "🚧 Checking blocking milestones..."
    python3 "$SCRIPTS_DIR/sync_milestones.py" --blocking
}

milestone_report() {
    print_status $BLUE "📊 Generating milestone report..."
    python3 "$SCRIPTS_DIR/sync_milestones.py" --report
}

# Testing and validation
run_integration_tests() {
    print_status $BLUE "🧪 Running integration tests..."
    python3 "$SCRIPTS_DIR/run_integration_tests.py" --all
}

check_system_readiness() {
    print_status $BLUE "✅ Checking system readiness..."
    python3 "$SCRIPTS_DIR/run_integration_tests.py" --readiness
}

test_notebooks() {
    print_status $BLUE "📓 Testing notebook execution..."
    python3 "$SCRIPTS_DIR/run_integration_tests.py" --notebooks
}

test_communication() {
    print_status $BLUE "📡 Testing terminal communication..."
    python3 "$SCRIPTS_DIR/run_integration_tests.py" --communication
}

# Quick status check
quick_status() {
    print_status $BLUE "⚡ Quick Status Check"
    echo "===================="
    
    # Check coordination system
    check_coordination_system
    
    # Check dependencies
    echo ""
    print_status $YELLOW "Dependencies:"
    python3 "$SCRIPTS_DIR/check_dependencies.py" --all | head -20
    
    # Check milestones
    echo ""
    print_status $YELLOW "Milestones:"
    python3 "$SCRIPTS_DIR/sync_milestones.py" --report | head -10
}

# Emergency procedures
emergency_stop() {
    print_status $RED "🚨 EMERGENCY STOP"
    python3 "$SCRIPTS_DIR/coordination_master.py" --emergency-stop
}

reset_coordination() {
    print_status $YELLOW "🔄 Resetting coordination system..."
    stop_coordination
    sleep 5
    start_coordination
}

# Utility functions
show_logs() {
    local lines=${1:-50}
    print_status $BLUE "📄 Recent logs (last $lines lines):"
    find "$COORDINATION_DIR/documentation/progress_logs" -name "*.json" -type f -exec ls -t {} + | head -1 | xargs cat | tail -$lines
}

show_coordination_help() {
    echo "Terminal Coordination Commands"
    echo "============================="
    echo ""
    echo "🚀 System Control:"
    echo "  start_coordination          - Start coordination system"
    echo "  stop_coordination           - Stop coordination system"
    echo "  system_status               - Get system status"
    echo "  reset_coordination          - Reset coordination system"
    echo "  emergency_stop              - Emergency stop all activities"
    echo ""
    echo "📋 Terminal Management:"
    echo "  terminal1_status            - Get Terminal 1 status"
    echo "  terminal1_update <comp> [progress] [status] - Update Terminal 1"
    echo "  terminal2_status            - Get Terminal 2 status"
    echo "  terminal2_update <comp> [progress] [status] - Update Terminal 2"
    echo ""
    echo "🔗 Dependencies:"
    echo "  check_dependencies          - Check all dependencies"
    echo "  wait_for_dependency <dep>   - Wait for specific dependency"
    echo "  check_terminal1_ready       - Check Terminal 1 readiness"
    echo "  check_terminal2_ready       - Check Terminal 2 readiness"
    echo ""
    echo "🎯 Milestones:"
    echo "  sync_milestones             - Sync all milestones"
    echo "  check_blocking_milestones   - Check blocking milestones"
    echo "  milestone_report            - Generate milestone report"
    echo ""
    echo "🧪 Testing:"
    echo "  run_integration_tests       - Run all integration tests"
    echo "  check_system_readiness      - Check system readiness"
    echo "  test_notebooks              - Test notebook execution"
    echo "  test_communication          - Test terminal communication"
    echo ""
    echo "⚡ Quick Commands:"
    echo "  quick_status                - Quick status overview"
    echo "  show_logs [lines]           - Show recent logs"
    echo "  show_coordination_help      - Show this help"
    echo ""
    echo "📝 Examples:"
    echo "  terminal1_update risk_management 50 in_progress"
    echo "  terminal2_update strategic_training 75 in_progress"
    echo "  wait_for_dependency strategic_models 2"
}

# Export functions so they can be used from other terminals
export -f start_coordination stop_coordination system_status
export -f terminal1_status terminal1_update terminal2_status terminal2_update
export -f check_dependencies wait_for_dependency check_terminal1_ready check_terminal2_ready
export -f sync_milestones check_blocking_milestones milestone_report
export -f run_integration_tests check_system_readiness test_notebooks test_communication
export -f quick_status emergency_stop reset_coordination show_logs show_coordination_help

# Main command dispatcher
case "${1:-help}" in
    "start")
        start_coordination
        ;;
    "stop")
        stop_coordination
        ;;
    "status")
        system_status
        ;;
    "t1-status")
        terminal1_status
        ;;
    "t2-status")
        terminal2_status
        ;;
    "deps")
        check_dependencies
        ;;
    "milestones")
        sync_milestones
        ;;
    "tests")
        run_integration_tests
        ;;
    "quick")
        quick_status
        ;;
    "emergency")
        emergency_stop
        ;;
    "reset")
        reset_coordination
        ;;
    "help"|*)
        show_coordination_help
        ;;
esac