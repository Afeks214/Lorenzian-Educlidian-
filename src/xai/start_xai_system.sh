#!/bin/bash

# XAI Trading System Startup Script
# Starts both frontend and backend components

set -e

echo "🚀 Starting XAI Trading System..."
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_error "Node.js is required but not installed"
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check npm
    if ! command -v npm &> /dev/null; then
        print_error "npm is required but not installed"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    # Install frontend dependencies
    if [ -f "frontend/package.json" ]; then
        print_status "Installing frontend dependencies..."
        cd frontend
        npm install
        cd ..
        print_success "Frontend dependencies installed"
    else
        print_warning "Frontend package.json not found, skipping frontend dependencies"
    fi
    
    # Install backend dependencies
    print_status "Installing backend dependencies..."
    pip3 install fastapi uvicorn websockets pydantic python-multipart
    print_success "Backend dependencies installed"
}

# Setup environment
setup_environment() {
    print_status "Setting up environment..."
    
    # Create frontend .env if it doesn't exist
    if [ ! -f "frontend/.env" ] && [ -f "frontend/.env.example" ]; then
        print_status "Creating frontend environment file..."
        cp frontend/.env.example frontend/.env
        print_success "Frontend .env created from example"
    fi
    
    print_success "Environment setup complete"
}

# Start backend server
start_backend() {
    print_status "Starting backend server..."
    
    cd backend
    python3 api/xai_api.py &
    BACKEND_PID=$!
    echo $BACKEND_PID > ../backend.pid
    cd ..
    
    print_success "Backend server started (PID: $BACKEND_PID)"
    print_status "Backend API available at: http://localhost:8001"
    print_status "API Documentation available at: http://localhost:8001/docs"
}

# Start frontend server
start_frontend() {
    print_status "Starting frontend server..."
    
    cd frontend
    npm start &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > ../frontend.pid
    cd ..
    
    print_success "Frontend server started (PID: $FRONTEND_PID)"
    print_status "Frontend application available at: http://localhost:3000"
}

# Wait for servers to be ready
wait_for_servers() {
    print_status "Waiting for servers to be ready..."
    
    # Wait for backend
    for i in {1..30}; do
        if curl -s http://localhost:8001/health > /dev/null 2>&1; then
            print_success "Backend server is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "Backend server failed to start within 30 seconds"
            exit 1
        fi
        sleep 1
    done
    
    # Wait for frontend
    for i in {1..60}; do
        if curl -s http://localhost:3000 > /dev/null 2>&1; then
            print_success "Frontend server is ready"
            break
        fi
        if [ $i -eq 60 ]; then
            print_warning "Frontend server may still be starting (this is normal for React apps)"
            break
        fi
        sleep 1
    done
}

# Cleanup function
cleanup() {
    print_status "Shutting down XAI Trading System..."
    
    # Kill backend
    if [ -f "backend.pid" ]; then
        BACKEND_PID=$(cat backend.pid)
        kill $BACKEND_PID 2>/dev/null || true
        rm backend.pid
        print_success "Backend server stopped"
    fi
    
    # Kill frontend
    if [ -f "frontend.pid" ]; then
        FRONTEND_PID=$(cat frontend.pid)
        kill $FRONTEND_PID 2>/dev/null || true
        rm frontend.pid
        print_success "Frontend server stopped"
    fi
    
    print_success "XAI Trading System shutdown complete"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Main execution
main() {
    # Change to script directory
    cd "$(dirname "$0")"
    
    # Parse command line arguments
    case "${1:-start}" in
        "start")
            check_prerequisites
            install_dependencies
            setup_environment
            start_backend
            sleep 2
            start_frontend
            wait_for_servers
            ;;
        "stop")
            cleanup
            ;;
        "restart")
            cleanup
            sleep 2
            main start
            ;;
        "status")
            if [ -f "backend.pid" ] && [ -f "frontend.pid" ]; then
                BACKEND_PID=$(cat backend.pid)
                FRONTEND_PID=$(cat frontend.pid)
                if kill -0 $BACKEND_PID 2>/dev/null && kill -0 $FRONTEND_PID 2>/dev/null; then
                    print_success "XAI Trading System is running"
                    print_status "Backend PID: $BACKEND_PID"
                    print_status "Frontend PID: $FRONTEND_PID"
                else
                    print_warning "XAI Trading System is partially running or stopped"
                fi
            else
                print_warning "XAI Trading System is not running"
            fi
            ;;
        "logs")
            print_status "Showing system logs..."
            # In production, this would tail actual log files
            print_status "Backend logs: Check console output or implement proper logging"
            print_status "Frontend logs: Check browser console"
            ;;
        "help"|*)
            echo "XAI Trading System Control Script"
            echo ""
            echo "Usage: $0 {start|stop|restart|status|logs|help}"
            echo ""
            echo "Commands:"
            echo "  start    - Start the XAI Trading System (default)"
            echo "  stop     - Stop the XAI Trading System"
            echo "  restart  - Restart the XAI Trading System"
            echo "  status   - Check system status"
            echo "  logs     - Show system logs"
            echo "  help     - Show this help message"
            echo ""
            echo "URLs:"
            echo "  Frontend: http://localhost:3000"
            echo "  Backend:  http://localhost:8001"
            echo "  API Docs: http://localhost:8001/docs"
            exit 0
            ;;
    esac
    
    if [ "${1:-start}" == "start" ]; then
        print_success "XAI Trading System started successfully!"
        echo ""
        echo "🎯 Access Points:"
        echo "   Frontend Application: http://localhost:3000"
        echo "   Backend API:         http://localhost:8001"
        echo "   API Documentation:   http://localhost:8001/docs"
        echo "   WebSocket:           ws://localhost:8001/ws/{client_id}"
        echo ""
        echo "🔧 Controls:"
        echo "   Stop system:    Ctrl+C or '$0 stop'"
        echo "   Restart system: '$0 restart'"
        echo "   Check status:   '$0 status'"
        echo ""
        echo "📝 Features:"
        echo "   ✅ ChatGPT-like AI trading explanations"
        echo "   ✅ Advanced analytics dashboard"
        echo "   ✅ Real-time WebSocket updates"
        echo "   ✅ Natural language queries"
        echo "   ✅ Mobile-responsive design"
        echo "   ✅ Voice input/output capabilities"
        echo ""
        print_status "System is running... Press Ctrl+C to stop"
        
        # Keep script running
        while true; do
            sleep 1
        done
    fi
}

# Run main function
main "$@"