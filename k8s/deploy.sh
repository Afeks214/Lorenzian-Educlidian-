#!/bin/bash
# GrandModel Kubernetes Deployment Script - Agent 5 Production Deployment
set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="grandmodel"
TIMEOUT="600s"
VERBOSE=${VERBOSE:-false}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if kubectl is available
check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_success "kubectl is available and cluster is reachable"
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    check_kubectl
    
    # Check required files
    local required_files=(
        "namespace.yaml"
        "rbac.yaml"
        "configmaps.yaml"
        "storage.yaml"
        "deployments.yaml"
        "services.yaml"
        "hpa.yaml"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "${SCRIPT_DIR}/${file}" ]]; then
            log_error "Required file ${file} not found"
            exit 1
        fi
    done
    
    log_success "All prerequisites met"
}

# Function to create namespace and basic resources
deploy_namespace() {
    log_info "Deploying namespace and basic resources..."
    
    kubectl apply -f "${SCRIPT_DIR}/namespace.yaml"
    
    # Wait for namespace to be ready
    kubectl wait --for=condition=Ready namespace/${NAMESPACE} --timeout=${TIMEOUT} || true
    
    log_success "Namespace deployed successfully"
}

# Function to create secrets
create_secrets() {
    log_info "Creating secrets..."
    
    # Check if secrets already exist
    if kubectl get secret postgres-secret -n ${NAMESPACE} &> /dev/null; then
        log_warning "postgres-secret already exists, skipping creation"
    else
        # Generate random passwords
        POSTGRES_PASSWORD=$(openssl rand -base64 32)
        kubectl create secret generic postgres-secret \
            -n ${NAMESPACE} \
            --from-literal=password="${POSTGRES_PASSWORD}"
        log_success "PostgreSQL secret created"
    fi
    
    if kubectl get secret redis-secret -n ${NAMESPACE} &> /dev/null; then
        log_warning "redis-secret already exists, skipping creation"
    else
        REDIS_PASSWORD=$(openssl rand -base64 32)
        kubectl create secret generic redis-secret \
            -n ${NAMESPACE} \
            --from-literal=password="${REDIS_PASSWORD}"
        log_success "Redis secret created"
    fi
    
    # Create TLS secrets (if certificates are available)
    if [[ -f "${SCRIPT_DIR}/../configs/ssl/tls.crt" && -f "${SCRIPT_DIR}/../configs/ssl/tls.key" ]]; then
        kubectl create secret tls grandmodel-tls \
            -n ${NAMESPACE} \
            --cert="${SCRIPT_DIR}/../configs/ssl/tls.crt" \
            --key="${SCRIPT_DIR}/../configs/ssl/tls.key" \
            --dry-run=client -o yaml | kubectl apply -f -
        log_success "TLS secret created"
    else
        log_warning "TLS certificates not found, skipping TLS secret creation"
    fi
}

# Function to deploy RBAC
deploy_rbac() {
    log_info "Deploying RBAC configuration..."
    
    kubectl apply -f "${SCRIPT_DIR}/rbac.yaml"
    
    log_success "RBAC deployed successfully"
}

# Function to deploy ConfigMaps
deploy_configmaps() {
    log_info "Deploying ConfigMaps..."
    
    kubectl apply -f "${SCRIPT_DIR}/configmaps.yaml"
    
    # Wait for ConfigMaps to be ready
    kubectl wait --for=condition=Ready configmap/grandmodel-config -n ${NAMESPACE} --timeout=${TIMEOUT} || true
    
    log_success "ConfigMaps deployed successfully"
}

# Function to deploy storage
deploy_storage() {
    log_info "Deploying storage resources..."
    
    kubectl apply -f "${SCRIPT_DIR}/storage.yaml"
    
    # Wait for PVCs to be bound
    log_info "Waiting for PVCs to be bound..."
    local pvcs=("models-pvc" "logs-pvc" "data-pvc" "metrics-pvc" "backup-pvc")
    
    for pvc in "${pvcs[@]}"; do
        if kubectl get pvc ${pvc} -n ${NAMESPACE} &> /dev/null; then
            kubectl wait --for=condition=Bound pvc/${pvc} -n ${NAMESPACE} --timeout=${TIMEOUT} || log_warning "PVC ${pvc} not bound within timeout"
        fi
    done
    
    log_success "Storage deployed successfully"
}

# Function to deploy applications
deploy_applications() {
    log_info "Deploying application services..."
    
    # Deploy services first
    kubectl apply -f "${SCRIPT_DIR}/services.yaml"
    
    # Deploy applications
    kubectl apply -f "${SCRIPT_DIR}/deployments.yaml"
    
    # Wait for deployments to be ready
    log_info "Waiting for deployments to be ready..."
    local deployments=("strategic-deployment" "tactical-deployment" "risk-deployment" "nginx-gateway")
    
    for deployment in "${deployments[@]}"; do
        if kubectl get deployment ${deployment} -n ${NAMESPACE} &> /dev/null; then
            kubectl wait --for=condition=Available deployment/${deployment} -n ${NAMESPACE} --timeout=${TIMEOUT}
            log_success "Deployment ${deployment} is ready"
        fi
    done
    
    log_success "Applications deployed successfully"
}

# Function to deploy autoscaling
deploy_autoscaling() {
    log_info "Deploying autoscaling configuration..."
    
    # Check if metrics server is available
    if kubectl get apiservice v1beta1.metrics.k8s.io -o jsonpath='{.status.conditions[?(@.type=="Available")].status}' | grep -q True; then
        kubectl apply -f "${SCRIPT_DIR}/hpa.yaml"
        log_success "Autoscaling deployed successfully"
    else
        log_warning "Metrics server not available, skipping HPA deployment"
        log_warning "Install metrics server to enable autoscaling: kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml"
    fi
}

# Function to verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check namespace
    kubectl get namespace ${NAMESPACE}
    
    # Check deployments
    kubectl get deployments -n ${NAMESPACE}
    
    # Check services
    kubectl get services -n ${NAMESPACE}
    
    # Check pods
    kubectl get pods -n ${NAMESPACE}
    
    # Check HPA (if deployed)
    kubectl get hpa -n ${NAMESPACE} 2>/dev/null || true
    
    # Health check
    log_info "Performing health checks..."
    local services=("strategic-service" "tactical-service" "risk-service")
    
    for service in "${services[@]}"; do
        if kubectl get service ${service} -n ${NAMESPACE} &> /dev/null; then
            local cluster_ip=$(kubectl get service ${service} -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}')
            log_info "Service ${service} available at ${cluster_ip}:8000"
        fi
    done
    
    log_success "Deployment verification completed"
}

# Function to get deployment status
get_status() {
    log_info "Getting deployment status..."
    
    echo "=================== NAMESPACE STATUS ==================="
    kubectl get namespace ${NAMESPACE} -o wide
    
    echo "=================== DEPLOYMENT STATUS ==================="
    kubectl get deployments -n ${NAMESPACE} -o wide
    
    echo "=================== SERVICE STATUS ==================="
    kubectl get services -n ${NAMESPACE} -o wide
    
    echo "=================== POD STATUS ==================="
    kubectl get pods -n ${NAMESPACE} -o wide
    
    echo "=================== PVC STATUS ==================="
    kubectl get pvc -n ${NAMESPACE} -o wide
    
    echo "=================== HPA STATUS ==================="
    kubectl get hpa -n ${NAMESPACE} -o wide 2>/dev/null || echo "No HPAs found"
    
    echo "=================== INGRESS STATUS ==================="
    kubectl get ingress -n ${NAMESPACE} -o wide 2>/dev/null || echo "No Ingresses found"
}

# Function to clean up deployment
cleanup() {
    log_warning "Cleaning up deployment..."
    
    read -p "Are you sure you want to delete the entire ${NAMESPACE} deployment? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Delete in reverse order
        kubectl delete -f "${SCRIPT_DIR}/hpa.yaml" --ignore-not-found=true
        kubectl delete -f "${SCRIPT_DIR}/deployments.yaml" --ignore-not-found=true
        kubectl delete -f "${SCRIPT_DIR}/services.yaml" --ignore-not-found=true
        kubectl delete -f "${SCRIPT_DIR}/storage.yaml" --ignore-not-found=true
        kubectl delete -f "${SCRIPT_DIR}/configmaps.yaml" --ignore-not-found=true
        kubectl delete -f "${SCRIPT_DIR}/rbac.yaml" --ignore-not-found=true
        
        # Delete secrets
        kubectl delete secret postgres-secret redis-secret grandmodel-tls -n ${NAMESPACE} --ignore-not-found=true
        
        # Delete namespace (this will delete everything in it)
        kubectl delete namespace ${NAMESPACE} --ignore-not-found=true
        
        log_success "Cleanup completed"
    else
        log_info "Cleanup cancelled"
    fi
}

# Function to show usage
usage() {
    cat << EOF
GrandModel Kubernetes Deployment Script - Agent 5

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    deploy      Deploy the complete GrandModel system
    status      Get deployment status
    cleanup     Clean up the deployment
    help        Show this help message

Options:
    --namespace NAMESPACE   Set the Kubernetes namespace (default: grandmodel)
    --timeout TIMEOUT       Set timeout for operations (default: 600s)
    --verbose               Enable verbose output

Examples:
    $0 deploy                    # Deploy the complete system
    $0 status                    # Check deployment status
    $0 cleanup                   # Clean up deployment
    $0 deploy --verbose          # Deploy with verbose output

EOF
}

# Main deployment function
main_deploy() {
    log_info "Starting GrandModel Kubernetes deployment..."
    log_info "Agent 5 - System Integration & Production Deployment Validation"
    echo "============================================================"
    
    check_prerequisites
    deploy_namespace
    create_secrets
    deploy_rbac
    deploy_configmaps
    deploy_storage
    deploy_applications
    deploy_autoscaling
    verify_deployment
    
    echo "============================================================"
    log_success "GrandModel deployment completed successfully!"
    log_info "Access the system through the gateway service"
    log_info "Monitor the system using the metrics endpoints"
    log_info "Use 'kubectl get pods -n ${NAMESPACE}' to check pod status"
    
    # Display access information
    if kubectl get service gateway-service -n ${NAMESPACE} &> /dev/null; then
        local external_ip=$(kubectl get service gateway-service -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "Pending")
        if [[ "${external_ip}" != "Pending" && -n "${external_ip}" ]]; then
            log_info "External access: http://${external_ip}"
        else
            log_info "External IP pending. Use 'kubectl get service gateway-service -n ${NAMESPACE}' to check status"
        fi
    fi
}

# Parse command line arguments
COMMAND=""
while [[ $# -gt 0 ]]; do
    case $1 in
        deploy|status|cleanup|help)
            COMMAND="$1"
            shift
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Enable verbose mode
if [[ "${VERBOSE}" == "true" ]]; then
    set -x
fi

# Execute command
case "${COMMAND}" in
    deploy)
        main_deploy
        ;;
    status)
        get_status
        ;;
    cleanup)
        cleanup
        ;;
    help|"")
        usage
        ;;
    *)
        log_error "Unknown command: ${COMMAND}"
        usage
        exit 1
        ;;
esac