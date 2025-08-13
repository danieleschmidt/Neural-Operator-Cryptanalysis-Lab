#!/bin/bash

# Neural Cryptanalysis Framework - Production Deployment Script
# This script automates the deployment process for production environments

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
NAMESPACE="neural-cryptanalysis"
ENVIRONMENT="${ENVIRONMENT:-production}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_TESTS="${SKIP_TESTS:-false}"
VERBOSE="${VERBOSE:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARN: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS: $1${NC}"
}

# Help function
show_help() {
    cat << EOF
Neural Cryptanalysis Framework - Production Deployment Script

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -e, --environment ENV   Set environment (dev|staging|production) [default: production]
    -n, --namespace NS      Set Kubernetes namespace [default: neural-cryptanalysis]
    -d, --dry-run          Show what would be deployed without making changes
    -s, --skip-tests       Skip pre-deployment tests
    -v, --verbose          Enable verbose output
    --skip-secrets         Skip secrets deployment (use existing)
    --skip-database        Skip database deployment
    --skip-monitoring      Skip monitoring deployment
    --force                Force deployment even if validation fails

EXAMPLES:
    # Deploy to production with all components
    $0 --environment production

    # Dry run deployment to staging
    $0 --environment staging --dry-run

    # Deploy only application (skip infrastructure)
    $0 --skip-database --skip-monitoring

    # Deploy with verbose output
    $0 --verbose --environment production

ENVIRONMENT VARIABLES:
    KUBECONFIG             Path to kubectl configuration file
    DOCKER_REGISTRY        Docker registry URL
    IMAGE_TAG              Docker image tag to deploy
    DATABASE_PASSWORD      Database password (if not using secrets)
    REDIS_PASSWORD         Redis password (if not using secrets)

EOF
}

# Parse command line arguments
parse_args() {
    SKIP_SECRETS=false
    SKIP_DATABASE=false
    SKIP_MONITORING=false
    FORCE=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -s|--skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            --skip-secrets)
                SKIP_SECRETS=true
                shift
                ;;
            --skip-database)
                SKIP_DATABASE=true
                shift
                ;;
            --skip-monitoring)
                SKIP_MONITORING=true
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            *)
                error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Validate environment
validate_environment() {
    log "Validating environment..."

    # Check required tools
    local tools=("kubectl" "docker" "helm")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error "$tool is required but not installed"
            return 1
        fi
    done

    # Check kubectl connectivity
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
        return 1
    fi

    # Check environment validity
    if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|production)$ ]]; then
        error "Invalid environment: $ENVIRONMENT"
        return 1
    fi

    # Check Docker registry access
    if [[ -n "${DOCKER_REGISTRY:-}" ]]; then
        if ! docker pull "${DOCKER_REGISTRY}/neural-cryptanalysis:${IMAGE_TAG:-latest}" &> /dev/null; then
            warn "Cannot pull image from registry, using local image"
        fi
    fi

    success "Environment validation passed"
}

# Pre-deployment checks
pre_deployment_checks() {
    log "Running pre-deployment checks..."

    # Check cluster resources
    local available_cpu
    available_cpu=$(kubectl top nodes --no-headers | awk '{sum += $3} END {print sum}' | sed 's/m//')
    if [[ $available_cpu -lt 2000 ]]; then
        warn "Low CPU resources available: ${available_cpu}m"
        if [[ "$FORCE" != "true" ]]; then
            error "Insufficient resources. Use --force to override"
            return 1
        fi
    fi

    # Check for existing deployments
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        if kubectl get deployment -n "$NAMESPACE" | grep -q neural-crypto-api; then
            warn "Existing deployment found in namespace $NAMESPACE"
            if [[ "$FORCE" != "true" ]] && [[ "$DRY_RUN" != "true" ]]; then
                read -p "Continue with update? (y/N): " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    log "Deployment cancelled"
                    exit 0
                fi
            fi
        fi
    fi

    # Run tests if not skipped
    if [[ "$SKIP_TESTS" != "true" ]]; then
        log "Running pre-deployment tests..."
        if ! python -m pytest tests/ -x --tb=short; then
            error "Tests failed"
            return 1
        fi
        success "Pre-deployment tests passed"
    fi

    success "Pre-deployment checks passed"
}

# Apply Kubernetes manifests
apply_manifests() {
    local manifest_dir="$1"
    local description="$2"

    log "Applying $description..."

    if [[ "$DRY_RUN" == "true" ]]; then
        kubectl apply -f "$manifest_dir" --dry-run=client --validate=true
    else
        kubectl apply -f "$manifest_dir"
    fi

    success "$description applied successfully"
}

# Wait for deployment readiness
wait_for_deployment() {
    local deployment="$1"
    local timeout="${2:-600}"

    log "Waiting for deployment $deployment to be ready..."

    if [[ "$DRY_RUN" != "true" ]]; then
        if ! kubectl rollout status deployment/"$deployment" -n "$NAMESPACE" --timeout="${timeout}s"; then
            error "Deployment $deployment failed to become ready"
            kubectl describe deployment/"$deployment" -n "$NAMESPACE"
            kubectl logs deployment/"$deployment" -n "$NAMESPACE" --tail=50
            return 1
        fi
    fi

    success "Deployment $deployment is ready"
}

# Deploy infrastructure components
deploy_infrastructure() {
    log "Deploying infrastructure components..."

    # Create namespace
    log "Creating namespace $NAMESPACE..."
    if [[ "$DRY_RUN" == "true" ]]; then
        kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml
    else
        kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    fi

    # Apply RBAC
    apply_manifests "$PROJECT_ROOT/deployment/security/rbac/" "RBAC configuration"

    # Apply security policies
    apply_manifests "$PROJECT_ROOT/deployment/security/policies/" "Security policies"

    # Apply secrets (if not skipped)
    if [[ "$SKIP_SECRETS" != "true" ]]; then
        apply_manifests "$PROJECT_ROOT/deployment/kubernetes/secrets.yaml" "Secrets"
    fi

    # Apply ConfigMaps
    apply_manifests "$PROJECT_ROOT/deployment/kubernetes/configmap.yaml" "ConfigMaps"

    success "Infrastructure components deployed"
}

# Deploy database
deploy_database() {
    if [[ "$SKIP_DATABASE" == "true" ]]; then
        log "Skipping database deployment"
        return 0
    fi

    log "Deploying database..."

    apply_manifests "$PROJECT_ROOT/deployment/kubernetes/postgresql.yaml" "PostgreSQL database"

    # Wait for database to be ready
    wait_for_deployment "neural-crypto-postgresql" 300

    # Run database migrations
    log "Running database migrations..."
    if [[ "$DRY_RUN" != "true" ]]; then
        kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: database-migration-$(date +%s)
  namespace: $NAMESPACE
spec:
  template:
    spec:
      containers:
      - name: migration
        image: ${DOCKER_REGISTRY:-neural-cryptanalysis}:${IMAGE_TAG:-latest}
        command:
        - python
        - -m
        - neural_cryptanalysis.migrations
        - --apply
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: neural-crypto-database-credentials
              key: DATABASE_URL
      restartPolicy: OnFailure
  backoffLimit: 3
EOF

        # Wait for migration job to complete
        local job_name
        job_name=$(kubectl get jobs -n "$NAMESPACE" --sort-by=.metadata.creationTimestamp -o name | tail -1)
        kubectl wait --for=condition=complete "$job_name" -n "$NAMESPACE" --timeout=300s
    fi

    success "Database deployed and migrated"
}

# Deploy cache
deploy_cache() {
    log "Deploying cache..."

    apply_manifests "$PROJECT_ROOT/deployment/kubernetes/redis.yaml" "Redis cache"

    # Wait for Redis to be ready
    wait_for_deployment "neural-crypto-redis" 180

    success "Cache deployed"
}

# Deploy application
deploy_application() {
    log "Deploying application..."

    # Update image tag in deployment
    local deployment_file="/tmp/api-deployment.yaml"
    cp "$PROJECT_ROOT/deployment/kubernetes/api-deployment.yaml" "$deployment_file"
    
    if [[ -n "${IMAGE_TAG:-}" ]]; then
        sed -i "s|neural-cryptanalysis:latest|${DOCKER_REGISTRY:-neural-cryptanalysis}:${IMAGE_TAG}|g" "$deployment_file"
    fi

    apply_manifests "$deployment_file" "Neural Crypto API"

    # Wait for application to be ready
    wait_for_deployment "neural-crypto-api" 600

    # Deploy HPA
    apply_manifests "$PROJECT_ROOT/deployment/kubernetes/hpa.yaml" "Horizontal Pod Autoscaler"

    # Deploy VPA (if available)
    if kubectl get crd verticalpodautoscalers.autoscaling.k8s.io &> /dev/null; then
        apply_manifests "$PROJECT_ROOT/deployment/kubernetes/vertical-pod-autoscaler.yaml" "Vertical Pod Autoscaler"
    else
        warn "VPA CRDs not found, skipping VPA deployment"
    fi

    success "Application deployed"
}

# Deploy load balancer and ingress
deploy_networking() {
    log "Deploying networking components..."

    apply_manifests "$PROJECT_ROOT/deployment/kubernetes/load-balancer.yaml" "Load balancer"
    apply_manifests "$PROJECT_ROOT/deployment/kubernetes/ingress.yaml" "Ingress"

    # Wait for ingress to get external IP
    if [[ "$DRY_RUN" != "true" ]]; then
        log "Waiting for external IP assignment..."
        local retries=0
        local max_retries=30
        
        while [[ $retries -lt $max_retries ]]; do
            local external_ip
            external_ip=$(kubectl get ingress neural-crypto-ingress -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
            
            if [[ -n "$external_ip" ]]; then
                success "External IP assigned: $external_ip"
                break
            fi
            
            retries=$((retries + 1))
            sleep 10
        done
        
        if [[ $retries -eq $max_retries ]]; then
            warn "External IP not assigned within timeout"
        fi
    fi

    success "Networking components deployed"
}

# Deploy monitoring
deploy_monitoring() {
    if [[ "$SKIP_MONITORING" == "true" ]]; then
        log "Skipping monitoring deployment"
        return 0
    fi

    log "Deploying monitoring components..."

    # Check if monitoring namespace exists
    if ! kubectl get namespace monitoring &> /dev/null; then
        kubectl create namespace monitoring
    fi

    # Deploy Prometheus configuration
    apply_manifests "$PROJECT_ROOT/deployment/monitoring/prometheus.yml" "Prometheus configuration"

    # Deploy alerts
    apply_manifests "$PROJECT_ROOT/deployment/monitoring/alerts/" "Alerting rules"

    # Deploy Grafana dashboards
    kubectl create configmap neural-crypto-dashboards \
        --from-file="$PROJECT_ROOT/deployment/monitoring/grafana/dashboards/" \
        -n monitoring \
        --dry-run=client -o yaml | kubectl apply -f -

    success "Monitoring components deployed"
}

# Post-deployment verification
post_deployment_verification() {
    log "Running post-deployment verification..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log "Skipping verification in dry-run mode"
        return 0
    fi

    # Health checks
    log "Checking application health..."
    local max_retries=10
    local retry=0
    
    while [[ $retry -lt $max_retries ]]; do
        if kubectl exec deployment/neural-crypto-api -n "$NAMESPACE" -- \
            python -c "from neural_cryptanalysis.utils.monitoring import health_check; health_check()" &> /dev/null; then
            success "Application health check passed"
            break
        fi
        
        retry=$((retry + 1))
        sleep 10
    done
    
    if [[ $retry -eq $max_retries ]]; then
        error "Application health check failed"
        return 1
    fi

    # Database connectivity check
    log "Checking database connectivity..."
    if kubectl exec deployment/neural-crypto-api -n "$NAMESPACE" -- \
        python -c "
import os
import psycopg2
conn = psycopg2.connect(os.environ['DATABASE_URL'])
print('Database connection successful')
" &> /dev/null; then
        success "Database connectivity check passed"
    else
        error "Database connectivity check failed"
        return 1
    fi

    # API endpoint check
    log "Checking API endpoints..."
    kubectl port-forward service/neural-crypto-api 8080:80 -n "$NAMESPACE" &
    local port_forward_pid=$!
    sleep 5
    
    if curl -f http://localhost:8080/health &> /dev/null; then
        success "API endpoint check passed"
    else
        error "API endpoint check failed"
        kill $port_forward_pid 2>/dev/null || true
        return 1
    fi
    
    kill $port_forward_pid 2>/dev/null || true

    success "Post-deployment verification completed"
}

# Generate deployment report
generate_deployment_report() {
    log "Generating deployment report..."

    local report_file="/tmp/deployment-report-$(date +%Y%m%d-%H%M%S).txt"
    
    cat > "$report_file" << EOF
Neural Cryptanalysis Framework - Deployment Report
==================================================

Deployment Date: $(date)
Environment: $ENVIRONMENT
Namespace: $NAMESPACE
Image Tag: ${IMAGE_TAG:-latest}
Dry Run: $DRY_RUN

Deployment Summary:
------------------
EOF

    if [[ "$DRY_RUN" != "true" ]]; then
        kubectl get pods -n "$NAMESPACE" -o wide >> "$report_file"
        echo "" >> "$report_file"
        kubectl get services -n "$NAMESPACE" >> "$report_file"
        echo "" >> "$report_file"
        kubectl get ingress -n "$NAMESPACE" >> "$report_file"
        echo "" >> "$report_file"
        
        # Get external endpoints
        local external_ip
        external_ip=$(kubectl get ingress neural-crypto-ingress -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "Pending")
        
        echo "External Endpoints:" >> "$report_file"
        echo "API: https://$external_ip/" >> "$report_file"
        echo "Health Check: https://$external_ip/health" >> "$report_file"
        echo "Metrics: https://$external_ip/metrics" >> "$report_file"
    else
        echo "Dry run mode - no actual deployment performed" >> "$report_file"
    fi

    log "Deployment report saved to: $report_file"
    cat "$report_file"
}

# Cleanup function
cleanup() {
    local exit_code=$?
    
    if [[ $exit_code -ne 0 ]]; then
        error "Deployment failed with exit code $exit_code"
        
        if [[ "$DRY_RUN" != "true" ]]; then
            log "Collecting debug information..."
            kubectl get events --sort-by=.metadata.creationTimestamp -n "$NAMESPACE" --tail=20
            kubectl describe pods -n "$NAMESPACE" | grep -A 10 -B 5 "Warning\|Error"
        fi
    fi
    
    exit $exit_code
}

# Main deployment function
main() {
    # Set up trap for cleanup
    trap cleanup EXIT

    log "Starting Neural Cryptanalysis Framework deployment"
    log "Environment: $ENVIRONMENT"
    log "Namespace: $NAMESPACE"
    log "Dry Run: $DRY_RUN"

    # Validate environment
    validate_environment

    # Pre-deployment checks
    pre_deployment_checks

    # Deploy components in order
    deploy_infrastructure
    deploy_database
    deploy_cache
    deploy_application
    deploy_networking
    deploy_monitoring

    # Post-deployment verification
    post_deployment_verification

    # Generate report
    generate_deployment_report

    success "Deployment completed successfully!"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    parse_args "$@"
    
    if [[ "$VERBOSE" == "true" ]]; then
        set -x
    fi
    
    main
fi