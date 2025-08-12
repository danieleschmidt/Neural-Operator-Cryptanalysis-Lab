# Neural Cryptanalysis Framework - Production Deployment Guide

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Architecture](#architecture)
4. [Infrastructure Setup](#infrastructure-setup)
5. [Security Configuration](#security-configuration)
6. [Containerization](#containerization)
7. [Kubernetes Deployment](#kubernetes-deployment)
8. [CI/CD Pipeline](#cicd-pipeline)
9. [Monitoring & Observability](#monitoring--observability)
10. [Global-First Features](#global-first-features)
11. [Scaling & Performance](#scaling--performance)
12. [Maintenance & Operations](#maintenance--operations)
13. [Troubleshooting](#troubleshooting)
14. [Compliance & Governance](#compliance--governance)

## Overview

This guide provides comprehensive instructions for deploying the Neural Cryptanalysis Framework in production environments. The deployment architecture supports:

- **Multi-environment deployments** (dev, staging, production)
- **Auto-scaling** and high availability
- **Global-first** approach with i18n and compliance
- **Enterprise security** standards
- **Comprehensive monitoring** and observability
- **Zero-downtime deployments**

## Prerequisites

### Infrastructure Requirements

#### Minimum Production Requirements
- **Kubernetes cluster**: v1.21+ with 3+ nodes
- **CPU**: 8+ cores per node
- **Memory**: 16GB+ per node
- **Storage**: 100GB+ SSD per node
- **Network**: 1Gbps+ bandwidth

#### Recommended Production Requirements
- **Kubernetes cluster**: v1.24+ with 5+ nodes
- **CPU**: 16+ cores per node
- **Memory**: 32GB+ per node
- **Storage**: 500GB+ NVMe SSD per node
- **Network**: 10Gbps+ bandwidth

### Software Dependencies

```bash
# Required tools
kubectl >= 1.21
helm >= 3.8
docker >= 20.10
terraform >= 1.0 (optional, for infrastructure as code)

# Optional but recommended
istio >= 1.15 (service mesh)
prometheus-operator >= 0.60
grafana >= 8.0
jaeger >= 1.35
```

### Cloud Provider Setup

#### AWS
```bash
# Install AWS CLI and configure
aws configure

# Install AWS Load Balancer Controller
kubectl apply -k "github.com/aws/eks-charts/stable/aws-load-balancer-controller//crds?ref=master"

# Install EBS CSI driver
kubectl apply -k "github.com/kubernetes-sigs/aws-ebs-csi-driver/deploy/kubernetes/overlays/stable/?ref=release-1.12"
```

#### Azure
```bash
# Install Azure CLI
az login

# Install Azure CSI drivers
kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/azuredisk-csi-driver/master/deploy/example/snapshot/storageclass-azuredisk-snapshot.yaml
```

#### GCP
```bash
# Install gcloud CLI
gcloud auth login

# Enable required APIs
gcloud services enable container.googleapis.com
gcloud services enable monitoring.googleapis.com
```

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Internet/CDN                             │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                Load Balancer (ALB/NLB)                         │
│              ┌─────────────────────────┐                       │
│              │   WAF/Rate Limiting     │                       │
│              │   SSL Termination       │                       │
└──────────────┴─────────┬───────────────┴───────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                 Kubernetes Cluster                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Ingress   │  │   Service   │  │     Neural Crypto API   │  │
│  │ Controller  │  │    Mesh     │  │    (Auto-scaling)       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ PostgreSQL  │  │    Redis    │  │     Background          │  │
│  │  (Primary)  │  │   Cache     │  │     Workers             │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Prometheus  │  │   Grafana   │  │      Jaeger             │  │
│  │ Monitoring  │  │ Dashboard   │  │    Tracing              │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Component Architecture

#### API Layer
- **Neural Crypto API**: Core application services
- **Load Balancer**: Traffic distribution and SSL termination
- **API Gateway**: Rate limiting, authentication, routing

#### Data Layer
- **PostgreSQL**: Primary database for experiments and results
- **Redis**: Caching and session management
- **Object Storage**: Model artifacts and large datasets

#### Processing Layer
- **Background Workers**: Asynchronous task processing
- **Model Training**: GPU-accelerated neural network training
- **Batch Processing**: Large-scale cryptanalysis jobs

#### Observability Layer
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **Jaeger**: Distributed tracing
- **ELK Stack**: Log aggregation and analysis

## Infrastructure Setup

### 1. Kubernetes Cluster Setup

#### Using eksctl (AWS)
```bash
# Create EKS cluster
eksctl create cluster \
  --name neural-crypto-prod \
  --version 1.24 \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type m5.xlarge \
  --nodes 3 \
  --nodes-min 3 \
  --nodes-max 10 \
  --managed \
  --ssh-access \
  --ssh-public-key ~/.ssh/id_rsa.pub
```

#### Using Azure AKS
```bash
# Create resource group
az group create --name neural-crypto-rg --location westus2

# Create AKS cluster
az aks create \
  --resource-group neural-crypto-rg \
  --name neural-crypto-prod \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-addons monitoring \
  --generate-ssh-keys
```

#### Using Google GKE
```bash
# Create GKE cluster
gcloud container clusters create neural-crypto-prod \
  --zone us-west1-a \
  --num-nodes 3 \
  --machine-type n1-standard-4 \
  --enable-autoscaling \
  --min-nodes 3 \
  --max-nodes 10
```

### 2. Storage Classes

```bash
# Apply storage classes
kubectl apply -f - <<EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
  encrypted: "true"
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
EOF
```

### 3. Namespace and RBAC Setup

```bash
# Create namespace and apply basic configuration
kubectl apply -f deployment/kubernetes/namespace.yaml
kubectl apply -f deployment/security/rbac/rbac.yaml
```

## Security Configuration

### 1. Pod Security Policies

```bash
# Apply pod security policies
kubectl apply -f deployment/security/policies/pod-security-policy.yaml
```

### 2. Network Policies

```bash
# Apply network security policies
kubectl apply -f deployment/security/policies/network-policies.yaml
```

### 3. Secrets Management

#### Option A: External Secrets Operator (Recommended)

```bash
# Install External Secrets Operator
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets -n external-secrets-system --create-namespace

# Configure with HashiCorp Vault
kubectl apply -f deployment/security/secrets/secret-management.yaml
```

#### Option B: Sealed Secrets

```bash
# Install Sealed Secrets controller
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.18.0/controller.yaml

# Create sealed secrets
echo -n 'your-secret-password' | kubectl create secret generic neural-crypto-secrets --dry-run=client --from-file=password=/dev/stdin -o yaml | kubeseal -o yaml > sealed-secret.yaml
kubectl apply -f sealed-secret.yaml
```

### 4. TLS Certificates

#### Using cert-manager

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.9.1/cert-manager.yaml

# Create ClusterIssuer for Let's Encrypt
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@yourdomain.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

## Containerization

### 1. Build Production Images

```bash
# Build production image
docker build -f Dockerfile.production --target production -t neural-cryptanalysis:latest .

# Tag for registry
docker tag neural-cryptanalysis:latest your-registry.com/neural-cryptanalysis:v1.0.0

# Push to registry
docker push your-registry.com/neural-cryptanalysis:v1.0.0
```

### 2. Multi-Architecture Builds

```bash
# Setup buildx for multi-platform builds
docker buildx create --name multi-arch-builder --use

# Build for multiple architectures
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --file Dockerfile.production \
  --target production \
  --tag your-registry.com/neural-cryptanalysis:v1.0.0 \
  --push .
```

### 3. Security Scanning

```bash
# Scan images for vulnerabilities
trivy image your-registry.com/neural-cryptanalysis:v1.0.0

# Generate SBOM
syft your-registry.com/neural-cryptanalysis:v1.0.0 -o spdx-json > sbom.json
```

## Kubernetes Deployment

### 1. Database Deployment

```bash
# Deploy PostgreSQL
kubectl apply -f deployment/kubernetes/postgresql.yaml

# Wait for database to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/component=postgresql -n neural-cryptanalysis --timeout=300s

# Verify database connection
kubectl exec -it deployment/neural-crypto-postgresql -n neural-cryptanalysis -- psql -U crypto_user -d cryptanalysis -c "SELECT version();"
```

### 2. Cache Deployment

```bash
# Deploy Redis
kubectl apply -f deployment/kubernetes/redis.yaml

# Wait for Redis to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/component=redis -n neural-cryptanalysis --timeout=300s

# Verify Redis connection
kubectl exec -it deployment/neural-crypto-redis -n neural-cryptanalysis -- redis-cli ping
```

### 3. Application Deployment

```bash
# Apply ConfigMaps and Secrets
kubectl apply -f deployment/kubernetes/configmap.yaml
kubectl apply -f deployment/kubernetes/secrets.yaml

# Deploy application
kubectl apply -f deployment/kubernetes/api-deployment.yaml

# Wait for deployment to be ready
kubectl rollout status deployment/neural-crypto-api -n neural-cryptanalysis --timeout=600s
```

### 4. Ingress and Load Balancer

```bash
# Deploy load balancer configuration
kubectl apply -f deployment/kubernetes/load-balancer.yaml

# Deploy ingress
kubectl apply -f deployment/kubernetes/ingress.yaml

# Get external IP
kubectl get ingress neural-crypto-ingress -n neural-cryptanalysis
```

### 5. Auto-scaling Configuration

```bash
# Deploy HPA
kubectl apply -f deployment/kubernetes/hpa.yaml

# Deploy VPA (if enabled)
kubectl apply -f deployment/kubernetes/vertical-pod-autoscaler.yaml

# Deploy Cluster Autoscaler
kubectl apply -f deployment/kubernetes/cluster-autoscaler.yaml
```

## CI/CD Pipeline

### 1. GitHub Actions Setup

```bash
# Secrets to configure in GitHub repository
DOCKER_REGISTRY_URL
DOCKER_USERNAME
DOCKER_PASSWORD
KUBE_CONFIG_DATA
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
SLACK_WEBHOOK_URL
```

### 2. Pipeline Configuration

The CI/CD pipeline is automatically configured through:
- `.github/workflows/ci.yml` - Continuous Integration
- `.github/workflows/cd.yml` - Continuous Deployment

### 3. Deployment Workflow

1. **Code Commit** → Triggers CI pipeline
2. **Quality Gates** → Code quality, security scans, tests
3. **Container Build** → Multi-stage Docker build and push
4. **Staging Deployment** → Automatic deployment to staging
5. **Integration Tests** → Comprehensive testing suite
6. **Production Approval** → Manual approval gate
7. **Production Deployment** → Blue-green deployment
8. **Post-deployment Monitoring** → Health checks and alerts

## Monitoring & Observability

### 1. Prometheus Setup

```bash
# Install Prometheus Operator
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --values - <<EOF
prometheus:
  prometheusSpec:
    storageSpec:
      volumeClaimTemplate:
        spec:
          storageClassName: fast-ssd
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 50Gi
grafana:
  persistence:
    enabled: true
    storageClassName: fast-ssd
    size: 10Gi
EOF
```

### 2. Application Metrics

```bash
# Apply service monitors
kubectl apply -f deployment/monitoring/prometheus.yml
kubectl apply -f deployment/monitoring/alerts/neural-crypto-alerts.yml
```

### 3. Grafana Dashboards

```bash
# Import dashboards
kubectl create configmap neural-crypto-dashboard \
  --from-file=deployment/monitoring/grafana/dashboards/ \
  -n monitoring

# Configure data sources
kubectl apply -f deployment/monitoring/grafana/datasources/prometheus.yml
```

### 4. Distributed Tracing

```bash
# Install Jaeger Operator
kubectl create namespace observability
kubectl apply -f https://github.com/jaegertracing/jaeger-operator/releases/download/v1.35.0/jaeger-operator.yaml -n observability

# Deploy Jaeger instance
kubectl apply -f - <<EOF
apiVersion: jaegertracing.io/v1
kind: Jaeger
metadata:
  name: neural-crypto-jaeger
  namespace: neural-cryptanalysis
spec:
  strategy: production
  storage:
    type: elasticsearch
    elasticsearch:
      nodeCount: 3
      redundancyPolicy: SingleRedundancy
      storage:
        storageClassName: fast-ssd
        size: 20Gi
EOF
```

### 5. Log Aggregation

```bash
# Install ELK Stack
helm repo add elastic https://helm.elastic.co
helm install elasticsearch elastic/elasticsearch \
  --namespace logging \
  --create-namespace \
  --set replicas=3 \
  --set volumeClaimTemplate.storageClassName=fast-ssd

helm install kibana elastic/kibana --namespace logging
helm install filebeat elastic/filebeat --namespace logging
```

## Global-First Features

### 1. Internationalization

The framework includes built-in i18n support with:

```python
from neural_cryptanalysis.i18n import _, set_locale, format_date

# Set user locale
set_locale('es')  # Spanish

# Translate messages
message = _('experiments.create_experiment')  # "Crear Experimento"

# Format dates according to locale
formatted_date = format_date(datetime.now())  # "31/12/2023"
```

### 2. Compliance Framework

#### GDPR Compliance

```python
from neural_cryptanalysis.i18n.compliance import get_compliance_manager, DataSubjectRequest

compliance = get_compliance_manager()

# Handle data subject request
request = DataSubjectRequest(
    request_id="gdpr-001",
    request_type="erasure",
    data_subject_id="user123",
    request_date=datetime.now(),
    description="User requests data deletion"
)

response = compliance.submit_subject_request(request)
```

#### Data Retention Policies

```bash
# Configure automatic data cleanup
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: data-retention-cleanup
  namespace: neural-cryptanalysis
spec:
  schedule: "0 2 * * 0"  # Weekly cleanup
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: cleanup
            image: neural-cryptanalysis:latest
            command:
            - python
            - -m
            - neural_cryptanalysis.compliance
            - --cleanup-expired-data
          restartPolicy: OnFailure
EOF
```

### 3. Multi-Region Deployment

```bash
# Deploy to multiple regions
for region in us-west-2 us-east-1 eu-west-1; do
  kubectl config use-context neural-crypto-$region
  kubectl apply -k deployment/kubernetes/
done
```

## Scaling & Performance

### 1. Horizontal Pod Autoscaling

The HPA configuration automatically scales based on:
- CPU utilization (target: 70%)
- Memory utilization (target: 80%)
- Custom metrics (requests per second)

```bash
# Monitor scaling decisions
kubectl describe hpa neural-crypto-api-hpa -n neural-cryptanalysis

# View current scaling metrics
kubectl top pods -n neural-cryptanalysis
```

### 2. Vertical Pod Autoscaling

VPA automatically adjusts resource requests/limits:

```bash
# Check VPA recommendations
kubectl describe vpa neural-crypto-api-vpa -n neural-cryptanalysis
```

### 3. Cluster Autoscaling

The cluster autoscaler automatically manages node capacity:

```bash
# Monitor cluster autoscaler logs
kubectl logs -f deployment/cluster-autoscaler -n kube-system

# Check node status
kubectl get nodes
```

### 4. Performance Optimization

#### Database Optimization

```sql
-- Enable connection pooling
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '1GB';
ALTER SYSTEM SET effective_cache_size = '3GB';
ALTER SYSTEM SET work_mem = '16MB';

-- Create indexes for common queries
CREATE INDEX CONCURRENTLY idx_experiments_status_created 
ON experiments(status, created_at) WHERE status IN ('running', 'pending');

CREATE INDEX CONCURRENTLY idx_models_type_accuracy 
ON models(type, accuracy DESC) WHERE accuracy IS NOT NULL;
```

#### Redis Optimization

```bash
# Configure Redis for performance
kubectl patch configmap neural-crypto-redis-config -n neural-cryptanalysis --patch '
data:
  redis.conf: |
    maxmemory 2gb
    maxmemory-policy allkeys-lru
    tcp-keepalive 300
    timeout 0
'
```

#### Application Optimization

```python
# Enable async processing
import asyncio
from neural_cryptanalysis.optimization import AsyncProcessor

processor = AsyncProcessor(max_workers=8)
await processor.process_batch(experiments)
```

## Maintenance & Operations

### 1. Backup Procedures

#### Database Backups

```bash
# Automated daily backups
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgresql-backup
  namespace: neural-cryptanalysis
spec:
  schedule: "0 3 * * *"  # Daily at 3 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:15-alpine
            command:
            - /bin/bash
            - -c
            - |
              TIMESTAMP=$(date +%Y%m%d_%H%M%S)
              pg_dump \$DATABASE_URL > /backup/neural_crypto_\$TIMESTAMP.sql
              aws s3 cp /backup/neural_crypto_\$TIMESTAMP.sql s3://neural-crypto-backups/database/
            env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: neural-crypto-database-credentials
                  key: DATABASE_URL
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
          volumes:
          - name: backup-storage
            emptyDir: {}
          restartPolicy: OnFailure
EOF
```

#### Application Data Backups

```bash
# Backup model artifacts and results
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: data-backup
  namespace: neural-cryptanalysis
spec:
  schedule: "0 4 * * *"  # Daily at 4 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: amazon/aws-cli
            command:
            - aws
            - s3
            - sync
            - /app/data/
            - s3://neural-crypto-backups/data/
            volumeMounts:
            - name: data-volume
              mountPath: /app/data
              readOnly: true
          volumes:
          - name: data-volume
            persistentVolumeClaim:
              claimName: neural-crypto-data-pvc
          restartPolicy: OnFailure
EOF
```

### 2. Update Procedures

#### Rolling Updates

```bash
# Update application with zero downtime
kubectl set image deployment/neural-crypto-api \
  neural-crypto-api=neural-cryptanalysis:v1.1.0 \
  -n neural-cryptanalysis

# Monitor rollout
kubectl rollout status deployment/neural-crypto-api -n neural-cryptanalysis

# Rollback if needed
kubectl rollout undo deployment/neural-crypto-api -n neural-cryptanalysis
```

#### Database Migrations

```bash
# Run database migrations
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: database-migration-v110
  namespace: neural-cryptanalysis
spec:
  template:
    spec:
      containers:
      - name: migration
        image: neural-cryptanalysis:v1.1.0
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
EOF
```

### 3. Health Checks

#### Application Health

```bash
# Check application health
kubectl exec -it deployment/neural-crypto-api -n neural-cryptanalysis -- \
  python -c "
from neural_cryptanalysis.utils.monitoring import health_check
print(health_check())
"

# Check database connectivity
kubectl exec -it deployment/neural-crypto-postgresql -n neural-cryptanalysis -- \
  pg_isready -U crypto_user -d cryptanalysis
```

#### System Health

```bash
# Check cluster health
kubectl get nodes
kubectl top nodes
kubectl get pods --all-namespaces

# Check resource usage
kubectl describe node
kubectl top pods --all-namespaces --sort-by=memory
```

### 4. Log Management

#### Application Logs

```bash
# View application logs
kubectl logs -f deployment/neural-crypto-api -n neural-cryptanalysis

# Search logs with specific patterns
kubectl logs deployment/neural-crypto-api -n neural-cryptanalysis | grep ERROR

# Export logs for analysis
kubectl logs deployment/neural-crypto-api -n neural-cryptanalysis --since=24h > app-logs.txt
```

#### System Logs

```bash
# View system events
kubectl get events --sort-by=.metadata.creationTimestamp

# Check kubelet logs
journalctl -u kubelet -f

# Check container runtime logs
journalctl -u docker -f  # or containerd
```

## Troubleshooting

### Common Issues

#### 1. Pod Startup Failures

```bash
# Check pod status
kubectl get pods -n neural-cryptanalysis

# Describe failing pod
kubectl describe pod <pod-name> -n neural-cryptanalysis

# Check pod logs
kubectl logs <pod-name> -n neural-cryptanalysis --previous

# Common fixes:
# - Resource constraints: Increase CPU/memory limits
# - Image pull errors: Check registry credentials
# - Configuration errors: Validate ConfigMaps and Secrets
```

#### 2. Database Connection Issues

```bash
# Test database connectivity
kubectl exec -it deployment/neural-crypto-api -n neural-cryptanalysis -- \
  python -c "
import psycopg2
import os
conn = psycopg2.connect(os.environ['DATABASE_URL'])
print('Database connection successful')
"

# Check database logs
kubectl logs deployment/neural-crypto-postgresql -n neural-cryptanalysis

# Common fixes:
# - Check credentials in secrets
# - Verify network policies allow connection
# - Check database resource limits
```

#### 3. Performance Issues

```bash
# Check resource utilization
kubectl top pods -n neural-cryptanalysis
kubectl top nodes

# Check HPA status
kubectl describe hpa -n neural-cryptanalysis

# Check for resource bottlenecks
kubectl describe node <node-name>

# Common fixes:
# - Scale up replicas or resources
# - Optimize database queries
# - Add caching layers
# - Review application bottlenecks
```

#### 4. Network Connectivity Issues

```bash
# Test service connectivity
kubectl exec -it deployment/neural-crypto-api -n neural-cryptanalysis -- \
  curl http://neural-crypto-postgresql:5432

# Check network policies
kubectl get networkpolicy -n neural-cryptanalysis

# Check ingress status
kubectl describe ingress neural-crypto-ingress -n neural-cryptanalysis

# Common fixes:
# - Review network policies
# - Check ingress controller configuration
# - Verify DNS resolution
# - Check firewall rules
```

### Debugging Tools

#### 1. Debug Pod

```bash
# Create debug pod for troubleshooting
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: debug-pod
  namespace: neural-cryptanalysis
spec:
  containers:
  - name: debug
    image: nicolaka/netshoot
    command: ["/bin/bash"]
    args: ["-c", "while true; do sleep 30; done;"]
    securityContext:
      runAsUser: 0
  restartPolicy: Always
EOF

# Use debug pod
kubectl exec -it debug-pod -n neural-cryptanalysis -- /bin/bash
```

#### 2. Port Forwarding

```bash
# Forward database port for local debugging
kubectl port-forward service/neural-crypto-postgresql 5432:5432 -n neural-cryptanalysis

# Forward application port
kubectl port-forward service/neural-crypto-api 8000:80 -n neural-cryptanalysis

# Forward monitoring ports
kubectl port-forward service/prometheus-operated 9090:9090 -n monitoring
kubectl port-forward service/grafana 3000:80 -n monitoring
```

#### 3. Performance Profiling

```bash
# Enable profiling in the application
kubectl patch deployment neural-crypto-api -n neural-cryptanalysis --patch '
spec:
  template:
    spec:
      containers:
      - name: neural-crypto-api
        env:
        - name: NEURAL_CRYPTO_PROFILING_ENABLED
          value: "true"
'

# Access profiling endpoints
kubectl port-forward service/neural-crypto-api 8000:80 -n neural-cryptanalysis
curl http://localhost:8000/debug/pprof/
```

## Compliance & Governance

### 1. GDPR Compliance

#### Data Processing Documentation

```bash
# Generate data processing report
kubectl exec -it deployment/neural-crypto-api -n neural-cryptanalysis -- \
  python -c "
from neural_cryptanalysis.i18n.compliance import get_compliance_manager
manager = get_compliance_manager()
report = manager.generate_compliance_report()
print(json.dumps(report, indent=2))
"
```

#### Data Subject Rights

```bash
# Handle data access request
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: data-access-request
  namespace: neural-cryptanalysis
spec:
  template:
    spec:
      containers:
      - name: data-access
        image: neural-cryptanalysis:latest
        command:
        - python
        - -m
        - neural_cryptanalysis.compliance
        - --handle-request
        - --type=access
        - --subject-id=user123
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: neural-crypto-database-credentials
              key: DATABASE_URL
      restartPolicy: OnFailure
EOF
```

### 2. Security Compliance

#### Security Scanning

```bash
# Scan running containers
trivy k8s --report summary neural-cryptanalysis

# Check for security policy violations
kubectl get podsecuritypolicy
kubectl get networkpolicy -n neural-cryptanalysis

# Audit RBAC permissions
kubectl auth can-i --list --as=system:serviceaccount:neural-cryptanalysis:neural-crypto-api
```

#### Compliance Reporting

```bash
# Generate security compliance report
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: compliance-report
  namespace: neural-cryptanalysis
spec:
  schedule: "0 9 * * 1"  # Weekly on Monday at 9 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: compliance
            image: neural-cryptanalysis:latest
            command:
            - python
            - -m
            - neural_cryptanalysis.compliance
            - --generate-report
            - --output=/reports/
            volumeMounts:
            - name: reports
              mountPath: /reports
          volumes:
          - name: reports
            persistentVolumeClaim:
              claimName: compliance-reports-pvc
          restartPolicy: OnFailure
EOF
```

### 3. Audit Logging

```bash
# Enable Kubernetes audit logging
# Add to kube-apiserver configuration:
--audit-log-path=/var/log/audit.log
--audit-policy-file=/etc/kubernetes/audit-policy.yaml
--audit-log-maxage=30
--audit-log-maxbackup=10
--audit-log-maxsize=100

# Application audit logging is automatically enabled
# View audit logs:
kubectl logs deployment/neural-crypto-api -n neural-cryptanalysis | grep "AUDIT"
```

## Performance Tuning

### 1. Application Optimization

```bash
# Enable performance monitoring
kubectl patch deployment neural-crypto-api -n neural-cryptanalysis --patch '
spec:
  template:
    spec:
      containers:
      - name: neural-crypto-api
        env:
        - name: NEURAL_CRYPTO_PERFORMANCE_MONITORING
          value: "true"
        - name: NEURAL_CRYPTO_MAX_WORKERS
          value: "8"
        - name: NEURAL_CRYPTO_BATCH_SIZE
          value: "64"
'
```

### 2. Database Optimization

```sql
-- Apply performance settings
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET track_activity_query_size = 2048;
ALTER SYSTEM SET log_statement = 'all';
ALTER SYSTEM SET log_duration = on;

-- Analyze query performance
SELECT query, calls, total_time, mean_time 
FROM pg_stat_statements 
ORDER BY total_time DESC 
LIMIT 10;
```

### 3. Caching Strategy

```bash
# Configure Redis for optimal performance
kubectl patch configmap neural-crypto-redis-config -n neural-cryptanalysis --patch '
data:
  redis.conf: |
    # Memory optimization
    maxmemory 4gb
    maxmemory-policy allkeys-lru
    
    # Performance tuning
    tcp-keepalive 300
    timeout 0
    tcp-backlog 511
    
    # Persistence optimization
    save 900 1
    save 300 10
    save 60 10000
    
    # Network optimization
    tcp-nodelay yes
'
```

---

## Support and Maintenance

### Getting Help

- **Documentation**: [https://neural-crypto.terragonlabs.com/docs](https://neural-crypto.terragonlabs.com/docs)
- **Issue Tracker**: [https://github.com/terragonlabs/neural-cryptanalysis/issues](https://github.com/terragonlabs/neural-cryptanalysis/issues)
- **Community Forum**: [https://community.terragonlabs.com](https://community.terragonlabs.com)
- **Enterprise Support**: enterprise@terragonlabs.com

### Maintenance Schedule

- **Daily**: Automated backups, security scans, performance monitoring
- **Weekly**: Dependency updates, vulnerability assessments, compliance checks
- **Monthly**: Capacity planning, performance optimization, disaster recovery testing
- **Quarterly**: Major version updates, architecture reviews, business continuity planning

### Version Compatibility

| Component | Minimum Version | Recommended Version |
|-----------|----------------|-------------------|
| Kubernetes | 1.21 | 1.24+ |
| Docker | 20.10 | 23.0+ |
| Helm | 3.8 | 3.10+ |
| Python | 3.9 | 3.11+ |
| PostgreSQL | 13 | 15+ |
| Redis | 6.2 | 7.0+ |

---

*This deployment guide is maintained by the Terragon Labs team. For updates and improvements, please contribute to our [documentation repository](https://github.com/terragonlabs/neural-cryptanalysis-docs).*