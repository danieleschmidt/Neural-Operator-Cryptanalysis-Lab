# Terragon Neural Operator Cryptanalysis - Production Deployment Guide

**Version:** 1.0.0  
**Date:** August 2025  
**Status:** Production Ready  
**Deployment Target:** Enterprise/Research Infrastructure  

## Overview

This guide provides comprehensive instructions for deploying the Terragon Breakthrough Neural Operator Cryptanalysis framework to production environments. The deployment supports all four breakthrough implementations with enterprise-grade scalability, security, and monitoring.

## 🏗️ Architecture Overview

### Core Components

1. **Neural Architecture Search Service** - Automated architecture discovery and optimization
2. **Causal Analysis Engine** - Temporal cryptographic analysis with causality modeling  
3. **Foundation Model Server** - Pre-trained models with few-shot adaptation
4. **Neuro-Symbolic Reasoner** - Formal verification and symbolic reasoning
5. **Validation Framework** - Comprehensive benchmarking and statistical analysis

### Infrastructure Components

- **API Gateway** - Request routing and rate limiting
- **Load Balancer** - High-availability request distribution
- **Model Registry** - Versioned model storage and serving
- **Message Queue** - Asynchronous task processing
- **Monitoring Stack** - Comprehensive observability
- **Security Layer** - Authentication, authorization, encryption

## 🐳 Container Deployment

### Docker Images

#### Base Neural Operator Service
```dockerfile
# Dockerfile.neural-operators
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Set environment variables
ENV PYTHONPATH=/app/src
ENV NEURAL_OPERATOR_ENV=production

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run application
CMD ["python", "-m", "neural_cryptanalysis.cli", "serve", "--host", "0.0.0.0", "--port", "8080"]
```

#### NAS Service
```dockerfile
# Dockerfile.nas-service
FROM neural-operators-base:latest

WORKDIR /app

# Install NAS-specific dependencies
RUN pip install --no-cache-dir \
    scipy>=1.7.0 \
    matplotlib>=3.3.0 \
    seaborn>=0.11.0

# Copy NAS-specific configurations
COPY config/nas/ ./config/nas/

# Set service-specific environment
ENV SERVICE_TYPE=neural_architecture_search
ENV NAS_POPULATION_SIZE=20
ENV NAS_GENERATIONS=50

EXPOSE 8081

CMD ["python", "-m", "neural_cryptanalysis.neural_operators.architecture_search", "--serve"]
```

#### Causal Analysis Service
```dockerfile
# Dockerfile.causal-service
FROM neural-operators-base:latest

WORKDIR /app

# Install causal analysis dependencies
RUN pip install --no-cache-dir \
    networkx>=2.6.0 \
    statsmodels>=0.12.0

ENV SERVICE_TYPE=causal_analysis
ENV CAUSAL_MAX_LAG=20
ENV CAUSAL_SIGNIFICANCE_LEVEL=0.05

EXPOSE 8082

CMD ["python", "-m", "neural_cryptanalysis.neural_operators.causal_operators", "--serve"]
```

#### Foundation Models Service
```dockerfile
# Dockerfile.foundation-service
FROM neural-operators-base:latest

WORKDIR /app

# Install transformer dependencies
RUN pip install --no-cache-dir \
    transformers>=4.15.0 \
    tokenizers>=0.10.0

# Create model cache directory
RUN mkdir -p /app/models/cache

ENV SERVICE_TYPE=foundation_models
ENV MODEL_CACHE_DIR=/app/models/cache
ENV HUGGINGFACE_CACHE=/app/models/cache

EXPOSE 8083

VOLUME ["/app/models"]

CMD ["python", "-m", "neural_cryptanalysis.neural_operators.foundation_models", "--serve"]
```

#### Neuro-Symbolic Service
```dockerfile
# Dockerfile.neuro-symbolic-service
FROM neural-operators-base:latest

WORKDIR /app

# Install symbolic reasoning dependencies
RUN pip install --no-cache-dir \
    sympy>=1.8.0 \
    z3-solver>=4.8.0

# Copy knowledge graphs
COPY knowledge_graphs/ ./knowledge_graphs/

ENV SERVICE_TYPE=neuro_symbolic
ENV KNOWLEDGE_GRAPH_PATH=/app/knowledge_graphs

EXPOSE 8084

CMD ["python", "-m", "neural_cryptanalysis.neural_operators.neuro_symbolic", "--serve"]
```

### Docker Compose Configuration

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  # API Gateway
  api-gateway:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./config/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - nas-service
      - causal-service
      - foundation-service
      - neuro-symbolic-service
    restart: unless-stopped
    networks:
      - cryptanalysis-network

  # Neural Architecture Search Service
  nas-service:
    build:
      context: .
      dockerfile: Dockerfile.nas-service
    environment:
      - POSTGRES_URL=postgresql://user:pass@postgres:5432/cryptanalysis
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=INFO
    volumes:
      - nas-models:/app/models
      - nas-logs:/app/logs
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    networks:
      - cryptanalysis-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8081/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Causal Analysis Service
  causal-service:
    build:
      context: .
      dockerfile: Dockerfile.causal-service
    environment:
      - POSTGRES_URL=postgresql://user:pass@postgres:5432/cryptanalysis
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=INFO
    volumes:
      - causal-data:/app/data
      - causal-logs:/app/logs
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    networks:
      - cryptanalysis-network

  # Foundation Models Service
  foundation-service:
    build:
      context: .
      dockerfile: Dockerfile.foundation-service
    environment:
      - POSTGRES_URL=postgresql://user:pass@postgres:5432/cryptanalysis
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=INFO
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - foundation-models:/app/models
      - foundation-cache:/app/cache
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    networks:
      - cryptanalysis-network
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Neuro-Symbolic Service
  neuro-symbolic-service:
    build:
      context: .
      dockerfile: Dockerfile.neuro-symbolic-service
    environment:
      - POSTGRES_URL=postgresql://user:pass@postgres:5432/cryptanalysis
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=INFO
    volumes:
      - symbolic-knowledge:/app/knowledge_graphs
      - symbolic-logs:/app/logs
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    networks:
      - cryptanalysis-network

  # Database
  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=cryptanalysis
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=secure_password
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./config/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped
    networks:
      - cryptanalysis-network

  # Cache and Message Queue
  redis:
    image: redis:6-alpine
    command: redis-server --requirepass secure_redis_password
    volumes:
      - redis-data:/data
    restart: unless-stopped
    networks:
      - cryptanalysis-network

  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./config/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    ports:
      - "9090:9090"
    restart: unless-stopped
    networks:
      - cryptanalysis-network

  grafana:
    image: grafana/grafana:latest
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=secure_grafana_password
    volumes:
      - grafana-data:/var/lib/grafana
      - ./config/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./config/grafana/datasources:/etc/grafana/provisioning/datasources
    ports:
      - "3000:3000"
    restart: unless-stopped
    networks:
      - cryptanalysis-network

volumes:
  nas-models:
  nas-logs:
  causal-data:
  causal-logs:
  foundation-models:
  foundation-cache:
  symbolic-knowledge:
  symbolic-logs:
  postgres-data:
  redis-data:
  prometheus-data:
  grafana-data:

networks:
  cryptanalysis-network:
    driver: bridge
```

## ☸️ Kubernetes Deployment

### Namespace Configuration
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: terragon-cryptanalysis
  labels:
    name: terragon-cryptanalysis
    tier: production
```

### Neural Architecture Search Deployment
```yaml
# k8s/nas-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nas-service
  namespace: terragon-cryptanalysis
  labels:
    app: nas-service
    version: v1.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nas-service
  template:
    metadata:
      labels:
        app: nas-service
        version: v1.0.0
    spec:
      containers:
      - name: nas-service
        image: terragon/nas-service:v1.0.0
        ports:
        - containerPort: 8081
        env:
        - name: POSTGRES_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: postgres-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: cache-credentials
              key: redis-url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8081
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8081
          initialDelaySeconds: 30
          periodSeconds: 10
        volumeMounts:
        - name: nas-models
          mountPath: /app/models
      volumes:
      - name: nas-models
        persistentVolumeClaim:
          claimName: nas-models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: nas-service
  namespace: terragon-cryptanalysis
spec:
  selector:
    app: nas-service
  ports:
  - protocol: TCP
    port: 8081
    targetPort: 8081
  type: ClusterIP
```

### Foundation Models Deployment (GPU-Enabled)
```yaml
# k8s/foundation-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: foundation-service
  namespace: terragon-cryptanalysis
spec:
  replicas: 2
  selector:
    matchLabels:
      app: foundation-service
  template:
    metadata:
      labels:
        app: foundation-service
    spec:
      nodeSelector:
        accelerator: nvidia-tesla-v100
      containers:
      - name: foundation-service
        image: terragon/foundation-service:v1.0.0
        ports:
        - containerPort: 8083
        resources:
          requests:
            memory: "8Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
          limits:
            memory: "16Gi"
            cpu: "4000m"
            nvidia.com/gpu: 1
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: MODEL_CACHE_DIR
          value: "/app/models/cache"
        volumeMounts:
        - name: foundation-models
          mountPath: /app/models
        - name: gpu-cache
          mountPath: /app/cache
      volumes:
      - name: foundation-models
        persistentVolumeClaim:
          claimName: foundation-models-pvc
      - name: gpu-cache
        emptyDir:
          sizeLimit: 10Gi
```

### Ingress Configuration
```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: cryptanalysis-ingress
  namespace: terragon-cryptanalysis
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.cryptanalysis.terragon.ai
    secretName: cryptanalysis-tls
  rules:
  - host: api.cryptanalysis.terragon.ai
    http:
      paths:
      - path: /nas/
        pathType: Prefix
        backend:
          service:
            name: nas-service
            port:
              number: 8081
      - path: /causal/
        pathType: Prefix
        backend:
          service:
            name: causal-service
            port:
              number: 8082
      - path: /foundation/
        pathType: Prefix
        backend:
          service:
            name: foundation-service
            port:
              number: 8083
      - path: /symbolic/
        pathType: Prefix
        backend:
          service:
            name: neuro-symbolic-service
            port:
              number: 8084
```

## 🔧 Configuration Management

### Application Configuration
```yaml
# config/application.yml
terragon:
  cryptanalysis:
    # Global settings
    environment: production
    debug: false
    log_level: INFO
    
    # Database configuration
    database:
      url: ${POSTGRES_URL}
      pool_size: 20
      max_overflow: 30
      pool_timeout: 30
      
    # Cache configuration
    cache:
      url: ${REDIS_URL}
      ttl: 3600
      max_connections: 50
      
    # Model serving
    models:
      cache_dir: /app/models/cache
      max_model_size: 5GB
      inference_timeout: 300
      
    # Security
    security:
      api_key_required: true
      rate_limit:
        requests_per_minute: 100
        burst_limit: 150
      encryption:
        algorithm: AES-256-GCM
        key_rotation_interval: 24h
        
    # Monitoring
    monitoring:
      metrics_enabled: true
      tracing_enabled: true
      health_check_interval: 30s
      
    # Neural Architecture Search
    nas:
      population_size: ${NAS_POPULATION_SIZE:20}
      generations: ${NAS_GENERATIONS:50}
      mutation_rate: 0.15
      crossover_rate: 0.7
      elite_size: 5
      
    # Causal Analysis
    causal:
      max_lag: ${CAUSAL_MAX_LAG:20}
      significance_level: ${CAUSAL_SIGNIFICANCE_LEVEL:0.05}
      correction_method: fdr_bh
      
    # Foundation Models
    foundation:
      model_name: terragon/cryptanalysis-foundation-v1
      max_sequence_length: 2048
      batch_size: 16
      few_shot_examples: 5
      
    # Neuro-Symbolic
    symbolic:
      knowledge_graph_path: ${KNOWLEDGE_GRAPH_PATH}
      verification_enabled: true
      rule_confidence_threshold: 0.7
```

### Environment-Specific Configurations
```bash
# config/production.env
POSTGRES_URL=postgresql://cryptanalysis_user:secure_password@postgres.internal:5432/cryptanalysis_prod
REDIS_URL=redis://redis.internal:6379/0
LOG_LEVEL=INFO
METRICS_ENABLED=true
GPU_ENABLED=true
MODEL_PARALLELISM=true
DISTRIBUTED_TRAINING=true
```

## 📊 Monitoring and Observability

### Prometheus Configuration
```yaml
# config/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "cryptanalysis_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'nas-service'
    static_configs:
      - targets: ['nas-service:8081']
    scrape_interval: 30s
    metrics_path: '/metrics'
    
  - job_name: 'causal-service'
    static_configs:
      - targets: ['causal-service:8082']
    scrape_interval: 30s
    
  - job_name: 'foundation-service'
    static_configs:
      - targets: ['foundation-service:8083']
    scrape_interval: 30s
    
  - job_name: 'neuro-symbolic-service'
    static_configs:
      - targets: ['neuro-symbolic-service:8084']
    scrape_interval: 30s
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
```

### Grafana Dashboards
```json
{
  "dashboard": {
    "title": "Terragon Cryptanalysis - Production Overview",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m])) by (service)",
            "legendFormat": "{{service}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service))",
            "legendFormat": "{{service}} 95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{status=~\"5..\"}[5m])) by (service)",
            "legendFormat": "{{service}} errors"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "nvidia_gpu_utilization_gpu",
            "legendFormat": "GPU {{gpu}}"
          }
        ]
      },
      {
        "title": "Model Inference Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, sum(rate(model_inference_duration_seconds_bucket[5m])) by (le, model))",
            "legendFormat": "{{model}} inference time"
          }
        ]
      }
    ]
  }
}
```

### Alerting Rules
```yaml
# config/prometheus/cryptanalysis_rules.yml
groups:
  - name: cryptanalysis_alerts
    rules:
      - alert: HighErrorRate
        expr: sum(rate(http_requests_total{status=~"5.."}[5m])) by (service) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate for {{ $labels.service }}"
          description: "{{ $labels.service }} has an error rate of {{ $value }} errors per second"
          
      - alert: ModelInferenceTimeout
        expr: histogram_quantile(0.95, sum(rate(model_inference_duration_seconds_bucket[5m])) by (le)) > 30
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Model inference taking too long"
          description: "95th percentile inference time is {{ $value }} seconds"
          
      - alert: GPUMemoryHigh
        expr: nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes > 0.9
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "GPU memory usage is high"
          description: "GPU memory usage is {{ $value | humanizePercentage }}"
          
      - alert: ServiceDown
        expr: up{job=~"nas-service|causal-service|foundation-service|neuro-symbolic-service"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "{{ $labels.job }} is down"
          description: "{{ $labels.job }} has been down for more than 1 minute"
```

## 🔒 Security Configuration

### Network Security
```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: cryptanalysis-network-policy
  namespace: terragon-cryptanalysis
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
  - from:
    - podSelector:
        matchLabels:
          app: api-gateway
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
```

### RBAC Configuration
```yaml
# k8s/rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: cryptanalysis-service-account
  namespace: terragon-cryptanalysis
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: cryptanalysis-role
  namespace: terragon-cryptanalysis
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list"]
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: cryptanalysis-role-binding
  namespace: terragon-cryptanalysis
subjects:
- kind: ServiceAccount
  name: cryptanalysis-service-account
  namespace: terragon-cryptanalysis
roleRef:
  kind: Role
  name: cryptanalysis-role
  apiGroup: rbac.authorization.k8s.io
```

## 🚀 Deployment Scripts

### Production Deployment Script
```bash
#!/bin/bash
# scripts/deploy-production.sh

set -e

# Configuration
NAMESPACE="terragon-cryptanalysis"
RELEASE_VERSION="v1.0.0"
DOCKER_REGISTRY="terragon.azurecr.io"

echo "🚀 Deploying Terragon Cryptanalysis Framework v${RELEASE_VERSION}"

# Pre-deployment checks
echo "📋 Running pre-deployment checks..."
kubectl cluster-info
kubectl get nodes

# Create namespace if it doesn't exist
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

# Deploy secrets
echo "🔐 Deploying secrets..."
kubectl apply -f k8s/secrets/ -n ${NAMESPACE}

# Deploy persistent volumes
echo "💾 Deploying storage..."
kubectl apply -f k8s/storage/ -n ${NAMESPACE}

# Deploy database and cache
echo "🗄️  Deploying database and cache..."
kubectl apply -f k8s/postgres/ -n ${NAMESPACE}
kubectl apply -f k8s/redis/ -n ${NAMESPACE}

# Wait for database to be ready
echo "⏳ Waiting for database to be ready..."
kubectl wait --for=condition=ready pod -l app=postgres -n ${NAMESPACE} --timeout=300s

# Run database migrations
echo "📊 Running database migrations..."
kubectl apply -f k8s/migrations/ -n ${NAMESPACE}

# Deploy core services
echo "🧠 Deploying neural operator services..."
kubectl apply -f k8s/nas-deployment.yaml -n ${NAMESPACE}
kubectl apply -f k8s/causal-deployment.yaml -n ${NAMESPACE}
kubectl apply -f k8s/foundation-deployment.yaml -n ${NAMESPACE}
kubectl apply -f k8s/neuro-symbolic-deployment.yaml -n ${NAMESPACE}

# Deploy API gateway and ingress
echo "🌐 Deploying API gateway..."
kubectl apply -f k8s/ingress.yaml -n ${NAMESPACE}

# Deploy monitoring
echo "📊 Deploying monitoring stack..."
kubectl apply -f k8s/monitoring/ -n ${NAMESPACE}

# Wait for deployments to be ready
echo "⏳ Waiting for deployments to be ready..."
kubectl wait --for=condition=available deployment --all -n ${NAMESPACE} --timeout=600s

# Run health checks
echo "🏥 Running health checks..."
./scripts/health-check.sh

echo "✅ Deployment completed successfully!"
echo "🔗 Access the API at: https://api.cryptanalysis.terragon.ai"
echo "📊 Monitor at: https://monitoring.cryptanalysis.terragon.ai"
```

### Health Check Script
```bash
#!/bin/bash
# scripts/health-check.sh

NAMESPACE="terragon-cryptanalysis"
API_BASE_URL="https://api.cryptanalysis.terragon.ai"

echo "🏥 Running comprehensive health checks..."

# Check service endpoints
services=("nas" "causal" "foundation" "symbolic")
for service in "${services[@]}"; do
    echo "Checking ${service} service..."
    response=$(curl -s -o /dev/null -w "%{http_code}" "${API_BASE_URL}/${service}/health")
    if [ "$response" == "200" ]; then
        echo "✅ ${service} service is healthy"
    else
        echo "❌ ${service} service is unhealthy (HTTP $response)"
        exit 1
    fi
done

# Check database connectivity
echo "Checking database connectivity..."
kubectl exec -n ${NAMESPACE} $(kubectl get pod -n ${NAMESPACE} -l app=postgres -o jsonpath='{.items[0].metadata.name}') -- psql -U cryptanalysis_user -d cryptanalysis_prod -c "SELECT 1;" > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Database is accessible"
else
    echo "❌ Database connectivity failed"
    exit 1
fi

# Check GPU availability (if applicable)
if kubectl get nodes -l accelerator=nvidia-tesla-v100 | grep -q Ready; then
    echo "Checking GPU availability..."
    gpu_pods=$(kubectl get pods -n ${NAMESPACE} -l app=foundation-service -o jsonpath='{.items[*].metadata.name}')
    for pod in $gpu_pods; do
        gpu_status=$(kubectl exec -n ${NAMESPACE} $pod -- nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
        if [ $? -eq 0 ]; then
            echo "✅ GPU is available in pod $pod (utilization: $gpu_status%)"
        else
            echo "❌ GPU not available in pod $pod"
            exit 1
        fi
    done
fi

# Performance test
echo "Running performance tests..."
./scripts/performance-test.sh

echo "✅ All health checks passed!"
```

## 📈 Scaling and Performance

### Horizontal Pod Autoscaler
```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nas-service-hpa
  namespace: terragon-cryptanalysis
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: nas-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "50"
```

### Vertical Pod Autoscaler
```yaml
# k8s/vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: foundation-service-vpa
  namespace: terragon-cryptanalysis
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: foundation-service
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: foundation-service
      maxAllowed:
        memory: 32Gi
        cpu: 8
      minAllowed:
        memory: 4Gi
        cpu: 1
```

## 💾 Backup and Disaster Recovery

### Database Backup
```yaml
# k8s/backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup
  namespace: terragon-cryptanalysis
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: postgres-backup
            image: postgres:13
            env:
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: db-credentials
                  key: password
            command:
            - /bin/bash
            - -c
            - |
              pg_dump -h postgres -U cryptanalysis_user cryptanalysis_prod > /backup/backup-$(date +%Y%m%d-%H%M%S).sql
              aws s3 cp /backup/backup-$(date +%Y%m%d-%H%M%S).sql s3://terragon-backups/postgres/
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
          volumes:
          - name: backup-storage
            emptyDir: {}
          restartPolicy: OnFailure
```

### Model Backup and Versioning
```bash
#!/bin/bash
# scripts/backup-models.sh

NAMESPACE="terragon-cryptanalysis"
S3_BUCKET="terragon-model-backups"
BACKUP_DATE=$(date +%Y%m%d-%H%M%S)

echo "🔄 Starting model backup process..."

# Backup foundation models
kubectl exec -n ${NAMESPACE} $(kubectl get pod -n ${NAMESPACE} -l app=foundation-service -o jsonpath='{.items[0].metadata.name}') -- tar -czf /tmp/foundation-models-${BACKUP_DATE}.tar.gz /app/models/

# Copy to S3
kubectl exec -n ${NAMESPACE} $(kubectl get pod -n ${NAMESPACE} -l app=foundation-service -o jsonpath='{.items[0].metadata.name}') -- aws s3 cp /tmp/foundation-models-${BACKUP_DATE}.tar.gz s3://${S3_BUCKET}/foundation-models/

echo "✅ Model backup completed: foundation-models-${BACKUP_DATE}.tar.gz"
```

## 🔧 Maintenance and Updates

### Rolling Updates
```bash
#!/bin/bash
# scripts/rolling-update.sh

NAMESPACE="terragon-cryptanalysis"
NEW_VERSION="$1"

if [ -z "$NEW_VERSION" ]; then
    echo "Usage: $0 <new_version>"
    exit 1
fi

echo "🔄 Starting rolling update to version ${NEW_VERSION}..."

# Update deployments one by one
services=("nas-service" "causal-service" "foundation-service" "neuro-symbolic-service")

for service in "${services[@]}"; do
    echo "Updating ${service}..."
    kubectl set image deployment/${service} ${service}=terragon/${service}:${NEW_VERSION} -n ${NAMESPACE}
    
    # Wait for rollout to complete
    kubectl rollout status deployment/${service} -n ${NAMESPACE} --timeout=600s
    
    # Verify health
    if ! ./scripts/health-check-service.sh ${service}; then
        echo "❌ Health check failed for ${service}, rolling back..."
        kubectl rollout undo deployment/${service} -n ${NAMESPACE}
        exit 1
    fi
    
    echo "✅ ${service} updated successfully"
done

echo "✅ Rolling update completed successfully!"
```

## 📋 Troubleshooting Guide

### Common Issues and Solutions

#### High Memory Usage
```bash
# Check memory usage across pods
kubectl top pods -n terragon-cryptanalysis

# Investigate specific pod
kubectl describe pod <pod-name> -n terragon-cryptanalysis
kubectl logs <pod-name> -n terragon-cryptanalysis --tail=100
```

#### GPU Issues
```bash
# Check GPU allocation
kubectl describe node <gpu-node-name>

# Check GPU usage in pod
kubectl exec -n terragon-cryptanalysis <foundation-pod> -- nvidia-smi
```

#### Database Connection Issues
```bash
# Check database pod status
kubectl get pods -n terragon-cryptanalysis -l app=postgres

# Test database connectivity
kubectl exec -n terragon-cryptanalysis <app-pod> -- pg_isready -h postgres -p 5432
```

#### Model Loading Issues
```bash
# Check model storage
kubectl exec -n terragon-cryptanalysis <foundation-pod> -- ls -la /app/models/

# Check model download logs
kubectl logs -n terragon-cryptanalysis <foundation-pod> --tail=100 | grep "model"
```

## 🎯 Performance Tuning

### Optimization Recommendations

1. **Memory Management**
   - Set appropriate memory limits and requests
   - Enable swap accounting for containers
   - Use memory-mapped files for large models

2. **CPU Optimization**
   - Pin CPU cores for inference pods
   - Use CPU affinity for better cache performance
   - Enable CPU frequency scaling

3. **GPU Utilization**
   - Enable GPU sharing for multiple small models
   - Use TensorRT optimization for inference
   - Implement model quantization

4. **Network Performance**
   - Use persistent connections
   - Enable HTTP/2 for API calls
   - Implement connection pooling

5. **Storage Optimization**
   - Use fast SSD storage for model cache
   - Implement model sharding for large models
   - Enable compression for stored models

---

This production deployment guide provides comprehensive instructions for deploying the Terragon Breakthrough Neural Operator Cryptanalysis framework with enterprise-grade scalability, security, and monitoring capabilities. The configuration supports all four breakthrough implementations while maintaining high availability and performance in production environments.