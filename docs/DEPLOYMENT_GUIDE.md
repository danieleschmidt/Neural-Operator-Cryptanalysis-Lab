# Neural Operator Cryptanalysis Lab - Production Deployment Guide

## Overview

This comprehensive deployment guide covers production deployment of the Neural Operator Cryptanalysis Lab across various environments, from single-machine setups to distributed clusters with hardware-in-the-loop integration.

**Security Notice**: This framework is designed for defensive security research only. Ensure proper authorization and follow responsible disclosure practices.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Hardware Integration Setup](#hardware-integration-setup)
6. [Configuration Management](#configuration-management)
7. [Security Hardening](#security-hardening)
8. [Monitoring and Observability](#monitoring-and-observability)
9. [Backup and Recovery](#backup-and-recovery)
10. [Performance Tuning](#performance-tuning)
11. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements

| Component | Specification | Notes |
|-----------|---------------|-------|
| **CPU** | 8 cores, 2.4 GHz | Intel Xeon or AMD EPYC recommended |
| **Memory** | 32 GB RAM | More for large datasets |
| **Storage** | 500 GB SSD | Fast I/O for trace processing |
| **GPU** | NVIDIA RTX 3080 / V100 | CUDA 11.8+ support |
| **Network** | 1 Gbps | For distributed deployments |
| **OS** | Ubuntu 20.04 LTS | Python 3.9+ required |

### Recommended Production Setup

| Component | Specification | Rationale |
|-----------|---------------|-----------|
| **CPU** | 32 cores, 3.0 GHz | Parallel trace processing |
| **Memory** | 128 GB RAM | Large model caching |
| **Storage** | 2 TB NVMe SSD | High-throughput data pipeline |
| **GPU** | 4x NVIDIA A100 | Distributed training |
| **Network** | 10 Gbps | Multi-node coordination |
| **OS** | Ubuntu 22.04 LTS | Latest security updates |

### Hardware-in-the-Loop Additional Requirements

| Device Type | Recommended Models | Interface |
|-------------|-------------------|-----------|
| **Oscilloscope** | Picoscope 6404D, Keysight DSOX3034T | USB 3.0 / Ethernet |
| **EM Probe** | Langer RF-R 50-1, Tecknit EMC-25 | SMA connectors |
| **Target Board** | ChipWhisperer CW308, Arduino Uno | USB / SPI |
| **Power Supply** | Keysight E36312A | SCPI over Ethernet |

---

## Installation Methods

### Method 1: Package Installation (Recommended)

```bash
# Install from PyPI
pip install neural-operator-cryptanalysis

# With hardware support
pip install neural-operator-cryptanalysis[hardware]

# Full research installation
pip install neural-operator-cryptanalysis[research,hardware,visualization]
```

### Method 2: Development Installation

```bash
# Clone repository
git clone https://github.com/neural-cryptanalysis/Neural-Operator-Cryptanalysis-Lab.git
cd Neural-Operator-Cryptanalysis-Lab

# Create virtual environment
python -m venv neural_sca_env
source neural_sca_env/bin/activate  # On Windows: neural_sca_env\Scripts\activate

# Install in development mode
pip install -e ".[dev,research,hardware]"

# Install pre-commit hooks
pre-commit install
```

### Method 3: Conda Installation

```bash
# Create conda environment
conda create -n neural_sca python=3.9
conda activate neural_sca

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install neural-cryptanalysis
pip install neural-operator-cryptanalysis[all]
```

---

## Docker Deployment

### Single Container Deployment

#### Build Production Image

```dockerfile
# Dockerfile.production
FROM pytorch/pytorch:1.13-cuda11.6-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libusb-1.0-0-dev \
    libudev-dev \
    libhidapi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY setup.py .
COPY pyproject.toml .

# Install application
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 neural_sca && \
    chown -R neural_sca:neural_sca /app
USER neural_sca

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import neural_cryptanalysis; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "neural_cryptanalysis.api", "--host", "0.0.0.0", "--port", "8000"]
```

#### Build and Run

```bash
# Build production image
docker build -f Dockerfile.production -t neural-cryptanalysis:latest .

# Run with GPU support
docker run -d \
    --name neural-sca \
    --gpus all \
    -p 8000:8000 \
    -v /data/traces:/app/data \
    -v /data/models:/app/models \
    --restart unless-stopped \
    neural-cryptanalysis:latest
```

### Docker Compose Deployment

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  neural-sca:
    image: neural-cryptanalysis:latest
    container_name: neural-sca
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./config:/app/config
    environment:
      - NEURAL_SCA_CONFIG=/app/config/production.yaml
      - NEURAL_SCA_LOG_LEVEL=INFO
      - CUDA_VISIBLE_DEVICES=0,1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
    depends_on:
      - redis
      - postgresql
      - prometheus

  redis:
    image: redis:7-alpine
    container_name: neural-sca-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}

  postgresql:
    image: postgres:15-alpine
    container_name: neural-sca-db
    restart: unless-stopped
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=neural_sca
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}

  prometheus:
    image: prom/prometheus:latest
    container_name: neural-sca-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    container_name: neural-sca-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}

volumes:
  redis_data:
  postgres_data:
  prometheus_data:
  grafana_data:
```

#### Deploy with Docker Compose

```bash
# Set environment variables
export REDIS_PASSWORD=$(openssl rand -hex 32)
export DB_USER=neural_sca
export DB_PASSWORD=$(openssl rand -hex 32)
export GRAFANA_PASSWORD=$(openssl rand -hex 16)

# Deploy services
docker-compose -f docker-compose.production.yml up -d

# Check status
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs -f neural-sca
```

---

## Kubernetes Deployment

### Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install kubectl /usr/local/bin/

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Install NVIDIA GPU Operator (for GPU nodes)
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update
helm install --wait --generate-name \
    nvidia/gpu-operator \
    --set driver.enabled=false
```

### Namespace and RBAC

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: neural-cryptanalysis
  labels:
    name: neural-cryptanalysis
    security: restricted

---
# rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: neural-sca-service-account
  namespace: neural-cryptanalysis

---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: neural-sca-role
  namespace: neural-cryptanalysis
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: neural-sca-role-binding
  namespace: neural-cryptanalysis
subjects:
- kind: ServiceAccount
  name: neural-sca-service-account
  namespace: neural-cryptanalysis
roleRef:
  kind: Role
  name: neural-sca-role
  apiGroup: rbac.authorization.k8s.io
```

### ConfigMap

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: neural-sca-config
  namespace: neural-cryptanalysis
data:
  production.yaml: |
    neural_operators:
      fno:
        modes: 32
        width: 128
        depth: 4
    
    training:
      batch_size: 64
      learning_rate: 1e-3
      epochs: 100
    
    security:
      audit_logging: true
      rate_limiting: true
      max_requests_per_minute: 100
    
    monitoring:
      metrics_enabled: true
      metrics_port: 9090
      health_check_interval: 30
    
    storage:
      traces_path: /data/traces
      models_path: /data/models
      cache_size_gb: 10
```

### Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neural-cryptanalysis
  namespace: neural-cryptanalysis
  labels:
    app: neural-cryptanalysis
    version: v1.0.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: neural-cryptanalysis
  template:
    metadata:
      labels:
        app: neural-cryptanalysis
        version: v1.0.0
    spec:
      serviceAccountName: neural-sca-service-account
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: neural-sca
        image: neural-cryptanalysis:1.0.0
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: NEURAL_SCA_CONFIG
          value: /app/config/production.yaml
        - name: NEURAL_SCA_LOG_LEVEL
          value: INFO
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        - name: data-volume
          mountPath: /data
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
          limits:
            memory: "16Gi"
            cpu: "8"
            nvidia.com/gpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
      volumes:
      - name: config-volume
        configMap:
          name: neural-sca-config
      - name: data-volume
        persistentVolumeClaim:
          claimName: neural-sca-data-pvc
      nodeSelector:
        accelerator: nvidia-tesla-v100
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

### Service and Ingress

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: neural-cryptanalysis-service
  namespace: neural-cryptanalysis
  labels:
    app: neural-cryptanalysis
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  - port: 9090
    targetPort: 9090
    protocol: TCP
    name: metrics
  selector:
    app: neural-cryptanalysis

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: neural-cryptanalysis-ingress
  namespace: neural-cryptanalysis
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - neural-sca.example.com
    secretName: neural-sca-tls
  rules:
  - host: neural-sca.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: neural-cryptanalysis-service
            port:
              number: 80
```

### Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: neural-cryptanalysis-hpa
  namespace: neural-cryptanalysis
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: neural-cryptanalysis
  minReplicas: 3
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
        name: gpu_utilization
      target:
        type: AverageValue
        averageValue: "0.75"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 120
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

### Deploy to Kubernetes

```bash
# Apply configurations
kubectl apply -f namespace.yaml
kubectl apply -f rbac.yaml
kubectl apply -f configmap.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml
kubectl apply -f hpa.yaml

# Check deployment status
kubectl get pods -n neural-cryptanalysis
kubectl get services -n neural-cryptanalysis
kubectl get ingress -n neural-cryptanalysis

# View logs
kubectl logs -f deployment/neural-cryptanalysis -n neural-cryptanalysis

# Port forward for local access
kubectl port-forward service/neural-cryptanalysis-service 8000:80 -n neural-cryptanalysis
```

---

## Hardware Integration Setup

### Oscilloscope Integration

#### Picoscope Setup

```python
# hardware_setup.py
from neural_cryptanalysis.hardware import PicoscopeInterface

# Configure Picoscope 6404D
scope_config = {
    'model': 'picoscope_6404d',
    'channels': {
        'A': {
            'range': '100mV',
            'coupling': 'DC',
            'probe_attenuation': 1
        },
        'B': {
            'range': '50mV', 
            'coupling': 'AC',
            'probe_attenuation': 10
        }
    },
    'sampling_rate': 5e9,  # 5 GS/s
    'memory_depth': 1e6,
    'trigger': {
        'channel': 'C',
        'threshold': '2.5V',
        'direction': 'rising',
        'delay': 0
    }
}

scope = PicoscopeInterface(scope_config)
scope.connect()
```

#### ChipWhisperer Setup

```python
from neural_cryptanalysis.hardware import ChipWhispererInterface

# Configure ChipWhisperer CW308
cw_config = {
    'target_board': 'CW308_STM32F4',
    'clock_frequency': 24e6,
    'voltage': 3.3,
    'crypto_implementation': 'aes128_table',
    'programmer': 'openocd'
}

chipwhisperer = ChipWhispererInterface(cw_config)
chipwhisperer.connect()
chipwhisperer.program_target('firmware.hex')
```

### Automated Measurement Campaign

```python
from neural_cryptanalysis.hardware import MeasurementCampaign

# Configure automated campaign
campaign_config = {
    'n_traces': 50000,
    'operations': ['aes_encrypt', 'aes_key_schedule'],
    'input_generation': 'random',
    'trace_preprocessing': ['alignment', 'filtering'],
    'real_time_analysis': True,
    'storage_format': 'hdf5'
}

campaign = MeasurementCampaign(
    scope=scope,
    target=chipwhisperer,
    config=campaign_config
)

# Run campaign with real-time analysis
results = campaign.run_with_analysis(
    neural_operator='fourier_neural_operator',
    confidence_threshold=0.95
)
```

### Hardware Safety and Validation

```python
class HardwareSafetyManager:
    def __init__(self):
        self.safety_checks = [
            self.check_voltage_levels,
            self.check_current_limits,
            self.check_temperature,
            self.check_probe_integrity
        ]
    
    def validate_setup(self, hardware_config: Dict) -> bool:
        """Validate hardware setup for safety."""
        for check in self.safety_checks:
            if not check(hardware_config):
                return False
        return True
    
    def monitor_safety(self, measurement_session):
        """Monitor safety during measurement."""
        while measurement_session.active:
            if not self.validate_setup(measurement_session.config):
                measurement_session.emergency_stop()
                break
            time.sleep(1)
```

---

## Configuration Management

### Configuration Hierarchy

```
config/
├── default.yaml              # Default settings
├── development.yaml           # Development overrides
├── production.yaml           # Production overrides
├── security/
│   ├── authentication.yaml   # Auth settings
│   └── encryption.yaml       # Encryption settings
├── neural_operators/
│   ├── fno.yaml              # FNO configurations
│   ├── deeponet.yaml         # DeepONet configurations
│   └── graph_operators.yaml  # Graph operator settings
└── hardware/
    ├── oscilloscopes.yaml    # Scope configurations
    └── target_boards.yaml    # Target board settings
```

### Production Configuration

```yaml
# config/production.yaml
# Neural Operator Cryptanalysis Lab - Production Configuration

global:
  environment: production
  debug: false
  log_level: INFO
  random_seed: null  # Use random seed in production

api:
  host: 0.0.0.0
  port: 8000
  workers: 4
  timeout: 300
  max_request_size: 100MB

security:
  authentication:
    enabled: true
    method: jwt
    secret_key: ${JWT_SECRET_KEY}
    token_expiry: 3600
  
  authorization:
    enabled: true
    rbac_config: /app/config/rbac.yaml
  
  rate_limiting:
    enabled: true
    requests_per_minute: 60
    burst_size: 10
  
  audit_logging:
    enabled: true
    log_file: /var/log/neural_sca_audit.log
    include_request_body: false

neural_operators:
  default_architecture: fourier_neural_operator
  
  fno:
    modes: 32
    width: 128
    depth: 4
    activation: gelu
    dropout: 0.1
    normalization: layer_norm
  
  deeponet:
    branch_net: [512, 512, 512]
    trunk_net: [256, 256, 256]
    activation: relu
    dropout: 0.2
  
  graph_operator:
    node_features: 64
    edge_features: 32
    hidden_dim: 128
    n_layers: 4
    attention_heads: 8

training:
  batch_size: 64
  learning_rate: 1e-3
  weight_decay: 1e-4
  epochs: 100
  early_stopping:
    patience: 10
    min_delta: 1e-4
  
  optimizer: adamw
  scheduler: cosine_annealing
  
  mixed_precision: true
  gradient_clipping: 1.0
  
  checkpoint:
    save_every: 10
    keep_best: 3

data:
  storage_backend: s3
  s3_bucket: neural-sca-data
  cache_size_gb: 20
  
  preprocessing:
    standardization: true
    alignment: true
    filtering:
      enabled: true
      type: butterworth
      order: 4
      cutoff_hz: 100000
  
  augmentation:
    enabled: true
    techniques:
      - temporal_jitter
      - noise_addition
      - amplitude_scaling

monitoring:
  metrics:
    enabled: true
    port: 9090
    path: /metrics
  
  logging:
    level: INFO
    format: json
    file: /var/log/neural_sca.log
    max_size: 100MB
    backup_count: 10
  
  health_checks:
    enabled: true
    interval: 30
    timeout: 10

resources:
  gpu:
    enabled: true
    device_ids: [0, 1]
    memory_fraction: 0.9
  
  cpu:
    num_workers: 8
    pin_memory: true
  
  memory:
    cache_size: 10GB
    shared_memory: true

compliance:
  data_retention_days: 90
  anonymization: true
  gdpr_compliance: true
  audit_trail: true
```

### Environment-Specific Overrides

```bash
# Set environment variables
export NEURAL_SCA_CONFIG_ENV=production
export JWT_SECRET_KEY=$(openssl rand -hex 32)
export DATABASE_URL=postgresql://user:pass@localhost/neural_sca
export REDIS_URL=redis://localhost:6379/0
export S3_ACCESS_KEY=your_access_key
export S3_SECRET_KEY=your_secret_key
```

---

## Security Hardening

### Container Security

```dockerfile
# Security-hardened Dockerfile
FROM pytorch/pytorch:1.13-cuda11.6-cudnn8-runtime

# Update packages and install security patches
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Create non-root user with specific UID/GID
RUN groupadd -r neural_sca -g 1000 && \
    useradd -r -u 1000 -g neural_sca -d /app -s /bin/bash neural_sca

# Set secure file permissions
COPY --chown=neural_sca:neural_sca src/ /app/src/
COPY --chown=neural_sca:neural_sca requirements.txt /app/

# Remove unnecessary packages and files
RUN apt-get purge -y --auto-remove build-essential && \
    rm -rf /tmp/* /var/tmp/* /root/.cache

# Set resource limits
USER neural_sca
WORKDIR /app

# Security labels
LABEL \
    security.scan="enabled" \
    security.vulnerability.check="enabled" \
    maintainer="security@neural-cryptanalysis.org"

# Run with minimal privileges
CMD ["python", "-m", "neural_cryptanalysis.api"]
```

### Network Security

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: neural-sca-network-policy
  namespace: neural-cryptanalysis
spec:
  podSelector:
    matchLabels:
      app: neural-cryptanalysis
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 9090
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: database
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - namespaceSelector:
        matchLabels:
          name: cache
    ports:
    - protocol: TCP
      port: 6379
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS only
```

### Secrets Management

```yaml
# sealed-secret.yaml
apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: neural-sca-secrets
  namespace: neural-cryptanalysis
spec:
  encryptedData:
    jwt-secret: AgBy3i4OJSWK+PiTySYZZA9rO43cGDEQAM...
    db-password: AgBy3i4OJSWK+PiTySYZZA9rO43cGDEQAM...
    redis-password: AgBy3i4OJSWK+PiTySYZZA9rO43cGDEQAM...
  template:
    metadata:
      name: neural-sca-secrets
      namespace: neural-cryptanalysis
    type: Opaque
```

### Security Monitoring

```python
# security_monitoring.py
class SecurityMonitor:
    def __init__(self):
        self.threat_detector = ThreatDetector()
        self.audit_logger = AuditLogger()
        
    def monitor_api_requests(self, request_data):
        """Monitor API requests for security threats."""
        
        # Check for injection attacks
        if self.threat_detector.detect_injection(request_data):
            self.audit_logger.log_security_event(
                event_type='injection_attempt',
                request=request_data,
                severity='high'
            )
            return False
            
        # Check for unusual access patterns
        if self.threat_detector.detect_anomaly(request_data):
            self.audit_logger.log_security_event(
                event_type='anomalous_access',
                request=request_data,
                severity='medium'
            )
            
        return True
```

---

## Monitoring and Observability

### Prometheus Metrics

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
REQUEST_COUNT = Counter(
    'neural_sca_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'neural_sca_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

MODEL_TRAINING_DURATION = Histogram(
    'neural_sca_training_duration_seconds',
    'Model training duration in seconds',
    ['architecture', 'dataset_size']
)

ATTACK_SUCCESS_RATE = Gauge(
    'neural_sca_attack_success_rate',
    'Current attack success rate',
    ['target', 'method']
)

GPU_UTILIZATION = Gauge(
    'neural_sca_gpu_utilization_percent',
    'GPU utilization percentage',
    ['device_id']
)
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "id": null,
    "title": "Neural Cryptanalysis Lab",
    "tags": ["neural-sca", "cryptanalysis"],
    "timezone": "UTC",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(neural_sca_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "id": 2,
        "title": "Attack Success Rate",
        "type": "gauge",
        "targets": [
          {
            "expr": "neural_sca_attack_success_rate",
            "legendFormat": "{{target}} - {{method}}"
          }
        ]
      },
      {
        "id": 3,
        "title": "GPU Utilization",
        "type": "timeseries",
        "targets": [
          {
            "expr": "neural_sca_gpu_utilization_percent",
            "legendFormat": "GPU {{device_id}}"
          }
        ]
      }
    ]
  }
}
```

### Alerting Rules

```yaml
# alerts.yaml
groups:
- name: neural_sca_alerts
  rules:
  - alert: HighErrorRate
    expr: rate(neural_sca_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} requests/sec"
      
  - alert: LowAttackSuccessRate
    expr: neural_sca_attack_success_rate < 0.5
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Low attack success rate"
      description: "Success rate is {{ $value }}"
      
  - alert: GPUUtilizationHigh
    expr: neural_sca_gpu_utilization_percent > 95
    for: 15m
    labels:
      severity: critical
    annotations:
      summary: "GPU utilization very high"
      description: "GPU {{ $labels.device_id }} at {{ $value }}%"
```

---

## Backup and Recovery

### Data Backup Strategy

```bash
#!/bin/bash
# backup.sh - Automated backup script

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/neural_sca_${BACKUP_DATE}"

# Create backup directory
mkdir -p ${BACKUP_DIR}

# Backup PostgreSQL database
pg_dump neural_sca > ${BACKUP_DIR}/database.sql

# Backup trace data
tar -czf ${BACKUP_DIR}/traces.tar.gz /data/traces/

# Backup trained models
tar -czf ${BACKUP_DIR}/models.tar.gz /data/models/

# Backup configuration
cp -r /app/config ${BACKUP_DIR}/

# Upload to S3
aws s3 sync ${BACKUP_DIR} s3://neural-sca-backups/${BACKUP_DATE}/

# Clean up local backup
rm -rf ${BACKUP_DIR}

# Retain only last 30 days of backups
aws s3 ls s3://neural-sca-backups/ | \
    awk '{print $2}' | \
    sort | \
    head -n -30 | \
    xargs -I {} aws s3 rm s3://neural-sca-backups/{} --recursive
```

### Disaster Recovery Plan

```yaml
# disaster-recovery.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: disaster-recovery-runbook
data:
  runbook.md: |
    # Disaster Recovery Runbook
    
    ## Scenario 1: Complete Cluster Failure
    
    ### Recovery Steps:
    1. Provision new Kubernetes cluster
    2. Install required operators (GPU, monitoring)
    3. Restore from latest backup:
       ```bash
       kubectl apply -f backup/manifests/
       ```
    4. Restore data volumes from S3
    5. Verify service health
    
    ## Scenario 2: Data Corruption
    
    ### Recovery Steps:
    1. Stop affected services
    2. Identify corruption scope
    3. Restore from last known good backup
    4. Replay transaction logs if available
    5. Restart services
    
    ## Recovery Time Objectives (RTO):
    - Critical services: 1 hour
    - Full system: 4 hours
    
    ## Recovery Point Objectives (RPO):
    - Configuration: 15 minutes
    - Model data: 1 hour
    - Trace data: 4 hours
```

---

## Performance Tuning

### GPU Optimization

```python
# gpu_optimization.py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

def optimize_gpu_settings():
    """Optimize GPU settings for neural operator training."""
    
    # Enable TensorFloat-32 (TF32) for A100
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable optimized attention
    torch.backends.cuda.enable_flash_sdp(True)
    
    # Set memory allocation strategy
    torch.cuda.empty_cache()
    torch.cuda.memory.set_per_process_memory_fraction(0.9)
    
    # Compile model for optimization (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        return torch.compile
    return lambda x: x

def setup_distributed_training():
    """Setup distributed training across multiple GPUs."""
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=int(os.environ['WORLD_SIZE']),
        rank=int(os.environ['RANK'])
    )
    
    # Set device for current process
    device_id = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(device_id)
    
    return device_id
```

### Memory Optimization

```python
# memory_optimization.py
from torch.utils.checkpoint import checkpoint

class MemoryEfficientFNO(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.fno_layers = nn.ModuleList([
            FNOLayer(*args, **kwargs) for _ in range(4)
        ])
        
    def forward(self, x):
        # Use gradient checkpointing to trade compute for memory
        for layer in self.fno_layers:
            x = checkpoint(layer, x, use_reentrant=False)
        return x

def optimize_dataloader(dataset, batch_size, num_workers):
    """Create optimized DataLoader."""
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,          # Speed up GPU transfer
        persistent_workers=True,   # Keep workers alive
        prefetch_factor=2,        # Prefetch batches
        drop_last=True           # Consistent batch sizes
    )
```

### Storage Optimization

```python
# storage_optimization.py
import h5py
import blosc

class OptimizedTraceStorage:
    def __init__(self, file_path, compression='blosc'):
        self.file_path = file_path
        self.compression = compression
        
    def save_traces(self, traces, labels, metadata=None):
        """Save traces with optimized compression."""
        
        with h5py.File(self.file_path, 'w') as f:
            # Use chunking and compression
            trace_dataset = f.create_dataset(
                'traces',
                data=traces,
                chunks=True,
                compression='gzip',
                compression_opts=9,
                shuffle=True,
                fletcher32=True
            )
            
            label_dataset = f.create_dataset(
                'labels',
                data=labels,
                chunks=True,
                compression='gzip'
            )
            
            # Store metadata as attributes
            if metadata:
                for key, value in metadata.items():
                    f.attrs[key] = value
    
    def load_traces(self, indices=None):
        """Load traces efficiently."""
        
        with h5py.File(self.file_path, 'r') as f:
            if indices is None:
                traces = f['traces'][:]
                labels = f['labels'][:]
            else:
                traces = f['traces'][indices]
                labels = f['labels'][indices]
                
        return traces, labels
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory

**Problem**: GPU runs out of memory during training

**Solutions**:
```bash
# Reduce batch size
export NEURAL_SCA_BATCH_SIZE=32

# Enable gradient checkpointing
export NEURAL_SCA_GRADIENT_CHECKPOINTING=true

# Use mixed precision
export NEURAL_SCA_MIXED_PRECISION=true
```

#### 2. Hardware Connection Issues

**Problem**: Cannot connect to oscilloscope or target board

**Diagnosis**:
```bash
# Check USB devices
lsusb

# Check device permissions
ls -la /dev/ttyUSB*

# Check driver installation
modinfo ftdi_sio
```

**Solutions**:
```bash
# Add user to dialout group
sudo usermod -a -G dialout $USER

# Install missing drivers
sudo apt-get install libusb-1.0-0-dev

# Reset USB device
sudo usb_reset /dev/bus/usb/001/002
```

#### 3. Model Training Convergence Issues

**Problem**: Neural operator fails to converge

**Diagnosis**:
```python
# Check data quality
from neural_cryptanalysis.utils import TraceAnalyzer

analyzer = TraceAnalyzer()
quality_report = analyzer.analyze_dataset(traces, labels)
print(f"SNR: {quality_report.snr}")
print(f"Alignment quality: {quality_report.alignment_score}")
```

**Solutions**:
- Increase dataset size
- Improve trace preprocessing
- Adjust learning rate schedule
- Add data augmentation

#### 4. Performance Issues

**Problem**: Slow inference or training

**Diagnosis**:
```python
# Profile performance
from neural_cryptanalysis.utils import PerformanceProfiler

profiler = PerformanceProfiler()
with profiler.profile('training'):
    model.train(data)

profiler.print_summary()
```

**Solutions**:
- Enable model compilation (PyTorch 2.0+)
- Use appropriate batch sizes
- Optimize data loading pipeline
- Enable GPU optimizations

### Debug Commands

```bash
# Check container logs
docker logs neural-sca --tail=100 -f

# Check Kubernetes pod logs
kubectl logs -f deployment/neural-cryptanalysis -n neural-cryptanalysis

# Monitor resource usage
kubectl top pods -n neural-cryptanalysis

# Describe pod for troubleshooting
kubectl describe pod <pod-name> -n neural-cryptanalysis

# Execute into container
kubectl exec -it <pod-name> -n neural-cryptanalysis -- /bin/bash

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Check hardware connections
python -c "from neural_cryptanalysis.hardware import list_devices; print(list_devices())"
```

### Log Analysis

```bash
# Analyze error patterns
grep "ERROR" /var/log/neural_sca.log | tail -20

# Check authentication failures
grep "authentication" /var/log/neural_sca_audit.log

# Monitor performance metrics
grep "performance" /var/log/neural_sca.log | jq '.duration'

# Check security events
grep "security" /var/log/neural_sca_audit.log | jq '.event_type'
```

---

## Maintenance

### Regular Maintenance Tasks

```bash
#!/bin/bash
# maintenance.sh - Regular maintenance script

# Update container images
docker pull neural-cryptanalysis:latest

# Clean up old logs
find /var/log -name "*.log" -mtime +30 -delete

# Vacuum PostgreSQL database
psql -d neural_sca -c "VACUUM ANALYZE;"

# Clean up old model checkpoints
find /data/models -name "*.pth" -mtime +7 -not -name "*best*" -delete

# Update SSL certificates
certbot renew --quiet

# Run security scans
trivy image neural-cryptanalysis:latest

# Check backup integrity
aws s3 ls s3://neural-sca-backups/ --recursive | tail -10
```

### Health Checks

```python
# health_check.py
from neural_cryptanalysis import NeuralSCA
import torch

def health_check():
    """Comprehensive system health check."""
    
    checks = {
        'gpu_available': torch.cuda.is_available(),
        'model_loading': test_model_loading(),
        'hardware_connection': test_hardware_connection(),
        'database_connection': test_database_connection(),
        'storage_access': test_storage_access()
    }
    
    all_healthy = all(checks.values())
    
    return {
        'status': 'healthy' if all_healthy else 'unhealthy',
        'checks': checks,
        'timestamp': time.time()
    }

def test_model_loading():
    """Test that models can be loaded."""
    try:
        neural_sca = NeuralSCA()
        return True
    except Exception:
        return False
```

This deployment guide provides comprehensive coverage for production deployment of the Neural Operator Cryptanalysis Lab across various environments while maintaining security, reliability, and performance standards.