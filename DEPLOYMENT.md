# Neural Operator Cryptanalysis Lab - Deployment Guide

## 🚀 Production Deployment

This guide covers deployment of the Neural Operator Cryptanalysis Lab for production research environments.

### Prerequisites

- Python 3.9+ with pip
- CUDA 11.8+ (optional, for GPU acceleration)
- Docker (recommended for containerized deployment)
- Git for version control

### Installation Methods

#### Method 1: PyPI Installation (Recommended)

```bash
# Install stable release
pip install neural-operator-cryptanalysis

# With GPU support
pip install neural-operator-cryptanalysis[gpu]

# With development tools
pip install neural-operator-cryptanalysis[dev,research]
```

#### Method 2: Source Installation

```bash
# Clone repository
git clone https://github.com/danieleschmidt/Neural-Operator-Cryptanalysis-Lab.git
cd Neural-Operator-Cryptanalysis-Lab

# Install in development mode
pip install -e ".[dev,research]"
```

#### Method 3: Docker Deployment

```bash
# Pull latest image
docker pull neural-crypto-lab:latest

# Run interactive container
docker run -it --gpus all -p 8888:8888 neural-crypto-lab:research

# Production deployment
docker run -d --name neural-crypto-prod neural-crypto-lab:production
```

### Configuration

#### Environment Variables

```bash
# Core configuration
export NEURAL_CRYPTO_LOG_LEVEL=INFO
export NEURAL_CRYPTO_DATA_PATH=/path/to/data
export NEURAL_CRYPTO_MODEL_PATH=/path/to/models

# Security configuration
export NEURAL_CRYPTO_AUTH_TOKEN=your-secure-token
export NEURAL_CRYPTO_RATE_LIMIT=1000
export NEURAL_CRYPTO_AUDIT_LOG=true

# Performance configuration
export NEURAL_CRYPTO_BATCH_SIZE=128
export NEURAL_CRYPTO_NUM_WORKERS=4
export NEURAL_CRYPTO_CACHE_SIZE=1GB
```

#### Configuration File

Create `~/.neural_crypto/config.yaml`:

```yaml
# Neural Cryptanalysis Configuration
system:
  log_level: INFO
  data_path: "./data"
  model_path: "./models"
  temp_path: "/tmp/neural_crypto"

security:
  auth_required: true
  rate_limit: 1000
  audit_logging: true
  responsible_use_check: true

performance:
  batch_size: 128
  num_workers: 4
  cache_size: "1GB"
  gpu_enabled: true

research:
  experiment_tracking: true
  reproducible_seeds: true
  statistical_validation: true
```

### Production Architecture

#### Single Server Deployment

```
┌─────────────────────────────────────┐
│             Load Balancer           │
└─────────────┬───────────────────────┘
              │
┌─────────────▼───────────────────────┐
│      Neural Crypto Application     │
│  ┌─────────────┐ ┌───────────────┐  │
│  │   API       │ │   Research    │  │
│  │   Server    │ │   Jupyter     │  │
│  └─────────────┘ └───────────────┘  │
└─────────────┬───────────────────────┘
              │
┌─────────────▼───────────────────────┐
│         Storage Layer              │
│  ┌─────────────┐ ┌───────────────┐  │
│  │   Dataset   │ │   Model       │  │
│  │   Storage   │ │   Registry    │  │
│  └─────────────┘ └───────────────┘  │
└─────────────────────────────────────┘
```

#### Multi-Node Cluster Deployment

```
┌─────────────────────────────────────────┐
│            API Gateway                  │
└─────────────┬───────────────────────────┘
              │
    ┌─────────┼─────────┐
    │         │         │
┌───▼───┐ ┌───▼───┐ ┌───▼───┐
│Node 1 │ │Node 2 │ │Node 3 │
│       │ │       │ │       │
│ CPU   │ │ GPU   │ │ GPU   │
│ Only  │ │ Accel │ │ Accel │
└───┬───┘ └───┬───┘ └───┬───┘
    │         │         │
    └─────────┼─────────┘
              │
┌─────────────▼───────────────────────┐
│        Shared Storage              │
│  ┌─────────────┐ ┌───────────────┐  │
│  │   Datasets  │ │    Models     │  │
│  │    (NFS)    │ │   (Redis)     │  │
│  └─────────────┘ └───────────────┘  │
└─────────────────────────────────────┘
```

### Security Configuration

#### Authentication Setup

```bash
# Generate API tokens
neural-crypto auth generate-token --user researcher1
neural-crypto auth generate-token --user researcher2 --permissions read-only

# Configure LDAP/Active Directory (optional)
neural-crypto auth configure-ldap \
  --server ldap://your-ldap-server \
  --base-dn "dc=yourorg,dc=com" \
  --user-filter "(uid={username})"
```

#### Audit Logging

```yaml
# audit_config.yaml
audit:
  enabled: true
  log_file: "/var/log/neural_crypto/audit.log"
  rotation: daily
  retention_days: 90
  events:
    - dataset_access
    - model_training
    - attack_execution
    - key_recovery_attempts
```

#### Network Security

```bash
# Firewall configuration
ufw allow 22/tcp      # SSH
ufw allow 8888/tcp    # Jupyter
ufw allow 8080/tcp    # API server
ufw enable

# SSL/TLS certificate setup
neural-crypto ssl generate-cert --domain your-domain.com
neural-crypto ssl configure --cert-path /path/to/cert.pem
```

### Performance Optimization

#### GPU Configuration

```yaml
gpu:
  enabled: true
  devices: [0, 1, 2, 3]  # Use specific GPUs
  memory_fraction: 0.8   # Reserve GPU memory
  mixed_precision: true  # Enable FP16 training
```

#### Caching Strategy

```yaml
cache:
  redis:
    host: localhost
    port: 6379
    db: 0
    password: your-redis-password
  
  memory:
    size: "2GB"
    ttl: 3600  # 1 hour
  
  disk:
    path: "/var/cache/neural_crypto"
    size: "10GB"
    cleanup_threshold: 0.9
```

#### Database Configuration

```yaml
database:
  # Experiment tracking
  experiments:
    engine: postgresql
    host: localhost
    port: 5432
    name: neural_crypto_experiments
    user: neural_crypto
    password: your-db-password
  
  # Model registry
  models:
    engine: sqlite
    path: "./models/registry.db"
```

### Monitoring and Observability

#### Prometheus Metrics

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'neural-crypto'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
```

#### Health Checks

```bash
# Application health check
curl http://localhost:8080/health

# Detailed system status
curl http://localhost:8080/status
```

#### Logging Configuration

```yaml
logging:
  version: 1
  formatters:
    detailed:
      format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  handlers:
    file:
      class: logging.handlers.RotatingFileHandler
      filename: /var/log/neural_crypto/app.log
      maxBytes: 10485760  # 10MB
      backupCount: 5
      formatter: detailed
    console:
      class: logging.StreamHandler
      formatter: detailed
  root:
    level: INFO
    handlers: [file, console]
```

### Backup and Recovery

#### Data Backup

```bash
#!/bin/bash
# backup_script.sh

# Database backup
pg_dump neural_crypto_experiments > /backup/experiments_$(date +%Y%m%d).sql

# Model registry backup
cp ./models/registry.db /backup/registry_$(date +%Y%m%d).db

# Dataset backup (if local)
tar -czf /backup/datasets_$(date +%Y%m%d).tar.gz ./data/

# Configuration backup
cp -r ~/.neural_crypto /backup/config_$(date +%Y%m%d)/
```

#### Disaster Recovery

```bash
# Restore from backup
./scripts/restore_backup.sh /backup/2024MMDD/

# Verify system integrity
neural-crypto verify --full-check

# Restart services
systemctl restart neural-crypto
systemctl restart redis
systemctl restart postgresql
```

### Scaling and Load Balancing

#### Horizontal Scaling

```yaml
# kubernetes_deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neural-crypto-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: neural-crypto-api
  template:
    metadata:
      labels:
        app: neural-crypto-api
    spec:
      containers:
      - name: neural-crypto
        image: neural-crypto-lab:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

#### Load Balancer Configuration

```nginx
# nginx.conf
upstream neural_crypto_backend {
    server 127.0.0.1:8080;
    server 127.0.0.1:8081;
    server 127.0.0.1:8082;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://neural_crypto_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /health {
        access_log off;
        proxy_pass http://neural_crypto_backend;
    }
}
```

### Troubleshooting

#### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   export NEURAL_CRYPTO_BATCH_SIZE=64
   
   # Enable memory management
   neural-crypto config set gpu.memory_management true
   ```

2. **Authentication Failures**
   ```bash
   # Check token validity
   neural-crypto auth verify-token your-token
   
   # Regenerate tokens
   neural-crypto auth refresh-tokens
   ```

3. **Performance Issues**
   ```bash
   # Profile performance
   neural-crypto profile --duration 60s
   
   # Check resource usage
   neural-crypto status --detailed
   ```

#### Log Analysis

```bash
# Real-time log monitoring
tail -f /var/log/neural_crypto/app.log

# Error analysis
grep ERROR /var/log/neural_crypto/app.log | tail -20

# Performance metrics
grep "PERFORMANCE" /var/log/neural_crypto/app.log | tail -10
```

### Compliance and Security

#### GDPR Compliance

```yaml
privacy:
  data_retention_days: 90
  anonymization: true
  consent_tracking: true
  right_to_deletion: true
```

#### Security Hardening

```bash
# System hardening
neural-crypto security harden --level production

# Vulnerability scanning
neural-crypto security scan

# Penetration testing
neural-crypto security pentest --report /tmp/pentest_report.pdf
```

### Support and Maintenance

#### Regular Maintenance Tasks

```bash
# Weekly maintenance script
#!/bin/bash

# Update dependencies
pip install --upgrade neural-operator-cryptanalysis

# Clean temporary files
neural-crypto cleanup --temp-files --older-than 7d

# Rotate logs
logrotate /etc/logrotate.d/neural-crypto

# Database maintenance
neural-crypto db vacuum
neural-crypto db reindex

# Health check
neural-crypto health-check --comprehensive
```

#### Performance Monitoring

```bash
# Set up monitoring alerts
neural-crypto alerts configure \
  --cpu-threshold 80 \
  --memory-threshold 85 \
  --disk-threshold 90 \
  --email admin@yourorg.com
```

## 📞 Support

- Documentation: [https://neural-crypto-lab.readthedocs.io](https://neural-crypto-lab.readthedocs.io)
- Issues: [https://github.com/danieleschmidt/Neural-Operator-Cryptanalysis-Lab/issues](https://github.com/danieleschmidt/Neural-Operator-Cryptanalysis-Lab/issues)
- Security: [security@terragonlabs.com](mailto:security@terragonlabs.com)
- Community: [https://discord.gg/neural-crypto](https://discord.gg/neural-crypto)

---

**⚠️ Security Notice**: This is a research tool for defensive security analysis. Always ensure proper authorization before conducting any cryptanalysis and follow responsible disclosure practices.