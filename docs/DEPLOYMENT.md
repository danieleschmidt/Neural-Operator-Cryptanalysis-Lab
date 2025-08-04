# Deployment Guide

## GitHub Actions CI/CD Setup

Due to GitHub security restrictions, the CI/CD workflow needs to be created manually in the GitHub repository with appropriate permissions.

### Manual CI/CD Setup

1. Go to your GitHub repository settings
2. Navigate to Actions > General
3. Ensure "Allow all actions and reusable workflows" is selected
4. Create `.github/workflows/ci.yml` with the following content:

```yaml
name: Neural Operator Cryptanalysis Lab CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
        
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,research]"
        
    - name: Lint with flake8
      run: |
        flake8 src/neural_cryptanalysis --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 src/neural_cryptanalysis --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
        
    - name: Format check with black
      run: |
        black --check src/neural_cryptanalysis tests
        
    - name: Type check with mypy
      run: |
        mypy src/neural_cryptanalysis --ignore-missing-imports
        
    - name: Test with pytest
      run: |
        pytest tests/ --cov=src/neural_cryptanalysis --cov-report=xml --cov-fail-under=85
        
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
        
  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.9
        
    - name: Install security tools
      run: |
        python -m pip install --upgrade pip
        pip install bandit safety
        
    - name: Security check with bandit
      run: |
        bandit -r src/neural_cryptanalysis -f json -o bandit-report.json
        
    - name: Check dependencies with safety
      run: |
        safety check --json --output safety-report.json
        
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json
```

## Docker Deployment

### Development Environment

```bash
# Build and run development environment
docker-compose --profile dev up --build

# Access the container
docker exec -it neural-crypto-dev bash
```

### Research Environment

```bash
# Start Jupyter Lab research environment
docker-compose --profile research up --build

# Access Jupyter at http://localhost:8888
```

### Production Deployment

```bash
# Deploy production environment
docker-compose --profile prod up -d --build

# Monitor logs
docker-compose logs -f neural-crypto-prod
```

## Local Development Setup

### Prerequisites

- Python 3.9+
- Git
- Docker (optional)

### Installation

```bash
# Clone the repository
git clone https://github.com/danieleschmidt/Neural-Operator-Cryptanalysis-Lab.git
cd Neural-Operator-Cryptanalysis-Lab

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,research]"

# Run tests
pytest tests/ -v
```

### Environment Variables

Create a `.env` file for local development:

```bash
NEURAL_CRYPTO_LOG_LEVEL=DEBUG
NEURAL_CRYPTO_DATA_DIR=./data
NEURAL_CRYPTO_OUTPUT_DIR=./output
NEURAL_CRYPTO_REQUIRE_AUTHORIZATION=false  # Development only
NEURAL_CRYPTO_AUDIT_LOGGING=true
```

## Security Configuration

### Production Security Settings

```yaml
security:
  enable_responsible_disclosure: true
  max_attack_iterations: 100000
  require_authorization: true
  audit_logging: true
  rate_limit_attacks: true
  embargo_period_days: 90
  allowed_targets:
    - "test_implementations"
    - "research_platforms" 
    - "authorized_targets"
```

### Authorization Setup

For production deployments, ensure proper authorization:

```python
from neural_cryptanalysis.security import SecurityPolicy, ResponsibleDisclosure

# Configure security policy
policy = SecurityPolicy(
    max_traces_per_experiment=1000000,
    require_written_authorization=True,
    audit_all_operations=True
)

# Initialize responsible disclosure
disclosure = ResponsibleDisclosure(policy)
```

## Monitoring and Logging

### Log Configuration

```yaml
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  log_file: "neural_cryptanalysis.log"
  max_file_size: "10MB"
  backup_count: 5
```

### Monitoring Stack (Optional)

Use the monitoring profile for production observability:

```bash
# Start monitoring stack
docker-compose --profile monitoring up -d

# Access Grafana at http://localhost:3000
# Access Prometheus at http://localhost:9090
```

## Performance Optimization

### GPU Configuration

For GPU-accelerated neural operators:

```python
import torch

# Check GPU availability
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name()}")
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Configure neural operator
neural_sca = NeuralSCA(
    architecture='fourier_neural_operator',
    config={'neural_operator': {'device': str(device)}}
)
```

### Memory Management

```python
from neural_cryptanalysis.optimization import BatchProcessor

# Optimize batch processing
processor = BatchProcessor(
    batch_size=64,  # Adjust based on available memory
    use_gpu=True
)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed with `pip install -e ".[dev]"`
2. **GPU Issues**: Check CUDA installation and PyTorch compatibility
3. **Memory Errors**: Reduce batch size or use CPU processing
4. **Permission Errors**: Check file permissions and Docker user settings

### Debug Mode

Enable debug logging for troubleshooting:

```bash
export NEURAL_CRYPTO_LOG_LEVEL=DEBUG
python -m neural_cryptanalysis.cli --log-level DEBUG
```

### Performance Profiling

```python
from neural_cryptanalysis.optimization import PerformanceProfiler

profiler = PerformanceProfiler()

@profiler.profile_function
def your_function():
    # Your code here
    pass

# Get performance metrics
metrics = profiler.get_average_metrics()
print(f"Average execution time: {metrics.execution_time:.3f}s")
```