# GitHub Workflows Setup Guide 🚀

## Overview

Due to GitHub App permissions restrictions, the CI/CD workflow files could not be automatically committed. This guide provides the complete workflow configurations that should be manually added to enable full automation.

## Required GitHub Repository Settings

Before adding these workflows, ensure your repository has the following permissions:

1. **Actions**: Enable GitHub Actions
2. **Workflows**: Allow workflow files to be created/modified
3. **Secrets**: Configure the following repository secrets:
   - `DOCKER_HUB_USERNAME`
   - `DOCKER_HUB_ACCESS_TOKEN`
   - `KUBECONFIG` (for Kubernetes deployments)
   - `SLACK_WEBHOOK_URL` (for notifications)

## Workflow Files to Create

### 1. Continuous Integration (`.github/workflows/ci.yml`)

```yaml
name: Continuous Integration

on:
  push:
    branches: [ main, develop, 'feature/*', 'terragon/*' ]
  pull_request:
    branches: [ main, develop ]

env:
  PYTHON_VERSION: "3.9"
  POETRY_VERSION: "1.4.2"

jobs:
  code-quality:
    name: Code Quality Checks
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install black isort flake8 mypy bandit safety
        pip install -e .
    
    - name: Code formatting (Black)
      run: black --check --diff src/ tests/
    
    - name: Import sorting (isort)
      run: isort --check-only --diff src/ tests/
    
    - name: Linting (flake8)
      run: flake8 src/ tests/
    
    - name: Type checking (mypy)
      run: mypy src/
    
    - name: Security scan (Bandit)
      run: bandit -r src/
    
    - name: Dependency check (Safety)
      run: safety check

  testing:
    name: Test Suite
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ["3.9", "3.10", "3.11"]
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pytest pytest-cov pytest-xdist
        pip install -e .
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=src/neural_cryptanalysis --cov-report=xml --cov-fail-under=85
    
    - name: Upload coverage to Codecov
      if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.9'
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  security-scan:
    name: Security Scanning
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results to GitHub Security tab
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: 'trivy-results.sarif'

  build:
    name: Build and Test Docker Image
    runs-on: ubuntu-latest
    needs: [code-quality, testing]
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Build Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        file: ./Dockerfile.production
        push: false
        tags: neural-cryptanalysis:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
    
    - name: Test Docker image
      run: |
        docker run --rm neural-cryptanalysis:${{ github.sha }} python -c "import neural_cryptanalysis; print('✅ Import successful')"

  performance-test:
    name: Performance Benchmarks
    runs-on: ubuntu-latest
    needs: [testing]
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .
        pip install pytest-benchmark
    
    - name: Run performance tests
      run: |
        pytest tests/test_performance_benchmarks.py -v --benchmark-only
    
    - name: Store benchmark results
      uses: benchmark-action/github-action-benchmark@v1
      if: github.ref == 'refs/heads/main'
      with:
        name: Python Benchmark
        tool: 'pytest'
        output-file-path: benchmark-results.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true

  compliance-check:
    name: Compliance Validation
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .
    
    - name: GDPR Compliance Check
      run: |
        python scripts/compliance_check.py --standard gdpr
    
    - name: CCPA Compliance Check
      run: |
        python scripts/compliance_check.py --standard ccpa
    
    - name: Academic Ethics Check
      run: |
        python scripts/ethics_check.py --validate-research-ethics
```

### 2. Continuous Deployment (`.github/workflows/cd.yml`)

```yaml
name: Continuous Deployment

on:
  push:
    branches: [ main ]
    tags: [ 'v*.*.*' ]
  workflow_dispatch:
    inputs:
      environment:
        description: 'Deployment environment'
        required: true
        default: 'staging'
        type: choice
        options:
        - staging
        - production

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-push:
    name: Build and Push Container
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    outputs:
      image-digest: ${{ steps.build.outputs.digest }}
      image-tag: ${{ steps.meta.outputs.tags }}
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=sha,prefix={{branch}}-
    
    - name: Build and push Docker image
      id: build
      uses: docker/build-push-action@v5
      with:
        context: .
        file: ./Dockerfile.production
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
        platforms: linux/amd64,linux/arm64

  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: build-and-push
    if: github.ref == 'refs/heads/main'
    environment:
      name: staging
      url: https://staging.neural-cryptanalysis.com
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Configure kubectl
      run: |
        echo "${{ secrets.KUBECONFIG }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
    
    - name: Deploy to staging
      run: |
        export KUBECONFIG=kubeconfig
        kubectl set image deployment/neural-cryptanalysis-api \
          api=${{ needs.build-and-push.outputs.image-tag }} \
          -n staging
        kubectl rollout status deployment/neural-cryptanalysis-api -n staging --timeout=300s
    
    - name: Run smoke tests
      run: |
        export KUBECONFIG=kubeconfig
        kubectl run smoke-test --rm -i --restart=Never \
          --image=${{ needs.build-and-push.outputs.image-tag }} \
          -- python -c "
        import neural_cryptanalysis
        from neural_cryptanalysis import NeuralSCA
        print('✅ Smoke test passed')
        "

  security-scan:
    name: Container Security Scan
    runs-on: ubuntu-latest
    needs: build-and-push
    
    steps:
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: ${{ needs.build-and-push.outputs.image-tag }}
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: 'trivy-results.sarif'

  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: [build-and-push, deploy-staging, security-scan]
    if: startsWith(github.ref, 'refs/tags/v') || github.event.inputs.environment == 'production'
    environment:
      name: production
      url: https://neural-cryptanalysis.com
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Manual approval gate
      uses: trstringer/manual-approval@v1
      with:
        secret: ${{ secrets.GITHUB_TOKEN }}
        approvers: danieleschmidt
        minimum-approvals: 1
        issue-title: "Production Deployment Approval Required"
        issue-body: |
          Please review and approve the production deployment:
          - Image: ${{ needs.build-and-push.outputs.image-tag }}
          - Commit: ${{ github.sha }}
          - Security scan: Passed ✅
          - Staging tests: Passed ✅
    
    - name: Configure kubectl
      run: |
        echo "${{ secrets.KUBECONFIG }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
    
    - name: Blue-Green Deployment
      run: |
        export KUBECONFIG=kubeconfig
        
        # Deploy to green environment
        kubectl set image deployment/neural-cryptanalysis-api-green \
          api=${{ needs.build-and-push.outputs.image-tag }} \
          -n production
        
        # Wait for green deployment
        kubectl rollout status deployment/neural-cryptanalysis-api-green -n production --timeout=600s
        
        # Run production smoke tests
        kubectl run prod-smoke-test --rm -i --restart=Never \
          --image=${{ needs.build-and-push.outputs.image-tag }} \
          -n production \
          -- python -c "
        import neural_cryptanalysis
        print('✅ Production smoke test passed')
        "
        
        # Switch traffic to green
        kubectl patch service neural-cryptanalysis-api \
          -p '{"spec":{"selector":{"version":"green"}}}' \
          -n production
        
        # Scale down blue environment
        kubectl scale deployment/neural-cryptanalysis-api-blue --replicas=1 -n production
    
    - name: Post-deployment monitoring
      run: |
        # Wait 5 minutes and check system health
        sleep 300
        curl -f https://neural-cryptanalysis.com/health || exit 1
        echo "✅ Production deployment successful"

  notify:
    name: Deployment Notifications
    runs-on: ubuntu-latest
    needs: [deploy-staging, deploy-production]
    if: always()
    
    steps:
    - name: Notify Slack
      if: always()
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        channel: '#deployments'
        webhook_url: ${{ secrets.SLACK_WEBHOOK_URL }}
        fields: repo,message,commit,author,action,eventName,ref,workflow
```

## Setup Instructions

1. **Create workflow directory**:
   ```bash
   mkdir -p .github/workflows
   ```

2. **Add CI workflow**:
   ```bash
   # Copy the CI workflow content above to:
   .github/workflows/ci.yml
   ```

3. **Add CD workflow**:
   ```bash
   # Copy the CD workflow content above to:
   .github/workflows/cd.yml
   ```

4. **Configure repository secrets**:
   - Go to Repository Settings → Secrets and variables → Actions
   - Add the required secrets listed above

5. **Enable GitHub Actions**:
   - Go to Repository Settings → Actions → General
   - Enable "Allow all actions and reusable workflows"

6. **Test workflows**:
   - Push to a feature branch to trigger CI
   - Merge to main to trigger CD

## Benefits

Once configured, these workflows provide:

- **Automated quality gates** on every push
- **Multi-platform testing** (Linux, Windows, macOS)
- **Security scanning** with vulnerability detection
- **Performance regression** detection
- **Blue-green deployments** with zero downtime
- **Compliance validation** for GDPR/CCPA
- **Automated notifications** for deployment status

The workflows are designed to maintain the highest standards of quality, security, and reliability for the Neural Operator Cryptanalysis Lab platform.