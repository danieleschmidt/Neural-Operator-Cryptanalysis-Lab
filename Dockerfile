# Neural Operator Cryptanalysis Lab - Production Container
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user for security
RUN groupadd -r cryptanalysis && useradd -r -g cryptanalysis cryptanalysis

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY setup.py pyproject.toml ./
COPY src/neural_cryptanalysis/_version.py src/neural_cryptanalysis/ || true

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir -e ".[dev,research]"

# Copy source code
COPY . .

# Change ownership to non-root user
RUN chown -R cryptanalysis:cryptanalysis /app

USER cryptanalysis

# Command for development
CMD ["python", "-m", "neural_cryptanalysis.cli", "--help"]

# Production stage
FROM base as production

# Copy only necessary files
COPY src/ src/
COPY README.md LICENSE ./

# Install production dependencies only
RUN pip install --no-cache-dir .

# Change ownership to non-root user
RUN chown -R cryptanalysis:cryptanalysis /app

USER cryptanalysis

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import neural_cryptanalysis; print('OK')" || exit 1

# Command for production
CMD ["python", "-m", "neural_cryptanalysis.cli"]

# Research stage with Jupyter support
FROM development as research

# Install Jupyter and additional research tools
RUN pip install --no-cache-dir jupyter jupyterlab plotly tensorboard wandb

# Expose Jupyter port
EXPOSE 8888

# Create directories for data and notebooks
RUN mkdir -p /app/notebooks /app/data /app/results

# Command for research environment
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]