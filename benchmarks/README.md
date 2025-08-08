# Neural Operator Cryptanalysis - Benchmarks

This directory contains comprehensive benchmarking suites for evaluating neural operator performance in cryptanalysis tasks.

## 📊 Benchmark Categories

### 1. Architecture Comparison
- `architecture_benchmark.py` - Compare FNO vs DeepONet vs Custom architectures
- `scalability_benchmark.py` - Performance vs dataset size analysis
- `memory_efficiency.py` - Memory usage and optimization metrics

### 2. Attack Effectiveness
- `attack_success_rates.py` - Success rate vs trace count analysis
- `noise_robustness.py` - Performance under different SNR conditions
- `countermeasure_resistance.py` - Effectiveness against defenses

### 3. Post-Quantum Specific
- `pqc_benchmark_suite.py` - Comprehensive PQC implementation testing
- `kyber_ntt_benchmark.py` - Specialized Kyber NTT operation analysis
- `lattice_attack_comparison.py` - Traditional vs neural approaches

### 4. Hardware Platform Analysis
- `platform_comparison.py` - ARM vs RISC-V vs x86 analysis
- `device_portability.py` - Model transfer between platforms
- `real_hardware_validation.py` - Synthetic vs real trace comparison

## 🏃 Running Benchmarks

```bash
# Run full architecture comparison
python benchmarks/architecture_benchmark.py --output results/arch_comparison.json

# Test scalability
python benchmarks/scalability_benchmark.py --max-traces 100000 --step 5000

# PQC comprehensive benchmark
python benchmarks/pqc_benchmark_suite.py --algorithms kyber,dilithium,mceliece

# Generate benchmark report
python benchmarks/generate_report.py --input results/ --format html
```

## 📈 Benchmark Metrics

### Performance Metrics
- **Training Time**: Model training duration
- **Inference Speed**: Prediction latency per trace
- **Memory Usage**: Peak RAM and GPU memory consumption
- **Accuracy**: Attack success rate and confidence scores

### Research Metrics
- **Statistical Significance**: p-values for performance differences
- **Reproducibility**: Variance across multiple runs
- **Generalizability**: Cross-platform and cross-implementation performance

## 🎯 Baseline Comparisons

All benchmarks include comparisons against established baselines:
- Traditional correlation power analysis (CPA)
- Template attacks with Gaussian models
- Machine learning approaches (SVM, Random Forest)
- Deep neural networks (CNN, LSTM)

## 📊 Results Format

Benchmark results are saved in structured JSON format with metadata:

```json
{
  "benchmark_info": {
    "name": "architecture_comparison",
    "timestamp": "2024-01-15T10:30:00Z",
    "environment": {...}
  },
  "results": {
    "architectures": {...},
    "metrics": {...},
    "statistical_tests": {...}
  }
}
```