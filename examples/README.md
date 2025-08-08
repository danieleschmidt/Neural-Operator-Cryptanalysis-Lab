# Neural Operator Cryptanalysis - Examples

This directory contains practical examples demonstrating the Neural Operator Cryptanalysis Lab capabilities.

## 📚 Available Examples

### 1. Basic Usage Examples
- `basic_fno_attack.py` - Simple FNO-based side-channel attack
- `kyber_analysis.py` - Post-quantum Kyber implementation analysis
- `multi_channel_fusion.py` - Combining power and EM analysis

### 2. Advanced Research Examples
- `adaptive_attack_synthesis.py` - ML-driven attack parameter optimization
- `countermeasure_evaluation.py` - Testing masking and shuffling defenses
- `leakage_localization.py` - Spatial EM field analysis

### 3. Benchmarking Examples
- `performance_comparison.py` - Compare neural operator architectures
- `baseline_attacks.py` - Traditional vs neural operator approaches
- `scalability_analysis.py` - Performance vs dataset size analysis

## 🚀 Quick Start

```bash
# Run basic FNO attack example
python examples/basic_fno_attack.py

# Analyze Kyber implementation
python examples/kyber_analysis.py --variant kyber768 --traces 10000

# Compare architectures
python examples/performance_comparison.py --architectures fno,deeponet,custom
```

## 📊 Example Data

Each example uses synthetic data generation for reproducibility. Real hardware data collection examples are provided in the `hardware_examples/` subdirectory.