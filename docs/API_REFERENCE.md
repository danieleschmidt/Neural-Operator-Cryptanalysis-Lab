# Neural Operator Cryptanalysis Lab - API Reference

## Overview

The Neural Operator Cryptanalysis Lab provides a comprehensive API for defensive side-channel analysis using neural operators. This reference documents all public classes, methods, and configuration options.

**Warning**: This tool is for defensive security research only. Users must follow responsible disclosure practices and obtain proper authorization before testing on any systems.

## Table of Contents

1. [Core API](#core-api)
2. [Neural Operators](#neural-operators)
3. [Side-Channel Analysis](#side-channel-analysis)
4. [Target Implementations](#target-implementations)
5. [Datasets](#datasets)
6. [Visualization](#visualization)
7. [Utilities](#utilities)
8. [Configuration](#configuration)
9. [Examples](#examples)

---

## Core API

### NeuralSCA

Main interface for neural operator-based side-channel analysis.

```python
from neural_cryptanalysis import NeuralSCA

neural_sca = NeuralSCA(
    architecture='fourier_neural_operator',
    channels=['power', 'em_near'],
    config=None
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `architecture` | `str` | `'fourier_neural_operator'` | Neural operator architecture |
| `channels` | `List[str]` | `['power']` | Side-channel types to analyze |
| `config` | `Union[Dict, str, None]` | `None` | Configuration dict or path |

#### Supported Architectures

- `'fourier_neural_operator'` - Fourier Neural Operator for spectral analysis
- `'deep_operator_network'` - DeepONet for operator learning
- `'graph_neural_operator'` - Graph-based multi-modal fusion
- `'physics_informed_operator'` - Physics-constrained operators
- `'crypto_transformer'` - Transformer-based operator

#### Methods

##### `train(traces, labels, validation_split=0.2, **kwargs)`

Train the neural operator on trace data.

```python
model = neural_sca.train(
    traces=trace_data,           # TraceData object or numpy array
    labels=intermediate_values,  # Target intermediate values
    validation_split=0.2,        # Fraction for validation
    epochs=100,                  # Training epochs
    batch_size=64,              # Batch size
    learning_rate=1e-3,         # Learning rate
    early_stopping=True         # Enable early stopping
)
```

**Parameters:**
- `traces`: Training trace data (shape: [n_traces, trace_length] or TraceData object)
- `labels`: Target intermediate values (shape: [n_traces])
- `validation_split`: Fraction of data for validation (0.0-1.0)
- `**kwargs`: Additional training parameters

**Returns:**
- `NeuralOperatorModel`: Trained model object

**Example:**
```python
from neural_cryptanalysis import NeuralSCA
from neural_cryptanalysis.datasets import SyntheticDatasetGenerator

# Generate synthetic dataset
generator = SyntheticDatasetGenerator()
traces, labels = generator.generate_aes_dataset(n_traces=10000)

# Train neural operator
neural_sca = NeuralSCA(architecture='fourier_neural_operator')
model = neural_sca.train(traces, labels, epochs=50)
```

##### `attack(test_traces, model, strategy='template', **kwargs)`

Perform side-channel attack using trained model.

```python
results = neural_sca.attack(
    test_traces=attack_traces,   # Test trace data
    model=trained_model,         # Trained neural operator
    strategy='adaptive_template', # Attack strategy
    target_bytes=[0, 1, 2],     # Specific key bytes to attack
    confidence_threshold=0.9     # Minimum confidence
)
```

**Parameters:**
- `test_traces`: Attack trace data
- `model`: Trained neural operator model
- `strategy`: Attack strategy ('template', 'adaptive_template', 'profiling')
- `**kwargs`: Strategy-specific parameters

**Returns:**
- `AttackResult`: Attack results with success metrics

**Example:**
```python
# Perform attack
attack_results = neural_sca.attack(
    test_traces=test_data,
    model=model,
    strategy='adaptive_template',
    target_bytes=list(range(16))  # Attack all 16 AES key bytes
)

print(f"Attack success: {attack_results.success}")
print(f"Key recovery rate: {attack_results.key_recovery_rate:.2%}")
print(f"Traces needed: {attack_results.traces_needed}")
```

##### `evaluate_countermeasure(implementation, traces, **kwargs)`

Evaluate effectiveness of countermeasures.

```python
evaluation = neural_sca.evaluate_countermeasure(
    implementation=masked_aes,
    traces=countermeasure_traces,
    masking_order=2,
    statistical_tests=['t_test', 'mutual_information']
)
```

**Returns:**
- `CountermeasureEvaluation`: Security assessment results

### LeakageSimulator

Simulate side-channel leakage for various implementations.

```python
from neural_cryptanalysis import LeakageSimulator

simulator = LeakageSimulator(
    device_model='stm32f4',
    noise_model='realistic',
    channels=['power', 'em_near']
)
```

#### Methods

##### `simulate_traces(target, n_traces, operations, **kwargs)`

Generate synthetic leakage traces.

```python
traces = simulator.simulate_traces(
    target=kyber_implementation,
    n_traces=5000,
    operations=['ntt', 'polynomial_mul'],
    snr_db=10,
    alignment_jitter=2
)
```

---

## Neural Operators

### FourierNeuralOperator

Fourier Neural Operator for spectral analysis of side-channel traces.

```python
from neural_cryptanalysis.neural_operators import FourierNeuralOperator

fno = FourierNeuralOperator(
    modes=16,           # Number of Fourier modes
    width=64,           # Hidden dimension
    depth=4,            # Number of layers
    input_dim=1,        # Input channels
    output_dim=256      # Output classes (e.g., 256 for byte values)
)
```

#### Architecture Details

The FNO uses spectral convolutions in the Fourier domain to efficiently capture long-range dependencies in side-channel traces.

**Key Features:**
- **Spectral Convolutions**: Efficient global receptive field
- **Multi-Resolution**: Captures patterns at different frequencies
- **Translation Invariance**: Robust to temporal shifts
- **Parameter Efficiency**: Fewer parameters than CNNs

#### Configuration

```python
config = {
    'modes': 32,              # Higher modes = more frequency detail
    'width': 128,             # Hidden dimension
    'depth': 6,               # Number of FNO layers
    'input_dim': 1,           # Single channel input
    'output_dim': 256,        # 256-class classification
    'activation': 'gelu',     # Activation function
    'dropout': 0.1,           # Dropout rate
    'normalization': 'layer'  # Normalization type
}

fno = FourierNeuralOperator(**config)
```

### DeepOperatorNetwork

DeepONet implementation for operator learning between input functions and outputs.

```python
from neural_cryptanalysis.neural_operators import DeepOperatorNetwork

deeponet = DeepOperatorNetwork(
    branch_net=[512, 512, 512],  # Branch network architecture
    trunk_net=[256, 256, 256],   # Trunk network architecture  
    output_dim=256               # Output dimension
)
```

#### Architecture Components

- **Branch Network**: Processes trace segments
- **Trunk Network**: Processes temporal coordinates
- **Operator Layer**: Combines branch and trunk outputs

### GraphNeuralOperator

Graph-based neural operator for multi-modal sensor fusion.

```python
from neural_cryptanalysis.neural_operators import GraphNeuralOperator

graph_op = GraphNeuralOperator(
    node_features=64,     # Node feature dimension
    edge_features=32,     # Edge feature dimension
    hidden_dim=128,       # Hidden dimension
    n_layers=4,           # Number of graph layers
    attention_heads=8     # Multi-head attention
)
```

---

## Side-Channel Analysis

### SideChannelAnalyzer

Base class for side-channel analysis with various attack strategies.

```python
from neural_cryptanalysis.side_channels import SideChannelAnalyzer

analyzer = SideChannelAnalyzer(
    channel_type='power',
    attack_type='neural',
    config=analysis_config
)
```

#### Supported Channel Types

| Channel | Description | Typical Use |
|---------|-------------|-------------|
| `power` | Power consumption analysis | Most common SCA |
| `em_near` | Near-field EM emanations | High spatial resolution |
| `em_far` | Far-field EM emanations | Covert analysis |
| `acoustic` | Acoustic emanations | CPU operations |
| `optical` | Optical emanations | LED/display analysis |
| `timing` | Execution timing | Cache attacks |

#### Analysis Configuration

```python
from neural_cryptanalysis.side_channels import AnalysisConfig, ChannelType

config = AnalysisConfig(
    channel_type=ChannelType.POWER,
    sample_rate=1e6,              # 1 MHz sampling
    trace_length=10000,           # 10k samples per trace
    n_traces=50000,               # 50k traces for training
    preprocessing=['standardize', 'filter'],
    poi_method='mutual_information',
    n_pois=200,                   # Top 200 points of interest
    confidence_threshold=0.95
)
```

### PowerAnalyzer

Specialized analyzer for power consumption side-channels.

```python
from neural_cryptanalysis.side_channels import PowerAnalyzer

power_analyzer = PowerAnalyzer(
    amplifier_gain=100,
    frequency_range=(1e3, 1e6),
    current_probe_type='magnetic'
)
```

### EMAnalyzer

Electromagnetic emanation analysis with spatial modeling.

```python
from neural_cryptanalysis.side_channels import EMAnalyzer

em_analyzer = EMAnalyzer(
    probe_type='loop_antenna',
    frequency_range=(10e6, 1e9),
    spatial_resolution=0.1,  # mm
    near_field=True
)
```

---

## Target Implementations

### Post-Quantum Cryptography

#### KyberImplementation

NIST Kyber lattice-based key encapsulation mechanism.

```python
from neural_cryptanalysis.targets import KyberImplementation

kyber = KyberImplementation(
    variant='kyber768',           # Security level
    platform='arm_cortex_m4',     # Target platform
    countermeasures=['shuffling', 'masking'],
    optimization_level='O2'
)
```

**Supported Variants:**
- `kyber512` - Security level 1
- `kyber768` - Security level 3  
- `kyber1024` - Security level 5

**Vulnerable Operations:**
- Number Theoretic Transform (NTT)
- Polynomial multiplication
- Sampling operations
- Key generation

**Example Analysis:**
```python
# Analyze NTT operations
ntt_traces = kyber.collect_ntt_traces(n_traces=10000)
ntt_analyzer = NeuralSCA(architecture='fourier_neural_operator')
ntt_model = ntt_analyzer.train(ntt_traces.power, ntt_traces.intermediate_values)

# Attack key generation
keygen_attack = ntt_analyzer.attack(
    test_traces=keygen_traces,
    model=ntt_model,
    target_operation='ntt_coefficient_recovery'
)
```

#### DilithiumImplementation

NIST Dilithium lattice-based digital signature scheme.

```python
from neural_cryptanalysis.targets import DilithiumImplementation

dilithium = DilithiumImplementation(
    variant='dilithium3',
    side_channel_protection='basic'
)
```

#### ClassicMcElieceImplementation

Classic McEliece code-based key encapsulation.

```python
from neural_cryptanalysis.targets import ClassicMcElieceImplementation

mceliece = ClassicMcElieceImplementation(
    parameters='mceliece460896',
    decoder='patterson'
)
```

### Classical Cryptography

#### AESImplementation

Advanced Encryption Standard implementation analysis.

```python
from neural_cryptanalysis.targets import AESImplementation

aes = AESImplementation(
    key_size=128,
    implementation='table_lookup',
    countermeasures=['boolean_masking'],
    masking_order=1
)
```

---

## Datasets

### SyntheticDatasetGenerator

Generate realistic synthetic datasets for training and validation.

```python
from neural_cryptanalysis.datasets import SyntheticDatasetGenerator

generator = SyntheticDatasetGenerator(
    device_model='stm32f4',
    noise_characteristics='measured',
    random_seed=42
)
```

#### Dataset Generation

##### AES Datasets

```python
# Generate AES power traces
aes_traces, aes_labels = generator.generate_aes_dataset(
    n_traces=50000,
    operation='sbox_lookup',
    key_bytes=[0, 1, 2, 3],      # Target key bytes
    snr_db=15,                   # Signal-to-noise ratio
    trace_length=2000,           # Samples per trace
    alignment_jitter=3           # Random temporal shift
)
```

##### Post-Quantum Datasets

```python
# Generate Kyber NTT traces
kyber_traces, kyber_labels = generator.generate_kyber_dataset(
    n_traces=20000,
    operation='ntt_butterfly',
    polynomial_coefficients=True,
    noise_model='realistic'
)
```

#### Dataset Format

Datasets are returned as `TraceData` objects:

```python
@dataclass
class TraceData:
    traces: np.ndarray           # Shape: [n_traces, trace_length]
    labels: np.ndarray           # Shape: [n_traces]
    metadata: Dict[str, Any]     # Metadata dict
    plaintexts: Optional[np.ndarray]  # Input plaintexts
    keys: Optional[np.ndarray]   # Cryptographic keys
    intermediate_values: Optional[np.ndarray]  # Target values
```

### DatasetLoader

Load and preprocess existing datasets.

```python
from neural_cryptanalysis.datasets import DatasetLoader

loader = DatasetLoader()

# Load DPA contest data
dpa_v4 = loader.load_dpa_contest('v4.1')

# Load custom traces
custom_data = loader.load_traces(
    'path/to/traces.npz',
    format='numpy',
    preprocessing=['standardize', 'align']
)
```

---

## Visualization

### TraceVisualizer

Comprehensive trace visualization and analysis tools.

```python
from neural_cryptanalysis.visualization import TraceVisualizer

visualizer = TraceVisualizer(
    style='publication',
    figure_size=(12, 8),
    dpi=300
)
```

#### Visualization Methods

##### Trace Plots

```python
# Plot individual traces
visualizer.plot_traces(
    traces=sample_traces[:10],
    labels=sample_labels[:10],
    title="Power Consumption Traces",
    highlight_pois=True
)

# Plot average traces by key byte
visualizer.plot_average_traces(
    traces=traces,
    labels=labels,
    group_by='key_byte_0',
    overlay=True
)
```

##### Statistical Analysis

```python
# SNR analysis
snr_plot = visualizer.plot_snr(
    traces=traces,
    labels=labels,
    window_size=100,
    highlight_threshold=0.5
)

# Correlation analysis
corr_plot = visualizer.plot_correlation_matrix(
    features=poi_features,
    method='pearson'
)
```

##### Frequency Domain Analysis

```python
# Spectral analysis
spectrum_plot = visualizer.plot_spectrum(
    traces=traces,
    frequency_range=(1e3, 1e6),
    window='hann'
)

# Spectrogram
spectrogram = visualizer.plot_spectrogram(
    trace=example_trace,
    nperseg=256,
    overlap=128
)
```

##### Neural Operator Visualization

```python
# Attention weights
attention_plot = visualizer.plot_attention_weights(
    model=fno_model,
    trace=test_trace,
    layer_idx=2
)

# Feature maps
feature_plot = visualizer.plot_feature_maps(
    model=fno_model,
    trace=test_trace,
    n_features=16
)
```

### AttackDashboard

Real-time attack monitoring and visualization.

```python
from neural_cryptanalysis.visualization import AttackDashboard

dashboard = AttackDashboard()

# Start live monitoring
dashboard.monitor_attack(
    attack_instance=neural_attack,
    update_interval=1.0,  # seconds
    metrics=['success_rate', 'key_bytes_recovered', 'confidence']
)
```

---

## Utilities

### Configuration Management

#### ConfigManager

Centralized configuration management with validation.

```python
from neural_cryptanalysis.utils import ConfigManager

config = ConfigManager()

# Load configuration
config.load('config/production.yaml')

# Access nested values
neural_config = config.get('neural_operators.fno')
device = config.get('device', default='cpu')

# Validate configuration
config.validate_schema('schemas/neural_sca.json')
```

#### Configuration Schema

```yaml
# Example configuration file
neural_operators:
  fno:
    modes: 32
    width: 128
    depth: 4
    activation: 'gelu'
  
side_channels:
  power:
    sample_rate: 1e6
    amplifier_gain: 100
    filter:
      type: 'butterworth'
      order: 4
      cutoff: 1e5

training:
  batch_size: 64
  learning_rate: 1e-3
  epochs: 100
  early_stopping:
    patience: 10
    delta: 1e-4

device: 'cuda'
precision: 'float32'
random_seed: 42
```

### Performance Profiling

#### PerformanceProfiler

Profile neural operator performance and resource usage.

```python
from neural_cryptanalysis.utils import PerformanceProfiler

profiler = PerformanceProfiler()

with profiler.profile('training'):
    model = neural_sca.train(traces, labels)

# Get detailed results
results = profiler.get_results()
print(f"Training time: {results['training']['total_time']:.2f}s")
print(f"Peak memory: {results['training']['peak_memory_mb']:.1f}MB")
print(f"GPU utilization: {results['training']['gpu_utilization']:.1%}")
```

### Statistical Validation

#### StatisticalValidator

Validate research results with statistical rigor.

```python
from neural_cryptanalysis.utils import StatisticalValidator

validator = StatisticalValidator()

# Compare two methods
comparison = validator.compare_methods(
    method_a_results=traditional_sca_results,
    method_b_results=neural_operator_results,
    metric='success_rate',
    alpha=0.05
)

print(f"Statistically significant improvement: {comparison.significant}")
print(f"Effect size (Cohen's d): {comparison.effect_size:.3f}")
print(f"P-value: {comparison.p_value:.6f}")
```

### Error Handling

#### Custom Exceptions

```python
from neural_cryptanalysis.utils.errors import (
    NeuralCryptanalysisError,
    ValidationError,
    ModelError,
    DataError
)

try:
    model = neural_sca.train(invalid_traces, labels)
except ValidationError as e:
    print(f"Validation error: {e.message}")
    print(f"Field: {e.field}")
    print(f"Value: {e.value}")
except ModelError as e:
    print(f"Model error: {e.message}")
    print(f"Model state: {e.model_state}")
```

---

## Configuration

### Global Configuration

Set global framework defaults:

```python
import neural_cryptanalysis as nc

# Configure logging
nc.configure_logging(
    level='INFO',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    file='neural_sca.log'
)

# Set default device
nc.set_default_device('cuda')

# Configure random seed for reproducibility
nc.set_random_seed(42)

# Configure warnings
nc.configure_warnings(
    filter_level='once',
    responsible_use_notice=True
)
```

### Environment Variables

Control framework behavior via environment variables:

```bash
# Device selection
export NEURAL_SCA_DEVICE="cuda"

# Logging configuration  
export NEURAL_SCA_LOG_LEVEL="DEBUG"
export NEURAL_SCA_LOG_FILE="/var/log/neural_sca.log"

# Performance settings
export NEURAL_SCA_NUM_WORKERS="8"
export NEURAL_SCA_MEMORY_LIMIT="16GB"

# Security settings
export NEURAL_SCA_AUDIT_LOG="true"
export NEURAL_SCA_DEFENSIVE_ONLY="true"
```

---

## Examples

### Basic Usage

```python
from neural_cryptanalysis import NeuralSCA, LeakageSimulator
from neural_cryptanalysis.targets import AESImplementation

# 1. Setup target implementation
aes = AESImplementation(key_size=128, implementation='table_lookup')

# 2. Simulate leakage data
simulator = LeakageSimulator(device_model='atmega328p')
traces = simulator.simulate_traces(
    target=aes,
    n_traces=10000,
    operations=['sbox_lookup'],
    snr_db=10
)

# 3. Train neural operator
neural_sca = NeuralSCA(architecture='fourier_neural_operator')
model = neural_sca.train(
    traces=traces.power,
    labels=traces.intermediate_values,
    epochs=50
)

# 4. Perform attack
attack_results = neural_sca.attack(
    test_traces=test_traces,
    model=model,
    strategy='template'
)

print(f"Attack successful: {attack_results.success}")
print(f"Key recovered: {attack_results.recovered_key.hex()}")
```

### Advanced Multi-Modal Analysis

```python
from neural_cryptanalysis import NeuralSCA
from neural_cryptanalysis.multi_modal_fusion import MultiModalAnalyzer

# Multi-modal setup
analyzer = MultiModalAnalyzer(
    channels=['power', 'em_near', 'em_far'],
    fusion_strategy='graph_attention'
)

# Collect synchronized traces
multi_traces = analyzer.collect_synchronized_traces(
    target=kyber_implementation,
    n_traces=5000
)

# Train fusion model
fusion_model = analyzer.train_fusion_model(
    traces=multi_traces,
    architecture='graph_neural_operator'
)

# Perform multi-modal attack
fusion_attack = analyzer.attack(
    test_traces=test_multi_traces,
    model=fusion_model,
    fusion_weights='adaptive'
)
```

### Countermeasure Evaluation

```python
from neural_cryptanalysis import NeuralSCA
from neural_cryptanalysis.countermeasures import MaskingEvaluator

# Evaluate Boolean masking
evaluator = MaskingEvaluator()
masked_aes = AESImplementation(
    countermeasures=['boolean_masking'],
    masking_order=2
)

# Test with increasing trace counts
n_traces_list = [1000, 5000, 10000, 50000, 100000]
results = []

for n_traces in n_traces_list:
    traces = collect_traces(masked_aes, n_traces)
    
    attack_result = neural_sca.attack(
        test_traces=traces,
        model=model,
        strategy='adaptive_template'
    )
    
    results.append({
        'n_traces': n_traces,
        'success_rate': attack_result.success_rate,
        'confidence': attack_result.confidence
    })

# Analyze masking effectiveness
security_analysis = evaluator.analyze_security_order(results)
print(f"Effective masking order: {security_analysis.effective_order}")
print(f"Security margin: {security_analysis.security_margin}")
```

---

## Version Information

- **API Version**: 1.0.0
- **Framework Version**: 3.0.0
- **Python Compatibility**: 3.9+
- **PyTorch Compatibility**: 1.12+
- **Last Updated**: 2025-01-12

## Support

For API support and questions:
- Documentation: https://neural-cryptanalysis.readthedocs.io
- Issues: https://github.com/neural-cryptanalysis/issues
- Discussions: https://github.com/neural-cryptanalysis/discussions

## License

GPL-3.0 License - See LICENSE file for details.

**Responsible Use Notice**: This API is designed for defensive security research only. Users must follow responsible disclosure practices and obtain proper authorization before testing on any systems.