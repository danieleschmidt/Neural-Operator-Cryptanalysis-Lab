# Neural-Operator-Cryptanalysis-Lab 🔐🧠⚡

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-GPL%20v3-red.svg)](LICENSE)
[![ScienceDirect](https://img.shields.io/badge/Paper-ScienceDirect-orange.svg)](https://www.sciencedirect.com)

Applies neural operators to model side-channel leakages (power, EM) and synthesize adaptive attacks against post-quantum cryptographic implementations.

## ⚠️ Responsible Disclosure

This tool is for defensive security research only. We follow responsible disclosure practices and work with cryptographic implementers to improve security. See [SECURITY.md](SECURITY.md) for our disclosure policy.

## 🌟 Key Features

- **Neural Operator Learning**: Model complex side-channel leakage patterns
- **Post-Quantum Focus**: Specialized attacks on lattice, code, and hash-based schemes
- **Adaptive Attack Synthesis**: ML-driven attack parameter optimization
- **Multi-Channel Fusion**: Combine power, EM, acoustic, and optical emanations
- **Defense Evaluation**: Test countermeasure effectiveness
- **Hardware Modeling**: Accurate device-specific leakage simulation

## 🚀 Quick Start

### Installation

```bash
# Install core package
pip install neural-operator-cryptanalysis

# With hardware interfaces
pip install neural-operator-cryptanalysis[hardware]

# Development installation
git clone https://github.com/yourusername/Neural-Operator-Cryptanalysis-Lab.git
cd Neural-Operator-Cryptanalysis-Lab
pip install -e ".[dev,research]"
```

### Basic Usage

```python
from neural_cryptanalysis import NeuralSCA, LeakageSimulator
from neural_cryptanalysis.targets import KyberImplementation

# Load target implementation
target = KyberImplementation(
    version='kyber768',
    platform='arm_cortex_m4',
    countermeasures=['shuffling', 'masking']
)

# Simulate leakage traces
simulator = LeakageSimulator(
    device_model='stm32f4',
    noise_model='realistic'
)

traces = simulator.simulate_traces(
    target=target,
    n_traces=10000,
    operations=['ntt', 'polynomial_mul', 'sampling']
)

# Train neural operator
neural_sca = NeuralSCA(
    architecture='fourier_neural_operator',
    channels=['power', 'em_near_field']
)

model = neural_sca.train(
    traces=traces,
    labels=traces.intermediate_values,
    validation_split=0.2
)

# Perform attack
attack_results = neural_sca.attack(
    target_traces=real_traces,
    model=model,
    strategy='adaptive_template'
)

print(f"Key recovery success: {attack_results.success}")
print(f"Traces required: {attack_results.n_traces_needed}")
print(f"Confidence: {attack_results.confidence:.2%}")
```

## 🏗️ Architecture

```
neural-operator-cryptanalysis-lab/
├── neural_operators/       # Neural operator architectures
│   ├── fno/               # Fourier Neural Operators
│   │   ├── layers.py      # Spectral convolution layers
│   │   ├── models.py      # FNO architectures
│   │   └── training.py    # Training utilities
│   ├── deeponet/          # Deep Operator Networks
│   ├── mgno/              # Multipole Graph NO
│   └── custom/            # Custom architectures
├── side_channels/         # Side-channel modeling
│   ├── power/             # Power analysis
│   │   ├── models.py      # Power consumption models
│   │   ├── preprocessing.py
│   │   └── attacks.py     
│   ├── electromagnetic/   # EM analysis
│   ├── acoustic/          # Acoustic cryptanalysis
│   ├── optical/           # Optical emanations
│   └── fusion/            # Multi-channel fusion
├── targets/               # Cryptographic targets
│   ├── post_quantum/      # PQC implementations
│   │   ├── lattice/       # Kyber, Dilithium, etc.
│   │   ├── code_based/    # Classic McEliece
│   │   ├── hash_based/    # SPHINCS+
│   │   └── isogeny/       # SIKE (historical)
│   ├── classical/         # AES, RSA, ECC
│   └── implementations/   # Real implementations
├── attacks/               # Attack strategies
│   ├── template/          # Template attacks
│   ├── profiling/         # Profiling attacks
│   ├── collision/         # Collision attacks
│   ├── horizontal/        # Horizontal attacks
│   └── adaptive/          # ML-driven adaptation
├── countermeasures/       # Defense mechanisms
│   ├── masking/           # Boolean/arithmetic masking
│   ├── hiding/            # Hiding techniques
│   ├── shuffling/         # Operation shuffling
│   └── evaluation/        # Countermeasure testing
├── hardware/              # Hardware interfaces
│   ├── oscilloscopes/     # Scope interfaces
│   ├── probes/            # EM/power probes
│   ├── boards/            # Target boards
│   └── automation/        # Measurement automation
└── analysis/              # Analysis tools
    ├── visualization/     # Trace visualization
    ├── statistics/        # Statistical tests
    ├── reporting/         # Attack reports
    └── benchmarks/        # Performance metrics
```

## 🧠 Neural Operator Models

### Fourier Neural Operator for Side-Channels

```python
from neural_cryptanalysis.neural_operators import SideChannelFNO
import torch

class LeakageFNO(torch.nn.Module):
    def __init__(self, modes=16, width=64):
        super().__init__()
        self.modes = modes
        self.width = width
        
        # Fourier layer
        self.fourier = FourierLayer(self.modes, self.width)
        
        # Lift to higher dimension
        self.fc0 = torch.nn.Linear(2, self.width)  # input: trace + position
        
        # Spectral convolutions
        self.convs = torch.nn.ModuleList([
            SpectralConv1d(self.width, self.width, self.modes)
            for _ in range(4)
        ])
        
        # Non-linearity
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.width, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, self.width)
        )
        
        # Project to output
        self.fc1 = torch.nn.Linear(self.width, 256)  # 256 possible byte values
        
    def forward(self, x):
        # x shape: (batch, length, channels)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)  # (batch, channels, length)
        
        for conv in self.convs:
            x1 = conv(x)
            x2 = self.mlp(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = x1 + x2
            x = torch.nn.functional.gelu(x)
        
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        
        # Global pooling for key byte prediction
        return x.mean(dim=1)

# Train on leakage traces
model = LeakageFNO(modes=32, width=128)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    for traces, labels in dataloader:
        # traces: side-channel measurements
        # labels: secret key bytes
        
        pred = model(traces)
        loss = torch.nn.functional.cross_entropy(pred, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Adaptive Attack Strategy

```python
from neural_cryptanalysis.attacks import AdaptiveNeuralAttack

class AdaptiveAttacker:
    def __init__(self, neural_operator, target_impl):
        self.operator = neural_operator
        self.target = target_impl
        self.attack_params = self.initialize_params()
        
    def initialize_params(self):
        return {
            'window_size': 1000,
            'window_offset': 0,
            'preprocessing': 'standardize',
            'poi_selection': 'mutual_information',
            'n_pois': 50
        }
    
    def adapt_parameters(self, traces, success_rate):
        """Use reinforcement learning to adapt attack parameters"""
        # Analyze trace characteristics
        snr = self.compute_snr(traces)
        alignment = self.check_alignment(traces)
        
        # Adapt window parameters
        if snr < 0.1:
            self.attack_params['window_size'] *= 1.5
            self.attack_params['n_pois'] *= 2
        
        if alignment < 0.8:
            self.enable_alignment_correction()
        
        # Try different preprocessing
        if success_rate < 0.5:
            self.attack_params['preprocessing'] = 'filtering'
            self.attack_params['filter_params'] = {
                'type': 'butterworth',
                'order': 4,
                'cutoff': self.estimate_clock_frequency(traces) * 2
            }
        
        return self.attack_params
    
    def progressive_attack(self, traces):
        """Progressively refine attack using neural operator predictions"""
        recovered_bytes = {}
        confidence_threshold = 0.9
        
        for byte_idx in range(self.target.key_length):
            # Focus neural operator on specific byte
            byte_operator = self.operator.specialize(byte_idx)
            
            # Initial attack
            predictions = byte_operator.predict(traces[:100])
            
            # Refine with more traces if needed
            n_traces = 100
            while predictions.max_confidence() < confidence_threshold:
                n_traces = min(n_traces * 2, len(traces))
                predictions = byte_operator.predict(traces[:n_traces])
                
                # Adapt attack based on partial success
                self.adapt_parameters(traces[:n_traces], 
                                    predictions.max_confidence())
            
            recovered_bytes[byte_idx] = predictions.most_likely()
            
        return recovered_bytes
```

## 🔬 Post-Quantum Cryptanalysis

### Lattice-Based Schemes (Kyber)

```python
from neural_cryptanalysis.targets.pqc import KyberAnalyzer

kyber = KyberAnalyzer(variant='kyber768')

# Target NTT operations
class NTTLeakageModel(torch.nn.Module):
    def __init__(self, n=256, q=3329):
        super().__init__()
        self.n = n
        self.q = q
        
        # Model butterfly operations in NTT
        self.butterfly_model = NeuralButterfly(
            stages=int(np.log2(n)),
            modulus=q
        )
        
    def forward(self, power_trace):
        # Extract features specific to NTT patterns
        features = self.extract_ntt_features(power_trace)
        
        # Predict intermediate values
        butterflies = self.butterfly_model(features)
        
        # Recover coefficients
        coefficients = self.inverse_ntt_recovery(butterflies)
        
        return coefficients
    
    def extract_ntt_features(self, trace):
        # Identify modular multiplication patterns
        patterns = self.find_modular_patterns(trace)
        
        # Extract timing of butterfly operations  
        timings = self.extract_butterfly_timings(trace)
        
        return torch.cat([patterns, timings], dim=-1)

# Attack Kyber key generation
def attack_kyber_keygen(traces, labels=None):
    model = NTTLeakageModel()
    
    if labels is not None:
        # Supervised learning
        model = train_supervised(model, traces, labels)
    else:
        # Unsupervised learning with self-supervision
        model = train_selfsupervised(model, traces)
    
    # Recover secret key
    recovered_polynomials = []
    for trace in test_traces:
        coeffs = model(trace)
        recovered_polynomials.append(coeffs)
    
    # Reconstruct full key
    secret_key = kyber.reconstruct_key(recovered_polynomials)
    
    return secret_key
```

### Code-Based Schemes (Classic McEliece)

```python
from neural_cryptanalysis.targets.pqc import McElieceAnalyzer

mceliece = McElieceAnalyzer(params='mceliece460896')

class SyndromeLeakageNN(torch.nn.Module):
    """Neural network for syndrome decoding side-channels"""
    
    def __init__(self, n=4608, t=96):
        super().__init__()
        self.n = n
        self.t = t
        
        # Model Berlekamp-Massey operations
        self.bm_model = DeepONet(
            branch_net=[512, 512, 512],
            trunk_net=[256, 256, 256]
        )
        
        # Model Euclidean algorithm
        self.gcd_model = RecurrentNeuralOperator(
            hidden_size=256,
            n_layers=4
        )
        
    def forward(self, em_trace):
        # Identify algorithm phases
        phases = self.segment_trace(em_trace)
        
        # Extract features from each phase
        bm_features = self.bm_model(phases['berlekamp_massey'])
        gcd_features = self.gcd_model(phases['euclidean'])
        
        # Predict error positions
        error_positions = self.predict_errors(bm_features, gcd_features)
        
        return error_positions

# Attack Classic McEliece decryption
attacker = SyndromeLeakageNN()
error_positions = attacker(em_measurements)

# Recover plaintext
plaintext = mceliece.decode_with_errors(ciphertext, error_positions)
```

## 📊 Multi-Channel Analysis

### Sensor Fusion

```python
from neural_cryptanalysis.fusion import MultiChannelFusion

class FusionAttack:
    def __init__(self, channels):
        self.channels = channels
        self.fusion_model = self.build_fusion_model()
        
    def build_fusion_model(self):
        return MultiModalNeuralOperator(
            modalities={
                'power': FNO1d(width=64),
                'em_near': FNO2d(width=32),  # 2D for spatial
                'em_far': FNO1d(width=32),
                'acoustic': WaveNet(layers=10)
            },
            fusion_strategy='attention',
            output_dim=256  # Key space
        )
    
    def collect_synchronized_traces(self, n_traces=1000):
        """Collect time-synchronized multi-channel traces"""
        traces = {ch: [] for ch in self.channels}
        
        with SynchronizedCapture(self.channels) as capture:
            for _ in range(n_traces):
                # Trigger crypto operation
                trigger.arm()
                target.encrypt(random_plaintext())
                
                # Capture all channels
                multi_trace = capture.acquire()
                
                for ch in self.channels:
                    traces[ch].append(multi_trace[ch])
        
        return traces
    
    def joint_analysis(self, traces):
        """Jointly analyze all channels"""
        # Align traces
        aligned = self.align_channels(traces)
        
        # Extract channel-specific features
        features = {}
        for channel, channel_traces in aligned.items():
            features[channel] = self.extract_features(
                channel_traces,
                method=self.get_optimal_method(channel)
            )
        
        # Fusion through neural operator
        fused_prediction = self.fusion_model(features)
        
        return fused_prediction
```

### Advanced EM Analysis

```python
from neural_cryptanalysis.em import EMFieldAnalyzer

class SpatialEMAnalysis:
    def __init__(self, probe_array_config):
        self.probes = probe_array_config
        self.field_model = EMFieldNeuralOperator(
            spatial_resolution=0.1,  # mm
            frequency_range=(1e6, 1e9)  # Hz
        )
    
    def capture_em_field(self, target_position):
        """Capture EM field using probe array"""
        field_data = np.zeros((self.probes.n_x, self.probes.n_y, 
                              self.probes.n_samples))
        
        for i, j in itertools.product(range(self.probes.n_x), 
                                     range(self.probes.n_y)):
            # Position probe
            self.probes.move_to(i, j)
            
            # Capture EM emanations
            trace = self.probes.capture()
            field_data[i, j, :] = trace
        
        return field_data
    
    def locate_leakage_source(self, field_data):
        """Find physical location of leakage"""
        # Use neural operator to model field propagation
        sources = self.field_model.inverse_problem(field_data)
        
        # Identify crypto components
        components = self.map_to_layout(sources, 'chip_layout.gds')
        
        return components
    
    def optimize_probe_placement(self, initial_field):
        """Find optimal probe positions"""
        optimizer = ProbeOptimizer(self.field_model)
        
        optimal_positions = optimizer.optimize(
            initial_field,
            n_probes=4,
            objective='maximize_snr',
            constraints=['physical_access', 'mutual_interference']
        )
        
        return optimal_positions
```

## 🛡️ Countermeasure Evaluation

### Testing Masking Schemes

```python
from neural_cryptanalysis.countermeasures import MaskingEvaluator

evaluator = MaskingEvaluator()

# Test Boolean masking
def evaluate_boolean_masking(implementation, masking_order=2):
    # Theoretical security
    theory_security = evaluator.compute_theoretical_security(
        masking_order,
        noise_variance=implementation.noise_model.variance
    )
    
    # Practical evaluation with neural operators
    practical_eval = NeuralMaskingAttack(
        target=implementation,
        max_order=masking_order + 2
    )
    
    # Collect traces with increasing amounts
    n_traces = [1e3, 1e4, 1e5, 1e6, 1e7]
    success_rates = []
    
    for n in n_traces:
        traces = collect_traces(implementation, int(n))
        
        # Apply neural operator attack
        attack_result = practical_eval.attack(traces)
        success_rates.append(attack_result.success_rate)
        
        # Check if masking is broken
        if attack_result.success_rate > 0.5:
            print(f"Masking broken with {n:.0e} traces!")
            print(f"Effective order: {attack_result.estimated_order}")
            break
    
    return {
        'theoretical_security': theory_security,
        'practical_security': n_traces[success_rates.index(max(success_rates))],
        'security_margin': theory_security / practical_eval.n_traces_needed
    }

# Test shuffling countermeasures
def evaluate_shuffling(implementation):
    shuffling_attack = ShufflingNeuralAttack(
        sequence_model='transformer',
        max_permutations=implementation.shuffling_space
    )
    
    # Learn shuffling patterns
    pattern_model = shuffling_attack.learn_patterns(
        training_traces,
        n_epochs=50
    )
    
    # Attack with pattern knowledge
    results = shuffling_attack.attack_with_patterns(
        test_traces,
        pattern_model
    )
    
    return results
```

## 📈 Visualization & Analysis

### Leakage Visualization

```python
from neural_cryptanalysis.visualization import LeakageVisualizer

viz = LeakageVisualizer()

# SNR heatmap
snr_map = viz.compute_snr_map(
    traces,
    labels,
    window_size=100,
    overlap=50
)

viz.plot_snr_heatmap(
    snr_map,
    title="Signal-to-Noise Ratio",
    highlight_threshold=0.5
)

# Neural operator attention
attention_weights = model.get_attention_weights(test_trace)

viz.plot_attention(
    test_trace,
    attention_weights,
    title="Neural Operator Focus Areas",
    overlay_operations=['ntt_start', 'ntt_end', 'sampling']
)

# 3D leakage surface
viz.plot_3d_leakage(
    parameter_1=range(256),  # Key byte values
    parameter_2=range(1000),  # Time samples  
    leakage_function=lambda k, t: model.predict_leakage(k, t),
    title="Leakage Surface"
)
```

### Attack Metrics Dashboard

```python
from neural_cryptanalysis.analysis import AttackDashboard

dashboard = AttackDashboard()

# Real-time attack monitoring
dashboard.start_live_attack(
    attack=neural_attack,
    target=target_device,
    update_rate=10  # Hz
)

# Metrics tracked:
# - Success rate vs. number of traces
# - Confidence per key byte
# - Guessing entropy
# - Neural operator loss
# - Time remaining estimate

dashboard.add_panel('key_recovery', type='heatmap')
dashboard.add_panel('trace_quality', type='histogram')
dashboard.add_panel('attack_progress', type='timeline')
dashboard.add_panel('resource_usage', type='gauges')

# Export results
dashboard.export_report(
    format='latex',
    include_figures=True,
    template='conference_paper'
)
```

## 🔧 Hardware Setup

### Measurement Configuration

```python
from neural_cryptanalysis.hardware import MeasurementSetup

# Configure oscilloscope
scope = MeasurementSetup.create_scope(
    model='Picoscope_6404D',
    channels={
        'A': {'probe': 'current_probe', 'range': '100mV'},
        'B': {'probe': 'em_probe', 'range': '50mV'},
        'C': {'probe': 'trigger', 'range': '5V'}
    },
    sampling_rate=5e9,  # 5 GS/s
    memory_depth=1e6
)

# Configure target board
target = MeasurementSetup.create_target(
    board='CW308_STM32F4',
    clock=24e6,  # 24 MHz
    programmer='openocd'
)

# Automated measurement
automator = MeasurementAutomation(scope, target)

# Sweep attack parameters
results = automator.parameter_sweep({
    'clock_frequency': np.linspace(20e6, 30e6, 11),
    'voltage': np.linspace(2.8, 3.6, 9),
    'temperature': [20, 30, 40],
    'traces_per_point': 10000
})

# Find optimal attack point
optimal = results.get_optimal_parameters(metric='snr')
print(f"Optimal settings: {optimal}")
```

## 🚨 Responsible Use

### Disclosure Process

```python
from neural_cryptanalysis.disclosure import VulnerabilityReporter

reporter = VulnerabilityReporter()

# Document findings
vulnerability = reporter.create_report(
    title="Power Side-Channel in Kyber768 Implementation",
    severity="high",
    affected_versions=['1.0', '1.1'],
    attack_requirements={
        'physical_access': True,
        'traces_needed': 50000,
        'equipment_cost': '$5000',
        'expertise_level': 'moderate'
    },
    mitigation="Add first-order masking to NTT operations"
)

# Responsible disclosure
reporter.notify_vendor(
    vendor='crypto_library_maintainer',
    vulnerability=vulnerability,
    embargo_days=90
)

# Track disclosure timeline
reporter.track_remediation(vulnerability.id)
```

## 📚 Research & Citations

```bibtex
@article{neural_operator_sca2025,
  title={Neural Operators for Side-Channel Analysis of Post-Quantum Cryptography},
  author={Your Name et al.},
  journal={Journal of Cryptographic Engineering},
  year={2025}
}

@inproceedings{adaptive_pqc_attacks2024,
  title={Adaptive Neural Attacks on Masked Post-Quantum Implementations},
  author={Your Team},
  booktitle={CHES},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions in:
- New neural operator architectures
- Additional PQC targets
- Countermeasure evaluations
- Hardware interfaces

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

## ⚖️ License & Ethics

GPL v3 License - see [LICENSE](LICENSE)

This tool is for defensive research only. Users must:
- Follow responsible disclosure
- Not attack systems without permission
- Contribute defenses back to the community

## 🔗 Resources

- [Documentation](https://neural-cryptanalysis.readthedocs.io)
- [Tutorial Videos](https://youtube.com/neural-sca-lab)
- [PQC Implementation Database](https://pqc-sca.org)
- [Hardware Setup Guide](./docs/hardware_setup.md)
