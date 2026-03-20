# Neural Operator Cryptanalysis Lab

Neural operators for side-channel leakage modeling and adaptive attacks against post-quantum cryptography.

## Features
- **PowerTraceSimulator**: AES power trace simulation with Hamming weight leakage
- **NeuralOperatorLeakageModel**: FNO-inspired spectral convolution for leakage prediction
- **CorrelationPowerAnalysis**: Classical CPA baseline attack
- **AdaptiveAttacker**: Model-guided trace selection for efficient attacks

## Install

```bash
pip install numpy
```

## Usage

```python
from noclab.power_trace import PowerTraceSimulator
from noclab.cpa import CorrelationPowerAnalysis
import numpy as np

sim = PowerTraceSimulator(noise_std=0.05)
plaintexts = np.random.randint(0, 256, 100, dtype=np.uint8)
traces = sim.simulate(plaintexts, key_byte=0x2B)
cpa = CorrelationPowerAnalysis()
key = cpa.attack(traces, plaintexts)
print(f"Recovered key byte: {key:#04x}")
```

## Run Tests

```bash
~/anaconda3/bin/python3 -m pytest tests/ -v
```
