"""Demo: recover 1 AES key byte using CPA."""

import numpy as np
from .power_trace import PowerTraceSimulator
from .cpa import CorrelationPowerAnalysis


def demo_recover_key_byte() -> None:
    """Simulate 200 traces with a known key byte and recover it via CPA."""
    TRUE_KEY_BYTE = 0x2B
    N_TRACES = 200

    rng = np.random.default_rng(1234)
    plaintexts = rng.integers(0, 256, size=N_TRACES, dtype=np.uint8)

    sim = PowerTraceSimulator(noise_std=0.05, n_samples=500)
    traces = sim.simulate(plaintexts, key_byte=TRUE_KEY_BYTE, n_traces=N_TRACES)

    cpa = CorrelationPowerAnalysis()
    recovered = cpa.attack(traces, plaintexts)

    print(f"True key byte:      {TRUE_KEY_BYTE:#04x}")
    print(f"Recovered key byte: {recovered:#04x}")
    if recovered == TRUE_KEY_BYTE:
        print("✓ Key recovery SUCCESS")
    else:
        print("✗ Key recovery FAILED")


if __name__ == "__main__":
    demo_recover_key_byte()
