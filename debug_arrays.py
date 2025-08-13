#!/usr/bin/env python3

import sys
sys.path.insert(0, '/root/repo')

import numpy_mock as np

n_traces = 100
traces = np.random.randn(n_traces, 1000)
labels = np.random.randint(0, 256, (n_traces,))

print(f"traces shape: {traces.shape if hasattr(traces, 'shape') else len(traces)}")
print(f"labels shape: {labels.shape if hasattr(labels, 'shape') else len(labels)}")
print(f"traces length: {len(traces)}")
print(f"labels length: {len(labels)}")