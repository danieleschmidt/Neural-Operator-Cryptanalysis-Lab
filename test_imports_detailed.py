#!/usr/bin/env python3
"""Test script to identify missing imports and dependencies."""

import sys
import os
sys.path.insert(0, '/root/repo/src')

# Import the mocks first
sys.path.insert(0, '/root/repo')
import simple_typing_mock
import numpy_mock
import scipy_mock
import sklearn_mock
import yaml_mock
import simple_torch_mock

# Test individual modules
modules_to_test = [
    'neural_cryptanalysis.neural_operators.base',
    'neural_cryptanalysis.neural_operators.fno', 
    'neural_cryptanalysis.neural_operators.deeponet',
    'neural_cryptanalysis.neural_operators.custom',
    'neural_cryptanalysis.side_channels.base',
    'neural_cryptanalysis.core',
    'neural_cryptanalysis',
]

missing_dependencies = []
import_errors = []

for module in modules_to_test:
    try:
        print(f"Testing {module}...")
        __import__(module)
        print(f"✓ {module} imported successfully")
    except ImportError as e:
        print(f"✗ {module} failed: {e}")
        import_errors.append((module, str(e)))
    except Exception as e:
        print(f"✗ {module} failed with other error: {e}")
        import_errors.append((module, str(e)))

print("\n" + "="*50)
print("IMPORT ERROR SUMMARY:")
print("="*50)

for module, error in import_errors:
    print(f"{module}: {error}")