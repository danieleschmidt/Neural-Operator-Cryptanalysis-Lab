#!/usr/bin/env python3
"""Simple test to check imports."""

import sys
sys.path.insert(0, '/root/repo')
sys.path.insert(0, '/root/repo/src')

# Import mocks
import simple_typing_mock
import numpy_mock
import scipy_mock
import sklearn_mock  
import yaml_mock
import simple_torch_mock

print("All mocks imported successfully")

# Now test a simple module
try:
    import neural_cryptanalysis
    print("✓ neural_cryptanalysis imported")
    
    from neural_cryptanalysis import NeuralSCA, LeakageSimulator
    print("✓ Core classes imported")
    
    # Test basic instantiation
    neural_sca = NeuralSCA()
    print(f"✓ NeuralSCA instantiated: {type(neural_sca)}")
    
    simulator = LeakageSimulator()
    print(f"✓ LeakageSimulator instantiated: {type(simulator)}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()