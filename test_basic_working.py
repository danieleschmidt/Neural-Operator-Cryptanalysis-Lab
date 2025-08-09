#!/usr/bin/env python3
"""
Test basic functionality of Neural Cryptanalysis Lab
Generation 1: Make It Work
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import mock torch first
import simple_torch_mock

import numpy as np
from neural_cryptanalysis.core import NeuralSCA, LeakageSimulator
from neural_cryptanalysis.side_channels.base import TraceData

def test_basic_imports():
    """Test that all core modules can be imported."""
    try:
        from neural_cryptanalysis import NeuralSCA, LeakageSimulator
        from neural_cryptanalysis.neural_operators.base import NeuralOperatorBase
        from neural_cryptanalysis.side_channels.base import SideChannelAnalyzer
        print("✓ All core imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_trace_data_creation():
    """Test creating TraceData objects."""
    try:
        # Create simple trace data
        traces = np.random.randn(100, 1000)  # 100 traces, 1000 samples each
        labels = np.random.randint(0, 256, 100)  # Random labels
        
        trace_data = TraceData(traces=traces, labels=labels)
        print(f"✓ TraceData created with {len(trace_data)} traces")
        
        # Test indexing
        sample = trace_data[0]
        print(f"✓ Sample trace shape: {sample['trace'].shape}")
        return True
    except Exception as e:
        print(f"✗ TraceData creation failed: {e}")
        return False

def test_leakage_simulator():
    """Test basic leakage simulation."""
    try:
        simulator = LeakageSimulator(device_model='stm32f4')
        
        # Mock target for simulation
        class MockTarget:
            def __init__(self):
                self.key = np.random.randint(0, 256, 16, dtype=np.uint8)
            
            def compute_intermediate_values(self, plaintext):
                # Simple Hamming weight leakage model
                return np.array([bin(x ^ self.key[i % 16]).count('1') for i, x in enumerate(plaintext)])
        
        target = MockTarget()
        trace_data = simulator.simulate_traces(target, n_traces=10, trace_length=1000)
        
        print(f"✓ Simulated {len(trace_data)} traces")
        print(f"  Device model: {trace_data.metadata['device_model']}")
        return True
    except Exception as e:
        print(f"✗ Leakage simulation failed: {e}")
        return False

def test_neural_sca_initialization():
    """Test NeuralSCA initialization."""
    try:
        # Simple configuration
        config = {
            'operator': {
                'input_dim': 1000,
                'output_dim': 256,
                'hidden_dim': 64
            },
            'analysis': {
                'trace_length': 1000,
                'n_traces': 100
            }
        }
        
        neural_sca = NeuralSCA(
            architecture='side_channel_fno',
            channels=['power'],
            config=config
        )
        
        print("✓ NeuralSCA initialized successfully")
        return True
    except Exception as e:
        print(f"✗ NeuralSCA initialization failed: {e}")
        return False

def test_simple_attack_workflow():
    """Test a complete but simple attack workflow."""
    try:
        print("Testing simple attack workflow...")
        
        # 1. Create synthetic data
        simulator = LeakageSimulator()
        
        class SimpleTarget:
            def __init__(self):
                self.key = np.array([0x43] * 16, dtype=np.uint8)  # Fixed key for testing
            
            def compute_intermediate_values(self, plaintext):
                # S-box output (simplified)
                sbox_out = plaintext[0] ^ self.key[0]  # First byte only
                return np.array([bin(sbox_out).count('1')])  # Hamming weight
        
        target = SimpleTarget()
        
        # Generate training data
        train_data = simulator.simulate_traces(target, n_traces=50, trace_length=100)
        print(f"✓ Generated {len(train_data)} training traces")
        
        # 2. Initialize neural SCA
        neural_sca = NeuralSCA(
            architecture='side_channel_fno',
            config={
                'operator': {'input_dim': 100, 'output_dim': 256, 'hidden_dim': 32},
                'training': {'epochs': 1, 'batch_size': 16}  # Quick test
            }
        )
        
        # 3. Train (mock training for now)
        print("✓ Training initiated (mock)")
        
        # 4. Attack simulation
        test_data = simulator.simulate_traces(target, n_traces=10, trace_length=100)
        results = neural_sca.attack(test_data)
        
        print(f"✓ Attack completed with {len(results['predictions'])} predictions")
        print(f"  Average confidence: {results['avg_confidence']:.3f}")
        
        return True
    except Exception as e:
        print(f"✗ Attack workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧠 Neural Cryptanalysis Lab - Basic Functionality Test")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_trace_data_creation,
        test_leakage_simulator,
        test_neural_sca_initialization,
        test_simple_attack_workflow
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        try:
            if test():
                passed += 1
            else:
                print("  Test failed")
        except Exception as e:
            print(f"  Test error: {e}")
    
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic functionality tests PASSED!")
        print("Generation 1: MAKE IT WORK - COMPLETED")
        return True
    else:
        print("⚠️  Some tests failed - debugging needed")
        return False

if __name__ == "__main__":
    main()