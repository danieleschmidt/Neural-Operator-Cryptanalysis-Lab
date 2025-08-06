#!/usr/bin/env python3
"""Basic functionality test for Neural Operator Cryptanalysis Lab."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import torch
from neural_cryptanalysis.neural_operators import OperatorConfig, FourierNeuralOperator
from neural_cryptanalysis.side_channels.base import TraceData
from neural_cryptanalysis.core import NeuralSCA, LeakageSimulator
from neural_cryptanalysis.targets.post_quantum import KyberImplementation
from neural_cryptanalysis.targets.base import ImplementationConfig


def test_neural_operators():
    """Test neural operator implementations."""
    print("Testing Neural Operators...")
    
    # Test FourierNeuralOperator
    config = OperatorConfig(
        input_dim=1,
        output_dim=256,
        hidden_dim=32,
        num_layers=2,
        device='cpu'
    )
    
    try:
        model = FourierNeuralOperator(config, modes=8)
        
        # Test forward pass
        x = torch.randn(4, 100, 1)
        output = model(x)
        
        assert output.shape == (4, 256), f"Expected (4, 256), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        
        print("✓ FourierNeuralOperator test passed")
        
    except Exception as e:
        print(f"✗ FourierNeuralOperator test failed: {e}")
        return False
    
    return True


def test_side_channel_analysis():
    """Test side-channel analysis components."""
    print("Testing Side-Channel Analysis...")
    
    try:
        # Create synthetic trace data
        n_traces = 100
        trace_length = 1000
        traces = np.random.randn(n_traces, trace_length)
        labels = np.random.randint(0, 256, n_traces)
        
        trace_data = TraceData(
            traces=traces,
            labels=labels,
            metadata={'sample_rate': 1e6}
        )
        
        # Test data access
        assert len(trace_data) == n_traces, "TraceData length mismatch"
        sample = trace_data[0]
        assert 'trace' in sample, "Missing trace in sample"
        assert 'label' in sample, "Missing label in sample"
        
        # Test data splitting
        train_data, test_data = trace_data.split(0.8)
        assert len(train_data) + len(test_data) == n_traces, "Split size mismatch"
        
        print("✓ Side-channel analysis test passed")
        
    except Exception as e:
        print(f"✗ Side-channel analysis test failed: {e}")
        return False
    
    return True


def test_target_implementations():
    """Test cryptographic target implementations."""
    print("Testing Target Implementations...")
    
    try:
        # Test Kyber implementation
        config = ImplementationConfig(
            algorithm='kyber',
            variant='kyber768',
            platform='generic'
        )
        
        kyber = KyberImplementation(config)
        
        # Test key generation
        public_key, secret_key = kyber.generate_key()
        assert public_key is not None, "Public key generation failed"
        assert secret_key is not None, "Secret key generation failed"
        
        # Test encryption/decryption
        message = np.random.randint(0, 256, 32, dtype=np.uint8)
        ciphertext = kyber.encrypt(message)
        decrypted = kyber.decrypt(ciphertext)
        
        assert isinstance(ciphertext, np.ndarray), "Ciphertext not numpy array"
        assert isinstance(decrypted, np.ndarray), "Decrypted not numpy array"
        
        # Test intermediate value computation
        intermediates = kyber.compute_intermediate_values(ciphertext)
        assert len(intermediates) > 0, "No intermediate values computed"
        
        print("✓ Target implementations test passed")
        
    except Exception as e:
        print(f"✗ Target implementations test failed: {e}")
        return False
    
    return True


def test_core_api():
    """Test core API functionality."""
    print("Testing Core API...")
    
    try:
        # Test LeakageSimulator
        simulator = LeakageSimulator(
            device_model='stm32f4',
            noise_model='realistic'
        )
        
        # Create a dummy target
        config = ImplementationConfig(
            algorithm='kyber',
            variant='kyber768'
        )
        target = KyberImplementation(config)
        target.generate_key()
        
        # Simulate traces
        synthetic_data = simulator.simulate_traces(
            target=target,
            n_traces=10,
            operations=['ntt'],
            trace_length=1000
        )
        
        assert len(synthetic_data) == 10, "Wrong number of traces"
        assert synthetic_data.traces.shape[1] == 1000, "Wrong trace length"
        
        print("✓ Core API test passed")
        
    except Exception as e:
        print(f"✗ Core API test failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Neural Operator Cryptanalysis Lab - Basic Functionality Test")
    print("=" * 60)
    
    tests = [
        test_neural_operators,
        test_side_channel_analysis,
        test_target_implementations,
        test_core_api,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The implementation is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)