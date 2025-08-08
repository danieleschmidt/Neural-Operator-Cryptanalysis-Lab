#!/usr/bin/env python3
"""
Basic Fourier Neural Operator Side-Channel Attack Example

This example demonstrates a simple power analysis attack using FNO
against a simulated AES implementation.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_cryptanalysis import NeuralSCA, LeakageSimulator
from neural_cryptanalysis.targets import AESImplementation
from neural_cryptanalysis.neural_operators import FourierNeuralOperator
from neural_cryptanalysis.utils.visualization import plot_attack_results


def generate_synthetic_traces(n_traces: int = 5000) -> tuple:
    """Generate synthetic power traces for AES encryption."""
    
    # Create AES target
    target = AESImplementation(
        version='aes128',
        platform='software',
        countermeasures=[]
    )
    
    # Initialize leakage simulator
    simulator = LeakageSimulator(
        device_model='generic_mcu',
        noise_model='gaussian',
        snr_db=10.0
    )
    
    print(f"Generating {n_traces} synthetic power traces...")
    
    # Generate random plaintexts and fixed key
    plaintexts = np.random.randint(0, 256, (n_traces, 16), dtype=np.uint8)
    key = np.array([0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
                   0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c], dtype=np.uint8)
    
    traces = []
    labels = []
    
    for i, plaintext in enumerate(plaintexts):
        # Simulate encryption with leakage
        trace, intermediate_values = simulator.simulate_aes_encryption(
            plaintext=plaintext,
            key=key,
            target=target
        )
        
        traces.append(trace)
        # Use first S-box output as label
        labels.append(intermediate_values['sbox_output'][0])
        
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1}/{n_traces} traces")
    
    return np.array(traces), np.array(labels), plaintexts, key


def train_fno_model(traces: np.ndarray, labels: np.ndarray) -> torch.nn.Module:
    """Train FNO model for side-channel attack."""
    
    print("Training Fourier Neural Operator...")
    
    # Initialize Neural SCA with FNO
    neural_sca = NeuralSCA(
        architecture='fourier_neural_operator',
        channels=['power'],
        config={
            'fno': {
                'modes': 16,
                'width': 64,
                'n_layers': 4
            },
            'training': {
                'batch_size': 128,
                'learning_rate': 1e-3,
                'epochs': 50
            }
        }
    )
    
    # Convert to tensors
    X = torch.tensor(traces, dtype=torch.float32).unsqueeze(-1)  # Add channel dim
    y = torch.tensor(labels, dtype=torch.long)
    
    # Train model
    model = neural_sca.train(
        traces=X,
        labels=y,
        validation_split=0.2
    )
    
    return model, neural_sca


def perform_attack(model, neural_sca, test_traces: np.ndarray, 
                  test_plaintexts: np.ndarray, true_key: np.ndarray) -> dict:
    """Perform key recovery attack using trained model."""
    
    print("Performing key recovery attack...")
    
    # Convert test traces to tensor
    X_test = torch.tensor(test_traces, dtype=torch.float32).unsqueeze(-1)
    
    # Attack first key byte
    attack_results = neural_sca.attack(
        target_traces=X_test,
        model=model,
        strategy='template',
        target_byte=0,
        plaintexts=test_plaintexts
    )
    
    # Calculate success metrics
    recovered_key_byte = attack_results['predicted_key_byte']
    true_key_byte = true_key[0]
    
    success = recovered_key_byte == true_key_byte
    confidence = attack_results['confidence']
    
    results = {
        'success': success,
        'recovered_key_byte': recovered_key_byte,
        'true_key_byte': true_key_byte,
        'confidence': confidence,
        'traces_used': len(test_traces),
        'attack_results': attack_results
    }
    
    return results


def main():
    """Main execution function."""
    
    print("🔐 Neural Operator Cryptanalysis - Basic FNO Attack Example")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate synthetic data
    traces, labels, plaintexts, key = generate_synthetic_traces(n_traces=5000)
    
    # Split data
    split_idx = int(0.8 * len(traces))
    train_traces = traces[:split_idx]
    train_labels = labels[:split_idx]
    test_traces = traces[split_idx:]
    test_plaintexts = plaintexts[split_idx:]
    
    print(f"Training set: {len(train_traces)} traces")
    print(f"Test set: {len(test_traces)} traces")
    
    # Train FNO model
    model, neural_sca = train_fno_model(train_traces, train_labels)
    
    # Perform attack
    results = perform_attack(model, neural_sca, test_traces, test_plaintexts, key)
    
    # Display results
    print("\n🎯 Attack Results:")
    print(f"Success: {'✅ YES' if results['success'] else '❌ NO'}")
    print(f"True key byte: 0x{results['true_key_byte']:02x}")
    print(f"Recovered key byte: 0x{results['recovered_key_byte']:02x}")
    print(f"Confidence: {results['confidence']:.4f}")
    print(f"Traces used: {results['traces_used']}")
    
    # Create visualization if matplotlib available
    try:
        plot_attack_results(results)
        plt.savefig('basic_fno_attack_results.png', dpi=300, bbox_inches='tight')
        print(f"\nResults visualization saved to: basic_fno_attack_results.png")
    except ImportError:
        print("Matplotlib not available - skipping visualization")
    
    print("\n✅ Example completed successfully!")
    
    return results


if __name__ == "__main__":
    main()