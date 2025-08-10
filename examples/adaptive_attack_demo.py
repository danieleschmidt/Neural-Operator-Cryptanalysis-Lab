#!/usr/bin/env python3
"""
Demonstration of Adaptive Attack Engine with Reinforcement Learning.

This example shows how to use the autonomous attack optimization system
to automatically discover optimal attack parameters for side-channel analysis.
"""

import sys
import os
import numpy as np
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_cryptanalysis.core import NeuralSCA, LeakageSimulator, TraceData
from neural_cryptanalysis.adaptive_rl import AdaptiveAttackEngine, MetaLearningAdaptiveEngine
from neural_cryptanalysis.targets.base import CryptographicTarget
from neural_cryptanalysis.datasets.synthetic import SyntheticDatasetGenerator
from neural_cryptanalysis.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

class SimpleAESTarget(CryptographicTarget):
    """Simple AES target for demonstration."""
    
    def __init__(self, key: bytes = None):
        super().__init__()
        self.key = key or b'\x2b\x7e\x15\x16\x28\xae\xd2\xa6\xab\xf7\x15\x88\x09\xcf\x4f\x3c'
        self.sbox = [
            0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
            0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
            0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
            0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
            0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
            0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
            0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
            0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
            0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
            0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
            0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
            0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
            0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
            0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
            0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
            0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
        ]
    
    def compute_intermediate_values(self, plaintext: np.ndarray) -> np.ndarray:
        """Compute AES S-box intermediate values."""
        if len(plaintext.shape) == 1:
            plaintext = plaintext.reshape(1, -1)
        
        intermediate_values = []
        for pt in plaintext:
            # First round: AddRoundKey + SubBytes
            state = pt[:16] if len(pt) >= 16 else np.pad(pt, (0, 16 - len(pt)))
            key_bytes = np.frombuffer(self.key[:16], dtype=np.uint8)
            
            # AddRoundKey
            after_ark = state ^ key_bytes
            
            # SubBytes (S-box)
            after_sbox = np.array([self.sbox[b] for b in after_ark])
            
            intermediate_values.append(after_sbox)
        
        return np.array(intermediate_values)

def demonstrate_basic_adaptive_attack():
    """Demonstrate basic adaptive attack optimization."""
    print("\n" + "="*60)
    print("ADAPTIVE ATTACK ENGINE - BASIC DEMONSTRATION")
    print("="*60)
    
    # Create target
    target = SimpleAESTarget()
    
    # Generate synthetic traces
    simulator = LeakageSimulator(device_model='stm32f4', noise_model='realistic')
    traces = simulator.simulate_traces(
        target=target,
        n_traces=2000,
        operations=['sbox'],
        trace_length=5000
    )
    
    print(f"Generated {len(traces)} traces with {len(traces.traces[0])} samples each")
    
    # Initialize neural SCA
    neural_sca = NeuralSCA(architecture='fourier_neural_operator')
    
    # Create adaptive engine
    adaptive_engine = AdaptiveAttackEngine(
        neural_sca=neural_sca,
        learning_rate=1e-4,
        epsilon=0.9,
        memory_size=5000,
        device='cpu'
    )
    
    print("Training neural operator...")
    # Quick training
    model = neural_sca.train(traces, validation_split=0.2)
    
    print("Starting autonomous attack optimization...")
    # Perform autonomous attack
    results = adaptive_engine.autonomous_attack(
        traces=traces,
        target_success_rate=0.8,
        max_episodes=20,
        patience=5
    )
    
    print(f"\nOptimization Results:")
    print(f"  Final Success Rate: {results['success_rate']:.3f}")
    print(f"  Final Confidence: {results['confidence']:.3f}")
    print(f"  Training Episodes: {results['training_episodes']}")
    print(f"  Final Reward: {results['final_reward']:.2f}")
    
    print(f"\nOptimal Parameters:")
    for param, value in results['optimal_parameters'].items():
        print(f"  {param}: {value}")
    
    return results

def demonstrate_meta_learning_adaptation():
    """Demonstrate meta-learning for rapid adaptation to new targets."""
    print("\n" + "="*60)
    print("META-LEARNING ADAPTIVE ENGINE - DEMONSTRATION")
    print("="*60)
    
    # Create multiple targets with different characteristics
    targets = {
        'target_1': SimpleAESTarget(key=b'\x2b\x7e\x15\x16\x28\xae\xd2\xa6\xab\xf7\x15\x88\x09\xcf\x4f\x3c'),
        'target_2': SimpleAESTarget(key=b'\x00\x11\x22\x33\x44\x55\x66\x77\x88\x99\xaa\xbb\xcc\xdd\xee\xff'),
        'target_3': SimpleAESTarget(key=b'\xff\xee\xdd\xcc\xbb\xaa\x99\x88\x77\x66\x55\x44\x33\x22\x11\x00')
    }
    
    # Generate traces for each target
    simulator = LeakageSimulator(device_model='stm32f4', noise_model='realistic')
    target_traces = {}
    
    for name, target in targets.items():
        traces = simulator.simulate_traces(
            target=target,
            n_traces=1000,
            operations=['sbox'],
            trace_length=5000
        )
        target_traces[name] = traces
        print(f"Generated traces for {name}")
    
    # Initialize neural SCA and meta-learning engine
    neural_sca = NeuralSCA(architecture='fourier_neural_operator')
    meta_engine = MetaLearningAdaptiveEngine(
        neural_sca=neural_sca,
        learning_rate=1e-4,
        device='cpu'
    )
    
    # Train on first two targets
    print("\nTraining meta-learner on known targets...")
    for name in ['target_1', 'target_2']:
        traces = target_traces[name]
        model = neural_sca.train(traces, validation_split=0.2)
        
        # Short training episode to build meta-knowledge
        reward = meta_engine.train_episode(traces, max_steps=10)
        print(f"Meta-training on {name}: reward={reward:.2f}")
    
    # Rapid adaptation to new target
    print("\nPerforming rapid adaptation to unknown target_3...")
    new_target_traces = target_traces['target_3']
    
    adaptation_results = meta_engine.rapid_adaptation(
        traces=new_target_traces,
        adaptation_steps=5
    )
    
    print(f"\nRapid Adaptation Results:")
    print(f"  Adapted Success Rate: {adaptation_results['adapted_success_rate']:.3f}")
    print(f"  Adapted Confidence: {adaptation_results['adapted_confidence']:.3f}")
    print(f"  Adaptation Steps: {adaptation_results['adaptation_steps']}")
    
    print(f"\nAdapted Parameters:")
    for param, value in adaptation_results['optimal_parameters'].items():
        print(f"  {param}: {value}")
    
    return adaptation_results

def demonstrate_parameter_sensitivity():
    """Demonstrate sensitivity analysis of attack parameters."""
    print("\n" + "="*60)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*60)
    
    # Create target and traces
    target = SimpleAESTarget()
    simulator = LeakageSimulator(device_model='stm32f4', noise_model='realistic')
    traces = simulator.simulate_traces(
        target=target,
        n_traces=1000,
        operations=['sbox'],
        trace_length=5000
    )
    
    # Initialize neural SCA
    neural_sca = NeuralSCA(architecture='fourier_neural_operator')
    model = neural_sca.train(traces, validation_split=0.2)
    
    # Test different parameter configurations
    parameter_configs = [
        {'window_size': 500, 'n_pois': 25, 'preprocessing': 'standardize'},
        {'window_size': 1000, 'n_pois': 50, 'preprocessing': 'standardize'},
        {'window_size': 2000, 'n_pois': 100, 'preprocessing': 'normalize'},
        {'window_size': 1000, 'n_pois': 50, 'preprocessing': 'filtering'},
        {'window_size': 1500, 'n_pois': 75, 'preprocessing': 'standardize'},
    ]
    
    print("Testing parameter configurations:")
    results = []
    
    for i, config in enumerate(parameter_configs):
        # Update neural SCA configuration
        neural_sca.config['analysis'].update(config)
        
        # Perform attack
        attack_results = neural_sca.attack(traces, strategy='direct')
        
        success_rate = attack_results.get('success', 0.0)
        confidence = attack_results.get('avg_confidence', 0.0)
        
        results.append({
            'config': config,
            'success_rate': success_rate,
            'confidence': confidence
        })
        
        print(f"  Config {i+1}: Success={success_rate:.3f}, Confidence={confidence:.3f}")
        print(f"    {config}")
    
    # Find best configuration
    best_result = max(results, key=lambda x: x['success_rate'] + x['confidence'])
    print(f"\nBest Configuration:")
    print(f"  Success Rate: {best_result['success_rate']:.3f}")
    print(f"  Confidence: {best_result['confidence']:.3f}")
    print(f"  Parameters: {best_result['config']}")
    
    return results

def demonstrate_noise_robustness():
    """Demonstrate adaptive attack under different noise conditions."""
    print("\n" + "="*60)
    print("NOISE ROBUSTNESS DEMONSTRATION")
    print("="*60)
    
    target = SimpleAESTarget()
    noise_levels = ['low_noise', 'realistic', 'high_noise']
    
    # Initialize adaptive engine
    neural_sca = NeuralSCA(architecture='fourier_neural_operator')
    adaptive_engine = AdaptiveAttackEngine(
        neural_sca=neural_sca,
        learning_rate=1e-4,
        device='cpu'
    )
    
    results = []
    
    for noise_level in noise_levels:
        print(f"\nTesting with {noise_level} noise...")
        
        # Generate traces with specific noise level
        simulator = LeakageSimulator(device_model='stm32f4', noise_model=noise_level)
        traces = simulator.simulate_traces(
            target=target,
            n_traces=1500,
            operations=['sbox'],
            trace_length=5000
        )
        
        # Train model
        model = neural_sca.train(traces, validation_split=0.2)
        
        # Adaptive attack
        attack_results = adaptive_engine.autonomous_attack(
            traces=traces,
            target_success_rate=0.7,
            max_episodes=15,
            patience=5
        )
        
        results.append({
            'noise_level': noise_level,
            'results': attack_results
        })
        
        print(f"  {noise_level}: Success={attack_results['success_rate']:.3f}, "
              f"Episodes={attack_results['training_episodes']}")
    
    # Summary
    print(f"\nNoise Robustness Summary:")
    for result in results:
        noise_level = result['noise_level']
        success = result['results']['success_rate']
        episodes = result['results']['training_episodes']
        print(f"  {noise_level:12s}: Success={success:.3f}, Episodes={episodes:2d}")
    
    return results

def main():
    """Run all demonstrations."""
    print("NEURAL OPERATOR CRYPTANALYSIS - ADAPTIVE ATTACK DEMONSTRATIONS")
    print("=" * 70)
    
    try:
        # Basic adaptive attack
        basic_results = demonstrate_basic_adaptive_attack()
        
        # Meta-learning adaptation
        meta_results = demonstrate_meta_learning_adaptation()
        
        # Parameter sensitivity
        sensitivity_results = demonstrate_parameter_sensitivity()
        
        # Noise robustness
        robustness_results = demonstrate_noise_robustness()
        
        print("\n" + "="*70)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("="*70)
        
        print(f"\nSummary:")
        print(f"  Basic Adaptive Attack: {basic_results['success_rate']:.3f} success rate")
        print(f"  Meta-Learning Adaptation: {meta_results['adapted_success_rate']:.3f} success rate")
        print(f"  Best Manual Config: {max(sensitivity_results, key=lambda x: x['success_rate'])['success_rate']:.3f} success rate")
        print(f"  Noise Robustness: Tested across 3 noise levels")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())