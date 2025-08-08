#!/usr/bin/env python3
"""
Post-Quantum Kyber Implementation Analysis Example

This example demonstrates neural operator-based analysis of Kyber
post-quantum cryptographic implementations with focus on NTT operations.
"""

import numpy as np
import torch
import argparse
from pathlib import Path
import sys
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_cryptanalysis import NeuralSCA, LeakageSimulator
from neural_cryptanalysis.targets.post_quantum import KyberImplementation
from neural_cryptanalysis.neural_operators.custom import SideChannelFNO
from neural_cryptanalysis.utils.config import load_config, save_config


class KyberNTTAnalyzer:
    """Specialized analyzer for Kyber NTT operations."""
    
    def __init__(self, variant: str = 'kyber768', platform: str = 'arm_cortex_m4'):
        self.variant = variant
        self.platform = platform
        
        # Initialize Kyber target
        self.target = KyberImplementation(
            variant=variant,
            platform=platform,
            countermeasures=['masking', 'shuffling']
        )
        
        # Initialize leakage simulator
        self.simulator = LeakageSimulator(
            device_model='stm32f4',
            noise_model='realistic',
            snr_db=8.0
        )
        
        print(f"Initialized Kyber {variant} analyzer for {platform}")
    
    def generate_ntt_traces(self, n_traces: int) -> tuple:
        """Generate traces focusing on NTT operations."""
        
        print(f"Generating {n_traces} NTT-focused traces...")
        
        traces = []
        coefficients = []
        ntt_outputs = []
        
        for i in range(n_traces):
            # Generate random polynomial coefficients
            coeffs = np.random.randint(0, self.target.q, self.target.n, dtype=np.int16)
            
            # Simulate NTT computation with leakage
            trace, intermediate_values = self.simulator.simulate_kyber_ntt(
                coefficients=coeffs,
                target=self.target
            )
            
            traces.append(trace)
            coefficients.append(coeffs)
            ntt_outputs.append(intermediate_values['ntt_output'])
            
            if (i + 1) % 500 == 0:
                print(f"Generated {i + 1}/{n_traces} traces")
        
        return np.array(traces), np.array(coefficients), np.array(ntt_outputs)
    
    def train_ntt_neural_operator(self, traces: np.ndarray, 
                                 coefficients: np.ndarray) -> torch.nn.Module:
        """Train specialized neural operator for NTT analysis."""
        
        print("Training NTT-specialized neural operator...")
        
        # Configure specialized FNO for NTT patterns
        config = {
            'side_channel_fno': {
                'modes': 32,  # Higher modes for NTT frequency patterns
                'width': 128,
                'n_layers': 6,
                'activation': 'gelu',
                'dropout': 0.1
            },
            'training': {
                'batch_size': 64,
                'learning_rate': 5e-4,
                'epochs': 100,
                'weight_decay': 1e-5
            }
        }
        
        # Initialize neural SCA with custom architecture
        neural_sca = NeuralSCA(
            architecture='side_channel_fno',
            channels=['power', 'em_near_field'],
            config=config
        )
        
        # Prepare training data
        X = torch.tensor(traces, dtype=torch.float32)
        y = torch.tensor(coefficients[:, :16], dtype=torch.long)  # First 16 coefficients
        
        # Train model
        model = neural_sca.train(
            traces=X,
            labels=y,
            validation_split=0.15
        )
        
        return model, neural_sca
    
    def analyze_ntt_leakage(self, model, traces: np.ndarray) -> dict:
        """Analyze NTT operation leakage patterns."""
        
        print("Analyzing NTT leakage patterns...")
        
        # Convert to tensor
        X = torch.tensor(traces, dtype=torch.float32)
        
        with torch.no_grad():
            # Get model predictions
            predictions = model(X)
            
            # Analyze attention patterns (if available)
            if hasattr(model, 'get_attention_weights'):
                attention_weights = model.get_attention_weights(X)
            else:
                attention_weights = None
            
            # Identify critical time points
            prediction_variance = torch.var(predictions, dim=0)
            critical_points = torch.topk(prediction_variance, k=10).indices
        
        analysis_results = {
            'predictions': predictions.numpy(),
            'attention_weights': attention_weights.numpy() if attention_weights is not None else None,
            'critical_timepoints': critical_points.numpy(),
            'leakage_variance': prediction_variance.numpy()
        }
        
        return analysis_results
    
    def evaluate_countermeasures(self, model, n_eval_traces: int = 1000) -> dict:
        """Evaluate effectiveness of countermeasures."""
        
        print("Evaluating countermeasure effectiveness...")
        
        results = {}
        
        # Test different countermeasure configurations
        countermeasure_configs = [
            [],  # No countermeasures
            ['masking'],  # Boolean masking only
            ['shuffling'],  # Operation shuffling only
            ['masking', 'shuffling']  # Both countermeasures
        ]
        
        for countermeasures in countermeasure_configs:
            config_name = '_'.join(countermeasures) if countermeasures else 'none'
            
            # Reconfigure target with different countermeasures
            eval_target = KyberImplementation(
                variant=self.variant,
                platform=self.platform,
                countermeasures=countermeasures
            )
            
            # Generate evaluation traces
            eval_traces, eval_coeffs, _ = self.generate_ntt_traces(n_eval_traces)
            
            # Test attack success rate
            X_eval = torch.tensor(eval_traces, dtype=torch.float32)
            
            with torch.no_grad():
                predictions = model(X_eval)
                
                # Calculate success rate (simplified metric)
                true_coeffs = torch.tensor(eval_coeffs[:, :16], dtype=torch.long)
                predicted_coeffs = torch.argmax(predictions, dim=-1)
                
                accuracy = (predicted_coeffs == true_coeffs).float().mean()
                
                results[config_name] = {
                    'accuracy': accuracy.item(),
                    'countermeasures': countermeasures,
                    'n_traces': n_eval_traces
                }
            
            print(f"Countermeasures {config_name}: Accuracy = {accuracy:.4f}")
        
        return results


def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(description='Kyber Neural Operator Analysis')
    parser.add_argument('--variant', choices=['kyber512', 'kyber768', 'kyber1024'], 
                       default='kyber768', help='Kyber variant to analyze')
    parser.add_argument('--traces', type=int, default=10000, 
                       help='Number of traces to generate')
    parser.add_argument('--platform', choices=['arm_cortex_m4', 'riscv', 'x86_64'], 
                       default='arm_cortex_m4', help='Target platform')
    parser.add_argument('--output', type=str, default='kyber_analysis_results.json', 
                       help='Output file for results')
    
    args = parser.parse_args()
    
    print("🔐 Neural Operator Cryptanalysis - Kyber Analysis Example")
    print("=" * 60)
    print(f"Variant: {args.variant}")
    print(f"Platform: {args.platform}")
    print(f"Traces: {args.traces}")
    
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Initialize analyzer
    analyzer = KyberNTTAnalyzer(variant=args.variant, platform=args.platform)
    
    # Generate training data
    start_time = time.time()
    traces, coefficients, ntt_outputs = analyzer.generate_ntt_traces(args.traces)
    data_gen_time = time.time() - start_time
    
    print(f"Data generation completed in {data_gen_time:.2f} seconds")
    
    # Split data for training and testing
    split_idx = int(0.8 * len(traces))
    train_traces = traces[:split_idx]
    train_coeffs = coefficients[:split_idx]
    test_traces = traces[split_idx:]
    test_coeffs = coefficients[split_idx:]
    
    # Train neural operator
    start_time = time.time()
    model, neural_sca = analyzer.train_ntt_neural_operator(train_traces, train_coeffs)
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Analyze leakage patterns
    analysis_results = analyzer.analyze_ntt_leakage(model, test_traces[:100])
    
    # Evaluate countermeasures
    countermeasure_results = analyzer.evaluate_countermeasures(model, n_eval_traces=1000)
    
    # Compile final results
    final_results = {
        'configuration': {
            'variant': args.variant,
            'platform': args.platform,
            'n_traces': args.traces,
            'data_generation_time': data_gen_time,
            'training_time': training_time
        },
        'leakage_analysis': {
            'critical_timepoints': analysis_results['critical_timepoints'].tolist(),
            'max_leakage_variance': float(np.max(analysis_results['leakage_variance']))
        },
        'countermeasure_evaluation': countermeasure_results
    }
    
    # Save results
    save_config(final_results, args.output)
    
    # Display summary
    print("\n🎯 Analysis Results Summary:")
    print(f"Critical timepoints identified: {len(analysis_results['critical_timepoints'])}")
    print(f"Maximum leakage variance: {final_results['leakage_analysis']['max_leakage_variance']:.6f}")
    print("\nCountermeasure effectiveness:")
    for config, result in countermeasure_results.items():
        print(f"  {config}: {result['accuracy']:.4f} accuracy")
    
    print(f"\n✅ Results saved to: {args.output}")
    print("Analysis completed successfully!")
    
    return final_results


if __name__ == "__main__":
    main()