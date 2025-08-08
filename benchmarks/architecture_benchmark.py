#!/usr/bin/env python3
"""
Neural Operator Architecture Benchmark Suite

Comprehensive comparison of FNO, DeepONet, and custom neural operator
architectures for side-channel cryptanalysis tasks.
"""

import numpy as np
import torch
import time
import json
import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import psutil
import gc

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_cryptanalysis import NeuralSCA, LeakageSimulator
from neural_cryptanalysis.targets import AESImplementation
from neural_cryptanalysis.neural_operators import (
    FourierNeuralOperator, DeepOperatorNetwork, SideChannelFNO
)
from neural_cryptanalysis.utils.performance import PerformanceProfiler
from neural_cryptanalysis.utils.validation import StatisticalValidator


class ArchitectureBenchmark:
    """Comprehensive architecture benchmarking framework."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.profiler = PerformanceProfiler()
        self.validator = StatisticalValidator()
        
        # Initialize target and simulator
        self.target = AESImplementation(
            version='aes128',
            platform='software',
            countermeasures=[]
        )
        
        self.simulator = LeakageSimulator(
            device_model='generic_mcu',
            noise_model='gaussian',
            snr_db=config.get('snr_db', 10.0)
        )
        
        print("Initialized architecture benchmark framework")
    
    def generate_benchmark_dataset(self, n_traces: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate standardized benchmark dataset."""
        
        print(f"Generating benchmark dataset with {n_traces} traces...")
        
        # Fixed key for reproducibility
        key = np.array([0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
                       0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c], dtype=np.uint8)
        
        # Generate random plaintexts
        plaintexts = np.random.randint(0, 256, (n_traces, 16), dtype=np.uint8)
        
        traces = []
        labels = []
        
        for i, plaintext in enumerate(plaintexts):
            trace, intermediate_values = self.simulator.simulate_aes_encryption(
                plaintext=plaintext,
                key=key,
                target=self.target
            )
            
            traces.append(trace)
            labels.append(intermediate_values['sbox_output'][0])  # First S-box output
            
            if (i + 1) % 1000 == 0:
                print(f"Generated {i + 1}/{n_traces} traces")
        
        return np.array(traces), np.array(labels)
    
    def benchmark_architecture(self, architecture: str, traces: np.ndarray, 
                             labels: np.ndarray) -> Dict:
        """Benchmark a specific neural operator architecture."""
        
        print(f"\nBenchmarking {architecture} architecture...")
        
        # Architecture-specific configurations
        arch_configs = {
            'fourier_neural_operator': {
                'fno': {
                    'modes': 16,
                    'width': 64,
                    'n_layers': 4
                }
            },
            'deep_operator_network': {
                'deeponet': {
                    'branch_net': [128, 128, 128],
                    'trunk_net': [64, 64, 64],
                    'activation': 'relu'
                }
            },
            'side_channel_fno': {
                'side_channel_fno': {
                    'modes': 24,
                    'width': 96,
                    'n_layers': 5,
                    'activation': 'gelu'
                }
            }
        }
        
        config = {
            **arch_configs.get(architecture, {}),
            'training': {
                'batch_size': 128,
                'learning_rate': 1e-3,
                'epochs': self.config.get('epochs', 50)
            }
        }
        
        # Initialize neural SCA
        neural_sca = NeuralSCA(
            architecture=architecture,
            channels=['power'],
            config=config
        )
        
        # Prepare data
        X = torch.tensor(traces, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(labels, dtype=torch.long)
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Benchmark training
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        with self.profiler.profile(f"{architecture}_training"):
            model = neural_sca.train(
                traces=X_train,
                labels=y_train,
                validation_split=0.2
            )
        training_time = time.time() - start_time
        
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_usage = memory_after - memory_before
        
        # Benchmark inference
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            with self.profiler.profile(f"{architecture}_inference"):
                predictions = model(X_test)
            inference_time = time.time() - start_time
        
        # Calculate accuracy
        predicted_labels = torch.argmax(predictions, dim=1)
        accuracy = (predicted_labels == y_test).float().mean().item()
        
        # Calculate additional metrics
        confidence_scores = torch.softmax(predictions, dim=1).max(dim=1)[0]
        mean_confidence = confidence_scores.mean().item()
        
        # Model complexity metrics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        results = {
            'architecture': architecture,
            'performance': {
                'accuracy': accuracy,
                'mean_confidence': mean_confidence,
                'training_time': training_time,
                'inference_time': inference_time,
                'inference_time_per_trace': inference_time / len(X_test),
                'memory_usage_mb': memory_usage
            },
            'model_complexity': {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': total_params * 4 / 1024 / 1024  # Assuming float32
            },
            'training_config': config,
            'dataset_info': {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'trace_length': traces.shape[1]
            }
        }
        
        print(f"{architecture} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Inference time per trace: {inference_time / len(X_test):.6f}s")
        print(f"  Memory usage: {memory_usage:.2f} MB")
        print(f"  Parameters: {total_params:,}")
        
        return results
    
    def run_statistical_analysis(self, results: List[Dict]) -> Dict:
        """Run statistical analysis on benchmark results."""
        
        print("\nRunning statistical analysis...")
        
        # Extract metrics for comparison
        architectures = [r['architecture'] for r in results]
        accuracies = [r['performance']['accuracy'] for r in results]
        training_times = [r['performance']['training_time'] for r in results]
        inference_times = [r['performance']['inference_time_per_trace'] for r in results]
        
        # Statistical tests
        statistical_results = {
            'accuracy_comparison': self.validator.compare_groups(
                {arch: [acc] for arch, acc in zip(architectures, accuracies)},
                metric_name='accuracy'
            ),
            'training_time_comparison': {
                'fastest': architectures[np.argmin(training_times)],
                'slowest': architectures[np.argmax(training_times)],
                'speedup_ratio': max(training_times) / min(training_times)
            },
            'inference_speed_comparison': {
                'fastest': architectures[np.argmin(inference_times)],
                'slowest': architectures[np.argmax(inference_times)],
                'speedup_ratio': max(inference_times) / min(inference_times)
            }
        }
        
        return statistical_results
    
    def generate_benchmark_report(self, results: List[Dict], 
                                statistical_analysis: Dict) -> Dict:
        """Generate comprehensive benchmark report."""
        
        # Find best performing architecture for each metric
        best_accuracy = max(results, key=lambda x: x['performance']['accuracy'])
        fastest_training = min(results, key=lambda x: x['performance']['training_time'])
        fastest_inference = min(results, key=lambda x: x['performance']['inference_time_per_trace'])
        smallest_model = min(results, key=lambda x: x['model_complexity']['total_parameters'])
        
        report = {
            'benchmark_summary': {
                'total_architectures_tested': len(results),
                'dataset_size': results[0]['dataset_info']['train_size'] + results[0]['dataset_info']['test_size'],
                'trace_length': results[0]['dataset_info']['trace_length']
            },
            'best_performers': {
                'highest_accuracy': {
                    'architecture': best_accuracy['architecture'],
                    'accuracy': best_accuracy['performance']['accuracy']
                },
                'fastest_training': {
                    'architecture': fastest_training['architecture'],
                    'time': fastest_training['performance']['training_time']
                },
                'fastest_inference': {
                    'architecture': fastest_inference['architecture'],
                    'time_per_trace': fastest_inference['performance']['inference_time_per_trace']
                },
                'smallest_model': {
                    'architecture': smallest_model['architecture'],
                    'parameters': smallest_model['model_complexity']['total_parameters']
                }
            },
            'detailed_results': results,
            'statistical_analysis': statistical_analysis,
            'recommendations': self._generate_recommendations(results)
        }
        
        return report
    
    def _generate_recommendations(self, results: List[Dict]) -> Dict:
        """Generate architecture recommendations based on results."""
        
        recommendations = {}
        
        # Accuracy-focused recommendation
        best_accuracy = max(results, key=lambda x: x['performance']['accuracy'])
        recommendations['for_accuracy'] = {
            'architecture': best_accuracy['architecture'],
            'reason': f"Highest accuracy: {best_accuracy['performance']['accuracy']:.4f}"
        }
        
        # Speed-focused recommendation
        fastest_inference = min(results, key=lambda x: x['performance']['inference_time_per_trace'])
        recommendations['for_speed'] = {
            'architecture': fastest_inference['architecture'],
            'reason': f"Fastest inference: {fastest_inference['performance']['inference_time_per_trace']:.6f}s per trace"
        }
        
        # Balanced recommendation (accuracy * speed score)
        for result in results:
            result['balance_score'] = (
                result['performance']['accuracy'] / 
                result['performance']['inference_time_per_trace']
            )
        
        best_balanced = max(results, key=lambda x: x['balance_score'])
        recommendations['for_balance'] = {
            'architecture': best_balanced['architecture'],
            'reason': f"Best accuracy/speed balance: {best_balanced['balance_score']:.2f}"
        }
        
        return recommendations


def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(description='Neural Operator Architecture Benchmark')
    parser.add_argument('--traces', type=int, default=5000,
                       help='Number of traces for benchmark')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Training epochs per architecture')
    parser.add_argument('--snr-db', type=float, default=10.0,
                       help='Signal-to-noise ratio in dB')
    parser.add_argument('--output', type=str, default='architecture_benchmark_results.json',
                       help='Output file for results')
    parser.add_argument('--architectures', type=str, 
                       default='fourier_neural_operator,deep_operator_network,side_channel_fno',
                       help='Comma-separated list of architectures to benchmark')
    
    args = parser.parse_args()
    
    print("🏁 Neural Operator Architecture Benchmark Suite")
    print("=" * 60)
    
    # Parse architectures
    architectures = [arch.strip() for arch in args.architectures.split(',')]
    
    print(f"Architectures to benchmark: {architectures}")
    print(f"Dataset size: {args.traces} traces")
    print(f"Training epochs: {args.epochs}")
    print(f"SNR: {args.snr_db} dB")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Initialize benchmark
    config = {
        'epochs': args.epochs,
        'snr_db': args.snr_db
    }
    
    benchmark = ArchitectureBenchmark(config)
    
    # Generate benchmark dataset
    traces, labels = benchmark.generate_benchmark_dataset(args.traces)
    
    # Run benchmarks for each architecture
    results = []
    for architecture in architectures:
        try:
            result = benchmark.benchmark_architecture(architecture, traces, labels)
            results.append(result)
        except Exception as e:
            print(f"Error benchmarking {architecture}: {e}")
            continue
    
    if not results:
        print("No successful benchmarks completed!")
        return
    
    # Run statistical analysis
    statistical_analysis = benchmark.run_statistical_analysis(results)
    
    # Generate comprehensive report
    report = benchmark.generate_benchmark_report(results, statistical_analysis)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Display summary
    print("\n🎯 Benchmark Results Summary:")
    print(f"Best accuracy: {report['best_performers']['highest_accuracy']['architecture']} "
          f"({report['best_performers']['highest_accuracy']['accuracy']:.4f})")
    print(f"Fastest training: {report['best_performers']['fastest_training']['architecture']} "
          f"({report['best_performers']['fastest_training']['time']:.2f}s)")
    print(f"Fastest inference: {report['best_performers']['fastest_inference']['architecture']} "
          f"({report['best_performers']['fastest_inference']['time_per_trace']:.6f}s/trace)")
    
    print("\n📋 Recommendations:")
    for use_case, rec in report['recommendations'].items():
        print(f"  {use_case}: {rec['architecture']} - {rec['reason']}")
    
    print(f"\n✅ Complete results saved to: {args.output}")
    
    return report


if __name__ == "__main__":
    main()