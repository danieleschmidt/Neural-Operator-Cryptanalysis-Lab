"""Command Line Interface for Neural Operator Cryptanalysis Lab."""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

from .core import NeuralSCA, LeakageSimulator
from .targets import KyberImplementation, DilithiumImplementation
from .utils.config import ConfigManager, get_default_config, validate_config
from .utils.logging_utils import setup_logging, get_logger
from .security import ResponsibleDisclosure, OperationType, SecurityPolicy
from .optimization import PerformanceProfiler, BatchProcessor


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Neural Operator Cryptanalysis Lab - Defensive Security Research Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic attack simulation
  neural-sca simulate --algorithm kyber --traces 10000
  
  # Train neural operator model
  neural-sca train --config config.yaml --data traces.npy
  
  # Evaluate countermeasures
  neural-sca evaluate --target kyber768 --countermeasure masking
  
  # Run comprehensive benchmark
  neural-sca benchmark --architectures fno,deeponet --targets kyber,dilithium

For more information, visit: https://github.com/danieleschmidt/Neural-Operator-Cryptanalysis-Lab
        """
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Neural Operator Cryptanalysis Lab 0.1.0'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (YAML or JSON)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Output directory for results'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Simulate command
    simulate_parser = subparsers.add_parser(
        'simulate',
        help='Simulate side-channel attack on cryptographic implementation'
    )
    simulate_parser.add_argument('--algorithm', required=True, choices=['kyber', 'dilithium', 'aes'],
                               help='Target cryptographic algorithm')
    simulate_parser.add_argument('--traces', type=int, default=10000,
                               help='Number of traces to simulate')
    simulate_parser.add_argument('--architecture', default='fourier_neural_operator',
                               help='Neural operator architecture')
    simulate_parser.add_argument('--noise-level', type=float, default=0.01,
                               help='Noise level for simulation')
    
    # Train command
    train_parser = subparsers.add_parser(
        'train',
        help='Train neural operator model on trace data'
    )
    train_parser.add_argument('--data', required=True,
                            help='Path to training trace data')
    train_parser.add_argument('--labels',
                            help='Path to training labels')
    train_parser.add_argument('--epochs', type=int, default=100,
                            help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=64,
                            help='Training batch size')
    train_parser.add_argument('--lr', type=float, default=1e-3,
                            help='Learning rate')
    
    # Attack command
    attack_parser = subparsers.add_parser(
        'attack',
        help='Perform side-channel attack using trained model'
    )
    attack_parser.add_argument('--model', required=True,
                             help='Path to trained model')
    attack_parser.add_argument('--traces', required=True,
                             help='Path to target traces')
    attack_parser.add_argument('--strategy', default='direct',
                             choices=['direct', 'template', 'adaptive'],
                             help='Attack strategy')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate countermeasure effectiveness'
    )
    evaluate_parser.add_argument('--target', required=True,
                               help='Target implementation')
    evaluate_parser.add_argument('--countermeasure', required=True,
                               choices=['masking', 'shuffling', 'hiding'],
                               help='Countermeasure to evaluate')
    evaluate_parser.add_argument('--order', type=int, default=1,
                               help='Countermeasure order/strength')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser(
        'benchmark',
        help='Run performance benchmarks'
    )
    benchmark_parser.add_argument('--architectures',
                                help='Comma-separated list of architectures to benchmark')
    benchmark_parser.add_argument('--targets',
                                help='Comma-separated list of targets to benchmark')
    benchmark_parser.add_argument('--trace-counts', default='1000,10000,100000',
                                help='Comma-separated list of trace counts')
    
    # Generate command
    generate_parser = subparsers.add_parser(
        'generate',
        help='Generate synthetic trace data'
    )
    generate_parser.add_argument('--algorithm', required=True,
                               help='Target algorithm for trace generation')
    generate_parser.add_argument('--count', type=int, default=10000,
                               help='Number of traces to generate')
    generate_parser.add_argument('--output', required=True,
                               help='Output file path')
    generate_parser.add_argument('--format', choices=['numpy', 'hdf5'], default='numpy',
                               help='Output format')
    
    return parser


def setup_environment(args: argparse.Namespace) -> Dict[str, Any]:
    """Setup environment and load configuration."""
    # Load configuration
    if args.config:
        config_manager = ConfigManager(args.config)
        config = config_manager.to_dict()
    else:
        config = get_default_config()
    
    # Override with command line arguments
    if hasattr(args, 'log_level'):
        config['logging']['level'] = args.log_level
    
    config['data']['output_dir'] = args.output_dir
    
    # Validate configuration
    issues = validate_config(config)
    if issues:
        print("Configuration validation errors:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    
    # Setup logging
    setup_logging(
        level=config['logging']['level'],
        log_file=Path(args.output_dir) / "neural_cryptanalysis.log",
        experiment_name=config['experiment']['name'],
        audit_logging=config.get('security', {}).get('audit_logging', True)
    )
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    return config


def command_simulate(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    """Execute simulate command."""
    logger = get_logger(__name__)
    logger.info(f"Starting simulation of {args.algorithm} with {args.traces} traces")
    
    try:
        # Setup security policy
        policy = SecurityPolicy(
            max_traces_per_experiment=1000000,
            require_written_authorization=False  # Relaxed for simulation
        )
        disclosure = ResponsibleDisclosure(policy)
        
        # Request authorization
        auth_token = disclosure.ensure_authorized(
            OperationType.TRACE_COLLECTION,
            target=f"{args.algorithm}_simulation",
            justification="Neural operator research simulation"
        )
        
        # Validate operation
        if not disclosure.validate_operation(auth_token, OperationType.TRACE_COLLECTION, n_traces=args.traces):
            logger.error("Operation not authorized or exceeds limits")
            return 1
        
        # Create target implementation
        if args.algorithm == 'kyber':
            from .targets.base import ImplementationConfig
            target_config = ImplementationConfig(
                algorithm='kyber',
                variant='kyber768',
                platform='arm_cortex_m4'
            )
            target = KyberImplementation(target_config)
        elif args.algorithm == 'dilithium':
            target_config = ImplementationConfig(
                algorithm='dilithium',
                variant='dilithium3'
            )
            target = DilithiumImplementation(target_config)
        else:
            logger.error(f"Unsupported algorithm: {args.algorithm}")
            return 1
        
        # Generate key
        public_key, secret_key = target.generate_key()
        logger.info("Generated cryptographic keys")
        
        # Setup simulator
        simulator = LeakageSimulator(
            device_model='stm32f4',
            noise_model='realistic'
        )
        
        # Generate traces
        trace_data = simulator.simulate_traces(
            target=target,
            n_traces=args.traces,
            operations=['sbox', 'ntt'],
            trace_length=config['side_channel']['trace_length']
        )
        
        logger.info(f"Generated {len(trace_data)} traces")
        
        # Setup neural operator
        neural_sca = NeuralSCA(
            architecture=args.architecture,
            channels=['power'],
            config=config
        )
        
        # Train model
        profiler = PerformanceProfiler()
        train_fn = profiler.profile_function(neural_sca.train)
        
        model = train_fn(trace_data, validation_split=0.2)
        
        # Perform attack
        attack_results = neural_sca.attack(trace_data, model)
        
        # Report results
        logger.info(f"Attack success rate: {attack_results['success']:.2%}")
        logger.info(f"Average confidence: {attack_results['avg_confidence']:.3f}")
        
        # Save results
        results = {
            'algorithm': args.algorithm,
            'traces': args.traces,
            'architecture': args.architecture,
            'attack_results': attack_results,
            'performance_metrics': profiler.get_average_metrics().__dict__
        }
        
        output_path = Path(args.output_dir) / f"simulation_{args.algorithm}_{args.traces}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return 1


def command_train(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    """Execute train command."""
    logger = get_logger(__name__)
    logger.info(f"Starting training with data from {args.data}")
    
    try:
        import numpy as np
        
        # Load training data
        traces = np.load(args.data)
        
        if args.labels:
            labels = np.load(args.labels)
        else:
            # Generate synthetic labels
            labels = np.random.randint(0, 256, len(traces))
        
        logger.info(f"Loaded {len(traces)} traces for training")
        
        # Setup neural operator
        config['training']['epochs'] = args.epochs
        config['training']['batch_size'] = args.batch_size
        config['training']['learning_rate'] = args.lr
        
        neural_sca = NeuralSCA(config=config)
        
        # Setup batch processor for optimization
        batch_processor = BatchProcessor(
            batch_size=args.batch_size,
            use_gpu=True
        )
        
        # Train model with profiling
        profiler = PerformanceProfiler()
        train_fn = profiler.profile_function(neural_sca.train)
        
        from .side_channels.base import TraceData
        trace_data = TraceData(traces=traces, labels=labels)
        
        model = train_fn(trace_data)
        
        # Save trained model
        model_path = Path(args.output_dir) / "trained_model.pth"
        neural_sca.neural_operator.save_checkpoint(str(model_path))
        
        logger.info(f"Model saved to {model_path}")
        
        # Save training metrics
        metrics = {
            'training_history': neural_sca.training_history,
            'performance_metrics': profiler.get_average_metrics().__dict__
        }
        
        metrics_path = Path(args.output_dir) / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


def command_benchmark(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    """Execute benchmark command."""
    logger = get_logger(__name__)
    logger.info("Starting performance benchmarks")
    
    try:
        from .optimization import ScalabilityTester
        
        architectures = ['fourier_neural_operator', 'deep_operator_network']
        if args.architectures:
            architectures = args.architectures.split(',')
        
        trace_counts = [1000, 10000, 100000]
        if args.trace_counts:
            trace_counts = [int(x.strip()) for x in args.trace_counts.split(',')]
        
        tester = ScalabilityTester()
        benchmark_results = {}
        
        for arch in architectures:
            logger.info(f"Benchmarking {arch}")
            
            # Create neural operator
            neural_sca = NeuralSCA(architecture=arch, config=config)
            
            # Test scalability
            input_sizes = [1000, 5000, 10000]
            batch_sizes = [16, 32, 64, 128]
            
            scalability_results = tester.test_model_scalability(
                neural_sca.neural_operator,
                input_sizes,
                batch_sizes
            )
            
            benchmark_results[arch] = scalability_results
        
        # Save benchmark results
        results_path = Path(args.output_dir) / "benchmark_results.json"
        with open(results_path, 'w') as f:
            json.dump(benchmark_results, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to {results_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup environment
    try:
        config = setup_environment(args)
    except Exception as e:
        print(f"Environment setup failed: {e}")
        return 1
    
    # Execute command
    if args.command == 'simulate':
        return command_simulate(args, config)
    elif args.command == 'train':
        return command_train(args, config)
    elif args.command == 'attack':
        print("Attack command not yet implemented")
        return 1
    elif args.command == 'evaluate':
        print("Evaluate command not yet implemented")
        return 1
    elif args.command == 'benchmark':
        return command_benchmark(args, config)
    elif args.command == 'generate':
        print("Generate command not yet implemented")
        return 1
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())