#!/usr/bin/env python3
"""
Comprehensive testing framework for Neural Operator Cryptanalysis Lab.

This module provides extensive test coverage including unit tests, integration tests,
performance benchmarks, and security validation tests.
"""

import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys
import time
import warnings
from unittest.mock import Mock

# Mock torch if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create a mock torch module
    torch = Mock()
    torch.tensor = Mock(return_value=Mock())
    torch.nn = Mock()
    torch.optim = Mock()
    torch.cuda = Mock()
    torch.cuda.is_available = Mock(return_value=False)
    torch.manual_seed = Mock()
    torch.randn = Mock(return_value=Mock())
    torch.randint = Mock(return_value=Mock())
    torch.zeros = Mock(return_value=Mock())
    torch.ones = Mock(return_value=Mock())
    torch.no_grad = Mock(return_value=Mock())
    torch.float32 = "float32"
    torch.long = "long"
    # Add method mocks for tensor operations
    mock_tensor = Mock()
    mock_tensor.unsqueeze = Mock(return_value=Mock())
    mock_tensor.requires_grad = False
    torch.tensor = Mock(return_value=mock_tensor)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_cryptanalysis import NeuralSCA, LeakageSimulator
from neural_cryptanalysis.neural_operators import FourierNeuralOperator, DeepOperatorNetwork
from neural_cryptanalysis.targets.post_quantum import KyberImplementation
from neural_cryptanalysis.datasets.synthetic import SyntheticDatasetGenerator
from neural_cryptanalysis.utils.validation import StatisticalValidator
from neural_cryptanalysis.utils.security import SecurityPolicy
from neural_cryptanalysis.utils.performance import PerformanceProfiler


class TestNeuralOperatorArchitectures(unittest.TestCase):
    """Test neural operator architectures."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        self.batch_size = 32
        self.trace_length = 1000
        self.n_channels = 1
        
        # Create test data
        self.test_traces = torch.randn(self.batch_size, self.trace_length, self.n_channels)
        self.test_labels = torch.randint(0, 256, (self.batch_size,))
    
    def test_fourier_neural_operator_initialization(self):
        """Test FNO initialization and basic forward pass."""
        fno = FourierNeuralOperator(
            modes=16,
            width=64,
            n_layers=4,
            in_channels=self.n_channels,
            out_channels=256
        )
        
        # Test forward pass
        output = fno(self.test_traces)
        
        self.assertEqual(output.shape, (self.batch_size, 256))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_deep_operator_network_initialization(self):
        """Test DeepONet initialization and forward pass."""
        deeponet = DeepOperatorNetwork(
            branch_net=[128, 128, 128],
            trunk_net=[64, 64, 64],
            output_dim=256
        )
        
        # Test forward pass
        output = deeponet(self.test_traces)
        
        self.assertEqual(output.shape, (self.batch_size, 256))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_neural_operator_gradients(self):
        """Test gradient computation for neural operators."""
        fno = FourierNeuralOperator(
            modes=8,
            width=32,
            n_layers=2,
            in_channels=self.n_channels,
            out_channels=256
        )
        
        # Forward pass
        output = fno(self.test_traces)
        loss = torch.nn.functional.cross_entropy(output, self.test_labels)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist and are non-zero
        gradient_norm = 0
        for param in fno.parameters():
            if param.grad is not None:
                gradient_norm += param.grad.norm().item()
        
        self.assertGreater(gradient_norm, 0, "No gradients computed")
    
    def test_neural_operator_reproducibility(self):
        """Test deterministic behavior with fixed seeds."""
        torch.manual_seed(123)
        fno1 = FourierNeuralOperator(modes=8, width=32, n_layers=2)
        output1 = fno1(self.test_traces)
        
        torch.manual_seed(123)
        fno2 = FourierNeuralOperator(modes=8, width=32, n_layers=2)
        output2 = fno2(self.test_traces)
        
        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))


class TestSideChannelAnalysis(unittest.TestCase):
    """Test side-channel analysis components."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        torch.manual_seed(42)
        
        self.neural_sca = NeuralSCA(
            architecture='fourier_neural_operator',
            channels=['power'],
            config={
                'fno': {'modes': 8, 'width': 32, 'n_layers': 2},
                'training': {'batch_size': 32, 'epochs': 3, 'learning_rate': 1e-3}
            }
        )
        
        # Generate synthetic test data
        self.n_traces = 500
        self.traces = torch.randn(self.n_traces, 1000, 1)
        self.labels = torch.randint(0, 256, (self.n_traces,))
    
    def test_neural_sca_training(self):
        """Test neural SCA training process."""
        # Split data
        split_idx = int(0.8 * len(self.traces))
        train_traces = self.traces[:split_idx]
        train_labels = self.labels[:split_idx]
        
        # Train model
        model = self.neural_sca.train(
            traces=train_traces,
            labels=train_labels,
            validation_split=0.2
        )
        
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'forward'))
        
        # Test inference
        with torch.no_grad():
            predictions = model(self.traces[:10])
            self.assertEqual(predictions.shape, (10, 256))
    
    def test_attack_simulation(self):
        """Test attack simulation and evaluation."""
        # Quick training for testing
        model = self.neural_sca.train(
            traces=self.traces[:200],
            labels=self.labels[:200],
            validation_split=0.2
        )
        
        # Simulate attack
        test_traces = self.traces[200:250]
        test_plaintexts = np.random.randint(0, 256, (50, 16), dtype=np.uint8)
        
        attack_results = self.neural_sca.attack(
            target_traces=test_traces,
            model=model,
            strategy='template',
            target_byte=0,
            plaintexts=test_plaintexts
        )
        
        self.assertIn('predicted_key_byte', attack_results)
        self.assertIn('confidence', attack_results)
        self.assertIsInstance(attack_results['confidence'], (int, float))
    
    def test_noise_robustness(self):
        """Test robustness to different noise levels."""
        noise_levels = [0.1, 0.5, 1.0]
        accuracies = []
        
        for noise_level in noise_levels:
            # Add noise to traces
            noisy_traces = self.traces + torch.randn_like(self.traces) * noise_level
            
            # Quick training
            model = self.neural_sca.train(
                traces=noisy_traces[:200],
                labels=self.labels[:200],
                validation_split=0.2
            )
            
            # Test accuracy
            with torch.no_grad():
                predictions = model(noisy_traces[200:250])
                predicted_labels = torch.argmax(predictions, dim=1)
                accuracy = (predicted_labels == self.labels[200:250]).float().mean().item()
                accuracies.append(accuracy)
        
        # Accuracy should decrease with increased noise
        self.assertGreaterEqual(accuracies[0], accuracies[1])
        self.assertGreaterEqual(accuracies[1], accuracies[2])


class TestPostQuantumCryptanalysis(unittest.TestCase):
    """Test post-quantum cryptography analysis components."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
        self.kyber_target = KyberImplementation(
            variant='kyber768',
            platform='arm_cortex_m4',
            countermeasures=[]
        )
    
    def test_kyber_implementation_initialization(self):
        """Test Kyber implementation initialization."""
        self.assertEqual(self.kyber_target.variant, 'kyber768')
        self.assertEqual(self.kyber_target.platform, 'arm_cortex_m4')
        self.assertIsInstance(self.kyber_target.q, int)
        self.assertIsInstance(self.kyber_target.n, int)
    
    def test_kyber_ntt_simulation(self):
        """Test Kyber NTT operation simulation."""
        simulator = LeakageSimulator(
            device_model='stm32f4',
            noise_model='gaussian',
            snr_db=15.0
        )
        
        # Generate random polynomial coefficients
        coefficients = np.random.randint(0, self.kyber_target.q, self.kyber_target.n, dtype=np.int16)
        
        # Simulate NTT operation
        trace, intermediate_values = simulator.simulate_kyber_ntt(
            coefficients=coefficients,
            target=self.kyber_target
        )
        
        self.assertIsInstance(trace, np.ndarray)
        self.assertIn('ntt_output', intermediate_values)
        self.assertEqual(len(intermediate_values['ntt_output']), self.kyber_target.n)
    
    def test_countermeasure_evaluation(self):
        """Test countermeasure effectiveness evaluation."""
        # Test different countermeasure configurations
        configs = [
            [],  # No countermeasures
            ['masking'],  # Boolean masking
            ['shuffling'],  # Operation shuffling
            ['masking', 'shuffling']  # Both
        ]
        
        for countermeasures in configs:
            target = KyberImplementation(
                variant='kyber512',  # Smaller variant for faster testing
                platform='arm_cortex_m4',
                countermeasures=countermeasures
            )
            
            self.assertEqual(target.countermeasures, countermeasures)
            
            # Simulate with countermeasures
            simulator = LeakageSimulator(device_model='stm32f4', noise_model='gaussian')
            coeffs = np.random.randint(0, target.q, target.n, dtype=np.int16)
            
            trace, intermediate_values = simulator.simulate_kyber_ntt(
                coefficients=coeffs,
                target=target
            )
            
            self.assertIsInstance(trace, np.ndarray)
            self.assertIn('ntt_output', intermediate_values)


class TestDatasetGeneration(unittest.TestCase):
    """Test synthetic dataset generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
        self.generator = SyntheticDatasetGenerator(random_seed=42)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_aes_dataset_generation(self):
        """Test AES dataset generation."""
        n_traces = 100
        dataset = self.generator.generate_aes_dataset(
            n_traces=n_traces,
            target_bytes=[0],
            trace_length=500
        )
        
        self.assertIn('power_traces', dataset)
        self.assertIn('labels', dataset)
        self.assertIn('plaintexts', dataset)
        self.assertIn('key', dataset)
        self.assertIn('metadata', dataset)
        
        self.assertEqual(dataset['power_traces'].shape, (n_traces, 500))
        self.assertEqual(len(dataset['labels'][0]), n_traces)
        self.assertEqual(dataset['plaintexts'].shape, (n_traces, 16))
        self.assertEqual(len(dataset['key']), 16)
    
    def test_noise_model_configuration(self):
        """Test different noise model configurations."""
        from neural_cryptanalysis.datasets.synthetic import NoiseModel
        
        noise_configs = [
            NoiseModel(noise_type='gaussian', snr_db=10),
            NoiseModel(noise_type='uniform', snr_db=5),
            NoiseModel(noise_type='laplace', snr_db=15)
        ]
        
        for noise_config in noise_configs:
            generator = SyntheticDatasetGenerator(noise_model=noise_config, random_seed=42)
            
            # Generate small dataset
            dataset = generator.generate_aes_dataset(n_traces=50, trace_length=100)
            
            self.assertEqual(dataset['power_traces'].shape, (50, 100))
            self.assertEqual(dataset['metadata']['noise_model']['type'], noise_config.noise_type)
            self.assertEqual(dataset['metadata']['noise_model']['snr_db'], noise_config.snr_db)
    
    def test_dataset_save_load(self):
        """Test dataset saving and loading."""
        dataset = self.generator.generate_aes_dataset(n_traces=50, trace_length=100)
        
        # Test .npz format
        npz_path = Path(self.temp_dir) / "test_dataset.npz"
        self.generator.save_dataset(dataset, npz_path)
        
        loaded_dataset = self.generator.load_dataset(npz_path)
        
        np.testing.assert_array_equal(dataset['power_traces'], loaded_dataset['power_traces'])
        np.testing.assert_array_equal(dataset['key'], loaded_dataset['key'])
        
        # Test .pt format
        pt_path = Path(self.temp_dir) / "test_dataset.pt"
        self.generator.save_dataset(dataset, pt_path)
        
        loaded_dataset_pt = self.generator.load_dataset(pt_path)
        
        np.testing.assert_array_equal(dataset['power_traces'], loaded_dataset_pt['power_traces'])


class TestSecurityAndValidation(unittest.TestCase):
    """Test security policies and validation frameworks."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.security_policy = SecurityPolicy()
        self.validator = StatisticalValidator()
    
    def test_security_policy_enforcement(self):
        """Test security policy enforcement."""
        # Test rate limiting
        with self.assertRaises(Exception):
            for _ in range(self.security_policy.max_requests_per_hour + 1):
                self.security_policy.check_rate_limit("test_user")
    
    def test_input_validation(self):
        """Test input validation for security."""
        from neural_cryptanalysis.utils.validation import validate_input
        
        # Valid inputs should pass
        validate_input(100, int, min_value=1, max_value=1000)
        validate_input([1, 2, 3], list, min_length=1, max_length=10)
        
        # Invalid inputs should raise exceptions
        with self.assertRaises(ValueError):
            validate_input(-1, int, min_value=0)
        
        with self.assertRaises(ValueError):
            validate_input([], list, min_length=1)
    
    def test_statistical_validation(self):
        """Test statistical validation framework."""
        # Generate test data
        group1 = np.random.normal(0.8, 0.1, 100)
        group2 = np.random.normal(0.6, 0.1, 100)
        
        groups = {'group1': group1, 'group2': group2}
        
        results = self.validator.compare_groups(groups, metric_name='accuracy')
        
        self.assertIn('statistical_test', results)
        self.assertIn('p_value', results)
        self.assertIn('effect_size', results)
        self.assertIsInstance(results['p_value'], float)
    
    def test_responsible_use_compliance(self):
        """Test responsible use compliance checking."""
        # This should be implemented based on the specific responsible use framework
        # For now, test that the security policy exists and has required methods
        
        self.assertTrue(hasattr(self.security_policy, 'check_authorization'))
        self.assertTrue(hasattr(self.security_policy, 'audit_log'))
        self.assertTrue(hasattr(self.security_policy, 'check_rate_limit'))


class TestPerformanceBenchmarks(unittest.TestCase):
    """Test performance benchmarks and optimization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.profiler = PerformanceProfiler()
        torch.manual_seed(42)
        
        # Create test data
        self.test_data = torch.randn(100, 1000, 1)
        self.test_labels = torch.randint(0, 256, (100,))
    
    def test_training_performance(self):
        """Test training performance benchmarks."""
        neural_sca = NeuralSCA(
            architecture='fourier_neural_operator',
            channels=['power'],
            config={
                'fno': {'modes': 8, 'width': 32, 'n_layers': 2},
                'training': {'batch_size': 32, 'epochs': 2}
            }
        )
        
        # Benchmark training
        start_time = time.time()
        with self.profiler.profile("training_test"):
            model = neural_sca.train(
                traces=self.test_data,
                labels=self.test_labels,
                validation_split=0.2
            )
        training_time = time.time() - start_time
        
        # Training should complete in reasonable time
        self.assertLess(training_time, 60.0)  # Less than 1 minute
        self.assertIsNotNone(model)
    
    def test_inference_performance(self):
        """Test inference performance benchmarks."""
        # Quick model training
        neural_sca = NeuralSCA(
            architecture='fourier_neural_operator',
            channels=['power'],
            config={
                'fno': {'modes': 4, 'width': 16, 'n_layers': 1},
                'training': {'batch_size': 32, 'epochs': 1}
            }
        )
        
        model = neural_sca.train(
            traces=self.test_data[:50],
            labels=self.test_labels[:50],
            validation_split=0.2
        )
        
        # Benchmark inference
        test_batch = self.test_data[:32]
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):  # Multiple runs for average
                predictions = model(test_batch)
        inference_time = time.time() - start_time
        
        avg_time_per_batch = inference_time / 10
        avg_time_per_trace = avg_time_per_batch / 32
        
        # Inference should be fast
        self.assertLess(avg_time_per_trace, 0.01)  # Less than 10ms per trace
    
    def test_memory_efficiency(self):
        """Test memory usage during training and inference."""
        import psutil
        import gc
        
        # Measure baseline memory
        gc.collect()
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Create and train model
        neural_sca = NeuralSCA(
            architecture='fourier_neural_operator',
            channels=['power'],
            config={
                'fno': {'modes': 8, 'width': 32, 'n_layers': 2},
                'training': {'batch_size': 16, 'epochs': 1}
            }
        )
        
        model = neural_sca.train(
            traces=self.test_data[:50],
            labels=self.test_labels[:50],
            validation_split=0.2
        )
        
        # Measure peak memory
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_usage = peak_memory - baseline_memory
        
        # Clean up
        del model, neural_sca
        gc.collect()
        
        # Memory usage should be reasonable (less than 1GB for this small test)
        self.assertLess(memory_usage, 1024)
    
    def test_scalability_limits(self):
        """Test system behavior at scale limits."""
        # Test with larger dataset
        large_traces = torch.randn(1000, 2000, 1)  # Larger traces
        large_labels = torch.randint(0, 256, (1000,))
        
        neural_sca = NeuralSCA(
            architecture='fourier_neural_operator',
            channels=['power'],
            config={
                'fno': {'modes': 4, 'width': 16, 'n_layers': 1},
                'training': {'batch_size': 64, 'epochs': 1}
            }
        )
        
        # This should complete without memory errors
        try:
            model = neural_sca.train(
                traces=large_traces[:200],  # Subset for testing
                labels=large_labels[:200],
                validation_split=0.2
            )
            self.assertIsNotNone(model)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.skipTest("Insufficient memory for scalability test")
            else:
                raise


class TestIntegrationScenarios(unittest.TestCase):
    """Test end-to-end integration scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        torch.manual_seed(42)
        
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_full_attack_pipeline(self):
        """Test complete attack pipeline from data generation to key recovery."""
        # 1. Generate synthetic dataset
        generator = SyntheticDatasetGenerator(random_seed=42)
        dataset = generator.generate_aes_dataset(
            n_traces=200,
            target_bytes=[0],
            trace_length=500
        )
        
        # 2. Initialize neural SCA
        neural_sca = NeuralSCA(
            architecture='fourier_neural_operator',
            channels=['power'],
            config={
                'fno': {'modes': 8, 'width': 32, 'n_layers': 2},
                'training': {'batch_size': 32, 'epochs': 5}
            }
        )
        
        # 3. Train model
        traces = torch.tensor(dataset['power_traces'], dtype=torch.float32).unsqueeze(-1)
        labels = torch.tensor(dataset['labels'][0], dtype=torch.long)
        
        model = neural_sca.train(
            traces=traces[:150],
            labels=labels[:150],
            validation_split=0.2
        )
        
        # 4. Perform attack
        attack_results = neural_sca.attack(
            target_traces=traces[150:],
            model=model,
            strategy='template',
            target_byte=0,
            plaintexts=dataset['plaintexts'][150:]
        )
        
        # 5. Validate results
        self.assertIn('predicted_key_byte', attack_results)
        self.assertIn('confidence', attack_results)
        
        # Check if attack was successful (should be with synthetic data)
        true_key_byte = dataset['key'][0]
        predicted_key_byte = attack_results['predicted_key_byte']
        
        # With good synthetic data, attack should succeed
        self.assertEqual(predicted_key_byte, true_key_byte)
        self.assertGreater(attack_results['confidence'], 0.5)
    
    def test_multi_architecture_comparison(self):
        """Test comparison of multiple neural operator architectures."""
        # Generate test data
        generator = SyntheticDatasetGenerator(random_seed=42)
        dataset = generator.generate_aes_dataset(n_traces=100, trace_length=200)
        
        traces = torch.tensor(dataset['power_traces'], dtype=torch.float32).unsqueeze(-1)
        labels = torch.tensor(dataset['labels'][0], dtype=torch.long)
        
        architectures = ['fourier_neural_operator', 'deep_operator_network']
        results = {}
        
        for arch in architectures:
            try:
                neural_sca = NeuralSCA(
                    architecture=arch,
                    channels=['power'],
                    config={
                        'training': {'batch_size': 16, 'epochs': 2}
                    }
                )
                
                start_time = time.time()
                model = neural_sca.train(
                    traces=traces[:70],
                    labels=labels[:70],
                    validation_split=0.2
                )
                training_time = time.time() - start_time
                
                # Test accuracy
                with torch.no_grad():
                    predictions = model(traces[70:])
                    predicted_labels = torch.argmax(predictions, dim=1)
                    accuracy = (predicted_labels == labels[70:]).float().mean().item()
                
                results[arch] = {
                    'accuracy': accuracy,
                    'training_time': training_time
                }
                
            except Exception as e:
                warnings.warn(f"Architecture {arch} failed: {e}")
                continue
        
        # At least one architecture should work
        self.assertGreater(len(results), 0)
        
        # Results should contain expected metrics
        for arch, metrics in results.items():
            self.assertIn('accuracy', metrics)
            self.assertIn('training_time', metrics)
            self.assertIsInstance(metrics['accuracy'], float)
            self.assertIsInstance(metrics['training_time'], float)


def run_comprehensive_tests():
    """Run all comprehensive tests with detailed reporting."""
    
    # Configure test discovery
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestNeuralOperatorArchitectures,
        TestSideChannelAnalysis,
        TestPostQuantumCryptanalysis,
        TestDatasetGeneration,
        TestSecurityAndValidation,
        TestPerformanceBenchmarks,
        TestIntegrationScenarios
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        buffer=True
    )
    
    print("🧪 Running Comprehensive Neural Cryptanalysis Test Suite")
    print("=" * 70)
    
    result = runner.run(suite)
    
    # Generate summary report
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    success = total_tests - failures - errors - skipped
    
    print("\n" + "=" * 70)
    print("📊 Test Results Summary:")
    print(f"  Total tests: {total_tests}")
    print(f"  Successful: {success}")
    print(f"  Failures: {failures}")
    print(f"  Errors: {errors}")
    print(f"  Skipped: {skipped}")
    print(f"  Success rate: {(success/total_tests)*100:.1f}%")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed! Framework is production-ready.")
    else:
        print("\n❌ Some tests failed. Review output above.")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)