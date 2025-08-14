"""Comprehensive unit tests for all neural cryptanalysis components."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock
import asyncio

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

# Import test fixtures and utilities
from conftest import (
    neural_operator_architectures, countermeasure_types, 
    side_channel_modalities, skip_if_no_gpu
)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import core components
from neural_cryptanalysis.core import NeuralSCA, TraceData
from neural_cryptanalysis.neural_operators import (
    FourierNeuralOperator, DeepOperatorNetwork, OperatorConfig
)
from neural_cryptanalysis.neural_operators.custom import (
    SideChannelFNO, LeakageFNO, MultiModalOperator
)
from neural_cryptanalysis.side_channels import SideChannelAnalyzer, AnalysisConfig
from neural_cryptanalysis.targets.post_quantum import KyberImplementation
from neural_cryptanalysis.datasets.synthetic import SyntheticDatasetGenerator
from neural_cryptanalysis.utils.validation import validate_trace_data, ValidationContext
from neural_cryptanalysis.utils.errors import (
    ValidationError, ConfigurationError, ModelError, DataError
)


class TestNeuralSCACore:
    """Test the core NeuralSCA API."""
    
    def test_initialization_default(self):
        """Test NeuralSCA initialization with default parameters."""
        neural_sca = NeuralSCA()
        
        assert neural_sca.architecture == 'fourier_neural_operator'
        assert neural_sca.channels == ['power']
        assert neural_sca.config is not None
        assert isinstance(neural_sca.config, dict)
    
    def test_initialization_custom(self, neural_sca_config):
        """Test NeuralSCA initialization with custom parameters."""
        neural_sca = NeuralSCA(
            architecture='deep_operator_network',
            channels=['power', 'em_near'],
            config=neural_sca_config
        )
        
        assert neural_sca.architecture == 'deep_operator_network'
        assert neural_sca.channels == ['power', 'em_near']
        assert neural_sca.config == neural_sca_config
    
    def test_initialization_validation_errors(self):
        """Test NeuralSCA initialization validation."""
        # Test invalid architecture
        with pytest.raises(ValidationError):
            NeuralSCA(architecture="")
        
        with pytest.raises(ValidationError):
            NeuralSCA(architecture=123)
        
        # Test invalid channels
        with pytest.raises(ValidationError):
            NeuralSCA(channels="power")  # Should be list
        
        with pytest.raises(ValidationError):
            NeuralSCA(channels=[123, 456])  # Should be strings
    
    def test_model_creation(self, neural_sca_config):
        """Test neural operator model creation."""
        neural_sca = NeuralSCA(
            architecture='fourier_neural_operator',
            config=neural_sca_config
        )
        
        # Create model should work without errors
        model = neural_sca._create_model()
        assert model is not None
        assert hasattr(model, 'forward')
    
    @neural_operator_architectures
    def test_architecture_compatibility(self, architecture, config_params, neural_sca_config):
        """Test compatibility across different neural operator architectures."""
        neural_sca_config.update({architecture.replace('_', ''): config_params})
        
        neural_sca = NeuralSCA(
            architecture=architecture,
            config=neural_sca_config
        )
        
        # Should create model without errors
        model = neural_sca._create_model()
        assert model is not None
    
    def test_training_basic(self, sample_traces, sample_labels, neural_sca_config):
        """Test basic training functionality."""
        # Use minimal config for fast testing
        config = neural_sca_config.copy()
        config['training']['epochs'] = 1
        config['training']['batch_size'] = 16
        
        neural_sca = NeuralSCA(config=config)
        
        # Convert to tensors
        traces = torch.tensor(sample_traces[:50], dtype=torch.float32).unsqueeze(-1)
        labels = torch.tensor(sample_labels[:50], dtype=torch.long)
        
        # Training should complete without errors
        model = neural_sca.train(traces, labels, validation_split=0.2)
        assert model is not None
        assert hasattr(model, 'forward')
    
    def test_inference(self, sample_traces, sample_labels, neural_sca_config):
        """Test model inference functionality."""
        config = neural_sca_config.copy()
        config['training']['epochs'] = 1
        
        neural_sca = NeuralSCA(config=config)
        
        # Quick training
        traces = torch.tensor(sample_traces[:30], dtype=torch.float32).unsqueeze(-1)
        labels = torch.tensor(sample_labels[:30], dtype=torch.long)
        
        model = neural_sca.train(traces, labels, validation_split=0.2)
        
        # Test inference
        test_traces = torch.tensor(sample_traces[30:40], dtype=torch.float32).unsqueeze(-1)
        
        with torch.no_grad():
            predictions = model(test_traces)
        
        assert predictions.shape == (10, 256)  # 10 traces, 256 classes
        assert not torch.isnan(predictions).any()
        assert not torch.isinf(predictions).any()
    
    def test_attack_simulation(self, sample_traces, sample_plaintexts, neural_sca_config):
        """Test attack simulation functionality."""
        config = neural_sca_config.copy()
        config['training']['epochs'] = 1
        
        neural_sca = NeuralSCA(config=config)
        
        # Quick training
        traces = torch.tensor(sample_traces[:30], dtype=torch.float32).unsqueeze(-1)
        labels = torch.tensor(sample_plaintexts[:30, 0], dtype=torch.long)  # First byte
        
        model = neural_sca.train(traces, labels, validation_split=0.2)
        
        # Test attack
        attack_traces = torch.tensor(sample_traces[30:40], dtype=torch.float32).unsqueeze(-1)
        attack_plaintexts = sample_plaintexts[30:40]
        
        attack_results = neural_sca.attack(
            target_traces=attack_traces,
            model=model,
            strategy='template',
            target_byte=0,
            plaintexts=attack_plaintexts
        )
        
        assert 'predicted_key_byte' in attack_results
        assert 'confidence' in attack_results
        assert isinstance(attack_results['predicted_key_byte'], (int, np.integer))
        assert isinstance(attack_results['confidence'], (float, np.floating))


class TestNeuralOperators:
    """Test neural operator implementations."""
    
    def test_fourier_neural_operator_creation(self):
        """Test FourierNeuralOperator creation and basic functionality."""
        config = OperatorConfig(
            input_dim=1,
            output_dim=256,
            hidden_dim=64,
            num_layers=2
        )
        
        fno = FourierNeuralOperator(config, modes=16)
        
        assert fno.modes == 16
        assert len(fno.layers) == 2
        assert hasattr(fno, 'fc0')
        assert hasattr(fno, 'fc1')
        assert hasattr(fno, 'fc2')
    
    def test_fourier_neural_operator_forward_pass(self):
        """Test FNO forward pass with different input shapes."""
        config = OperatorConfig(input_dim=1, output_dim=256, hidden_dim=32)
        fno = FourierNeuralOperator(config, modes=8)
        
        # Test 2D input (batch_size, sequence_length)
        x_2d = torch.randn(4, 100)
        output_2d = fno(x_2d)
        assert output_2d.shape == (4, 256)
        
        # Test 3D input (batch_size, sequence_length, channels)
        x_3d = torch.randn(4, 100, 1)
        output_3d = fno(x_3d)
        assert output_3d.shape == (4, 256)
        
        # Test no NaN or Inf values
        assert not torch.isnan(output_2d).any()
        assert not torch.isinf(output_2d).any()
        assert not torch.isnan(output_3d).any()
        assert not torch.isinf(output_3d).any()
    
    def test_deep_operator_network_creation(self):
        """Test DeepOperatorNetwork creation and functionality."""
        config = OperatorConfig(
            input_dim=50,  # Number of sensors
            output_dim=256,
            hidden_dim=64
        )
        
        deeponet = DeepOperatorNetwork(
            config,
            branch_layers=[128, 128],
            trunk_layers=[64, 64],
            coord_dim=1
        )
        
        assert hasattr(deeponet, 'branch_net')
        assert hasattr(deeponet, 'trunk_net')
        assert hasattr(deeponet, 'output_layer')
    
    def test_deep_operator_network_forward_pass(self):
        """Test DeepONet forward pass."""
        config = OperatorConfig(input_dim=50, output_dim=256)
        deeponet = DeepOperatorNetwork(
            config,
            branch_layers=[64, 64],
            trunk_layers=[64, 64],
            coord_dim=1
        )
        
        # Test with sensor measurements and evaluation points
        u = torch.randn(4, 50)  # Sensor measurements
        y = torch.randn(4, 20, 1)  # Evaluation points
        
        output = deeponet(u, y)
        assert output.shape == (4, 256)
        assert not torch.isnan(output).any()
        
        # Test with default evaluation points
        output_default = deeponet(u)
        assert output_default.shape == (4, 256)
        assert not torch.isnan(output_default).any()
    
    def test_side_channel_fno(self):
        """Test SideChannelFNO specialized implementation."""
        config = OperatorConfig(input_dim=1, output_dim=256, hidden_dim=32)
        sc_fno = SideChannelFNO(
            config,
            modes=8,
            trace_length=1000,
            preprocessing='standardize'
        )
        
        assert sc_fno.modes == 8
        assert sc_fno.trace_length == 1000
        assert sc_fno.preprocessing == 'standardize'
        
        # Test forward pass
        x = torch.randn(4, 1000)
        output = sc_fno(x)
        assert output.shape == (4, 256)
        assert not torch.isnan(output).any()
    
    def test_leakage_fno(self):
        """Test LeakageFNO specialized implementation."""
        config = OperatorConfig(input_dim=1, output_dim=256, hidden_dim=32)
        leakage_fno = LeakageFNO(config, operation_type='aes_sbox')
        
        assert leakage_fno.operation_type == 'aes_sbox'
        assert len(leakage_fno.scales) == 3
        assert len(leakage_fno.spectral_branches) == 3
        
        # Test forward pass
        x = torch.randn(4, 500)
        output = leakage_fno(x)
        assert output.shape == (4, 256)
        assert not torch.isnan(output).any()
        
        # Test with intermediate values
        intermediate_values = torch.randint(0, 256, (4,))
        output_with_values = leakage_fno(x, intermediate_values)
        assert output_with_values.shape == (4, 256)
        assert not torch.isnan(output_with_values).any()
    
    def test_multi_modal_operator(self):
        """Test MultiModalOperator implementation."""
        config = OperatorConfig(input_dim=2, output_dim=256, hidden_dim=64)
        mm_operator = MultiModalOperator(
            config,
            modalities=['power', 'em_near'],
            fusion_method='attention'
        )
        
        assert mm_operator.modalities == ['power', 'em_near']
        assert mm_operator.fusion_method == 'attention'
        
        # Test forward pass with multiple modalities
        power_traces = torch.randn(4, 1000)
        em_traces = torch.randn(4, 1000)
        
        output = mm_operator([power_traces, em_traces])
        assert output.shape == (4, 256)
        assert not torch.isnan(output).any()
    
    def test_parameter_counting(self):
        """Test parameter counting functionality."""
        config = OperatorConfig(input_dim=1, output_dim=256, hidden_dim=64)
        fno = FourierNeuralOperator(config, modes=16)
        
        param_count = fno.count_parameters()
        assert param_count > 0
        assert isinstance(param_count, int)
        
        # Verify by manual counting
        manual_count = sum(p.numel() for p in fno.parameters())
        assert param_count == manual_count
    
    def test_gradient_computation(self):
        """Test gradient computation through neural operators."""
        config = OperatorConfig(input_dim=1, output_dim=256, hidden_dim=32)
        fno = FourierNeuralOperator(config, modes=8)
        
        # Create input requiring gradients
        x = torch.randn(4, 100, 1, requires_grad=True)
        target = torch.randint(0, 256, (4,))
        
        # Forward pass
        output = fno(x)
        loss = torch.nn.functional.cross_entropy(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist and are non-zero
        assert x.grad is not None
        gradient_norm = sum(p.grad.norm().item() for p in fno.parameters() if p.grad is not None)
        assert gradient_norm > 0


class TestSideChannelComponents:
    """Test side-channel analysis components."""
    
    def test_trace_data_creation(self, sample_traces, sample_labels):
        """Test TraceData creation and validation."""
        trace_data = TraceData(traces=sample_traces, labels=sample_labels)
        
        assert trace_data.traces.shape == sample_traces.shape
        assert len(trace_data.labels) == len(sample_labels)
        assert trace_data.n_traces == len(sample_traces)
        assert trace_data.trace_length == sample_traces.shape[1]
    
    def test_trace_data_validation(self):
        """Test TraceData validation functionality."""
        # Test mismatched traces and labels
        traces = np.random.randn(100, 1000)
        labels = np.random.randint(0, 256, 50)  # Wrong length
        
        with pytest.raises(ValidationError):
            TraceData(traces=traces, labels=labels)
        
        # Test invalid trace shape
        invalid_traces = np.random.randn(100)  # 1D instead of 2D
        valid_labels = np.random.randint(0, 256, 100)
        
        with pytest.raises(ValidationError):
            TraceData(traces=invalid_traces, labels=valid_labels)
    
    def test_side_channel_analyzer_initialization(self):
        """Test SideChannelAnalyzer initialization."""
        config = AnalysisConfig(
            channel_types=['power'],
            preprocessing_methods=['standardize'],
            analysis_methods=['correlation', 'mutual_information']
        )
        
        analyzer = SideChannelAnalyzer(config)
        assert analyzer.config == config
        assert len(analyzer.preprocessors) == 1
        assert len(analyzer.analyzers) == 2
    
    def test_preprocessing_methods(self, sample_traces):
        """Test various preprocessing methods."""
        config = AnalysisConfig(
            channel_types=['power'],
            preprocessing_methods=['standardize', 'normalize', 'filter']
        )
        
        analyzer = SideChannelAnalyzer(config)
        
        # Test standardization
        standardized = analyzer.preprocess(sample_traces, method='standardize')
        assert np.abs(np.mean(standardized)) < 1e-10  # Should be close to 0
        assert np.abs(np.std(standardized) - 1.0) < 1e-6  # Should be close to 1
        
        # Test normalization
        normalized = analyzer.preprocess(sample_traces, method='normalize')
        assert normalized.min() >= 0
        assert normalized.max() <= 1
    
    def test_leakage_analysis_methods(self, sample_traces, sample_labels):
        """Test different leakage analysis methods."""
        config = AnalysisConfig(
            channel_types=['power'],
            analysis_methods=['correlation', 'mutual_information', 't_test']
        )
        
        analyzer = SideChannelAnalyzer(config)
        trace_data = TraceData(traces=sample_traces, labels=sample_labels)
        
        # Test correlation analysis
        correlation_results = analyzer.analyze_leakage(trace_data, method='correlation')
        assert 'correlations' in correlation_results
        assert len(correlation_results['correlations']) == sample_traces.shape[1]
        
        # Test mutual information
        mi_results = analyzer.analyze_leakage(trace_data, method='mutual_information')
        assert 'mutual_information' in mi_results
        assert len(mi_results['mutual_information']) == sample_traces.shape[1]
        
        # Test t-test
        ttest_results = analyzer.analyze_leakage(trace_data, method='t_test')
        assert 't_statistics' in ttest_results
        assert len(ttest_results['t_statistics']) == sample_traces.shape[1]


class TestPostQuantumTargets:
    """Test post-quantum cryptography target implementations."""
    
    def test_kyber_implementation_creation(self):
        """Test Kyber implementation creation."""
        kyber = KyberImplementation(
            variant='kyber768',
            platform='arm_cortex_m4',
            countermeasures=[]
        )
        
        assert kyber.variant == 'kyber768'
        assert kyber.platform == 'arm_cortex_m4'
        assert kyber.countermeasures == []
        assert isinstance(kyber.q, int)
        assert isinstance(kyber.n, int)
        assert kyber.q > 0
        assert kyber.n > 0
    
    def test_kyber_parameter_validation(self):
        """Test Kyber parameter validation."""
        # Test invalid variant
        with pytest.raises(ValidationError):
            KyberImplementation(variant='invalid_variant')
        
        # Test invalid platform
        with pytest.raises(ValidationError):
            KyberImplementation(platform='unknown_platform')
    
    def test_kyber_ntt_operations(self):
        """Test Kyber NTT operation simulation."""
        kyber = KyberImplementation(variant='kyber512')
        
        # Generate random polynomial coefficients
        coefficients = np.random.randint(0, kyber.q, kyber.n, dtype=np.int16)
        
        # Test NTT forward transform
        ntt_result = kyber.ntt_forward(coefficients)
        assert len(ntt_result) == kyber.n
        assert ntt_result.dtype == np.int16
        
        # Test NTT inverse transform
        intt_result = kyber.ntt_inverse(ntt_result)
        assert len(intt_result) == kyber.n
        
        # Should be close to original (modulo arithmetic)
        diff = (coefficients - intt_result) % kyber.q
        assert np.allclose(diff, 0, atol=1)
    
    def test_kyber_with_countermeasures(self):
        """Test Kyber implementation with countermeasures."""
        kyber_masked = KyberImplementation(
            variant='kyber768',
            countermeasures=['masking', 'shuffling']
        )
        
        assert 'masking' in kyber_masked.countermeasures
        assert 'shuffling' in kyber_masked.countermeasures
        
        # Operations should still work with countermeasures
        coefficients = np.random.randint(0, kyber_masked.q, kyber_masked.n, dtype=np.int16)
        ntt_result = kyber_masked.ntt_forward(coefficients)
        assert len(ntt_result) == kyber_masked.n


class TestDatasetGeneration:
    """Test synthetic dataset generation."""
    
    def test_synthetic_dataset_generator_creation(self):
        """Test SyntheticDatasetGenerator creation."""
        generator = SyntheticDatasetGenerator(random_seed=42)
        assert generator.random_seed == 42
        assert generator.rng is not None
    
    def test_aes_dataset_generation(self):
        """Test AES dataset generation."""
        generator = SyntheticDatasetGenerator(random_seed=42)
        
        dataset = generator.generate_aes_dataset(
            n_traces=100,
            target_bytes=[0, 1],
            trace_length=500
        )
        
        assert 'power_traces' in dataset
        assert 'labels' in dataset
        assert 'plaintexts' in dataset
        assert 'key' in dataset
        assert 'metadata' in dataset
        
        assert dataset['power_traces'].shape == (100, 500)
        assert len(dataset['labels']) == 2  # Two target bytes
        assert len(dataset['labels'][0]) == 100
        assert len(dataset['labels'][1]) == 100
        assert dataset['plaintexts'].shape == (100, 16)
        assert len(dataset['key']) == 16
    
    def test_kyber_dataset_generation(self):
        """Test Kyber dataset generation."""
        generator = SyntheticDatasetGenerator(random_seed=42)
        
        dataset = generator.generate_kyber_dataset(
            n_traces=50,
            variant='kyber512',
            trace_length=2000
        )
        
        assert 'power_traces' in dataset
        assert 'coefficients' in dataset
        assert 'ntt_outputs' in dataset
        assert 'metadata' in dataset
        
        assert dataset['power_traces'].shape == (50, 2000)
        assert len(dataset['coefficients']) == 50
        assert len(dataset['ntt_outputs']) == 50
    
    def test_noise_model_configuration(self):
        """Test different noise model configurations."""
        from neural_cryptanalysis.datasets.synthetic import NoiseModel
        
        # Test Gaussian noise
        gaussian_noise = NoiseModel(noise_type='gaussian', snr_db=10)
        generator = SyntheticDatasetGenerator(noise_model=gaussian_noise, random_seed=42)
        
        dataset = generator.generate_aes_dataset(n_traces=20, trace_length=100)
        assert dataset['metadata']['noise_model']['type'] == 'gaussian'
        assert dataset['metadata']['noise_model']['snr_db'] == 10
        
        # Test uniform noise
        uniform_noise = NoiseModel(noise_type='uniform', snr_db=5)
        generator_uniform = SyntheticDatasetGenerator(noise_model=uniform_noise, random_seed=42)
        
        dataset_uniform = generator_uniform.generate_aes_dataset(n_traces=20, trace_length=100)
        assert dataset_uniform['metadata']['noise_model']['type'] == 'uniform'
    
    def test_dataset_save_load(self, temp_directory):
        """Test dataset saving and loading functionality."""
        generator = SyntheticDatasetGenerator(random_seed=42)
        dataset = generator.generate_aes_dataset(n_traces=30, trace_length=200)
        
        # Test .npz format
        npz_path = temp_directory / "test_dataset.npz"
        generator.save_dataset(dataset, npz_path)
        assert npz_path.exists()
        
        loaded_dataset = generator.load_dataset(npz_path)
        np.testing.assert_array_equal(dataset['power_traces'], loaded_dataset['power_traces'])
        np.testing.assert_array_equal(dataset['key'], loaded_dataset['key'])
        
        # Test .pt format
        pt_path = temp_directory / "test_dataset.pt"
        generator.save_dataset(dataset, pt_path)
        assert pt_path.exists()
        
        loaded_dataset_pt = generator.load_dataset(pt_path)
        np.testing.assert_array_equal(dataset['power_traces'], loaded_dataset_pt['power_traces'])


class TestUtilities:
    """Test utility functions and classes."""
    
    def test_validation_functions(self, sample_traces, sample_labels):
        """Test validation utility functions."""
        # Test trace data validation
        context = ValidationContext()
        
        # Valid data should pass
        validate_trace_data(sample_traces, sample_labels, context)
        
        # Invalid data should raise errors
        with pytest.raises(ValidationError):
            validate_trace_data(sample_traces, sample_labels[:-10], context)  # Mismatched lengths
        
        with pytest.raises(ValidationError):
            validate_trace_data(sample_traces.flatten(), sample_labels, context)  # Wrong shape
    
    def test_error_handling(self):
        """Test error handling and custom exceptions."""
        # Test ValidationError
        error = ValidationError("Test message", field="test_field", value="test_value")
        assert error.field == "test_field"
        assert error.value == "test_value"
        assert "Test message" in str(error)
        
        # Test ConfigurationError
        config_error = ConfigurationError("Config test", config_key="test_key")
        assert config_error.config_key == "test_key"
        
        # Test ModelError
        model_error = ModelError("Model test", model_type="FNO")
        assert model_error.model_type == "FNO"
        
        # Test DataError
        data_error = DataError("Data test", data_type="traces")
        assert data_error.data_type == "traces"
    
    def test_config_management(self, temp_directory):
        """Test configuration loading and saving."""
        from neural_cryptanalysis.utils.config import load_config, save_config
        
        # Test config saving
        test_config = {
            'architecture': 'fourier_neural_operator',
            'training': {'epochs': 10, 'batch_size': 32},
            'fno': {'modes': 16, 'width': 64}
        }
        
        config_path = temp_directory / "test_config.yaml"
        save_config(test_config, config_path)
        assert config_path.exists()
        
        # Test config loading
        loaded_config = load_config(config_path)
        assert loaded_config['architecture'] == 'fourier_neural_operator'
        assert loaded_config['training']['epochs'] == 10
        assert loaded_config['fno']['modes'] == 16
    
    def test_logging_utilities(self):
        """Test logging utility functions."""
        from neural_cryptanalysis.utils.logging_utils import get_logger, setup_logging
        
        # Test logger creation
        logger = get_logger(__name__)
        assert logger is not None
        assert logger.name == __name__
        
        # Test logging setup
        setup_logging(level='INFO')
        logger.info("Test log message")  # Should not raise errors
    
    def test_performance_monitoring(self):
        """Test performance monitoring utilities."""
        from neural_cryptanalysis.utils.performance import PerformanceProfiler
        
        profiler = PerformanceProfiler()
        
        # Test profiling context
        with profiler.profile("test_operation"):
            # Simulate some work
            time.sleep(0.1)
        
        stats = profiler.get_stats("test_operation")
        assert stats is not None
        assert stats['total_time'] >= 0.1
        assert stats['call_count'] == 1


class TestReproducibility:
    """Test reproducibility and deterministic behavior."""
    
    def test_random_seed_reproducibility(self):
        """Test that random seeds produce reproducible results."""
        # Test with NeuralSCA
        neural_sca1 = NeuralSCA(architecture='fourier_neural_operator')
        neural_sca2 = NeuralSCA(architecture='fourier_neural_operator')
        
        # Create identical test data
        torch.manual_seed(123)
        traces1 = torch.randn(20, 100, 1)
        labels1 = torch.randint(0, 256, (20,))
        
        torch.manual_seed(123)
        traces2 = torch.randn(20, 100, 1)
        labels2 = torch.randint(0, 256, (20,))
        
        # Should be identical
        assert torch.allclose(traces1, traces2)
        assert torch.equal(labels1, labels2)
    
    def test_model_reproducibility(self, neural_sca_config):
        """Test that model training is reproducible with fixed seeds."""
        config = neural_sca_config.copy()
        config['training']['epochs'] = 1
        
        # Train two identical models
        torch.manual_seed(456)
        np.random.seed(456)
        neural_sca1 = NeuralSCA(config=config)
        
        torch.manual_seed(456)
        np.random.seed(456)
        neural_sca2 = NeuralSCA(config=config)
        
        # Generate identical training data
        torch.manual_seed(789)
        traces = torch.randn(30, 100, 1)
        labels = torch.randint(0, 256, (30,))
        
        # Train both models
        model1 = neural_sca1.train(traces, labels, validation_split=0.2)
        model2 = neural_sca2.train(traces, labels, validation_split=0.2)
        
        # Test on same input
        torch.manual_seed(999)
        test_input = torch.randn(5, 100, 1)
        
        with torch.no_grad():
            output1 = model1(test_input)
            output2 = model2(test_input)
        
        # Outputs should be very close (allowing for small numerical differences)
        assert torch.allclose(output1, output2, atol=1e-4)
    
    def test_dataset_reproducibility(self):
        """Test that dataset generation is reproducible."""
        # Generate two datasets with same seed
        generator1 = SyntheticDatasetGenerator(random_seed=101)
        dataset1 = generator1.generate_aes_dataset(n_traces=50, trace_length=100)
        
        generator2 = SyntheticDatasetGenerator(random_seed=101)
        dataset2 = generator2.generate_aes_dataset(n_traces=50, trace_length=100)
        
        # Should be identical
        np.testing.assert_array_equal(dataset1['power_traces'], dataset2['power_traces'])
        np.testing.assert_array_equal(dataset1['plaintexts'], dataset2['plaintexts'])
        np.testing.assert_array_equal(dataset1['key'], dataset2['key'])


@pytest.mark.performance
class TestPerformanceRequirements:
    """Test performance requirements and benchmarks."""
    
    def test_training_performance(self, sample_traces, sample_labels, performance_benchmarks):
        """Test that training meets performance requirements."""
        neural_sca = NeuralSCA(config={
            'training': {'epochs': 1, 'batch_size': 32},
            'fno': {'modes': 8, 'width': 32, 'n_layers': 2}
        })
        
        traces = torch.tensor(sample_traces[:100], dtype=torch.float32).unsqueeze(-1)
        labels = torch.tensor(sample_labels[:100], dtype=torch.long)
        
        import time
        start_time = time.perf_counter()
        
        model = neural_sca.train(traces, labels, validation_split=0.2)
        
        training_time = time.perf_counter() - start_time
        
        # Should complete within reasonable time
        assert training_time < performance_benchmarks['training_time_per_epoch_seconds']
        assert model is not None
    
    def test_inference_performance(self, sample_traces, performance_benchmarks):
        """Test inference performance requirements."""
        neural_sca = NeuralSCA(config={
            'training': {'epochs': 1, 'batch_size': 16},
            'fno': {'modes': 4, 'width': 16, 'n_layers': 1}
        })
        
        # Quick training
        traces = torch.tensor(sample_traces[:50], dtype=torch.float32).unsqueeze(-1)
        labels = torch.randint(0, 256, (50,))
        model = neural_sca.train(traces, labels, validation_split=0.2)
        
        # Benchmark inference
        test_traces = torch.tensor(sample_traces[50:60], dtype=torch.float32).unsqueeze(-1)
        
        import time
        times = []
        with torch.no_grad():
            for _ in range(10):  # Multiple runs for average
                start_time = time.perf_counter()
                predictions = model(test_traces)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
        
        avg_time_per_batch = np.mean(times)
        avg_time_per_trace_ms = (avg_time_per_batch / len(test_traces)) * 1000
        
        assert avg_time_per_trace_ms < performance_benchmarks['inference_time_per_trace_ms']
    
    def test_memory_usage(self, sample_traces, performance_benchmarks):
        """Test memory usage requirements."""
        import psutil
        process = psutil.Process()
        
        # Measure baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and train model
        neural_sca = NeuralSCA(config={
            'training': {'epochs': 1, 'batch_size': 32},
            'fno': {'modes': 8, 'width': 32, 'n_layers': 2}
        })
        
        traces = torch.tensor(sample_traces, dtype=torch.float32).unsqueeze(-1)
        labels = torch.randint(0, 256, (len(traces),))
        
        model = neural_sca.train(traces, labels, validation_split=0.2)
        
        # Measure peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = peak_memory - baseline_memory
        
        assert memory_used < performance_benchmarks['memory_usage_mb']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])