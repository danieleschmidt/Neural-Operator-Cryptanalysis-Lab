"""Test configuration and fixtures for neural cryptanalysis testing framework."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys
import warnings
from unittest.mock import Mock, patch

# Mock torch if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create a mock torch module
    torch = Mock()
    torch.manual_seed = Mock()
    torch.cuda = Mock()
    torch.cuda.is_available = Mock(return_value=False)
    torch.cuda.manual_seed = Mock()
    torch.cuda.empty_cache = Mock()
    torch.backends = Mock()
    torch.backends.cudnn = Mock()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Filter warnings for cleaner test output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up the test environment for the entire test session."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    if TORCH_AVAILABLE:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Mock torch operations
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
    
    yield
    
    # Cleanup after all tests
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif not TORCH_AVAILABLE:
        torch.cuda.empty_cache()


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_traces():
    """Generate sample power traces for testing."""
    n_traces = 100
    trace_length = 1000
    
    # Generate traces with some realistic characteristics
    traces = np.random.randn(n_traces, trace_length) * 0.1
    
    # Add some signal components
    for i in range(n_traces):
        signal = np.sin(np.linspace(0, 4*np.pi, trace_length)) * 0.05
        noise = np.random.randn(trace_length) * 0.02
        traces[i] += signal + noise
    
    return traces


@pytest.fixture
def sample_labels():
    """Generate sample labels for testing."""
    return np.random.randint(0, 256, 100)


@pytest.fixture
def sample_plaintexts():
    """Generate sample AES plaintexts for testing."""
    return np.random.randint(0, 256, (100, 16), dtype=np.uint8)


@pytest.fixture
def sample_key():
    """Generate a sample AES key for testing."""
    return np.random.randint(0, 256, 16, dtype=np.uint8)


@pytest.fixture
def multimodal_test_data():
    """Generate multi-modal test data."""
    n_traces = 50
    trace_length = 500
    
    return {
        'power_traces': np.random.randn(n_traces, trace_length) * 0.1,
        'em_near_traces': np.random.randn(n_traces, trace_length) * 0.08,
        'acoustic_traces': np.random.randn(n_traces, trace_length) * 0.05
    }


@pytest.fixture
def neural_sca_config():
    """Standard configuration for NeuralSCA testing."""
    return {
        'fno': {
            'modes': 8,
            'width': 32,
            'n_layers': 2,
            'in_channels': 1,
            'out_channels': 256
        },
        'deeponet': {
            'branch_layers': [64, 64],
            'trunk_layers': [64, 64],
            'output_dim': 256
        },
        'training': {
            'batch_size': 32,
            'epochs': 3,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'early_stopping_patience': 5
        },
        'attack': {
            'strategy': 'template',
            'target_byte': 0,
            'confidence_threshold': 0.8
        }
    }


@pytest.fixture
def mock_hardware_device():
    """Mock hardware device for testing hardware integration."""
    device = Mock()
    device.is_connected = False
    device.connect = Mock(return_value=True)
    device.disconnect = Mock(return_value=True)
    device.configure = Mock(return_value=True)
    device.measure = Mock(return_value=np.random.randn(1000))
    device.get_status = Mock(return_value={'status': 'ready'})
    return device


@pytest.fixture
def performance_benchmarks():
    """Performance benchmark thresholds for testing."""
    return {
        'training_time_per_epoch_seconds': 30.0,
        'inference_time_per_trace_ms': 10.0,
        'memory_usage_mb': 1024.0,
        'cpu_utilization_percent': 80.0,
        'accuracy_threshold': 0.7,
        'convergence_epochs': 10
    }


@pytest.fixture
def security_config():
    """Security configuration for testing."""
    return {
        'max_traces_per_hour': 10000,
        'max_memory_usage_gb': 8,
        'allowed_operations': ['train', 'predict', 'evaluate'],
        'audit_logging': True,
        'rate_limiting': True,
        'input_validation': True
    }


class TestDataGenerator:
    """Helper class for generating various types of test data."""
    
    @staticmethod
    def generate_kyber_ntt_traces(n_traces=100, trace_length=2000):
        """Generate synthetic Kyber NTT traces."""
        traces = np.random.randn(n_traces, trace_length) * 0.1
        
        # Add NTT-specific leakage patterns
        for i in range(n_traces):
            # Simulate butterfly operations
            butterfly_positions = np.random.choice(trace_length, 8, replace=False)
            for pos in butterfly_positions:
                if pos + 50 < trace_length:
                    traces[i, pos:pos+50] += np.random.randn(50) * 0.02
        
        return traces
    
    @staticmethod
    def generate_aes_sbox_traces(n_traces=100, trace_length=1000):
        """Generate synthetic AES S-box traces."""
        traces = np.random.randn(n_traces, trace_length) * 0.05
        
        # Add S-box leakage patterns
        sbox_window = slice(400, 600)  # S-box operation window
        for i in range(n_traces):
            hamming_weight = np.random.randint(0, 9)  # 0-8 bits set
            leakage_amplitude = hamming_weight * 0.01
            traces[i, sbox_window] += np.random.randn(200) * leakage_amplitude
        
        return traces
    
    @staticmethod
    def generate_noisy_traces(clean_traces, snr_db=10):
        """Add noise to clean traces with specified SNR."""
        signal_power = np.mean(clean_traces**2)
        noise_power = signal_power / (10**(snr_db/10))
        noise = np.random.randn(*clean_traces.shape) * np.sqrt(noise_power)
        return clean_traces + noise


@pytest.fixture
def test_data_generator():
    """Provide test data generator instance."""
    return TestDataGenerator()


# Performance monitoring fixtures
@pytest.fixture
def performance_monitor():
    """Monitor test performance and resource usage."""
    import psutil
    import time
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            self.process = psutil.Process()
        
        def start(self):
            self.start_time = time.perf_counter()
            self.start_memory = self.process.memory_info().rss
        
        def stop(self):
            if self.start_time is None:
                return None
            
            end_time = time.perf_counter()
            end_memory = self.process.memory_info().rss
            
            return {
                'execution_time': end_time - self.start_time,
                'memory_used_mb': (end_memory - self.start_memory) / 1024 / 1024,
                'peak_memory_mb': self.process.memory_info().rss / 1024 / 1024
            }
    
    return PerformanceMonitor()


# Quality gate markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: unit tests")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "performance: performance tests")
    config.addinivalue_line("markers", "security: security tests")
    config.addinivalue_line("markers", "research: research quality tests")
    config.addinivalue_line("markers", "slow: slow running tests")
    config.addinivalue_line("markers", "gpu: tests requiring GPU")
    config.addinivalue_line("markers", "hardware: tests requiring hardware")


# Skip conditions
skip_if_no_gpu = pytest.mark.skipif(
    not TORCH_AVAILABLE or not torch.cuda.is_available(),
    reason="GPU not available"
)

skip_if_no_hardware = pytest.mark.skipif(
    True,  # Always skip hardware tests in CI
    reason="Hardware not available in test environment"
)

# Parameterized test data
neural_operator_architectures = pytest.mark.parametrize(
    "architecture,config_params",
    [
        ("fourier_neural_operator", {"modes": 8, "width": 32}),
        ("deep_operator_network", {"branch_layers": [64, 64], "trunk_layers": [64, 64]}),
        ("side_channel_fno", {"modes": 8, "preprocessing": "normalize"}),
        ("leakage_fno", {"operation_type": "aes_sbox"}),
    ]
)

countermeasure_types = pytest.mark.parametrize(
    "countermeasure_type,order",
    [
        ("boolean_masking", 1),
        ("boolean_masking", 2),
        ("arithmetic_masking", 1),
        ("temporal_shuffling", None),
    ]
)

side_channel_modalities = pytest.mark.parametrize(
    "modalities",
    [
        (["power"]),
        (["power", "em_near"]),
        (["power", "em_near", "acoustic"]),
        (["em_far", "acoustic"]),
    ]
)