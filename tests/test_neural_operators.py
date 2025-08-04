"""Tests for neural operator implementations."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from neural_cryptanalysis.neural_operators import (
    NeuralOperatorBase, OperatorConfig, FourierNeuralOperator,
    DeepOperatorNetwork, SideChannelFNO, LeakageFNO
)


class TestOperatorConfig:
    """Test OperatorConfig functionality."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = OperatorConfig()
        
        assert config.input_dim == 1
        assert config.output_dim == 256
        assert config.hidden_dim == 64
        assert config.num_layers == 4
        assert config.activation == "gelu"
        assert config.dropout == 0.1
        assert config.device == "cpu"
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = OperatorConfig(
            input_dim=2,
            output_dim=128,
            hidden_dim=32,
            activation="relu"
        )
        
        assert config.input_dim == 2
        assert config.output_dim == 128
        assert config.hidden_dim == 32
        assert config.activation == "relu"


class TestFourierNeuralOperator:
    """Test FourierNeuralOperator implementation."""
    
    @pytest.fixture
    def fno_config(self):
        """Create test configuration for FNO."""
        return OperatorConfig(
            input_dim=1,
            output_dim=256,
            hidden_dim=32,
            num_layers=2,
            device="cpu"
        )
    
    @pytest.fixture
    def fno_model(self, fno_config):
        """Create FNO model for testing."""
        return FourierNeuralOperator(fno_config, modes=8)
    
    def test_initialization(self, fno_model):
        """Test FNO initialization."""
        assert isinstance(fno_model, FourierNeuralOperator)
        assert fno_model.modes == 8
        assert len(fno_model.layers) == 2
        assert hasattr(fno_model, 'fc0')
        assert hasattr(fno_model, 'fc1')
        assert hasattr(fno_model, 'fc2')
    
    def test_forward_pass_2d_input(self, fno_model):
        """Test forward pass with 2D input."""
        batch_size = 4
        sequence_length = 100
        
        x = torch.randn(batch_size, sequence_length)
        output = fno_model(x)
        
        assert output.shape == (batch_size, 256)
        assert not torch.isnan(output).any()
    
    def test_forward_pass_3d_input(self, fno_model):
        """Test forward pass with 3D input."""
        batch_size = 4
        sequence_length = 100
        channels = 1
        
        x = torch.randn(batch_size, sequence_length, channels)
        output = fno_model(x)
        
        assert output.shape == (batch_size, 256)
        assert not torch.isnan(output).any()
    
    def test_parameter_count(self, fno_model):
        """Test parameter counting."""
        param_count = fno_model.count_parameters()
        assert param_count > 0
        assert isinstance(param_count, int)
    
    def test_spectral_features(self, fno_model):
        """Test spectral feature extraction."""
        x = torch.randn(2, 100, 1)
        features = fno_model.get_spectral_features(x, layer_idx=0)
        
        assert features is not None
        assert features.shape[0] == 2  # Batch size
        assert not torch.isnan(features).any()
    
    def test_checkpoint_save_load(self, fno_model, tmp_path):
        """Test model checkpoint saving and loading."""
        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pth"
        fno_model.save_checkpoint(str(checkpoint_path), epoch=10, loss=0.5)
        
        assert checkpoint_path.exists()
        
        # Load checkpoint
        loaded_model, checkpoint_data = FourierNeuralOperator.load_checkpoint(str(checkpoint_path))
        
        assert isinstance(loaded_model, FourierNeuralOperator)
        assert checkpoint_data['epoch'] == 10
        assert checkpoint_data['loss'] == 0.5


class TestDeepOperatorNetwork:
    """Test DeepOperatorNetwork implementation."""
    
    @pytest.fixture
    def deeponet_config(self):
        """Create test configuration for DeepONet."""
        return OperatorConfig(
            input_dim=50,  # Number of sensors
            output_dim=256,
            hidden_dim=32,
            device="cpu"
        )
    
    @pytest.fixture
    def deeponet_model(self, deeponet_config):
        """Create DeepONet model for testing."""
        return DeepOperatorNetwork(
            deeponet_config,
            branch_layers=[64, 64],
            trunk_layers=[64, 64],
            coord_dim=1
        )
    
    def test_initialization(self, deeponet_model):
        """Test DeepONet initialization."""
        assert isinstance(deeponet_model, DeepOperatorNetwork)
        assert hasattr(deeponet_model, 'branch_net')
        assert hasattr(deeponet_model, 'trunk_net')
        assert hasattr(deeponet_model, 'output_layer')
    
    def test_forward_pass(self, deeponet_model):
        """Test forward pass."""
        batch_size = 4
        n_sensors = 50
        n_points = 20
        
        u = torch.randn(batch_size, n_sensors)  # Sensor measurements
        y = torch.randn(batch_size, n_points, 1)  # Evaluation points
        
        output = deeponet_model(u, y)
        
        assert output.shape == (batch_size, 256)
        assert not torch.isnan(output).any()
    
    def test_forward_pass_default_points(self, deeponet_model):
        """Test forward pass with default evaluation points."""
        batch_size = 4
        n_sensors = 50
        
        u = torch.randn(batch_size, n_sensors)
        output = deeponet_model(u)  # No evaluation points provided
        
        assert output.shape == (batch_size, 256)
        assert not torch.isnan(output).any()
    
    def test_operator_weights(self, deeponet_model):
        """Test operator weight extraction."""
        batch_size = 4
        n_sensors = 50
        
        u = torch.randn(batch_size, n_sensors)
        weights = deeponet_model.get_operator_weights(u)
        
        assert weights.shape == (batch_size, 64)  # Branch network output dim
        assert not torch.isnan(weights).any()


class TestSideChannelFNO:
    """Test SideChannelFNO implementation."""
    
    @pytest.fixture
    def sc_config(self):
        """Create test configuration for SideChannelFNO."""
        return OperatorConfig(
            input_dim=1,
            output_dim=256,
            hidden_dim=32,
            num_layers=2,
            device="cpu"
        )
    
    @pytest.fixture
    def sc_model(self, sc_config):
        """Create SideChannelFNO model for testing."""
        return SideChannelFNO(
            sc_config,
            modes=8,
            trace_length=1000,
            preprocessing='standardize'
        )
    
    def test_initialization(self, sc_model):
        """Test SideChannelFNO initialization."""
        assert isinstance(sc_model, SideChannelFNO)
        assert sc_model.modes == 8
        assert sc_model.trace_length == 1000
        assert sc_model.preprocessing == 'standardize'
    
    def test_forward_pass(self, sc_model):
        """Test forward pass."""
        batch_size = 4
        trace_length = 1000
        
        x = torch.randn(batch_size, trace_length)
        output = sc_model(x)
        
        assert output.shape == (batch_size, 256)
        assert not torch.isnan(output).any()
    
    def test_attention_weights(self, sc_model):
        """Test attention weight extraction."""
        batch_size = 2
        trace_length = 1000
        
        x = torch.randn(batch_size, trace_length)
        attention_weights = sc_model.get_attention_weights(x)
        
        assert attention_weights.shape[0] == batch_size
        assert not torch.isnan(attention_weights).any()


class TestLeakageFNO:
    """Test LeakageFNO implementation."""
    
    @pytest.fixture
    def leakage_config(self):
        """Create test configuration for LeakageFNO."""
        return OperatorConfig(
            input_dim=1,
            output_dim=256,
            hidden_dim=32,
            device="cpu"
        )
    
    @pytest.fixture
    def leakage_model(self, leakage_config):
        """Create LeakageFNO model for testing."""
        return LeakageFNO(leakage_config, operation_type='aes_sbox')
    
    def test_initialization(self, leakage_model):
        """Test LeakageFNO initialization."""
        assert isinstance(leakage_model, LeakageFNO)
        assert leakage_model.operation_type == 'aes_sbox'
        assert len(leakage_model.scales) == 3
        assert len(leakage_model.spectral_branches) == 3
    
    def test_forward_pass(self, leakage_model):
        """Test forward pass."""
        batch_size = 4
        trace_length = 500
        
        x = torch.randn(batch_size, trace_length)
        output = leakage_model(x)
        
        assert output.shape == (batch_size, 256)
        assert not torch.isnan(output).any()
    
    def test_forward_pass_with_intermediate_values(self, leakage_model):
        """Test forward pass with intermediate values."""
        batch_size = 4
        trace_length = 500
        
        x = torch.randn(batch_size, trace_length)
        intermediate_values = torch.randint(0, 256, (batch_size,))
        
        output = leakage_model(x, intermediate_values)
        
        assert output.shape == (batch_size, 256)
        assert not torch.isnan(output).any()


class TestSpectralConvolutions:
    """Test spectral convolution layers."""
    
    def test_spectral_conv1d(self):
        """Test 1D spectral convolution."""
        from neural_cryptanalysis.neural_operators.fno import SpectralConv1d
        
        conv = SpectralConv1d(in_channels=32, out_channels=32, modes=8)
        
        batch_size = 4
        channels = 32
        length = 100
        
        x = torch.randn(batch_size, channels, length)
        output = conv(x)
        
        assert output.shape == (batch_size, 32, length)
        assert not torch.isnan(output).any()
    
    def test_spectral_conv2d(self):
        """Test 2D spectral convolution."""
        from neural_cryptanalysis.neural_operators.fno import SpectralConv2d
        
        conv = SpectralConv2d(
            in_channels=16, 
            out_channels=16, 
            modes1=8, 
            modes2=8
        )
        
        batch_size = 4
        channels = 16
        height = 32
        width = 32
        
        x = torch.randn(batch_size, channels, height, width)
        output = conv(x)
        
        assert output.shape == (batch_size, 16, height, width)
        assert not torch.isnan(output).any()


class TestOperatorLoss:
    """Test custom loss functions."""
    
    def test_mse_loss(self):
        """Test MSE loss."""
        from neural_cryptanalysis.neural_operators.base import OperatorLoss
        
        loss_fn = OperatorLoss(loss_type='mse')
        
        pred = torch.randn(4, 10)
        target = torch.randn(4, 10)
        
        loss = loss_fn(pred, target)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_cross_entropy_loss(self):
        """Test cross-entropy loss."""
        from neural_cryptanalysis.neural_operators.base import OperatorLoss
        
        loss_fn = OperatorLoss(loss_type='cross_entropy')
        
        pred = torch.randn(4, 256)  # Logits for 256 classes
        target = torch.randint(0, 256, (4,))  # Class indices
        
        loss = loss_fn(pred, target)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_focal_loss(self):
        """Test focal loss."""
        from neural_cryptanalysis.neural_operators.base import OperatorLoss
        
        loss_fn = OperatorLoss(loss_type='focal')
        
        pred = torch.randn(4, 256)
        target = torch.randint(0, 256, (4,))
        
        loss = loss_fn(pred, target)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)


@pytest.mark.parametrize("architecture,config_params", [
    ("FourierNeuralOperator", {"modes": 16}),
    ("DeepOperatorNetwork", {"branch_layers": [64, 64], "trunk_layers": [64, 64]}),
    ("SideChannelFNO", {"modes": 8, "preprocessing": "normalize"}),
    ("LeakageFNO", {"operation_type": "kyber_ntt"}),
])
def test_architecture_integration(architecture, config_params):
    """Test integration of different architectures."""
    config = OperatorConfig(
        input_dim=1,
        output_dim=256,
        hidden_dim=32,
        num_layers=2,
        device="cpu"
    )
    
    # Create model based on architecture
    if architecture == "FourierNeuralOperator":
        model = FourierNeuralOperator(config, **config_params)
    elif architecture == "DeepOperatorNetwork":
        model = DeepOperatorNetwork(config, **config_params)
    elif architecture == "SideChannelFNO":
        model = SideChannelFNO(config, **config_params)
    elif architecture == "LeakageFNO":
        model = LeakageFNO(config, **config_params)
    
    # Test forward pass
    x = torch.randn(2, 100, 1) if architecture != "DeepOperatorNetwork" else torch.randn(2, 50)
    
    try:
        output = model(x)
        assert output.shape[0] == 2  # Batch size
        assert not torch.isnan(output).any()
    except Exception as e:
        pytest.fail(f"Forward pass failed for {architecture}: {e}")


def test_memory_efficiency():
    """Test memory efficiency of neural operators."""
    config = OperatorConfig(
        input_dim=1,
        output_dim=256,
        hidden_dim=64,
        num_layers=4,
        device="cpu"
    )
    
    model = FourierNeuralOperator(config, modes=16)
    
    # Test with large input
    large_input = torch.randn(8, 10000, 1)
    
    # Should not cause memory issues
    try:
        output = model(large_input)
        assert output.shape == (8, 256)
    except RuntimeError as e:
        if "out of memory" in str(e):
            pytest.skip("Insufficient memory for large input test")
        else:
            raise


def test_gradient_flow():
    """Test gradient flow through neural operators."""
    config = OperatorConfig(
        input_dim=1,
        output_dim=256,
        hidden_dim=32,
        num_layers=2,
        device="cpu"
    )
    
    model = FourierNeuralOperator(config, modes=8)
    
    # Create dummy data and target
    x = torch.randn(4, 100, 1, requires_grad=True)
    target = torch.randint(0, 256, (4,))
    
    # Forward pass
    output = model(x)
    loss = torch.nn.functional.cross_entropy(output, target)
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    assert x.grad is not None
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()