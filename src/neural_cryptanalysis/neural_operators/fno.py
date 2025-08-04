"""Fourier Neural Operator implementation for side-channel analysis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

from .base import NeuralOperatorBase, OperatorConfig


class SpectralConv1d(nn.Module):
    """1D Spectral Convolution layer for FNO."""
    
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # Initialize Fourier weights
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )
        
    def compl_mul1d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex multiplication for 1D spectral convolution."""
        return torch.einsum("bix,iox->box", input, weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through spectral convolution.
        
        Args:
            x: Input tensor [batch, channels, length]
            
        Returns:
            Output tensor [batch, channels, length]
        """
        batch_size = x.shape[0]
        
        # Apply FFT
        x_ft = torch.fft.rfft(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batch_size, self.out_channels, x.size(-1)//2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        out_ft[:, :, :self.modes] = self.compl_mul1d(
            x_ft[:, :, :self.modes], self.weights1
        )
        
        # Apply inverse FFT
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        
        return x


class SpectralConv2d(nn.Module):
    """2D Spectral Convolution layer for FNO."""
    
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        
    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex multiplication for 2D spectral convolution."""
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through 2D spectral convolution."""
        batch_size = x.shape[0]
        
        # Apply 2D FFT
        x_ft = torch.fft.rfft2(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batch_size, self.out_channels, x.size(-2), x.size(-1)//2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )
        
        # Apply inverse 2D FFT
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        
        return x


class FourierLayer(nn.Module):
    """Single Fourier layer combining spectral and local operations."""
    
    def __init__(self, modes: int, width: int, activation: nn.Module = nn.GELU()):
        super().__init__()
        self.modes = modes
        self.width = width
        self.activation = activation
        
        self.conv = SpectralConv1d(width, width, modes)
        self.w = nn.Conv1d(width, width, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining spectral and local operations."""
        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        x = self.activation(x)
        return x


class FourierNeuralOperator(NeuralOperatorBase):
    """Fourier Neural Operator for side-channel analysis.
    
    Implements FNO architecture optimized for processing side-channel traces
    with efficient spectral convolutions in the Fourier domain.
    """
    
    def __init__(self, config: OperatorConfig, modes: int = 16):
        super().__init__(config)
        self.modes = modes
        
        # Lift to higher dimension
        self.fc0 = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Fourier layers
        self.layers = nn.ModuleList([
            FourierLayer(modes, config.hidden_dim, self.activation)
            for _ in range(config.num_layers)
        ])
        
        # Project to output
        self.fc1 = nn.Linear(config.hidden_dim, 128)
        self.fc2 = nn.Linear(128, config.output_dim)
        
        # Normalization layers
        if config.normalization != 'none':
            self.norms = nn.ModuleList([
                self._get_normalization(config.hidden_dim)
                for _ in range(config.num_layers)
            ])
        else:
            self.norms = None
            
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        self.initialize_weights()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FNO.
        
        Args:
            x: Input tensor [batch, length, channels] or [batch, length]
            
        Returns:
            Output tensor [batch, output_dim]
        """
        # Handle different input shapes
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)  # Add channel dimension
            
        # Lift to higher dimension
        x = self.fc0(x)  # [batch, length, hidden_dim]
        x = x.permute(0, 2, 1)  # [batch, hidden_dim, length]
        
        # Apply Fourier layers
        for i, layer in enumerate(self.layers):
            residual = x if self.config.use_residual else 0
            x = layer(x)
            
            if self.norms is not None:
                x = x.permute(0, 2, 1)  # For normalization
                x = self.norms[i](x)
                x = x.permute(0, 2, 1)
                
            x = x + residual
            x = self.dropout(x)
        
        # Global pooling and projection
        x = x.permute(0, 2, 1)  # [batch, length, hidden_dim]
        x = torch.mean(x, dim=1)  # Global average pooling
        
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_spectral_features(self, x: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        """Extract spectral features from a specific layer."""
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
            
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        
        # Forward to specific layer
        for i in range(min(layer_idx + 1, len(self.layers))):
            if i == layer_idx:
                # Extract spectral features before activation
                x_ft = torch.fft.rfft(x)
                spectral_power = torch.abs(x_ft)
                return spectral_power
            x = self.layers[i](x)
            
        return None


class AdaptiveFNO(FourierNeuralOperator):
    """Adaptive FNO that adjusts modes based on input characteristics."""
    
    def __init__(self, config: OperatorConfig, max_modes: int = 32, min_modes: int = 4):
        super().__init__(config, modes=max_modes)
        self.max_modes = max_modes
        self.min_modes = min_modes
        self.current_modes = max_modes
        
        # Mode adaptation network
        self.mode_predictor = nn.Sequential(
            nn.Linear(config.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def adapt_modes(self, x: torch.Tensor) -> int:
        """Predict optimal number of modes for input."""
        # Compute signal characteristics
        signal_energy = torch.mean(x ** 2, dim=1, keepdim=True)
        mode_ratio = self.mode_predictor(signal_energy)
        
        # Scale to mode range
        adapted_modes = int(
            self.min_modes + mode_ratio.mean().item() * (self.max_modes - self.min_modes)
        )
        
        return max(self.min_modes, min(adapted_modes, self.max_modes))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive mode selection."""
        # Adapt modes based on input
        optimal_modes = self.adapt_modes(x)
        
        if optimal_modes != self.current_modes:
            self.current_modes = optimal_modes
            # Update spectral convolution layers
            for layer in self.layers:
                layer.conv.modes = optimal_modes
                
        return super().forward(x)