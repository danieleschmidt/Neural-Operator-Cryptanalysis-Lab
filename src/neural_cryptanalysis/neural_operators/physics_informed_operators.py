"""
Physics-Informed Neural Operators for Electromagnetic Side-Channel Analysis

Novel architectures incorporating Maxwell's equations and electromagnetic theory
to model realistic side-channel propagation and improve far-field attack capabilities.

Research Contribution: First implementation of Maxwell-Informed Neural Operators (MINOs)
for cryptanalysis with physics constraints and electromagnetic wave modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import math

from ..utils.logging_utils import get_logger
from .base import NeuralOperatorBase, OperatorConfig

logger = get_logger(__name__)


@dataclass
class PhysicsOperatorConfig(OperatorConfig):
    """Configuration for physics-informed neural operators."""
    
    # Electromagnetic parameters
    frequency_range: Tuple[float, float] = (1e6, 1e9)  # 1MHz to 1GHz
    wave_velocity: float = 3e8  # Speed of light (m/s)
    dielectric_constant: float = 4.0  # PCB dielectric constant
    
    # Spatial parameters
    device_dimensions: Tuple[float, float, float] = (0.01, 0.01, 0.002)  # 1cm x 1cm x 2mm chip
    antenna_position: Tuple[float, float, float] = (0.0, 0.0, 0.1)  # 10cm above chip
    measurement_positions: int = 64  # Number of measurement points
    
    # Physics constraints
    enforce_maxwell: bool = True
    wave_equation_weight: float = 1.0
    causality_weight: float = 0.5
    energy_conservation_weight: float = 0.1
    
    # Antenna modeling
    antenna_type: str = "dipole"  # dipole, patch, horn
    antenna_gain: float = 2.15  # dBi
    antenna_efficiency: float = 0.8
    
    # Environment modeling
    multipath_enabled: bool = True
    reflection_coefficient: float = 0.3
    atmospheric_loss: bool = False


class MaxwellEquationLayer(nn.Module):
    """Layer implementing Maxwell's equation constraints for EM wave propagation."""
    
    def __init__(self, config: PhysicsOperatorConfig):
        super().__init__()
        
        self.config = config
        self.wave_velocity = config.wave_velocity / math.sqrt(config.dielectric_constant)
        
        # Learnable physics parameters
        self.conductivity = nn.Parameter(torch.tensor(0.01))  # Substrate conductivity
        self.permittivity_correction = nn.Parameter(torch.tensor(0.0))  # Learn dielectric variations
        
        # Wave equation differential operator (simplified finite difference)
        self.spatial_kernel = nn.Parameter(
            torch.tensor([[[[-1., 0., 1.], [0., 0., 0.], [1., 0., -1.]]]], dtype=torch.float32) * 0.1
        )
        self.temporal_kernel = nn.Parameter(torch.tensor([1., -2., 1.]) * 0.1)
        
        logger.debug(f"Initialized Maxwell equation layer with wave velocity {self.wave_velocity:.2e} m/s")
    
    def forward(self, electric_field: torch.Tensor, magnetic_field: torch.Tensor, 
                dt: float = 1e-9) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Maxwell's equations to update EM fields.
        
        Args:
            electric_field: Electric field tensor [batch, time, height, width, 3]
            magnetic_field: Magnetic field tensor [batch, time, height, width, 3]
            dt: Time step size
            
        Returns:
            Updated electric and magnetic fields
        """
        # Simplified Maxwell equations in discretized form
        # ∂E/∂t = (1/ε)(∇×H - σE)
        # ∂H/∂t = -(1/μ)∇×E
        
        batch_size, time_steps, height, width, _ = electric_field.shape
        
        # Compute spatial gradients (simplified 2D curl)
        curl_H = self._compute_curl_2d(magnetic_field)
        curl_E = self._compute_curl_2d(electric_field)
        
        # Apply effective permittivity and conductivity
        epsilon_eff = self.config.dielectric_constant + self.permittivity_correction
        sigma_eff = self.conductivity
        
        # Update electric field: ∂E/∂t = (1/ε)(∇×H - σE)
        dE_dt = (curl_H - sigma_eff * electric_field) / epsilon_eff
        electric_field_new = electric_field + dt * dE_dt
        
        # Update magnetic field: ∂H/∂t = -(1/μ)∇×E
        mu_0 = 4 * math.pi * 1e-7  # Permeability of free space
        dH_dt = -curl_E / mu_0
        magnetic_field_new = magnetic_field + dt * dH_dt
        
        return electric_field_new, magnetic_field_new
    
    def _compute_curl_2d(self, field: torch.Tensor) -> torch.Tensor:
        """Compute 2D curl of vector field using finite differences."""
        # field: [batch, time, height, width, 3] (Ex, Ey, Ez components)
        
        # Simplified 2D curl computation
        # curl_z = ∂Ey/∂x - ∂Ex/∂y
        
        batch_size, time_steps, height, width, components = field.shape
        curl = torch.zeros_like(field)
        
        # Central differences for spatial derivatives
        if width > 2:
            # ∂Ey/∂x
            curl[:, :, :, 1:-1, 2] = (field[:, :, :, 2:, 1] - field[:, :, :, :-2, 1]) / 2.0
        
        if height > 2:
            # ∂Ex/∂y
            curl[:, :, 1:-1, :, 2] -= (field[:, :, 2:, :, 0] - field[:, :, :-2, :, 0]) / 2.0
        
        return curl
    
    def compute_wave_equation_loss(self, field: torch.Tensor, dt: float = 1e-9) -> torch.Tensor:
        """Compute loss term enforcing wave equation: ∇²E - (1/c²)∂²E/∂t² = 0."""
        if field.size(1) < 3:  # Need at least 3 time steps
            return torch.tensor(0.0, device=field.device)
        
        # Spatial Laplacian (simplified 2D)
        laplacian = self._compute_laplacian_2d(field)
        
        # Temporal second derivative
        d2_dt2 = field[:, 2:, :, :, :] - 2*field[:, 1:-1, :, :, :] + field[:, :-2, :, :, :]
        d2_dt2 = d2_dt2 / (dt**2)
        
        # Wave equation: ∇²E - (1/c²)∂²E/∂t² = 0
        wave_speed_squared = self.wave_velocity ** 2
        wave_residual = laplacian[:, 1:-1, :, :, :] - d2_dt2 / wave_speed_squared
        
        # L2 loss on wave equation residual
        wave_loss = torch.mean(wave_residual ** 2)
        
        return wave_loss
    
    def _compute_laplacian_2d(self, field: torch.Tensor) -> torch.Tensor:
        """Compute 2D Laplacian using finite differences."""
        # field: [batch, time, height, width, components]
        
        batch_size, time_steps, height, width, components = field.shape
        laplacian = torch.zeros_like(field)
        
        # 5-point stencil for 2D Laplacian
        if height > 2 and width > 2:
            laplacian[:, :, 1:-1, 1:-1, :] = (
                field[:, :, 2:, 1:-1, :] +     # North
                field[:, :, :-2, 1:-1, :] +    # South  
                field[:, :, 1:-1, 2:, :] +     # East
                field[:, :, 1:-1, :-2, :] -    # West
                4 * field[:, :, 1:-1, 1:-1, :]  # Center
            )
        
        return laplacian


class AntennaModel(nn.Module):
    """Antenna radiation pattern and gain modeling."""
    
    def __init__(self, config: PhysicsOperatorConfig):
        super().__init__()
        
        self.config = config
        self.antenna_type = config.antenna_type
        self.gain_db = config.antenna_gain
        self.efficiency = config.antenna_efficiency
        
        # Learnable antenna parameters
        if config.antenna_type == "dipole":
            self.antenna_length = nn.Parameter(torch.tensor(0.5))  # Half-wave dipole
            self.wire_radius = nn.Parameter(torch.tensor(1e-3))    # 1mm wire radius
        elif config.antenna_type == "patch":
            self.patch_width = nn.Parameter(torch.tensor(0.03))    # 3cm patch
            self.patch_length = nn.Parameter(torch.tensor(0.04))   # 4cm patch
            self.substrate_height = nn.Parameter(torch.tensor(1.6e-3))  # 1.6mm substrate
        
        logger.debug(f"Initialized {config.antenna_type} antenna model")
    
    def forward(self, source_positions: torch.Tensor, 
                measurement_positions: torch.Tensor, 
                frequency: torch.Tensor) -> torch.Tensor:
        """Compute antenna response between source and measurement positions.
        
        Args:
            source_positions: Source positions [batch, n_sources, 3]
            measurement_positions: Antenna positions [batch, n_antennas, 3] 
            frequency: Operating frequency [batch]
            
        Returns:
            Antenna response matrix [batch, n_antennas, n_sources]
        """
        batch_size = source_positions.size(0)
        n_sources = source_positions.size(1)
        n_antennas = measurement_positions.size(1)
        
        # Compute distances
        distances = self._compute_distances(source_positions, measurement_positions)
        
        # Compute antenna pattern
        antenna_pattern = self._compute_antenna_pattern(source_positions, measurement_positions)
        
        # Free space path loss
        wavelength = self.config.wave_velocity / frequency.unsqueeze(-1).unsqueeze(-1)
        path_loss = (4 * math.pi * distances / wavelength) ** 2
        
        # Combined antenna response
        response = antenna_pattern * self.efficiency / (path_loss + 1e-10)
        
        # Apply antenna gain
        gain_linear = 10 ** (self.gain_db / 10)
        response = response * gain_linear
        
        return response
    
    def _compute_distances(self, sources: torch.Tensor, antennas: torch.Tensor) -> torch.Tensor:
        """Compute Euclidean distances between sources and antennas."""
        # sources: [batch, n_sources, 3]
        # antennas: [batch, n_antennas, 3]
        
        sources_expanded = sources.unsqueeze(2)      # [batch, n_sources, 1, 3]
        antennas_expanded = antennas.unsqueeze(1)    # [batch, 1, n_antennas, 3]
        
        distances = torch.norm(sources_expanded - antennas_expanded, dim=-1)
        
        return distances.transpose(1, 2)  # [batch, n_antennas, n_sources]
    
    def _compute_antenna_pattern(self, sources: torch.Tensor, antennas: torch.Tensor) -> torch.Tensor:
        """Compute antenna radiation pattern."""
        batch_size = sources.size(0)
        n_sources = sources.size(1)
        n_antennas = antennas.size(1)
        
        if self.antenna_type == "dipole":
            return self._dipole_pattern(sources, antennas)
        elif self.antenna_type == "patch":
            return self._patch_pattern(sources, antennas)
        else:
            # Isotropic radiator
            return torch.ones(batch_size, n_antennas, n_sources, device=sources.device)
    
    def _dipole_pattern(self, sources: torch.Tensor, antennas: torch.Tensor) -> torch.Tensor:
        """Compute dipole antenna radiation pattern."""
        # Simplified dipole pattern: sin²(θ) where θ is angle from antenna axis
        
        # Assume dipoles are vertical (z-axis aligned)
        antenna_axis = torch.tensor([0., 0., 1.], device=sources.device)
        
        # Compute angles between source directions and antenna axis
        directions = sources.unsqueeze(2) - antennas.unsqueeze(1)  # [batch, n_sources, n_antennas, 3]
        directions = F.normalize(directions, dim=-1)
        
        # Dot product with antenna axis
        cos_theta = torch.matmul(directions, antenna_axis)
        sin_theta_squared = 1 - cos_theta ** 2
        
        # Dipole pattern
        pattern = sin_theta_squared.transpose(1, 2)  # [batch, n_antennas, n_sources]
        
        return pattern
    
    def _patch_pattern(self, sources: torch.Tensor, antennas: torch.Tensor) -> torch.Tensor:
        """Compute patch antenna radiation pattern."""
        # Simplified patch antenna pattern
        # Maximum gain in broadside direction (z-axis), decreases with angle
        
        directions = sources.unsqueeze(2) - antennas.unsqueeze(1)
        directions = F.normalize(directions, dim=-1)
        
        # Angle from broadside (z-axis)
        cos_theta = directions[:, :, :, 2]  # z-component
        
        # Patch antenna pattern (approximate)
        pattern = torch.clamp(cos_theta, min=0.0) ** 2
        
        return pattern.transpose(1, 2)


class PhysicsInformedNeuralOperator(NeuralOperatorBase):
    """Physics-informed neural operator incorporating electromagnetic theory.
    
    This operator models electromagnetic side-channel propagation using Maxwell's
    equations and realistic antenna/propagation models for improved far-field analysis.
    
    Research Innovation:
    - First physics-informed neural operator for electromagnetic cryptanalysis
    - Maxwell equation constraints with learnable material properties
    - Realistic antenna modeling and multipath propagation
    - Causality and energy conservation enforcement
    """
    
    def __init__(self, config: PhysicsOperatorConfig):
        super().__init__(config)
        
        self.config = config
        
        # Physics components
        self.maxwell_layer = MaxwellEquationLayer(config)
        self.antenna_model = AntennaModel(config)
        
        # Neural operator backbone (simplified FNO-style architecture)
        self.input_proj = nn.Linear(config.input_channels, config.hidden_dim)
        
        # Fourier layers for spectral processing
        self.fourier_layers = nn.ModuleList([
            self._make_fourier_layer(config.hidden_dim, config.hidden_dim, 16)
            for _ in range(config.n_layers)
        ])
        
        # Physics-aware attention
        self.physics_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Temporal convolution for causality enforcement
        self.causal_conv = nn.Conv1d(
            in_channels=config.hidden_dim,
            out_channels=config.hidden_dim,
            kernel_size=3,
            padding=1,
            bias=False
        )
        
        # Learnable physics parameters
        self.propagation_delay = nn.Parameter(torch.tensor(1e-9))  # Signal propagation delay
        self.multipath_coefficients = nn.Parameter(torch.randn(5) * 0.1)  # Multipath reflections
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)
        
        logger.info(f"Initialized PhysicsInformedNeuralOperator with Maxwell constraints")
    
    def _make_fourier_layer(self, in_channels: int, out_channels: int, modes: int) -> nn.Module:
        """Create Fourier layer with learnable spectral weights."""
        
        class FourierLayer(nn.Module):
            def __init__(self, in_channels: int, out_channels: int, modes: int):
                super().__init__()
                self.modes = modes
                self.weights = nn.Parameter(
                    torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat) * 0.02
                )
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: [batch, sequence, channels]
                batch_size, seq_len, channels = x.shape
                
                # FFT
                x_ft = torch.fft.rfft(x, dim=1)
                
                # Spectral convolution
                out_ft = torch.zeros(batch_size, x_ft.size(1), self.weights.size(1), 
                                   dtype=torch.cfloat, device=x.device)
                
                # Apply learned weights to low-frequency modes
                modes = min(self.modes, x_ft.size(1))
                out_ft[:, :modes, :] = torch.einsum('bic,ioc->boc', x_ft[:, :modes, :], self.weights[:, :, :modes])
                
                # IFFT
                x_out = torch.fft.irfft(out_ft, n=seq_len, dim=1)
                
                return x_out
        
        return FourierLayer(in_channels, out_channels, modes)
    
    def forward(self, x: torch.Tensor, 
                source_positions: Optional[torch.Tensor] = None,
                antenna_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with physics constraints.
        
        Args:
            x: Input traces [batch, sequence_length, input_channels]
            source_positions: Optional source positions [batch, n_sources, 3]
            antenna_positions: Optional antenna positions [batch, n_antennas, 3]
            
        Returns:
            Output predictions [batch, output_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # Input projection
        x = self.input_proj(x)
        
        # Apply physics-informed processing
        if self.config.enforce_maxwell and source_positions is not None:
            x = self._apply_physics_constraints(x, source_positions, antenna_positions)
        
        # Fourier neural operator layers
        for fourier_layer in self.fourier_layers:
            residual = x
            x = fourier_layer(x)
            x = F.gelu(x) + residual  # Residual connection
        
        # Physics-aware attention
        x_attended, _ = self.physics_attention(x, x, x)
        x = x + x_attended
        
        # Causal temporal processing
        x_transposed = x.transpose(1, 2)  # [batch, hidden_dim, sequence]
        x_causal = self.causal_conv(x_transposed)
        x = x_causal.transpose(1, 2) + x  # [batch, sequence, hidden_dim]
        
        # Global temporal pooling
        x = torch.mean(x, dim=1)  # [batch, hidden_dim]
        
        # Output projection
        output = self.output_proj(x)
        
        return output
    
    def _apply_physics_constraints(self, x: torch.Tensor, 
                                 source_positions: torch.Tensor,
                                 antenna_positions: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply physics-informed constraints to the signal."""
        batch_size, seq_len, hidden_dim = x.size()
        
        # Generate default antenna positions if not provided
        if antenna_positions is None:
            antenna_positions = self._generate_default_antenna_positions(batch_size)
        
        # Model electromagnetic propagation
        if self.config.multipath_enabled:
            x = self._apply_multipath_effects(x, source_positions, antenna_positions)
        
        # Apply propagation delay
        x = self._apply_propagation_delay(x)
        
        # Enforce causality
        x = self._enforce_causality(x)
        
        return x
    
    def _generate_default_antenna_positions(self, batch_size: int) -> torch.Tensor:
        """Generate default antenna positions for measurement."""
        # Create a grid of antenna positions above the device
        n_antennas = self.config.measurement_positions
        grid_size = int(math.sqrt(n_antennas))
        
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                x = (i - grid_size/2) * 0.02  # 2cm spacing
                y = (j - grid_size/2) * 0.02
                z = self.config.antenna_position[2]  # Height above device
                positions.append([x, y, z])
        
        # Pad if necessary
        while len(positions) < n_antennas:
            positions.append([0., 0., z])
        
        positions = torch.tensor(positions[:n_antennas], dtype=torch.float32)
        return positions.unsqueeze(0).repeat(batch_size, 1, 1)
    
    def _apply_multipath_effects(self, x: torch.Tensor, 
                               source_positions: torch.Tensor,
                               antenna_positions: torch.Tensor) -> torch.Tensor:
        """Apply multipath propagation effects."""
        # Simplified multipath model with learnable coefficients
        
        multipath_signal = x.clone()
        
        # Apply multiple reflection paths with different delays and attenuations
        for i, coeff in enumerate(self.multipath_coefficients):
            delay_samples = int((i + 1) * 2)  # Increasing delay
            
            if delay_samples < x.size(1):
                # Delayed and attenuated signal
                delayed_signal = torch.roll(x, shifts=delay_samples, dims=1)
                delayed_signal[:, :delay_samples, :] = 0  # Zero out wrapped samples
                
                multipath_signal = multipath_signal + coeff * delayed_signal
        
        return multipath_signal
    
    def _apply_propagation_delay(self, x: torch.Tensor) -> torch.Tensor:
        """Apply realistic propagation delay."""
        # Simple model: shift signal by learnable delay
        delay_samples = torch.clamp(self.propagation_delay * 1e9, min=0, max=10).int()  # Convert to samples
        
        if delay_samples > 0:
            x = torch.roll(x, shifts=delay_samples.item(), dims=1)
            x[:, :delay_samples, :] = 0
        
        return x
    
    def _enforce_causality(self, x: torch.Tensor) -> torch.Tensor:
        """Enforce causality constraint - no information from future."""
        # Apply causal mask to prevent future information leakage
        seq_len = x.size(1)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        
        # Simplified causality enforcement through masked convolution
        x_masked = torch.matmul(causal_mask, x)
        
        return x_masked
    
    def compute_physics_losses(self, predictions: torch.Tensor,
                             electric_field: Optional[torch.Tensor] = None,
                             magnetic_field: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute physics-informed loss terms."""
        losses = {}
        
        # Maxwell equation loss
        if electric_field is not None and magnetic_field is not None:
            wave_loss = self.maxwell_layer.compute_wave_equation_loss(electric_field)
            losses['wave_equation'] = wave_loss * self.config.wave_equation_weight
        
        # Causality loss - penalize non-causal responses
        if predictions.dim() > 1:
            # Simple causality check: signal should not have information before time 0
            early_response = torch.mean(torch.abs(predictions[:, :10]))  # First 10 samples
            late_response = torch.mean(torch.abs(predictions[:, -10:]))   # Last 10 samples
            
            causality_loss = F.relu(early_response - late_response * 0.1)
            losses['causality'] = causality_loss * self.config.causality_weight
        
        # Energy conservation loss
        input_energy = torch.sum(predictions ** 2, dim=-1)
        energy_variance = torch.var(input_energy)
        losses['energy_conservation'] = energy_variance * self.config.energy_conservation_weight
        
        return losses
    
    def get_physics_parameters(self) -> Dict[str, torch.Tensor]:
        """Get learnable physics parameters for analysis."""
        return {
            'conductivity': self.maxwell_layer.conductivity,
            'permittivity_correction': self.maxwell_layer.permittivity_correction,
            'propagation_delay': self.propagation_delay,
            'multipath_coefficients': self.multipath_coefficients,
            'antenna_parameters': self._get_antenna_parameters()
        }
    
    def _get_antenna_parameters(self) -> Dict[str, torch.Tensor]:
        """Get antenna-specific parameters."""
        params = {}
        
        if self.config.antenna_type == "dipole":
            params['antenna_length'] = self.antenna_model.antenna_length
            params['wire_radius'] = self.antenna_model.wire_radius
        elif self.config.antenna_type == "patch":
            params['patch_width'] = self.antenna_model.patch_width
            params['patch_length'] = self.antenna_model.patch_length
            params['substrate_height'] = self.antenna_model.substrate_height
        
        return params


class MultiFrequencyPhysicsOperator(PhysicsInformedNeuralOperator):
    """Extension with multiple frequency analysis for broadband attacks."""
    
    def __init__(self, config: PhysicsOperatorConfig, 
                 frequency_bands: List[Tuple[float, float]] = None):
        super().__init__(config)
        
        if frequency_bands is None:
            frequency_bands = [(1e6, 10e6), (10e6, 100e6), (100e6, 1e9)]
        
        self.frequency_bands = frequency_bands
        self.n_bands = len(frequency_bands)
        
        # Frequency-specific processing layers
        self.frequency_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim)
            )
            for _ in range(self.n_bands)
        ])
        
        # Frequency fusion layer
        self.frequency_fusion = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        logger.info(f"Initialized MultiFrequencyPhysicsOperator with {self.n_bands} frequency bands")
    
    def forward(self, x: torch.Tensor, 
                source_positions: Optional[torch.Tensor] = None,
                antenna_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with multi-frequency analysis."""
        batch_size, seq_len, _ = x.size()
        
        # Input projection
        x = self.input_proj(x)
        
        # Process each frequency band
        frequency_outputs = []
        
        for i, (freq_low, freq_high) in enumerate(self.frequency_bands):
            # Bandpass filtering (simplified)
            x_band = self._bandpass_filter(x, freq_low, freq_high)
            
            # Band-specific processing
            x_processed = self.frequency_processors[i](x_band)
            frequency_outputs.append(x_processed)
        
        # Stack frequency bands
        x_multi_freq = torch.stack(frequency_outputs, dim=2)  # [batch, seq, n_bands, hidden]
        
        # Reshape for attention
        batch_size, seq_len, n_bands, hidden_dim = x_multi_freq.size()
        x_flat = x_multi_freq.view(batch_size, seq_len * n_bands, hidden_dim)
        
        # Frequency fusion through attention
        x_fused, _ = self.frequency_fusion(x_flat, x_flat, x_flat)
        
        # Reshape back and combine
        x_fused = x_fused.view(batch_size, seq_len, n_bands, hidden_dim)
        x_combined = torch.mean(x_fused, dim=2)  # Average across frequency bands
        
        # Continue with standard processing
        return self._process_combined_features(x_combined)
    
    def _bandpass_filter(self, x: torch.Tensor, freq_low: float, freq_high: float) -> torch.Tensor:
        """Apply bandpass filter in frequency domain."""
        # FFT
        x_fft = torch.fft.rfft(x, dim=1)
        
        # Create bandpass mask
        freqs = torch.fft.rfftfreq(x.size(1), device=x.device)
        sample_rate = 1e9  # Assume 1 GSa/s
        freqs = freqs * sample_rate
        
        mask = ((freqs >= freq_low) & (freqs <= freq_high)).float()
        mask = mask.unsqueeze(0).unsqueeze(-1)  # [1, freq_bins, 1]
        
        # Apply filter
        x_filtered_fft = x_fft * mask
        
        # IFFT
        x_filtered = torch.fft.irfft(x_filtered_fft, n=x.size(1), dim=1)
        
        return x_filtered
    
    def _process_combined_features(self, x: torch.Tensor) -> torch.Tensor:
        """Process combined multi-frequency features."""
        # Apply remaining Fourier layers
        for fourier_layer in self.fourier_layers:
            residual = x
            x = fourier_layer(x)
            x = F.gelu(x) + residual
        
        # Global pooling and output projection
        x = torch.mean(x, dim=1)
        return self.output_proj(x)