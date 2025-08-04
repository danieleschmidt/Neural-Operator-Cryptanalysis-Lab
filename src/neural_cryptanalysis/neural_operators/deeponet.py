"""Deep Operator Network implementation for cryptanalysis."""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
import numpy as np

from .base import NeuralOperatorBase, OperatorConfig


class BranchNet(nn.Module):
    """Branch network for encoding input functions."""
    
    def __init__(self, input_dim: int, branch_layers: List[int], activation: nn.Module = nn.ReLU()):
        super().__init__()
        self.activation = activation
        
        layers = []
        prev_dim = input_dim
        
        for dim in branch_layers:
            layers.extend([
                nn.Linear(prev_dim, dim),
                activation,
                nn.Dropout(0.1)
            ])
            prev_dim = dim
            
        self.network = nn.Sequential(*layers[:-2])  # Remove last dropout
        self.output_dim = branch_layers[-1]
        
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Encode input function u.
        
        Args:
            u: Input function values [batch, sensors]
            
        Returns:
            Branch encoding [batch, branch_dim]
        """
        return self.network(u)


class TrunkNet(nn.Module):
    """Trunk network for encoding evaluation points."""
    
    def __init__(self, coord_dim: int, trunk_layers: List[int], activation: nn.Module = nn.ReLU()):
        super().__init__()
        self.activation = activation
        
        layers = []
        prev_dim = coord_dim
        
        for dim in trunk_layers:
            layers.extend([
                nn.Linear(prev_dim, dim),
                activation,
                nn.Dropout(0.1)
            ])
            prev_dim = dim
            
        self.network = nn.Sequential(*layers[:-2])  # Remove last dropout
        self.output_dim = trunk_layers[-1]
        
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Encode evaluation points y.
        
        Args:
            y: Evaluation coordinates [batch, points, coord_dim]
            
        Returns:
            Trunk encoding [batch, points, trunk_dim]
        """
        batch_size, n_points = y.shape[:2]
        
        # Flatten for processing
        y_flat = y.view(-1, y.shape[-1])
        trunk_out = self.network(y_flat)
        
        # Reshape back
        return trunk_out.view(batch_size, n_points, -1)


class DeepOperatorNetwork(NeuralOperatorBase):
    """Deep Operator Network for learning operators between function spaces.
    
    DeepONet learns mappings from input functions to output functions,
    making it suitable for modeling relationships between side-channel
    traces and cryptographic operations.
    """
    
    def __init__(self, config: OperatorConfig, 
                 branch_layers: List[int] = [128, 128, 128],
                 trunk_layers: List[int] = [128, 128, 128],
                 coord_dim: int = 1,
                 sensor_locations: Optional[torch.Tensor] = None):
        super().__init__(config)
        
        self.branch_layers = branch_layers
        self.trunk_layers = trunk_layers
        self.coord_dim = coord_dim
        
        # Initialize networks
        self.branch_net = BranchNet(config.input_dim, branch_layers, self.activation)
        self.trunk_net = TrunkNet(coord_dim, trunk_layers, self.activation)
        
        # Ensure same output dimension
        assert branch_layers[-1] == trunk_layers[-1], \
            "Branch and trunk networks must have same output dimension"
        
        self.latent_dim = branch_layers[-1]
        
        # Output projection
        self.output_layer = nn.Linear(self.latent_dim, config.output_dim)
        
        # Sensor locations for branch network
        if sensor_locations is not None:
            self.register_buffer('sensor_locations', sensor_locations)
        else:
            # Default uniform sensor locations
            sensors = torch.linspace(0, 1, config.input_dim).unsqueeze(0)
            self.register_buffer('sensor_locations', sensors)
            
        self.initialize_weights()
        
    def forward(self, u: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through DeepONet.
        
        Args:
            u: Input function values at sensor locations [batch, sensors]
            y: Evaluation points [batch, points, coord_dim]. If None, use default points.
            
        Returns:
            Output function values at evaluation points [batch, points, output_dim]
        """
        batch_size = u.shape[0]
        
        # Use default evaluation points if not provided
        if y is None:
            n_points = 100
            y = torch.linspace(0, 1, n_points).unsqueeze(0).unsqueeze(-1)
            y = y.repeat(batch_size, 1, 1).to(u.device)
        
        # Encode input function and evaluation points
        branch_out = self.branch_net(u)  # [batch, latent_dim]
        trunk_out = self.trunk_net(y)    # [batch, points, latent_dim]
        
        # Compute dot product
        branch_expanded = branch_out.unsqueeze(1)  # [batch, 1, latent_dim]
        operator_out = torch.sum(branch_expanded * trunk_out, dim=-1)  # [batch, points]
        
        # Apply output projection
        operator_out = operator_out.unsqueeze(-1)  # [batch, points, 1]
        output = self.output_layer(operator_out.view(-1, 1))
        output = output.view(batch_size, -1, self.config.output_dim)
        
        # Global pooling for classification tasks
        if self.config.output_dim > 1:
            output = torch.mean(output, dim=1)  # [batch, output_dim]
        
        return output
    
    def get_operator_weights(self, u: torch.Tensor) -> torch.Tensor:
        """Get operator weights for input function."""
        return self.branch_net(u)
    
    def evaluate_at_points(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Evaluate operator at specific points."""
        branch_out = self.branch_net(u)
        trunk_out = self.trunk_net(y)
        
        branch_expanded = branch_out.unsqueeze(1)
        return torch.sum(branch_expanded * trunk_out, dim=-1)


class ModifiedDeepONet(DeepOperatorNetwork):
    """Modified DeepONet with additional features for cryptanalysis."""
    
    def __init__(self, config: OperatorConfig, **kwargs):
        super().__init__(config, **kwargs)
        
        # Add attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Add residual connections
        self.branch_residual = nn.Linear(config.input_dim, self.latent_dim)
        self.trunk_residual = nn.Linear(self.coord_dim, self.latent_dim)
        
    def forward(self, u: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with attention and residual connections."""
        batch_size = u.shape[0]
        
        if y is None:
            n_points = 100
            y = torch.linspace(0, 1, n_points).unsqueeze(0).unsqueeze(-1)
            y = y.repeat(batch_size, 1, 1).to(u.device)
        
        # Branch network with residual
        branch_out = self.branch_net(u)
        branch_residual = self.branch_residual(u)
        branch_out = branch_out + branch_residual
        
        # Trunk network with residual
        trunk_out = self.trunk_net(y)
        trunk_residual = self.trunk_residual(y)
        trunk_out = trunk_out + trunk_residual
        
        # Apply attention
        branch_expanded = branch_out.unsqueeze(1).repeat(1, trunk_out.shape[1], 1)
        attended_features, _ = self.attention(
            branch_expanded, trunk_out, trunk_out
        )
        
        # Compute operator output
        operator_out = torch.sum(attended_features * trunk_out, dim=-1)
        
        # Output projection
        operator_out = operator_out.unsqueeze(-1)
        output = self.output_layer(operator_out.view(-1, 1))
        output = output.view(batch_size, -1, self.config.output_dim)
        
        if self.config.output_dim > 1:
            output = torch.mean(output, dim=1)
            
        return output


class HierarchicalDeepONet(DeepOperatorNetwork):
    """Hierarchical DeepONet for multi-scale cryptanalysis."""
    
    def __init__(self, config: OperatorConfig, n_scales: int = 3, **kwargs):
        super().__init__(config, **kwargs)
        self.n_scales = n_scales
        
        # Multi-scale branch networks
        self.branch_networks = nn.ModuleList([
            BranchNet(config.input_dim // (2**i), 
                     [128 // (2**i)] * 3, 
                     self.activation)
            for i in range(n_scales)
        ])
        
        # Feature fusion
        self.fusion_layer = nn.Linear(
            sum([128 // (2**i) for i in range(n_scales)]),
            self.latent_dim
        )
        
    def forward(self, u: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with hierarchical processing."""
        batch_size = u.shape[0]
        
        if y is None:
            n_points = 100
            y = torch.linspace(0, 1, n_points).unsqueeze(0).unsqueeze(-1)
            y = y.repeat(batch_size, 1, 1).to(u.device)
        
        # Multi-scale branch processing
        branch_features = []
        for i, branch_net in enumerate(self.branch_networks):
            # Downsample input
            scale_factor = 2 ** i
            if scale_factor > 1:
                u_scaled = F.avg_pool1d(
                    u.unsqueeze(1), 
                    kernel_size=scale_factor,
                    stride=scale_factor
                ).squeeze(1)
            else:
                u_scaled = u
                
            branch_out = branch_net(u_scaled)
            branch_features.append(branch_out)
        
        # Fuse multi-scale features
        fused_branch = self.fusion_layer(torch.cat(branch_features, dim=-1))
        
        # Trunk network
        trunk_out = self.trunk_net(y)
        
        # Compute operator output
        branch_expanded = fused_branch.unsqueeze(1)
        operator_out = torch.sum(branch_expanded * trunk_out, dim=-1)
        
        # Output projection
        operator_out = operator_out.unsqueeze(-1)
        output = self.output_layer(operator_out.view(-1, 1))
        output = output.view(batch_size, -1, self.config.output_dim)
        
        if self.config.output_dim > 1:
            output = torch.mean(output, dim=1)
            
        return output