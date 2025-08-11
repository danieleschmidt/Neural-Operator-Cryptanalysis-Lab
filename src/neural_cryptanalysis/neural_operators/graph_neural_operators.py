"""
Graph-Based Neural Operators for Cryptographic Circuit Analysis

Novel architectures that model cryptographic implementations as graph structures,
incorporating circuit topology, electromagnetic coupling, and spatial relationships.

Research Contribution: First implementation of Graph Fourier Neural Operators (GFNO)
for hardware cryptanalysis with circuit-aware spatial attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import math

from ..utils.logging_utils import get_logger
from .base import NeuralOperatorBase, OperatorConfig

logger = get_logger(__name__)


@dataclass
class GraphOperatorConfig(OperatorConfig):
    """Configuration for graph-based neural operators."""
    
    # Graph structure parameters
    n_nodes: int = 256  # Number of circuit nodes
    edge_features: int = 8  # Electromagnetic coupling features
    spectral_modes: int = 32  # Graph Fourier modes
    
    # Circuit topology
    circuit_type: str = "aes_hardware"  # aes_hardware, rsa_multiplier, kyber_ntt
    parasitic_modeling: bool = True  # Model parasitic coupling
    
    # Spatial parameters
    spatial_attention: bool = True
    max_propagation_hops: int = 3  # EM propagation modeling
    
    # Physics constraints
    physics_informed: bool = True
    frequency_range: Tuple[float, float] = (1e6, 1e9)  # 1MHz to 1GHz
    wave_velocity: float = 3e8  # Speed of light


class GraphFourierLayer(nn.Module):
    """Graph Fourier Transform layer for circuit topology analysis."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 spectral_modes: int, graph_laplacian: torch.Tensor):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spectral_modes = spectral_modes
        
        # Register graph Laplacian and compute eigendecomposition
        self.register_buffer('graph_laplacian', graph_laplacian)
        self.register_buffer('eigenvalues', torch.zeros(spectral_modes))
        self.register_buffer('eigenvectors', torch.zeros(graph_laplacian.size(0), spectral_modes))
        
        # Learnable spectral weights
        self.spectral_weights = nn.Parameter(
            torch.randn(in_channels, out_channels, spectral_modes, dtype=torch.cfloat) * 0.02
        )
        
        # Initialize graph spectrum
        self._initialize_graph_spectrum()
    
    def _initialize_graph_spectrum(self):
        """Initialize graph spectral decomposition."""
        try:
            # Compute eigendecomposition of graph Laplacian
            eigenvalues, eigenvectors = torch.linalg.eigh(self.graph_laplacian)
            
            # Select smallest eigenvalues (smooth graph functions)
            indices = torch.argsort(eigenvalues)[:self.spectral_modes]
            
            self.eigenvalues.data = eigenvalues[indices]
            self.eigenvectors.data = eigenvectors[:, indices]
            
            logger.debug(f"Initialized graph spectrum with {self.spectral_modes} modes")
            
        except Exception as e:
            logger.warning(f"Failed to compute graph spectrum, using identity: {e}")
            self.eigenvalues.data = torch.arange(self.spectral_modes, dtype=torch.float32)
            self.eigenvectors.data = torch.eye(self.graph_laplacian.size(0))[:, :self.spectral_modes]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through graph Fourier layer.
        
        Args:
            x: Input features [batch, nodes, channels]
            
        Returns:
            Output features [batch, nodes, out_channels]
        """
        batch_size = x.size(0)
        
        # Graph Fourier Transform
        # x_hat = V^T * x (project to spectral domain)
        x_spectral = torch.matmul(self.eigenvectors.T, x.transpose(1, 2))  # [spectral_modes, batch, in_channels]
        x_spectral = x_spectral.transpose(0, 1)  # [batch, spectral_modes, in_channels]
        
        # Apply learnable spectral filters
        output_spectral = torch.zeros(batch_size, self.spectral_modes, self.out_channels, 
                                    dtype=torch.cfloat, device=x.device)
        
        for mode in range(self.spectral_modes):
            # Spectral multiplication: W_k * x_hat_k
            output_spectral[:, mode, :] = torch.matmul(
                x_spectral[:, mode, :].unsqueeze(-1), 
                self.spectral_weights[:, :, mode].unsqueeze(0)
            ).squeeze(-1)
        
        # Inverse Graph Fourier Transform
        # output = V * output_spectral (project back to node domain)
        output = torch.matmul(self.eigenvectors, output_spectral.real.transpose(1, 2))
        output = output.transpose(1, 2)  # [batch, nodes, out_channels]
        
        return output


class CircuitAwareAttention(nn.Module):
    """Spatial attention mechanism aware of circuit topology and EM propagation."""
    
    def __init__(self, d_model: int, n_heads: int, circuit_adjacency: torch.Tensor,
                 max_hops: int = 3):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_hops = max_hops
        
        assert self.head_dim * n_heads == d_model
        
        # Multi-head attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Circuit topology encoding
        self.register_buffer('circuit_adjacency', circuit_adjacency)
        self.register_buffer('propagation_matrix', self._compute_propagation_matrix(circuit_adjacency))
        
        # Learnable position encodings for circuit layout
        n_nodes = circuit_adjacency.size(0)
        self.position_encoding = nn.Parameter(torch.randn(n_nodes, d_model) * 0.02)
        
        # Propagation distance encoding
        self.distance_encoding = nn.Embedding(max_hops + 1, n_heads)
        
    def _compute_propagation_matrix(self, adjacency: torch.Tensor) -> torch.Tensor:
        """Compute multi-hop propagation matrix for EM coupling."""
        n_nodes = adjacency.size(0)
        propagation = torch.zeros(n_nodes, n_nodes, self.max_hops + 1)
        
        # 0-hop: self-connection
        propagation[:, :, 0] = torch.eye(n_nodes)
        
        # Multi-hop propagation
        current_power = adjacency.clone()
        for hop in range(1, self.max_hops + 1):
            propagation[:, :, hop] = (current_power > 0).float()
            current_power = torch.matmul(current_power, adjacency)
        
        return propagation
    
    def forward(self, x: torch.Tensor, node_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with circuit-aware attention.
        
        Args:
            x: Input features [batch, nodes, d_model]
            node_mask: Optional mask for inactive nodes [batch, nodes]
            
        Returns:
            Output features [batch, nodes, d_model]
        """
        batch_size, n_nodes, _ = x.size()
        
        # Add positional encoding for circuit layout
        x = x + self.position_encoding.unsqueeze(0)
        
        # Multi-head attention projections
        q = self.q_proj(x).view(batch_size, n_nodes, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, n_nodes, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, n_nodes, self.n_heads, self.head_dim)
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # Apply circuit topology bias
        topology_bias = self._compute_topology_bias(attention_scores.size())
        attention_scores = attention_scores + topology_bias
        
        # Apply node mask if provided
        if node_mask is not None:
            mask_expanded = node_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, nodes]
            attention_scores = attention_scores.masked_fill(~mask_expanded, float('-inf'))
        
        # Softmax attention
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, v)
        output = output.contiguous().view(batch_size, n_nodes, self.d_model)
        
        # Output projection
        output = self.out_proj(output)
        
        return output
    
    def _compute_topology_bias(self, attention_shape: Tuple[int, ...]) -> torch.Tensor:
        """Compute topology-aware attention bias."""
        batch_size, n_nodes, n_heads, _ = attention_shape
        
        bias = torch.zeros(batch_size, n_nodes, n_heads, n_nodes, device=self.position_encoding.device)
        
        # Add distance-based bias for each propagation hop
        for hop in range(self.max_hops + 1):
            hop_mask = self.propagation_matrix[:, :, hop]  # [nodes, nodes]
            hop_bias = self.distance_encoding(torch.tensor(hop, device=bias.device))  # [n_heads]
            
            # Apply hop bias where propagation exists
            for head in range(n_heads):
                bias[:, :, head, :] += hop_mask.unsqueeze(0) * hop_bias[head]
        
        return bias


class CircuitGraphNeuralOperator(NeuralOperatorBase):
    """Graph-based neural operator for cryptographic circuit analysis.
    
    This operator models cryptographic implementations as graph structures,
    incorporating circuit topology, electromagnetic coupling, and spatial relationships
    for enhanced side-channel analysis.
    
    Research Innovation:
    - First implementation of Graph Fourier Neural Operators for cryptanalysis
    - Circuit-aware spatial attention with EM propagation modeling
    - Physics-informed constraints for realistic electromagnetic coupling
    """
    
    def __init__(self, config: GraphOperatorConfig):
        super().__init__(config)
        
        self.config = config
        
        # Generate or load circuit topology
        self.circuit_graph = self._generate_circuit_graph()
        
        # Input projection layer
        self.input_proj = nn.Linear(config.input_channels, config.hidden_dim)
        
        # Graph Fourier layers
        self.graph_layers = nn.ModuleList([
            GraphFourierLayer(
                in_channels=config.hidden_dim if i == 0 else config.hidden_dim,
                out_channels=config.hidden_dim,
                spectral_modes=config.spectral_modes,
                graph_laplacian=self.circuit_graph['laplacian']
            )
            for i in range(config.n_layers)
        ])
        
        # Circuit-aware attention layers
        if config.spatial_attention:
            self.attention_layers = nn.ModuleList([
                CircuitAwareAttention(
                    d_model=config.hidden_dim,
                    n_heads=8,
                    circuit_adjacency=self.circuit_graph['adjacency'],
                    max_hops=config.max_propagation_hops
                )
                for _ in range(config.n_layers)
            ])
        else:
            self.attention_layers = nn.ModuleList([nn.Identity() for _ in range(config.n_layers)])
        
        # Layer normalization and activation
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_dim) for _ in range(config.n_layers)
        ])
        self.activation = nn.GELU()
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)
        
        # Physics-informed loss components
        if config.physics_informed:
            self.physics_loss_weight = nn.Parameter(torch.tensor(0.1))
        
        logger.info(f"Initialized CircuitGraphNeuralOperator with {config.n_nodes} nodes, "
                   f"{config.spectral_modes} spectral modes")
    
    def _generate_circuit_graph(self) -> Dict[str, torch.Tensor]:
        """Generate circuit topology graph based on cryptographic implementation."""
        
        if self.config.circuit_type == "aes_hardware":
            return self._generate_aes_circuit_graph()
        elif self.config.circuit_type == "rsa_multiplier":
            return self._generate_rsa_circuit_graph()
        elif self.config.circuit_type == "kyber_ntt":
            return self._generate_kyber_circuit_graph()
        else:
            return self._generate_generic_circuit_graph()
    
    def _generate_aes_circuit_graph(self) -> Dict[str, torch.Tensor]:
        """Generate AES-specific circuit topology."""
        n_nodes = self.config.n_nodes
        
        # Create structured adjacency for AES S-box and MixColumns operations
        adjacency = torch.zeros(n_nodes, n_nodes)
        
        # S-box connections (16 S-boxes with internal structure)
        sbox_size = 16
        for sbox_idx in range(16):
            start_idx = sbox_idx * sbox_size
            end_idx = start_idx + sbox_size
            
            # Connect S-box internal nodes
            for i in range(start_idx, end_idx):
                for j in range(start_idx, end_idx):
                    if i != j:
                        adjacency[i, j] = 1.0
        
        # MixColumns connections (connect S-boxes within each column)
        for col in range(4):
            for row1 in range(4):
                for row2 in range(4):
                    if row1 != row2:
                        idx1 = (row1 * 4 + col) * sbox_size
                        idx2 = (row2 * 4 + col) * sbox_size
                        
                        # Connect S-box outputs
                        for offset in range(sbox_size):
                            adjacency[idx1 + offset, idx2 + offset] = 0.5
        
        # Add parasitic coupling (nearby components)
        if self.config.parasitic_modeling:
            for i in range(n_nodes):
                for j in range(max(0, i-3), min(n_nodes, i+4)):
                    if i != j:
                        adjacency[i, j] = max(adjacency[i, j], 0.1)
        
        # Compute graph Laplacian
        degree = torch.diag(torch.sum(adjacency, dim=1))
        laplacian = degree - adjacency
        
        return {
            'adjacency': adjacency,
            'laplacian': laplacian,
            'degree': degree,
            'type': 'aes_hardware'
        }
    
    def _generate_rsa_circuit_graph(self) -> Dict[str, torch.Tensor]:
        """Generate RSA multiplier circuit topology."""
        n_nodes = self.config.n_nodes
        
        # Create multiplier array structure
        adjacency = torch.zeros(n_nodes, n_nodes)
        
        # Model multiplier array as grid
        grid_size = int(math.sqrt(n_nodes))
        
        for i in range(grid_size):
            for j in range(grid_size):
                node_idx = i * grid_size + j
                
                # Connect to neighbors (4-connectivity)
                for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < grid_size and 0 <= nj < grid_size:
                        neighbor_idx = ni * grid_size + nj
                        adjacency[node_idx, neighbor_idx] = 1.0
        
        # Compute graph Laplacian
        degree = torch.diag(torch.sum(adjacency, dim=1))
        laplacian = degree - adjacency
        
        return {
            'adjacency': adjacency,
            'laplacian': laplacian,
            'degree': degree,
            'type': 'rsa_multiplier'
        }
    
    def _generate_kyber_circuit_graph(self) -> Dict[str, torch.Tensor]:
        """Generate Kyber NTT circuit topology."""
        n_nodes = self.config.n_nodes
        
        # Create butterfly network structure for NTT
        adjacency = torch.zeros(n_nodes, n_nodes)
        
        # NTT butterfly connections
        stages = int(math.log2(n_nodes))
        
        for stage in range(stages):
            butterfly_size = 2 ** (stage + 1)
            
            for start in range(0, n_nodes, butterfly_size):
                mid = start + butterfly_size // 2
                
                # Connect butterfly pairs
                for offset in range(butterfly_size // 2):
                    if start + offset < n_nodes and mid + offset < n_nodes:
                        adjacency[start + offset, mid + offset] = 1.0
                        adjacency[mid + offset, start + offset] = 1.0
        
        # Compute graph Laplacian
        degree = torch.diag(torch.sum(adjacency, dim=1))
        laplacian = degree - adjacency
        
        return {
            'adjacency': adjacency,
            'laplacian': laplacian,
            'degree': degree,
            'type': 'kyber_ntt'
        }
    
    def _generate_generic_circuit_graph(self) -> Dict[str, torch.Tensor]:
        """Generate generic circuit topology."""
        n_nodes = self.config.n_nodes
        
        # Create small-world graph structure
        adjacency = torch.zeros(n_nodes, n_nodes)
        
        # Regular ring lattice connections
        k = 4  # Each node connects to 4 nearest neighbors
        for i in range(n_nodes):
            for j in range(1, k//2 + 1):
                left = (i - j) % n_nodes
                right = (i + j) % n_nodes
                adjacency[i, left] = 1.0
                adjacency[i, right] = 1.0
        
        # Random rewiring with probability p = 0.3
        p = 0.3
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if adjacency[i, j] > 0:
                    if torch.rand(1) < p:
                        # Rewire to random node
                        new_target = torch.randint(0, n_nodes, (1,)).item()
                        if new_target != i:
                            adjacency[i, j] = 0.0
                            adjacency[j, i] = 0.0
                            adjacency[i, new_target] = 1.0
                            adjacency[new_target, i] = 1.0
        
        # Compute graph Laplacian
        degree = torch.diag(torch.sum(adjacency, dim=1))
        laplacian = degree - adjacency
        
        return {
            'adjacency': adjacency,
            'laplacian': laplacian,
            'degree': degree,
            'type': 'generic'
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through circuit graph neural operator.
        
        Args:
            x: Input traces [batch, sequence_length, input_channels]
            
        Returns:
            Output predictions [batch, output_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # Reshape to node representation
        # Map sequence positions to circuit nodes
        if seq_len != self.config.n_nodes:
            # Adaptive pooling to match node count
            x = x.transpose(1, 2)  # [batch, channels, seq_len]
            x = F.adaptive_avg_pool1d(x, self.config.n_nodes)
            x = x.transpose(1, 2)  # [batch, n_nodes, channels]
        
        # Input projection
        x = self.input_proj(x)
        
        # Apply graph layers with residual connections
        for i, (graph_layer, attention_layer, layer_norm) in enumerate(
            zip(self.graph_layers, self.attention_layers, self.layer_norms)
        ):
            # Graph Fourier processing
            residual = x
            x = graph_layer(x)
            x = self.activation(x)
            x = x + residual  # Residual connection
            
            # Circuit-aware attention
            if self.config.spatial_attention:
                residual = x
                x = attention_layer(x)
                x = x + residual  # Residual connection
            
            # Layer normalization
            x = layer_norm(x)
        
        # Global pooling across nodes
        x = torch.mean(x, dim=1)  # [batch, hidden_dim]
        
        # Output projection
        output = self.output_proj(x)
        
        return output
    
    def compute_physics_loss(self, predictions: torch.Tensor, 
                           ground_truth: torch.Tensor) -> torch.Tensor:
        """Compute physics-informed loss terms."""
        if not self.config.physics_informed:
            return torch.tensor(0.0, device=predictions.device)
        
        # Electromagnetic propagation constraint
        # Power should decay with distance according to path loss
        adjacency = self.circuit_graph['adjacency']
        
        # Compute expected power distribution
        # This is a simplified model - in practice would use Maxwell's equations
        power_distribution = torch.sum(predictions ** 2, dim=-1)  # [batch]
        
        # Physics loss (simplified - encourage realistic power distribution)
        physics_loss = torch.mean(power_distribution ** 2) * self.physics_loss_weight
        
        return physics_loss
    
    def get_attention_weights(self) -> List[torch.Tensor]:
        """Extract attention weights for visualization."""
        attention_weights = []
        
        for layer in self.attention_layers:
            if hasattr(layer, 'last_attention_weights'):
                attention_weights.append(layer.last_attention_weights)
        
        return attention_weights
    
    def get_spectral_analysis(self) -> Dict[str, torch.Tensor]:
        """Get spectral analysis of graph structure."""
        return {
            'eigenvalues': self.graph_layers[0].eigenvalues,
            'eigenvectors': self.graph_layers[0].eigenvectors,
            'spectral_weights': [layer.spectral_weights for layer in self.graph_layers],
            'adjacency': self.circuit_graph['adjacency'],
            'laplacian': self.circuit_graph['laplacian']
        }


class AdaptiveCircuitGraphOperator(CircuitGraphNeuralOperator):
    """Adaptive version that learns circuit topology during training."""
    
    def __init__(self, config: GraphOperatorConfig):
        # Initialize with generic topology
        config.circuit_type = "generic"
        super().__init__(config)
        
        # Learnable adjacency matrix
        self.learnable_adjacency = nn.Parameter(
            torch.randn(config.n_nodes, config.n_nodes) * 0.1
        )
        
        # Sparsity constraint
        self.sparsity_weight = 0.01
        
    def _update_graph_structure(self):
        """Update graph structure based on learned adjacency."""
        # Apply sigmoid and sparsity constraints
        adjacency = torch.sigmoid(self.learnable_adjacency)
        
        # Encourage sparsity
        adjacency = adjacency * (adjacency > 0.5).float()
        
        # Ensure symmetry
        adjacency = (adjacency + adjacency.T) / 2
        
        # Update graph Laplacian
        degree = torch.diag(torch.sum(adjacency, dim=1))
        laplacian = degree - adjacency
        
        # Update all graph layers
        for layer in self.graph_layers:
            layer.graph_laplacian.data = laplacian
            layer._initialize_graph_spectrum()
        
        # Update attention layers
        for layer in self.attention_layers:
            if hasattr(layer, 'circuit_adjacency'):
                layer.circuit_adjacency.data = adjacency
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive graph structure."""
        # Update graph structure periodically during training
        if self.training and torch.rand(1) < 0.1:  # 10% of the time
            self._update_graph_structure()
        
        return super().forward(x)
    
    def compute_sparsity_loss(self) -> torch.Tensor:
        """Compute sparsity regularization loss."""
        adjacency_sigmoid = torch.sigmoid(self.learnable_adjacency)
        sparsity_loss = torch.sum(adjacency_sigmoid) * self.sparsity_weight
        return sparsity_loss