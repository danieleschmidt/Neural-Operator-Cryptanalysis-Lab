"""Custom neural operator architectures for side-channel analysis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .base import NeuralOperatorBase, OperatorConfig
from .fno import FourierLayer, SpectralConv1d


class SideChannelFNO(NeuralOperatorBase):
    """Specialized FNO for side-channel trace analysis.
    
    Optimized for processing power/EM traces with built-in preprocessing
    and cryptographic operation detection capabilities.
    """
    
    def __init__(self, config: OperatorConfig, modes: int = 16, 
                 trace_length: int = 10000, preprocessing: str = 'standardize'):
        super().__init__(config)
        self.modes = modes
        self.trace_length = trace_length
        self.preprocessing = preprocessing
        
        # Preprocessing layers
        self.preprocess_layer = self._build_preprocessing(preprocessing)
        
        # Fourier layers with specialized initialization
        self.fc0 = nn.Linear(config.input_dim, config.hidden_dim)
        
        self.fourier_layers = nn.ModuleList([
            FourierLayer(modes, config.hidden_dim, self.activation)
            for _ in range(config.num_layers)
        ])
        
        # Attention mechanism for point-of-interest detection
        self.poi_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, 256),
            self.activation,
            nn.Dropout(config.dropout),
            nn.Linear(256, config.output_dim)
        )
        
        self.initialize_weights()
        
    def _build_preprocessing(self, method: str) -> nn.Module:
        """Build preprocessing layer."""
        if method == 'standardize':
            return StandardizationLayer()
        elif method == 'normalize':
            return NormalizationLayer()
        elif method == 'bandpass':
            return BandpassFilter()
        else:
            return nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SideChannelFNO.
        
        Args:
            x: Input traces [batch, length] or [batch, length, channels]
            
        Returns:
            Classification logits [batch, num_classes]
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
            
        # Preprocessing
        x = self.preprocess_layer(x)
        
        # Lift to higher dimension
        x = self.fc0(x)  # [batch, length, hidden_dim]
        x = x.permute(0, 2, 1)  # [batch, hidden_dim, length]
        
        # Apply Fourier layers
        for layer in self.fourier_layers:
            residual = x if self.config.use_residual else 0
            x = layer(x) + residual
        
        # Convert back for attention
        x = x.permute(0, 2, 1)  # [batch, length, hidden_dim]
        
        # Point-of-interest attention
        attended_x, attention_weights = self.poi_attention(x, x, x)
        
        # Global pooling with attention weights
        pooled = torch.sum(attended_x * attention_weights.mean(dim=1, keepdim=True), dim=1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights for visualization."""
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
            
        x = self.preprocess_layer(x)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        
        for layer in self.fourier_layers:
            x = layer(x)
            
        x = x.permute(0, 2, 1)
        _, attention_weights = self.poi_attention(x, x, x)
        
        return attention_weights.mean(dim=1)  # Average over heads


class LeakageFNO(NeuralOperatorBase):
    """FNO specialized for modeling leakage patterns in crypto operations."""
    
    def __init__(self, config: OperatorConfig, operation_type: str = 'aes_sbox'):
        super().__init__(config)
        self.operation_type = operation_type
        
        # Operation-specific parameters
        self.op_params = self._get_operation_params(operation_type)
        
        # Multi-scale spectral processing
        self.scales = [8, 16, 32]
        self.spectral_branches = nn.ModuleList([
            SpectralBranch(config.input_dim, config.hidden_dim, modes)
            for modes in self.scales
        ])
        
        # Feature fusion
        self.fusion = nn.Linear(
            len(self.scales) * config.hidden_dim,
            config.hidden_dim
        )
        
        # Leakage model specific layers
        self.leakage_decoder = LeakageDecoder(
            config.hidden_dim,
            self.op_params['intermediate_values'],
            config.output_dim
        )
        
    def _get_operation_params(self, operation: str) -> Dict:
        """Get parameters for specific cryptographic operations."""
        params = {
            'aes_sbox': {
                'intermediate_values': 256,
                'hamming_weight': True,
                'timing_variance': 'low'
            },
            'kyber_ntt': {
                'intermediate_values': 3329,  # q modulus
                'hamming_weight': False,
                'timing_variance': 'high'
            },
            'rsa_modexp': {
                'intermediate_values': 65536,
                'hamming_weight': True,
                'timing_variance': 'very_high'
            }
        }
        return params.get(operation, params['aes_sbox'])
    
    def forward(self, x: torch.Tensor, 
                intermediate_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass modeling leakage for specific operations."""
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        
        # Multi-scale spectral processing
        branch_outputs = []
        for branch in self.spectral_branches:
            branch_out = branch(x)
            branch_outputs.append(branch_out)
        
        # Fuse multi-scale features
        fused = torch.cat(branch_outputs, dim=-1)
        fused = self.fusion(fused)
        
        # Decode leakage patterns
        leakage_pred = self.leakage_decoder(fused, intermediate_values)
        
        return leakage_pred


class MultiModalOperator(NeuralOperatorBase):
    """Multi-modal neural operator for fusing different side-channel sources."""
    
    def __init__(self, config: OperatorConfig, 
                 modalities: Dict[str, int],
                 fusion_strategy: str = 'attention'):
        super().__init__(config)
        self.modalities = modalities
        self.fusion_strategy = fusion_strategy
        
        # Modality-specific encoders
        self.encoders = nn.ModuleDict()
        for modality, input_dim in modalities.items():
            self.encoders[modality] = ModalityEncoder(
                input_dim, config.hidden_dim, modality
            )
        
        # Fusion mechanism
        if fusion_strategy == 'attention':
            self.fusion = AttentionFusion(config.hidden_dim, len(modalities))
        elif fusion_strategy == 'concat':
            self.fusion = ConcatenationFusion(config.hidden_dim, len(modalities))
        else:
            self.fusion = AverageFusion()
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, 256),
            self.activation,
            nn.Dropout(config.dropout),
            nn.Linear(256, config.output_dim)
        )
        
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with multi-modal fusion.
        
        Args:
            inputs: Dictionary of modality tensors
            
        Returns:
            Fused classification output
        """
        # Encode each modality
        encoded_modalities = {}
        for modality, data in inputs.items():
            if modality in self.encoders:
                encoded_modalities[modality] = self.encoders[modality](data)
        
        # Fuse modalities
        fused_features = self.fusion(encoded_modalities)
        
        # Classification
        output = self.classifier(fused_features)
        
        return output


# Helper classes for custom architectures

class StandardizationLayer(nn.Module):
    """Learnable standardization layer."""
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        return (x - mean) / (std + self.eps)


class NormalizationLayer(nn.Module):
    """Min-max normalization layer."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_min = x.min(dim=1, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0]
        return (x - x_min) / (x_max - x_min + 1e-8)


class BandpassFilter(nn.Module):
    """Learnable bandpass filter using 1D convolutions."""
    
    def __init__(self, kernel_size: int = 15):
        super().__init__()
        self.filter = nn.Conv1d(1, 1, kernel_size, padding='same')
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x.squeeze(-1)
        x = x.unsqueeze(1)  # Add channel dim
        filtered = self.filter(x)
        return filtered.squeeze(1).unsqueeze(-1)


class SpectralBranch(nn.Module):
    """Single spectral processing branch."""
    
    def __init__(self, input_dim: int, hidden_dim: int, modes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.spectral_conv = SpectralConv1d(hidden_dim, hidden_dim, modes)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)  # [batch, length, hidden_dim]
        x = x.permute(0, 2, 1)  # [batch, hidden_dim, length]
        x = self.spectral_conv(x)
        x = self.activation(x)
        x = x.permute(0, 2, 1)  # [batch, length, hidden_dim]
        return torch.mean(x, dim=1)  # Global average pooling


class LeakageDecoder(nn.Module):
    """Decoder for leakage pattern prediction."""
    
    def __init__(self, hidden_dim: int, n_intermediate: int, output_dim: int):
        super().__init__()
        self.n_intermediate = n_intermediate
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, features: torch.Tensor, 
                intermediate_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode leakage patterns."""
        # If intermediate values provided, use them for supervised learning
        if intermediate_values is not None:
            # Combine features with intermediate value embeddings
            iv_embed = F.one_hot(intermediate_values, self.n_intermediate).float()
            combined = torch.cat([features, iv_embed], dim=-1)
            return self.decoder(combined)
        else:
            return self.decoder(features)


class ModalityEncoder(nn.Module):
    """Encoder for specific modality."""
    
    def __init__(self, input_dim: int, hidden_dim: int, modality_type: str):
        super().__init__()
        self.modality_type = modality_type
        
        if modality_type == 'power':
            # Power traces - focus on amplitude
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        elif modality_type == 'em':
            # EM traces - focus on frequency content
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            # Generic encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2:
            x = x.mean(dim=1)  # Global pooling for sequence data
        elif len(x.shape) == 3:
            x = x.mean(dim=1)  # Global pooling
        return self.encoder(x)


class AttentionFusion(nn.Module):
    """Attention-based fusion of multiple modalities."""
    
    def __init__(self, hidden_dim: int, n_modalities: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
    def forward(self, modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Stack modalities
        stacked = torch.stack(list(modalities.values()), dim=1)
        
        # Apply self-attention
        attended, _ = self.attention(stacked, stacked, stacked)
        
        # Global pooling
        return attended.mean(dim=1)


class ConcatenationFusion(nn.Module):
    """Simple concatenation fusion."""
    
    def __init__(self, hidden_dim: int, n_modalities: int):
        super().__init__()
        self.projection = nn.Linear(hidden_dim * n_modalities, hidden_dim)
        
    def forward(self, modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
        concatenated = torch.cat(list(modalities.values()), dim=-1)
        return self.projection(concatenated)


class AverageFusion(nn.Module):
    """Simple average fusion."""
    
    def forward(self, modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack(list(modalities.values()), dim=0).mean(dim=0)