"""
Transformer-Based Neural Operators for Cryptographic Sequence Analysis

Novel architectures leveraging self-attention mechanisms to model temporal
dependencies in cryptographic operations and side-channel traces.

Research Contribution: First implementation of CryptoTransformers specialized
for sequential cryptographic operations with hierarchical attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from ..utils.logging_utils import get_logger
from .base import NeuralOperatorBase, OperatorConfig

logger = get_logger(__name__)


@dataclass
class TransformerOperatorConfig(OperatorConfig):
    """Configuration for transformer-based neural operators."""
    
    # Transformer parameters
    n_heads: int = 8
    n_transformer_layers: int = 6
    dropout: float = 0.1
    
    # Cryptographic structure modeling
    crypto_operation_types: List[str] = None  # ["sbox", "mixcol", "ntt", "modmul"]
    max_sequence_length: int = 4096
    operation_embedding_dim: int = 64
    
    # Hierarchical attention
    hierarchical_levels: List[int] = None  # [1, 4, 16, 64] - temporal scales
    cross_scale_attention: bool = True
    
    # Positional encoding
    positional_encoding_type: str = "learned"  # "learned", "sinusoidal", "rotary"
    max_position: int = 8192
    
    # Temporal modeling
    causal_attention: bool = False  # True for autoregressive modeling
    temporal_window_size: int = 256  # Local attention window
    global_attention_rate: int = 4   # Every 4th layer uses global attention
    
    def __post_init__(self):
        if self.crypto_operation_types is None:
            self.crypto_operation_types = ["sbox", "mixcol", "keyadd", "shiftrows"]
        if self.hierarchical_levels is None:
            self.hierarchical_levels = [1, 4, 16, 64]


class CryptographicPositionalEncoding(nn.Module):
    """Specialized positional encoding for cryptographic operations."""
    
    def __init__(self, d_model: int, max_len: int = 8192, 
                 encoding_type: str = "sinusoidal"):
        super().__init__()
        
        self.d_model = d_model
        self.encoding_type = encoding_type
        
        if encoding_type == "sinusoidal":
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                               (-math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            self.register_buffer('pe', pe)
            
        elif encoding_type == "learned":
            self.pe = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
            
        elif encoding_type == "rotary":
            # Rotary Position Embedding (RoPE)
            inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
            self.register_buffer('inv_freq', inv_freq)
        
        # Cryptographic round encoding
        self.round_embedding = nn.Embedding(16, d_model)  # Up to 16 rounds (AES, etc.)
        
        # Operation type encoding
        self.operation_embedding = nn.Embedding(32, d_model)  # Various crypto operations
        
    def forward(self, x: torch.Tensor, 
                round_indices: Optional[torch.Tensor] = None,
                operation_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply positional encoding to input.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            round_indices: Cryptographic round indices [batch, seq_len]
            operation_indices: Operation type indices [batch, seq_len]
            
        Returns:
            Positionally encoded tensor [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.size()
        
        if self.encoding_type == "sinusoidal":
            pos_enc = self.pe[:seq_len].unsqueeze(0)
            x = x + pos_enc
            
        elif self.encoding_type == "learned":
            pos_enc = self.pe[:seq_len].unsqueeze(0)
            x = x + pos_enc
            
        elif self.encoding_type == "rotary":
            x = self._apply_rotary_encoding(x)
        
        # Add cryptographic structure encoding
        if round_indices is not None:
            round_enc = self.round_embedding(round_indices)
            x = x + round_enc
            
        if operation_indices is not None:
            op_enc = self.operation_embedding(operation_indices)
            x = x + op_enc
        
        return x
    
    def _apply_rotary_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary positional encoding."""
        batch_size, seq_len, d_model = x.size()
        
        # Generate position indices
        position_ids = torch.arange(seq_len, device=x.device, dtype=torch.float)
        
        # Compute rotation angles
        freqs = torch.outer(position_ids, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Apply rotation
        cos_emb = emb.cos()[None, :, :]
        sin_emb = emb.sin()[None, :, :]
        
        # Rotate x
        x_rot = self._rotate_half(x)
        x = x * cos_emb + x_rot * sin_emb
        
        return x
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the dimensions."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)


class HierarchicalAttention(nn.Module):
    """Multi-scale hierarchical attention for cryptographic sequences."""
    
    def __init__(self, d_model: int, n_heads: int, hierarchical_levels: List[int],
                 cross_scale: bool = True):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.hierarchical_levels = hierarchical_levels
        self.cross_scale = cross_scale
        
        # Scale-specific attention modules
        self.scale_attentions = nn.ModuleDict({
            f"scale_{scale}": nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                batch_first=True,
                dropout=0.1
            )
            for scale in hierarchical_levels
        })
        
        # Cross-scale fusion
        if cross_scale:
            self.cross_scale_fusion = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                batch_first=True,
                dropout=0.1
            )
        
        # Scale-specific projections
        self.scale_projections = nn.ModuleDict({
            f"proj_{scale}": nn.Linear(d_model, d_model)
            for scale in hierarchical_levels
        })
        
        # Output combination
        self.output_projection = nn.Linear(d_model * len(hierarchical_levels), d_model)
        
        logger.debug(f"Initialized hierarchical attention with scales: {hierarchical_levels}")
    
    def forward(self, x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through hierarchical attention.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional attention mask [batch, seq_len, seq_len]
            
        Returns:
            Multi-scale attended features [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.size()
        scale_outputs = []
        
        # Process each temporal scale
        for scale in self.hierarchical_levels:
            # Downsample for coarser scales
            if scale > 1:
                x_downsampled = self._temporal_downsample(x, scale)
                mask_downsampled = self._downsample_mask(attention_mask, scale) if attention_mask is not None else None
            else:
                x_downsampled = x
                mask_downsampled = attention_mask
            
            # Scale-specific attention
            attended_output, _ = self.scale_attentions[f"scale_{scale}"](
                x_downsampled, x_downsampled, x_downsampled,
                attn_mask=mask_downsampled
            )
            
            # Upsample back to original resolution
            if scale > 1:
                attended_output = self._temporal_upsample(attended_output, scale, seq_len)
            
            # Scale-specific projection
            attended_output = self.scale_projections[f"proj_{scale}"](attended_output)
            scale_outputs.append(attended_output)
        
        # Combine multi-scale features
        combined_features = torch.cat(scale_outputs, dim=-1)  # [batch, seq_len, d_model * n_scales]
        
        # Final projection
        output = self.output_projection(combined_features)
        
        # Cross-scale attention fusion
        if self.cross_scale:
            output_fused, _ = self.cross_scale_fusion(output, output, output)
            output = output + output_fused  # Residual connection
        
        return output
    
    def _temporal_downsample(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        """Downsample temporal dimension by averaging."""
        batch_size, seq_len, d_model = x.size()
        
        # Pad if necessary
        pad_len = (scale - seq_len % scale) % scale
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        
        # Reshape and average
        new_seq_len = (seq_len + pad_len) // scale
        x_reshaped = x.view(batch_size, new_seq_len, scale, d_model)
        x_downsampled = torch.mean(x_reshaped, dim=2)
        
        return x_downsampled
    
    def _temporal_upsample(self, x: torch.Tensor, scale: int, target_len: int) -> torch.Tensor:
        """Upsample temporal dimension by interpolation."""
        batch_size, seq_len, d_model = x.size()
        
        # Repeat each timestep 'scale' times
        x_repeated = x.unsqueeze(2).repeat(1, 1, scale, 1)  # [batch, seq_len, scale, d_model]
        x_upsampled = x_repeated.view(batch_size, seq_len * scale, d_model)
        
        # Trim to target length
        if x_upsampled.size(1) > target_len:
            x_upsampled = x_upsampled[:, :target_len, :]
        elif x_upsampled.size(1) < target_len:
            # Pad if necessary
            pad_len = target_len - x_upsampled.size(1)
            x_upsampled = F.pad(x_upsampled, (0, 0, 0, pad_len))
        
        return x_upsampled
    
    def _downsample_mask(self, mask: torch.Tensor, scale: int) -> torch.Tensor:
        """Downsample attention mask."""
        if mask is None:
            return None
        
        # Simple downsampling by selecting every scale-th element
        return mask[::scale, ::scale]


class CryptographicOperationLayer(nn.Module):
    """Layer specialized for modeling specific cryptographic operations."""
    
    def __init__(self, d_model: int, operation_types: List[str]):
        super().__init__()
        
        self.d_model = d_model
        self.operation_types = operation_types
        
        # Operation-specific experts
        self.operation_experts = nn.ModuleDict({
            op_type: nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
                nn.Dropout(0.1)
            )
            for op_type in operation_types
        })
        
        # Operation detection (which expert to use)
        self.operation_classifier = nn.Linear(d_model, len(operation_types))
        
        # Gating mechanism
        self.gate = nn.Linear(d_model, len(operation_types))
        
    def forward(self, x: torch.Tensor, 
                operation_hints: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through operation-specific processing.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            operation_hints: Optional operation type hints [batch, seq_len]
            
        Returns:
            Operation-aware processed features [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.size()
        
        # Operation classification (which expert to use)
        if operation_hints is not None:
            # Use provided hints
            operation_probs = F.one_hot(operation_hints, num_classes=len(self.operation_types)).float()
        else:
            # Classify operations automatically
            operation_logits = self.operation_classifier(x)
            operation_probs = F.softmax(operation_logits, dim=-1)
        
        # Compute gating weights
        gate_weights = torch.sigmoid(self.gate(x))  # [batch, seq_len, n_ops]
        
        # Apply operation-specific experts
        expert_outputs = []
        for i, op_type in enumerate(self.operation_types):
            expert_output = self.operation_experts[op_type](x)
            expert_outputs.append(expert_output)
        
        # Stack expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # [batch, seq_len, d_model, n_ops]
        
        # Weighted combination based on operation probabilities and gating
        combined_weights = operation_probs.unsqueeze(-2) * gate_weights.unsqueeze(-2)  # [batch, seq_len, 1, n_ops]
        
        # Weighted sum of expert outputs
        output = torch.sum(expert_outputs * combined_weights, dim=-1)
        
        return output


class CryptoTransformerOperator(NeuralOperatorBase):
    """Transformer-based neural operator for cryptographic sequence analysis.
    
    This operator leverages self-attention mechanisms to model complex temporal
    dependencies in cryptographic operations and side-channel traces.
    
    Research Innovation:
    - First Transformer architecture specialized for cryptographic sequence analysis
    - Hierarchical attention across multiple temporal scales
    - Operation-specific expert networks with gating
    - Cryptographic positional encoding with round and operation awareness
    """
    
    def __init__(self, config: TransformerOperatorConfig):
        super().__init__(config)
        
        self.config = config
        
        # Input projection and embedding
        self.input_projection = nn.Linear(config.input_channels, config.hidden_dim)
        
        # Cryptographic positional encoding
        self.positional_encoding = CryptographicPositionalEncoding(
            d_model=config.hidden_dim,
            max_len=config.max_position,
            encoding_type=config.positional_encoding_type
        )
        
        # Transformer layers with mixed local/global attention
        self.transformer_layers = nn.ModuleList()
        
        for layer_idx in range(config.n_transformer_layers):
            # Alternate between hierarchical and standard attention
            if layer_idx % config.global_attention_rate == 0:
                attention_module = HierarchicalAttention(
                    d_model=config.hidden_dim,
                    n_heads=config.n_heads,
                    hierarchical_levels=config.hierarchical_levels,
                    cross_scale=config.cross_scale_attention
                )
            else:
                attention_module = nn.MultiheadAttention(
                    embed_dim=config.hidden_dim,
                    num_heads=config.n_heads,
                    batch_first=True,
                    dropout=config.dropout
                )
            
            # Cryptographic operation layer
            crypto_layer = CryptographicOperationLayer(
                d_model=config.hidden_dim,
                operation_types=config.crypto_operation_types
            )
            
            # Standard transformer components
            feed_forward = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim * 4, config.hidden_dim),
                nn.Dropout(config.dropout)
            )
            
            layer_norm1 = nn.LayerNorm(config.hidden_dim)
            layer_norm2 = nn.LayerNorm(config.hidden_dim)
            layer_norm3 = nn.LayerNorm(config.hidden_dim)
            
            self.transformer_layers.append(nn.ModuleDict({
                'attention': attention_module,
                'crypto_ops': crypto_layer,
                'feed_forward': feed_forward,
                'norm1': layer_norm1,
                'norm2': layer_norm2,
                'norm3': layer_norm3
            }))
        
        # Output processing
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.output_projection = nn.Linear(config.hidden_dim, config.output_dim)
        
        # Temporal pooling strategies
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=1,
            batch_first=True
        )
        
        # Learnable pooling token
        self.pool_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)
        
        logger.info(f"Initialized CryptoTransformerOperator with {config.n_transformer_layers} layers, "
                   f"{config.n_heads} heads, hierarchical scales: {config.hierarchical_levels}")
    
    def forward(self, x: torch.Tensor,
                round_indices: Optional[torch.Tensor] = None,
                operation_indices: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through crypto transformer.
        
        Args:
            x: Input traces [batch, sequence_length, input_channels]
            round_indices: Cryptographic round indices [batch, sequence_length]
            operation_indices: Operation type indices [batch, sequence_length]
            attention_mask: Optional attention mask [batch, sequence_length]
            
        Returns:
            Output predictions [batch, output_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding with cryptographic structure
        x = self.positional_encoding(x, round_indices, operation_indices)
        
        # Create causal mask if needed
        if self.config.causal_attention:
            causal_mask = self._create_causal_mask(seq_len, x.device)
            if attention_mask is not None:
                attention_mask = attention_mask & causal_mask
            else:
                attention_mask = causal_mask
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            # Self-attention
            residual = x
            
            if isinstance(layer['attention'], HierarchicalAttention):
                # Hierarchical attention
                x_attended = layer['attention'](x, attention_mask)
            else:
                # Standard attention
                x_attended, _ = layer['attention'](x, x, x, attn_mask=attention_mask)
            
            x = layer['norm1'](x + x_attended)
            
            # Cryptographic operations processing
            residual = x
            x_crypto = layer['crypto_ops'](x, operation_indices)
            x = layer['norm2'](x + x_crypto)
            
            # Feed-forward
            residual = x
            x_ff = layer['feed_forward'](x)
            x = layer['norm3'](x + x_ff)
        
        # Final normalization
        x = self.output_norm(x)
        
        # Temporal pooling for sequence-to-vector conversion
        pooled_features = self._temporal_pooling(x, attention_mask)
        
        # Output projection
        output = self.output_projection(pooled_features)
        
        return output
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()
    
    def _temporal_pooling(self, x: torch.Tensor, 
                         attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Advanced temporal pooling for sequence features."""
        batch_size, seq_len, hidden_dim = x.size()
        
        # Strategy 1: Attention-based pooling with learnable query
        pool_token = self.pool_token.expand(batch_size, -1, -1)
        pooled_attn, _ = self.attention_pool(pool_token, x, x)
        pooled_attn = pooled_attn.squeeze(1)  # [batch, hidden_dim]
        
        # Strategy 2: Masked global average pooling
        if attention_mask is not None:
            # Apply mask and average
            mask_expanded = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
            pooled_avg = torch.sum(x * mask_expanded, dim=1) / (torch.sum(mask_expanded, dim=1) + 1e-9)
        else:
            pooled_avg = torch.mean(x, dim=1)
        
        # Strategy 3: Max pooling
        pooled_max = torch.max(x, dim=1)[0]
        
        # Strategy 4: Last token (if causal)
        if self.config.causal_attention:
            if attention_mask is not None:
                # Find last valid position for each sequence
                last_indices = torch.sum(attention_mask.float(), dim=1) - 1
                last_indices = torch.clamp(last_indices.long(), min=0, max=seq_len-1)
                pooled_last = x[torch.arange(batch_size), last_indices]
            else:
                pooled_last = x[:, -1, :]  # Last token
        else:
            pooled_last = pooled_avg  # Fallback
        
        # Combine pooling strategies
        combined_pooled = torch.cat([
            pooled_attn,
            pooled_avg,
            pooled_max,
            pooled_last
        ], dim=-1)
        
        # Final combination layer
        if not hasattr(self, 'pooling_combiner'):
            self.pooling_combiner = nn.Linear(hidden_dim * 4, hidden_dim).to(x.device)
        
        final_pooled = self.pooling_combiner(combined_pooled)
        
        return final_pooled
    
    def get_attention_weights(self, layer_idx: int = -1) -> Dict[str, torch.Tensor]:
        """Extract attention weights for visualization."""
        # This would need to be implemented by storing attention weights during forward pass
        # For now, return empty dict
        return {}
    
    def get_operation_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Get operation type predictions for interpretability."""
        batch_size, seq_len, _ = x.size()
        
        # Project input
        x_proj = self.input_projection(x)
        
        # Use first layer's operation classifier
        if hasattr(self.transformer_layers[0]['crypto_ops'], 'operation_classifier'):
            op_logits = self.transformer_layers[0]['crypto_ops'].operation_classifier(x_proj)
            return F.softmax(op_logits, dim=-1)
        
        return torch.zeros(batch_size, seq_len, len(self.config.crypto_operation_types))


class AdaptiveWindowTransformer(CryptoTransformerOperator):
    """Transformer with adaptive attention window based on cryptographic patterns."""
    
    def __init__(self, config: TransformerOperatorConfig):
        super().__init__(config)
        
        # Adaptive window predictor
        self.window_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Window size range
        self.min_window = 16
        self.max_window = config.temporal_window_size
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass with adaptive attention windows."""
        batch_size, seq_len, _ = x.size()
        
        # Project input
        x = self.input_projection(x)
        
        # Predict adaptive window sizes
        window_scores = self.window_predictor(x)  # [batch, seq_len, 1]
        window_sizes = (self.min_window + 
                       (self.max_window - self.min_window) * window_scores.squeeze(-1))
        
        # Apply positional encoding
        x = self.positional_encoding(x, 
                                   kwargs.get('round_indices'),
                                   kwargs.get('operation_indices'))
        
        # Create adaptive attention masks
        attention_mask = self._create_adaptive_mask(seq_len, window_sizes)
        
        # Continue with standard transformer processing
        return self._process_with_mask(x, attention_mask, **kwargs)
    
    def _create_adaptive_mask(self, seq_len: int, window_sizes: torch.Tensor) -> torch.Tensor:
        """Create adaptive attention masks based on predicted window sizes."""
        batch_size = window_sizes.size(0)
        device = window_sizes.device
        
        # Create base mask
        mask = torch.zeros(batch_size, seq_len, seq_len, device=device, dtype=torch.bool)
        
        for i in range(seq_len):
            for batch in range(batch_size):
                window_size = int(window_sizes[batch, i].item())
                start = max(0, i - window_size // 2)
                end = min(seq_len, i + window_size // 2 + 1)
                mask[batch, i, start:end] = True
        
        return mask
    
    def _process_with_mask(self, x: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """Process transformer layers with adaptive mask."""
        # Apply transformer layers (simplified)
        for layer in self.transformer_layers:
            if isinstance(layer['attention'], nn.MultiheadAttention):
                x_attended, _ = layer['attention'](x, x, x, attn_mask=attention_mask)
                x = layer['norm1'](x + x_attended)
            
            # Continue with other layers...
            x_crypto = layer['crypto_ops'](x, kwargs.get('operation_indices'))
            x = layer['norm2'](x + x_crypto)
            
            x_ff = layer['feed_forward'](x)
            x = layer['norm3'](x + x_ff)
        
        # Final processing
        x = self.output_norm(x)
        pooled_features = self._temporal_pooling(x, None)
        output = self.output_projection(pooled_features)
        
        return output