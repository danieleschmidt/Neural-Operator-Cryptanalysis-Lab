"""Base neural operator classes and configurations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np


@dataclass
class OperatorConfig:
    """Configuration for neural operators.
    
    Attributes:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of layers
        activation: Activation function name
        dropout: Dropout rate
        normalization: Normalization type ('batch', 'layer', 'instance')
        use_residual: Whether to use residual connections
        device: Computing device ('cpu', 'cuda')
    """
    input_dim: int = 1
    output_dim: int = 256
    hidden_dim: int = 64
    num_layers: int = 4
    activation: str = "gelu"
    dropout: float = 0.1
    normalization: str = "layer"
    use_residual: bool = True
    device: str = "cpu"


class NeuralOperatorBase(nn.Module, ABC):
    """Abstract base class for neural operators.
    
    Provides common functionality for all neural operator architectures
    including initialization, forward pass structure, and utility methods.
    """
    
    def __init__(self, config: OperatorConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize activation
        self.activation = self._get_activation(config.activation)
        
        # Initialize normalization
        self.normalization = config.normalization
        
        # Performance metrics
        self.training_history = []
        
    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(),
            'silu': nn.SiLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
        }
        
        if activation_name not in activations:
            raise ValueError(f"Unsupported activation: {activation_name}")
            
        return activations[activation_name]
    
    def _get_normalization(self, dim: int) -> nn.Module:
        """Get normalization layer."""
        if self.normalization == 'batch':
            return nn.BatchNorm1d(dim)
        elif self.normalization == 'layer':
            return nn.LayerNorm(dim)
        elif self.normalization == 'instance':
            return nn.InstanceNorm1d(dim)
        else:
            return nn.Identity()
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the neural operator.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**2
            memory_cached = torch.cuda.memory_reserved(self.device) / 1024**2
            return {
                'allocated_mb': memory_allocated,
                'cached_mb': memory_cached,
                'parameters': self.count_parameters()
            }
        else:
            return {'parameters': self.count_parameters()}
    
    def save_checkpoint(self, path: str, epoch: int = 0, loss: float = 0.0):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'epoch': epoch,
            'loss': loss,
            'training_history': self.training_history,
        }
        torch.save(checkpoint, path)
    
    @classmethod
    def load_checkpoint(cls, path: str) -> Tuple['NeuralOperatorBase', Dict[str, Any]]:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        
        # Create model instance
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore training state
        model.training_history = checkpoint.get('training_history', [])
        
        return model, checkpoint
    
    def freeze_layers(self, layer_names: List[str]):
        """Freeze specific layers."""
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
    
    def unfreeze_layers(self, layer_names: List[str]):
        """Unfreeze specific layers."""
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
    
    def get_layer_gradients(self) -> Dict[str, float]:
        """Get gradient norms for each layer."""
        gradients = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.norm().item()
            else:
                gradients[name] = 0.0
        return gradients
    
    def initialize_weights(self, method: str = 'xavier_uniform'):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if method == 'xavier_uniform':
                    nn.init.xavier_uniform_(module.weight)
                elif method == 'xavier_normal':
                    nn.init.xavier_normal_(module.weight)
                elif method == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                elif method == 'kaiming_normal':
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class OperatorLoss(nn.Module):
    """Custom loss functions for neural operators."""
    
    def __init__(self, loss_type: str = 'mse', reduction: str = 'mean'):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss between predictions and targets."""
        if self.loss_type == 'mse':
            loss = nn.functional.mse_loss(pred, target, reduction=self.reduction)
        elif self.loss_type == 'mae':
            loss = nn.functional.l1_loss(pred, target, reduction=self.reduction)
        elif self.loss_type == 'cross_entropy':
            loss = nn.functional.cross_entropy(pred, target, reduction=self.reduction)
        elif self.loss_type == 'focal':
            loss = self._focal_loss(pred, target)
        elif self.loss_type == 'ranking':
            loss = self._ranking_loss(pred, target)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
            
        return loss
    
    def _focal_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                   alpha: float = 1.0, gamma: float = 2.0) -> torch.Tensor:
        """Focal loss for imbalanced classification."""
        ce_loss = nn.functional.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
    def _ranking_loss(self, pred: torch.Tensor, target: torch.Tensor,
                     margin: float = 1.0) -> torch.Tensor:
        """Ranking loss for key recovery attacks."""
        # Assume target contains correct key byte index
        batch_size = pred.size(0)
        n_classes = pred.size(1)
        
        correct_scores = pred.gather(1, target.unsqueeze(1))
        
        # Compute margin loss
        losses = []
        for i in range(n_classes):
            if i != target[0].item():  # Skip correct class
                incorrect_scores = pred[:, i:i+1]
                loss = nn.functional.relu(margin - correct_scores + incorrect_scores)
                losses.append(loss)
        
        ranking_loss = torch.cat(losses, dim=1).mean()
        return ranking_loss