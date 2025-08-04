"""Neural Operator architectures for side-channel analysis.

This module provides various neural operator architectures optimized for
modeling side-channel leakage patterns in cryptographic implementations.

Available Architectures:
- Fourier Neural Operator (FNO): Spectral domain processing
- Deep Operator Network (DeepONet): Function approximation
- Multipole Graph Neural Operator (MGNO): Graph-based modeling
- Custom architectures for specific cryptographic operations

Key Features:
- Efficient spectral convolutions for signal processing
- Multi-modal fusion capabilities
- Adaptive architectures for different trace characteristics
- Built-in regularization and normalization
"""

from .base import NeuralOperatorBase, OperatorConfig
from .fno import FourierNeuralOperator, SpectralConv1d, SpectralConv2d
from .deeponet import DeepOperatorNetwork, BranchNet, TrunkNet
from .custom import SideChannelFNO, LeakageFNO, MultiModalOperator

__all__ = [
    "NeuralOperatorBase",
    "OperatorConfig", 
    "FourierNeuralOperator",
    "SpectralConv1d",
    "SpectralConv2d",
    "DeepOperatorNetwork",
    "BranchNet",
    "TrunkNet",
    "SideChannelFNO",
    "LeakageFNO", 
    "MultiModalOperator",
]