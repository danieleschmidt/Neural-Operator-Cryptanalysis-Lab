"""Cryptographic target implementations for side-channel analysis.

This module provides implementations and models of various cryptographic
algorithms, with focus on post-quantum cryptography schemes.

Target Categories:
- Post-Quantum: Lattice, code-based, hash-based, and isogeny schemes
- Classical: AES, RSA, ECC for comparison and validation
- Implementations: Real-world implementation models with countermeasures

Features:
- Accurate intermediate value computation
- Countermeasure modeling (masking, shuffling, hiding)
- Timing and power consumption models
- Hardware-specific implementation variants
"""

from .base import CryptographicTarget, ImplementationConfig
from .post_quantum import (
    KyberImplementation, DilithiumImplementation, 
    ClassicMcElieceImplementation, SPHINCSImplementation
)
from .classical import AESImplementation, RSAImplementation, ECCImplementation

__all__ = [
    "CryptographicTarget",
    "ImplementationConfig",
    "KyberImplementation", 
    "DilithiumImplementation",
    "ClassicMcElieceImplementation",
    "SPHINCSImplementation",
    "AESImplementation",
    "RSAImplementation", 
    "ECCImplementation",
]