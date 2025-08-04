"""Neural Operator Cryptanalysis Lab - Defensive Security Research Tool.

This package provides neural operator-based tools for defensive side-channel analysis
of post-quantum cryptographic implementations. It is designed for security researchers,
cryptographic implementers, and academic use.

Key Components:
- Neural Operators: FNO, DeepONet, and specialized architectures
- Side-Channel Analysis: Power, EM, acoustic, and optical analysis
- Post-Quantum Targets: Lattice, code-based, hash-based, and isogeny schemes
- Attack Strategies: Template, profiling, and adaptive attacks
- Countermeasure Evaluation: Masking, hiding, and shuffling assessment

Usage:
    from neural_cryptanalysis import NeuralSCA, LeakageSimulator
    from neural_cryptanalysis.targets import KyberImplementation
    
    # Basic usage example
    target = KyberImplementation(version='kyber768')
    simulator = LeakageSimulator(device_model='stm32f4')
    neural_sca = NeuralSCA(architecture='fourier_neural_operator')

Warning:
    This tool is for defensive security research only. Users must follow
    responsible disclosure practices and obtain proper authorization before
    testing on any systems.
"""

from typing import List, Optional

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

__author__ = "Terragon Labs"
__email__ = "research@terragonlabs.com"
__license__ = "GPL-3.0"

# Core imports for convenience
from .neural_operators import NeuralOperatorBase
from .side_channels import SideChannelAnalyzer
from .utils import config, logging_utils

# Main API classes
from .core import NeuralSCA, LeakageSimulator

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "__license__",
    "NeuralSCA",
    "LeakageSimulator",
    "NeuralOperatorBase",
    "SideChannelAnalyzer",
]

# Responsible use notice
def _show_responsible_use_notice() -> None:
    """Display responsible use notice on first import."""
    import warnings
    warnings.warn(
        "Neural Operator Cryptanalysis Lab - For Defensive Security Research Only\n"
        "This tool must be used responsibly and ethically. Users must:\n"
        "- Follow responsible disclosure practices\n" 
        "- Obtain proper authorization before testing\n"
        "- Contribute defensive improvements to the community\n"
        "See LICENSE and SECURITY.md for full terms.",
        UserWarning,
        stacklevel=2
    )

# Show notice on import
_show_responsible_use_notice()