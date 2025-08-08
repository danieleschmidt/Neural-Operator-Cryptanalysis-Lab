"""
Dataset generation and management for neural cryptanalysis.

This module provides synthetic and real-world dataset generation capabilities
for training and evaluating neural operator-based side-channel attacks.
"""

from .synthetic import SyntheticDatasetGenerator
from .loaders import DatasetLoader, TraceDataset
from .augmentation import DataAugmentation
from .validation import DatasetValidator

__all__ = [
    'SyntheticDatasetGenerator',
    'DatasetLoader', 
    'TraceDataset',
    'DataAugmentation',
    'DatasetValidator'
]