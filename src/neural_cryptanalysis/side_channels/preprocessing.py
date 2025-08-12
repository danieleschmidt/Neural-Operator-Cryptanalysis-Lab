"""Preprocessing utilities for side-channel traces."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from .base import TraceData


class TracePreprocessor:
    """Preprocessor for side-channel traces."""
    
    def __init__(self, methods: List[str] = ['standardize']):
        self.methods = methods
        self.fitted_params = {}
        
    def fit(self, traces: np.ndarray):
        """Fit preprocessing parameters."""
        for method in self.methods:
            if method == 'standardize':
                self.fitted_params['standardize'] = {
                    'mean': np.mean(traces, axis=0),
                    'std': np.std(traces, axis=0) + 1e-8  # Add small epsilon
                }
            elif method == 'normalize':
                self.fitted_params['normalize'] = {
                    'min': np.min(traces, axis=0),
                    'max': np.max(traces, axis=0)
                }
        return self
    
    def transform(self, traces: np.ndarray) -> np.ndarray:
        """Apply preprocessing transformations."""
        result = traces.copy()
        
        for method in self.methods:
            if method == 'standardize' and 'standardize' in self.fitted_params:
                params = self.fitted_params['standardize']
                result = (result - params['mean']) / params['std']
            elif method == 'normalize' and 'normalize' in self.fitted_params:
                params = self.fitted_params['normalize']
                result = (result - params['min']) / (params['max'] - params['min'] + 1e-8)
            elif method == 'center':
                result = result - np.mean(result, axis=0)
        
        return result
    
    def fit_transform(self, traces: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(traces).transform(traces)


class FeatureExtractor:
    """Feature extraction for side-channel analysis."""
    
    def __init__(self, feature_types: List[str] = ['statistical']):
        self.feature_types = feature_types
    
    def extract_features(self, traces: np.ndarray) -> np.ndarray:
        """Extract features from traces."""
        features = []
        
        for feature_type in self.feature_types:
            if feature_type == 'statistical':
                features.extend(self._extract_statistical_features(traces))
            elif feature_type == 'spectral':
                features.extend(self._extract_spectral_features(traces))
            elif feature_type == 'temporal':
                features.extend(self._extract_temporal_features(traces))
        
        return np.array(features).T if features else np.array([])
    
    def _extract_statistical_features(self, traces: np.ndarray) -> List[np.ndarray]:
        """Extract statistical features."""
        features = []
        features.append(np.mean(traces, axis=1))  # Mean
        features.append(np.std(traces, axis=1))   # Standard deviation
        features.append(np.var(traces, axis=1))   # Variance
        features.append(np.max(traces, axis=1))   # Maximum
        features.append(np.min(traces, axis=1))   # Minimum
        return features
    
    def _extract_spectral_features(self, traces: np.ndarray) -> List[np.ndarray]:
        """Extract spectral features."""
        features = []
        # Simple spectral features using numpy
        for i in range(traces.shape[0]):
            fft_trace = np.fft.fft(traces[i])
            features.append([
                np.sum(np.abs(fft_trace)),  # Spectral energy
                np.argmax(np.abs(fft_trace)),  # Dominant frequency
                np.mean(np.abs(fft_trace))  # Mean spectral magnitude
            ])
        return [np.array(features)]
    
    def _extract_temporal_features(self, traces: np.ndarray) -> List[np.ndarray]:
        """Extract temporal features."""
        features = []
        # Simple temporal features
        for i in range(traces.shape[0]):
            trace = traces[i]
            features.append([
                np.argmax(trace),  # Peak location
                np.sum(np.diff(trace) ** 2),  # Total variation
                len(trace)  # Length
            ])
        return [np.array(features)]