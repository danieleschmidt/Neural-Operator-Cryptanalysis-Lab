"""Data utilities for neural cryptanalysis."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


class TraceLoader:
    """Utility for loading trace data from various formats."""
    
    def __init__(self, format_type: str = 'numpy'):
        self.format_type = format_type
        
    def load_traces(self, file_path: str) -> np.ndarray:
        """Load traces from file."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Trace file not found: {file_path}")
        
        if self.format_type == 'numpy':
            return np.load(file_path)
        elif self.format_type == 'text':
            return np.loadtxt(file_path)
        else:
            # Default to random data for mock
            return np.random.randn(1000, 10000)
    
    def save_traces(self, traces: np.ndarray, file_path: str):
        """Save traces to file."""
        if self.format_type == 'numpy':
            np.save(file_path, traces)
        elif self.format_type == 'text':
            np.savetxt(file_path, traces)


class DataValidator:
    """Validator for trace data quality and integrity."""
    
    def __init__(self):
        self.validation_rules = {
            'min_traces': 100,
            'min_trace_length': 1000,
            'max_nan_ratio': 0.1,
            'max_inf_ratio': 0.01
        }
    
    def validate_traces(self, traces: np.ndarray) -> Dict[str, Any]:
        """Validate trace data and return validation report."""
        n_traces, trace_length = traces.shape
        
        # Check basic requirements
        issues = []
        if n_traces < self.validation_rules['min_traces']:
            issues.append(f"Too few traces: {n_traces} < {self.validation_rules['min_traces']}")
        
        if trace_length < self.validation_rules['min_trace_length']:
            issues.append(f"Traces too short: {trace_length} < {self.validation_rules['min_trace_length']}")
        
        # Check for problematic values
        nan_ratio = np.sum(np.isnan(traces)) / traces.size
        if nan_ratio > self.validation_rules['max_nan_ratio']:
            issues.append(f"Too many NaN values: {nan_ratio:.3f} > {self.validation_rules['max_nan_ratio']}")
        
        inf_ratio = np.sum(np.isinf(traces)) / traces.size
        if inf_ratio > self.validation_rules['max_inf_ratio']:
            issues.append(f"Too many infinite values: {inf_ratio:.3f} > {self.validation_rules['max_inf_ratio']}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'n_traces': n_traces,
            'trace_length': trace_length,
            'nan_ratio': nan_ratio,
            'inf_ratio': inf_ratio,
            'data_range': (float(np.min(traces)), float(np.max(traces)))
        }
    
    def clean_traces(self, traces: np.ndarray) -> np.ndarray:
        """Clean trace data by removing/fixing problematic values."""
        cleaned = traces.copy()
        
        # Replace NaN with median
        if np.any(np.isnan(cleaned)):
            median_val = np.nanmedian(cleaned)
            cleaned[np.isnan(cleaned)] = median_val
        
        # Replace infinite values with clipped values
        if np.any(np.isinf(cleaned)):
            finite_mask = np.isfinite(cleaned)
            if np.any(finite_mask):
                min_val = np.min(cleaned[finite_mask])
                max_val = np.max(cleaned[finite_mask])
                cleaned[np.isposinf(cleaned)] = max_val
                cleaned[np.isneginf(cleaned)] = min_val
        
        return cleaned


class StatisticalAnalyzer:
    """Statistical analysis utilities for trace data."""
    
    def __init__(self):
        pass
    
    def compute_statistics(self, traces: np.ndarray) -> Dict[str, Any]:
        """Compute comprehensive statistics for trace data."""
        stats = {
            'n_traces': traces.shape[0],
            'trace_length': traces.shape[1],
            'mean': float(np.mean(traces)),
            'std': float(np.std(traces)),
            'min': float(np.min(traces)),
            'max': float(np.max(traces)),
            'median': float(np.median(traces)),
        }
        
        # Per-trace statistics
        stats['trace_means'] = np.mean(traces, axis=1)
        stats['trace_stds'] = np.std(traces, axis=1)
        
        # Per-sample statistics
        stats['sample_means'] = np.mean(traces, axis=0)
        stats['sample_stds'] = np.std(traces, axis=0)
        stats['sample_vars'] = np.var(traces, axis=0)
        
        return stats
    
    def find_points_of_interest(self, traces: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Find points of interest in traces."""
        if labels is not None:
            # Use variance-based POI selection with labels
            poi_scores = np.var(traces, axis=0)
        else:
            # Use simple variance-based selection
            poi_scores = np.var(traces, axis=0)
        
        # Return indices of top POIs
        n_pois = min(100, len(poi_scores) // 10)
        return np.argsort(poi_scores)[-n_pois:]
    
    def correlation_analysis(self, traces: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Compute correlation between traces and reference."""
        correlations = []
        for i in range(traces.shape[1]):
            corr = np.corrcoef(traces[:, i], reference)[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0.0)
        
        return np.array(correlations)