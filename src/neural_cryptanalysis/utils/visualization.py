"""Visualization utilities for neural cryptanalysis."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class PlotManager:
    """Manager for creating and organizing plots."""
    
    def __init__(self, backend: str = 'mock'):
        self.backend = backend
        self.plots = {}
        
    def create_trace_plot(self, traces: np.ndarray, title: str = "Traces") -> str:
        """Create a trace plot."""
        plot_id = f"trace_{len(self.plots)}"
        self.plots[plot_id] = {
            'type': 'trace',
            'data': traces,
            'title': title,
            'n_traces': traces.shape[0] if len(traces.shape) > 1 else 1,
            'trace_length': traces.shape[1] if len(traces.shape) > 1 else len(traces)
        }
        return plot_id
    
    def create_spectrum_plot(self, spectrum: np.ndarray, frequencies: Optional[np.ndarray] = None) -> str:
        """Create a spectrum plot."""
        plot_id = f"spectrum_{len(self.plots)}"
        self.plots[plot_id] = {
            'type': 'spectrum',
            'data': spectrum,
            'frequencies': frequencies,
            'title': 'Power Spectrum'
        }
        return plot_id
    
    def create_correlation_plot(self, correlations: np.ndarray) -> str:
        """Create a correlation plot."""
        plot_id = f"correlation_{len(self.plots)}"
        self.plots[plot_id] = {
            'type': 'correlation',
            'data': correlations,
            'title': 'Correlation Analysis'
        }
        return plot_id
    
    def save_plot(self, plot_id: str, filename: str):
        """Save plot to file (mock implementation)."""
        if plot_id in self.plots:
            print(f"Mock: Saving plot {plot_id} to {filename}")
        else:
            raise ValueError(f"Plot {plot_id} not found")
    
    def show_plot(self, plot_id: str):
        """Show plot (mock implementation)."""
        if plot_id in self.plots:
            plot = self.plots[plot_id]
            print(f"Mock: Showing {plot['type']} plot '{plot['title']}'")
        else:
            raise ValueError(f"Plot {plot_id} not found")


class AttentionVisualizer:
    """Visualizer for neural operator attention patterns."""
    
    def __init__(self):
        self.attention_maps = {}
    
    def visualize_attention(self, attention_weights: np.ndarray, input_traces: np.ndarray) -> str:
        """Visualize attention patterns."""
        vis_id = f"attention_{len(self.attention_maps)}"
        
        # Simple attention analysis
        attention_summary = {
            'max_attention': float(np.max(attention_weights)),
            'min_attention': float(np.min(attention_weights)),
            'attention_entropy': self._compute_entropy(attention_weights),
            'peak_locations': self._find_attention_peaks(attention_weights)
        }
        
        self.attention_maps[vis_id] = {
            'weights': attention_weights,
            'traces': input_traces,
            'summary': attention_summary
        }
        
        return vis_id
    
    def _compute_entropy(self, weights: np.ndarray) -> float:
        """Compute entropy of attention weights."""
        # Normalize weights to probabilities
        weights_norm = weights / (np.sum(weights) + 1e-8)
        # Compute entropy
        entropy = -np.sum(weights_norm * np.log(weights_norm + 1e-8))
        return float(entropy)
    
    def _find_attention_peaks(self, weights: np.ndarray, n_peaks: int = 5) -> List[int]:
        """Find top attention peak locations."""
        if len(weights.shape) == 1:
            peak_indices = np.argsort(weights)[-n_peaks:]
            return peak_indices.tolist()
        else:
            # For multi-dimensional, find peaks in flattened version
            flat_weights = weights.flatten()
            peak_indices = np.argsort(flat_weights)[-n_peaks:]
            return peak_indices.tolist()


class LeakageVisualizer:
    """Visualizer for leakage analysis results."""
    
    def __init__(self):
        self.leakage_maps = {}
    
    def visualize_leakage(self, traces: np.ndarray, leakage_model: str = 'hamming_weight') -> str:
        """Visualize leakage patterns in traces."""
        vis_id = f"leakage_{len(self.leakage_maps)}"
        
        # Compute simple leakage metrics
        trace_variance = np.var(traces, axis=0)
        trace_mean = np.mean(traces, axis=0)
        
        # Find potential leakage points (high variance areas)
        leakage_candidates = np.where(trace_variance > np.percentile(trace_variance, 95))[0]
        
        leakage_info = {
            'model': leakage_model,
            'variance_profile': trace_variance,
            'mean_profile': trace_mean,
            'leakage_candidates': leakage_candidates.tolist(),
            'max_variance': float(np.max(trace_variance)),
            'leakage_strength': float(np.std(trace_variance))
        }
        
        self.leakage_maps[vis_id] = {
            'traces': traces,
            'info': leakage_info
        }
        
        return vis_id
    
    def create_leakage_heatmap(self, vis_id: str) -> Dict[str, Any]:
        """Create a heatmap of leakage intensity."""
        if vis_id not in self.leakage_maps:
            raise ValueError(f"Leakage visualization {vis_id} not found")
        
        leakage_map = self.leakage_maps[vis_id]
        traces = leakage_map['traces']
        
        # Create simple heatmap data
        heatmap_data = {
            'variance_heatmap': leakage_map['info']['variance_profile'],
            'shape': traces.shape,
            'colormap': 'viridis',
            'title': f'Leakage Heatmap ({leakage_map["info"]["model"]})'
        }
        
        return heatmap_data
    
    def export_leakage_report(self, vis_id: str) -> Dict[str, Any]:
        """Export leakage analysis report."""
        if vis_id not in self.leakage_maps:
            raise ValueError(f"Leakage visualization {vis_id} not found")
        
        leakage_map = self.leakage_maps[vis_id]
        return {
            'visualization_id': vis_id,
            'analysis_summary': leakage_map['info'],
            'n_traces': leakage_map['traces'].shape[0],
            'trace_length': leakage_map['traces'].shape[1],
            'recommendations': self._generate_recommendations(leakage_map['info'])
        }
    
    def _generate_recommendations(self, leakage_info: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on leakage analysis."""
        recommendations = []
        
        if leakage_info['leakage_strength'] > 0.1:
            recommendations.append("High leakage detected - consider countermeasures")
        
        if len(leakage_info['leakage_candidates']) > 10:
            recommendations.append("Multiple leakage points found - comprehensive protection needed")
        
        if leakage_info['max_variance'] > 1.0:
            recommendations.append("High variance regions detected - focus protection efforts here")
        
        if not recommendations:
            recommendations.append("Low leakage detected - implementation appears resistant")
        
        return recommendations