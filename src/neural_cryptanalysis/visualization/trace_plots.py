"""
Trace visualization utilities for neural cryptanalysis.

Provides comprehensive plotting capabilities for side-channel traces,
signal analysis, and leakage visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# Set style for professional plots
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")


class TracePlotter:
    """Comprehensive trace plotting and visualization utilities."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """Initialize trace plotter.
        
        Args:
            figsize: Default figure size
            dpi: Figure DPI for high-quality plots
        """
        self.figsize = figsize
        self.dpi = dpi
        
        # Color schemes for different trace types
        self.color_schemes = {
            'power': '#1f77b4',      # Blue
            'em': '#ff7f0e',         # Orange  
            'acoustic': '#2ca02c',   # Green
            'optical': '#d62728',    # Red
            'multi': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        }
    
    def plot_traces_overview(self, 
                           traces: np.ndarray,
                           labels: Optional[np.ndarray] = None,
                           trace_type: str = 'power',
                           max_traces: int = 10,
                           highlight_poi: Optional[List[int]] = None) -> plt.Figure:
        """Plot overview of multiple traces.
        
        Args:
            traces: Trace data (n_traces, trace_length) or (n_traces, trace_length, n_channels)
            labels: Optional labels for traces
            trace_type: Type of traces ('power', 'em', 'acoustic')
            max_traces: Maximum number of traces to plot
            highlight_poi: Points of interest to highlight
            
        Returns:
            Matplotlib figure
        """
        if traces.ndim == 3:
            # Multi-channel traces
            return self._plot_multichannel_overview(traces, labels, max_traces, highlight_poi)
        
        # Single-channel traces
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi)
        
        # Select subset of traces to plot
        n_plot = min(max_traces, len(traces))
        plot_indices = np.linspace(0, len(traces) - 1, n_plot, dtype=int)
        
        color = self.color_schemes.get(trace_type, '#1f77b4')
        
        # Plot individual traces
        for i, idx in enumerate(plot_indices):
            alpha = 0.7 if n_plot <= 5 else 0.5
            label = f"Trace {idx}" if labels is None else f"Label {labels[idx]}"
            ax1.plot(traces[idx], alpha=alpha, color=color, linewidth=1, label=label if i < 5 else "")
        
        ax1.set_title(f'{trace_type.capitalize()} Traces Overview (n={n_plot})')
        ax1.set_xlabel('Time Samples')
        ax1.set_ylabel(f'{trace_type.capitalize()} Amplitude')
        ax1.grid(True, alpha=0.3)
        
        if n_plot <= 5:
            ax1.legend()
        
        # Highlight points of interest
        if highlight_poi:
            for poi in highlight_poi:
                ax1.axvline(x=poi, color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        # Plot average trace
        mean_trace = np.mean(traces, axis=0)
        std_trace = np.std(traces, axis=0)
        
        ax2.plot(mean_trace, color=color, linewidth=2, label='Average')
        ax2.fill_between(range(len(mean_trace)), 
                        mean_trace - std_trace, 
                        mean_trace + std_trace,
                        alpha=0.3, color=color, label='±1 std')
        
        ax2.set_title('Average Trace with Standard Deviation')
        ax2.set_xlabel('Time Samples')
        ax2.set_ylabel(f'{trace_type.capitalize()} Amplitude')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Highlight points of interest
        if highlight_poi:
            for poi in highlight_poi:
                ax2.axvline(x=poi, color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        plt.tight_layout()
        
        logger.info(f"Generated trace overview plot for {len(traces)} {trace_type} traces")
        
        return fig
    
    def _plot_multichannel_overview(self, 
                                  traces: np.ndarray,
                                  labels: Optional[np.ndarray],
                                  max_traces: int,
                                  highlight_poi: Optional[List[int]]) -> plt.Figure:
        """Plot overview for multi-channel traces."""
        
        n_channels = traces.shape[2]
        fig, axes = plt.subplots(n_channels, 1, figsize=(self.figsize[0], self.figsize[1] * n_channels // 2), 
                                dpi=self.dpi, sharex=True)
        
        if n_channels == 1:
            axes = [axes]
        
        n_plot = min(max_traces, len(traces))
        plot_indices = np.linspace(0, len(traces) - 1, n_plot, dtype=int)
        
        colors = self.color_schemes['multi']
        
        for ch in range(n_channels):
            color = colors[ch % len(colors)]
            
            # Plot individual traces for this channel
            for i, idx in enumerate(plot_indices):
                alpha = 0.7 if n_plot <= 5 else 0.5
                axes[ch].plot(traces[idx, :, ch], alpha=alpha, color=color, linewidth=1)
            
            # Plot average
            mean_trace = np.mean(traces[:, :, ch], axis=0)
            axes[ch].plot(mean_trace, color='black', linewidth=2, label='Average')
            
            axes[ch].set_title(f'Channel {ch + 1}')
            axes[ch].set_ylabel('Amplitude')
            axes[ch].grid(True, alpha=0.3)
            axes[ch].legend()
            
            # Highlight points of interest
            if highlight_poi:
                for poi in highlight_poi:
                    axes[ch].axvline(x=poi, color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        axes[-1].set_xlabel('Time Samples')
        plt.suptitle(f'Multi-Channel Traces Overview (n={n_plot})')
        plt.tight_layout()
        
        return fig
    
    def plot_snr_analysis(self, 
                         traces: np.ndarray,
                         labels: np.ndarray,
                         window_size: int = 100,
                         overlap: int = 50) -> plt.Figure:
        """Plot Signal-to-Noise Ratio analysis.
        
        Args:
            traces: Trace data
            labels: True labels for traces
            window_size: Window size for SNR computation
            overlap: Overlap between windows
            
        Returns:
            SNR analysis figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi)
        
        # Compute SNR for each time point
        snr_values = self._compute_snr(traces, labels)
        
        # Plot SNR over time
        ax1.plot(snr_values, color='blue', linewidth=2)
        ax1.set_title('Signal-to-Noise Ratio Over Time')
        ax1.set_xlabel('Time Samples')
        ax1.set_ylabel('SNR')
        ax1.grid(True, alpha=0.3)
        
        # Highlight high-SNR regions
        threshold = np.percentile(snr_values, 90)
        high_snr_points = np.where(snr_values > threshold)[0]
        ax1.scatter(high_snr_points, snr_values[high_snr_points], 
                   color='red', s=20, alpha=0.6, label=f'Top 10% SNR')
        ax1.legend()
        
        # Plot windowed SNR heatmap
        windowed_snr = self._compute_windowed_snr(traces, labels, window_size, overlap)
        
        im = ax2.imshow(windowed_snr.T, aspect='auto', cmap='viridis', origin='lower')
        ax2.set_title(f'Windowed SNR Heatmap (window={window_size}, overlap={overlap})')
        ax2.set_xlabel('Time Windows')
        ax2.set_ylabel('Label Values')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('SNR')
        
        plt.tight_layout()
        
        logger.info(f"Generated SNR analysis with max SNR: {np.max(snr_values):.4f}")
        
        return fig
    
    def plot_frequency_analysis(self, 
                               traces: np.ndarray,
                               sampling_rate: float = 1.0,
                               max_freq: Optional[float] = None) -> plt.Figure:
        """Plot frequency domain analysis of traces.
        
        Args:
            traces: Trace data
            sampling_rate: Sampling rate in Hz
            max_freq: Maximum frequency to plot
            
        Returns:
            Frequency analysis figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi)
        
        # Compute average power spectral density
        freqs, psd = self._compute_psd(traces, sampling_rate)
        
        if max_freq:
            freq_mask = freqs <= max_freq
            freqs = freqs[freq_mask]
            psd = psd[freq_mask]
        
        # Plot PSD
        ax1.semilogy(freqs, psd, color='blue', linewidth=2)
        ax1.set_title('Power Spectral Density')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('PSD (dB/Hz)')
        ax1.grid(True, alpha=0.3)
        
        # Plot spectrogram for first trace
        if len(traces) > 0:
            f, t, Sxx = self._compute_spectrogram(traces[0], sampling_rate)
            
            if max_freq:
                freq_mask = f <= max_freq
                f = f[freq_mask]
                Sxx = Sxx[freq_mask, :]
            
            im = ax2.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
            ax2.set_title('Spectrogram (First Trace)')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Frequency (Hz)')
            
            cbar = plt.colorbar(im, ax=ax2)
            cbar.set_label('Power (dB)')
        
        plt.tight_layout()
        
        logger.info(f"Generated frequency analysis up to {max_freq or 'Nyquist'} Hz")
        
        return fig
    
    def plot_correlation_analysis(self, 
                                 traces: np.ndarray,
                                 target_values: np.ndarray,
                                 correlation_type: str = 'pearson') -> plt.Figure:
        """Plot correlation analysis between traces and target values.
        
        Args:
            traces: Trace data
            target_values: Target intermediate values
            correlation_type: Type of correlation ('pearson', 'spearman')
            
        Returns:
            Correlation analysis figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi)
        
        # Compute correlations
        correlations = self._compute_correlations(traces, target_values, correlation_type)
        
        # Plot correlation over time
        ax1.plot(correlations, color='green', linewidth=2)
        ax1.set_title(f'{correlation_type.capitalize()} Correlation with Target Values')
        ax1.set_xlabel('Time Samples')
        ax1.set_ylabel('Correlation Coefficient')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Highlight significant correlations
        threshold = 3 * np.std(correlations)
        significant_points = np.abs(correlations) > threshold
        ax1.scatter(np.where(significant_points)[0], correlations[significant_points],
                   color='red', s=20, alpha=0.8, label='Significant correlations')
        ax1.legend()
        
        # Plot correlation histogram
        ax2.hist(correlations, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.set_title('Distribution of Correlation Coefficients')
        ax2.set_xlabel('Correlation Coefficient')
        ax2.set_ylabel('Frequency')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        max_corr = np.max(np.abs(correlations))
        logger.info(f"Generated correlation analysis with max |correlation|: {max_corr:.4f}")
        
        return fig
    
    def _compute_snr(self, traces: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute SNR for each time point."""
        unique_labels = np.unique(labels)
        
        # Group traces by label
        signal_variance = np.zeros(traces.shape[1])
        noise_variance = np.zeros(traces.shape[1])
        
        for label in unique_labels:
            mask = labels == label
            if np.sum(mask) > 1:
                group_traces = traces[mask]
                group_mean = np.mean(group_traces, axis=0)
                group_var = np.var(group_traces, axis=0)
                
                signal_variance += group_mean ** 2
                noise_variance += group_var
        
        signal_variance /= len(unique_labels)
        noise_variance /= len(unique_labels)
        
        # Avoid division by zero
        snr = np.divide(signal_variance, noise_variance + 1e-10)
        
        return snr
    
    def _compute_windowed_snr(self, traces: np.ndarray, labels: np.ndarray, 
                             window_size: int, overlap: int) -> np.ndarray:
        """Compute SNR for sliding windows."""
        step = window_size - overlap
        n_windows = (traces.shape[1] - window_size) // step + 1
        unique_labels = np.unique(labels)
        
        windowed_snr = np.zeros((n_windows, len(unique_labels)))
        
        for i in range(n_windows):
            start = i * step
            end = start + window_size
            window_traces = traces[:, start:end]
            
            for j, label in enumerate(unique_labels):
                mask = labels == label
                if np.sum(mask) > 1:
                    group_traces = window_traces[mask]
                    signal_var = np.var(np.mean(group_traces, axis=0))
                    noise_var = np.mean(np.var(group_traces, axis=0))
                    windowed_snr[i, j] = signal_var / (noise_var + 1e-10)
        
        return windowed_snr
    
    def _compute_psd(self, traces: np.ndarray, sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute power spectral density."""
        from scipy import signal
        
        # Compute average PSD across all traces
        freqs = None
        psd_sum = None
        
        for trace in traces[:min(100, len(traces))]:  # Limit for computational efficiency
            f, p = signal.welch(trace, fs=sampling_rate, nperseg=min(256, len(trace) // 4))
            
            if freqs is None:
                freqs = f
                psd_sum = p
            else:
                psd_sum += p
        
        psd_avg = psd_sum / min(100, len(traces))
        
        return freqs, psd_avg
    
    def _compute_spectrogram(self, trace: np.ndarray, sampling_rate: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute spectrogram for a single trace."""
        from scipy import signal
        
        f, t, Sxx = signal.spectrogram(trace, fs=sampling_rate, nperseg=min(128, len(trace) // 8))
        
        return f, t, Sxx
    
    def _compute_correlations(self, traces: np.ndarray, target_values: np.ndarray, 
                             correlation_type: str) -> np.ndarray:
        """Compute correlations between traces and target values."""
        correlations = np.zeros(traces.shape[1])
        
        for i in range(traces.shape[1]):
            if correlation_type == 'pearson':
                corr = np.corrcoef(traces[:, i], target_values)[0, 1]
            elif correlation_type == 'spearman':
                from scipy.stats import spearmanr
                corr, _ = spearmanr(traces[:, i], target_values)
            else:
                raise ValueError(f"Unknown correlation type: {correlation_type}")
            
            correlations[i] = corr if not np.isnan(corr) else 0
        
        return correlations
    
    def save_plot(self, fig: plt.Figure, filepath: Union[str, Path], 
                 format: str = 'png', dpi: Optional[int] = None) -> None:
        """Save plot to file.
        
        Args:
            fig: Matplotlib figure
            filepath: Output file path
            format: Output format ('png', 'pdf', 'svg')
            dpi: Output DPI (uses default if None)
        """
        dpi = dpi or self.dpi
        fig.savefig(filepath, format=format, dpi=dpi, bbox_inches='tight')
        logger.info(f"Plot saved to {filepath}")
    
    def close_all(self) -> None:
        """Close all matplotlib figures to free memory."""
        plt.close('all')
        logger.info("Closed all matplotlib figures")