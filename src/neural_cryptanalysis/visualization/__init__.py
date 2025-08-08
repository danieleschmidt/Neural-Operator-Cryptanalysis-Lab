"""
Visualization utilities for neural cryptanalysis.

Provides comprehensive visualization tools for traces, attack results,
neural operator analysis, and performance metrics.
"""

from .trace_plots import TracePlotter
from .attack_analysis import AttackVisualizer
from .performance_plots import PerformancePlotter
from .interactive_dashboard import InteractiveDashboard

__all__ = [
    'TracePlotter',
    'AttackVisualizer', 
    'PerformancePlotter',
    'InteractiveDashboard'
]

# Convenience functions
def plot_attack_results(results):
    """Quick plot of attack results."""
    visualizer = AttackVisualizer()
    return visualizer.plot_attack_summary(results)

def plot_traces(traces, labels=None):
    """Quick plot of trace data."""
    plotter = TracePlotter()
    return plotter.plot_traces_overview(traces, labels)

def plot_performance_comparison(results):
    """Quick plot of performance comparison."""
    plotter = PerformancePlotter()
    return plotter.plot_architecture_comparison(results)