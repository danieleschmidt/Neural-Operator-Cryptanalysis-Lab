"""
Advanced Countermeasure Evaluation Suite.

This module provides comprehensive automated testing of masking, hiding, shuffling,
and other side-channel countermeasures using advanced statistical techniques and 
neural operators.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from scipy.stats import ttest_ind, ks_2samp, entropy
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns

from .core import TraceData, NeuralSCA
from .utils.logging_utils import setup_logger
from .utils.performance import PerformanceMonitor

logger = setup_logger(__name__)

@dataclass
class CountermeasureConfig:
    """Configuration for countermeasure implementation."""
    countermeasure_type: str  # 'masking', 'hiding', 'shuffling', 'combined'
    order: int = 1
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    description: str = ""

@dataclass
class EvaluationMetrics:
    """Metrics for countermeasure evaluation."""
    theoretical_security_order: int
    practical_security_order: int
    traces_needed_90_percent: int
    traces_needed_95_percent: int
    max_success_rate: float
    snr_reduction_factor: float
    mutual_information: float
    t_test_statistics: Dict[str, float]
    detection_threshold: Optional[float] = None

class CountermeasureImplementation(ABC):
    """Abstract base class for countermeasure implementations."""
    
    def __init__(self, config: CountermeasureConfig):
        self.config = config
        self.is_enabled = config.enabled
        
    @abstractmethod
    def apply_countermeasure(self, traces: np.ndarray, 
                           intermediate_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply countermeasure to traces and intermediate values."""
        pass
    
    @abstractmethod
    def get_leakage_model(self) -> Callable:
        """Get leakage model for countermeasure."""
        pass
    
    @abstractmethod
    def estimate_security_order(self) -> int:
        """Estimate theoretical security order."""
        pass

class BooleanMasking(CountermeasureImplementation):
    """Boolean masking countermeasure implementation."""
    
    def __init__(self, config: CountermeasureConfig):
        super().__init__(config)
        self.masking_order = config.order
        self.refresh_randomness = config.parameters.get('refresh_randomness', True)
        
    def apply_countermeasure(self, traces: np.ndarray, 
                           intermediate_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Boolean masking to traces."""
        n_traces, trace_length = traces.shape
        n_shares = self.masking_order + 1
        
        # Generate masks
        masks = np.random.randint(0, 256, size=(n_traces, self.masking_order), dtype=np.uint8)
        
        # Create masked intermediate values
        masked_values = []
        for i in range(n_traces):
            # Split sensitive variable into shares
            shares = []
            cumulative_mask = 0
            
            for j in range(self.masking_order):
                shares.append(masks[i, j])
                cumulative_mask ^= masks[i, j]
            
            # Last share contains the secret
            if len(intermediate_values) > i:
                shares.append(intermediate_values[i] ^ cumulative_mask)
            else:
                shares.append(np.random.randint(0, 256) ^ cumulative_mask)
            
            masked_values.append(shares)
        
        # Simulate masked trace leakage
        masked_traces = self._simulate_masked_leakage(traces, masked_values)
        
        return masked_traces, np.array(masked_values)
    
    def _simulate_masked_leakage(self, original_traces: np.ndarray, 
                               masked_values: List[List[int]]) -> np.ndarray:
        """Simulate leakage from masked implementation."""
        n_traces, trace_length = original_traces.shape
        masked_traces = np.copy(original_traces)
        
        # Reduce leakage proportionally to masking order
        leakage_reduction = 1.0 / (2 ** self.masking_order)
        
        # Add leakage from individual shares
        for i in range(n_traces):
            shares = masked_values[i]
            
            # Each share contributes independent leakage
            for j, share in enumerate(shares):
                share_leakage_pos = (j * trace_length // len(shares)) + np.random.randint(0, 100)
                if share_leakage_pos < trace_length:
                    # Hamming weight leakage model
                    hw = bin(share).count('1')
                    leakage_amplitude = leakage_reduction * hw * 0.001
                    
                    # Add Gaussian spread around leakage point
                    leakage_spread = 50
                    start_idx = max(0, share_leakage_pos - leakage_spread)
                    end_idx = min(trace_length, share_leakage_pos + leakage_spread)
                    
                    for k in range(start_idx, end_idx):
                        distance = abs(k - share_leakage_pos)
                        weight = np.exp(-distance**2 / (2 * (leakage_spread / 3)**2))
                        masked_traces[i, k] += leakage_amplitude * weight
        
        return masked_traces
    
    def get_leakage_model(self) -> Callable:
        """Get leakage model for Boolean masking."""
        def masked_leakage_model(plaintext: np.ndarray, key: np.ndarray) -> np.ndarray:
            # For d-th order masking, attacker needs (d+1)-th order moments
            sbox_output = self._sbox_lookup(plaintext ^ key)
            return sbox_output
        
        return masked_leakage_model
    
    def _sbox_lookup(self, x: np.ndarray) -> np.ndarray:
        """AES S-box lookup."""
        sbox = [
            0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
            0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0
        ]
        return np.array([sbox[val & 0xFF] for val in x.flatten()]).reshape(x.shape)
    
    def estimate_security_order(self) -> int:
        """Estimate theoretical security order."""
        return self.masking_order

class ArithmeticMasking(CountermeasureImplementation):
    """Arithmetic masking countermeasure implementation."""
    
    def __init__(self, config: CountermeasureConfig):
        super().__init__(config)
        self.masking_order = config.order
        self.modulus = config.parameters.get('modulus', 2**32)
    
    def apply_countermeasure(self, traces: np.ndarray, 
                           intermediate_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply arithmetic masking to traces."""
        n_traces, trace_length = traces.shape
        
        # Generate arithmetic masks
        masks = np.random.randint(0, self.modulus, size=(n_traces, self.masking_order), dtype=np.uint32)
        
        # Create masked values
        masked_values = []
        for i in range(n_traces):
            shares = []
            cumulative_mask = 0
            
            for j in range(self.masking_order):
                shares.append(int(masks[i, j]))
                cumulative_mask = (cumulative_mask + masks[i, j]) % self.modulus
            
            # Last share
            if len(intermediate_values) > i:
                last_share = (int(intermediate_values[i]) - cumulative_mask) % self.modulus
                shares.append(last_share)
            else:
                last_share = (np.random.randint(0, 256) - cumulative_mask) % self.modulus
                shares.append(last_share)
            
            masked_values.append(shares)
        
        # Simulate arithmetic masking leakage
        masked_traces = self._simulate_arithmetic_leakage(traces, masked_values)
        
        return masked_traces, np.array(masked_values, dtype=object)
    
    def _simulate_arithmetic_leakage(self, original_traces: np.ndarray, 
                                   masked_values: List[List[int]]) -> np.ndarray:
        """Simulate leakage from arithmetic masking."""
        n_traces, trace_length = original_traces.shape
        masked_traces = np.copy(original_traces)
        
        # Arithmetic masking has different leakage characteristics
        for i in range(n_traces):
            shares = masked_values[i]
            
            for j, share in enumerate(shares):
                leakage_pos = (j * trace_length // len(shares)) + np.random.randint(0, 100)
                if leakage_pos < trace_length:
                    # Weight-based leakage model for arithmetic operations
                    weight = bin(share & 0xFFFF).count('1')  # Lower 16 bits
                    leakage_amplitude = weight * 0.0005
                    
                    # Add leakage with temporal spread
                    spread = 30
                    start_idx = max(0, leakage_pos - spread)
                    end_idx = min(trace_length, leakage_pos + spread)
                    
                    for k in range(start_idx, end_idx):
                        distance = abs(k - leakage_pos)
                        weight_factor = np.exp(-distance**2 / (2 * (spread / 3)**2))
                        masked_traces[i, k] += leakage_amplitude * weight_factor
        
        return masked_traces
    
    def get_leakage_model(self) -> Callable:
        """Get leakage model for arithmetic masking."""
        def arithmetic_leakage_model(plaintext: np.ndarray, key: np.ndarray) -> np.ndarray:
            result = (plaintext.astype(np.uint32) + key.astype(np.uint32)) % self.modulus
            return result.astype(np.uint8)
        
        return arithmetic_leakage_model
    
    def estimate_security_order(self) -> int:
        """Estimate theoretical security order."""
        return self.masking_order

class TemporalShuffling(CountermeasureImplementation):
    """Temporal shuffling countermeasure implementation."""
    
    def __init__(self, config: CountermeasureConfig):
        super().__init__(config)
        self.shuffle_window = config.parameters.get('shuffle_window', 1000)
        self.n_operations = config.parameters.get('n_operations', 16)
    
    def apply_countermeasure(self, traces: np.ndarray, 
                           intermediate_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply temporal shuffling to traces."""
        n_traces, trace_length = traces.shape
        shuffled_traces = np.zeros_like(traces)
        shuffled_intermediate_values = np.copy(intermediate_values)
        
        for i in range(n_traces):
            # Generate random permutation for operations
            operation_order = np.random.permutation(self.n_operations)
            
            # Apply shuffling
            shuffled_trace = self._shuffle_single_trace(traces[i], operation_order)
            shuffled_traces[i] = shuffled_trace
        
        return shuffled_traces, shuffled_intermediate_values
    
    def _shuffle_single_trace(self, trace: np.ndarray, operation_order: np.ndarray) -> np.ndarray:
        """Shuffle single trace based on operation order."""
        trace_length = len(trace)
        shuffled_trace = np.copy(trace)
        
        # Define operation blocks
        block_size = trace_length // self.n_operations
        
        # Create shuffled version
        shuffled_blocks = []
        for op_idx in operation_order:
            start_idx = op_idx * block_size
            end_idx = min((op_idx + 1) * block_size, trace_length)
            
            if start_idx < trace_length:
                block = trace[start_idx:end_idx]
                
                # Add random delay within shuffle window
                delay = np.random.randint(0, min(self.shuffle_window, block_size // 2))
                
                # Pad or truncate block
                if delay > 0:
                    padded_block = np.concatenate([np.zeros(delay), block])[:block_size]
                else:
                    padded_block = block
                
                shuffled_blocks.append(padded_block)
        
        # Reconstruct shuffled trace
        if shuffled_blocks:
            shuffled_trace = np.concatenate(shuffled_blocks)[:trace_length]
            
            # Pad if necessary
            if len(shuffled_trace) < trace_length:
                padding = trace_length - len(shuffled_trace)
                shuffled_trace = np.concatenate([shuffled_trace, np.zeros(padding)])
        
        return shuffled_trace
    
    def get_leakage_model(self) -> Callable:
        """Get leakage model for shuffling."""
        def shuffled_leakage_model(plaintext: np.ndarray, key: np.ndarray) -> np.ndarray:
            # Shuffling doesn't change the leakage model, just timing
            return self._sbox_lookup(plaintext ^ key)
        
        return shuffled_leakage_model
    
    def _sbox_lookup(self, x: np.ndarray) -> np.ndarray:
        """AES S-box lookup."""
        sbox = [0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76]
        return np.array([sbox[val & 0x0F] for val in x.flatten()]).reshape(x.shape)
    
    def estimate_security_order(self) -> int:
        """Estimate theoretical security order."""
        # Shuffling provides logarithmic security increase
        return int(np.log2(self.n_operations))

class AdvancedCountermeasureEvaluator:
    """Comprehensive evaluation system for side-channel countermeasures."""
    
    def __init__(self, neural_sca: NeuralSCA):
        self.neural_sca = neural_sca
        self.performance_monitor = PerformanceMonitor()
        
        # Evaluation parameters
        self.evaluation_params = {
            'min_traces': 1000,
            'max_traces': 100000,
            'batch_size': 1000,
            'confidence_level': 0.95,
            'success_threshold': 0.9
        }
        
        logger.info("Advanced countermeasure evaluator initialized")
    
    def evaluate_countermeasure(self, 
                               countermeasure: CountermeasureImplementation,
                               original_traces: TraceData,
                               max_traces: int = 50000) -> EvaluationMetrics:
        """Comprehensive evaluation of a countermeasure."""
        logger.info(f"Evaluating {countermeasure.config.countermeasure_type} "
                   f"countermeasure (order {countermeasure.config.order})")
        
        with self.performance_monitor.measure("countermeasure_evaluation"):
            # Apply countermeasure
            protected_traces, masked_values = countermeasure.apply_countermeasure(
                original_traces.traces, 
                original_traces.labels if hasattr(original_traces, 'labels') and original_traces.labels is not None else np.random.randint(0, 256, len(original_traces))
            )
            
            # Create protected trace data
            protected_trace_data = TraceData(
                traces=protected_traces,
                labels=original_traces.labels if hasattr(original_traces, 'labels') else None
            )
            
            # Statistical analysis
            stats_results = self._statistical_analysis(
                original_traces.traces, protected_traces, masked_values
            )
            
            # Attack evaluation
            attack_results = self._evaluate_attack_effectiveness(
                original_trace_data=original_traces,
                protected_trace_data=protected_trace_data,
                max_traces=max_traces
            )
            
            # Higher-order analysis
            higher_order_results = self._higher_order_analysis(
                protected_traces, masked_values, countermeasure.config.order
            )
            
            # Compile evaluation metrics
            metrics = EvaluationMetrics(
                theoretical_security_order=countermeasure.estimate_security_order(),
                practical_security_order=higher_order_results['practical_order'],
                traces_needed_90_percent=attack_results['traces_90_percent'],
                traces_needed_95_percent=attack_results['traces_95_percent'],
                max_success_rate=attack_results['max_success_rate'],
                snr_reduction_factor=stats_results['snr_reduction'],
                mutual_information=stats_results['mutual_information'],
                t_test_statistics=stats_results['t_test_results'],
                detection_threshold=higher_order_results.get('detection_threshold')
            )
        
        self._log_evaluation_results(countermeasure, metrics)
        return metrics
    
    def _statistical_analysis(self, 
                            original_traces: np.ndarray, 
                            protected_traces: np.ndarray,
                            masked_values: np.ndarray) -> Dict[str, Any]:
        """Perform statistical analysis of countermeasure effectiveness."""
        
        # SNR analysis
        original_snr = self._compute_snr(original_traces)
        protected_snr = self._compute_snr(protected_traces)
        snr_reduction = original_snr / (protected_snr + 1e-10)
        
        # Mutual information analysis
        if len(protected_traces) > 0:
            # Compute mutual information between traces and intermediate values
            flattened_traces = protected_traces.flatten()
            if hasattr(masked_values, '__len__') and len(masked_values) > 0:
                # Handle different masked value formats
                if isinstance(masked_values[0], list):
                    flattened_values = np.array([val[0] if val else 0 for val in masked_values])
                else:
                    flattened_values = masked_values.flatten()[:len(flattened_traces)]
                
                # Discretize for mutual information
                trace_bins = np.digitize(flattened_traces, 
                                       np.linspace(flattened_traces.min(), flattened_traces.max(), 50))
                value_bins = np.digitize(flattened_values,
                                       np.linspace(flattened_values.min(), flattened_values.max(), 50))
                
                mutual_info = mutual_info_score(trace_bins, value_bins)
            else:
                mutual_info = 0.0
        else:
            mutual_info = 0.0
        
        # T-test analysis (TVLA)
        t_test_results = self._perform_ttest_analysis(protected_traces)
        
        return {
            'snr_reduction': float(snr_reduction),
            'mutual_information': float(mutual_info),
            't_test_results': t_test_results,
            'original_snr': float(original_snr),
            'protected_snr': float(protected_snr)
        }
    
    def _compute_snr(self, traces: np.ndarray) -> float:
        """Compute Signal-to-Noise Ratio."""
        if len(traces) == 0:
            return 0.0
        
        signal_variance = np.var(np.mean(traces, axis=0))
        noise_variance = np.mean(np.var(traces, axis=1))
        
        return signal_variance / (noise_variance + 1e-10)
    
    def _perform_ttest_analysis(self, traces: np.ndarray) -> Dict[str, float]:
        """Perform T-test analysis (TVLA - Test Vector Leakage Assessment)."""
        if len(traces) < 2:
            return {'max_t_statistic': 0.0, 'points_above_threshold': 0}
        
        # Split traces into two groups (fixed vs random)
        mid_point = len(traces) // 2
        group1 = traces[:mid_point]
        group2 = traces[mid_point:]
        
        # Perform t-test for each time point
        t_statistics = []
        for i in range(min(group1.shape[1], group2.shape[1])):
            if np.var(group1[:, i]) > 1e-10 and np.var(group2[:, i]) > 1e-10:
                t_stat, _ = ttest_ind(group1[:, i], group2[:, i])
                t_statistics.append(abs(t_stat))
            else:
                t_statistics.append(0.0)
        
        t_statistics = np.array(t_statistics)
        
        # TVLA threshold (commonly 4.5)
        threshold = 4.5
        points_above_threshold = np.sum(t_statistics > threshold)
        
        return {
            'max_t_statistic': float(np.max(t_statistics)) if len(t_statistics) > 0 else 0.0,
            'mean_t_statistic': float(np.mean(t_statistics)) if len(t_statistics) > 0 else 0.0,
            'points_above_threshold': int(points_above_threshold),
            'total_points': len(t_statistics)
        }
    
    def _evaluate_attack_effectiveness(self,
                                     original_trace_data: TraceData,
                                     protected_trace_data: TraceData,
                                     max_traces: int) -> Dict[str, Any]:
        """Evaluate attack effectiveness against countermeasure."""
        
        # Test with increasing number of traces
        trace_counts = [1000, 2500, 5000, 10000, 20000, 50000]
        trace_counts = [n for n in trace_counts if n <= max_traces and n <= len(protected_trace_data)]
        
        success_rates = []
        
        for n_traces in trace_counts:
            # Use subset of traces
            subset_traces = TraceData(
                traces=protected_trace_data.traces[:n_traces],
                labels=protected_trace_data.labels[:n_traces] if hasattr(protected_trace_data, 'labels') and protected_trace_data.labels is not None else None
            )
            
            # Perform attack
            try:
                attack_results = self.neural_sca.attack(subset_traces)
                success_rate = attack_results.get('success', 0.0)
                success_rates.append(success_rate)
                
                logger.debug(f"Attack with {n_traces} traces: success_rate={success_rate:.3f}")
                
            except Exception as e:
                logger.warning(f"Attack failed with {n_traces} traces: {e}")
                success_rates.append(0.0)
        
        # Find traces needed for different success levels
        traces_90_percent = self._find_traces_needed(trace_counts, success_rates, 0.9)
        traces_95_percent = self._find_traces_needed(trace_counts, success_rates, 0.95)
        max_success_rate = max(success_rates) if success_rates else 0.0
        
        return {
            'success_rates': success_rates,
            'trace_counts': trace_counts,
            'traces_90_percent': traces_90_percent,
            'traces_95_percent': traces_95_percent,
            'max_success_rate': max_success_rate
        }
    
    def _find_traces_needed(self, trace_counts: List[int], 
                          success_rates: List[float], 
                          target_success: float) -> int:
        """Find number of traces needed for target success rate."""
        for i, success_rate in enumerate(success_rates):
            if success_rate >= target_success:
                return trace_counts[i]
        
        # If target not reached, extrapolate or return max
        return trace_counts[-1] * 2 if trace_counts else 100000
    
    def _higher_order_analysis(self, 
                             protected_traces: np.ndarray, 
                             masked_values: np.ndarray,
                             theoretical_order: int) -> Dict[str, Any]:
        """Perform higher-order statistical analysis."""
        
        # Compute moments up to theoretical order + 2
        max_order = min(theoretical_order + 2, 6)  # Limit computational complexity
        
        moments_analysis = {}
        for order in range(1, max_order + 1):
            moments = self._compute_centered_moments(protected_traces, order)
            moments_analysis[f'order_{order}_variance'] = np.var(moments)
            moments_analysis[f'order_{order}_mean'] = np.mean(np.abs(moments))
        
        # Estimate practical security order
        practical_order = self._estimate_practical_order(moments_analysis, theoretical_order)
        
        # Detection threshold analysis
        detection_threshold = self._compute_detection_threshold(protected_traces)
        
        return {
            'moments_analysis': moments_analysis,
            'practical_order': practical_order,
            'detection_threshold': detection_threshold
        }
    
    def _compute_centered_moments(self, traces: np.ndarray, order: int) -> np.ndarray:
        """Compute centered moments of specified order."""
        if len(traces) == 0:
            return np.array([])
        
        # Compute centered moments for each time point
        mean_traces = np.mean(traces, axis=0)
        centered_traces = traces - mean_traces
        
        moments = np.mean(centered_traces ** order, axis=0)
        return moments
    
    def _estimate_practical_order(self, moments_analysis: Dict[str, float], 
                                theoretical_order: int) -> int:
        """Estimate practical security order from moments analysis."""
        
        # Look for first moment order where variance drops significantly
        threshold_ratio = 0.1  # 10x reduction indicates security level
        
        for order in range(1, len(moments_analysis) // 2 + 1):
            current_key = f'order_{order}_variance'
            next_key = f'order_{order + 1}_variance'
            
            if current_key in moments_analysis and next_key in moments_analysis:
                current_var = moments_analysis[current_key]
                next_var = moments_analysis[next_key]
                
                if next_var < current_var * threshold_ratio:
                    return order
        
        # Default to theoretical order if no clear drop
        return theoretical_order
    
    def _compute_detection_threshold(self, traces: np.ndarray) -> float:
        """Compute statistical detection threshold."""
        if len(traces) == 0:
            return 0.0
        
        # Use variance of trace means as detection threshold
        trace_means = np.mean(traces, axis=1)
        return float(np.var(trace_means))
    
    def compare_countermeasures(self, 
                               countermeasures: List[CountermeasureImplementation],
                               original_traces: TraceData) -> Dict[str, Any]:
        """Compare multiple countermeasures side by side."""
        logger.info(f"Comparing {len(countermeasures)} countermeasures")
        
        comparison_results = {}
        
        for i, countermeasure in enumerate(countermeasures):
            cm_name = f"{countermeasure.config.countermeasure_type}_order_{countermeasure.config.order}"
            
            logger.info(f"Evaluating countermeasure {i+1}/{len(countermeasures)}: {cm_name}")
            
            metrics = self.evaluate_countermeasure(countermeasure, original_traces)
            comparison_results[cm_name] = metrics
        
        # Generate comparison summary
        summary = self._generate_comparison_summary(comparison_results)
        
        return {
            'individual_results': comparison_results,
            'summary': summary
        }
    
    def _generate_comparison_summary(self, results: Dict[str, EvaluationMetrics]) -> Dict[str, Any]:
        """Generate summary of countermeasure comparison."""
        
        summary = {
            'best_snr_reduction': {'name': '', 'value': 0.0},
            'best_traces_needed': {'name': '', 'value': float('inf')},
            'highest_security_order': {'name': '', 'value': 0},
            'lowest_success_rate': {'name': '', 'value': 1.0}
        }
        
        for name, metrics in results.items():
            # Best SNR reduction
            if metrics.snr_reduction_factor > summary['best_snr_reduction']['value']:
                summary['best_snr_reduction'] = {'name': name, 'value': metrics.snr_reduction_factor}
            
            # Best traces needed (highest)
            if metrics.traces_needed_90_percent < summary['best_traces_needed']['value']:
                summary['best_traces_needed'] = {'name': name, 'value': metrics.traces_needed_90_percent}
            
            # Highest security order
            if metrics.practical_security_order > summary['highest_security_order']['value']:
                summary['highest_security_order'] = {'name': name, 'value': metrics.practical_security_order}
            
            # Lowest success rate
            if metrics.max_success_rate < summary['lowest_success_rate']['value']:
                summary['lowest_success_rate'] = {'name': name, 'value': metrics.max_success_rate}
        
        return summary
    
    def _log_evaluation_results(self, 
                              countermeasure: CountermeasureImplementation,
                              metrics: EvaluationMetrics):
        """Log detailed evaluation results."""
        cm_name = countermeasure.config.countermeasure_type
        
        logger.info(f"Countermeasure Evaluation Results: {cm_name}")
        logger.info(f"  Theoretical Security Order: {metrics.theoretical_security_order}")
        logger.info(f"  Practical Security Order: {metrics.practical_security_order}")
        logger.info(f"  Traces Needed (90%): {metrics.traces_needed_90_percent}")
        logger.info(f"  Traces Needed (95%): {metrics.traces_needed_95_percent}")
        logger.info(f"  Max Success Rate: {metrics.max_success_rate:.3f}")
        logger.info(f"  SNR Reduction Factor: {metrics.snr_reduction_factor:.2f}")
        logger.info(f"  Mutual Information: {metrics.mutual_information:.6f}")
        
        if metrics.t_test_statistics:
            logger.info(f"  T-test Max Statistic: {metrics.t_test_statistics.get('max_t_statistic', 0):.2f}")
            logger.info(f"  T-test Points Above Threshold: {metrics.t_test_statistics.get('points_above_threshold', 0)}")

# Factory functions for creating countermeasures
def create_boolean_masking(order: int = 1, **kwargs) -> BooleanMasking:
    """Create Boolean masking countermeasure."""
    config = CountermeasureConfig(
        countermeasure_type='boolean_masking',
        order=order,
        parameters=kwargs,
        description=f"Boolean masking of order {order}"
    )
    return BooleanMasking(config)

def create_arithmetic_masking(order: int = 1, modulus: int = 2**32, **kwargs) -> ArithmeticMasking:
    """Create arithmetic masking countermeasure."""
    config = CountermeasureConfig(
        countermeasure_type='arithmetic_masking',
        order=order,
        parameters={'modulus': modulus, **kwargs},
        description=f"Arithmetic masking of order {order} (mod {modulus})"
    )
    return ArithmeticMasking(config)

def create_temporal_shuffling(n_operations: int = 16, shuffle_window: int = 1000, **kwargs) -> TemporalShuffling:
    """Create temporal shuffling countermeasure."""
    config = CountermeasureConfig(
        countermeasure_type='temporal_shuffling',
        order=int(np.log2(n_operations)),
        parameters={'n_operations': n_operations, 'shuffle_window': shuffle_window, **kwargs},
        description=f"Temporal shuffling with {n_operations} operations"
    )
    return TemporalShuffling(config)