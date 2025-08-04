"""Base classes for side-channel analysis."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import numpy as np
import torch
import torch.nn as nn


class ChannelType(Enum):
    """Types of side-channels."""
    POWER = "power"
    EM_NEAR = "em_near" 
    EM_FAR = "em_far"
    ACOUSTIC = "acoustic"
    OPTICAL = "optical"
    TIMING = "timing"
    CACHE = "cache"


class AttackType(Enum):
    """Types of side-channel attacks."""
    CPA = "correlation_power_analysis"
    DPA = "differential_power_analysis"
    TEMPLATE = "template_attack"
    PROFILING = "profiling_attack"
    COLLISION = "collision_attack"
    NEURAL = "neural_attack"


@dataclass
class AnalysisConfig:
    """Configuration for side-channel analysis.
    
    Attributes:
        channel_type: Type of side-channel
        attack_type: Type of attack
        sample_rate: Sampling rate in Hz
        trace_length: Length of traces in samples
        n_traces: Number of traces to collect/analyze
        preprocessing: Preprocessing steps to apply
        poi_method: Point-of-interest selection method
        n_pois: Number of points of interest
        neural_operator: Neural operator configuration
        device: Computing device
    """
    channel_type: ChannelType = ChannelType.POWER
    attack_type: AttackType = AttackType.NEURAL
    sample_rate: float = 1e6  # 1 MHz
    trace_length: int = 10000
    n_traces: int = 10000
    preprocessing: List[str] = field(default_factory=lambda: ['standardize'])
    poi_method: str = 'mutual_information'
    n_pois: int = 100
    neural_operator: Dict[str, Any] = field(default_factory=dict)
    device: str = 'cpu'
    
    # Analysis parameters
    confidence_threshold: float = 0.9
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    
    # Security parameters
    leakage_assessment: bool = True
    statistical_tests: List[str] = field(default_factory=lambda: ['t_test', 'chi2'])


class LeakageModel(nn.Module):
    """Base class for leakage models."""
    
    def __init__(self, model_type: str = 'hamming_weight'):
        super().__init__()
        self.model_type = model_type
        
    def forward(self, intermediate_value: torch.Tensor) -> torch.Tensor:
        """Compute leakage for intermediate values.
        
        Args:
            intermediate_value: Cryptographic intermediate values
            
        Returns:
            Expected leakage values
        """
        if self.model_type == 'hamming_weight':
            return self.hamming_weight(intermediate_value)
        elif self.model_type == 'hamming_distance':
            return self.hamming_distance(intermediate_value)
        elif self.model_type == 'identity':
            return intermediate_value.float()
        else:
            raise ValueError(f"Unknown leakage model: {self.model_type}")
    
    def hamming_weight(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Hamming weight of values."""
        if x.dtype != torch.uint8:
            x = x.byte()
        
        # Count bits using bit manipulation
        hw = torch.zeros_like(x, dtype=torch.float32)
        for i in range(8):
            hw += (x >> i) & 1
        
        return hw
    
    def hamming_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute Hamming distance between values."""
        return self.hamming_weight(x1 ^ x2)


class TraceData:
    """Container for side-channel traces and metadata."""
    
    def __init__(self, traces: np.ndarray, 
                 metadata: Optional[Dict[str, Any]] = None,
                 labels: Optional[np.ndarray] = None,
                 plaintexts: Optional[np.ndarray] = None,
                 keys: Optional[np.ndarray] = None):
        self.traces = traces
        self.metadata = metadata or {}
        self.labels = labels
        self.plaintexts = plaintexts
        self.keys = keys
        
        # Validate data consistency
        self._validate_data()
    
    def _validate_data(self):
        """Validate data consistency."""
        n_traces = len(self.traces)
        
        if self.labels is not None and len(self.labels) != n_traces:
            raise ValueError("Labels length must match traces length")
            
        if self.plaintexts is not None and len(self.plaintexts) != n_traces:
            raise ValueError("Plaintexts length must match traces length")
            
        if self.keys is not None and len(self.keys) != n_traces:
            raise ValueError("Keys length must match traces length")
    
    def __len__(self) -> int:
        return len(self.traces)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        """Get trace data by index."""
        item = {'trace': self.traces[idx]}
        
        if self.labels is not None:
            item['label'] = self.labels[idx]
        if self.plaintexts is not None:
            item['plaintext'] = self.plaintexts[idx]
        if self.keys is not None:
            item['key'] = self.keys[idx]
            
        return item
    
    def split(self, train_ratio: float = 0.8) -> Tuple['TraceData', 'TraceData']:
        """Split data into training and testing sets."""
        n_train = int(len(self) * train_ratio)
        indices = np.random.permutation(len(self))
        
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        train_data = TraceData(
            traces=self.traces[train_idx],
            metadata=self.metadata.copy(),
            labels=self.labels[train_idx] if self.labels is not None else None,
            plaintexts=self.plaintexts[train_idx] if self.plaintexts is not None else None,
            keys=self.keys[train_idx] if self.keys is not None else None
        )
        
        test_data = TraceData(
            traces=self.traces[test_idx],
            metadata=self.metadata.copy(),
            labels=self.labels[test_idx] if self.labels is not None else None,
            plaintexts=self.plaintexts[test_idx] if self.plaintexts is not None else None,
            keys=self.keys[test_idx] if self.keys is not None else None
        )
        
        return train_data, test_data


class SideChannelAnalyzer(ABC):
    """Abstract base class for side-channel analyzers."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.leakage_model = LeakageModel()
        
        # Analysis state
        self.is_trained = False
        self.training_history = []
        self.attack_results = {}
        
    @abstractmethod
    def preprocess_traces(self, traces: np.ndarray) -> np.ndarray:
        """Preprocess raw traces.
        
        Args:
            traces: Raw trace data
            
        Returns:
            Preprocessed traces
        """
        pass
    
    @abstractmethod
    def extract_features(self, traces: np.ndarray) -> np.ndarray:
        """Extract features from traces.
        
        Args:
            traces: Preprocessed traces
            
        Returns:
            Feature vectors
        """
        pass
    
    @abstractmethod
    def train(self, data: TraceData) -> Dict[str, Any]:
        """Train the analyzer on trace data.
        
        Args:
            data: Training trace data
            
        Returns:
            Training results and metrics
        """
        pass
    
    @abstractmethod
    def attack(self, data: TraceData) -> Dict[str, Any]:
        """Perform attack on trace data.
        
        Args:
            data: Attack trace data
            
        Returns:
            Attack results and key recovery metrics
        """
        pass
    
    def select_pois(self, traces: np.ndarray, 
                   intermediate_values: np.ndarray) -> np.ndarray:
        """Select points of interest from traces.
        
        Args:
            traces: Trace data
            intermediate_values: Intermediate cryptographic values
            
        Returns:
            Indices of selected points of interest
        """
        if self.config.poi_method == 'mutual_information':
            return self._select_pois_mi(traces, intermediate_values)
        elif self.config.poi_method == 'correlation':
            return self._select_pois_correlation(traces, intermediate_values)
        elif self.config.poi_method == 'variance':
            return self._select_pois_variance(traces)
        else:
            # Uniform sampling as fallback
            return np.linspace(0, traces.shape[1]-1, self.config.n_pois, dtype=int)
    
    def _select_pois_mi(self, traces: np.ndarray, 
                       intermediate_values: np.ndarray) -> np.ndarray:
        """Select POIs using mutual information."""
        from sklearn.feature_selection import mutual_info_regression
        
        mi_scores = []
        for i in range(traces.shape[1]):
            mi = mutual_info_regression(
                traces[:, i:i+1], 
                intermediate_values.ravel()
            )[0]
            mi_scores.append(mi)
        
        # Select top POIs
        poi_indices = np.argsort(mi_scores)[-self.config.n_pois:]
        return np.sort(poi_indices)
    
    def _select_pois_correlation(self, traces: np.ndarray,
                               intermediate_values: np.ndarray) -> np.ndarray:
        """Select POIs using correlation analysis."""
        correlations = []
        leakage = self.leakage_model(torch.tensor(intermediate_values)).numpy()
        
        for i in range(traces.shape[1]):
            corr = np.corrcoef(traces[:, i], leakage)[0, 1]
            correlations.append(abs(corr))
        
        poi_indices = np.argsort(correlations)[-self.config.n_pois:]
        return np.sort(poi_indices)
    
    def _select_pois_variance(self, traces: np.ndarray) -> np.ndarray:
        """Select POIs with highest variance."""
        variances = np.var(traces, axis=0)
        poi_indices = np.argsort(variances)[-self.config.n_pois:]
        return np.sort(poi_indices)
    
    def assess_leakage(self, traces: np.ndarray,
                      intermediate_values: np.ndarray) -> Dict[str, float]:
        """Assess information leakage in traces.
        
        Args:
            traces: Trace data
            intermediate_values: Intermediate values
            
        Returns:
            Leakage assessment metrics
        """
        results = {}
        
        # Signal-to-noise ratio
        leakage = self.leakage_model(torch.tensor(intermediate_values)).numpy()
        signal_power = np.var(leakage)
        noise_power = np.mean(np.var(traces, axis=0))
        results['snr'] = signal_power / noise_power if noise_power > 0 else float('inf')
        
        # Mutual information
        from sklearn.feature_selection import mutual_info_regression
        mi_per_sample = []
        for i in range(0, traces.shape[1], max(1, traces.shape[1] // 100)):
            mi = mutual_info_regression(
                traces[:, i:i+1],
                intermediate_values.ravel()
            )[0]
            mi_per_sample.append(mi)
        
        results['max_mutual_info'] = max(mi_per_sample)
        results['avg_mutual_info'] = np.mean(mi_per_sample)
        
        # Correlation analysis
        max_correlation = 0
        for i in range(0, traces.shape[1], max(1, traces.shape[1] // 100)):
            corr = abs(np.corrcoef(traces[:, i], leakage)[0, 1])
            if not np.isnan(corr):
                max_correlation = max(max_correlation, corr)
        
        results['max_correlation'] = max_correlation
        
        return results
    
    def statistical_test(self, group1: np.ndarray, 
                        group2: np.ndarray,
                        test_type: str = 't_test') -> Dict[str, float]:
        """Perform statistical test for leakage detection.
        
        Args:
            group1: First group of traces
            group2: Second group of traces
            test_type: Type of statistical test
            
        Returns:
            Test statistics and p-values
        """
        from scipy import stats
        
        if test_type == 't_test':
            statistic, p_value = stats.ttest_ind(
                group1.reshape(group1.shape[0], -1),
                group2.reshape(group2.shape[0], -1),
                axis=0
            )
            return {
                'max_t_statistic': np.max(np.abs(statistic)),
                'min_p_value': np.min(p_value),
                'significant_samples': np.sum(p_value < 0.05)
            }
        elif test_type == 'chi2':
            # Chi-square test for categorical data
            combined = np.vstack([group1, group2])
            chi2_stats = []
            p_values = []
            
            for i in range(combined.shape[1]):
                # Bin the data
                bins = np.linspace(combined[:, i].min(), combined[:, i].max(), 10)
                obs1, _ = np.histogram(group1[:, i], bins=bins)
                obs2, _ = np.histogram(group2[:, i], bins=bins)
                
                # Avoid empty bins
                if np.sum(obs1) > 0 and np.sum(obs2) > 0:
                    chi2, p = stats.chi2_contingency([obs1, obs2])[:2]
                    chi2_stats.append(chi2)
                    p_values.append(p)
            
            return {
                'max_chi2_statistic': np.max(chi2_stats) if chi2_stats else 0,
                'min_p_value': np.min(p_values) if p_values else 1,
                'significant_samples': np.sum(np.array(p_values) < 0.05)
            }
        else:
            raise ValueError(f"Unknown test type: {test_type}")
    
    def get_success_rate(self, predictions: np.ndarray, 
                        true_values: np.ndarray,
                        top_k: int = 1) -> float:
        """Calculate attack success rate.
        
        Args:
            predictions: Predicted key bytes (probabilities)
            true_values: True key bytes
            top_k: Consider top-k predictions as success
            
        Returns:
            Success rate [0, 1]
        """
        if len(predictions.shape) == 1:
            # Single prediction per sample
            return np.mean(predictions == true_values)
        else:
            # Probability predictions
            top_predictions = np.argsort(predictions, axis=1)[:, -top_k:]
            success = np.any(top_predictions == true_values.reshape(-1, 1), axis=1)
            return np.mean(success)
    
    def save_model(self, filepath: str):
        """Save trained model."""
        if hasattr(self, 'model') and self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'training_history': self.training_history,
                'attack_results': self.attack_results
            }, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.config = checkpoint['config']
        self.training_history = checkpoint['training_history']
        self.attack_results = checkpoint['attack_results']
        
        if hasattr(self, 'model') and self.model is not None:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.is_trained = True