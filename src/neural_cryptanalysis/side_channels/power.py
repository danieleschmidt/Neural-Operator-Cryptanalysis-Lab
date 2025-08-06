"""Power analysis implementations for side-channel attacks."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture

from .base import SideChannelAnalyzer, AnalysisConfig, TraceData, LeakageModel
from ..neural_operators import NeuralOperatorBase, FourierNeuralOperator, SideChannelFNO


class PowerAnalyzer(SideChannelAnalyzer):
    """Base power analysis implementation with neural operators."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        
        # Initialize neural operator for power analysis
        from ..neural_operators import OperatorConfig
        op_config = OperatorConfig(
            input_dim=1,
            output_dim=256,
            hidden_dim=config.neural_operator.get('hidden_dim', 64),
            num_layers=config.neural_operator.get('num_layers', 4),
            device=config.device
        )
        
        self.neural_operator = SideChannelFNO(
            op_config,
            modes=config.neural_operator.get('modes', 16),
            trace_length=config.trace_length,
            preprocessing=config.preprocessing[0] if config.preprocessing else 'standardize'
        ).to(self.device)
        
        # Power-specific leakage model
        self.leakage_model = PowerLeakageModel()
        
    def preprocess_traces(self, traces: np.ndarray) -> np.ndarray:
        """Preprocess power traces."""
        processed = traces.copy()
        
        for method in self.config.preprocessing:
            if method == 'standardize':
                processed = (processed - np.mean(processed, axis=1, keepdims=True)) / \
                           (np.std(processed, axis=1, keepdims=True) + 1e-8)
            elif method == 'normalize':
                processed = (processed - np.min(processed, axis=1, keepdims=True)) / \
                           (np.max(processed, axis=1, keepdims=True) - 
                            np.min(processed, axis=1, keepdims=True) + 1e-8)
            elif method == 'filtering':
                processed = self._apply_bandpass_filter(processed)
            elif method == 'alignment':
                processed = self._align_traces(processed)
                
        return processed
    
    def _apply_bandpass_filter(self, traces: np.ndarray, 
                              low_freq: float = 1000, high_freq: float = 100000) -> np.ndarray:
        """Apply bandpass filter to remove noise."""
        from scipy.signal import butter, filtfilt
        
        nyquist = self.config.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        b, a = butter(4, [low, high], btype='band')
        
        filtered_traces = []
        for trace in traces:
            filtered_trace = filtfilt(b, a, trace)
            filtered_traces.append(filtered_trace)
            
        return np.array(filtered_traces)
    
    def _align_traces(self, traces: np.ndarray) -> np.ndarray:
        """Align traces using cross-correlation."""
        reference = traces[0]
        aligned = [reference]
        
        for trace in traces[1:]:
            correlation = np.correlate(trace, reference, mode='full')
            offset = correlation.argmax() - len(reference) + 1
            
            if offset > 0:
                aligned_trace = np.concatenate([trace[offset:], np.zeros(offset)])
            elif offset < 0:
                aligned_trace = np.concatenate([np.zeros(-offset), trace[:offset]])
            else:
                aligned_trace = trace
                
            # Ensure same length
            aligned_trace = aligned_trace[:len(reference)]
            if len(aligned_trace) < len(reference):
                aligned_trace = np.concatenate([aligned_trace, 
                                              np.zeros(len(reference) - len(aligned_trace))])
            
            aligned.append(aligned_trace)
            
        return np.array(aligned)
    
    def extract_features(self, traces: np.ndarray) -> np.ndarray:
        """Extract power-specific features."""
        features = []
        
        for trace in traces:
            # Statistical features
            mean_power = np.mean(trace)
            std_power = np.std(trace)
            max_power = np.max(trace)
            min_power = np.min(trace)
            
            # Spectral features
            fft = np.fft.fft(trace)
            spectral_power = np.abs(fft)[:len(fft)//2]
            dominant_freq = np.argmax(spectral_power)
            spectral_entropy = stats.entropy(spectral_power + 1e-8)
            
            # Temporal features
            peak_indices = self._find_peaks(trace)
            n_peaks = len(peak_indices)
            peak_spacing = np.mean(np.diff(peak_indices)) if n_peaks > 1 else 0
            
            feature_vector = [
                mean_power, std_power, max_power, min_power,
                dominant_freq, spectral_entropy,
                n_peaks, peak_spacing
            ]
            
            features.append(feature_vector)
            
        return np.array(features)
    
    def _find_peaks(self, trace: np.ndarray, threshold: float = None) -> np.ndarray:
        """Find peaks in power trace."""
        if threshold is None:
            threshold = np.mean(trace) + 2 * np.std(trace)
            
        peaks = []
        for i in range(1, len(trace) - 1):
            if trace[i] > threshold and trace[i] > trace[i-1] and trace[i] > trace[i+1]:
                peaks.append(i)
                
        return np.array(peaks)
    
    def train(self, data: TraceData) -> Dict[str, Any]:
        """Train neural operator on power traces."""
        # Preprocess traces
        preprocessed_traces = self.preprocess_traces(data.traces)
        
        # Convert to tensors
        traces_tensor = torch.tensor(preprocessed_traces, dtype=torch.float32).to(self.device)
        labels_tensor = torch.tensor(data.labels, dtype=torch.long).to(self.device)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(traces_tensor, labels_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=64, shuffle=True
        )
        
        # Training setup
        optimizer = torch.optim.Adam(self.neural_operator.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.neural_operator.train()
        training_history = []
        
        for epoch in range(100):  # Fixed epochs for simplicity
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_traces, batch_labels in dataloader:
                optimizer.zero_grad()
                
                outputs = self.neural_operator(batch_traces)
                loss = criterion(outputs, batch_labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
            
            accuracy = 100 * correct / total
            avg_loss = total_loss / len(dataloader)
            
            training_history.append({
                'epoch': epoch,
                'loss': avg_loss,
                'accuracy': accuracy
            })
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
        
        self.is_trained = True
        self.training_history = training_history
        
        return {
            'final_loss': training_history[-1]['loss'],
            'final_accuracy': training_history[-1]['accuracy'],
            'training_history': training_history
        }
    
    def attack(self, data: TraceData) -> Dict[str, Any]:
        """Perform neural power attack."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before attacking")
        
        # Preprocess traces
        preprocessed_traces = self.preprocess_traces(data.traces)
        traces_tensor = torch.tensor(preprocessed_traces, dtype=torch.float32).to(self.device)
        
        # Perform attack
        self.neural_operator.eval()
        with torch.no_grad():
            predictions = self.neural_operator(traces_tensor)
            probabilities = torch.softmax(predictions, dim=1)
            predicted_keys = torch.argmax(probabilities, dim=1)
            confidences = torch.max(probabilities, dim=1)[0]
        
        # Calculate success rate if true labels available
        success_rate = 0
        if data.labels is not None:
            true_labels = torch.tensor(data.labels, dtype=torch.long).to(self.device)
            success_rate = (predicted_keys == true_labels).float().mean().item()
        
        results = {
            'predictions': predicted_keys.cpu().numpy(),
            'probabilities': probabilities.cpu().numpy(),
            'confidences': confidences.cpu().numpy(),
            'success_rate': success_rate,
            'avg_confidence': confidences.mean().item()
        }
        
        self.attack_results = results
        return results


class CPAnalyzer(PowerAnalyzer):
    """Correlation Power Analysis with neural operators."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.correlation_threshold = 0.1
        
    def train(self, data: TraceData) -> Dict[str, Any]:
        """Train using correlation-guided neural operator learning."""
        # First perform traditional CPA to identify POIs
        pois = self._perform_traditional_cpa(data.traces, data.labels)
        
        # Focus neural operator on high-correlation regions
        focused_traces = data.traces[:, pois]
        focused_data = TraceData(
            traces=focused_traces,
            labels=data.labels,
            metadata=data.metadata
        )
        
        # Train neural operator on focused data
        return super().train(focused_data)
    
    def _perform_traditional_cpa(self, traces: np.ndarray, 
                                labels: np.ndarray) -> np.ndarray:
        """Perform traditional CPA to find points of interest."""
        correlations = []
        
        # Compute Hamming weight of labels
        hw_labels = np.array([bin(label).count('1') for label in labels])
        
        # Compute correlation for each time point
        for i in range(traces.shape[1]):
            corr = np.corrcoef(traces[:, i], hw_labels)[0, 1]
            correlations.append(abs(corr) if not np.isnan(corr) else 0)
        
        # Select points above threshold
        correlations = np.array(correlations)
        pois = np.where(correlations > self.correlation_threshold)[0]
        
        # If no POIs found, select top percentile
        if len(pois) == 0:
            threshold = np.percentile(correlations, 90)
            pois = np.where(correlations >= threshold)[0]
        
        return pois


class DPAnalyzer(PowerAnalyzer):
    """Differential Power Analysis with neural enhancement."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.bit_position = 0  # Target bit position
        
    def train(self, data: TraceData) -> Dict[str, Any]:
        """Train using differential power analysis principles."""
        # Partition traces based on target bit
        group0_traces, group1_traces = self._partition_traces(data.traces, data.labels)
        
        # Compute differential trace
        differential_trace = np.mean(group1_traces, axis=0) - np.mean(group0_traces, axis=0)
        
        # Find POIs from differential trace
        pois = self._find_differential_pois(differential_trace)
        
        # Create augmented training data with differential features
        augmented_traces = self._create_augmented_traces(data.traces, differential_trace, pois)
        augmented_data = TraceData(
            traces=augmented_traces,
            labels=data.labels,
            metadata=data.metadata
        )
        
        return super().train(augmented_data)
    
    def _partition_traces(self, traces: np.ndarray, 
                         labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Partition traces based on target bit."""
        bit_values = np.array([(label >> self.bit_position) & 1 for label in labels])
        
        group0_indices = np.where(bit_values == 0)[0]
        group1_indices = np.where(bit_values == 1)[0]
        
        return traces[group0_indices], traces[group1_indices]
    
    def _find_differential_pois(self, differential_trace: np.ndarray) -> np.ndarray:
        """Find points of interest from differential trace."""
        # Find peaks in absolute differential trace
        abs_diff = np.abs(differential_trace)
        threshold = np.mean(abs_diff) + 2 * np.std(abs_diff)
        
        pois = []
        for i in range(1, len(abs_diff) - 1):
            if (abs_diff[i] > threshold and 
                abs_diff[i] > abs_diff[i-1] and 
                abs_diff[i] > abs_diff[i+1]):
                pois.append(i)
        
        return np.array(pois) if pois else np.array([np.argmax(abs_diff)])
    
    def _create_augmented_traces(self, traces: np.ndarray, 
                               differential_trace: np.ndarray,
                               pois: np.ndarray) -> np.ndarray:
        """Create augmented traces with differential features."""
        # Original traces + differential features at POIs
        diff_features = np.tile(differential_trace[pois], (len(traces), 1))
        
        # Combine original traces with differential features
        augmented = np.concatenate([traces, diff_features], axis=1)
        
        return augmented


class PowerLeakageModel(LeakageModel):
    """Specialized leakage model for power consumption."""
    
    def __init__(self, model_type: str = 'hamming_weight_extended'):
        super().__init__(model_type)
        
        # Power-specific parameters
        self.base_power = 0.1  # Base power consumption
        self.switching_factor = 0.01  # Power per bit switch
        
    def forward(self, intermediate_value: torch.Tensor) -> torch.Tensor:
        """Compute power leakage for intermediate values."""
        if self.model_type == 'hamming_weight_extended':
            hw = self.hamming_weight(intermediate_value)
            # Add base power and scaling
            return self.base_power + self.switching_factor * hw
        else:
            return super().forward(intermediate_value)
    
    def power_model_with_noise(self, intermediate_value: torch.Tensor,
                              noise_std: float = 0.01) -> torch.Tensor:
        """Power model with realistic noise."""
        clean_power = self.forward(intermediate_value)
        noise = torch.normal(0, noise_std, size=clean_power.shape)
        return clean_power + noise.to(clean_power.device)


class AdvancedPowerAnalyzer(PowerAnalyzer):
    """Advanced power analyzer with multiple attack strategies."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        
        # Multiple neural operators for different strategies
        self.analyzers = {
            'cpa': CPAnalyzer(config),
            'dpa': DPAnalyzer(config),
            'template': TemplateBasedAnalyzer(config)
        }
        
    def train(self, data: TraceData) -> Dict[str, Any]:
        """Train multiple analyzers and ensemble."""
        results = {}
        
        for name, analyzer in self.analyzers.items():
            print(f"Training {name} analyzer...")
            analyzer_result = analyzer.train(data)
            results[name] = analyzer_result
            
        self.is_trained = True
        return results
    
    def attack(self, data: TraceData) -> Dict[str, Any]:
        """Ensemble attack using multiple strategies."""
        if not self.is_trained:
            raise RuntimeError("Analyzers must be trained before attacking")
        
        # Get predictions from each analyzer
        predictions = {}
        confidences = {}
        
        for name, analyzer in self.analyzers.items():
            result = analyzer.attack(data)
            predictions[name] = result['predictions']
            confidences[name] = result['confidences']
        
        # Ensemble voting
        ensemble_predictions = self._ensemble_vote(predictions, confidences)
        
        # Calculate ensemble success rate
        success_rate = 0
        if data.labels is not None:
            success_rate = np.mean(ensemble_predictions == data.labels)
        
        return {
            'ensemble_predictions': ensemble_predictions,
            'individual_predictions': predictions,
            'individual_confidences': confidences,
            'success_rate': success_rate
        }
    
    def _ensemble_vote(self, predictions: Dict[str, np.ndarray], 
                      confidences: Dict[str, np.ndarray]) -> np.ndarray:
        """Weighted ensemble voting based on confidence."""
        n_samples = len(next(iter(predictions.values())))
        ensemble_preds = []
        
        for i in range(n_samples):
            # Weighted vote based on confidence
            votes = {}
            for name in predictions.keys():
                pred = predictions[name][i]
                conf = confidences[name][i]
                votes[pred] = votes.get(pred, 0) + conf
            
            # Select prediction with highest weighted vote
            best_pred = max(votes, key=votes.get)
            ensemble_preds.append(best_pred)
        
        return np.array(ensemble_preds)


class TemplateBasedAnalyzer(PowerAnalyzer):
    """Template-based power analysis with neural operators."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.templates = {}
        self.n_components = 5  # Number of Gaussian components per template
        
    def train(self, data: TraceData) -> Dict[str, Any]:
        """Build templates using Gaussian Mixture Models."""
        # Preprocess traces
        preprocessed_traces = self.preprocess_traces(data.traces)
        
        # Select POIs
        pois = self.select_pois(preprocessed_traces, data.labels)
        poi_traces = preprocessed_traces[:, pois]
        
        # Build template for each key byte value
        for key_byte in range(256):
            mask = data.labels == key_byte
            if np.sum(mask) > 0:  # Only if we have traces for this key byte
                byte_traces = poi_traces[mask]
                
                # Fit Gaussian Mixture Model
                gmm = GaussianMixture(n_components=self.n_components)
                gmm.fit(byte_traces)
                
                self.templates[key_byte] = gmm
        
        # Also train neural operator for comparison
        neural_result = super().train(data)
        
        self.is_trained = True
        
        return {
            'n_templates': len(self.templates),
            'n_pois': len(pois),
            'neural_results': neural_result
        }
    
    def attack(self, data: TraceData) -> Dict[str, Any]:
        """Perform template attack."""
        if not self.is_trained:
            raise RuntimeError("Templates must be built before attacking")
        
        # Preprocess traces
        preprocessed_traces = self.preprocess_traces(data.traces)
        
        # Use same POIs as training
        pois = self.select_pois(preprocessed_traces, 
                               np.zeros(len(preprocessed_traces)))  # Dummy labels
        poi_traces = preprocessed_traces[:, pois]
        
        # Template matching
        predictions = []
        confidences = []
        
        for trace in poi_traces:
            best_key = 0
            best_score = float('-inf')
            
            for key_byte, template in self.templates.items():
                score = template.score_samples([trace])[0]
                if score > best_score:
                    best_score = score
                    best_key = key_byte
            
            predictions.append(best_key)
            confidences.append(np.exp(best_score))  # Convert log-likelihood to likelihood
        
        # Calculate success rate
        success_rate = 0
        if data.labels is not None:
            success_rate = np.mean(np.array(predictions) == data.labels)
        
        return {
            'predictions': np.array(predictions),
            'confidences': np.array(confidences),
            'success_rate': success_rate
        }