"""Electromagnetic (EM) analysis implementations for side-channel attacks."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy import signal, spatial
from sklearn.decomposition import PCA

from .base import SideChannelAnalyzer, AnalysisConfig, TraceData, LeakageModel
from ..neural_operators import NeuralOperatorBase, FourierNeuralOperator, SideChannelFNO


class EMAnalyzer(SideChannelAnalyzer):
    """Base electromagnetic analysis implementation."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        
        # EM-specific configuration
        self.frequency_bands = self._setup_frequency_bands()
        self.spatial_resolution = 1.0  # mm
        
        # Initialize neural operator for EM analysis
        from ..neural_operators import OperatorConfig
        op_config = OperatorConfig(
            input_dim=1,
            output_dim=256,
            hidden_dim=config.neural_operator.get('hidden_dim', 128),
            num_layers=config.neural_operator.get('num_layers', 6),
            device=config.device
        )
        
        self.neural_operator = EMNeuralOperator(
            op_config,
            frequency_bands=self.frequency_bands,
            modes=config.neural_operator.get('modes', 32)
        ).to(self.device)
        
        # EM-specific leakage model
        self.leakage_model = EMLeakageModel()
        
    def _setup_frequency_bands(self) -> List[Tuple[float, float]]:
        """Setup frequency bands for EM analysis."""
        return [
            (1e6, 10e6),    # 1-10 MHz
            (10e6, 100e6),  # 10-100 MHz  
            (100e6, 1e9),   # 100 MHz - 1 GHz
            (1e9, 5e9),     # 1-5 GHz
        ]
    
    def preprocess_traces(self, traces: np.ndarray) -> np.ndarray:
        """Preprocess EM traces with spectral analysis."""
        processed = traces.copy()
        
        for method in self.config.preprocessing:
            if method == 'spectral_filtering':
                processed = self._apply_spectral_filtering(processed)
            elif method == 'spatial_averaging':
                processed = self._apply_spatial_averaging(processed)
            elif method == 'phase_alignment':
                processed = self._align_phase(processed)
            elif method == 'standardize':
                processed = (processed - np.mean(processed, axis=1, keepdims=True)) / \
                           (np.std(processed, axis=1, keepdims=True) + 1e-8)
        
        return processed
    
    def _apply_spectral_filtering(self, traces: np.ndarray) -> np.ndarray:
        """Apply frequency-selective filtering."""
        filtered_traces = []
        
        for trace in traces:
            # Apply FFT
            fft_trace = np.fft.fft(trace)
            freqs = np.fft.fftfreq(len(trace), 1/self.config.sample_rate)
            
            # Filter each frequency band
            filtered_components = []
            for low_freq, high_freq in self.frequency_bands:
                mask = (np.abs(freqs) >= low_freq) & (np.abs(freqs) <= high_freq)
                band_component = np.fft.ifft(fft_trace * mask)
                filtered_components.append(np.real(band_component))
            
            # Combine filtered components
            filtered_trace = np.sum(filtered_components, axis=0)
            filtered_traces.append(filtered_trace)
        
        return np.array(filtered_traces)
    
    def _apply_spatial_averaging(self, traces: np.ndarray, 
                                kernel_size: int = 5) -> np.ndarray:
        """Apply spatial averaging to reduce noise."""
        # Simple moving average filter
        kernel = np.ones(kernel_size) / kernel_size
        averaged_traces = []
        
        for trace in traces:
            averaged_trace = np.convolve(trace, kernel, mode='same')
            averaged_traces.append(averaged_trace)
        
        return np.array(averaged_traces)
    
    def _align_phase(self, traces: np.ndarray) -> np.ndarray:
        """Align phases of EM signals."""
        reference_fft = np.fft.fft(traces[0])
        reference_phase = np.angle(reference_fft)
        
        aligned_traces = [traces[0]]  # Reference trace
        
        for trace in traces[1:]:
            trace_fft = np.fft.fft(trace)
            trace_magnitude = np.abs(trace_fft)
            
            # Use reference phase
            aligned_fft = trace_magnitude * np.exp(1j * reference_phase)
            aligned_trace = np.real(np.fft.ifft(aligned_fft))
            aligned_traces.append(aligned_trace)
        
        return np.array(aligned_traces)
    
    def extract_features(self, traces: np.ndarray) -> np.ndarray:
        """Extract EM-specific features."""
        features = []
        
        for trace in traces:
            # Spectral features
            fft_trace = np.fft.fft(trace)
            power_spectrum = np.abs(fft_trace) ** 2
            freqs = np.fft.fftfreq(len(trace), 1/self.config.sample_rate)
            
            # Power in different frequency bands
            band_powers = []
            for low_freq, high_freq in self.frequency_bands:
                mask = (np.abs(freqs) >= low_freq) & (np.abs(freqs) <= high_freq)
                band_power = np.sum(power_spectrum[mask])
                band_powers.append(band_power)
            
            # Statistical features
            mean_power = np.mean(power_spectrum)
            std_power = np.std(power_spectrum)
            peak_freq_idx = np.argmax(power_spectrum[:len(power_spectrum)//2])
            peak_freq = freqs[peak_freq_idx]
            
            # Phase coherence
            phase = np.angle(fft_trace)
            phase_coherence = np.abs(np.mean(np.exp(1j * phase)))
            
            # Combine features
            feature_vector = band_powers + [
                mean_power, std_power, peak_freq, phase_coherence
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def train(self, data: TraceData) -> Dict[str, Any]:
        """Train EM neural operator."""
        # Preprocess traces
        preprocessed_traces = self.preprocess_traces(data.traces)
        
        # Convert to tensors
        traces_tensor = torch.tensor(preprocessed_traces, dtype=torch.float32).to(self.device)
        labels_tensor = torch.tensor(data.labels, dtype=torch.long).to(self.device)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(traces_tensor, labels_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=True
        )
        
        # Training setup
        optimizer = torch.optim.Adam(self.neural_operator.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # Training loop
        self.neural_operator.train()
        training_history = []
        
        for epoch in range(150):  # More epochs for EM analysis
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
            scheduler.step(avg_loss)
            
            training_history.append({
                'epoch': epoch,
                'loss': avg_loss,
                'accuracy': accuracy,
                'lr': optimizer.param_groups[0]['lr']
            })
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
        
        self.is_trained = True
        self.training_history = training_history
        
        return {
            'final_loss': training_history[-1]['loss'],
            'final_accuracy': training_history[-1]['accuracy'],
            'training_history': training_history
        }
    
    def attack(self, data: TraceData) -> Dict[str, Any]:
        """Perform EM attack."""
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
        
        # Calculate success rate
        success_rate = 0
        if data.labels is not None:
            true_labels = torch.tensor(data.labels, dtype=torch.long).to(self.device)
            success_rate = (predicted_keys == true_labels).float().mean().item()
        
        return {
            'predictions': predicted_keys.cpu().numpy(),
            'probabilities': probabilities.cpu().numpy(),
            'confidences': confidences.cpu().numpy(),
            'success_rate': success_rate,
            'frequency_analysis': self._analyze_frequency_contributions(traces_tensor)
        }
    
    def _analyze_frequency_contributions(self, traces: torch.Tensor) -> Dict[str, float]:
        """Analyze contribution of different frequency bands."""
        contributions = {}
        
        with torch.no_grad():
            full_prediction = self.neural_operator(traces)
            
            for i, (low_freq, high_freq) in enumerate(self.frequency_bands):
                # Filter traces to specific band
                filtered_traces = self._filter_to_band(traces, low_freq, high_freq)
                band_prediction = self.neural_operator(filtered_traces)
                
                # Measure prediction similarity
                similarity = torch.cosine_similarity(
                    full_prediction.flatten(),
                    band_prediction.flatten(),
                    dim=0
                ).item()
                
                contributions[f"band_{i+1}_{low_freq/1e6:.0f}-{high_freq/1e6:.0f}MHz"] = similarity
        
        return contributions
    
    def _filter_to_band(self, traces: torch.Tensor, low_freq: float, 
                       high_freq: float) -> torch.Tensor:
        """Filter traces to specific frequency band."""
        filtered_traces = []
        
        for trace in traces:
            trace_np = trace.cpu().numpy()
            fft_trace = np.fft.fft(trace_np)
            freqs = np.fft.fftfreq(len(trace_np), 1/self.config.sample_rate)
            
            # Apply band filter
            mask = (np.abs(freqs) >= low_freq) & (np.abs(freqs) <= high_freq)
            filtered_fft = fft_trace * mask
            filtered_trace = np.real(np.fft.ifft(filtered_fft))
            
            filtered_traces.append(torch.tensor(filtered_trace, dtype=torch.float32))
        
        return torch.stack(filtered_traces).to(self.device)


class NearFieldEM(EMAnalyzer):
    """Near-field EM analysis with spatial modeling."""
    
    def __init__(self, config: AnalysisConfig, probe_distance: float = 1.0):
        super().__init__(config)
        self.probe_distance = probe_distance  # mm
        self.spatial_model = None
        
    def build_spatial_model(self, positions: np.ndarray, 
                           measurements: np.ndarray) -> None:
        """Build spatial model of EM field."""
        self.spatial_model = {
            'positions': positions,
            'field_map': self._interpolate_field(positions, measurements)
        }
    
    def _interpolate_field(self, positions: np.ndarray, 
                          measurements: np.ndarray) -> callable:
        """Create interpolation function for EM field."""
        from scipy.interpolate import griddata
        
        def field_interpolator(query_points):
            return griddata(positions, measurements, query_points, method='cubic')
        
        return field_interpolator
    
    def optimize_probe_position(self, initial_measurements: np.ndarray,
                               positions: np.ndarray) -> np.ndarray:
        """Find optimal probe position for maximum SNR."""
        snr_scores = []
        
        for i, measurement in enumerate(initial_measurements):
            # Compute SNR for this position
            signal_power = np.var(measurement)
            noise_floor = np.percentile(np.abs(measurement), 10)
            snr = signal_power / (noise_floor ** 2) if noise_floor > 0 else 0
            snr_scores.append(snr)
        
        # Find position with maximum SNR
        best_idx = np.argmax(snr_scores)
        return positions[best_idx]


class FarFieldEM(EMAnalyzer):
    """Far-field EM analysis for distant measurements."""
    
    def __init__(self, config: AnalysisConfig, distance: float = 100.0):
        super().__init__(config)
        self.distance = distance  # meters
        
        # Far-field specific frequency bands (higher frequencies)
        self.frequency_bands = [
            (100e6, 500e6),   # VHF
            (500e6, 2e9),     # UHF
            (2e9, 6e9),       # S-band
            (6e9, 18e9),      # C/X-band
        ]
        
    def preprocess_traces(self, traces: np.ndarray) -> np.ndarray:
        """Far-field specific preprocessing."""
        # Apply distance-based attenuation model
        attenuation_factor = 1 / (4 * np.pi * self.distance) ** 2
        attenuated_traces = traces * attenuation_factor
        
        # Apply atmospheric effects (simplified)
        atmospheric_factor = self._compute_atmospheric_attenuation()
        compensated_traces = attenuated_traces / atmospheric_factor
        
        # Standard preprocessing
        return super().preprocess_traces(compensated_traces)
    
    def _compute_atmospheric_attenuation(self) -> float:
        """Compute atmospheric attenuation factor."""
        # Simplified atmospheric model
        # In reality, this would depend on frequency, humidity, etc.
        frequency = np.mean([band[0] for band in self.frequency_bands])
        
        if frequency < 1e9:
            return 1.0  # Minimal attenuation below 1 GHz
        else:
            # Approximate attenuation in dB/km converted to linear scale
            attenuation_db_per_km = 0.1 * (frequency / 1e9)
            attenuation_db = attenuation_db_per_km * (self.distance / 1000)
            return 10 ** (attenuation_db / 20)


class EMLeakageModel(LeakageModel):
    """EM-specific leakage model."""
    
    def __init__(self, model_type: str = 'current_flow'):
        super().__init__(model_type)
        
    def forward(self, intermediate_value: torch.Tensor) -> torch.Tensor:
        """Model EM leakage based on current flow."""
        if self.model_type == 'current_flow':
            # EM leakage is proportional to rate of change of current
            hw = self.hamming_weight(intermediate_value)
            # Approximate current change with Hamming weight difference
            return hw * 0.1  # Scaling factor
        elif self.model_type == 'switching_activity':
            # Model based on bit transitions
            return self.hamming_distance(intermediate_value, 
                                       torch.roll(intermediate_value, 1, dims=0))
        else:
            return super().forward(intermediate_value)


class EMNeuralOperator(nn.Module):
    """Specialized neural operator for EM analysis."""
    
    def __init__(self, config, frequency_bands: List[Tuple[float, float]], 
                 modes: int = 32):
        super().__init__()
        self.config = config
        self.frequency_bands = frequency_bands
        self.modes = modes
        
        # Multi-frequency processing branches
        self.frequency_branches = nn.ModuleList([
            FrequencyBranch(config.input_dim, config.hidden_dim, modes // len(frequency_bands))
            for _ in frequency_bands
        ])
        
        # Cross-frequency attention
        self.cross_freq_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * len(frequency_bands), 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, config.output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process EM traces through multi-frequency analysis."""
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        
        # Process each frequency band
        branch_outputs = []
        for branch in self.frequency_branches:
            branch_out = branch(x)
            branch_outputs.append(branch_out)
        
        # Stack for cross-attention
        stacked_features = torch.stack(branch_outputs, dim=1)
        
        # Apply cross-frequency attention
        attended_features, _ = self.cross_freq_attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # Flatten for classification
        flattened = attended_features.flatten(start_dim=1)
        
        # Final classification
        output = self.classifier(flattened)
        
        return output


class FrequencyBranch(nn.Module):
    """Processing branch for specific frequency band."""
    
    def __init__(self, input_dim: int, hidden_dim: int, modes: int):
        super().__init__()
        
        self.preprocess = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Spectral processing
        from ..neural_operators.fno import SpectralConv1d
        self.spectral_conv = SpectralConv1d(hidden_dim, hidden_dim, modes)
        
        # Local processing
        self.local_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through frequency-specific branch."""
        # Preprocess
        x = self.preprocess(x)  # [batch, length, hidden_dim]
        
        # Spectral processing
        x_perm = x.permute(0, 2, 1)  # [batch, hidden_dim, length]
        x_spectral = self.spectral_conv(x_perm)
        x_local = self.local_conv(x_perm)
        
        # Combine spectral and local
        x_combined = x_spectral + x_local
        x_combined = self.activation(x_combined)
        
        # Back to original format and normalize
        x_combined = x_combined.permute(0, 2, 1)
        x_combined = self.norm(x_combined)
        
        # Global pooling
        return torch.mean(x_combined, dim=1)