"""Acoustic side-channel analysis implementations."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy import signal
from scipy.fft import fft, fftfreq

from .base import SideChannelAnalyzer, AnalysisConfig, TraceData


class AcousticAnalyzer(SideChannelAnalyzer):
    """Acoustic side-channel analysis for cryptographic operations."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        
        # Acoustic-specific parameters
        self.frequency_bands = [
            (20, 200),      # Sub-bass
            (200, 500),     # Bass
            (500, 2000),    # Midrange
            (2000, 8000),   # Treble
            (8000, 20000),  # High treble
        ]
        
        self.microphone_config = {
            'sensitivity': -38,  # dBV/Pa
            'max_spl': 130,     # dB SPL
            'frequency_response': 'flat'
        }
        
    def preprocess_traces(self, traces: np.ndarray) -> np.ndarray:
        """Preprocess acoustic traces."""
        processed = traces.copy()
        
        for method in self.config.preprocessing:
            if method == 'bandpass_filter':
                processed = self._apply_acoustic_bandpass(processed)
            elif method == 'noise_reduction':
                processed = self._reduce_environmental_noise(processed)
            elif method == 'amplitude_normalization':
                processed = self._normalize_amplitude(processed)
            elif method == 'spectral_subtraction':
                processed = self._spectral_subtraction(processed)
        
        return processed
    
    def _apply_acoustic_bandpass(self, traces: np.ndarray, 
                                low_freq: float = 500, 
                                high_freq: float = 8000) -> np.ndarray:
        """Apply bandpass filter for cryptographic frequencies."""
        nyquist = self.config.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        sos = signal.butter(6, [low, high], btype='band', output='sos')
        filtered_traces = []
        
        for trace in traces:
            filtered = signal.sosfilt(sos, trace)
            filtered_traces.append(filtered)
        
        return np.array(filtered_traces)
    
    def _reduce_environmental_noise(self, traces: np.ndarray) -> np.ndarray:
        """Reduce environmental noise using spectral gating."""
        # Estimate noise floor from quiet periods
        noise_threshold = np.percentile(np.abs(traces), 10)
        
        denoised_traces = []
        for trace in traces:
            # Apply spectral gating
            fft_trace = fft(trace)
            magnitude = np.abs(fft_trace)
            phase = np.angle(fft_trace)
            
            # Gate frequencies below noise threshold
            gated_magnitude = np.where(magnitude > noise_threshold * 2, 
                                     magnitude, magnitude * 0.1)
            
            # Reconstruct signal
            gated_fft = gated_magnitude * np.exp(1j * phase)
            denoised_trace = np.real(np.fft.ifft(gated_fft))
            denoised_traces.append(denoised_trace)
        
        return np.array(denoised_traces)
    
    def extract_features(self, traces: np.ndarray) -> np.ndarray:
        """Extract acoustic-specific features."""
        features = []
        
        for trace in traces:
            feature_vector = []
            
            # Time domain features
            feature_vector.extend([
                np.mean(np.abs(trace)),        # RMS amplitude
                np.std(trace),                 # Standard deviation
                np.max(np.abs(trace)),         # Peak amplitude
                len(self._find_acoustic_events(trace))  # Event count
            ])
            
            # Frequency domain features
            fft_trace = fft(trace)
            freqs = fftfreq(len(trace), 1/self.config.sample_rate)
            power_spectrum = np.abs(fft_trace) ** 2
            
            # Power in frequency bands
            for low_freq, high_freq in self.frequency_bands:
                mask = (freqs >= low_freq) & (freqs <= high_freq)
                band_power = np.sum(power_spectrum[mask])
                feature_vector.append(band_power)
            
            # Spectral features
            feature_vector.extend([
                np.sum(power_spectrum),                    # Total power
                freqs[np.argmax(power_spectrum)],         # Dominant frequency
                np.sum(freqs * power_spectrum) / np.sum(power_spectrum),  # Spectral centroid
                self._compute_spectral_rolloff(freqs, power_spectrum)      # Spectral rolloff
            ])
            
            # Cepstral features
            cepstrum = np.fft.ifft(np.log(power_spectrum + 1e-10))
            feature_vector.extend([
                np.real(cepstrum[1]),  # First cepstral coefficient
                np.real(cepstrum[2]),  # Second cepstral coefficient
            ])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _find_acoustic_events(self, trace: np.ndarray, 
                             threshold_factor: float = 3.0) -> List[int]:
        """Find significant acoustic events in trace."""
        # Detect events using energy-based threshold
        energy = np.abs(trace) ** 2
        threshold = np.mean(energy) + threshold_factor * np.std(energy)
        
        events = []
        above_threshold = energy > threshold
        
        # Find rising edges (start of events)
        for i in range(1, len(above_threshold)):
            if above_threshold[i] and not above_threshold[i-1]:
                events.append(i)
        
        return events
    
    def _compute_spectral_rolloff(self, freqs: np.ndarray, 
                                 power_spectrum: np.ndarray, 
                                 percentile: float = 0.85) -> float:
        """Compute spectral rolloff frequency."""
        cumulative_power = np.cumsum(power_spectrum)
        total_power = cumulative_power[-1]
        rolloff_threshold = percentile * total_power
        
        rolloff_idx = np.where(cumulative_power >= rolloff_threshold)[0]
        if len(rolloff_idx) > 0:
            return freqs[rolloff_idx[0]]
        else:
            return freqs[-1]
    
    def train(self, data: TraceData) -> Dict[str, Any]:
        """Train acoustic analyzer."""
        # Extract acoustic features
        features = self.extract_features(data.traces)
        
        # Use traditional ML for acoustic analysis
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, accuracy_score
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, data.labels, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train classifier
        self.classifier = SVC(kernel='rbf', probability=True)
        self.classifier.fit(X_train_scaled, y_train)
        
        # Evaluate
        predictions = self.classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        
        self.is_trained = True
        
        return {
            'training_accuracy': accuracy,
            'n_features': features.shape[1],
            'classification_report': classification_report(y_test, predictions, output_dict=True)
        }
    
    def attack(self, data: TraceData) -> Dict[str, Any]:
        """Perform acoustic attack."""
        if not self.is_trained:
            raise RuntimeError("Analyzer must be trained before attacking")
        
        # Extract features
        features = self.extract_features(data.traces)
        features_scaled = self.scaler.transform(features)
        
        # Predict
        predictions = self.classifier.predict(features_scaled)
        probabilities = self.classifier.predict_proba(features_scaled)
        confidences = np.max(probabilities, axis=1)
        
        # Calculate success rate
        success_rate = 0
        if data.labels is not None:
            success_rate = np.mean(predictions == data.labels)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'confidences': confidences,
            'success_rate': success_rate,
            'feature_importance': self._analyze_feature_importance(features)
        }
    
    def _analyze_feature_importance(self, features: np.ndarray) -> Dict[str, float]:
        """Analyze importance of different acoustic features."""
        # Simple variance-based importance
        feature_names = [
            'rms_amplitude', 'std_dev', 'peak_amplitude', 'event_count'
        ] + [f'band_power_{i}' for i in range(len(self.frequency_bands))] + [
            'total_power', 'dominant_freq', 'spectral_centroid', 'spectral_rolloff',
            'cepstral_1', 'cepstral_2'
        ]
        
        variances = np.var(features, axis=0)
        total_variance = np.sum(variances)
        
        importance = {}
        for i, name in enumerate(feature_names):
            if i < len(variances):
                importance[name] = variances[i] / total_variance
        
        return importance


class SoundBasedAttack(AcousticAnalyzer):
    """Specialized sound-based cryptanalytic attack."""
    
    def __init__(self, config: AnalysisConfig, attack_type: str = 'keystroke'):
        super().__init__(config)
        self.attack_type = attack_type
        
        # Attack-specific parameters
        if attack_type == 'keystroke':
            self.target_frequencies = [1000, 2000, 4000, 8000]  # Typical key click frequencies
        elif attack_type == 'coil_whine':
            self.target_frequencies = [15000, 20000, 25000]     # High-frequency coil noise
        elif attack_type == 'fan_noise':
            self.target_frequencies = [100, 200, 500, 1000]    # Fan modulation frequencies
    
    def preprocess_traces(self, traces: np.ndarray) -> np.ndarray:
        """Attack-specific preprocessing."""
        processed = super().preprocess_traces(traces)
        
        if self.attack_type == 'keystroke':
            # Focus on transient events
            processed = self._enhance_transients(processed)
        elif self.attack_type == 'coil_whine':
            # Focus on high-frequency components
            processed = self._enhance_high_frequency(processed)
        elif self.attack_type == 'fan_noise':
            # Focus on modulation patterns
            processed = self._enhance_modulation(processed)
        
        return processed
    
    def _enhance_transients(self, traces: np.ndarray) -> np.ndarray:
        """Enhance transient acoustic events."""
        enhanced_traces = []
        
        for trace in traces:
            # Compute energy envelope
            envelope = np.abs(signal.hilbert(trace))
            
            # Detect transients using derivative
            transient_detector = np.abs(np.diff(envelope))
            transient_mask = transient_detector > np.percentile(transient_detector, 90)
            
            # Enhance regions with transients
            enhanced = trace.copy()
            for i in range(len(transient_mask)):
                if transient_mask[i]:
                    start = max(0, i - 10)
                    end = min(len(enhanced), i + 10)
                    enhanced[start:end] *= 2.0
            
            enhanced_traces.append(enhanced)
        
        return np.array(enhanced_traces)
    
    def _enhance_high_frequency(self, traces: np.ndarray) -> np.ndarray:
        """Enhance high-frequency components."""
        enhanced_traces = []
        
        for trace in traces:
            # Apply high-pass filter
            nyquist = self.config.sample_rate / 2
            high_freq = 10000 / nyquist
            
            sos = signal.butter(4, high_freq, btype='high', output='sos')
            enhanced = signal.sosfilt(sos, trace)
            enhanced_traces.append(enhanced)
        
        return np.array(enhanced_traces)
    
    def _enhance_modulation(self, traces: np.ndarray) -> np.ndarray:
        """Enhance amplitude/frequency modulation patterns."""
        enhanced_traces = []
        
        for trace in traces:
            # Compute instantaneous amplitude
            analytic_signal = signal.hilbert(trace)
            amplitude_envelope = np.abs(analytic_signal)
            
            # Detect modulation by analyzing envelope variations
            envelope_variations = np.abs(np.diff(amplitude_envelope))
            modulation_strength = np.std(envelope_variations)
            
            # Enhance based on modulation strength
            enhancement_factor = 1.0 + modulation_strength
            enhanced = trace * enhancement_factor
            enhanced_traces.append(enhanced)
        
        return np.array(enhanced_traces)
    
    def analyze_acoustic_signature(self, traces: np.ndarray) -> Dict[str, Any]:
        """Analyze acoustic signature of cryptographic operations."""
        signatures = {}
        
        for i, trace in enumerate(traces):
            signature = {
                'duration': len(trace) / self.config.sample_rate,
                'peak_frequency': self._find_peak_frequency(trace),
                'frequency_spread': self._compute_frequency_spread(trace),
                'temporal_pattern': self._analyze_temporal_pattern(trace),
                'harmonic_content': self._analyze_harmonics(trace)
            }
            signatures[f'trace_{i}'] = signature
        
        return signatures
    
    def _find_peak_frequency(self, trace: np.ndarray) -> float:
        """Find the dominant frequency in the trace."""
        fft_trace = fft(trace)
        freqs = fftfreq(len(trace), 1/self.config.sample_rate)
        power_spectrum = np.abs(fft_trace) ** 2
        
        # Only consider positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        positive_power = power_spectrum[:len(power_spectrum)//2]
        
        peak_idx = np.argmax(positive_power)
        return positive_freqs[peak_idx]
    
    def _compute_frequency_spread(self, trace: np.ndarray) -> float:
        """Compute the spread of frequency content."""
        fft_trace = fft(trace)
        freqs = fftfreq(len(trace), 1/self.config.sample_rate)
        power_spectrum = np.abs(fft_trace) ** 2
        
        # Compute spectral centroid and spread
        positive_freqs = freqs[:len(freqs)//2]
        positive_power = power_spectrum[:len(power_spectrum)//2]
        
        if np.sum(positive_power) > 0:
            centroid = np.sum(positive_freqs * positive_power) / np.sum(positive_power)
            spread = np.sqrt(np.sum(((positive_freqs - centroid) ** 2) * positive_power) / np.sum(positive_power))
            return spread
        else:
            return 0.0
    
    def _analyze_temporal_pattern(self, trace: np.ndarray) -> Dict[str, float]:
        """Analyze temporal patterns in the acoustic trace."""
        # Compute energy over time
        window_size = int(0.01 * self.config.sample_rate)  # 10ms windows
        energy_pattern = []
        
        for i in range(0, len(trace) - window_size, window_size // 2):
            window = trace[i:i + window_size]
            energy = np.sum(window ** 2)
            energy_pattern.append(energy)
        
        energy_pattern = np.array(energy_pattern)
        
        return {
            'energy_variance': np.var(energy_pattern),
            'energy_peaks': len(signal.find_peaks(energy_pattern)[0]),
            'attack_time': self._compute_attack_time(energy_pattern),
            'decay_time': self._compute_decay_time(energy_pattern)
        }
    
    def _compute_attack_time(self, energy_pattern: np.ndarray) -> float:
        """Compute attack time (rise time to peak)."""
        if len(energy_pattern) == 0:
            return 0.0
        
        peak_idx = np.argmax(energy_pattern)
        peak_energy = energy_pattern[peak_idx]
        
        # Find 10% and 90% points
        target_10 = 0.1 * peak_energy
        target_90 = 0.9 * peak_energy
        
        idx_10 = np.where(energy_pattern[:peak_idx] >= target_10)[0]
        idx_90 = np.where(energy_pattern[:peak_idx] >= target_90)[0]
        
        if len(idx_10) > 0 and len(idx_90) > 0:
            return (idx_90[0] - idx_10[0]) * 0.005  # Convert to seconds (5ms per window)
        else:
            return 0.0
    
    def _compute_decay_time(self, energy_pattern: np.ndarray) -> float:
        """Compute decay time from peak."""
        if len(energy_pattern) == 0:
            return 0.0
        
        peak_idx = np.argmax(energy_pattern)
        peak_energy = energy_pattern[peak_idx]
        
        # Find 90% and 10% points after peak
        target_90 = 0.9 * peak_energy
        target_10 = 0.1 * peak_energy
        
        post_peak = energy_pattern[peak_idx:]
        idx_90 = np.where(post_peak <= target_90)[0]
        idx_10 = np.where(post_peak <= target_10)[0]
        
        if len(idx_90) > 0 and len(idx_10) > 0:
            return (idx_10[0] - idx_90[0]) * 0.005  # Convert to seconds
        else:
            return 0.0
    
    def _analyze_harmonics(self, trace: np.ndarray) -> Dict[str, float]:
        """Analyze harmonic content of the acoustic signal."""
        fft_trace = fft(trace)
        freqs = fftfreq(len(trace), 1/self.config.sample_rate)
        power_spectrum = np.abs(fft_trace) ** 2
        
        # Find fundamental frequency
        positive_freqs = freqs[:len(freqs)//2]
        positive_power = power_spectrum[:len(power_spectrum)//2]
        
        if len(positive_power) == 0:
            return {'fundamental': 0.0, 'harmonic_ratio': 0.0, 'total_harmonic_distortion': 0.0}
        
        fundamental_idx = np.argmax(positive_power)
        fundamental_freq = positive_freqs[fundamental_idx]
        fundamental_power = positive_power[fundamental_idx]
        
        # Analyze harmonics
        harmonic_power = 0.0
        harmonic_count = 0
        
        for harmonic in range(2, 6):  # Check 2nd through 5th harmonics
            harmonic_freq = harmonic * fundamental_freq
            if harmonic_freq < positive_freqs[-1]:
                # Find closest frequency bin
                harmonic_idx = np.argmin(np.abs(positive_freqs - harmonic_freq))
                harmonic_power += positive_power[harmonic_idx]
                harmonic_count += 1
        
        # Calculate metrics
        harmonic_ratio = harmonic_power / fundamental_power if fundamental_power > 0 else 0
        total_power = np.sum(positive_power)
        thd = harmonic_power / total_power if total_power > 0 else 0
        
        return {
            'fundamental': fundamental_freq,
            'harmonic_ratio': harmonic_ratio,
            'total_harmonic_distortion': thd
        }