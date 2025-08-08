"""
Synthetic dataset generation for neural cryptanalysis research.

Provides comprehensive synthetic trace generation with realistic noise models,
device characteristics, and cryptographic operation patterns.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json

from ..targets.base import CryptographicTarget
from ..utils.config import load_config, save_config
from ..utils.logging_utils import get_logger
from ..utils.validation import validate_input


logger = get_logger(__name__)


@dataclass
class NoiseModel:
    """Noise model configuration for synthetic trace generation."""
    
    noise_type: str = 'gaussian'  # 'gaussian', 'uniform', 'laplace'
    snr_db: float = 10.0
    baseline_noise: float = 0.01
    frequency_dependent: bool = True
    correlation_factor: float = 0.0  # Trace-to-trace correlation


@dataclass 
class DeviceModel:
    """Device characteristics model for realistic trace generation."""
    
    device_type: str = 'microcontroller'  # 'microcontroller', 'fpga', 'asic'
    clock_frequency: float = 24e6  # Hz
    voltage_range: Tuple[float, float] = (3.0, 3.6)  # V
    temperature_range: Tuple[float, float] = (20, 85)  # Celsius
    process_variation: float = 0.05  # Percentage variation
    aging_factor: float = 0.0  # Long-term drift


class SyntheticDatasetGenerator:
    """Advanced synthetic dataset generator for neural cryptanalysis."""
    
    def __init__(self, 
                 noise_model: Optional[NoiseModel] = None,
                 device_model: Optional[DeviceModel] = None,
                 random_seed: Optional[int] = None):
        """Initialize synthetic dataset generator.
        
        Args:
            noise_model: Noise characteristics configuration
            device_model: Device characteristics configuration  
            random_seed: Random seed for reproducibility
        """
        self.noise_model = noise_model or NoiseModel()
        self.device_model = device_model or DeviceModel()
        
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
        
        logger.info(f"Initialized synthetic dataset generator with "
                   f"SNR={self.noise_model.snr_db}dB, "
                   f"device={self.device_model.device_type}")
    
    def generate_power_leakage(self, 
                              data_values: np.ndarray,
                              hamming_weight_model: bool = True) -> np.ndarray:
        """Generate power consumption traces based on processed data.
        
        Args:
            data_values: Processed data values (e.g., S-box outputs)
            hamming_weight_model: Use Hamming weight leakage model
            
        Returns:
            Power consumption traces
        """
        if hamming_weight_model:
            # Hamming weight leakage model
            hamming_weights = np.array([bin(val).count('1') for val in data_values.flatten()])
            leakage = hamming_weights.reshape(data_values.shape)
        else:
            # Identity leakage model
            leakage = data_values.astype(float)
        
        # Scale by device characteristics
        voltage = np.random.uniform(*self.device_model.voltage_range)
        power_scale = voltage ** 2  # P ∝ V²
        
        leakage = leakage * power_scale * 1e-3  # Convert to mW
        
        return leakage
    
    def generate_em_leakage(self, 
                           data_values: np.ndarray,
                           spatial_components: int = 3) -> np.ndarray:
        """Generate electromagnetic emanation traces.
        
        Args:
            data_values: Processed data values
            spatial_components: Number of spatial EM components
            
        Returns:
            EM emanation traces with spatial components
        """
        n_samples = len(data_values)
        em_traces = np.zeros((n_samples, spatial_components))
        
        for i in range(spatial_components):
            # Different spatial components have different sensitivities
            sensitivity = 0.8 ** i  # Exponentially decreasing sensitivity
            
            # EM leakage with bit-level dependencies
            bit_leakage = np.array([
                sum((val >> bit) & 1 for bit in range(8)) 
                for val in data_values.flatten()
            ])
            
            em_traces[:, i] = bit_leakage * sensitivity
        
        return em_traces
    
    def add_realistic_noise(self, clean_traces: np.ndarray) -> np.ndarray:
        """Add realistic noise to clean traces.
        
        Args:
            clean_traces: Clean leakage traces
            
        Returns:
            Noisy traces
        """
        signal_power = np.var(clean_traces)
        noise_power = signal_power / (10 ** (self.noise_model.snr_db / 10))
        
        if self.noise_model.noise_type == 'gaussian':
            noise = np.random.normal(0, np.sqrt(noise_power), clean_traces.shape)
        elif self.noise_model.noise_type == 'uniform':
            noise_std = np.sqrt(noise_power)
            noise = np.random.uniform(-noise_std * np.sqrt(3), 
                                    noise_std * np.sqrt(3), 
                                    clean_traces.shape)
        elif self.noise_model.noise_type == 'laplace':
            noise_scale = np.sqrt(noise_power / 2)
            noise = np.random.laplace(0, noise_scale, clean_traces.shape)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_model.noise_type}")
        
        # Add baseline noise
        baseline = np.random.normal(0, self.noise_model.baseline_noise, 
                                  clean_traces.shape)
        
        # Frequency-dependent noise (pink noise)
        if self.noise_model.frequency_dependent:
            freq_noise = self._generate_pink_noise(clean_traces.shape)
            noise += freq_noise * np.sqrt(noise_power) * 0.3
        
        noisy_traces = clean_traces + noise + baseline
        
        return noisy_traces
    
    def _generate_pink_noise(self, shape: Tuple) -> np.ndarray:
        """Generate pink (1/f) noise."""
        if len(shape) == 1:
            shape = (shape[0], 1)
        
        n_samples = shape[-1]
        pink_noise = np.zeros(shape)
        
        for i in range(shape[0]):
            # Generate white noise
            white = np.random.randn(n_samples)
            
            # Apply 1/f filter in frequency domain
            freqs = np.fft.fftfreq(n_samples)
            freqs[0] = 1e-10  # Avoid division by zero
            
            white_fft = np.fft.fft(white)
            pink_fft = white_fft / np.sqrt(np.abs(freqs))
            pink_noise[i] = np.real(np.fft.ifft(pink_fft))
        
        if len(shape) == 1:
            pink_noise = pink_noise.flatten()
        
        return pink_noise
    
    def generate_temporal_alignment_variation(self, 
                                            traces: np.ndarray,
                                            max_shift: int = 10) -> np.ndarray:
        """Add temporal misalignment to traces.
        
        Args:
            traces: Input traces
            max_shift: Maximum time shift in samples
            
        Returns:
            Misaligned traces
        """
        aligned_traces = np.zeros_like(traces)
        
        for i, trace in enumerate(traces):
            # Random shift for each trace
            shift = np.random.randint(-max_shift, max_shift + 1)
            
            if shift > 0:
                aligned_traces[i, shift:] = trace[:-shift]
            elif shift < 0:
                aligned_traces[i, :shift] = trace[-shift:]
            else:
                aligned_traces[i] = trace
        
        return aligned_traces
    
    def generate_aes_dataset(self, 
                           n_traces: int,
                           key: Optional[np.ndarray] = None,
                           plaintexts: Optional[np.ndarray] = None,
                           target_bytes: List[int] = [0],
                           trace_length: int = 1000) -> Dict:
        """Generate comprehensive AES side-channel dataset.
        
        Args:
            n_traces: Number of traces to generate
            key: AES key (random if None)
            plaintexts: Input plaintexts (random if None)
            target_bytes: Key bytes to include as labels
            trace_length: Length of each trace
            
        Returns:
            Dictionary containing traces, labels, and metadata
        """
        validate_input(n_traces, int, min_value=1, max_value=1000000)
        
        if key is None:
            key = np.random.randint(0, 256, 16, dtype=np.uint8)
        
        if plaintexts is None:
            plaintexts = np.random.randint(0, 256, (n_traces, 16), dtype=np.uint8)
        
        logger.info(f"Generating AES dataset: {n_traces} traces, "
                   f"target bytes: {target_bytes}")
        
        # AES S-box
        sbox = np.array([
            0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
            0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
            0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
            0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
            0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
            0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
            0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
            0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
            0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x02, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
            0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
            0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
            0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
            0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
            0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
            0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
            0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
        ], dtype=np.uint8)
        
        # Generate intermediate values and traces
        power_traces = []
        em_traces = []
        labels = {byte_idx: [] for byte_idx in target_bytes}
        
        for i, plaintext in enumerate(plaintexts):
            # Compute S-box outputs for target bytes
            sbox_outputs = {}
            for byte_idx in target_bytes:
                intermediate = plaintext[byte_idx] ^ key[byte_idx]
                sbox_output = sbox[intermediate]
                sbox_outputs[byte_idx] = sbox_output
                labels[byte_idx].append(sbox_output)
            
            # Generate power leakage (focus on first target byte)
            main_sbox_output = sbox_outputs[target_bytes[0]]
            power_leakage = self.generate_power_leakage(np.array([main_sbox_output]))
            
            # Create full power trace with operation timeline
            power_trace = np.zeros(trace_length)
            
            # S-box operation occurs at specific time points
            sbox_start = trace_length // 4
            sbox_end = sbox_start + 50
            
            # Add S-box leakage
            power_trace[sbox_start:sbox_end] = np.repeat(power_leakage[0], sbox_end - sbox_start)
            
            # Add background activity
            background = np.random.normal(0.5, 0.1, trace_length)
            power_trace += background
            
            power_traces.append(power_trace)
            
            # Generate EM traces
            em_leakage = self.generate_em_leakage(np.array([main_sbox_output]))
            em_trace = np.zeros((trace_length, em_leakage.shape[1]))
            em_trace[sbox_start:sbox_end] = np.repeat(em_leakage, sbox_end - sbox_start, axis=0)
            em_traces.append(em_trace)
        
        power_traces = np.array(power_traces)
        em_traces = np.array(em_traces)
        
        # Add noise and misalignment
        power_traces = self.add_realistic_noise(power_traces)
        em_traces = self.add_realistic_noise(em_traces)
        
        power_traces = self.generate_temporal_alignment_variation(power_traces)
        
        # Convert labels to arrays
        for byte_idx in target_bytes:
            labels[byte_idx] = np.array(labels[byte_idx])
        
        dataset = {
            'power_traces': power_traces,
            'em_traces': em_traces,
            'labels': labels,
            'plaintexts': plaintexts,
            'key': key,
            'metadata': {
                'n_traces': n_traces,
                'trace_length': trace_length,
                'target_bytes': target_bytes,
                'noise_model': {
                    'type': self.noise_model.noise_type,
                    'snr_db': self.noise_model.snr_db
                },
                'device_model': {
                    'type': self.device_model.device_type,
                    'clock_freq': self.device_model.clock_frequency
                },
                'generation_timestamp': np.datetime64('now').astype(str)
            }
        }
        
        logger.info(f"Generated AES dataset with {n_traces} traces successfully")
        
        return dataset
    
    def save_dataset(self, dataset: Dict, filepath: Path) -> None:
        """Save dataset to file.
        
        Args:
            dataset: Dataset dictionary
            filepath: Output file path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.suffix == '.npz':
            # Save as NumPy compressed format
            save_dict = {}
            for key, value in dataset.items():
                if key == 'metadata':
                    # Save metadata as JSON string
                    save_dict[key] = json.dumps(value)
                else:
                    save_dict[key] = value
            
            np.savez_compressed(filepath, **save_dict)
            
        elif filepath.suffix == '.pt':
            # Save as PyTorch format
            torch.save(dataset, filepath)
            
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Dataset saved to {filepath}")
    
    def load_dataset(self, filepath: Path) -> Dict:
        """Load dataset from file.
        
        Args:
            filepath: Input file path
            
        Returns:
            Dataset dictionary
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.npz':
            # Load NumPy format
            loaded = np.load(filepath, allow_pickle=True)
            dataset = dict(loaded)
            
            # Parse metadata JSON
            if 'metadata' in dataset:
                dataset['metadata'] = json.loads(str(dataset['metadata']))
                
        elif filepath.suffix == '.pt':
            # Load PyTorch format  
            dataset = torch.load(filepath)
            
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Dataset loaded from {filepath}")
        
        return dataset