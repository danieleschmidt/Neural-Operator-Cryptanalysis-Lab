"""Base classes for cryptographic target implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import numpy as np


class TargetType(Enum):
    """Types of cryptographic targets."""
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    HASH = "hash"
    POST_QUANTUM = "post_quantum"


class CountermeasureType(Enum):
    """Types of countermeasures."""
    MASKING = "masking"
    SHUFFLING = "shuffling"
    HIDING = "hiding"
    BLINDING = "blinding"
    DUMMY_OPERATIONS = "dummy_operations"


@dataclass
class ImplementationConfig:
    """Configuration for cryptographic implementations.
    
    Attributes:
        algorithm: Cryptographic algorithm name
        variant: Algorithm variant (e.g., 'kyber768', 'aes128')
        platform: Target platform (e.g., 'arm_cortex_m4', 'x86_64')
        optimization: Optimization level ('speed', 'size', 'constant_time')
        countermeasures: List of enabled countermeasures
        parameters: Algorithm-specific parameters
        noise_model: Noise characteristics for simulation
    """
    algorithm: str
    variant: str = "default"
    platform: str = "generic"
    optimization: str = "speed"
    countermeasures: List[CountermeasureType] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    noise_model: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.algorithm not in self._get_supported_algorithms():
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
            
        # Set default parameters based on algorithm
        if not self.parameters:
            self.parameters = self._get_default_parameters()
    
    def _get_supported_algorithms(self) -> List[str]:
        """Get list of supported algorithms."""
        return [
            'kyber', 'dilithium', 'classic_mceliece', 'sphincs',
            'aes', 'rsa', 'ecdsa'
        ]
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for algorithm."""
        defaults = {
            'kyber': {'n': 256, 'q': 3329, 'k': 3, 'eta': 2},
            'dilithium': {'n': 256, 'q': 8380417, 'k': 6, 'l': 5},
            'classic_mceliece': {'n': 4608, 't': 96, 'm': 13},
            'sphincs': {'n': 32, 'h': 64, 'w': 16},
            'aes': {'rounds': 10, 'key_size': 128},
            'rsa': {'key_size': 2048, 'e': 65537},
            'ecdsa': {'curve': 'P-256', 'key_size': 256}
        }
        return defaults.get(self.algorithm, {})


class CryptographicTarget(ABC):
    """Abstract base class for cryptographic target implementations.
    
    Provides interface for cryptographic operations with side-channel
    simulation capabilities. Subclasses implement specific algorithms
    with detailed intermediate value tracking.
    """
    
    def __init__(self, config: ImplementationConfig):
        self.config = config
        self.target_type = self._get_target_type()
        
        # Cryptographic state
        self.key = None
        self.state_history = []
        self.operation_count = 0
        
        # Countermeasure parameters
        self.masking_order = 0
        self.shuffle_masks = []
        
        # Initialize implementation-specific components
        self._initialize_implementation()
    
    @abstractmethod
    def _get_target_type(self) -> TargetType:
        """Get the type of cryptographic target."""
        pass
    
    @abstractmethod
    def _initialize_implementation(self):
        """Initialize implementation-specific components."""
        pass
    
    @abstractmethod
    def generate_key(self) -> np.ndarray:
        """Generate cryptographic key.
        
        Returns:
            Generated key as numpy array
        """
        pass
    
    @abstractmethod
    def set_key(self, key: np.ndarray):
        """Set cryptographic key.
        
        Args:
            key: Key to set
        """
        pass
    
    @abstractmethod
    def encrypt(self, plaintext: np.ndarray) -> np.ndarray:
        """Encrypt plaintext.
        
        Args:
            plaintext: Input plaintext
            
        Returns:
            Encrypted ciphertext
        """
        pass
    
    @abstractmethod
    def decrypt(self, ciphertext: np.ndarray) -> np.ndarray:
        """Decrypt ciphertext.
        
        Args:
            ciphertext: Input ciphertext
            
        Returns:
            Decrypted plaintext
        """
        pass
    
    @abstractmethod
    def compute_intermediate_values(self, input_data: np.ndarray) -> List[np.ndarray]:
        """Compute intermediate values during cryptographic operation.
        
        This method tracks all intermediate computational values that
        could potentially leak through side channels.
        
        Args:
            input_data: Input data for operation
            
        Returns:
            List of intermediate values at each operation step
        """
        pass
    
    def apply_countermeasure(self, countermeasure: CountermeasureType, 
                           parameters: Optional[Dict[str, Any]] = None):
        """Apply countermeasure to implementation.
        
        Args:
            countermeasure: Type of countermeasure to apply
            parameters: Countermeasure-specific parameters
        """
        if countermeasure not in self.config.countermeasures:
            self.config.countermeasures.append(countermeasure)
        
        if countermeasure == CountermeasureType.MASKING:
            self._apply_masking(parameters or {})
        elif countermeasure == CountermeasureType.SHUFFLING:
            self._apply_shuffling(parameters or {})
        elif countermeasure == CountermeasureType.HIDING:
            self._apply_hiding(parameters or {})
        elif countermeasure == CountermeasureType.BLINDING:
            self._apply_blinding(parameters or {})
        elif countermeasure == CountermeasureType.DUMMY_OPERATIONS:
            self._apply_dummy_operations(parameters or {})
    
    def _apply_masking(self, parameters: Dict[str, Any]):
        """Apply masking countermeasure."""
        self.masking_order = parameters.get('order', 1)
        print(f"Applied {self.masking_order}-order masking")
    
    def _apply_shuffling(self, parameters: Dict[str, Any]):
        """Apply shuffling countermeasure."""
        shuffle_space = parameters.get('space_size', 16)
        self.shuffle_masks = np.random.permutation(shuffle_space)
        print(f"Applied shuffling with space size {shuffle_space}")
    
    def _apply_hiding(self, parameters: Dict[str, Any]):
        """Apply hiding countermeasure."""
        noise_level = parameters.get('noise_level', 0.1)
        print(f"Applied hiding with noise level {noise_level}")
    
    def _apply_blinding(self, parameters: Dict[str, Any]):
        """Apply blinding countermeasure."""
        blind_factor = parameters.get('blind_factor', 'random')
        print(f"Applied blinding with factor {blind_factor}")
    
    def _apply_dummy_operations(self, parameters: Dict[str, Any]):
        """Apply dummy operations countermeasure."""
        dummy_ratio = parameters.get('ratio', 0.5)
        print(f"Applied dummy operations with ratio {dummy_ratio}")
    
    def get_leakage_model(self) -> str:
        """Get appropriate leakage model for this target."""
        if CountermeasureType.MASKING in self.config.countermeasures:
            return 'higher_order'
        else:
            return 'hamming_weight'
    
    def simulate_timing(self, operation: str) -> float:
        """Simulate timing for specific operation.
        
        Args:
            operation: Name of operation
            
        Returns:
            Simulated execution time in seconds
        """
        base_timings = {
            'key_generation': 1e-3,
            'encryption': 5e-4,
            'decryption': 5e-4,
            'signature': 1e-3,
            'verification': 5e-4
        }
        
        base_time = base_timings.get(operation, 1e-4)
        
        # Add platform-specific scaling
        platform_scaling = {
            'arm_cortex_m4': 2.0,
            'arm_cortex_m0': 4.0,
            'x86_64': 0.5,
            'risc_v': 1.5
        }
        
        scaling = platform_scaling.get(self.config.platform, 1.0)
        
        # Add noise for realistic timing
        noise = np.random.normal(0, base_time * 0.1)
        
        return base_time * scaling + noise
    
    def simulate_power_consumption(self, intermediate_values: List[np.ndarray]) -> np.ndarray:
        """Simulate power consumption based on intermediate values.
        
        Args:
            intermediate_values: List of intermediate computational values
            
        Returns:
            Simulated power trace
        """
        # Base power consumption
        base_power = 0.1  # mW
        
        # Power trace
        trace_length = len(intermediate_values) * 10  # 10 samples per operation
        power_trace = np.full(trace_length, base_power)
        
        # Add operation-specific power consumption
        for i, values in enumerate(intermediate_values):
            start_idx = i * 10
            end_idx = start_idx + 10
            
            if values.size > 0:
                # Hamming weight leakage model
                if isinstance(values, np.ndarray) and values.dtype == np.uint8:
                    hw = np.sum([(values >> bit) & 1 for bit in range(8)], axis=0)
                    leakage = np.mean(hw) * 0.001  # 1 µW per bit
                else:
                    leakage = np.mean(np.abs(values)) * 0.0001
                    
                power_trace[start_idx:end_idx] += leakage
        
        # Add countermeasure effects
        if CountermeasureType.MASKING in self.config.countermeasures:
            # Masking reduces leakage but adds overhead
            power_trace *= (1.0 / (self.masking_order + 1))
            power_trace += 0.02 * self.masking_order  # Masking overhead
        
        if CountermeasureType.SHUFFLING in self.config.countermeasures:
            # Shuffling changes timing but not average power
            shuffle_indices = np.random.permutation(len(power_trace))
            power_trace = power_trace[shuffle_indices]
        
        # Add noise
        noise_std = self.config.noise_model.get('power_noise_std', 0.01)
        noise = np.random.normal(0, noise_std, len(power_trace))
        
        return power_trace + noise
    
    def get_implementation_details(self) -> Dict[str, Any]:
        """Get detailed implementation information."""
        return {
            'algorithm': self.config.algorithm,
            'variant': self.config.variant,
            'platform': self.config.platform,
            'optimization': self.config.optimization,
            'countermeasures': [cm.value for cm in self.config.countermeasures],
            'parameters': self.config.parameters,
            'target_type': self.target_type.value,
            'operation_count': self.operation_count,
            'leakage_model': self.get_leakage_model()
        }
    
    def reset_state(self):
        """Reset implementation state."""
        self.state_history.clear()
        self.operation_count = 0
    
    def benchmark_operation(self, operation: str, n_iterations: int = 1000) -> Dict[str, float]:
        """Benchmark specific operation.
        
        Args:
            operation: Operation to benchmark
            n_iterations: Number of iterations
            
        Returns:
            Benchmark results
        """
        import time
        
        # Generate test data
        if operation in ['encrypt', 'decrypt']:
            test_data = [np.random.randint(0, 256, 16, dtype=np.uint8) 
                        for _ in range(n_iterations)]
        else:
            test_data = [None] * n_iterations
        
        # Time the operation
        start_time = time.time()
        
        for data in test_data:
            if operation == 'encrypt' and data is not None:
                self.encrypt(data)
            elif operation == 'decrypt' and data is not None:
                self.decrypt(data)
            elif operation == 'key_generation':
                self.generate_key()
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / n_iterations
        
        return {
            'operation': operation,
            'total_time': total_time,
            'average_time': avg_time,
            'operations_per_second': n_iterations / total_time,
            'iterations': n_iterations
        }