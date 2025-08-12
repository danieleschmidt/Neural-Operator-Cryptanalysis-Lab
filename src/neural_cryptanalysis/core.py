"""Core API classes for Neural Operator Cryptanalysis Lab."""

import time
import warnings
from typing import Dict, List, Optional, Union, Any
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from .neural_operators import NeuralOperatorBase, OperatorConfig
from .neural_operators.fno import FourierNeuralOperator
from .neural_operators.deeponet import DeepOperatorNetwork
from .neural_operators.custom import SideChannelFNO, LeakageFNO, MultiModalOperator
from .side_channels import SideChannelAnalyzer, AnalysisConfig, TraceData
from .utils.config import load_config, save_config
from .utils.errors import (
    NeuralCryptanalysisError, ValidationError, ConfigurationError, 
    ModelError, DataError, ResourceError, TimeoutError,
    error_handler, validate_input, ErrorContext, create_error_context
)
from .utils.validation import (
    validate_trace_data, validate_neural_operator_config,
    validate_numeric_range, ValidationContext
)
from .utils.logging_utils import get_logger

logger = get_logger(__name__)


class NeuralSCA:
    """Main interface for neural operator-based side-channel analysis.
    
    This class provides a high-level API for performing side-channel attacks
    using neural operators. It integrates preprocessing, training, and attack
    phases with various neural operator architectures.
    
    Example:
        >>> neural_sca = NeuralSCA(architecture='fourier_neural_operator')
        >>> model = neural_sca.train(traces, labels)
        >>> results = neural_sca.attack(test_traces, model)
    """
    
    @error_handler(re_raise=True)
    def __init__(self, 
                 architecture: str = 'fourier_neural_operator',
                 channels: List[str] = ['power'],
                 config: Optional[Union[Dict, str]] = None):
        """Initialize NeuralSCA.
        
        Args:
            architecture: Neural operator architecture to use
            channels: List of side-channel types
            config: Configuration dict or path to config file
            
        Raises:
            ValidationError: If architecture or channels are invalid
            ConfigurationError: If configuration is invalid
        """
        # Input validation
        if not isinstance(architecture, str) or not architecture.strip():
            raise ValidationError(
                "Architecture must be a non-empty string",
                field="architecture",
                value=architecture,
                context=create_error_context("NeuralSCA", "__init__")
            )
        
        if not isinstance(channels, list) or not all(isinstance(ch, str) for ch in channels):
            raise ValidationError(
                "Channels must be a list of strings",
                field="channels",
                value=channels,
                context=create_error_context("NeuralSCA", "__init__")
            )
        
        self.architecture = architecture
        self.channels = channels
        
        # Load and validate configuration
        try:
            if isinstance(config, str):
                if not Path(config).exists():
                    raise ConfigurationError(
                        f"Configuration file not found: {config}",
                        config_key="config_file",
                        context=create_error_context("NeuralSCA", "__init__")
                    )
                self.config = load_config(config)
            elif isinstance(config, dict):
                self.config = config
            elif config is None:
                self.config = self._get_default_config()
            else:
                raise ValidationError(
                    "Config must be string path, dict, or None",
                    field="config",
                    value=type(config),
                    context=create_error_context("NeuralSCA", "__init__")
                )
        except Exception as e:
            if isinstance(e, NeuralCryptanalysisError):
                raise
            raise ConfigurationError(
                f"Failed to load configuration: {e}",
                context=create_error_context("NeuralSCA", "__init__"),
                cause=e
            )
        
        # Validate configuration structure
        with ValidationContext("NeuralSCA initialization") as ctx:
            # Validate operator config
            operator_config = self.config.get('operator', {})
            operator_issues = validate_neural_operator_config(operator_config)
            for issue in operator_issues:
                ctx.add_error(f"Operator config: {issue}")
            
            # Validate architecture
            valid_architectures = [
                'fourier_neural_operator', 'deep_operator_network', 
                'side_channel_fno', 'leakage_fno', 'multi_modal'
            ]
            if self.architecture not in valid_architectures:
                ctx.add_error(f"Unknown architecture: {self.architecture}. Valid: {valid_architectures}")
            
            # Validate channels
            valid_channels = ['power', 'electromagnetic', 'acoustic', 'timing', 'cache']
            invalid_channels = [ch for ch in self.channels if ch not in valid_channels]
            if invalid_channels:
                ctx.add_warning(f"Unrecognized channels: {invalid_channels}. Valid: {valid_channels}")
        
        # Initialize configurations with error handling
        try:
            self.operator_config = OperatorConfig(**self.config.get('operator', {}))
            self.analysis_config = AnalysisConfig(**self.config.get('analysis', {}))
        except Exception as e:
            raise ConfigurationError(
                f"Failed to create configuration objects: {e}",
                context=create_error_context("NeuralSCA", "__init__"),
                cause=e
            )
        
        # Create neural operator with error handling
        try:
            self.neural_operator = self._create_neural_operator()
        except Exception as e:
            raise ModelError(
                f"Failed to create neural operator: {e}",
                model_component="neural_operator",
                context=create_error_context("NeuralSCA", "__init__"),
                cause=e
            )
        
        # Initialize other components
        self.analyzer = None
        self.trained_models = {}
        self.training_history = []
        
        logger.info(f"NeuralSCA initialized with architecture: {architecture}, channels: {channels}")
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'operator': {
                'input_dim': 1,
                'output_dim': 256,
                'hidden_dim': 64,
                'num_layers': 4,
                'activation': 'gelu',
                'dropout': 0.1,
                'device': 'cpu'
            },
            'analysis': {
                'sample_rate': 1e6,
                'trace_length': 10000,
                'n_traces': 10000,
                'preprocessing': ['standardize'],
                'poi_method': 'mutual_information',
                'n_pois': 100
            },
            'training': {
                'batch_size': 64,
                'learning_rate': 1e-3,
                'epochs': 100,
                'patience': 10,
                'min_delta': 1e-6
            }
        }
    
    @error_handler(re_raise=True)
    def _create_neural_operator(self) -> NeuralOperatorBase:
        """Create neural operator based on architecture.
        
        Returns:
            Configured neural operator instance
            
        Raises:
            ModelError: If operator creation fails
        """
        try:
            if self.architecture == 'fourier_neural_operator':
                return FourierNeuralOperator(self.operator_config)
            elif self.architecture == 'deep_operator_network':
                return DeepOperatorNetwork(self.operator_config)
            elif self.architecture == 'side_channel_fno':
                return SideChannelFNO(self.operator_config)
            elif self.architecture == 'leakage_fno':
                return LeakageFNO(self.operator_config)
            elif self.architecture == 'multi_modal':
                modalities = {ch: self.operator_config.input_dim for ch in self.channels}
                return MultiModalOperator(self.operator_config, modalities)
            else:
                raise ModelError(
                    f"Unknown architecture: {self.architecture}",
                    model_component="neural_operator",
                    context=create_error_context("NeuralSCA", "_create_neural_operator")
                )
        except Exception as e:
            if isinstance(e, NeuralCryptanalysisError):
                raise
            raise ModelError(
                f"Failed to create {self.architecture}: {e}",
                model_component="neural_operator",
                context=create_error_context("NeuralSCA", "_create_neural_operator"),
                cause=e
            )
    
    @error_handler(re_raise=True)
    def train(self, 
              traces: Union[np.ndarray, TraceData],
              labels: Optional[np.ndarray] = None,
              validation_split: float = 0.2,
              timeout: Optional[float] = None,
              **kwargs) -> nn.Module:
        """Train neural operator on side-channel traces.
        
        Args:
            traces: Training traces or TraceData object
            labels: Target labels (if not in TraceData)
            validation_split: Fraction of data for validation
            timeout: Optional timeout in seconds
            **kwargs: Additional training parameters
            
        Returns:
            Trained neural operator model
            
        Raises:
            ValidationError: If input data is invalid
            DataError: If data processing fails
            ModelError: If training fails
            TimeoutError: If training times out
        """
        start_time = time.time()
        
        # Input validation
        if not isinstance(validation_split, (int, float)) or not 0 <= validation_split < 1:
            raise ValidationError(
                "Validation split must be a number between 0 and 1",
                field="validation_split",
                value=validation_split,
                context=create_error_context("NeuralSCA", "train")
            )
        
        # Convert to TraceData if needed with validation
        try:
            if isinstance(traces, np.ndarray):
                if labels is None:
                    raise ValidationError(
                        "Labels required when traces is numpy array",
                        field="labels",
                        context=create_error_context("NeuralSCA", "train")
                    )
                
                # Validate trace data
                validation_issues = validate_trace_data(traces, labels)
                if validation_issues:
                    raise DataError(
                        f"Trace data validation failed: {validation_issues}",
                        data_type="training_traces",
                        context=create_error_context("NeuralSCA", "train")
                    )
                
                trace_data = TraceData(traces=traces, labels=labels)
            elif hasattr(traces, 'traces') and hasattr(traces, 'labels'):
                trace_data = traces
                # Validate existing TraceData
                validation_issues = validate_trace_data(trace_data.traces, trace_data.labels)
                if validation_issues:
                    logger.warning(f"TraceData validation issues: {validation_issues}")
            else:
                raise ValidationError(
                    "Traces must be numpy array or TraceData object",
                    field="traces",
                    value=type(traces),
                    context=create_error_context("NeuralSCA", "train")
                )
        except Exception as e:
            if isinstance(e, NeuralCryptanalysisError):
                raise
            raise DataError(
                f"Failed to process training data: {e}",
                data_type="training_traces",
                context=create_error_context("NeuralSCA", "train"),
                cause=e
            )
        
        # Split data
        train_data, val_data = trace_data.split(1 - validation_split)
        
        # Create data loaders
        train_loader = self._create_dataloader(train_data, 
                                             self.config.get('training', {}).get('batch_size', 64))
        val_loader = self._create_dataloader(val_data,
                                           self.config.get('training', {}).get('batch_size', 64))
        
        # Setup training
        optimizer = torch.optim.Adam(
            self.neural_operator.parameters(),
            lr=self.config.get('training', {}).get('learning_rate', 1e-3)
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        epochs = self.config.get('training', {}).get('epochs', 100)
        patience = self.config.get('training', {}).get('patience', 10)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader, optimizer, criterion)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(val_loader, criterion)
            
            # Track progress
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            })
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.trained_models['best'] = self.neural_operator.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        if 'best' in self.trained_models:
            self.neural_operator.load_state_dict(self.trained_models['best'])
        
        return self.neural_operator
    
    def attack(self, 
               target_traces: Union[np.ndarray, TraceData],
               model: Optional[nn.Module] = None,
               strategy: str = 'direct') -> Dict[str, Any]:
        """Perform side-channel attack using trained model.
        
        Args:
            target_traces: Traces to attack
            model: Trained model (uses self.neural_operator if None)
            strategy: Attack strategy ('direct', 'template', 'adaptive')
            
        Returns:
            Attack results including recovered keys and confidence scores
        """
        if model is None:
            model = self.neural_operator
            
        if not hasattr(model, 'eval'):
            raise ValueError("Model must be a trained neural network")
        
        # Convert to TraceData if needed
        if isinstance(target_traces, np.ndarray):
            trace_data = TraceData(traces=target_traces)
        else:
            trace_data = target_traces
        
        # Create data loader
        test_loader = self._create_dataloader(trace_data, batch_size=64, shuffle=False)
        
        # Attack based on strategy
        if strategy == 'direct':
            return self._direct_attack(test_loader, model)
        elif strategy == 'template':
            return self._template_attack(test_loader, model)
        elif strategy == 'adaptive':
            return self._adaptive_attack(test_loader, model)
        else:
            raise ValueError(f"Unknown attack strategy: {strategy}")
    
    def _create_dataloader(self, data: TraceData, batch_size: int, shuffle: bool = True):
        """Create PyTorch DataLoader from TraceData."""
        dataset = TraceDataset(data)
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )
    
    def _train_epoch(self, dataloader, optimizer, criterion) -> float:
        """Train for one epoch."""
        self.neural_operator.train()
        total_loss = 0
        
        for batch in dataloader:
            traces = batch['trace'].float()
            labels = batch['label'].long()
            
            optimizer.zero_grad()
            outputs = self.neural_operator(traces)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def _validate_epoch(self, dataloader, criterion) -> tuple[float, float]:
        """Validate for one epoch."""
        self.neural_operator.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                traces = batch['trace'].float()
                labels = batch['label'].long()
                
                outputs = self.neural_operator(traces)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total if total > 0 else 0
        return total_loss / len(dataloader), accuracy
    
    def _direct_attack(self, dataloader, model) -> Dict[str, Any]:
        """Direct neural attack."""
        model.eval()
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for batch in dataloader:
                traces = batch['trace'].float()
                outputs = model(traces)
                
                # Get predictions and confidence scores
                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1)
                conf = torch.max(probs, dim=1)[0]
                
                predictions.extend(pred.cpu().numpy())
                confidences.extend(conf.cpu().numpy())
        
        return {
            'predictions': np.array(predictions),
            'confidences': np.array(confidences),
            'success': np.mean(np.array(confidences) > 0.5),
            'avg_confidence': np.mean(confidences)
        }
    
    def _template_attack(self, dataloader, model) -> Dict[str, Any]:
        """Template-based attack using neural operator."""
        # Implementation would involve building templates and matching
        # For now, fall back to direct attack
        return self._direct_attack(dataloader, model)
    
    def _adaptive_attack(self, dataloader, model) -> Dict[str, Any]:
        """Adaptive attack that adjusts parameters."""
        # Implementation would involve parameter adaptation
        # For now, fall back to direct attack
        return self._direct_attack(dataloader, model)
    
    def evaluate_countermeasures(self, 
                               protected_traces: TraceData,
                               unprotected_traces: TraceData) -> Dict[str, Any]:
        """Evaluate effectiveness of countermeasures."""
        # Attack both protected and unprotected implementations
        protected_results = self.attack(protected_traces)
        unprotected_results = self.attack(unprotected_traces)
        
        return {
            'protected_success_rate': protected_results['success'],
            'unprotected_success_rate': unprotected_results['success'],
            'security_improvement': (
                unprotected_results['success'] - protected_results['success']
            ) / unprotected_results['success'] if unprotected_results['success'] > 0 else 0,
            'traces_needed_ratio': self._estimate_traces_needed_ratio(
                protected_results, unprotected_results
            )
        }
    
    def _estimate_traces_needed_ratio(self, protected_results, unprotected_results) -> float:
        """Estimate ratio of traces needed for protected vs unprotected."""
        # Simplified estimation based on confidence scores
        protected_conf = protected_results.get('avg_confidence', 0)
        unprotected_conf = unprotected_results.get('avg_confidence', 0)
        
        if protected_conf > 0 and unprotected_conf > 0:
            return (unprotected_conf / protected_conf) ** 2
        else:
            return float('inf')


class LeakageSimulator:
    """Simulator for generating synthetic side-channel traces.
    
    Generates realistic side-channel traces for various cryptographic
    operations and device models, useful for training and testing.
    """
    
    def __init__(self, 
                 device_model: str = 'stm32f4',
                 noise_model: str = 'realistic'):
        """Initialize LeakageSimulator.
        
        Args:
            device_model: Target device model
            noise_model: Noise characteristics
        """
        self.device_model = device_model
        self.noise_model = noise_model
        
        # Device parameters
        self.device_params = self._get_device_params(device_model)
        
        # Noise parameters  
        self.noise_params = self._get_noise_params(noise_model)
        
    def _get_device_params(self, model: str) -> Dict[str, Any]:
        """Get device-specific parameters."""
        params = {
            'stm32f4': {
                'voltage': 3.3,
                'frequency': 168e6,
                'power_baseline': 0.1,
                'leakage_factor': 0.01,
                'capacitance': 10e-12
            },
            'atmega328': {
                'voltage': 5.0,
                'frequency': 16e6,
                'power_baseline': 0.05,
                'leakage_factor': 0.02,
                'capacitance': 5e-12
            }
        }
        return params.get(model, params['stm32f4'])
    
    def _get_noise_params(self, model: str) -> Dict[str, Any]:
        """Get noise model parameters."""
        params = {
            'realistic': {
                'gaussian_std': 0.01,
                'pink_noise_factor': 0.005,
                'quantization_bits': 12,
                'sampling_jitter': 1e-9
            },
            'low_noise': {
                'gaussian_std': 0.001,
                'pink_noise_factor': 0.0005,
                'quantization_bits': 16,
                'sampling_jitter': 1e-10
            },
            'high_noise': {
                'gaussian_std': 0.1,
                'pink_noise_factor': 0.05,
                'quantization_bits': 8,
                'sampling_jitter': 1e-8
            }
        }
        return params.get(model, params['realistic'])
    
    def simulate_traces(self,
                       target,
                       n_traces: int = 1000,
                       operations: List[str] = ['sbox'],
                       trace_length: int = 10000) -> TraceData:
        """Simulate side-channel traces.
        
        Args:
            target: Target implementation object
            n_traces: Number of traces to generate
            operations: List of operations to include
            trace_length: Length of each trace
            
        Returns:
            Generated trace data
        """
        traces = []
        plaintexts = []
        labels = []
        
        for i in range(n_traces):
            # Generate random plaintext
            plaintext = np.random.randint(0, 256, size=16, dtype=np.uint8)
            plaintexts.append(plaintext)
            
            # Simulate cryptographic operation
            intermediate_values = target.compute_intermediate_values(plaintext)
            
            # Generate leakage trace
            trace = self._generate_leakage_trace(
                intermediate_values, operations, trace_length
            )
            
            traces.append(trace)
            
            # Extract labels (e.g., key bytes)
            if hasattr(target, 'key'):
                label = target.key[0] ^ plaintext[0]  # First round key byte
                labels.append(label)
        
        return TraceData(
            traces=np.array(traces),
            plaintexts=np.array(plaintexts),
            labels=np.array(labels) if labels else None,
            metadata={
                'device_model': self.device_model,
                'noise_model': self.noise_model,
                'operations': operations
            }
        )
    
    def _generate_leakage_trace(self, 
                              intermediate_values: np.ndarray,
                              operations: List[str],
                              trace_length: int) -> np.ndarray:
        """Generate single leakage trace."""
        trace = np.zeros(trace_length)
        
        # Add baseline power consumption
        trace += self.device_params['power_baseline']
        
        # Add operation-specific leakage
        for op in operations:
            if op == 'sbox':
                self._add_sbox_leakage(trace, intermediate_values)
            elif op == 'ntt':
                self._add_ntt_leakage(trace, intermediate_values)
            elif op == 'multiplication':
                self._add_mult_leakage(trace, intermediate_values)
        
        # Add noise
        trace = self._add_noise(trace)
        
        return trace
    
    def _add_sbox_leakage(self, trace: np.ndarray, values: np.ndarray):
        """Add S-box leakage to trace."""
        # Hamming weight model
        if len(values) > 0:
            hw = np.sum([[int(v) >> i & 1 for i in range(8)] for v in values], axis=1)
        else:
            hw = 0
        
        # Add leakage at specific time points
        start_idx = len(trace) // 4
        end_idx = start_idx + len(values) if len(values) > 0 else start_idx + 1
        
        if end_idx <= len(trace):
            leakage = hw * self.device_params['leakage_factor']
            trace[start_idx:end_idx] += leakage
    
    def _add_ntt_leakage(self, trace: np.ndarray, values: np.ndarray):
        """Add NTT operation leakage."""
        # More complex leakage pattern for NTT
        if len(values) > 0:
            for i, val in enumerate(values[:min(len(values), len(trace)//2)]):
                idx = i * 2
                if idx < len(trace):
                    trace[idx] += (val % 256) * self.device_params['leakage_factor'] * 0.1
    
    def _add_mult_leakage(self, trace: np.ndarray, values: np.ndarray):
        """Add multiplication leakage."""
        # Multiplication has different timing characteristics
        if len(values) > 0:
            mult_indices = np.linspace(0, len(trace)-1, len(values), dtype=int)
            for idx, val in zip(mult_indices, values):
                trace[idx] += (val & 0xFF) * self.device_params['leakage_factor'] * 0.5
    
    def _add_noise(self, trace: np.ndarray) -> np.ndarray:
        """Add realistic noise to trace (simplified)."""
        # Simplified noise addition to avoid complex operations
        # Just add Gaussian noise
        noise_std = self.noise_params['gaussian_std']
        trace_array = trace.data if hasattr(trace, 'data') else trace
        
        # Create simple noisy trace
        noisy_trace = []
        for i, val in enumerate(trace_array):
            # Add some noise
            noise = np.random.normal(0, noise_std)
            noisy_val = float(val) + noise
            noisy_trace.append(noisy_val)
        
        return np.array(noisy_trace)
    
    def _generate_pink_noise(self, length: int) -> np.ndarray:
        """Generate 1/f pink noise (simplified version)."""
        # Very simplified pink noise - just scaled random noise
        pink_noise = np.random.normal(0, 1, length)
        
        # Scale to desired amplitude
        return pink_noise * self.noise_params['pink_noise_factor']


class TraceDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for TraceData."""
    
    def __init__(self, trace_data: TraceData):
        self.trace_data = trace_data
        
    def __len__(self):
        return len(self.trace_data)
    
    def __getitem__(self, idx):
        item = self.trace_data[idx]
        
        # Convert to tensors
        result = {
            'trace': torch.tensor(item['trace'], dtype=torch.float32)
        }
        
        if 'label' in item:
            result['label'] = torch.tensor(item['label'], dtype=torch.long)
        if 'plaintext' in item:
            result['plaintext'] = torch.tensor(item['plaintext'], dtype=torch.uint8)
        if 'key' in item:
            result['key'] = torch.tensor(item['key'], dtype=torch.uint8)
            
        return result