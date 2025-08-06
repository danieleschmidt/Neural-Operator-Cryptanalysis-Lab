"""Validation utilities for neural cryptanalysis components."""

import numpy as np
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import re


class ValidationError(Exception):
    """Custom exception for validation errors."""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        super().__init__(message)
        self.field = field
        self.value = value
        self.message = message
    
    def __str__(self):
        if self.field:
            return f"Validation error in '{self.field}': {self.message}"
        return f"Validation error: {self.message}"


class SecurityValidationError(ValidationError):
    """Exception for security-related validation failures."""
    pass


class PerformanceWarning(UserWarning):
    """Warning for potential performance issues."""
    pass


class SecurityWarning(UserWarning):
    """Warning for potential security issues."""
    pass


def validate_trace_data(traces: np.ndarray, labels: Optional[np.ndarray] = None,
                       plaintexts: Optional[np.ndarray] = None,
                       keys: Optional[np.ndarray] = None) -> List[str]:
    """Validate side-channel trace data.
    
    Args:
        traces: Trace array [n_traces, trace_length]
        labels: Optional labels array [n_traces]
        plaintexts: Optional plaintexts array [n_traces, plaintext_length]
        keys: Optional keys array [n_traces, key_length]
        
    Returns:
        List of validation warnings/errors
    """
    issues = []
    
    # Validate traces
    if not isinstance(traces, np.ndarray):
        issues.append("Traces must be numpy array")
        return issues
    
    if traces.ndim != 2:
        issues.append(f"Traces must be 2D array, got {traces.ndim}D")
    
    if traces.size == 0:
        issues.append("Traces array is empty")
    
    n_traces, trace_length = traces.shape
    
    # Check for reasonable trace length
    if trace_length < 100:
        issues.append(f"Trace length {trace_length} is very short, may not contain enough information")
    elif trace_length > 1000000:
        warnings.warn(f"Trace length {trace_length} is very long, may cause memory issues", 
                     PerformanceWarning)
    
    # Check for reasonable number of traces
    if n_traces < 10:
        issues.append(f"Number of traces {n_traces} is very small for meaningful analysis")
    elif n_traces > 1000000:
        warnings.warn(f"Number of traces {n_traces} is very large, may cause memory/performance issues",
                     PerformanceWarning)
    
    # Check for data quality issues
    if np.any(np.isnan(traces)):
        issues.append("Traces contain NaN values")
    
    if np.any(np.isinf(traces)):
        issues.append("Traces contain infinite values")
    
    # Check for constant traces (likely measurement errors)
    constant_traces = np.var(traces, axis=1) == 0
    if np.any(constant_traces):
        n_constant = np.sum(constant_traces)
        issues.append(f"{n_constant} traces are constant (no variation)")
    
    # Check dynamic range
    trace_ranges = np.ptp(traces, axis=1)  # Peak-to-peak
    if np.min(trace_ranges) == 0:
        issues.append("Some traces have zero dynamic range")
    
    avg_range = np.mean(trace_ranges)
    if avg_range < 1e-6:
        warnings.warn("Very small signal dynamic range, may indicate measurement issues",
                     UserWarning)
    
    # Validate labels if provided
    if labels is not None:
        if not isinstance(labels, np.ndarray):
            issues.append("Labels must be numpy array")
        elif len(labels) != n_traces:
            issues.append(f"Labels length {len(labels)} doesn't match traces {n_traces}")
        elif labels.dtype not in [np.uint8, np.int32, np.int64]:
            issues.append(f"Labels should be integer type, got {labels.dtype}")
        elif np.any(labels < 0) or np.any(labels > 255):
            issues.append("Labels should be in range 0-255 for key byte values")
    
    # Validate plaintexts if provided
    if plaintexts is not None:
        if not isinstance(plaintexts, np.ndarray):
            issues.append("Plaintexts must be numpy array")
        elif len(plaintexts) != n_traces:
            issues.append(f"Plaintexts length {len(plaintexts)} doesn't match traces {n_traces}")
        elif plaintexts.dtype != np.uint8:
            issues.append(f"Plaintexts should be uint8, got {plaintexts.dtype}")
    
    # Validate keys if provided
    if keys is not None:
        if not isinstance(keys, np.ndarray):
            issues.append("Keys must be numpy array")
        elif len(keys) != n_traces:
            issues.append(f"Keys length {len(keys)} doesn't match traces {n_traces}")
        elif keys.dtype != np.uint8:
            issues.append(f"Keys should be uint8, got {keys.dtype}")
    
    return issues


def validate_neural_operator_config(config: Dict[str, Any]) -> List[str]:
    """Validate neural operator configuration.
    
    Args:
        config: Neural operator configuration dictionary
        
    Returns:
        List of validation issues
    """
    issues = []
    
    # Required fields
    required_fields = ['input_dim', 'output_dim', 'hidden_dim', 'num_layers']
    for field in required_fields:
        if field not in config:
            issues.append(f"Missing required field: {field}")
        elif not isinstance(config[field], int) or config[field] <= 0:
            issues.append(f"{field} must be positive integer")
    
    # Validate dimensions
    if config.get('input_dim', 0) > 10000:
        warnings.warn(f"Very high input dimension {config['input_dim']}, may cause memory issues",
                     PerformanceWarning)
    
    if config.get('output_dim', 0) > 1000:
        warnings.warn(f"High output dimension {config['output_dim']}, ensure this is correct for your task",
                     PerformanceWarning)
    
    if config.get('num_layers', 0) > 20:
        warnings.warn(f"Very deep network with {config['num_layers']} layers, may be hard to train",
                     PerformanceWarning)
    
    # Validate activation
    valid_activations = ['relu', 'gelu', 'elu', 'leaky_relu', 'silu', 'tanh', 'sigmoid']
    if 'activation' in config and config['activation'] not in valid_activations:
        issues.append(f"Invalid activation {config['activation']}, must be one of: {valid_activations}")
    
    # Validate dropout
    if 'dropout' in config:
        dropout = config['dropout']
        if not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            issues.append(f"Dropout must be float in [0, 1), got {dropout}")
    
    # Validate device
    if 'device' in config and config['device'] not in ['cpu', 'cuda', 'auto']:
        issues.append(f"Invalid device {config['device']}, must be 'cpu', 'cuda', or 'auto'")
    
    return issues


def validate_attack_parameters(n_traces: int, confidence_threshold: float,
                              max_iterations: int = 1000000) -> List[str]:
    """Validate attack parameters for security compliance.
    
    Args:
        n_traces: Number of traces to use in attack
        confidence_threshold: Confidence threshold for attack success
        max_iterations: Maximum allowed attack iterations
        
    Returns:
        List of validation issues
    """
    issues = []
    
    # Security validation - prevent excessive attacks
    if n_traces > max_iterations:
        raise SecurityValidationError(
            f"Attack with {n_traces} traces exceeds security limit of {max_iterations}",
            field="n_traces",
            value=n_traces
        )
    
    # Performance warnings
    if n_traces > 100000:
        warnings.warn(f"Large number of traces ({n_traces}) may require significant compute resources",
                     PerformanceWarning)
    
    if confidence_threshold < 0.1:
        warnings.warn(f"Very low confidence threshold {confidence_threshold} may lead to false positives",
                     UserWarning)
    elif confidence_threshold > 0.99:
        warnings.warn(f"Very high confidence threshold {confidence_threshold} may be too restrictive",
                     UserWarning)
    
    return issues


def validate_cryptographic_implementation(algorithm: str, variant: str,
                                       countermeasures: List[str]) -> List[str]:
    """Validate cryptographic implementation parameters.
    
    Args:
        algorithm: Cryptographic algorithm name
        variant: Algorithm variant
        countermeasures: List of enabled countermeasures
        
    Returns:
        List of validation issues
    """
    issues = []
    
    # Validate algorithm
    supported_algorithms = [
        'kyber', 'dilithium', 'classic_mceliece', 'sphincs',
        'aes', 'rsa', 'ecdsa'
    ]
    
    if algorithm not in supported_algorithms:
        issues.append(f"Unsupported algorithm {algorithm}, supported: {supported_algorithms}")
    
    # Validate variants for specific algorithms
    variant_mappings = {
        'kyber': ['kyber512', 'kyber768', 'kyber1024'],
        'dilithium': ['dilithium2', 'dilithium3', 'dilithium5'],
        'classic_mceliece': ['mceliece348864', 'mceliece460896', 'mceliece6688128'],
        'aes': ['aes128', 'aes192', 'aes256']
    }
    
    if algorithm in variant_mappings and variant not in variant_mappings[algorithm]:
        issues.append(f"Invalid variant {variant} for {algorithm}, valid: {variant_mappings[algorithm]}")
    
    # Validate countermeasures
    valid_countermeasures = ['masking', 'shuffling', 'hiding', 'blinding', 'dummy_operations']
    invalid_cm = [cm for cm in countermeasures if cm not in valid_countermeasures]
    if invalid_cm:
        issues.append(f"Invalid countermeasures: {invalid_cm}, valid: {valid_countermeasures}")
    
    # Security warnings
    if not countermeasures:
        warnings.warn("No countermeasures enabled - implementation may be vulnerable to side-channel attacks",
                     SecurityWarning)
    
    if 'masking' in countermeasures and 'shuffling' not in countermeasures:
        warnings.warn("Masking without shuffling may still be vulnerable to some attacks",
                     SecurityWarning)
    
    return issues


def validate_experimental_setup(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Comprehensive validation of experimental setup.
    
    Args:
        config: Complete experimental configuration
        
    Returns:
        Tuple of (errors, warnings)
    """
    errors = []
    warnings_list = []
    
    # Validate required sections
    required_sections = ['neural_operator', 'side_channel', 'target', 'training']
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required configuration section: {section}")
    
    # Validate individual sections
    if 'neural_operator' in config:
        no_issues = validate_neural_operator_config(config['neural_operator'])
        errors.extend([issue for issue in no_issues if 'must' in issue])
        warnings_list.extend([issue for issue in no_issues if 'must' not in issue])
    
    if 'target' in config:
        target_config = config['target']
        target_issues = validate_cryptographic_implementation(
            target_config.get('algorithm', ''),
            target_config.get('variant', ''),
            target_config.get('countermeasures', [])
        )
        errors.extend([issue for issue in target_issues if 'Unsupported' in issue or 'Invalid' in issue])
        warnings_list.extend([issue for issue in target_issues if 'warning' in issue.lower()])
    
    # Cross-validation between sections
    if 'neural_operator' in config and 'side_channel' in config:
        # Check consistency between neural operator output and target classes
        output_dim = config['neural_operator'].get('output_dim', 0)
        if output_dim != 256:
            warnings_list.append(f"Neural operator output_dim {output_dim} != 256 may not be suitable for key byte prediction")
    
    # Resource validation
    if 'training' in config:
        batch_size = config['training'].get('batch_size', 64)
        if 'side_channel' in config:
            trace_length = config['side_channel'].get('trace_length', 10000)
            estimated_memory_gb = (batch_size * trace_length * 4) / (1024**3)  # Rough estimate
            
            if estimated_memory_gb > 8:
                warnings_list.append(f"Estimated memory usage {estimated_memory_gb:.1f}GB may exceed system limits")
    
    return errors, warnings_list


def sanitize_input_string(input_str: str, max_length: int = 1000,
                         allowed_chars: str = None) -> str:
    """Sanitize input string for security.
    
    Args:
        input_str: Input string to sanitize
        max_length: Maximum allowed length
        allowed_chars: Regex pattern for allowed characters
        
    Returns:
        Sanitized string
        
    Raises:
        ValidationError: If input is invalid
    """
    if not isinstance(input_str, str):
        raise ValidationError("Input must be string", value=input_str)
    
    if len(input_str) > max_length:
        raise ValidationError(f"Input too long: {len(input_str)} > {max_length}",
                            value=input_str)
    
    # Default allowed characters: alphanumeric, dash, underscore, dot
    if allowed_chars is None:
        allowed_chars = r'^[a-zA-Z0-9\-_\.]+$'
    
    if not re.match(allowed_chars, input_str):
        raise ValidationError(f"Input contains invalid characters: {input_str}",
                            value=input_str)
    
    return input_str


def validate_file_path(path: Union[str, Path], must_exist: bool = True,
                      allowed_extensions: List[str] = None) -> Path:
    """Validate and sanitize file path.
    
    Args:
        path: File path to validate
        must_exist: Whether file must exist
        allowed_extensions: List of allowed file extensions
        
    Returns:
        Validated Path object
        
    Raises:
        ValidationError: If path is invalid
    """
    if isinstance(path, str):
        path = Path(path)
    
    if not isinstance(path, Path):
        raise ValidationError("Path must be string or Path object", value=path)
    
    # Security check - prevent path traversal
    if '..' in str(path):
        raise SecurityValidationError("Path traversal not allowed", value=str(path))
    
    # Check if file exists
    if must_exist and not path.exists():
        raise ValidationError(f"File does not exist: {path}", value=str(path))
    
    # Check extension
    if allowed_extensions and path.suffix.lower() not in allowed_extensions:
        raise ValidationError(f"Invalid file extension {path.suffix}, allowed: {allowed_extensions}",
                            value=str(path))
    
    return path


def validate_numeric_range(value: Union[int, float], min_val: float = None,
                          max_val: float = None, name: str = "value") -> Union[int, float]:
    """Validate numeric value is within range.
    
    Args:
        value: Numeric value to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        name: Name of the parameter for error messages
        
    Returns:
        Validated value
        
    Raises:
        ValidationError: If value is out of range
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be numeric, got {type(value)}", 
                            field=name, value=value)
    
    if min_val is not None and value < min_val:
        raise ValidationError(f"{name} must be >= {min_val}, got {value}",
                            field=name, value=value)
    
    if max_val is not None and value > max_val:
        raise ValidationError(f"{name} must be <= {max_val}, got {value}",
                            field=name, value=value)
    
    return value


def check_responsible_use_compliance(experiment_config: Dict[str, Any]) -> List[str]:
    """Check experiment compliance with responsible use guidelines.
    
    Args:
        experiment_config: Complete experiment configuration
        
    Returns:
        List of compliance issues
    """
    issues = []
    
    # Check if responsible disclosure is enabled
    security_config = experiment_config.get('security', {})
    if not security_config.get('enable_responsible_disclosure', True):
        issues.append("Responsible disclosure must be enabled for ethical research")
    
    # Check for excessive attack parameters
    side_channel_config = experiment_config.get('side_channel', {})
    n_traces = side_channel_config.get('n_traces', 0)
    
    if n_traces > 10000000:  # 10M traces
        issues.append(f"Excessive number of traces ({n_traces}) may indicate misuse")
    
    # Check if authorization is required
    if not security_config.get('require_authorization', True):
        issues.append("Authorization requirement should be enabled")
    
    # Check if audit logging is enabled
    if not security_config.get('audit_logging', True):
        issues.append("Audit logging should be enabled for security compliance")
    
    # Check experiment description for inappropriate content
    experiment_info = experiment_config.get('experiment', {})
    description = experiment_info.get('description', '').lower()
    
    prohibited_terms = ['attack', 'exploit', 'break', 'crack', 'hack']
    found_terms = [term for term in prohibited_terms if term in description]
    
    if found_terms:
        warnings.warn(f"Experiment description contains potentially concerning terms: {found_terms}. "
                     "Ensure this is for defensive research only.", SecurityWarning)
    
    return issues


class ValidationContext:
    """Context manager for validation with detailed error reporting."""
    
    def __init__(self, component: str = "Unknown"):
        self.component = component
        self.errors = []
        self.warnings = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is ValidationError:
            self.errors.append(str(exc_val))
            return True  # Suppress the exception
        
        if self.errors:
            error_msg = f"Validation failed for {self.component}:\n" + "\n".join(f"- {e}" for e in self.errors)
            if self.warnings:
                error_msg += f"\nWarnings:\n" + "\n".join(f"- {w}" for w in self.warnings)
            raise ValidationError(error_msg)
    
    def add_error(self, message: str):
        """Add validation error."""
        self.errors.append(message)
    
    def add_warning(self, message: str):
        """Add validation warning."""
        self.warnings.append(message)
    
    def validate(self, condition: bool, message: str):
        """Add error if condition is False."""
        if not condition:
            self.add_error(message)
    
    def warn_if(self, condition: bool, message: str):
        """Add warning if condition is True."""
        if condition:
            self.add_warning(message)