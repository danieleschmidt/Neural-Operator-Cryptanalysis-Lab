"""Configuration management utilities."""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import os


class ConfigManager:
    """Configuration manager for neural cryptanalysis experiments."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else None
        self.config = {}
        
        if self.config_path and self.config_path.exists():
            self.load()
    
    def load(self, path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Load configuration from file."""
        if path:
            self.config_path = Path(path)
        
        if not self.config_path or not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            if self.config_path.suffix.lower() in ['.yml', '.yaml']:
                self.config = yaml.safe_load(f)
            elif self.config_path.suffix.lower() == '.json':
                self.config = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {self.config_path.suffix}")
        
        return self.config
    
    def save(self, path: Optional[Union[str, Path]] = None):
        """Save configuration to file."""
        if path:
            self.config_path = Path(path)
        
        if not self.config_path:
            raise ValueError("No config path specified")
        
        # Create directory if it doesn't exist
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            if self.config_path.suffix.lower() in ['.yml', '.yaml']:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            elif self.config_path.suffix.lower() == '.json':
                json.dump(self.config, f, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {self.config_path.suffix}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by key."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with dictionary."""
        def recursive_update(d: dict, u: dict):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = recursive_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = recursive_update(self.config, updates)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()
    
    def from_dict(self, config_dict: Dict[str, Any]):
        """Load configuration from dictionary."""
        self.config = config_dict.copy()


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from file.
    
    Args:
        path: Path to configuration file (JSON or YAML)
        
    Returns:
        Configuration dictionary
    """
    manager = ConfigManager(path)
    return manager.to_dict()


def save_config(config: Dict[str, Any], path: Union[str, Path]):
    """Save configuration to file.
    
    Args:
        config: Configuration dictionary
        path: Path to save configuration file
    """
    manager = ConfigManager()
    manager.from_dict(config)
    manager.save(path)


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for neural cryptanalysis."""
    return {
        "experiment": {
            "name": "neural_cryptanalysis_experiment",
            "description": "Neural operator based side-channel analysis",
            "version": "1.0.0"
        },
        "neural_operator": {
            "architecture": "fourier_neural_operator",
            "input_dim": 1,
            "output_dim": 256,
            "hidden_dim": 64,
            "num_layers": 4,
            "modes": 16,
            "activation": "gelu",
            "dropout": 0.1,
            "normalization": "layer",
            "use_residual": True,
            "device": "cpu"
        },
        "side_channel": {
            "channel_type": "power",
            "attack_type": "neural",
            "sample_rate": 1000000.0,
            "trace_length": 10000,
            "n_traces": 10000,
            "preprocessing": ["standardize"],
            "poi_method": "mutual_information",
            "n_pois": 100,
            "confidence_threshold": 0.9
        },
        "target": {
            "algorithm": "kyber",
            "variant": "kyber768",
            "platform": "arm_cortex_m4",
            "optimization": "speed",
            "countermeasures": [],
            "parameters": {}
        },
        "training": {
            "batch_size": 64,
            "learning_rate": 0.001,
            "epochs": 100,
            "patience": 10,
            "min_delta": 1e-6,
            "validation_split": 0.2,
            "optimizer": "adam",
            "scheduler": "reduce_on_plateau",
            "loss_function": "cross_entropy"
        },
        "data": {
            "data_dir": "./data",
            "output_dir": "./output",
            "cache_dir": "./cache",
            "trace_format": "numpy",
            "metadata_format": "json",
            "compression": "gzip"
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "log_file": "neural_cryptanalysis.log",
            "max_file_size": "10MB",
            "backup_count": 5
        },
        "visualization": {
            "backend": "matplotlib",
            "style": "seaborn",
            "figure_size": [12, 8],
            "dpi": 150,
            "color_palette": "viridis",
            "save_format": "png"
        },
        "security": {
            "enable_responsible_disclosure": True,
            "max_attack_iterations": 1000000,
            "require_authorization": True,
            "audit_logging": True
        },
        "performance": {
            "enable_profiling": False,
            "memory_limit": "8GB",
            "cpu_threads": -1,
            "gpu_memory_fraction": 0.8,
            "mixed_precision": False
        }
    }


def create_experiment_config(
    experiment_name: str,
    architecture: str = "fourier_neural_operator",
    target_algorithm: str = "kyber",
    **kwargs
) -> Dict[str, Any]:
    """Create experiment configuration with common parameters.
    
    Args:
        experiment_name: Name of the experiment
        architecture: Neural operator architecture
        target_algorithm: Target cryptographic algorithm
        **kwargs: Additional configuration overrides
        
    Returns:
        Experiment configuration dictionary
    """
    config = get_default_config()
    
    # Update experiment info
    config["experiment"]["name"] = experiment_name
    config["neural_operator"]["architecture"] = architecture
    config["target"]["algorithm"] = target_algorithm
    
    # Apply additional overrides
    manager = ConfigManager()
    manager.from_dict(config)
    manager.update(kwargs)
    
    return manager.to_dict()


# Alias for backwards compatibility  
Config = ConfigManager


def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate configuration and return list of issues.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        List of validation error messages
    """
    issues = []
    
    # Required sections
    required_sections = [
        "neural_operator", "side_channel", "target", "training"
    ]
    
    for section in required_sections:
        if section not in config:
            issues.append(f"Missing required section: {section}")
    
    # Neural operator validation
    if "neural_operator" in config:
        no_config = config["neural_operator"]
        
        if "architecture" not in no_config:
            issues.append("Missing neural_operator.architecture")
        
        if no_config.get("input_dim", 0) <= 0:
            issues.append("neural_operator.input_dim must be positive")
        
        if no_config.get("output_dim", 0) <= 0:
            issues.append("neural_operator.output_dim must be positive")
    
    # Side channel validation
    if "side_channel" in config:
        sc_config = config["side_channel"]
        
        valid_channels = ["power", "em_near", "em_far", "acoustic", "optical"]
        if sc_config.get("channel_type") not in valid_channels:
            issues.append(f"Invalid channel_type, must be one of: {valid_channels}")
        
        if sc_config.get("trace_length", 0) <= 0:
            issues.append("side_channel.trace_length must be positive")
        
        if sc_config.get("n_traces", 0) <= 0:
            issues.append("side_channel.n_traces must be positive")
    
    # Training validation
    if "training" in config:
        train_config = config["training"]
        
        if train_config.get("batch_size", 0) <= 0:
            issues.append("training.batch_size must be positive")
        
        if not (0 < train_config.get("learning_rate", 0) < 1):
            issues.append("training.learning_rate must be between 0 and 1")
        
        if train_config.get("epochs", 0) <= 0:
            issues.append("training.epochs must be positive")
    
    return issues


def setup_experiment_directory(config: Dict[str, Any]) -> Path:
    """Setup experiment directory structure.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Path to experiment directory
    """
    experiment_name = config["experiment"]["name"]
    base_dir = Path(config.get("data", {}).get("output_dir", "./output"))
    
    exp_dir = base_dir / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    subdirs = ["models", "results", "logs", "plots", "checkpoints"]
    for subdir in subdirs:
        (exp_dir / subdir).mkdir(exist_ok=True)
    
    # Save experiment configuration
    config_path = exp_dir / "config.yaml"
    save_config(config, config_path)
    
    return exp_dir


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    def deep_merge(dict1: dict, dict2: dict) -> dict:
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    merged = {}
    for config in configs:
        merged = deep_merge(merged, config)
    
    return merged


def config_from_env(prefix: str = "NEURAL_CRYPTO_") -> Dict[str, Any]:
    """Load configuration from environment variables.
    
    Args:
        prefix: Prefix for environment variables
        
    Returns:
        Configuration dictionary from environment
    """
    config = {}
    
    for key, value in os.environ.items():
        if key.startswith(prefix):
            # Remove prefix and convert to lowercase
            config_key = key[len(prefix):].lower()
            
            # Convert dots to nested structure
            keys = config_key.split('_')
            current = config
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Try to parse value as appropriate type
            try:
                if value.lower() in ['true', 'false']:
                    current[keys[-1]] = value.lower() == 'true'
                elif value.isdigit():
                    current[keys[-1]] = int(value)
                elif '.' in value and all(part.isdigit() for part in value.split('.')):
                    current[keys[-1]] = float(value)
                else:
                    current[keys[-1]] = value
            except ValueError:
                current[keys[-1]] = value
    
    return config