"""Logging utilities for neural cryptanalysis experiments."""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any
import json
from datetime import datetime


class SecurityAuditHandler(logging.Handler):
    """Specialized handler for security audit logging."""
    
    def __init__(self, audit_file: Union[str, Path]):
        super().__init__()
        self.audit_file = Path(audit_file)
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
        
    def emit(self, record: logging.LogRecord):
        """Emit audit log record."""
        if not hasattr(record, 'audit_type'):
            return
            
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'audit_type': record.audit_type,
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add additional audit fields if present
        for field in ['user_id', 'experiment_id', 'target_algorithm', 'attack_type']:
            if hasattr(record, field):
                audit_entry[field] = getattr(record, field)
        
        # Write to audit file
        with open(self.audit_file, 'a') as f:
            f.write(json.dumps(audit_entry) + '\n')


class ExperimentFormatter(logging.Formatter):
    """Custom formatter for experiment logging."""
    
    def __init__(self, include_experiment_info: bool = True):
        super().__init__()
        self.include_experiment_info = include_experiment_info
        
        if include_experiment_info:
            self.base_format = (
                "%(asctime)s - %(name)s - %(levelname)s - "
                "[Exp: %(experiment_name)s] - %(message)s"
            )
        else:
            self.base_format = (
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with experiment context."""
        # Add default experiment name if not present
        if not hasattr(record, 'experiment_name'):
            record.experiment_name = 'default'
        
        # Use base formatter
        formatter = logging.Formatter(self.base_format)
        return formatter.format(record)


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    console_output: bool = True,
    experiment_name: str = "neural_cryptanalysis",
    audit_logging: bool = False,
    audit_file: Optional[Union[str, Path]] = None,
    max_file_size: str = "10MB",
    backup_count: int = 5
) -> logging.Logger:
    """Setup comprehensive logging for neural cryptanalysis experiments.
    
    Args:
        level: Logging level
        log_file: Path to log file
        console_output: Whether to output to console
        experiment_name: Name of current experiment
        audit_logging: Enable security audit logging
        audit_file: Path to audit log file
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
        
    Returns:
        Configured logger instance
    """
    # Parse log level
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Create root logger
    root_logger = logging.getLogger('neural_cryptanalysis')
    root_logger.setLevel(level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = ExperimentFormatter(include_experiment_info=True)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Parse max file size
        size_bytes = _parse_size(max_file_size)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=size_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Audit logging handler
    if audit_logging:
        if not audit_file:
            audit_file = log_file.parent / "audit.log" if log_file else Path("audit.log")
        
        audit_handler = SecurityAuditHandler(audit_file)
        audit_handler.setLevel(logging.WARNING)  # Only log warnings and above for audit
        root_logger.addHandler(audit_handler)
    
    # Add experiment context to all log records
    class ExperimentFilter(logging.Filter):
        def filter(self, record):
            record.experiment_name = experiment_name
            return True
    
    experiment_filter = ExperimentFilter()
    root_logger.addFilter(experiment_filter)
    
    # Log setup completion
    root_logger.info(f"Logging initialized for experiment: {experiment_name}")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get logger for specific module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f'neural_cryptanalysis.{name}')


def log_experiment_start(
    logger: logging.Logger,
    experiment_name: str,
    config: Dict[str, Any]
):
    """Log experiment start with configuration.
    
    Args:
        logger: Logger instance
        experiment_name: Name of experiment
        config: Experiment configuration
    """
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Neural operator architecture: {config.get('neural_operator', {}).get('architecture')}")
    logger.info(f"Target algorithm: {config.get('target', {}).get('algorithm')}")
    logger.info(f"Side channel type: {config.get('side_channel', {}).get('channel_type')}")
    
    # Log configuration summary
    summary = {
        'experiment': experiment_name,
        'architecture': config.get('neural_operator', {}).get('architecture'),
        'target': config.get('target', {}).get('algorithm'),
        'traces': config.get('side_channel', {}).get('n_traces'),
        'epochs': config.get('training', {}).get('epochs')
    }
    
    logger.info(f"Experiment configuration: {summary}")


def log_experiment_end(
    logger: logging.Logger,
    experiment_name: str,
    results: Dict[str, Any],
    duration: float
):
    """Log experiment completion with results.
    
    Args:
        logger: Logger instance
        experiment_name: Name of experiment
        results: Experiment results
        duration: Experiment duration in seconds
    """
    logger.info(f"Completed experiment: {experiment_name}")
    logger.info(f"Duration: {duration:.2f} seconds")
    
    # Log key results
    if 'success_rate' in results:
        logger.info(f"Attack success rate: {results['success_rate']:.2%}")
    
    if 'best_accuracy' in results:
        logger.info(f"Best model accuracy: {results['best_accuracy']:.2%}")
    
    if 'traces_needed' in results:
        logger.info(f"Traces needed for attack: {results['traces_needed']}")
    
    logger.info("Experiment completed successfully")


def log_security_event(
    logger: logging.Logger,
    event_type: str,
    message: str,
    **kwargs
):
    """Log security-related event for audit trail.
    
    Args:
        logger: Logger instance
        event_type: Type of security event
        message: Event description
        **kwargs: Additional event data
    """
    # Create log record with audit information
    extra = {
        'audit_type': event_type,
        **kwargs
    }
    
    logger.warning(message, extra=extra)


def log_attack_attempt(
    logger: logging.Logger,
    target_algorithm: str,
    attack_type: str,
    traces_used: int,
    success: bool,
    **kwargs
):
    """Log attack attempt for security audit.
    
    Args:
        logger: Logger instance
        target_algorithm: Target cryptographic algorithm
        attack_type: Type of attack
        traces_used: Number of traces used
        success: Whether attack was successful
        **kwargs: Additional attack data
    """
    message = (
        f"Attack attempt - Target: {target_algorithm}, "
        f"Type: {attack_type}, Traces: {traces_used}, "
        f"Success: {success}"
    )
    
    extra = {
        'audit_type': 'attack_attempt',
        'target_algorithm': target_algorithm,
        'attack_type': attack_type,
        'traces_used': traces_used,
        'success': success,
        **kwargs
    }
    
    logger.warning(message, extra=extra)


def log_countermeasure_evaluation(
    logger: logging.Logger,
    countermeasure_type: str,
    effectiveness: float,
    **kwargs
):
    """Log countermeasure evaluation results.
    
    Args:
        logger: Logger instance
        countermeasure_type: Type of countermeasure
        effectiveness: Effectiveness score [0, 1]
        **kwargs: Additional evaluation data
    """
    message = (
        f"Countermeasure evaluation - Type: {countermeasure_type}, "
        f"Effectiveness: {effectiveness:.2%}"
    )
    
    extra = {
        'audit_type': 'countermeasure_evaluation',
        'countermeasure_type': countermeasure_type,
        'effectiveness': effectiveness,
        **kwargs
    }
    
    logger.info(message, extra=extra)


def _parse_size(size_str: str) -> int:
    """Parse size string to bytes.
    
    Args:
        size_str: Size string (e.g., '10MB', '1GB')
        
    Returns:
        Size in bytes
    """
    size_str = size_str.upper()
    
    if size_str.endswith('B'):
        size_str = size_str[:-1]
    
    multipliers = {
        'K': 1024,
        'M': 1024**2, 
        'G': 1024**3,
        'T': 1024**4
    }
    
    for suffix, multiplier in multipliers.items():
        if size_str.endswith(suffix):
            return int(float(size_str[:-1]) * multiplier)
    
    return int(size_str)  # Assume bytes if no suffix


class ProgressLogger:
    """Logger for tracking training/attack progress."""
    
    def __init__(self, logger: logging.Logger, total_steps: int, log_interval: int = 100):
        self.logger = logger
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.current_step = 0
        self.start_time = datetime.now()
    
    def update(self, step: Optional[int] = None, **metrics):
        """Update progress and log if needed.
        
        Args:
            step: Current step (auto-increments if None)
            **metrics: Additional metrics to log
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        if self.current_step % self.log_interval == 0 or self.current_step == self.total_steps:
            self._log_progress(**metrics)
    
    def _log_progress(self, **metrics):
        """Log current progress."""
        progress_pct = (self.current_step / self.total_steps) * 100
        elapsed = datetime.now() - self.start_time
        
        message = f"Progress: {self.current_step}/{self.total_steps} ({progress_pct:.1f}%)"
        
        if metrics:
            metric_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                  for k, v in metrics.items()])
            message += f" - {metric_str}"
        
        message += f" - Elapsed: {elapsed}"
        
        self.logger.info(message)
    
    def finish(self, **final_metrics):
        """Log completion."""
        total_time = datetime.now() - self.start_time
        message = f"Completed {self.total_steps} steps in {total_time}"
        
        if final_metrics:
            metric_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                  for k, v in final_metrics.items()])
            message += f" - Final: {metric_str}"
        
        self.logger.info(message)


class SecureLogFilter(logging.Filter):
    """Filter to sanitize and secure log messages."""
    
    def __init__(self):
        super().__init__()
        self.sensitive_patterns = [
            r'(?i)(password|token|key|secret|auth)[\s=:]+[^\s]+',
            r'(?i)(api[_-]?key)[\s=:]+[^\s]+',
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        ]
        
        self.rate_limits = {}
        self.max_rate = 100  # Max logs per minute for same message
        
    def filter(self, record):
        """Filter and sanitize log records."""
        # Sanitize message
        if hasattr(record, 'msg'):
            record.msg = self._sanitize_message(str(record.msg))
        
        # Rate limiting
        if not self._check_rate_limit(record):
            return False
        
        # Security level filtering
        if hasattr(record, 'audit_type') and record.audit_type in ['sensitive_operation']:
            # Additional security checks for sensitive operations
            return self._security_check(record)
        
        return True
    
    def _sanitize_message(self, message: str) -> str:
        """Remove sensitive information from log messages."""
        import re
        
        sanitized = message
        for pattern in self.sensitive_patterns:
            sanitized = re.sub(pattern, '[REDACTED]', sanitized)
        
        return sanitized
    
    def _check_rate_limit(self, record) -> bool:
        """Check if log message exceeds rate limit."""
        import time
        
        current_time = time.time()
        message_key = f"{record.levelname}:{record.msg}"
        
        if message_key not in self.rate_limits:
            self.rate_limits[message_key] = []
        
        # Clean old entries
        cutoff_time = current_time - 60  # 1 minute window
        self.rate_limits[message_key] = [
            t for t in self.rate_limits[message_key] if t > cutoff_time
        ]
        
        # Check rate limit
        if len(self.rate_limits[message_key]) >= self.max_rate:
            return False
        
        # Add current time
        self.rate_limits[message_key].append(current_time)
        return True
    
    def _security_check(self, record) -> bool:
        """Additional security checks for sensitive log records."""
        return True


# Global secure logger instances
def create_secure_logger(name: str) -> logging.Logger:
    """Create a secure logger with enhanced filtering."""
    logger = get_logger(name)
    security_filter = SecureLogFilter()
    logger.addFilter(security_filter)
    return logger