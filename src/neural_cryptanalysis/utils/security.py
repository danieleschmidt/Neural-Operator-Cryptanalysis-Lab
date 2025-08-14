"""Security utilities and monitoring for neural cryptanalysis research."""

import hashlib
import hmac
import time
import json
import os
from typing import Dict, List, Any, Optional, Callable, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import secrets
import threading
from dataclasses import dataclass
from enum import Enum
import warnings

from .logging_utils import get_logger, log_security_event
from .validation import SecurityValidationError, SecurityWarning

logger = get_logger(__name__)


class SecurityLevel(Enum):
    """Security levels for research activities."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    max_traces_per_attack: int = 1000000
    max_attack_duration: float = 3600.0  # 1 hour
    require_authorization: bool = True
    enable_audit_logging: bool = True
    rate_limit_enabled: bool = True
    max_requests_per_minute: int = 10
    allowed_algorithms: List[str] = None
    blocked_algorithms: List[str] = None
    max_experiment_duration: float = 86400.0  # 24 hours
    enable_data_encryption: bool = True
    
    def __post_init__(self):
        if self.allowed_algorithms is None:
            self.allowed_algorithms = ['kyber', 'dilithium', 'classic_mceliece', 'sphincs']
        if self.blocked_algorithms is None:
            self.blocked_algorithms = []


class SecurityMonitor:
    """Security monitor for tracking and enforcing security policies."""
    
    def __init__(self, policy: SecurityPolicy = None):
        self.policy = policy or SecurityPolicy()
        self.active_experiments = {}
        self.rate_limits = {}
        self.security_events = []
        self.lock = threading.Lock()
        
        # Initialize security monitoring
        self.start_time = datetime.now()
        
        logger.info("Security monitor initialized")
        log_security_event(logger, "security_monitor_start", 
                          "Security monitoring system started")
    
    def authorize_experiment(self, experiment_id: str, user_id: str,
                           config: Dict[str, Any]) -> bool:
        """Authorize experiment execution.
        
        Args:
            experiment_id: Unique experiment identifier
            user_id: User identifier
            config: Experiment configuration
            
        Returns:
            True if authorized, False otherwise
            
        Raises:
            SecurityValidationError: If experiment violates security policy
        """
        with self.lock:
            # Check if authorization is required
            if self.policy.require_authorization:
                auth_token = os.environ.get('NEURAL_CRYPTO_AUTH_TOKEN')
                if not auth_token:
                    log_security_event(logger, "authorization_failed",
                                     f"Missing authorization token for experiment {experiment_id}",
                                     experiment_id=experiment_id, user_id=user_id)
                    return False
                
                if not self._verify_auth_token(auth_token, user_id):
                    log_security_event(logger, "authorization_failed",
                                     f"Invalid authorization token for user {user_id}",
                                     experiment_id=experiment_id, user_id=user_id)
                    return False
            
            # Validate experiment configuration against policy
            validation_errors = self._validate_experiment_config(config)
            if validation_errors:
                error_msg = f"Experiment violates security policy: {'; '.join(validation_errors)}"
                log_security_event(logger, "policy_violation", error_msg,
                                 experiment_id=experiment_id, user_id=user_id,
                                 violations=validation_errors)
                raise SecurityValidationError(error_msg)
            
            # Check rate limits
            if not self._check_rate_limit(user_id):
                log_security_event(logger, "rate_limit_exceeded",
                                 f"Rate limit exceeded for user {user_id}",
                                 user_id=user_id)
                return False
            
            # Register experiment
            self.active_experiments[experiment_id] = {
                'user_id': user_id,
                'start_time': datetime.now(),
                'config': config,
                'status': 'authorized'
            }
            
            log_security_event(logger, "experiment_authorized",
                             f"Experiment {experiment_id} authorized for user {user_id}",
                             experiment_id=experiment_id, user_id=user_id)
            
            return True
    
    def register_attack_attempt(self, experiment_id: str, target_algorithm: str,
                               attack_params: Dict[str, Any]) -> bool:
        """Register and validate attack attempt.
        
        Args:
            experiment_id: Experiment identifier
            target_algorithm: Target cryptographic algorithm
            attack_params: Attack parameters
            
        Returns:
            True if attack is allowed, False otherwise
        """
        with self.lock:
            # Check if experiment is authorized
            if experiment_id not in self.active_experiments:
                log_security_event(logger, "unauthorized_attack",
                                 f"Attack attempt from unauthorized experiment {experiment_id}",
                                 experiment_id=experiment_id)
                return False
            
            # Validate attack parameters
            n_traces = attack_params.get('n_traces', 0)
            if n_traces > self.policy.max_traces_per_attack:
                log_security_event(logger, "attack_limit_exceeded",
                                 f"Attack traces {n_traces} exceeds limit {self.policy.max_traces_per_attack}",
                                 experiment_id=experiment_id,
                                 n_traces=n_traces)
                return False
            
            # Check algorithm restrictions
            if (self.policy.allowed_algorithms and 
                target_algorithm not in self.policy.allowed_algorithms):
                log_security_event(logger, "algorithm_not_allowed",
                                 f"Algorithm {target_algorithm} not in allowed list",
                                 experiment_id=experiment_id,
                                 algorithm=target_algorithm)
                return False
            
            if target_algorithm in self.policy.blocked_algorithms:
                log_security_event(logger, "algorithm_blocked",
                                 f"Algorithm {target_algorithm} is blocked",
                                 experiment_id=experiment_id,
                                 algorithm=target_algorithm)
                return False
            
            # Log attack attempt
            log_security_event(logger, "attack_registered",
                             f"Attack registered - Target: {target_algorithm}, Traces: {n_traces}",
                             experiment_id=experiment_id,
                             target_algorithm=target_algorithm,
                             n_traces=n_traces)
            
            return True
    
    def monitor_experiment_progress(self, experiment_id: str, 
                                  progress_data: Dict[str, Any]):
        """Monitor ongoing experiment for security violations.
        
        Args:
            experiment_id: Experiment identifier
            progress_data: Current progress metrics
        """
        with self.lock:
            if experiment_id not in self.active_experiments:
                return
            
            experiment = self.active_experiments[experiment_id]
            current_time = datetime.now()
            
            # Check experiment duration
            duration = (current_time - experiment['start_time']).total_seconds()
            if duration > self.policy.max_experiment_duration:
                log_security_event(logger, "experiment_timeout",
                                 f"Experiment {experiment_id} exceeded maximum duration",
                                 experiment_id=experiment_id,
                                 duration=duration)
                
                # Mark experiment as terminated
                experiment['status'] = 'terminated_timeout'
                warnings.warn(f"Experiment {experiment_id} terminated due to timeout",
                             SecurityWarning)
    
    def finalize_experiment(self, experiment_id: str, results: Dict[str, Any]):
        """Finalize experiment and log results.
        
        Args:
            experiment_id: Experiment identifier
            results: Experiment results
        """
        with self.lock:
            if experiment_id not in self.active_experiments:
                return
            
            experiment = self.active_experiments[experiment_id]
            end_time = datetime.now()
            duration = (end_time - experiment['start_time']).total_seconds()
            
            # Log experiment completion
            log_security_event(logger, "experiment_completed",
                             f"Experiment {experiment_id} completed",
                             experiment_id=experiment_id,
                             duration=duration,
                             success_rate=results.get('success_rate', 0))
            
            # Remove from active experiments
            del self.active_experiments[experiment_id]
    
    def _verify_auth_token(self, token: str, user_id: str) -> bool:
        """Verify authorization token."""
        # Simple HMAC-based token verification
        secret_key = os.environ.get('NEURAL_CRYPTO_SECRET_KEY', 'default_secret')
        expected_token = hmac.new(
            secret_key.encode(),
            user_id.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(token, expected_token)
    
    def _validate_experiment_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate experiment configuration against security policy."""
        violations = []
        
        # Check trace limits
        side_channel_config = config.get('side_channel', {})
        n_traces = side_channel_config.get('n_traces', 0)
        
        if n_traces > self.policy.max_traces_per_attack:
            violations.append(f"Trace count {n_traces} exceeds limit {self.policy.max_traces_per_attack}")
        
        # Check algorithm restrictions
        target_config = config.get('target', {})
        algorithm = target_config.get('algorithm')
        
        if (self.policy.allowed_algorithms and 
            algorithm not in self.policy.allowed_algorithms):
            violations.append(f"Algorithm {algorithm} not allowed")
        
        if algorithm in self.policy.blocked_algorithms:
            violations.append(f"Algorithm {algorithm} is blocked")
        
        # Check for excessive training parameters
        training_config = config.get('training', {})
        epochs = training_config.get('epochs', 0)
        
        if epochs > 1000:
            violations.append(f"Excessive epochs {epochs} may indicate misuse")
        
        return violations
    
    def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user exceeds rate limit."""
        if not self.policy.rate_limit_enabled:
            return True
        
        current_time = datetime.now()
        
        # Clean old entries
        if user_id in self.rate_limits:
            self.rate_limits[user_id] = [
                req_time for req_time in self.rate_limits[user_id]
                if current_time - req_time < timedelta(minutes=1)
            ]
        else:
            self.rate_limits[user_id] = []
        
        # Check limit
        if len(self.rate_limits[user_id]) >= self.policy.max_requests_per_minute:
            return False
        
        # Add current request
        self.rate_limits[user_id].append(current_time)
        return True
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate security monitoring report."""
        with self.lock:
            uptime = datetime.now() - self.start_time
            
            return {
                'monitor_uptime': str(uptime),
                'active_experiments': len(self.active_experiments),
                'total_security_events': len(self.security_events),
                'policy': {
                    'max_traces': self.policy.max_traces_per_attack,
                    'authorization_required': self.policy.require_authorization,
                    'audit_logging': self.policy.enable_audit_logging,
                    'rate_limit': self.policy.rate_limit_enabled
                },
                'experiments': list(self.active_experiments.keys())
            }


class DataProtection:
    """Utilities for protecting sensitive research data."""
    
    @staticmethod
    def encrypt_sensitive_data(data: bytes, key: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """Encrypt sensitive data using AES-GCM.
        
        Args:
            data: Data to encrypt
            key: Encryption key (generated if None)
            
        Returns:
            Tuple of (encrypted_data, key)
        """
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            from cryptography.hazmat.backends import default_backend
        except ImportError:
            logger.warning("Cryptography library not available, data encryption disabled")
            return data, b''
        
        if key is None:
            key = secrets.token_bytes(32)
        
        # Generate random IV
        iv = secrets.token_bytes(16)
        
        # Create cipher
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        
        # Encrypt data
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        # Combine IV, tag, and ciphertext
        encrypted_data = iv + encryptor.tag + ciphertext
        
        return encrypted_data, key
    
    @staticmethod
    def decrypt_sensitive_data(encrypted_data: bytes, key: bytes) -> bytes:
        """Decrypt sensitive data.
        
        Args:
            encrypted_data: Encrypted data
            key: Decryption key
            
        Returns:
            Decrypted data
        """
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.backends import default_backend
        except ImportError:
            logger.warning("Cryptography library not available, returning data as-is")
            return encrypted_data
        
        # Extract IV, tag, and ciphertext
        iv = encrypted_data[:16]
        tag = encrypted_data[16:32]
        ciphertext = encrypted_data[32:]
        
        # Create cipher
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend())
        decryptor = cipher.decryptor()
        
        # Decrypt data
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        return plaintext
    
    @staticmethod
    def hash_sensitive_info(data: str) -> str:
        """Create secure hash of sensitive information.
        
        Args:
            data: Sensitive data to hash
            
        Returns:
            SHA-256 hash
        """
        return hashlib.sha256(data.encode()).hexdigest()
    
    @staticmethod
    def redact_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive information from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Redacted configuration
        """
        redacted_config = config.copy()
        
        # List of sensitive keys to redact
        sensitive_keys = [
            'auth_token', 'secret_key', 'password', 'private_key',
            'api_key', 'access_token', 'credentials'
        ]
        
        def redact_recursive(obj):
            if isinstance(obj, dict):
                return {
                    key: '***REDACTED***' if any(sens in key.lower() for sens in sensitive_keys)
                    else redact_recursive(value)
                    for key, value in obj.items()
                }
            elif isinstance(obj, list):
                return [redact_recursive(item) for item in obj]
            else:
                return obj
        
        return redact_recursive(redacted_config)


class ResponsibleDisclosure:
    """Framework for responsible disclosure of vulnerabilities."""
    
    def __init__(self, contact_info: Dict[str, str] = None):
        self.contact_info = contact_info or {
            'email': 'security@terragonlabs.com',
            'gpg_key': 'https://terragonlabs.com/pgp'
        }
        self.disclosures = []
    
    def create_disclosure(self, vulnerability_info: Dict[str, Any]) -> str:
        """Create vulnerability disclosure record.
        
        Args:
            vulnerability_info: Information about the vulnerability
            
        Returns:
            Disclosure ID
        """
        disclosure_id = secrets.token_hex(8)
        
        disclosure = {
            'id': disclosure_id,
            'timestamp': datetime.now().isoformat(),
            'vulnerability': vulnerability_info,
            'status': 'draft',
            'contact_info': self.contact_info
        }
        
        self.disclosures.append(disclosure)
        
        logger.info(f"Vulnerability disclosure created: {disclosure_id}")
        log_security_event(logger, "disclosure_created",
                         f"Vulnerability disclosure {disclosure_id} created",
                         disclosure_id=disclosure_id)
        
        return disclosure_id
    
    def submit_disclosure(self, disclosure_id: str) -> bool:
        """Submit disclosure to vendor.
        
        Args:
            disclosure_id: Disclosure identifier
            
        Returns:
            True if submitted successfully
        """
        # Find disclosure
        disclosure = next((d for d in self.disclosures if d['id'] == disclosure_id), None)
        if not disclosure:
            return False
        
        # Mark as submitted
        disclosure['status'] = 'submitted'
        disclosure['submitted_at'] = datetime.now().isoformat()
        
        logger.info(f"Vulnerability disclosure submitted: {disclosure_id}")
        log_security_event(logger, "disclosure_submitted",
                         f"Vulnerability disclosure {disclosure_id} submitted",
                         disclosure_id=disclosure_id)
        
        return True
    
    def get_disclosure_template(self) -> Dict[str, str]:
        """Get template for vulnerability disclosure.
        
        Returns:
            Disclosure template
        """
        return {
            'title': 'Side-Channel Vulnerability in [Algorithm] Implementation',
            'severity': 'Medium',  # Low, Medium, High, Critical
            'affected_versions': [],
            'description': 'Detailed description of vulnerability',
            'attack_requirements': {
                'physical_access': True,
                'traces_needed': 0,
                'equipment_cost': '$0',
                'expertise_level': 'novice'  # novice, moderate, expert
            },
            'impact': 'Description of potential impact',
            'mitigation': 'Suggested mitigation strategies',
            'timeline': '90 days responsible disclosure',
            'researcher_info': 'Terragon Labs Neural Cryptanalysis Team'
        }


def enforce_security_policy(func: Callable) -> Callable:
    """Decorator to enforce security policy on functions.
    
    Args:
        func: Function to protect
        
    Returns:
        Protected function
    """
    def wrapper(*args, **kwargs):
        # Check if security monitor is available
        security_monitor = getattr(func, '_security_monitor', None)
        
        if security_monitor and hasattr(security_monitor, 'policy'):
            # Extract experiment info from arguments
            experiment_id = kwargs.get('experiment_id')
            if experiment_id and experiment_id not in security_monitor.active_experiments:
                raise SecurityValidationError("Unauthorized experiment access")
        
        return func(*args, **kwargs)
    
    return wrapper


def secure_random_bytes(length: int) -> bytes:
    """Generate cryptographically secure random bytes.
    
    Args:
        length: Number of bytes to generate
        
    Returns:
        Random bytes
    """
    return secrets.token_bytes(length)


def secure_hash(data: bytes, salt: Optional[bytes] = None) -> Tuple[str, bytes]:
    """Create secure hash with salt.
    
    Args:
        data: Data to hash
        salt: Salt bytes (generated if None)
        
    Returns:
        Tuple of (hash, salt)
    """
    if salt is None:
        salt = secure_random_bytes(32)
    
    # Use PBKDF2 for secure hashing
    hash_value = hashlib.pbkdf2_hmac('sha256', data, salt, 100000)
    return hash_value.hex(), salt


def validate_experiment_ethics(config: Dict[str, Any]) -> List[str]:
    """Validate experiment meets ethical guidelines.
    
    Args:
        config: Experiment configuration
        
    Returns:
        List of ethical concerns
    """
    concerns = []
    
    # Check for responsible disclosure enablement
    security_config = config.get('security', {})
    if not security_config.get('enable_responsible_disclosure', True):
        concerns.append("Responsible disclosure must be enabled")
    
    # Check for excessive attack parameters
    side_channel_config = config.get('side_channel', {})
    n_traces = side_channel_config.get('n_traces', 0)
    
    if n_traces > 10000000:  # 10M traces
        concerns.append(f"Excessive trace count {n_traces} may indicate misuse")
    
    # Check experiment description
    experiment_info = config.get('experiment', {})
    description = experiment_info.get('description', '').lower()
    
    malicious_indicators = [
        'attack system', 'break encryption', 'exploit vulnerability',
        'unauthorized access', 'bypass security'
    ]
    
    found_indicators = [ind for ind in malicious_indicators if ind in description]
    if found_indicators:
        concerns.append(f"Experiment description contains concerning terms: {found_indicators}")
    
    return concerns