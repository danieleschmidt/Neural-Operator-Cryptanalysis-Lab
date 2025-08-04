"""Security and validation utilities for responsible cryptanalysis research."""

import hashlib
import hmac
import time
import secrets
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .utils.logging_utils import get_logger, log_security_event


class AuthorizationLevel(Enum):
    """Authorization levels for cryptanalysis operations."""
    RESEARCH = "research"
    EDUCATIONAL = "educational"
    AUDIT = "audit"
    DEVELOPMENT = "development"


class OperationType(Enum):
    """Types of cryptanalysis operations."""
    TRACE_COLLECTION = "trace_collection"
    KEY_RECOVERY = "key_recovery"
    COUNTERMEASURE_EVALUATION = "countermeasure_evaluation"
    VULNERABILITY_ASSESSMENT = "vulnerability_assessment"


@dataclass
class SecurityPolicy:
    """Security policy for cryptanalysis operations."""
    max_traces_per_experiment: int = 1000000
    max_attack_iterations: int = 1000000
    require_written_authorization: bool = True
    audit_all_operations: bool = True
    rate_limit_attacks: bool = True
    embargo_period_days: int = 90
    allowed_targets: List[str] = None
    
    def __post_init__(self):
        if self.allowed_targets is None:
            self.allowed_targets = [
                'test_implementations',
                'research_platforms',
                'authorized_targets'
            ]


class AuthorizationManager:
    """Manages authorization for cryptanalysis operations."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.logger = get_logger(__name__)
        self.active_sessions = {}
        self.operation_history = []
        
    def request_authorization(self,
                            operation: OperationType,
                            target_description: str,
                            justification: str,
                            authorization_level: AuthorizationLevel) -> str:
        """Request authorization for cryptanalysis operation.
        
        Args:
            operation: Type of operation
            target_description: Description of target system
            justification: Research justification
            authorization_level: Required authorization level
            
        Returns:
            Authorization token if approved
            
        Raises:
            SecurityError: If authorization is denied
        """
        # Generate authorization request
        request_id = self._generate_request_id()
        
        # Log authorization request
        log_security_event(
            self.logger,
            'authorization_request',
            f"Authorization requested for {operation.value}",
            operation=operation.value,
            target=target_description,
            justification=justification,
            level=authorization_level.value,
            request_id=request_id
        )
        
        # Check policy compliance
        if not self._check_policy_compliance(operation, target_description):
            raise SecurityError(f"Operation not permitted by security policy")
        
        # For demonstration, auto-approve research operations
        # In practice, this would involve human review
        if authorization_level in [AuthorizationLevel.RESEARCH, AuthorizationLevel.EDUCATIONAL]:
            token = self._generate_authorization_token(request_id, operation)
            
            self.active_sessions[token] = {
                'request_id': request_id,
                'operation': operation,
                'target': target_description,
                'authorization_level': authorization_level,
                'timestamp': time.time(),
                'usage_count': 0
            }
            
            log_security_event(
                self.logger,
                'authorization_granted',
                f"Authorization granted for {operation.value}",
                request_id=request_id,
                token=token[:8] + "..."  # Log partial token only
            )
            
            return token
        else:
            raise SecurityError(f"Authorization level {authorization_level.value} requires manual approval")
    
    def validate_authorization(self, token: str, operation: OperationType) -> bool:
        """Validate authorization token for operation.
        
        Args:
            token: Authorization token
            operation: Operation being attempted
            
        Returns:
            True if authorized, False otherwise
        """
        if token not in self.active_sessions:
            log_security_event(
                self.logger,
                'authorization_validation_failed',
                "Invalid authorization token used",
                token=token[:8] + "..." if len(token) > 8 else token
            )
            return False
        
        session = self.active_sessions[token]
        
        # Check if token matches operation
        if session['operation'] != operation:
            log_security_event(
                self.logger,
                'authorization_mismatch',
                f"Token authorized for {session['operation'].value} but used for {operation.value}",
                token=token[:8] + "..."
            )
            return False
        
        # Check if token has expired (24 hour limit)
        if time.time() - session['timestamp'] > 86400:
            log_security_event(
                self.logger,
                'authorization_expired',
                "Expired authorization token used",
                token=token[:8] + "..."
            )
            del self.active_sessions[token]
            return False
        
        # Update usage counter
        session['usage_count'] += 1
        
        return True
    
    def revoke_authorization(self, token: str):
        """Revoke authorization token."""
        if token in self.active_sessions:
            log_security_event(
                self.logger,
                'authorization_revoked',
                "Authorization token revoked",
                token=token[:8] + "..."
            )
            del self.active_sessions[token]
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        return secrets.token_hex(16)
    
    def _generate_authorization_token(self, request_id: str, operation: OperationType) -> str:
        """Generate authorization token."""
        data = f"{request_id}:{operation.value}:{time.time()}".encode()
        return hashlib.sha256(data).hexdigest()
    
    def _check_policy_compliance(self, operation: OperationType, target: str) -> bool:
        """Check if operation complies with security policy."""
        # Check if target is in allowed list
        allowed = any(allowed_target in target.lower() 
                     for allowed_target in self.policy.allowed_targets)
        
        if not allowed:
            return False
        
        # Additional policy checks can be added here
        return True


class OperationLimiter:
    """Rate limiting and resource controls for cryptanalysis operations."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.logger = get_logger(__name__)
        self.operation_counts = {}
        self.rate_limits = {}
        
    def check_limits(self, operation: OperationType, **kwargs) -> bool:
        """Check if operation is within limits.
        
        Args:
            operation: Type of operation
            **kwargs: Operation parameters
            
        Returns:
            True if within limits, False otherwise
        """
        current_time = time.time()
        
        # Check trace collection limits
        if operation == OperationType.TRACE_COLLECTION:
            n_traces = kwargs.get('n_traces', 0)
            if n_traces > self.policy.max_traces_per_experiment:
                self.logger.warning(
                    f"Trace collection request ({n_traces}) exceeds policy limit "
                    f"({self.policy.max_traces_per_experiment})"
                )
                return False
        
        # Check attack iteration limits
        elif operation == OperationType.KEY_RECOVERY:
            max_iterations = kwargs.get('max_iterations', 0)
            if max_iterations > self.policy.max_attack_iterations:
                self.logger.warning(
                    f"Attack iteration request ({max_iterations}) exceeds policy limit "
                    f"({self.policy.max_attack_iterations})"
                )
                return False
        
        # Rate limiting
        if self.policy.rate_limit_attacks and operation == OperationType.KEY_RECOVERY:
            if not self._check_rate_limit(operation, current_time):
                return False
        
        return True
    
    def _check_rate_limit(self, operation: OperationType, current_time: float) -> bool:
        """Check rate limits for operation."""
        if operation not in self.rate_limits:
            self.rate_limits[operation] = []
        
        # Remove old entries (1 hour window)
        window = 3600  # 1 hour
        self.rate_limits[operation] = [
            t for t in self.rate_limits[operation] 
            if current_time - t < window
        ]
        
        # Check if within rate limit (max 10 attacks per hour)
        max_per_hour = 10
        if len(self.rate_limits[operation]) >= max_per_hour:
            self.logger.warning(
                f"Rate limit exceeded for {operation.value}: "
                f"{len(self.rate_limits[operation])} operations in last hour"
            )
            return False
        
        # Record this operation
        self.rate_limits[operation].append(current_time)
        return True


class VulnerabilityReporter:
    """Handles responsible disclosure of discovered vulnerabilities."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.logger = get_logger(__name__)
        self.reported_vulnerabilities = {}
        
    def create_vulnerability_report(self,
                                  title: str,
                                  description: str,
                                  severity: str,
                                  affected_systems: List[str],
                                  attack_requirements: Dict[str, Any],
                                  mitigation_suggestions: str) -> str:
        """Create vulnerability report.
        
        Args:
            title: Vulnerability title
            description: Detailed description
            severity: Severity level (low, medium, high, critical)
            affected_systems: List of affected systems
            attack_requirements: Requirements for successful attack
            mitigation_suggestions: Suggested mitigations
            
        Returns:
            Vulnerability report ID
        """
        report_id = secrets.token_hex(8)
        
        report = {
            'id': report_id,
            'title': title,
            'description': description,
            'severity': severity,
            'affected_systems': affected_systems,
            'attack_requirements': attack_requirements,
            'mitigation_suggestions': mitigation_suggestions,
            'discovery_date': time.time(),
            'status': 'discovered',
            'embargo_end': time.time() + (self.policy.embargo_period_days * 86400),
            'disclosure_timeline': []
        }
        
        self.reported_vulnerabilities[report_id] = report
        
        log_security_event(
            self.logger,
            'vulnerability_reported',
            f"New vulnerability reported: {title}",
            report_id=report_id,
            severity=severity,
            affected_systems=len(affected_systems)
        )
        
        # Automatically notify if critical
        if severity == 'critical':
            self._initiate_emergency_disclosure(report_id)
        
        return report_id
    
    def _initiate_emergency_disclosure(self, report_id: str):
        """Initiate emergency disclosure for critical vulnerabilities."""
        self.logger.critical(
            f"CRITICAL VULNERABILITY DISCOVERED: {report_id} - "
            "Initiating emergency disclosure procedures"
        )
        
        # In practice, this would contact relevant parties immediately
        # For now, just log the action
        log_security_event(
            self.logger,
            'emergency_disclosure',
            f"Emergency disclosure initiated for {report_id}",
            report_id=report_id
        )


class DataProtection:
    """Protects sensitive data in cryptanalysis experiments."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._encryption_key = None
        
    def sanitize_traces(self, traces: List[Any]) -> List[Any]:
        """Remove sensitive information from traces.
        
        Args:
            traces: Raw trace data
            
        Returns:
            Sanitized traces
        """
        # Remove absolute values that could reveal secrets
        sanitized = []
        
        for trace in traces:
            if hasattr(trace, 'copy'):
                sanitized_trace = trace.copy()
                
                # Add noise to prevent exact recovery
                if hasattr(sanitized_trace, 'shape'):
                    import numpy as np
                    noise = np.random.normal(0, 0.001, sanitized_trace.shape)
                    sanitized_trace += noise
                
                sanitized.append(sanitized_trace)
            else:
                sanitized.append(trace)
        
        self.logger.info(f"Sanitized {len(traces)} traces for sharing")
        return sanitized
    
    def anonymize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize attack results for publication.
        
        Args:
            results: Raw attack results
            
        Returns:
            Anonymized results
        """
        anonymized = results.copy()
        
        # Remove specific key values
        sensitive_keys = ['recovered_key', 'secret_key', 'private_key']
        for key in sensitive_keys:
            if key in anonymized:
                # Replace with success indicator only
                anonymized[key] = 'SUCCESS' if anonymized[key] is not None else 'FAILED'
        
        # Generalize specific implementation details
        if 'target_implementation' in anonymized:
            anonymized['target_implementation'] = 'ANONYMIZED'
        
        self.logger.info("Anonymized attack results for publication")
        return anonymized


class SecurityError(Exception):
    """Exception raised for security policy violations."""
    pass


class ResponsibleDisclosure:
    """Manages the responsible disclosure process."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.authorization_manager = AuthorizationManager(policy)
        self.operation_limiter = OperationLimiter(policy)
        self.vulnerability_reporter = VulnerabilityReporter(policy)
        self.data_protection = DataProtection()
        self.logger = get_logger(__name__)
        
    def ensure_authorized(self, operation: OperationType, **kwargs) -> str:
        """Ensure operation is authorized.
        
        Args:
            operation: Type of operation
            **kwargs: Operation parameters
            
        Returns:
            Authorization token
        """
        # Check if authorization is required
        if not self.policy.require_written_authorization:
            # Generate a basic token for logging purposes
            return f"auto_auth_{secrets.token_hex(8)}"
        
        # For automated systems, this would integrate with an approval system
        # For now, auto-approve research operations with proper logging
        token = self.authorization_manager.request_authorization(
            operation=operation,
            target_description=kwargs.get('target', 'research_target'),
            justification=kwargs.get('justification', 'security_research'),
            authorization_level=AuthorizationLevel.RESEARCH
        )
        
        return token
    
    def validate_operation(self, token: str, operation: OperationType, **kwargs) -> bool:
        """Validate that operation is authorized and within limits.
        
        Args:
            token: Authorization token
            operation: Type of operation
            **kwargs: Operation parameters
            
        Returns:
            True if valid, False otherwise
        """
        # Validate authorization
        if not self.authorization_manager.validate_authorization(token, operation):
            return False
        
        # Check limits
        if not self.operation_limiter.check_limits(operation, **kwargs):
            return False
        
        return True
    
    def report_findings(self,
                       findings: Dict[str, Any],
                       target_system: str) -> str:
        """Report security findings through responsible disclosure.
        
        Args:
            findings: Security findings
            target_system: Affected system
            
        Returns:
            Report ID
        """
        # Determine severity
        severity = self._assess_severity(findings)
        
        # Create vulnerability report
        report_id = self.vulnerability_reporter.create_vulnerability_report(
            title=f"Side-channel vulnerability in {target_system}",
            description=findings.get('description', 'Neural operator-based attack successful'),
            severity=severity,
            affected_systems=[target_system],
            attack_requirements=findings.get('requirements', {}),
            mitigation_suggestions=findings.get('mitigations', 'Apply side-channel countermeasures')
        )
        
        self.logger.info(f"Security findings reported: {report_id}")
        return report_id
    
    def _assess_severity(self, findings: Dict[str, Any]) -> str:
        """Assess severity of security findings."""
        success_rate = findings.get('success_rate', 0)
        traces_needed = findings.get('traces_needed', float('inf'))
        
        if success_rate > 0.9 and traces_needed < 10000:
            return 'critical'
        elif success_rate > 0.7 and traces_needed < 100000:
            return 'high'
        elif success_rate > 0.5 and traces_needed < 1000000:
            return 'medium'
        else:
            return 'low'