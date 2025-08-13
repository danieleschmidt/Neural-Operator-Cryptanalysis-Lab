"""Security penetration testing and vulnerability scanning framework."""

import pytest
import numpy as np
import torch
import time
import hashlib
import hmac
import os
import tempfile
import subprocess
import json
from pathlib import Path
import sys
import asyncio
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Import test fixtures
from conftest import security_config

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import components for security testing
from neural_cryptanalysis.core import NeuralSCA, TraceData
from neural_cryptanalysis.utils.security import SecurityPolicy, SecurityValidator
from neural_cryptanalysis.utils.errors import SecurityError, ValidationError
from neural_cryptanalysis.datasets.synthetic import SyntheticDatasetGenerator
from neural_cryptanalysis.adaptive_rl import AdaptiveAttackEngine
from neural_cryptanalysis.hardware_integration import HardwareInTheLoopSystem


@dataclass
class SecurityFinding:
    """Security vulnerability finding."""
    severity: str  # critical, high, medium, low
    category: str  # injection, exposure, validation, etc.
    description: str
    location: str
    recommendation: str
    cve_reference: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'severity': self.severity,
            'category': self.category,
            'description': self.description,
            'location': self.location,
            'recommendation': self.recommendation,
            'cve_reference': self.cve_reference
        }


class SecurityScanner:
    """Comprehensive security vulnerability scanner."""
    
    def __init__(self):
        self.findings: List[SecurityFinding] = []
        self.scan_results = {}
    
    def add_finding(self, finding: SecurityFinding):
        """Add a security finding."""
        self.findings.append(finding)
    
    def scan_input_validation(self, component) -> List[SecurityFinding]:
        """Scan for input validation vulnerabilities."""
        findings = []
        
        try:
            # Test with various malicious inputs
            malicious_inputs = [
                None,  # Null injection
                "",    # Empty string
                "' OR '1'='1",  # SQL injection pattern
                "<script>alert('xss')</script>",  # XSS pattern
                "../../../etc/passwd",  # Path traversal
                "A" * 10000,  # Buffer overflow attempt
                float('inf'),  # Infinity
                float('nan'),  # NaN
                -1,  # Negative values
                sys.maxsize + 1,  # Integer overflow
            ]
            
            for malicious_input in malicious_inputs:
                try:
                    if hasattr(component, '__call__'):
                        component(malicious_input)
                    elif hasattr(component, 'process'):
                        component.process(malicious_input)
                    
                    # If no exception, might be vulnerable
                    findings.append(SecurityFinding(
                        severity='medium',
                        category='input_validation',
                        description=f'Component accepts potentially malicious input: {repr(malicious_input)}',
                        location=str(type(component)),
                        recommendation='Implement strict input validation and sanitization'
                    ))
                    
                except (ValidationError, ValueError, TypeError):
                    # Good - input was rejected
                    pass
                except Exception as e:
                    # Unexpected error might indicate vulnerability
                    findings.append(SecurityFinding(
                        severity='high',
                        category='input_validation',
                        description=f'Unexpected error with input {repr(malicious_input)}: {e}',
                        location=str(type(component)),
                        recommendation='Review error handling and input validation'
                    ))
        
        except Exception as e:
            findings.append(SecurityFinding(
                severity='high',
                category='scan_error',
                description=f'Error during input validation scan: {e}',
                location='SecurityScanner.scan_input_validation',
                recommendation='Review scanner implementation'
            ))
        
        return findings
    
    def scan_memory_safety(self, component) -> List[SecurityFinding]:
        """Scan for memory safety issues."""
        findings = []
        
        try:
            import psutil
            process = psutil.Process()
            
            # Baseline memory
            baseline_memory = process.memory_info().rss
            
            # Test with large inputs to detect memory leaks
            large_inputs = [
                np.random.randn(10000, 5000),  # Large array
                torch.randn(1000, 1000, 100),  # Large tensor
                ["large_string" * 1000] * 1000,  # Large list
            ]
            
            for large_input in large_inputs:
                try:
                    if hasattr(component, '__call__'):
                        result = component(large_input)
                        del result
                    
                    current_memory = process.memory_info().rss
                    memory_increase = current_memory - baseline_memory
                    
                    # Check for excessive memory usage
                    if memory_increase > 1024 * 1024 * 1024:  # 1GB
                        findings.append(SecurityFinding(
                            severity='high',
                            category='memory_safety',
                            description='Excessive memory allocation detected',
                            location=str(type(component)),
                            recommendation='Implement memory usage limits and proper cleanup'
                        ))
                    
                    del large_input
                    
                except Exception as e:
                    if 'memory' in str(e).lower() or 'out of' in str(e).lower():
                        findings.append(SecurityFinding(
                            severity='medium',
                            category='memory_safety',
                            description=f'Memory-related error: {e}',
                            location=str(type(component)),
                            recommendation='Implement proper memory management and limits'
                        ))
        
        except Exception as e:
            findings.append(SecurityFinding(
                severity='medium',
                category='scan_error',
                description=f'Error during memory safety scan: {e}',
                location='SecurityScanner.scan_memory_safety',
                recommendation='Review memory scanning implementation'
            ))
        
        return findings
    
    def scan_timing_attacks(self, component) -> List[SecurityFinding]:
        """Scan for timing attack vulnerabilities."""
        findings = []
        
        try:
            # Test with different inputs to detect timing differences
            test_inputs = [
                torch.randn(100, 100),
                torch.randn(100, 100) + 1000,  # Different magnitude
                torch.zeros(100, 100),
                torch.ones(100, 100),
            ]
            
            timings = []
            
            for test_input in test_inputs:
                start_time = time.perf_counter()
                
                try:
                    if hasattr(component, '__call__'):
                        result = component(test_input)
                    elif hasattr(component, 'process'):
                        result = component.process(test_input)
                    
                    end_time = time.perf_counter()
                    timings.append(end_time - start_time)
                    
                except Exception:
                    timings.append(float('inf'))  # Error case
            
            # Check for significant timing differences
            valid_timings = [t for t in timings if t != float('inf')]
            if len(valid_timings) >= 2:
                timing_variance = np.var(valid_timings)
                timing_mean = np.mean(valid_timings)
                
                if timing_variance > (timing_mean * 0.5)**2:  # High variance
                    findings.append(SecurityFinding(
                        severity='medium',
                        category='timing_attack',
                        description='Significant timing variations detected',
                        location=str(type(component)),
                        recommendation='Implement constant-time operations where appropriate'
                    ))
        
        except Exception as e:
            findings.append(SecurityFinding(
                severity='low',
                category='scan_error',
                description=f'Error during timing attack scan: {e}',
                location='SecurityScanner.scan_timing_attacks',
                recommendation='Review timing analysis implementation'
            ))
        
        return findings
    
    def scan_dependency_vulnerabilities(self) -> List[SecurityFinding]:
        """Scan for known dependency vulnerabilities."""
        findings = []
        
        try:
            # Check for known vulnerable packages
            vulnerable_patterns = {
                'pillow': '< 8.3.2',  # CVE-2021-34552
                'numpy': '< 1.21.0',  # CVE-2021-33430
                'torch': '< 1.13.0',  # Various security fixes
            }
            
            import pkg_resources
            
            for package_name, vulnerable_version in vulnerable_patterns.items():
                try:
                    package = pkg_resources.get_distribution(package_name)
                    # Simplified version check (real implementation would use proper version parsing)
                    findings.append(SecurityFinding(
                        severity='medium',
                        category='dependency_vulnerability',
                        description=f'Dependency {package_name} version {package.version} should be checked for vulnerabilities',
                        location='dependencies',
                        recommendation=f'Update {package_name} to latest secure version'
                    ))
                except pkg_resources.DistributionNotFound:
                    pass  # Package not installed
        
        except Exception as e:
            findings.append(SecurityFinding(
                severity='low',
                category='scan_error',
                description=f'Error during dependency scan: {e}',
                location='SecurityScanner.scan_dependency_vulnerabilities',
                recommendation='Review dependency scanning implementation'
            ))
        
        return findings
    
    def scan_authentication_bypass(self, component) -> List[SecurityFinding]:
        """Scan for authentication bypass vulnerabilities."""
        findings = []
        
        try:
            # Test authentication bypass patterns
            bypass_attempts = [
                {'user': 'admin', 'password': ''},
                {'user': 'admin', 'password': 'admin'},
                {'user': '', 'password': ''},
                {'user': None, 'password': None},
                {'token': 'invalid_token'},
                {'auth': False},
            ]
            
            if hasattr(component, 'authenticate') or hasattr(component, 'check_authorization'):
                for attempt in bypass_attempts:
                    try:
                        if hasattr(component, 'authenticate'):
                            result = component.authenticate(**attempt)
                        else:
                            result = component.check_authorization(**attempt)
                        
                        if result:  # Authentication succeeded with invalid creds
                            findings.append(SecurityFinding(
                                severity='critical',
                                category='authentication_bypass',
                                description=f'Authentication bypass possible with: {attempt}',
                                location=str(type(component)),
                                recommendation='Implement proper authentication validation'
                            ))
                    
                    except (ValidationError, SecurityError):
                        pass  # Good - authentication failed as expected
                    except Exception as e:
                        findings.append(SecurityFinding(
                            severity='high',
                            category='authentication_bypass',
                            description=f'Unexpected auth error with {attempt}: {e}',
                            location=str(type(component)),
                            recommendation='Review authentication error handling'
                        ))
        
        except Exception as e:
            findings.append(SecurityFinding(
                severity='medium',
                category='scan_error',
                description=f'Error during authentication scan: {e}',
                location='SecurityScanner.scan_authentication_bypass',
                recommendation='Review authentication scanning implementation'
            ))
        
        return findings
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        severity_counts = {}
        category_counts = {}
        
        for finding in self.findings:
            severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1
            category_counts[finding.category] = category_counts.get(finding.category, 0) + 1
        
        critical_findings = [f for f in self.findings if f.severity == 'critical']
        high_findings = [f for f in self.findings if f.severity == 'high']
        
        return {
            'total_findings': len(self.findings),
            'severity_breakdown': severity_counts,
            'category_breakdown': category_counts,
            'critical_findings': [f.to_dict() for f in critical_findings],
            'high_findings': [f.to_dict() for f in high_findings],
            'all_findings': [f.to_dict() for f in self.findings],
            'risk_score': self._calculate_risk_score(),
            'recommendations': self._generate_recommendations()
        }
    
    def _calculate_risk_score(self) -> float:
        """Calculate overall risk score (0-100)."""
        if not self.findings:
            return 0.0
        
        severity_weights = {
            'critical': 10,
            'high': 7,
            'medium': 4,
            'low': 1
        }
        
        total_score = sum(severity_weights.get(f.severity, 1) for f in self.findings)
        max_possible = len(self.findings) * 10  # All critical
        
        return (total_score / max_possible) * 100 if max_possible > 0 else 0.0
    
    def _generate_recommendations(self) -> List[str]:
        """Generate prioritized security recommendations."""
        recommendations = set()
        
        # Add recommendations from critical and high severity findings
        for finding in self.findings:
            if finding.severity in ['critical', 'high']:
                recommendations.add(finding.recommendation)
        
        return list(recommendations)


@pytest.mark.security
class TestSecurityValidation:
    """Security validation and penetration testing."""
    
    def test_input_validation_security(self):
        """Test input validation security across components."""
        print("\n=== Testing Input Validation Security ===")
        
        scanner = SecurityScanner()
        
        # Test NeuralSCA input validation
        neural_sca = NeuralSCA()
        findings = scanner.scan_input_validation(neural_sca)
        scanner.findings.extend(findings)
        
        # Test with synthetic dataset generator
        generator = SyntheticDatasetGenerator()
        generator_findings = scanner.scan_input_validation(generator)
        scanner.findings.extend(generator_findings)
        
        # Test with various malicious inputs
        malicious_traces = [
            np.array([float('inf')] * 1000),  # Infinity values
            np.array([float('nan')] * 1000),  # NaN values
            np.random.randn(0, 0),  # Empty array
            np.random.randn(10**8),  # Extremely large array
        ]
        
        for traces in malicious_traces:
            try:
                # Should either handle gracefully or raise appropriate validation error
                if traces.size > 0:
                    trace_data = TraceData(traces=traces, labels=np.zeros(len(traces)))
                    # If this succeeds without validation, it's a security issue
                    if np.any(~np.isfinite(traces)):
                        scanner.add_finding(SecurityFinding(
                            severity='medium',
                            category='input_validation',
                            description='Non-finite values accepted in trace data',
                            location='TraceData',
                            recommendation='Validate input data for finite values'
                        ))
            except (ValidationError, ValueError):
                pass  # Good - validation working
            except Exception as e:
                scanner.add_finding(SecurityFinding(
                    severity='high',
                    category='input_validation',
                    description=f'Unexpected error with malicious input: {e}',
                    location='TraceData',
                    recommendation='Improve input validation and error handling'
                ))
        
        print(f"✓ Input validation scan completed: {len(scanner.findings)} findings")
        
        # Should have minimal critical findings
        critical_findings = [f for f in scanner.findings if f.severity == 'critical']
        assert len(critical_findings) == 0, f"Critical input validation vulnerabilities found: {critical_findings}"
        
        return scanner.findings
    
    def test_authentication_security(self, security_config):
        """Test authentication and authorization security."""
        print("\n=== Testing Authentication Security ===")
        
        scanner = SecurityScanner()
        
        # Test security policy
        security_policy = SecurityPolicy(config=security_config)
        auth_findings = scanner.scan_authentication_bypass(security_policy)
        scanner.findings.extend(auth_findings)
        
        # Test rate limiting
        try:
            for i in range(security_config['max_traces_per_hour'] + 10):
                try:
                    security_policy.check_rate_limit('test_user')
                except SecurityError:
                    # Good - rate limiting is working
                    break
            else:
                scanner.add_finding(SecurityFinding(
                    severity='high',
                    category='rate_limiting',
                    description='Rate limiting not properly enforced',
                    location='SecurityPolicy.check_rate_limit',
                    recommendation='Implement proper rate limiting controls'
                ))
        except Exception as e:
            scanner.add_finding(SecurityFinding(
                severity='medium',
                category='rate_limiting',
                description=f'Rate limiting test error: {e}',
                location='SecurityPolicy.check_rate_limit',
                recommendation='Review rate limiting implementation'
            ))
        
        # Test authorization bypass attempts
        bypass_tests = [
            {'operation': 'admin_operation', 'user': 'regular_user'},
            {'operation': 'train', 'user': None},
            {'operation': 'attack', 'user': ''},
        ]
        
        for test in bypass_tests:
            try:
                authorized = security_policy.check_authorization(
                    test['user'], test['operation']
                )
                if authorized and test['user'] in [None, '']:
                    scanner.add_finding(SecurityFinding(
                        severity='critical',
                        category='authorization_bypass',
                        description=f'Authorization bypass possible: {test}',
                        location='SecurityPolicy.check_authorization',
                        recommendation='Implement strict authorization validation'
                    ))
            except (ValidationError, SecurityError):
                pass  # Good - authorization properly denied
            except Exception as e:
                scanner.add_finding(SecurityFinding(
                    severity='high',
                    category='authorization_bypass',
                    description=f'Authorization error: {e}',
                    location='SecurityPolicy.check_authorization',
                    recommendation='Review authorization error handling'
                ))
        
        print(f"✓ Authentication security scan completed")
        
        # No critical auth vulnerabilities should exist
        critical_auth = [f for f in scanner.findings if f.severity == 'critical' and 'auth' in f.category]
        assert len(critical_auth) == 0, f"Critical authentication vulnerabilities: {critical_auth}"
        
        return scanner.findings
    
    def test_memory_safety_security(self):
        """Test memory safety and resource exhaustion protection."""
        print("\n=== Testing Memory Safety Security ===")
        
        scanner = SecurityScanner()
        
        # Test neural SCA memory safety
        neural_sca = NeuralSCA(config={
            'training': {'batch_size': 8, 'epochs': 1}
        })
        
        memory_findings = scanner.scan_memory_safety(neural_sca)
        scanner.findings.extend(memory_findings)
        
        # Test with extremely large inputs (DoS attempt)
        try:
            # This should either be handled gracefully or fail safely
            huge_traces = torch.randn(10000, 10000)  # ~400MB tensor
            huge_labels = torch.randint(0, 256, (10000,))
            
            import psutil
            process = psutil.Process()
            baseline_memory = process.memory_info().rss
            
            try:
                model = neural_sca.train(huge_traces[:100], huge_labels[:100], validation_split=0.2)
                
                current_memory = process.memory_info().rss
                memory_increase = current_memory - baseline_memory
                
                # Check for reasonable memory usage
                if memory_increase > 2 * 1024**3:  # 2GB increase
                    scanner.add_finding(SecurityFinding(
                        severity='high',
                        category='memory_safety',
                        description='Excessive memory consumption with large inputs',
                        location='NeuralSCA.train',
                        recommendation='Implement memory usage limits and input size validation'
                    ))
                
            except (MemoryError, RuntimeError) as e:
                if 'memory' in str(e).lower():
                    # This is actually good - system protected itself
                    pass
                else:
                    raise
            
        except Exception as e:
            if 'memory' not in str(e).lower():
                scanner.add_finding(SecurityFinding(
                    severity='medium',
                    category='memory_safety',
                    description=f'Unexpected error with large input: {e}',
                    location='NeuralSCA',
                    recommendation='Implement proper input size validation'
                ))
        
        print(f"✓ Memory safety scan completed")
        
        # High memory safety issues should be minimal
        high_memory = [f for f in scanner.findings if f.severity == 'high' and 'memory' in f.category]
        assert len(high_memory) <= 1, f"Too many high-severity memory issues: {high_memory}"
        
        return scanner.findings
    
    def test_timing_attack_resistance(self):
        """Test resistance to timing-based side-channel attacks."""
        print("\n=== Testing Timing Attack Resistance ===")
        
        scanner = SecurityScanner()
        
        # Test neural SCA timing consistency
        neural_sca = NeuralSCA(config={
            'training': {'batch_size': 16, 'epochs': 1}
        })
        
        # Quick training to get a model
        traces = torch.randn(50, 100, 1)
        labels = torch.randint(0, 256, (50,))
        model = neural_sca.train(traces, labels, validation_split=0.2)
        
        # Test inference timing with different inputs
        timing_findings = scanner.scan_timing_attacks(model)
        scanner.findings.extend(timing_findings)
        
        # Test specific timing scenarios
        test_scenarios = [
            torch.zeros(10, 100, 1),  # All zeros
            torch.ones(10, 100, 1),   # All ones
            torch.randn(10, 100, 1),  # Random
            torch.randn(10, 100, 1) * 1000,  # Large magnitude
        ]
        
        timings = []
        
        with torch.no_grad():
            for scenario in test_scenarios:
                start_time = time.perf_counter()
                predictions = model(scenario)
                end_time = time.perf_counter()
                timings.append(end_time - start_time)
        
        # Check timing consistency
        if len(timings) > 1:
            timing_std = np.std(timings)
            timing_mean = np.mean(timings)
            cv = timing_std / timing_mean if timing_mean > 0 else 0
            
            if cv > 0.3:  # Coefficient of variation > 30%
                scanner.add_finding(SecurityFinding(
                    severity='medium',
                    category='timing_attack',
                    description=f'High timing variation detected: CV={cv:.3f}',
                    location='Neural model inference',
                    recommendation='Implement constant-time operations or add timing noise'
                ))
        
        print(f"✓ Timing attack resistance scan completed")
        print(f"  Mean inference time: {np.mean(timings)*1000:.3f}ms")
        print(f"  Timing variation (CV): {cv:.3f}")
        
        return scanner.findings
    
    def test_data_leakage_protection(self):
        """Test protection against data leakage vulnerabilities."""
        print("\n=== Testing Data Leakage Protection ===")
        
        scanner = SecurityScanner()
        
        # Generate test dataset with known patterns
        generator = SyntheticDatasetGenerator(random_seed=42)
        dataset = generator.generate_aes_dataset(n_traces=100, trace_length=200)
        
        # Test that training data patterns don't leak in unexpected ways
        neural_sca = NeuralSCA(config={
            'training': {'batch_size': 16, 'epochs': 2}
        })
        
        traces = torch.tensor(dataset['power_traces'], dtype=torch.float32).unsqueeze(-1)
        labels = torch.tensor(dataset['labels'][0], dtype=torch.long)
        
        # Split data
        train_traces = traces[:70]
        train_labels = labels[:70]
        test_traces = traces[70:]
        test_labels = labels[70:]
        
        model = neural_sca.train(train_traces, train_labels, validation_split=0.2)
        
        # Test for potential overfitting/memorization
        with torch.no_grad():
            train_predictions = model(train_traces)
            test_predictions = model(test_traces)
            
            train_accuracy = (torch.argmax(train_predictions, dim=1) == train_labels).float().mean()
            test_accuracy = (torch.argmax(test_predictions, dim=1) == test_labels).float().mean()
            
            # Check for excessive overfitting (potential memorization)
            if train_accuracy > 0.95 and test_accuracy < 0.3:
                scanner.add_finding(SecurityFinding(
                    severity='medium',
                    category='data_leakage',
                    description='Potential data memorization detected (high train, low test accuracy)',
                    location='Neural model training',
                    recommendation='Implement regularization and proper validation'
                ))
        
        # Test that model doesn't leak training data through gradients
        try:
            # Enable gradients
            test_input = torch.randn(1, 200, 1, requires_grad=True)
            output = model(test_input)
            loss = output.mean()
            loss.backward()
            
            # Check gradient magnitude (shouldn't be excessive)
            grad_norm = test_input.grad.norm().item()
            if grad_norm > 1000:  # Arbitrary threshold for excessive gradients
                scanner.add_finding(SecurityFinding(
                    severity='low',
                    category='data_leakage',
                    description='Large gradient magnitudes detected',
                    location='Neural model gradients',
                    recommendation='Review gradient clipping and normalization'
                ))
        
        except Exception as e:
            scanner.add_finding(SecurityFinding(
                severity='low',
                category='data_leakage',
                description=f'Gradient computation error: {e}',
                location='Neural model gradients',
                recommendation='Review gradient computation security'
            ))
        
        print(f"✓ Data leakage protection scan completed")
        print(f"  Train accuracy: {train_accuracy:.3f}")
        print(f"  Test accuracy: {test_accuracy:.3f}")
        
        return scanner.findings
    
    def test_cryptographic_security(self):
        """Test cryptographic implementation security."""
        print("\n=== Testing Cryptographic Security ===")
        
        scanner = SecurityScanner()
        
        # Test RNG security
        try:
            # Check if secure random is used where appropriate
            generator = SyntheticDatasetGenerator(random_seed=None)  # Should use secure random
            
            # Generate multiple datasets and check for patterns
            datasets = []
            for _ in range(5):
                dataset = generator.generate_aes_dataset(n_traces=10, trace_length=50)
                datasets.append(dataset['key'])
            
            # Check for key reuse (security issue)
            unique_keys = set(tuple(key) for key in datasets)
            if len(unique_keys) < len(datasets):
                scanner.add_finding(SecurityFinding(
                    severity='critical',
                    category='cryptographic_security',
                    description='Key reuse detected in dataset generation',
                    location='SyntheticDatasetGenerator',
                    recommendation='Ensure cryptographically secure random key generation'
                ))
            
            # Check key entropy
            for i, key in enumerate(datasets):
                key_entropy = -np.sum([p * np.log2(p) for p in np.bincount(key) / len(key) if p > 0])
                if key_entropy < 7.0:  # Should be close to 8 for random bytes
                    scanner.add_finding(SecurityFinding(
                        severity='high',
                        category='cryptographic_security',
                        description=f'Low entropy key detected: {key_entropy:.2f} bits',
                        location='SyntheticDatasetGenerator',
                        recommendation='Use cryptographically secure random number generator'
                    ))
        
        except Exception as e:
            scanner.add_finding(SecurityFinding(
                severity='medium',
                category='cryptographic_security',
                description=f'Cryptographic security test error: {e}',
                location='SyntheticDatasetGenerator',
                recommendation='Review cryptographic implementation'
            ))
        
        print(f"✓ Cryptographic security scan completed")
        
        # No critical crypto issues should exist
        critical_crypto = [f for f in scanner.findings if f.severity == 'critical' and 'crypto' in f.category]
        assert len(critical_crypto) == 0, f"Critical cryptographic vulnerabilities: {critical_crypto}"
        
        return scanner.findings
    
    @pytest.mark.asyncio
    async def test_async_security_vulnerabilities(self):
        """Test async operation security vulnerabilities."""
        print("\n=== Testing Async Security Vulnerabilities ===")
        
        scanner = SecurityScanner()
        
        # Test adaptive RL for async vulnerabilities
        neural_sca = NeuralSCA()
        adaptive_engine = AdaptiveAttackEngine(neural_sca, device='cpu')
        
        # Test for race conditions in async operations
        try:
            # Simulate concurrent access
            async def concurrent_evaluation():
                trace_data = TraceData(
                    traces=np.random.randn(50, 100),
                    labels=np.random.randint(0, 256, 50)
                )
                
                # Mock the evaluation to avoid actual training
                async def mock_eval(state, traces):
                    await asyncio.sleep(0.01)  # Simulate work
                    return 0.5, 0.6, 0.4
                
                with patch.object(adaptive_engine, 'evaluate_attack_performance', side_effect=mock_eval):
                    results = adaptive_engine.autonomous_attack(
                        traces=trace_data,
                        max_episodes=2,
                        patience=1
                    )
                return results
            
            # Run multiple concurrent evaluations
            tasks = [concurrent_evaluation() for _ in range(3)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check for exceptions that might indicate race conditions
            exceptions = [r for r in results if isinstance(r, Exception)]
            if exceptions:
                scanner.add_finding(SecurityFinding(
                    severity='medium',
                    category='concurrency_security',
                    description=f'Async operation exceptions: {len(exceptions)} failures',
                    location='AdaptiveAttackEngine',
                    recommendation='Review thread safety and async operation handling'
                ))
        
        except Exception as e:
            scanner.add_finding(SecurityFinding(
                severity='medium',
                category='concurrency_security',
                description=f'Async security test error: {e}',
                location='AdaptiveAttackEngine',
                recommendation='Review async operation security'
            ))
        
        print(f"✓ Async security scan completed")
        
        return scanner.findings
    
    def test_dependency_vulnerabilities(self):
        """Test for known dependency vulnerabilities."""
        print("\n=== Testing Dependency Vulnerabilities ===")
        
        scanner = SecurityScanner()
        dependency_findings = scanner.scan_dependency_vulnerabilities()
        scanner.findings.extend(dependency_findings)
        
        # Additional dependency checks
        try:
            import pkg_resources
            
            # Check for development/debug packages in production
            debug_packages = ['ipdb', 'pdb', 'debugpy', 'pytest-xdist']
            
            for debug_pkg in debug_packages:
                try:
                    pkg_resources.get_distribution(debug_pkg)
                    scanner.add_finding(SecurityFinding(
                        severity='low',
                        category='dependency_vulnerability',
                        description=f'Debug package {debug_pkg} present in environment',
                        location='dependencies',
                        recommendation='Remove debug packages from production environment'
                    ))
                except pkg_resources.DistributionNotFound:
                    pass  # Good - debug package not present
            
            # Check for packages with known security issues
            security_sensitive_packages = {
                'torch': 'Ensure using latest version with security patches',
                'numpy': 'Check for buffer overflow vulnerabilities',
                'pillow': 'Verify version includes security fixes',
            }
            
            for pkg_name, recommendation in security_sensitive_packages.items():
                try:
                    package = pkg_resources.get_distribution(pkg_name)
                    scanner.add_finding(SecurityFinding(
                        severity='low',
                        category='dependency_security',
                        description=f'Security-sensitive package: {pkg_name} v{package.version}',
                        location='dependencies',
                        recommendation=recommendation
                    ))
                except pkg_resources.DistributionNotFound:
                    pass
        
        except Exception as e:
            scanner.add_finding(SecurityFinding(
                severity='low',
                category='dependency_security',
                description=f'Dependency check error: {e}',
                location='dependency scanner',
                recommendation='Review dependency security scanning'
            ))
        
        print(f"✓ Dependency vulnerability scan completed")
        
        return scanner.findings


@pytest.mark.security
class TestComplianceValidation:
    """Test compliance with security standards and regulations."""
    
    def test_defensive_use_compliance(self):
        """Test compliance with defensive use requirements."""
        print("\n=== Testing Defensive Use Compliance ===")
        
        # Check for responsible use notices in documentation
        docs_paths = [
            Path('README.md'),
            Path('SECURITY.md'),
            Path('docs/DEPLOYMENT.md')
        ]
        
        responsible_use_found = False
        defensive_focus_found = False
        
        for doc_path in docs_paths:
            if doc_path.exists():
                content = doc_path.read_text().lower()
                if 'responsible' in content or 'ethical' in content:
                    responsible_use_found = True
                if 'defensive' in content or 'protection' in content:
                    defensive_focus_found = True
        
        assert responsible_use_found, "Responsible use documentation not found"
        assert defensive_focus_found, "Defensive use focus not documented"
        
        print("✓ Defensive use compliance verified")
    
    def test_data_protection_compliance(self):
        """Test data protection and privacy compliance."""
        print("\n=== Testing Data Protection Compliance ===")
        
        # Test that no sensitive data is logged or exposed
        generator = SyntheticDatasetGenerator(random_seed=42)
        dataset = generator.generate_aes_dataset(n_traces=10, trace_length=100)
        
        # Check that key material is not exposed in logs or outputs
        key_hex = dataset['key'].hex()
        
        # Simulate logging to check for key exposure
        import io
        import contextlib
        
        log_capture = io.StringIO()
        
        with contextlib.redirect_stdout(log_capture):
            neural_sca = NeuralSCA()
            traces = torch.tensor(dataset['power_traces'], dtype=torch.float32).unsqueeze(-1)
            labels = torch.tensor(dataset['labels'][0], dtype=torch.long)
            model = neural_sca.train(traces, labels, validation_split=0.2)
        
        log_output = log_capture.getvalue().lower()
        
        # Check that key material doesn't appear in logs
        assert key_hex.lower() not in log_output, "Key material exposed in logs"
        assert str(dataset['key']).lower() not in log_output, "Key array exposed in logs"
        
        print("✓ Data protection compliance verified")
    
    def test_audit_trail_compliance(self):
        """Test audit trail and logging compliance."""
        print("\n=== Testing Audit Trail Compliance ===")
        
        # Test that security-relevant events are properly logged
        security_policy = SecurityPolicy()
        
        # Check that audit logging is enabled
        assert hasattr(security_policy, 'audit_log'), "Audit logging not implemented"
        
        # Test audit log functionality
        test_events = [
            {'action': 'train', 'user': 'test_user', 'timestamp': time.time()},
            {'action': 'attack', 'user': 'test_user', 'timestamp': time.time()},
        ]
        
        for event in test_events:
            try:
                security_policy.audit_log(event)
            except Exception as e:
                pytest.fail(f"Audit logging failed: {e}")
        
        print("✓ Audit trail compliance verified")


def generate_security_report(findings: List[SecurityFinding], output_path: Path = None) -> Dict[str, Any]:
    """Generate comprehensive security report."""
    scanner = SecurityScanner()
    scanner.findings = findings
    
    report = scanner.generate_report()
    
    # Add compliance information
    report['compliance'] = {
        'defensive_use': True,  # Assumes compliance tests pass
        'data_protection': True,
        'audit_trail': True,
    }
    
    # Add executive summary
    report['executive_summary'] = {
        'total_tests_run': len(findings),
        'critical_issues': len([f for f in findings if f.severity == 'critical']),
        'high_issues': len([f for f in findings if f.severity == 'high']),
        'overall_security_rating': _calculate_security_rating(report['risk_score']),
        'ready_for_production': report['risk_score'] < 30  # Threshold for production readiness
    }
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    return report


def _calculate_security_rating(risk_score: float) -> str:
    """Calculate security rating based on risk score."""
    if risk_score < 10:
        return 'EXCELLENT'
    elif risk_score < 25:
        return 'GOOD'
    elif risk_score < 50:
        return 'FAIR'
    elif risk_score < 75:
        return 'POOR'
    else:
        return 'CRITICAL'


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "security"])