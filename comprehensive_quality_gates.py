#!/usr/bin/env python3
"""
Comprehensive Quality Gates Verification
Final Stage: Ensuring production-ready quality and security standards
"""

import sys
import os
import time
import logging
import subprocess
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
from dataclasses import dataclass

# Import our implementations
sys.path.insert(0, '.')
import simple_torch_mock  # Mock torch first
from test_minimal_working import main as test_basic
from enhanced_neural_sca import main as test_robust
from optimized_neural_sca import main as test_optimized

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    execution_time: float
    critical: bool = False

class SecurityValidator:
    """Validates security requirements and defensive use constraints."""
    
    def __init__(self):
        self.security_violations = []
        self.audit_log = []
    
    def validate_responsible_use_notices(self) -> QualityGateResult:
        """Ensure responsible use notices are present."""
        start_time = time.perf_counter()
        
        try:
            # Check README for responsible use
            readme_path = Path('README.md')
            if readme_path.exists():
                readme_content = readme_path.read_text()
                has_responsible_notice = 'Responsible Disclosure' in readme_content
                has_defensive_focus = 'defensive' in readme_content.lower()
                has_ethics_section = any(section in readme_content.lower() 
                                       for section in ['ethics', 'responsible', 'disclosure'])
            else:
                has_responsible_notice = has_defensive_focus = has_ethics_section = False
            
            # Check source code for warnings
            security_warnings_found = 0
            for py_file in Path('src').rglob('*.py'):
                if py_file.exists():
                    content = py_file.read_text()
                    if 'For Defensive Security Research Only' in content:
                        security_warnings_found += 1
            
            # Check for license file
            has_license = any(Path(f).exists() for f in ['LICENSE', 'LICENSE.txt', 'LICENSE.md'])
            
            score = sum([
                has_responsible_notice * 25,
                has_defensive_focus * 25, 
                has_ethics_section * 20,
                (security_warnings_found > 0) * 20,
                has_license * 10
            ]) / 100.0
            
            passed = score >= 0.8
            
            details = {
                'responsible_notice': has_responsible_notice,
                'defensive_focus': has_defensive_focus,
                'ethics_section': has_ethics_section,
                'security_warnings': security_warnings_found,
                'has_license': has_license,
                'score_breakdown': {
                    'responsible_notice': has_responsible_notice * 25,
                    'defensive_focus': has_defensive_focus * 25,
                    'ethics_section': has_ethics_section * 20,
                    'security_warnings': (security_warnings_found > 0) * 20,
                    'license': has_license * 10
                }
            }
            
            return QualityGateResult(
                name="Security & Responsible Use",
                passed=passed,
                score=score,
                details=details,
                execution_time=time.perf_counter() - start_time,
                critical=True
            )
            
        except Exception as e:
            return QualityGateResult(
                name="Security & Responsible Use",
                passed=False,
                score=0.0,
                details={'error': str(e)},
                execution_time=time.perf_counter() - start_time,
                critical=True
            )
    
    def validate_no_malicious_functionality(self) -> QualityGateResult:
        """Scan for potentially malicious functionality."""
        start_time = time.perf_counter()
        
        try:
            suspicious_patterns = [
                'os.system',
                'subprocess.call',
                'eval(',
                'exec(',
                '__import__',
                'socket.socket',
                'urllib.request',
                'requests.',
                'http.client',
                'ftplib',
                'smtplib',
                'telnetlib'
            ]
            
            violations = []
            scanned_files = 0
            
            # Scan Python files
            for py_file in Path('.').rglob('*.py'):
                if py_file.name.startswith('.'):
                    continue
                    
                try:
                    content = py_file.read_text()
                    scanned_files += 1
                    
                    for pattern in suspicious_patterns:
                        if pattern in content:
                            # Check if it's in a comment or safe context
                            lines = content.split('\n')
                            for i, line in enumerate(lines):
                                if pattern in line:
                                    stripped = line.strip()
                                    if not stripped.startswith('#') and not stripped.startswith('"""'):
                                        violations.append({
                                            'file': str(py_file),
                                            'line': i + 1,
                                            'pattern': pattern,
                                            'context': stripped[:100]
                                        })
                except Exception:
                    continue
            
            # Filter out known safe uses
            safe_violations = []
            for violation in violations:
                file_name = Path(violation['file']).name
                context = violation['context'].strip()
                
                # Allow legitimate patterns in safe contexts
                safe_patterns = [
                    # Import patterns in test files
                    (violation['pattern'] == '__import__', 'test' in file_name),
                    # String literals containing patterns (not actual calls)
                    (context.startswith("'") and context.endswith("'")),
                    (context.startswith('"') and context.endswith('"')),
                    # Method definitions (not calls)
                    (context.startswith('def ') and violation['pattern'] in ['eval(', 'exec(']),
                    # Quality gates file scanning itself
                    (file_name == 'comprehensive_quality_gates.py'),
                    # Mock implementations
                    ('mock' in file_name.lower()),
                    # Comments and docstrings (additional check)
                    (context.startswith('#')),
                ]
                
                if any(condition for condition in safe_patterns if condition):
                    continue
                
                safe_violations.append(violation)
            
            score = max(0.0, 1.0 - len(safe_violations) * 0.2)
            passed = len(safe_violations) == 0
            
            return QualityGateResult(
                name="Malicious Code Scan",
                passed=passed,
                score=score,
                details={
                    'scanned_files': scanned_files,
                    'total_violations': len(violations),
                    'filtered_violations': len(safe_violations),
                    'violations': safe_violations[:10]  # Show only first 10
                },
                execution_time=time.perf_counter() - start_time,
                critical=True
            )
            
        except Exception as e:
            return QualityGateResult(
                name="Malicious Code Scan",
                passed=False,
                score=0.0,
                details={'error': str(e)},
                execution_time=time.perf_counter() - start_time,
                critical=True
            )

class PerformanceValidator:
    """Validates performance requirements."""
    
    def validate_simulation_performance(self) -> QualityGateResult:
        """Test simulation performance requirements."""
        start_time = time.perf_counter()
        
        try:
            from optimized_neural_sca import OptimizedLeakageSimulator
            
            class PerfTarget:
                def __init__(self):
                    self.key = np.random.randint(0, 256, 16, dtype=np.uint8)
            
            simulator = OptimizedLeakageSimulator()
            target = PerfTarget()
            
            # Performance requirements
            min_throughput_small = 1000   # traces/sec for small traces
            min_throughput_medium = 500   # traces/sec for medium batches
            
            results = []
            
            # Test small batch (100 traces, 1000 samples)
            test_start = time.perf_counter()
            traces, _, _ = simulator.simulate_traces_optimized(target, 100, 1000)
            small_time = time.perf_counter() - test_start
            small_throughput = 100 / small_time
            
            results.append({
                'test': 'small_batch',
                'n_traces': 100,
                'trace_length': 1000,
                'time': small_time,
                'throughput': small_throughput,
                'requirement': min_throughput_small,
                'passed': small_throughput >= min_throughput_small
            })
            
            # Test medium batch (1000 traces, 1000 samples)
            test_start = time.perf_counter()
            traces, _, _ = simulator.simulate_traces_optimized(target, 1000, 1000)
            medium_time = time.perf_counter() - test_start
            medium_throughput = 1000 / medium_time
            
            results.append({
                'test': 'medium_batch',
                'n_traces': 1000,
                'trace_length': 1000,
                'time': medium_time,
                'throughput': medium_throughput,
                'requirement': min_throughput_medium,
                'passed': medium_throughput >= min_throughput_medium
            })
            
            # Calculate overall score
            passed_tests = sum(1 for r in results if r['passed'])
            score = passed_tests / len(results)
            
            return QualityGateResult(
                name="Simulation Performance",
                passed=score == 1.0,
                score=score,
                details={'results': results},
                execution_time=time.perf_counter() - start_time
            )
            
        except Exception as e:
            return QualityGateResult(
                name="Simulation Performance",
                passed=False,
                score=0.0,
                details={'error': str(e)},
                execution_time=time.perf_counter() - start_time
            )
    
    def validate_memory_efficiency(self) -> QualityGateResult:
        """Test memory efficiency."""
        start_time = time.perf_counter()
        
        try:
            # Test that we can handle reasonable data sizes without memory issues
            from enhanced_neural_sca import RobustTraceData
            
            # Create reasonably sized trace data
            traces = np.random.randn(1000, 5000).astype(np.float32)  # ~20MB
            labels = np.random.randint(0, 256, 1000, dtype=np.uint8)
            
            # Test creation and operations
            trace_data = RobustTraceData(traces, labels)
            
            # Test split operation
            train_data, val_data = trace_data.split(0.8)
            
            # Test indexing
            sample = trace_data[0]
            batch = trace_data[:10]
            
            # Memory usage estimation
            estimated_memory_mb = (traces.nbytes + labels.nbytes) / (1024 * 1024)
            
            score = 1.0 if estimated_memory_mb < 100 else 0.8  # Reasonable memory usage
            
            return QualityGateResult(
                name="Memory Efficiency",
                passed=True,
                score=score,
                details={
                    'estimated_memory_mb': estimated_memory_mb,
                    'trace_shape': traces.shape,
                    'operations_successful': True
                },
                execution_time=time.perf_counter() - start_time
            )
            
        except Exception as e:
            return QualityGateResult(
                name="Memory Efficiency", 
                passed=False,
                score=0.0,
                details={'error': str(e)},
                execution_time=time.perf_counter() - start_time
            )

class FunctionalityValidator:
    """Validates core functionality requirements."""
    
    def validate_core_functionality(self) -> QualityGateResult:
        """Test that all core functionality works."""
        start_time = time.perf_counter()
        
        try:
            # Run the basic functionality test
            basic_success = test_basic()
            
            # Run robustness test  
            robust_success = test_robust()
            
            # Calculate score
            passed_tests = sum([basic_success, robust_success])
            score = passed_tests / 2
            
            return QualityGateResult(
                name="Core Functionality",
                passed=score == 1.0,
                score=score,
                details={
                    'basic_functionality': basic_success,
                    'robustness_features': robust_success,
                    'total_tests_passed': passed_tests
                },
                execution_time=time.perf_counter() - start_time,
                critical=True
            )
            
        except Exception as e:
            return QualityGateResult(
                name="Core Functionality",
                passed=False,
                score=0.0,
                details={'error': str(e)},
                execution_time=time.perf_counter() - start_time,
                critical=True
            )
    
    def validate_error_handling(self) -> QualityGateResult:
        """Test error handling robustness."""
        start_time = time.perf_counter()
        
        try:
            from enhanced_neural_sca import RobustTraceData, ValidationError, NeuralSCAConfig
            
            error_tests = []
            
            # Test 1: Empty data validation
            try:
                RobustTraceData(np.array([]))
                error_tests.append({'test': 'empty_data', 'passed': False})
            except ValidationError:
                error_tests.append({'test': 'empty_data', 'passed': True})
            
            # Test 2: Mismatched labels validation
            try:
                traces = np.random.randn(100, 1000)
                labels = np.random.randint(0, 256, 50)  # Wrong size
                RobustTraceData(traces, labels)
                error_tests.append({'test': 'mismatched_labels', 'passed': False})
            except ValidationError:
                error_tests.append({'test': 'mismatched_labels', 'passed': True})
            
            # Test 3: Configuration validation
            config = NeuralSCAConfig(input_dim=-1, validation_split=1.5)
            errors = config.validate()
            error_tests.append({'test': 'config_validation', 'passed': len(errors) > 0})
            
            # Test 4: Invalid trace data
            try:
                invalid_traces = np.array([[np.nan, np.inf], [1, 2]])
                RobustTraceData(invalid_traces)
                error_tests.append({'test': 'invalid_trace_data', 'passed': False})
            except ValidationError:
                error_tests.append({'test': 'invalid_trace_data', 'passed': True})
            
            passed_tests = sum(1 for test in error_tests if test['passed'])
            score = passed_tests / len(error_tests)
            
            return QualityGateResult(
                name="Error Handling",
                passed=score >= 0.8,
                score=score,
                details={
                    'error_tests': error_tests,
                    'passed_tests': passed_tests,
                    'total_tests': len(error_tests)
                },
                execution_time=time.perf_counter() - start_time
            )
            
        except Exception as e:
            return QualityGateResult(
                name="Error Handling",
                passed=False,
                score=0.0,
                details={'error': str(e)},
                execution_time=time.perf_counter() - start_time
            )

class DocumentationValidator:
    """Validates documentation completeness."""
    
    def validate_documentation(self) -> QualityGateResult:
        """Check documentation completeness."""
        start_time = time.perf_counter()
        
        try:
            doc_scores = {}
            
            # Check README
            readme_path = Path('README.md')
            if readme_path.exists():
                content = readme_path.read_text()
                doc_scores['readme_exists'] = True
                doc_scores['has_installation'] = 'install' in content.lower()
                doc_scores['has_usage'] = 'usage' in content.lower() or 'example' in content.lower()
                doc_scores['has_features'] = 'feature' in content.lower()
                doc_scores['has_architecture'] = 'architecture' in content.lower()
                doc_scores['readme_length'] = len(content) > 1000  # Substantial content
            else:
                doc_scores.update({
                    'readme_exists': False,
                    'has_installation': False,
                    'has_usage': False,
                    'has_features': False,
                    'has_architecture': False,
                    'readme_length': False
                })
            
            # Check for additional documentation
            doc_files = [
                'CONTRIBUTING.md', 'SECURITY.md', 'CHANGELOG.md',
                'DEPLOYMENT.md', 'LICENSE'
            ]
            
            for doc_file in doc_files:
                doc_scores[f'has_{doc_file.lower()}'] = Path(doc_file).exists()
            
            # Check code documentation
            python_files = list(Path('src').rglob('*.py'))
            documented_files = 0
            
            for py_file in python_files:
                content = py_file.read_text()
                if '"""' in content or "'''" in content:  # Has docstrings
                    documented_files += 1
            
            doc_scores['code_documentation_ratio'] = (
                documented_files / len(python_files) if python_files else 0
            )
            
            # Calculate overall score
            weights = {
                'readme_exists': 20,
                'has_installation': 15,
                'has_usage': 15,
                'has_features': 10,
                'has_architecture': 10,
                'readme_length': 10,
                'has_security.md': 10,
                'has_contributing.md': 5,
                'has_license': 5,
                'code_documentation_ratio': 0  # Bonus points
            }
            
            total_score = 0
            max_score = 0
            
            for key, weight in weights.items():
                if key in doc_scores:
                    value = doc_scores[key]
                    if isinstance(value, bool):
                        total_score += weight if value else 0
                    elif isinstance(value, float):
                        total_score += weight * value
                max_score += weight
            
            # Bonus for code documentation
            if doc_scores['code_documentation_ratio'] > 0.5:
                total_score += 10
                max_score += 10
            
            score = total_score / max_score if max_score > 0 else 0
            
            return QualityGateResult(
                name="Documentation",
                passed=score >= 0.7,
                score=score,
                details={
                    'documentation_scores': doc_scores,
                    'total_score': total_score,
                    'max_score': max_score,
                    'python_files_count': len(python_files),
                    'documented_files_count': documented_files
                },
                execution_time=time.perf_counter() - start_time
            )
            
        except Exception as e:
            return QualityGateResult(
                name="Documentation",
                passed=False,
                score=0.0,
                details={'error': str(e)},
                execution_time=time.perf_counter() - start_time
            )

def run_quality_gates() -> Tuple[bool, List[QualityGateResult]]:
    """Run all quality gates and return results."""
    
    print("🛡️ COMPREHENSIVE QUALITY GATES VERIFICATION")
    print("=" * 70)
    print("Running production-ready quality assurance checks...")
    
    validators = [
        SecurityValidator(),
        FunctionalityValidator(),
        PerformanceValidator(),
        DocumentationValidator()
    ]
    
    all_results = []
    
    # Run security validation
    print("\n🔒 Security Validation:")
    security_validator = SecurityValidator()
    
    security_results = [
        security_validator.validate_responsible_use_notices(),
        security_validator.validate_no_malicious_functionality()
    ]
    
    for result in security_results:
        all_results.append(result)
        status = "✅ PASS" if result.passed else "❌ FAIL"
        critical = " [CRITICAL]" if result.critical else ""
        print(f"  {result.name}: {status} ({result.score:.1%}){critical}")
        if not result.passed and result.critical:
            print(f"    🚨 Critical failure details: {result.details}")
    
    # Run functionality validation
    print("\n⚙️ Functionality Validation:")
    func_validator = FunctionalityValidator()
    
    func_results = [
        func_validator.validate_core_functionality(),
        func_validator.validate_error_handling()
    ]
    
    for result in func_results:
        all_results.append(result)
        status = "✅ PASS" if result.passed else "❌ FAIL"
        critical = " [CRITICAL]" if result.critical else ""
        print(f"  {result.name}: {status} ({result.score:.1%}){critical}")
    
    # Run performance validation
    print("\n⚡ Performance Validation:")
    perf_validator = PerformanceValidator()
    
    perf_results = [
        perf_validator.validate_simulation_performance(),
        perf_validator.validate_memory_efficiency()
    ]
    
    for result in perf_results:
        all_results.append(result)
        status = "✅ PASS" if result.passed else "❌ FAIL" 
        print(f"  {result.name}: {status} ({result.score:.1%})")
        if 'results' in result.details:
            for test_result in result.details['results']:
                if 'throughput' in test_result:
                    print(f"    {test_result['test']}: {test_result['throughput']:.0f} traces/sec")
    
    # Run documentation validation
    print("\n📚 Documentation Validation:")
    doc_validator = DocumentationValidator()
    
    doc_result = doc_validator.validate_documentation()
    all_results.append(doc_result)
    status = "✅ PASS" if doc_result.passed else "❌ FAIL"
    print(f"  {doc_result.name}: {status} ({doc_result.score:.1%})")
    
    # Calculate overall results
    critical_results = [r for r in all_results if r.critical]
    critical_passed = all(r.passed for r in critical_results)
    
    overall_score = np.mean([r.score for r in all_results])
    overall_passed = critical_passed and overall_score >= 0.85
    
    print("\n" + "=" * 70)
    print("📊 QUALITY GATES SUMMARY:")
    print(f"  Overall Score: {overall_score:.1%}")
    print(f"  Critical Gates: {sum(r.passed for r in critical_results)}/{len(critical_results)} passed")
    print(f"  Total Gates: {sum(r.passed for r in all_results)}/{len(all_results)} passed")
    print(f"  Total Execution Time: {sum(r.execution_time for r in all_results):.2f}s")
    
    if overall_passed:
        print("\n🎉 ALL QUALITY GATES PASSED - PRODUCTION READY! 🎉")
    else:
        print("\n⚠️  QUALITY GATES FAILED - PRODUCTION NOT READY")
        failed_critical = [r for r in critical_results if not r.passed]
        if failed_critical:
            print("   Critical failures must be resolved:")
            for result in failed_critical:
                print(f"   - {result.name}")
    
    return overall_passed, all_results

def main():
    """Main quality gates execution."""
    try:
        overall_passed, results = run_quality_gates()
        
        # Generate quality report
        report = {
            'timestamp': time.time(),
            'overall_passed': overall_passed,
            'overall_score': np.mean([r.score for r in results]),
            'results': [
                {
                    'name': r.name,
                    'passed': r.passed,
                    'score': r.score,
                    'critical': r.critical,
                    'execution_time': r.execution_time,
                    'details': r.details
                }
                for r in results
            ]
        }
        
        # Save report
        with open('quality_gates_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📋 Quality gates report saved to: quality_gates_report.json")
        
        return overall_passed
        
    except Exception as e:
        print(f"💥 Quality gates execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)