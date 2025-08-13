#!/usr/bin/env python3
"""Comprehensive quality gates runner for neural cryptanalysis framework."""

import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import concurrent.futures
import threading


@dataclass
class QualityGateResult:
    """Result of a quality gate execution."""
    name: str
    passed: bool
    score: float
    execution_time: float
    details: Dict[str, Any]
    critical: bool = False
    error_message: str = ""


class QualityGateRunner:
    """Orchestrate and run all quality gates."""
    
    def __init__(self, parallel: bool = True, timeout: int = 600):
        self.parallel = parallel
        self.timeout = timeout
        self.results: List[QualityGateResult] = []
        self.start_time = time.time()
        
        # Define quality gates
        self.quality_gates = [
            {
                'name': 'Import Validation',
                'script': 'scripts/validate_imports.py',
                'args': ['src/neural_cryptanalysis/*.py'],
                'critical': True,
                'timeout': 60
            },
            {
                'name': 'Security Check',
                'script': 'scripts/security_check.py',
                'args': ['src/neural_cryptanalysis/*.py'],
                'critical': True,
                'timeout': 120
            },
            {
                'name': 'Performance Regression',
                'script': 'scripts/performance_check.py',
                'args': [],
                'critical': False,
                'timeout': 300
            },
            {
                'name': 'Test Coverage',
                'script': 'scripts/coverage_check.py',
                'args': ['85'],  # 85% minimum coverage
                'critical': True,
                'timeout': 240
            },
            {
                'name': 'Documentation Check',
                'script': 'scripts/docs_check.py',
                'args': ['src/neural_cryptanalysis/*.py'],
                'critical': False,
                'timeout': 120
            },
            {
                'name': 'Unit Tests',
                'command': [sys.executable, '-m', 'pytest', 'tests/test_comprehensive_unit_tests.py', '-v'],
                'critical': True,
                'timeout': 300
            },
            {
                'name': 'Integration Tests',
                'command': [sys.executable, '-m', 'pytest', 'tests/test_integration_workflows.py', '-v', '-m', 'integration'],
                'critical': True,
                'timeout': 600
            },
            {
                'name': 'Performance Tests',
                'command': [sys.executable, '-m', 'pytest', 'tests/test_performance_benchmarks.py', '-v', '-m', 'performance'],
                'critical': False,
                'timeout': 900
            },
            {
                'name': 'Security Tests',
                'command': [sys.executable, '-m', 'pytest', 'tests/test_security_validation.py', '-v', '-m', 'security'],
                'critical': True,
                'timeout': 300
            },
            {
                'name': 'Research Quality Gates',
                'command': [sys.executable, '-m', 'pytest', 'tests/test_research_quality_gates.py', '-v', '-m', 'research'],
                'critical': True,
                'timeout': 600
            }
        ]
    
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates."""
        print("🚀 Starting Comprehensive Quality Gates")
        print("=" * 60)
        
        if self.parallel:
            self._run_gates_parallel()
        else:
            self._run_gates_sequential()
        
        return self._generate_final_report()
    
    def _run_gates_parallel(self):
        """Run quality gates in parallel."""
        print("Running quality gates in parallel...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_gate = {}
            
            for gate in self.quality_gates:
                future = executor.submit(self._run_single_gate, gate)
                future_to_gate[future] = gate
            
            for future in concurrent.futures.as_completed(future_to_gate, timeout=self.timeout):
                gate = future_to_gate[future]
                try:
                    result = future.result()
                    self.results.append(result)
                    self._print_gate_result(result)
                except Exception as e:
                    error_result = QualityGateResult(
                        name=gate['name'],
                        passed=False,
                        score=0.0,
                        execution_time=0.0,
                        details={'error': str(e)},
                        critical=gate.get('critical', False),
                        error_message=str(e)
                    )
                    self.results.append(error_result)
                    self._print_gate_result(error_result)
    
    def _run_gates_sequential(self):
        """Run quality gates sequentially."""
        print("Running quality gates sequentially...")
        
        for gate in self.quality_gates:
            result = self._run_single_gate(gate)
            self.results.append(result)
            self._print_gate_result(result)
            
            # Stop on critical failures in sequential mode
            if not result.passed and result.critical:
                print(f"\n❌ Critical quality gate failed: {result.name}")
                print("Stopping execution due to critical failure.")
                break
    
    def _run_single_gate(self, gate: Dict[str, Any]) -> QualityGateResult:
        """Run a single quality gate."""
        gate_name = gate['name']
        gate_timeout = gate.get('timeout', 300)
        is_critical = gate.get('critical', False)
        
        start_time = time.time()
        
        try:
            # Prepare command
            if 'script' in gate:
                cmd = [sys.executable, gate['script']] + gate.get('args', [])
            elif 'command' in gate:
                cmd = gate['command']
            else:
                raise ValueError(f"No script or command defined for gate {gate_name}")
            
            # Expand glob patterns in arguments
            expanded_cmd = []
            for arg in cmd:
                if '*' in str(arg) and 'src/' in str(arg):
                    # Expand glob pattern
                    matches = list(Path('.').glob(str(arg)))
                    expanded_cmd.extend([str(m) for m in matches])
                else:
                    expanded_cmd.append(str(arg))
            
            # Run command
            result = subprocess.run(
                expanded_cmd,
                capture_output=True,
                text=True,
                timeout=gate_timeout
            )
            
            execution_time = time.time() - start_time
            
            # Parse result
            passed = result.returncode == 0
            
            # Try to parse JSON output for detailed results
            details = {}
            if result.stdout:
                try:
                    # Look for JSON in stdout
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if line.strip().startswith('{'):
                            details = json.loads(line.strip())
                            break
                except json.JSONDecodeError:
                    details = {'stdout': result.stdout, 'stderr': result.stderr}
            
            # Calculate score based on result
            score = 100.0 if passed else 0.0
            
            return QualityGateResult(
                name=gate_name,
                passed=passed,
                score=score,
                execution_time=execution_time,
                details=details,
                critical=is_critical,
                error_message=result.stderr if result.stderr else ""
            )
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return QualityGateResult(
                name=gate_name,
                passed=False,
                score=0.0,
                execution_time=execution_time,
                details={'error': 'Timeout'},
                critical=is_critical,
                error_message=f"Gate timed out after {gate_timeout} seconds"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return QualityGateResult(
                name=gate_name,
                passed=False,
                score=0.0,
                execution_time=execution_time,
                details={'error': str(e)},
                critical=is_critical,
                error_message=str(e)
            )
    
    def _print_gate_result(self, result: QualityGateResult):
        """Print result of a quality gate."""
        status_icon = "✅" if result.passed else "❌"
        critical_marker = " [CRITICAL]" if result.critical else ""
        
        print(f"{status_icon} {result.name}{critical_marker}")
        print(f"   Score: {result.score:.1f}% | Time: {result.execution_time:.2f}s")
        
        if not result.passed and result.error_message:
            print(f"   Error: {result.error_message}")
        
        print()
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        total_time = time.time() - self.start_time
        
        # Calculate statistics
        total_gates = len(self.results)
        passed_gates = sum(1 for r in self.results if r.passed)
        failed_gates = total_gates - passed_gates
        
        critical_results = [r for r in self.results if r.critical]
        critical_passed = sum(1 for r in critical_results if r.passed)
        critical_failed = len(critical_results) - critical_passed
        
        overall_score = (passed_gates / total_gates * 100) if total_gates > 0 else 0
        critical_score = (critical_passed / len(critical_results) * 100) if critical_results else 100
        
        # Determine overall status
        overall_passed = (critical_failed == 0 and failed_gates <= total_gates * 0.2)  # Allow 20% non-critical failures
        
        # Create detailed report
        report = {
            'timestamp': time.time(),
            'execution_time_seconds': total_time,
            'overall_status': 'PASSED' if overall_passed else 'FAILED',
            'overall_score': overall_score,
            'statistics': {
                'total_gates': total_gates,
                'passed_gates': passed_gates,
                'failed_gates': failed_gates,
                'critical_gates': len(critical_results),
                'critical_passed': critical_passed,
                'critical_failed': critical_failed
            },
            'scores': {
                'overall_score': overall_score,
                'critical_score': critical_score
            },
            'gate_results': [
                {
                    'name': r.name,
                    'passed': r.passed,
                    'score': r.score,
                    'execution_time': r.execution_time,
                    'critical': r.critical,
                    'error_message': r.error_message
                } for r in self.results
            ],
            'failed_gates': [
                {
                    'name': r.name,
                    'critical': r.critical,
                    'error': r.error_message,
                    'details': r.details
                } for r in self.results if not r.passed
            ],
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on failed gates."""
        recommendations = []
        
        failed_results = [r for r in self.results if not r.passed]
        critical_failures = [r for r in failed_results if r.critical]
        
        if critical_failures:
            recommendations.append("Fix critical quality gate failures before proceeding to production")
        
        if len(failed_results) > len(self.results) * 0.3:
            recommendations.append("High failure rate indicates systematic quality issues")
        
        # Specific recommendations based on gate types
        failed_gate_names = [r.name.lower() for r in failed_results]
        
        if any('security' in name for name in failed_gate_names):
            recommendations.append("Address security vulnerabilities before deployment")
        
        if any('coverage' in name for name in failed_gate_names):
            recommendations.append("Increase test coverage to meet quality standards")
        
        if any('performance' in name for name in failed_gate_names):
            recommendations.append("Investigate performance regressions and optimize where needed")
        
        if any('documentation' in name for name in failed_gate_names):
            recommendations.append("Improve code documentation and API documentation")
        
        return recommendations
    
    def print_final_summary(self, report: Dict[str, Any]):
        """Print final quality gates summary."""
        print("=" * 60)
        print("🏁 QUALITY GATES SUMMARY")
        print("=" * 60)
        
        status = report['overall_status']
        status_icon = "✅" if status == 'PASSED' else "❌"
        
        print(f"{status_icon} Overall Status: {status}")
        print(f"📊 Overall Score: {report['overall_score']:.1f}%")
        print(f"⏱️  Total Execution Time: {report['execution_time_seconds']:.1f}s")
        
        stats = report['statistics']
        print(f"\n📈 Statistics:")
        print(f"   Total Gates: {stats['total_gates']}")
        print(f"   Passed: {stats['passed_gates']}")
        print(f"   Failed: {stats['failed_gates']}")
        print(f"   Critical Passed: {stats['critical_passed']}/{stats['critical_gates']}")
        
        if report['failed_gates']:
            print(f"\n❌ Failed Gates:")
            for failed in report['failed_gates']:
                critical_marker = " [CRITICAL]" if failed['critical'] else ""
                print(f"   • {failed['name']}{critical_marker}")
                if failed['error']:
                    print(f"     Error: {failed['error']}")
        
        if report['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in report['recommendations']:
                print(f"   • {rec}")
        
        print("\n" + "=" * 60)


def main():
    """Main quality gates execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comprehensive quality gates')
    parser.add_argument('--sequential', action='store_true', 
                       help='Run gates sequentially instead of parallel')
    parser.add_argument('--timeout', type=int, default=1800,
                       help='Overall timeout in seconds (default: 1800)')
    parser.add_argument('--output', type=str, default='quality_gates_report.json',
                       help='Output report file (default: quality_gates_report.json)')
    
    args = parser.parse_args()
    
    # Create runner
    runner = QualityGateRunner(
        parallel=not args.sequential,
        timeout=args.timeout
    )
    
    try:
        # Run all quality gates
        report = runner.run_all_gates()
        
        # Save report
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        runner.print_final_summary(report)
        
        # Exit with appropriate code
        if report['overall_status'] == 'PASSED':
            print(f"✅ All quality gates passed! Report saved to {args.output}")
            sys.exit(0)
        else:
            print(f"❌ Quality gates failed! Report saved to {args.output}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⚠️  Quality gates execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Quality gates execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()