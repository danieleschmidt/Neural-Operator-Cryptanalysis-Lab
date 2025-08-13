#!/usr/bin/env python3
"""Test coverage validation for neural cryptanalysis framework."""

import sys
import subprocess
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional


class CoverageValidator:
    """Validate test coverage requirements."""
    
    def __init__(self, min_coverage: float = 85.0):
        self.min_coverage = min_coverage
        self.coverage_data = {}
        self.coverage_report = {}
    
    def run_coverage_analysis(self) -> Dict[str, Any]:
        """Run coverage analysis using pytest-cov."""
        try:
            # Run tests with coverage
            cmd = [
                sys.executable, '-m', 'pytest',
                '--cov=src/neural_cryptanalysis',
                '--cov-report=term-missing',
                '--cov-report=json:coverage.json',
                '--cov-report=xml:coverage.xml',
                '--cov-fail-under=0',  # Don't fail here, we'll check manually
                'tests/',
                '-v',
                '--tb=short'
            ]
            
            print("Running test coverage analysis...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Parse coverage results
            self._parse_coverage_results()
            
            return {
                'test_exit_code': result.returncode,
                'test_output': result.stdout,
                'test_errors': result.stderr,
                'coverage_data': self.coverage_data
            }
        
        except subprocess.TimeoutExpired:
            return {
                'test_exit_code': -1,
                'test_output': '',
                'test_errors': 'Test execution timed out',
                'coverage_data': {}
            }
        except Exception as e:
            return {
                'test_exit_code': -1,
                'test_output': '',
                'test_errors': str(e),
                'coverage_data': {}
            }
    
    def _parse_coverage_results(self):
        """Parse coverage results from JSON and XML reports."""
        # Parse JSON coverage report
        json_coverage_file = Path('coverage.json')
        if json_coverage_file.exists():
            try:
                with open(json_coverage_file, 'r') as f:
                    self.coverage_data = json.load(f)
            except Exception as e:
                print(f"Failed to parse JSON coverage: {e}")
        
        # Parse XML coverage report for additional details
        xml_coverage_file = Path('coverage.xml')
        if xml_coverage_file.exists():
            try:
                self._parse_xml_coverage(xml_coverage_file)
            except Exception as e:
                print(f"Failed to parse XML coverage: {e}")
    
    def _parse_xml_coverage(self, xml_file: Path):
        """Parse XML coverage report for detailed analysis."""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Extract package-level coverage
        packages = {}
        for package in root.findall('.//package'):
            package_name = package.get('name', '')
            package_coverage = {
                'line_rate': float(package.get('line-rate', 0)) * 100,
                'branch_rate': float(package.get('branch-rate', 0)) * 100,
                'classes': {}
            }
            
            # Extract class-level coverage
            for class_elem in package.findall('.//class'):
                class_name = class_elem.get('name', '')
                class_coverage = {
                    'line_rate': float(class_elem.get('line-rate', 0)) * 100,
                    'branch_rate': float(class_elem.get('branch-rate', 0)) * 100,
                    'methods': {}
                }
                
                package_coverage['classes'][class_name] = class_coverage
            
            packages[package_name] = package_coverage
        
        self.coverage_report['packages'] = packages
    
    def validate_coverage_requirements(self) -> Dict[str, Any]:
        """Validate coverage against requirements."""
        validation_results = {
            'overall_passed': False,
            'total_coverage': 0.0,
            'module_coverage': {},
            'missing_coverage': [],
            'critical_missing': [],
            'issues': []
        }
        
        if not self.coverage_data:
            validation_results['issues'].append({
                'type': 'no_coverage_data',
                'severity': 'critical',
                'message': 'No coverage data available',
                'recommendation': 'Ensure tests run successfully and coverage is collected'
            })
            return validation_results
        
        # Extract overall coverage
        totals = self.coverage_data.get('totals', {})
        total_coverage = totals.get('percent_covered', 0)
        validation_results['total_coverage'] = total_coverage
        
        # Check overall coverage requirement
        if total_coverage >= self.min_coverage:
            validation_results['overall_passed'] = True
        else:
            validation_results['issues'].append({
                'type': 'insufficient_overall_coverage',
                'severity': 'critical',
                'message': f'Overall coverage {total_coverage:.1f}% below minimum {self.min_coverage:.1f}%',
                'recommendation': f'Add tests to increase coverage by {self.min_coverage - total_coverage:.1f}%'
            })
        
        # Analyze file-level coverage
        files = self.coverage_data.get('files', {})
        critical_modules = [
            'neural_cryptanalysis/core.py',
            'neural_cryptanalysis/neural_operators/base.py',
            'neural_cryptanalysis/neural_operators/fno.py',
            'neural_cryptanalysis/security.py'
        ]
        
        for file_path, file_coverage in files.items():
            coverage_percent = file_coverage.get('summary', {}).get('percent_covered', 0)
            validation_results['module_coverage'][file_path] = coverage_percent
            
            # Check critical modules
            if any(critical in file_path for critical in critical_modules):
                if coverage_percent < self.min_coverage:
                    validation_results['critical_missing'].append({
                        'file': file_path,
                        'coverage': coverage_percent,
                        'required': self.min_coverage
                    })
            
            # Check for completely untested files
            if coverage_percent == 0:
                validation_results['missing_coverage'].append(file_path)
        
        # Validate test quality
        self._validate_test_quality(validation_results)
        
        return validation_results
    
    def _validate_test_quality(self, validation_results: Dict[str, Any]):
        """Validate quality of tests beyond just coverage."""
        
        # Check for test files
        test_dir = Path('tests')
        if not test_dir.exists():
            validation_results['issues'].append({
                'type': 'no_test_directory',
                'severity': 'critical',
                'message': 'No tests directory found',
                'recommendation': 'Create tests directory with test files'
            })
            return
        
        test_files = list(test_dir.glob('test_*.py'))
        if len(test_files) == 0:
            validation_results['issues'].append({
                'type': 'no_test_files',
                'severity': 'critical',
                'message': 'No test files found',
                'recommendation': 'Create test files following test_*.py naming convention'
            })
            return
        
        # Check test file coverage
        source_files = list(Path('src/neural_cryptanalysis').rglob('*.py'))
        source_files = [f for f in source_files if '__pycache__' not in str(f) and '__init__.py' not in str(f)]
        
        test_ratio = len(test_files) / len(source_files) if source_files else 0
        
        if test_ratio < 0.3:  # Less than 30% test-to-source ratio
            validation_results['issues'].append({
                'type': 'insufficient_test_files',
                'severity': 'medium',
                'message': f'Low test-to-source ratio: {test_ratio:.2f}',
                'recommendation': 'Consider adding more test files to improve coverage'
            })
        
        # Check for different types of tests
        test_types = {
            'unit': any('unit' in f.name for f in test_files),
            'integration': any('integration' in f.name for f in test_files),
            'performance': any('performance' in f.name for f in test_files),
            'security': any('security' in f.name for f in test_files)
        }
        
        missing_test_types = [test_type for test_type, exists in test_types.items() if not exists]
        
        if missing_test_types:
            validation_results['issues'].append({
                'type': 'missing_test_types',
                'severity': 'medium',
                'message': f'Missing test types: {", ".join(missing_test_types)}',
                'recommendation': 'Consider adding comprehensive test types for better coverage'
            })
    
    def generate_coverage_report(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive coverage report."""
        
        # Calculate statistics
        module_coverages = list(validation_results['module_coverage'].values())
        
        if module_coverages:
            avg_module_coverage = sum(module_coverages) / len(module_coverages)
            min_module_coverage = min(module_coverages)
            max_module_coverage = max(module_coverages)
        else:
            avg_module_coverage = min_module_coverage = max_module_coverage = 0
        
        # Generate recommendations
        recommendations = []
        
        if validation_results['total_coverage'] < self.min_coverage:
            gap = self.min_coverage - validation_results['total_coverage']
            recommendations.append(f"Increase overall coverage by {gap:.1f}% to meet minimum requirement")
        
        if validation_results['missing_coverage']:
            recommendations.append(f"Add tests for {len(validation_results['missing_coverage'])} completely untested files")
        
        if validation_results['critical_missing']:
            recommendations.append(f"Prioritize testing {len(validation_results['critical_missing'])} critical modules")
        
        # Calculate coverage grade
        total_coverage = validation_results['total_coverage']
        if total_coverage >= 95:
            grade = 'A+'
        elif total_coverage >= 90:
            grade = 'A'
        elif total_coverage >= 85:
            grade = 'B+'
        elif total_coverage >= 80:
            grade = 'B'
        elif total_coverage >= 70:
            grade = 'C+'
        elif total_coverage >= 60:
            grade = 'C'
        else:
            grade = 'F'
        
        return {
            'timestamp': Path.cwd().name,  # Simple timestamp
            'coverage_summary': {
                'total_coverage': validation_results['total_coverage'],
                'minimum_required': self.min_coverage,
                'grade': grade,
                'passed_requirements': validation_results['overall_passed']
            },
            'module_statistics': {
                'total_modules': len(validation_results['module_coverage']),
                'average_coverage': avg_module_coverage,
                'minimum_coverage': min_module_coverage,
                'maximum_coverage': max_module_coverage,
                'modules_below_threshold': len([c for c in module_coverages if c < self.min_coverage])
            },
            'quality_issues': {
                'total_issues': len(validation_results['issues']),
                'critical_issues': len([i for i in validation_results['issues'] if i['severity'] == 'critical']),
                'medium_issues': len([i for i in validation_results['issues'] if i['severity'] == 'medium']),
                'issues': validation_results['issues']
            },
            'missing_coverage': {
                'untested_files': validation_results['missing_coverage'],
                'critical_modules_missing': validation_results['critical_missing']
            },
            'recommendations': recommendations,
            'detailed_module_coverage': validation_results['module_coverage']
        }


def main():
    """Main coverage checking function."""
    
    # Set minimum coverage from command line or default
    min_coverage = 85.0
    if len(sys.argv) > 1:
        try:
            min_coverage = float(sys.argv[1])
        except ValueError:
            print(f"Invalid coverage threshold: {sys.argv[1]}")
            sys.exit(1)
    
    print(f"Checking test coverage (minimum: {min_coverage:.1f}%)...")
    
    validator = CoverageValidator(min_coverage=min_coverage)
    
    # Run coverage analysis
    analysis_results = validator.run_coverage_analysis()
    
    if analysis_results['test_exit_code'] not in [0, 1]:  # 1 is ok if tests fail but coverage runs
        print("❌ Test execution failed:")
        print(analysis_results['test_errors'])
        sys.exit(1)
    
    # Validate coverage requirements
    validation_results = validator.validate_coverage_requirements()
    
    # Generate comprehensive report
    coverage_report = validator.generate_coverage_report(validation_results)
    
    # Save report
    with open('coverage_validation_report.json', 'w') as f:
        json.dump(coverage_report, f, indent=2)
    
    # Print results
    summary = coverage_report['coverage_summary']
    print(f"\nCoverage Results:")
    print(f"  Total Coverage: {summary['total_coverage']:.1f}% (Grade: {summary['grade']})")
    print(f"  Required: {summary['minimum_required']:.1f}%")
    print(f"  Status: {'✅ PASSED' if summary['passed_requirements'] else '❌ FAILED'}")
    
    stats = coverage_report['module_statistics']
    print(f"\nModule Statistics:")
    print(f"  Total modules: {stats['total_modules']}")
    print(f"  Average coverage: {stats['average_coverage']:.1f}%")
    print(f"  Modules below threshold: {stats['modules_below_threshold']}")
    
    issues = coverage_report['quality_issues']
    if issues['total_issues'] > 0:
        print(f"\nQuality Issues: {issues['total_issues']}")
        for issue in issues['issues']:
            severity_icon = "🔴" if issue['severity'] == 'critical' else "🟡"
            print(f"  {severity_icon} {issue['message']}")
    
    if coverage_report['recommendations']:
        print(f"\nRecommendations:")
        for rec in coverage_report['recommendations']:
            print(f"  • {rec}")
    
    # Exit based on results
    if not summary['passed_requirements']:
        print(f"\n❌ Coverage validation failed")
        sys.exit(1)
    elif issues['critical_issues'] > 0:
        print(f"\n❌ Critical coverage issues found")
        sys.exit(1)
    else:
        print(f"\n✅ Coverage validation passed")
        sys.exit(0)


if __name__ == "__main__":
    main()