#!/usr/bin/env python3
"""Validate neural cryptanalysis imports for security and compatibility."""

import ast
import sys
from pathlib import Path
from typing import List, Set, Dict, Any
import json


class ImportValidator(ast.NodeVisitor):
    """AST visitor to validate imports in neural cryptanalysis code."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.issues: List[Dict[str, Any]] = []
        self.imports: Set[str] = set()
        self.from_imports: Dict[str, Set[str]] = {}
        
        # Define security-sensitive modules
        self.security_sensitive = {
            'subprocess', 'os', 'sys', 'eval', 'exec', 'pickle', 
            'shelve', 'marshal', 'tempfile', 'shutil'
        }
        
        # Define allowed external dependencies
        self.allowed_external = {
            'numpy', 'torch', 'scipy', 'matplotlib', 'seaborn',
            'pytest', 'unittest', 'pathlib', 'typing', 'dataclasses',
            'json', 'yaml', 'logging', 'time', 'datetime', 'hashlib',
            'hmac', 'secrets', 'random', 'collections', 'itertools',
            'functools', 'operator', 'warnings', 'contextlib',
            'asyncio', 'threading', 'multiprocessing', 'concurrent',
            'psutil', 'pkg_resources', 'statsmodels'
        }
        
        # Define neural cryptanalysis internal modules
        self.internal_modules = {
            'neural_cryptanalysis',
            'neural_cryptanalysis.core',
            'neural_cryptanalysis.neural_operators',
            'neural_cryptanalysis.side_channels',
            'neural_cryptanalysis.targets',
            'neural_cryptanalysis.datasets',
            'neural_cryptanalysis.utils',
            'neural_cryptanalysis.adaptive_rl',
            'neural_cryptanalysis.multi_modal_fusion',
            'neural_cryptanalysis.hardware_integration',
            'neural_cryptanalysis.advanced_countermeasures',
            'neural_cryptanalysis.distributed_computing',
            'neural_cryptanalysis.optimization',
            'neural_cryptanalysis.security',
            'neural_cryptanalysis.visualization'
        }
    
    def visit_Import(self, node: ast.Import):
        """Visit import statements."""
        for alias in node.names:
            module_name = alias.name
            self.imports.add(module_name)
            self._validate_import(module_name, node.lineno)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Visit from ... import statements."""
        if node.module:
            module_name = node.module
            if module_name not in self.from_imports:
                self.from_imports[module_name] = set()
            
            for alias in node.names:
                imported_name = alias.name
                self.from_imports[module_name].add(imported_name)
            
            self._validate_import(module_name, node.lineno)
        self.generic_visit(node)
    
    def _validate_import(self, module_name: str, line_number: int):
        """Validate a single import."""
        # Check for security-sensitive imports
        if any(sensitive in module_name for sensitive in self.security_sensitive):
            if not self._is_test_file():
                self.issues.append({
                    'type': 'security',
                    'severity': 'high',
                    'line': line_number,
                    'message': f'Security-sensitive import: {module_name}',
                    'recommendation': 'Use secure alternatives or add security validation'
                })
        
        # Check for prohibited dangerous imports
        dangerous_patterns = ['eval', 'exec', 'compile', '__import__']
        if any(pattern in module_name for pattern in dangerous_patterns):
            self.issues.append({
                'type': 'security',
                'severity': 'critical',
                'line': line_number,
                'message': f'Dangerous import pattern: {module_name}',
                'recommendation': 'Remove dangerous import or use safe alternatives'
            })
        
        # Check for external dependencies not in allowed list
        if self._is_external_module(module_name):
            top_level = module_name.split('.')[0]
            if top_level not in self.allowed_external:
                self.issues.append({
                    'type': 'dependency',
                    'severity': 'medium',
                    'line': line_number,
                    'message': f'Unregistered external dependency: {module_name}',
                    'recommendation': 'Add to allowed dependencies or remove if unnecessary'
                })
        
        # Check for relative imports in non-package files
        if module_name.startswith('.') and not self._is_package_file():
            self.issues.append({
                'type': 'structure',
                'severity': 'medium',
                'line': line_number,
                'message': f'Relative import in non-package file: {module_name}',
                'recommendation': 'Use absolute imports or ensure proper package structure'
            })
    
    def _is_external_module(self, module_name: str) -> bool:
        """Check if module is external (not stdlib or internal)."""
        # Standard library modules (simplified check)
        stdlib_modules = {
            'sys', 'os', 'time', 'datetime', 'json', 'logging', 'pathlib',
            'typing', 'collections', 'itertools', 'functools', 'operator',
            'hashlib', 'hmac', 'secrets', 'random', 'unittest', 'threading',
            'multiprocessing', 'asyncio', 'contextlib', 'warnings', 'tempfile'
        }
        
        top_level = module_name.split('.')[0]
        
        # Not stdlib and not internal
        return (top_level not in stdlib_modules and 
                not any(module_name.startswith(internal) for internal in self.internal_modules))
    
    def _is_test_file(self) -> bool:
        """Check if current file is a test file."""
        return 'test_' in self.filename or '/tests/' in self.filename
    
    def _is_package_file(self) -> bool:
        """Check if current file is part of a package."""
        return '/src/' in self.filename or '__init__.py' in self.filename
    
    def validate_file(self, content: str) -> List[Dict[str, Any]]:
        """Validate imports in file content."""
        try:
            tree = ast.parse(content, filename=self.filename)
            self.visit(tree)
        except SyntaxError as e:
            self.issues.append({
                'type': 'syntax',
                'severity': 'critical',
                'line': e.lineno or 0,
                'message': f'Syntax error: {e.msg}',
                'recommendation': 'Fix syntax error before proceeding'
            })
        
        return self.issues


def validate_imports_in_file(file_path: Path) -> List[Dict[str, Any]]:
    """Validate imports in a single file."""
    try:
        content = file_path.read_text(encoding='utf-8')
        validator = ImportValidator(str(file_path))
        return validator.validate_file(content)
    except Exception as e:
        return [{
            'type': 'error',
            'severity': 'high',
            'line': 0,
            'message': f'Failed to validate file: {e}',
            'recommendation': 'Check file encoding and accessibility'
        }]


def main():
    """Main validation function."""
    if len(sys.argv) < 2:
        print("Usage: validate_imports.py <file1> [file2] ...")
        sys.exit(1)
    
    all_issues = []
    
    for file_path in sys.argv[1:]:
        path = Path(file_path)
        if not path.exists():
            continue
        
        if path.suffix != '.py':
            continue
        
        issues = validate_imports_in_file(path)
        
        for issue in issues:
            issue['file'] = str(path)
            all_issues.append(issue)
    
    # Report issues
    critical_issues = [i for i in all_issues if i['severity'] == 'critical']
    high_issues = [i for i in all_issues if i['severity'] == 'high']
    medium_issues = [i for i in all_issues if i['severity'] == 'medium']
    
    if critical_issues:
        print("CRITICAL IMPORT ISSUES:")
        for issue in critical_issues:
            print(f"  {issue['file']}:{issue['line']} - {issue['message']}")
        print()
    
    if high_issues:
        print("HIGH PRIORITY IMPORT ISSUES:")
        for issue in high_issues:
            print(f"  {issue['file']}:{issue['line']} - {issue['message']}")
        print()
    
    if medium_issues:
        print("MEDIUM PRIORITY IMPORT ISSUES:")
        for issue in medium_issues:
            print(f"  {issue['file']}:{issue['line']} - {issue['message']}")
        print()
    
    # Save detailed report
    report = {
        'total_issues': len(all_issues),
        'critical_issues': len(critical_issues),
        'high_issues': len(high_issues),
        'medium_issues': len(medium_issues),
        'issues': all_issues
    }
    
    with open('import_validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Exit with error if critical issues found
    if critical_issues:
        print(f"❌ Import validation failed: {len(critical_issues)} critical issues")
        sys.exit(1)
    elif high_issues:
        print(f"⚠️  Import validation warning: {len(high_issues)} high priority issues")
        sys.exit(0)  # Don't fail build for warnings
    else:
        print("✅ Import validation passed")
        sys.exit(0)


if __name__ == "__main__":
    main()