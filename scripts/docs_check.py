#!/usr/bin/env python3
"""Documentation completeness validation for neural cryptanalysis framework."""

import ast
import sys
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import json


class DocumentationChecker(ast.NodeVisitor):
    """AST visitor to check documentation completeness."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.issues: List[Dict[str, Any]] = []
        self.documented_items = 0
        self.total_items = 0
        
        # Documentation requirements
        self.require_class_docstrings = True
        self.require_method_docstrings = True
        self.require_function_docstrings = True
        self.require_module_docstrings = True
        
        # Special method exclusions
        self.exclude_methods = {
            '__init__', '__str__', '__repr__', '__len__', '__getitem__',
            '__setitem__', '__delitem__', '__iter__', '__next__',
            '__enter__', '__exit__', '__call__'
        }
        
        # Documentation quality patterns
        self.docstring_patterns = {
            'args_section': r'Args?:|Arguments?:|Parameters?:',
            'returns_section': r'Returns?:|Return:',
            'raises_section': r'Raises?:|Exceptions?:',
            'examples_section': r'Examples?:|Example:',
            'note_section': r'Notes?:|Note:',
        }
    
    def visit_Module(self, node: ast.Module):
        """Check module-level documentation."""
        if self.require_module_docstrings:
            self.total_items += 1
            
            if ast.get_docstring(node):
                self.documented_items += 1
                docstring = ast.get_docstring(node)
                self._check_docstring_quality(docstring, 'module', 1)
            else:
                self.issues.append({
                    'type': 'missing_docstring',
                    'severity': 'medium',
                    'line': 1,
                    'item_type': 'module',
                    'item_name': self.filename,
                    'message': 'Module missing docstring',
                    'recommendation': 'Add module-level docstring explaining purpose and usage'
                })
        
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Check class documentation."""
        if self.require_class_docstrings:
            self.total_items += 1
            
            if ast.get_docstring(node):
                self.documented_items += 1
                docstring = ast.get_docstring(node)
                self._check_docstring_quality(docstring, 'class', node.lineno, node.name)
            else:
                self.issues.append({
                    'type': 'missing_docstring',
                    'severity': 'high',
                    'line': node.lineno,
                    'item_type': 'class',
                    'item_name': node.name,
                    'message': f'Class {node.name} missing docstring',
                    'recommendation': 'Add class docstring explaining purpose, attributes, and usage'
                })
        
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Check function/method documentation."""
        is_method = self._is_method(node)
        item_type = 'method' if is_method else 'function'
        
        # Check if we should require documentation for this item
        should_document = (
            (is_method and self.require_method_docstrings and node.name not in self.exclude_methods) or
            (not is_method and self.require_function_docstrings)
        )
        
        if should_document:
            self.total_items += 1
            
            if ast.get_docstring(node):
                self.documented_items += 1
                docstring = ast.get_docstring(node)
                self._check_docstring_quality(docstring, item_type, node.lineno, node.name, node)
            else:
                severity = 'high' if not is_method or not node.name.startswith('_') else 'medium'
                self.issues.append({
                    'type': 'missing_docstring',
                    'severity': severity,
                    'line': node.lineno,
                    'item_type': item_type,
                    'item_name': node.name,
                    'message': f'{item_type.title()} {node.name} missing docstring',
                    'recommendation': f'Add {item_type} docstring with description, parameters, and return value'
                })
        
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Check async function documentation."""
        # Treat async functions the same as regular functions
        self.visit_FunctionDef(node)
    
    def _is_method(self, node: ast.FunctionDef) -> bool:
        """Check if function is a method (inside a class)."""
        # Walk up the AST to find if we're inside a class
        for parent in ast.walk(ast.Module(body=[])):  # Simplified check
            pass
        # In practice, we'd need to track the AST context
        # For now, assume functions with 'self' parameter are methods
        if node.args.args and node.args.args[0].arg == 'self':
            return True
        return False
    
    def _check_docstring_quality(self, docstring: str, item_type: str, line_num: int, 
                                item_name: str = '', func_node: ast.FunctionDef = None):
        """Check quality of docstring content."""
        if not docstring:
            return
        
        docstring_lower = docstring.lower()
        
        # Check for basic description
        lines = docstring.strip().split('\n')
        if len(lines) < 1 or len(lines[0].strip()) < 10:
            self.issues.append({
                'type': 'poor_docstring_quality',
                'severity': 'medium',
                'line': line_num,
                'item_type': item_type,
                'item_name': item_name,
                'message': f'{item_type.title()} {item_name} has inadequate description',
                'recommendation': 'Provide a clear, descriptive summary of purpose and functionality'
            })
        
        # For functions/methods, check for parameter documentation
        if func_node and item_type in ['function', 'method']:
            # Check if function has parameters (excluding 'self')
            params = [arg.arg for arg in func_node.args.args]
            if item_type == 'method' and params and params[0] == 'self':
                params = params[1:]  # Remove 'self'
            
            if params:
                # Check for Args/Parameters section
                if not re.search(self.docstring_patterns['args_section'], docstring, re.IGNORECASE):
                    self.issues.append({
                        'type': 'missing_parameter_docs',
                        'severity': 'medium',
                        'line': line_num,
                        'item_type': item_type,
                        'item_name': item_name,
                        'message': f'{item_type.title()} {item_name} missing parameter documentation',
                        'recommendation': 'Add Args/Parameters section documenting each parameter'
                    })
                else:
                    # Check if all parameters are documented
                    for param in params:
                        if param not in docstring:
                            self.issues.append({
                                'type': 'undocumented_parameter',
                                'severity': 'low',
                                'line': line_num,
                                'item_type': item_type,
                                'item_name': item_name,
                                'message': f'Parameter {param} not documented in {item_name}',
                                'recommendation': f'Document parameter {param} in Args section'
                            })
            
            # Check for return documentation (if function returns something)
            if self._function_has_return(func_node):
                if not re.search(self.docstring_patterns['returns_section'], docstring, re.IGNORECASE):
                    self.issues.append({
                        'type': 'missing_return_docs',
                        'severity': 'medium',
                        'line': line_num,
                        'item_type': item_type,
                        'item_name': item_name,
                        'message': f'{item_type.title()} {item_name} missing return value documentation',
                        'recommendation': 'Add Returns section documenting return value'
                    })
            
            # Check for exception documentation (if function raises exceptions)
            if self._function_raises_exceptions(func_node):
                if not re.search(self.docstring_patterns['raises_section'], docstring, re.IGNORECASE):
                    self.issues.append({
                        'type': 'missing_exception_docs',
                        'severity': 'low',
                        'line': line_num,
                        'item_type': item_type,
                        'item_name': item_name,
                        'message': f'{item_type.title()} {item_name} missing exception documentation',
                        'recommendation': 'Add Raises section documenting possible exceptions'
                    })
        
        # Check for examples in public API functions
        if item_type in ['function', 'method'] and item_name and not item_name.startswith('_'):
            if not re.search(self.docstring_patterns['examples_section'], docstring, re.IGNORECASE):
                self.issues.append({
                    'type': 'missing_examples',
                    'severity': 'low',
                    'line': line_num,
                    'item_type': item_type,
                    'item_name': item_name,
                    'message': f'Public {item_type} {item_name} missing usage examples',
                    'recommendation': 'Consider adding Examples section with usage examples'
                })
    
    def _function_has_return(self, func_node: ast.FunctionDef) -> bool:
        """Check if function has return statements."""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Return) and node.value is not None:
                return True
        return False
    
    def _function_raises_exceptions(self, func_node: ast.FunctionDef) -> bool:
        """Check if function raises exceptions."""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Raise):
                return True
        return False
    
    def get_documentation_coverage(self) -> float:
        """Calculate documentation coverage percentage."""
        if self.total_items == 0:
            return 100.0
        return (self.documented_items / self.total_items) * 100.0


def check_external_documentation(repo_path: Path) -> List[Dict[str, Any]]:
    """Check for external documentation files."""
    issues = []
    
    # Required documentation files
    required_docs = {
        'README.md': 'Project overview and basic usage',
        'SECURITY.md': 'Security policies and reporting',
        'CONTRIBUTING.md': 'Contribution guidelines',
        'LICENSE': 'License information'
    }
    
    # Check for required documentation
    for doc_file, description in required_docs.items():
        doc_path = repo_path / doc_file
        if not doc_path.exists():
            issues.append({
                'type': 'missing_external_doc',
                'severity': 'medium',
                'line': 0,
                'item_type': 'documentation',
                'item_name': doc_file,
                'message': f'Missing {doc_file}',
                'recommendation': f'Create {doc_file} with {description}'
            })
        else:
            # Check file content quality
            try:
                content = doc_path.read_text()
                if len(content.strip()) < 100:  # Very short content
                    issues.append({
                        'type': 'inadequate_external_doc',
                        'severity': 'low',
                        'line': 0,
                        'item_type': 'documentation',
                        'item_name': doc_file,
                        'message': f'{doc_file} appears to have insufficient content',
                        'recommendation': f'Expand {doc_file} with more detailed information'
                    })
            except Exception:
                pass
    
    # Check for API documentation
    docs_dir = repo_path / 'docs'
    if docs_dir.exists():
        api_docs = list(docs_dir.glob('**/*.md'))
        if len(api_docs) == 0:
            issues.append({
                'type': 'missing_api_docs',
                'severity': 'medium',
                'line': 0,
                'item_type': 'documentation',
                'item_name': 'API documentation',
                'message': 'No API documentation found in docs directory',
                'recommendation': 'Create API documentation for main modules and classes'
            })
    else:
        issues.append({
            'type': 'missing_docs_directory',
            'severity': 'low',
            'line': 0,
            'item_type': 'documentation',
            'item_name': 'docs/',
            'message': 'No docs directory found',
            'recommendation': 'Consider creating docs directory for detailed documentation'
        })
    
    return issues


def check_code_comments(file_path: Path) -> List[Dict[str, Any]]:
    """Check for adequate code comments."""
    issues = []
    
    try:
        content = file_path.read_text()
        lines = content.split('\n')
        
        code_lines = 0
        comment_lines = 0
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Skip empty lines and docstrings
            if not stripped or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            
            if stripped.startswith('#'):
                comment_lines += 1
            elif not stripped.startswith('#') and len(stripped) > 0:
                code_lines += 1
        
        # Calculate comment ratio
        if code_lines > 0:
            comment_ratio = comment_lines / code_lines
            
            # Check for adequate commenting
            if comment_ratio < 0.1 and code_lines > 20:  # Less than 10% comments for files with >20 lines
                issues.append({
                    'type': 'insufficient_comments',
                    'severity': 'low',
                    'line': 0,
                    'item_type': 'code_comments',
                    'item_name': str(file_path),
                    'message': f'Low comment ratio: {comment_ratio:.1%}',
                    'recommendation': 'Consider adding more explanatory comments for complex logic'
                })
    
    except Exception:
        pass
    
    return issues


def check_file_documentation(file_path: Path) -> Dict[str, Any]:
    """Check documentation for a single file."""
    try:
        content = file_path.read_text()
        checker = DocumentationChecker(str(file_path))
        
        tree = ast.parse(content, filename=str(file_path))
        checker.visit(tree)
        
        # Add comment analysis
        comment_issues = check_code_comments(file_path)
        checker.issues.extend(comment_issues)
        
        return {
            'file': str(file_path),
            'documentation_coverage': checker.get_documentation_coverage(),
            'total_items': checker.total_items,
            'documented_items': checker.documented_items,
            'issues': checker.issues
        }
    
    except SyntaxError:
        return {
            'file': str(file_path),
            'documentation_coverage': 0.0,
            'total_items': 0,
            'documented_items': 0,
            'issues': [{
                'type': 'syntax_error',
                'severity': 'critical',
                'line': 0,
                'item_type': 'file',
                'item_name': str(file_path),
                'message': 'Syntax error prevents documentation analysis',
                'recommendation': 'Fix syntax errors before documentation analysis'
            }]
        }
    
    except Exception as e:
        return {
            'file': str(file_path),
            'documentation_coverage': 0.0,
            'total_items': 0,
            'documented_items': 0,
            'issues': [{
                'type': 'analysis_error',
                'severity': 'medium',
                'line': 0,
                'item_type': 'file',
                'item_name': str(file_path),
                'message': f'Documentation analysis failed: {e}',
                'recommendation': 'Check file accessibility and encoding'
            }]
        }


def main():
    """Main documentation checking function."""
    if len(sys.argv) < 2:
        print("Usage: docs_check.py <file1> [file2] ...")
        sys.exit(1)
    
    all_results = []
    all_issues = []
    
    # Check individual files
    for file_path in sys.argv[1:]:
        path = Path(file_path)
        if not path.exists() or path.suffix != '.py':
            continue
        
        result = check_file_documentation(path)
        all_results.append(result)
        all_issues.extend(result['issues'])
    
    # Check external documentation
    repo_path = Path.cwd()
    external_issues = check_external_documentation(repo_path)
    all_issues.extend(external_issues)
    
    # Calculate overall statistics
    total_items = sum(r['total_items'] for r in all_results)
    documented_items = sum(r['documented_items'] for r in all_results)
    overall_coverage = (documented_items / total_items * 100) if total_items > 0 else 100.0
    
    # Categorize issues
    critical_issues = [i for i in all_issues if i['severity'] == 'critical']
    high_issues = [i for i in all_issues if i['severity'] == 'high']
    medium_issues = [i for i in all_issues if i['severity'] == 'medium']
    low_issues = [i for i in all_issues if i['severity'] == 'low']
    
    # Generate report
    report = {
        'overall_coverage': overall_coverage,
        'total_items': total_items,
        'documented_items': documented_items,
        'files_analyzed': len(all_results),
        'issue_summary': {
            'total': len(all_issues),
            'critical': len(critical_issues),
            'high': len(high_issues),
            'medium': len(medium_issues),
            'low': len(low_issues)
        },
        'file_results': all_results,
        'external_documentation': {
            'issues': external_issues
        },
        'all_issues': all_issues
    }
    
    # Save report
    with open('documentation_check_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print results
    print(f"\nDocumentation Check Results:")
    print(f"  Overall Coverage: {overall_coverage:.1f}%")
    print(f"  Items Documented: {documented_items}/{total_items}")
    print(f"  Files Analyzed: {len(all_results)}")
    
    if all_issues:
        print(f"\nIssues Found: {len(all_issues)}")
        if critical_issues:
            print(f"  Critical: {len(critical_issues)}")
        if high_issues:
            print(f"  High: {len(high_issues)}")
        if medium_issues:
            print(f"  Medium: {len(medium_issues)}")
        if low_issues:
            print(f"  Low: {len(low_issues)}")
    
    # Print some example issues
    if critical_issues:
        print(f"\nCritical Issues:")
        for issue in critical_issues[:3]:
            print(f"  • {issue['message']}")
        if len(critical_issues) > 3:
            print(f"  ... and {len(critical_issues) - 3} more")
    
    if high_issues:
        print(f"\nHigh Priority Issues:")
        for issue in high_issues[:3]:
            print(f"  • {issue['message']}")
        if len(high_issues) > 3:
            print(f"  ... and {len(high_issues) - 3} more")
    
    # Exit based on results
    min_coverage = 70.0  # Minimum documentation coverage
    
    if critical_issues:
        print(f"\n❌ Documentation check failed: Critical issues found")
        sys.exit(1)
    elif overall_coverage < min_coverage:
        print(f"\n❌ Documentation check failed: Coverage {overall_coverage:.1f}% below minimum {min_coverage:.1f}%")
        sys.exit(1)
    elif len(high_issues) > 10:  # Too many high priority issues
        print(f"\n❌ Documentation check failed: Too many high priority issues ({len(high_issues)})")
        sys.exit(1)
    else:
        print(f"\n✅ Documentation check passed")
        sys.exit(0)


if __name__ == "__main__":
    main()