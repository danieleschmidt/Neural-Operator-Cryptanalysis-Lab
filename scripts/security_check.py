#!/usr/bin/env python3
"""Security validation script for neural cryptanalysis framework."""

import ast
import sys
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Set
import json


class SecurityChecker(ast.NodeVisitor):
    """AST visitor for security vulnerability detection."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.issues: List[Dict[str, Any]] = []
        
        # Define security patterns
        self.dangerous_functions = {
            'eval', 'exec', 'compile', '__import__', 'globals', 'locals',
            'getattr', 'setattr', 'delattr', 'hasattr'
        }
        
        self.shell_commands = {
            'os.system', 'os.popen', 'subprocess.call', 'subprocess.run',
            'subprocess.Popen', 'subprocess.check_call', 'subprocess.check_output'
        }
        
        self.file_operations = {
            'open', 'file', 'input', 'raw_input'
        }
        
        # Cryptographic requirements
        self.crypto_patterns = [
            r'random\.random\(',  # Insecure random
            r'random\.choice\(',  # Insecure random
            r'random\.randint\(',  # Insecure random
            r'md5\(',  # Weak hash
            r'sha1\(',  # Weak hash
        ]
        
        # Hardcoded secrets patterns
        self.secret_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'key\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
        ]
    
    def visit_Call(self, node: ast.Call):
        """Visit function calls for security issues."""
        func_name = self._get_function_name(node.func)
        
        # Check for dangerous functions
        if func_name in self.dangerous_functions:
            self.issues.append({
                'type': 'dangerous_function',
                'severity': 'critical',
                'line': node.lineno,
                'message': f'Dangerous function call: {func_name}',
                'recommendation': 'Use safer alternatives or add security validation'
            })
        
        # Check for shell command execution
        if func_name in self.shell_commands:
            self.issues.append({
                'type': 'shell_execution',
                'severity': 'high',
                'line': node.lineno,
                'message': f'Shell command execution: {func_name}',
                'recommendation': 'Use safer subprocess methods with proper input validation'
            })
        
        # Check for file operations with user input
        if func_name in self.file_operations:
            if self._has_user_input(node):
                self.issues.append({
                    'type': 'file_operation',
                    'severity': 'high',
                    'line': node.lineno,
                    'message': f'File operation with potential user input: {func_name}',
                    'recommendation': 'Validate and sanitize file paths'
                })
        
        # Check for pickle usage (deserialization vulnerability)
        if 'pickle' in func_name:
            self.issues.append({
                'type': 'deserialization',
                'severity': 'high',
                'line': node.lineno,
                'message': f'Pickle usage detected: {func_name}',
                'recommendation': 'Use safer serialization methods like JSON'
            })
        
        self.generic_visit(node)
    
    def visit_Assign(self, node: ast.Assign):
        """Visit assignments for hardcoded secrets."""
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            value = node.value.value
            
            # Check for hardcoded secrets
            for target in node.targets:
                if isinstance(target, ast.Name):
                    var_name = target.id.lower()
                    
                    # Check for suspicious variable names with string values
                    if any(secret in var_name for secret in ['password', 'secret', 'key', 'token']):
                        if len(value) > 8 and not self._is_test_value(value):
                            self.issues.append({
                                'type': 'hardcoded_secret',
                                'severity': 'critical',
                                'line': node.lineno,
                                'message': f'Potential hardcoded secret: {var_name}',
                                'recommendation': 'Use environment variables or secure configuration'
                            })
        
        self.generic_visit(node)
    
    def visit_Str(self, node: ast.Str):
        """Visit string literals for secrets and patterns."""
        if hasattr(node, 's'):  # Python < 3.8
            self._check_string_content(node.s, node.lineno)
        self.generic_visit(node)
    
    def visit_Constant(self, node: ast.Constant):
        """Visit constants for secrets and patterns."""
        if isinstance(node.value, str):
            self._check_string_content(node.value, node.lineno)
        self.generic_visit(node)
    
    def _get_function_name(self, node) -> str:
        """Extract function name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                return f"{node.value.id}.{node.attr}"
            elif isinstance(node.value, ast.Attribute):
                return f"{self._get_function_name(node.value)}.{node.attr}"
            else:
                return node.attr
        else:
            return ""
    
    def _has_user_input(self, node: ast.Call) -> bool:
        """Check if function call involves user input."""
        # Simplified check for user input patterns
        for arg in node.args:
            if isinstance(arg, ast.Call):
                func_name = self._get_function_name(arg.func)
                if func_name in ['input', 'raw_input', 'sys.argv']:
                    return True
        return False
    
    def _is_test_value(self, value: str) -> bool:
        """Check if string is likely a test value."""
        test_indicators = [
            'test', 'example', 'dummy', 'fake', 'mock', 'sample',
            '123', 'abc', 'xxx', 'placeholder'
        ]
        return any(indicator in value.lower() for indicator in test_indicators)
    
    def _check_string_content(self, content: str, line_number: int):
        """Check string content for security patterns."""
        # Check for crypto patterns
        for pattern in self.crypto_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                self.issues.append({
                    'type': 'weak_crypto',
                    'severity': 'medium',
                    'line': line_number,
                    'message': f'Weak cryptographic pattern detected',
                    'recommendation': 'Use cryptographically secure functions'
                })
        
        # Check for potential secrets in strings
        for pattern in self.secret_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                self.issues.append({
                    'type': 'potential_secret',
                    'severity': 'high',
                    'line': line_number,
                    'message': 'Potential secret in string literal',
                    'recommendation': 'Use environment variables or secure configuration'
                })
        
        # Check for SQL injection patterns
        sql_patterns = [
            r'SELECT.*FROM.*WHERE.*=.*\+',
            r'INSERT.*INTO.*VALUES.*\+',
            r'UPDATE.*SET.*WHERE.*\+',
            r'DELETE.*FROM.*WHERE.*\+'
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                self.issues.append({
                    'type': 'sql_injection',
                    'severity': 'critical',
                    'line': line_number,
                    'message': 'Potential SQL injection vulnerability',
                    'recommendation': 'Use parameterized queries'
                })


def check_file_permissions(file_path: Path) -> List[Dict[str, Any]]:
    """Check file permissions for security issues."""
    issues = []
    
    try:
        # Check if file is world-writable (Unix-like systems)
        import stat
        file_stat = file_path.stat()
        
        if file_stat.st_mode & stat.S_IWOTH:
            issues.append({
                'type': 'file_permissions',
                'severity': 'high',
                'line': 0,
                'message': 'File is world-writable',
                'recommendation': 'Restrict file permissions'
            })
        
        if file_stat.st_mode & stat.S_IROTH and 'secret' in str(file_path).lower():
            issues.append({
                'type': 'file_permissions',
                'severity': 'medium',
                'line': 0,
                'message': 'Potentially sensitive file is world-readable',
                'recommendation': 'Restrict access to sensitive files'
            })
    
    except Exception:
        pass  # Permission checks may not work on all systems
    
    return issues


def check_file_content_security(file_path: Path) -> List[Dict[str, Any]]:
    """Check file content for security issues."""
    issues = []
    
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # Check for common security anti-patterns in comments
        security_todos = [
            r'TODO.*security',
            r'FIXME.*security',
            r'HACK.*security',
            r'XXX.*security'
        ]
        
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern in security_todos:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append({
                        'type': 'security_todo',
                        'severity': 'medium',
                        'line': line_num,
                        'message': 'Security-related TODO/FIXME found',
                        'recommendation': 'Address security concerns before deployment'
                    })
        
        # Check for debug statements that might leak information
        debug_patterns = [
            r'print\s*\([^)]*password',
            r'print\s*\([^)]*secret',
            r'print\s*\([^)]*key',
            r'logging\.debug\([^)]*password',
            r'console\.log\([^)]*password'
        ]
        
        for line_num, line in enumerate(lines, 1):
            for pattern in debug_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append({
                        'type': 'information_leak',
                        'severity': 'high',
                        'line': line_num,
                        'message': 'Debug statement may leak sensitive information',
                        'recommendation': 'Remove or secure debug statements'
                    })
    
    except Exception as e:
        issues.append({
            'type': 'scan_error',
            'severity': 'low',
            'line': 0,
            'message': f'Could not scan file content: {e}',
            'recommendation': 'Ensure file is accessible and properly encoded'
        })
    
    return issues


def security_check_file(file_path: Path) -> List[Dict[str, Any]]:
    """Perform comprehensive security check on a file."""
    all_issues = []
    
    # AST-based security check
    try:
        content = file_path.read_text(encoding='utf-8')
        checker = SecurityChecker(str(file_path))
        tree = ast.parse(content, filename=str(file_path))
        checker.visit(tree)
        all_issues.extend(checker.issues)
    except SyntaxError:
        # Skip files with syntax errors (will be caught by other tools)
        pass
    except Exception as e:
        all_issues.append({
            'type': 'scan_error',
            'severity': 'medium',
            'line': 0,
            'message': f'AST parsing failed: {e}',
            'recommendation': 'Check file syntax and encoding'
        })
    
    # File permission check
    all_issues.extend(check_file_permissions(file_path))
    
    # Content-based security check
    all_issues.extend(check_file_content_security(file_path))
    
    return all_issues


def main():
    """Main security checking function."""
    if len(sys.argv) < 2:
        print("Usage: security_check.py <file1> [file2] ...")
        sys.exit(1)
    
    all_issues = []
    
    for file_path in sys.argv[1:]:
        path = Path(file_path)
        if not path.exists():
            continue
        
        if path.suffix != '.py':
            continue
        
        issues = security_check_file(path)
        
        for issue in issues:
            issue['file'] = str(path)
            all_issues.append(issue)
    
    # Categorize issues
    critical_issues = [i for i in all_issues if i['severity'] == 'critical']
    high_issues = [i for i in all_issues if i['severity'] == 'high']
    medium_issues = [i for i in all_issues if i['severity'] == 'medium']
    
    # Report issues
    if critical_issues:
        print("CRITICAL SECURITY ISSUES:")
        for issue in critical_issues:
            print(f"  {issue['file']}:{issue['line']} - {issue['message']}")
        print()
    
    if high_issues:
        print("HIGH PRIORITY SECURITY ISSUES:")
        for issue in high_issues:
            print(f"  {issue['file']}:{issue['line']} - {issue['message']}")
        print()
    
    if medium_issues:
        print("MEDIUM PRIORITY SECURITY ISSUES:")
        for issue in medium_issues:
            print(f"  {issue['file']}:{issue['line']} - {issue['message']}")
        print()
    
    # Generate security report
    report = {
        'timestamp': str(Path.cwd()),
        'total_issues': len(all_issues),
        'critical_issues': len(critical_issues),
        'high_issues': len(high_issues),
        'medium_issues': len(medium_issues),
        'security_score': max(0, 100 - (len(critical_issues) * 25 + len(high_issues) * 10 + len(medium_issues) * 5)),
        'issues': all_issues
    }
    
    with open('security_check_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Exit codes
    if critical_issues:
        print(f"❌ Security check failed: {len(critical_issues)} critical issues")
        sys.exit(1)
    elif high_issues and len(high_issues) > 5:  # Allow few high issues
        print(f"❌ Security check failed: too many high priority issues ({len(high_issues)})")
        sys.exit(1)
    elif high_issues:
        print(f"⚠️  Security check warning: {len(high_issues)} high priority issues")
        sys.exit(0)
    else:
        print("✅ Security check passed")
        sys.exit(0)


if __name__ == "__main__":
    main()