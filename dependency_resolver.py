#!/usr/bin/env python3
"""
Dependency Resolver for Neural Operator Cryptanalysis Lab
Provides lightweight mock implementations when ML libraries are not available.
"""

import sys
import os
from pathlib import Path

def setup_mock_dependencies():
    """Set up mock dependencies for testing without ML libraries."""
    project_root = Path(__file__).parent
    
    # Mock modules mapping
    mock_modules = {
        'numpy': 'numpy_mock.py',
        'torch': 'simple_torch_mock.py', 
        'scipy': 'scipy_mock.py',
        'sklearn': 'sklearn_mock.py',
        'yaml': 'yaml_mock.py',
        'typing_extensions': 'simple_typing_mock.py'
    }
    
    # Install mock modules
    for module_name, mock_file in mock_modules.items():
        mock_path = project_root / mock_file
        if mock_path.exists():
            # Load mock module
            spec = __import__('importlib.util').util.spec_from_file_location(module_name, mock_path)
            if spec and spec.loader:
                module = __import__('importlib.util').util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sys.modules[module_name] = module
                print(f"✅ Mock {module_name} installed")
    
    # Add project paths
    sys.path.insert(0, str(project_root / 'src'))
    
    print("🔧 Mock dependencies configured successfully")

if __name__ == "__main__":
    setup_mock_dependencies()