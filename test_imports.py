#!/usr/bin/env python3
"""Test imports and basic structure."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that our modules can be imported."""
    
    try:
        print("Testing neural_cryptanalysis package import...")
        from neural_cryptanalysis import __version__, __author__
        print(f"✓ Package imported successfully")
        print(f"  Version: {__version__}")
        print(f"  Author: {__author__}")
    except Exception as e:
        print(f"✗ Package import failed: {e}")
        return False
    
    try:
        print("\nTesting neural operator imports...")
        from neural_cryptanalysis.neural_operators.base import NeuralOperatorBase, OperatorConfig
        print("✓ Neural operator base classes imported")
    except Exception as e:
        print(f"✗ Neural operator import failed: {e}")
        return False
    
    try:
        print("\nTesting side-channel imports...")
        from neural_cryptanalysis.side_channels.base import SideChannelAnalyzer, TraceData
        print("✓ Side-channel classes imported")
    except Exception as e:
        print(f"✗ Side-channel import failed: {e}")
        return False
    
    try:
        print("\nTesting target imports...")
        from neural_cryptanalysis.targets.base import CryptographicTarget, ImplementationConfig
        print("✓ Target classes imported")
    except Exception as e:
        print(f"✗ Target import failed: {e}")
        return False
    
    return True

def test_class_instantiation():
    """Test that classes can be instantiated with minimal dependencies."""
    
    try:
        print("\nTesting OperatorConfig...")
        from neural_cryptanalysis.neural_operators.base import OperatorConfig
        config = OperatorConfig()
        print(f"✓ OperatorConfig created: input_dim={config.input_dim}, output_dim={config.output_dim}")
    except Exception as e:
        print(f"✗ OperatorConfig failed: {e}")
        return False
    
    try:
        print("\nTesting ImplementationConfig...")
        from neural_cryptanalysis.targets.base import ImplementationConfig
        config = ImplementationConfig(algorithm='kyber', variant='kyber768')
        print(f"✓ ImplementationConfig created: {config.algorithm}-{config.variant}")
    except Exception as e:
        print(f"✗ ImplementationConfig failed: {e}")
        return False
    
    return True

def main():
    """Run import tests."""
    print("=" * 60)
    print("Neural Operator Cryptanalysis Lab - Import Test")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 2
    
    if test_imports():
        tests_passed += 1
    
    if test_class_instantiation():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"Import Test Results: {tests_passed}/{total_tests} tests passed")
    print("=" * 60)
    
    if tests_passed == total_tests:
        print("🎉 All import tests passed! The basic structure is working.")
        return True
    else:
        print("⚠️  Some import tests failed.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)