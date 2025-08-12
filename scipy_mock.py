"""Simple mock implementation for SciPy functionality."""

import sys
import math
import random as python_random
from typing import Any, List, Tuple, Optional, Union

# Import numpy mock if available
try:
    import numpy as np
except ImportError:
    import numpy_mock as np

class MockResult:
    """Mock result object for statistical tests."""
    def __init__(self, statistic=0.0, pvalue=0.05):
        self.statistic = statistic
        self.pvalue = pvalue

class MockStats:
    """Mock scipy.stats module."""
    
    @staticmethod
    def ttest_ind(a, b, axis=0):
        """Mock independent t-test."""
        if hasattr(a, '__iter__') and hasattr(b, '__iter__'):
            return np.array([1.0] * len(a[0]) if hasattr(a[0], '__iter__') else [1.0]), np.array([0.05] * len(a[0]) if hasattr(a[0], '__iter__') else [0.05])
        return MockResult(1.0, 0.05)
    
    @staticmethod
    def ks_2samp(a, b):
        """Mock Kolmogorov-Smirnov test."""
        return MockResult(0.1, 0.05)
    
    @staticmethod
    def chi2_contingency(observed):
        """Mock chi-square test."""
        return 1.0, 0.05, 1, [[1, 1], [1, 1]]
    
    @staticmethod
    def spearmanr(a, b=None):
        """Mock Spearman correlation."""
        return MockResult(0.5, 0.05)
    
    @staticmethod
    def entropy(pk, qk=None, base=None):
        """Mock entropy calculation."""
        if hasattr(pk, '__iter__'):
            return sum(pk) * 0.1  # Simple mock
        return 0.5

class MockSignal:
    """Mock scipy.signal module."""
    
    @staticmethod
    def butter(N, Wn, btype='low', analog=False, output='ba'):
        """Mock Butterworth filter."""
        return [1, 0, 0], [1, 0, 0]  # Simple coefficients
    
    @staticmethod
    def filtfilt(b, a, x, axis=-1):
        """Mock zero-phase filter."""
        if hasattr(x, 'data'):
            return x  # Return input unchanged for mock
        return np.array(x) if hasattr(x, '__iter__') else x
    
    @staticmethod
    def spectrogram(x, fs=1.0, window='hann', nperseg=None, noverlap=None):
        """Mock spectrogram."""
        n = len(x) if hasattr(x, '__len__') else 100
        f = np.linspace(0, fs/2, n//2)
        t = np.linspace(0, n/fs, n//10)
        Sxx = np.random.random((len(f), len(t)))
        return f, t, Sxx
    
    @staticmethod
    def welch(x, fs=1.0, window='hann', nperseg=None, noverlap=None):
        """Mock Welch's method."""
        n = len(x) if hasattr(x, '__len__') else 100
        f = np.linspace(0, fs/2, n//2)
        Pxx = np.random.random(len(f))
        return f, Pxx

class MockSpatial:
    """Mock scipy.spatial module."""
    
    @staticmethod
    def distance_matrix(X, Y=None):
        """Mock distance matrix."""
        n = len(X) if hasattr(X, '__len__') else 3
        m = len(Y) if Y is not None and hasattr(Y, '__len__') else n
        return np.random.random((n, m))

class MockInterpolate:
    """Mock scipy.interpolate module."""
    
    @staticmethod
    def griddata(points, values, xi, method='linear', fill_value=np.nan):
        """Mock grid data interpolation."""
        if hasattr(xi, '__len__'):
            return np.random.random(len(xi))
        return 0.5

class MockFFT:
    """Mock scipy.fft module."""
    
    @staticmethod
    def fft(x, n=None, axis=-1):
        """Mock FFT."""
        if hasattr(x, '__len__'):
            length = n if n is not None else len(x)
            return np.array([complex(python_random.random(), python_random.random()) for _ in range(length)])
        return complex(1, 0)
    
    @staticmethod
    def ifft(x, n=None, axis=-1):
        """Mock inverse FFT."""
        return MockFFT.fft(x, n, axis)
    
    @staticmethod
    def fftfreq(n, d=1.0):
        """Mock FFT frequencies."""
        return np.linspace(-0.5/d, 0.5/d, n)

# Create scipy module structure
class SciPyModule:
    def __init__(self):
        self.stats = MockStats()
        self.signal = MockSignal()
        self.spatial = MockSpatial()
        self.interpolate = MockInterpolate()
        self.fft = MockFFT()

# Register in sys.modules
scipy_mock = SciPyModule()
sys.modules['scipy'] = scipy_mock
sys.modules['scipy.stats'] = MockStats()
sys.modules['scipy.signal'] = MockSignal()
sys.modules['scipy.spatial'] = MockSpatial()
sys.modules['scipy.interpolate'] = MockInterpolate()
sys.modules['scipy.fft'] = MockFFT()