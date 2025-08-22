"""Simple mock implementation for NumPy functionality."""

import sys
import math
import random as python_random
from typing import Any, Union, List, Tuple, Optional

class MockArray:
    """Mock numpy array class."""
    
    def __init__(self, data, dtype=None):
        if isinstance(data, (list, tuple)):
            self.data = list(data)
            self.shape = self._compute_shape(data)
        elif isinstance(data, MockArray):
            self.data = data.data
            self.shape = data.shape
        elif isinstance(data, (int, float)):
            self.data = [data]
            self.shape = ()
        else:
            self.data = list(data) if hasattr(data, '__iter__') else [data]
            self.shape = (len(self.data),)
        
        self.dtype = dtype or 'float64'
        self.ndim = len(self.shape) if isinstance(self.shape, tuple) else 0
        
    def _compute_shape(self, data):
        """Compute shape of nested list/tuple."""
        if not isinstance(data, (list, tuple)):
            return ()
        if len(data) == 0:
            return (0,)
        
        shape = [len(data)]
        if isinstance(data[0], (list, tuple)):
            inner_shape = self._compute_shape(data[0])
            if isinstance(inner_shape, tuple):
                shape.extend(inner_shape)
        return tuple(shape)
    
    def __len__(self):
        # For multi-dimensional arrays, return the first dimension
        if hasattr(self, 'shape') and len(self.shape) > 0:
            return self.shape[0]
        return len(self.data) if self.data else 0
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            return MockArray(self.data[key])
        elif isinstance(key, int):
            if len(self.shape) > 1:
                # Multi-dimensional indexing
                return MockArray(self.data[key])
            return self.data[key]
        elif isinstance(key, MockArray):
            # Handle numpy array indexing
            indices = key.data if hasattr(key, 'data') else key
            if isinstance(indices, list):
                try:
                    result_data = [self.data[int(i)] for i in indices]
                    return MockArray(result_data)
                except (TypeError, ValueError):
                    # Fallback to direct indexing
                    return MockArray(self.data)
            return MockArray(self.data)
        elif isinstance(key, tuple):
            # Handle multi-dimensional indexing
            result = self.data
            for k in key:
                if isinstance(result, list) and isinstance(k, int):
                    result = result[k]
                elif isinstance(result, list) and isinstance(k, slice):
                    result = result[k]
            return MockArray(result) if isinstance(result, list) else result
        elif isinstance(key, list):
            # Handle list indexing
            try:
                result_data = [self.data[int(i)] for i in key]
                return MockArray(result_data)
            except (TypeError, ValueError):
                return MockArray(self.data)
        return MockArray(self.data[key])
    
    def __setitem__(self, key, value):
        if isinstance(key, slice):
            if isinstance(value, MockArray):
                self.data[key] = value.data
            else:
                self.data[key] = [value] * len(range(*key.indices(len(self.data))))
        else:
            self.data[key] = value
    
    def astype(self, dtype):
        """Convert array to different dtype."""
        new_data = self.data.copy()
        if dtype in ['float32', 'float64']:
            new_data = [float(x) for x in self.data]
        elif dtype in ['int32', 'int64']:
            new_data = [int(x) for x in self.data]
        elif dtype == 'uint8':
            new_data = [int(x) % 256 for x in self.data]
        
        result = MockArray(new_data, dtype)
        result.shape = self.shape
        return result
    
    def copy(self):
        """Create a copy of the array."""
        result = MockArray(self.data.copy(), self.dtype)
        result.shape = self.shape
        return result
    
    def flatten(self):
        """Flatten the array."""
        def _flatten(data):
            result = []
            for item in data:
                if isinstance(item, list):
                    result.extend(_flatten(item))
                else:
                    result.append(item)
            return result
        
        return MockArray(_flatten(self.data))
    
    def reshape(self, *shape):
        """Reshape the array."""
        # Handle tuple input
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        
        flat_data = self.flatten().data
        total_size = 1
        for dim in shape:
            total_size *= dim
        
        # If sizes don't match, just create new data of the right size
        if len(flat_data) != total_size:
            # Create new data to match the requested shape
            if len(flat_data) > total_size:
                flat_data = flat_data[:total_size]
            else:
                # Repeat data to fill the shape
                repeat_factor = (total_size + len(flat_data) - 1) // len(flat_data)
                flat_data = (flat_data * repeat_factor)[:total_size]
        
        result = MockArray(flat_data)
        result.shape = shape
        return result
    
    def ravel(self):
        """Return flattened array."""
        return self.flatten()
    
    def sum(self, axis=None):
        """Sum array elements."""
        if axis is None:
            return sum(self.flatten().data)
        return MockArray([sum(self.data)])  # Simplified
    
    def mean(self, axis=None):
        """Mean of array elements."""
        flat = self.flatten()
        if len(flat.data) == 0:
            return 0
        return sum(flat.data) / len(flat.data)
    
    def std(self, axis=None):
        """Standard deviation."""
        flat = self.flatten()
        if len(flat.data) <= 1:
            return 0
        mean_val = flat.mean()
        variance = sum((x - mean_val) ** 2 for x in flat.data) / len(flat.data)
        return math.sqrt(variance)
    
    def var(self, axis=None):
        """Variance."""
        flat = self.flatten()
        if len(flat.data) <= 1:
            return 0
        mean_val = flat.mean()
        return sum((x - mean_val) ** 2 for x in flat.data) / len(flat.data)
    
    def max(self, axis=None):
        """Maximum value."""
        if axis is None:
            return max(self.flatten().data) if len(self.flatten().data) > 0 else 0
        return MockArray([max(self.data)])
    
    def min(self, axis=None):
        """Minimum value."""
        if axis is None:
            return min(self.flatten().data) if len(self.flatten().data) > 0 else 0
        return MockArray([min(self.data)])
    
    def __add__(self, other):
        if isinstance(other, MockArray):
            return MockArray([a + b for a, b in zip(self.data, other.data)])
        return MockArray([x + other for x in self.data])
    
    def __sub__(self, other):
        if isinstance(other, MockArray):
            return MockArray([a - b for a, b in zip(self.data, other.data)])
        return MockArray([x - other for x in self.data])
    
    def __mul__(self, other):
        if isinstance(other, MockArray):
            return MockArray([a * b for a, b in zip(self.data, other.data)])
        return MockArray([x * other for x in self.data])
    
    def __truediv__(self, other):
        if isinstance(other, MockArray):
            return MockArray([a / b if b != 0 else 0 for a, b in zip(self.data, other.data)])
        return MockArray([x / other if other != 0 else 0 for x in self.data])
    
    def __eq__(self, other):
        if isinstance(other, MockArray):
            return MockArray([a == b for a, b in zip(self.data, other.data)])
        return MockArray([x == other for x in self.data])

def array(data, dtype=None):
    """Create a mock numpy array."""
    return MockArray(data, dtype)

def zeros(shape, dtype='float64'):
    """Create array of zeros."""
    if isinstance(shape, int):
        return MockArray([0] * shape, dtype)
    elif isinstance(shape, tuple):
        total_size = 1
        for dim in shape:
            total_size *= dim
        result = MockArray([0] * total_size, dtype)
        result.shape = shape
        return result
    return MockArray([0], dtype)

def ones(shape, dtype='float64'):
    """Create array of ones."""
    if isinstance(shape, int):
        return MockArray([1] * shape, dtype)
    elif isinstance(shape, tuple):
        total_size = 1
        for dim in shape:
            total_size *= dim
        result = MockArray([1] * total_size, dtype)
        result.shape = shape
        return result
    return MockArray([1], dtype)

def zeros_like(arr):
    """Create zeros with same shape as input."""
    return zeros(arr.shape, arr.dtype)

def ones_like(arr):
    """Create ones with same shape as input."""
    return ones(arr.shape, arr.dtype)

def arange(start, stop=None, step=1, dtype=None):
    """Create array with evenly spaced values."""
    if stop is None:
        stop = start
        start = 0
    
    result = []
    current = start
    while current < stop:
        result.append(current)
        current += step
    
    return MockArray(result, dtype)

def linspace(start, stop, num=50, dtype=None):
    """Create array with linearly spaced values."""
    if num <= 1:
        return MockArray([start], dtype)
    
    step = (stop - start) / (num - 1)
    result = [start + i * step for i in range(num)]
    return MockArray(result, dtype)

def random_random(size=None):
    """Generate random numbers."""
    if size is None:
        return python_random.random()
    elif isinstance(size, int):
        return MockArray([python_random.random() for _ in range(size)])
    elif isinstance(size, tuple):
        total_size = 1
        for dim in size:
            total_size *= dim
        result = MockArray([python_random.random() for _ in range(total_size)])
        result.shape = size
        return result
    return MockArray([python_random.random()])

def random_randn(*size):
    """Generate random numbers from normal distribution."""
    if len(size) == 0:
        return python_random.gauss(0, 1)
    elif len(size) == 1:
        return MockArray([python_random.gauss(0, 1) for _ in range(size[0])])
    else:
        total_size = 1
        for dim in size:
            total_size *= dim
        result = MockArray([python_random.gauss(0, 1) for _ in range(total_size)])
        result.shape = size
        return result

def random_randint(low, high=None, size=None, dtype='int'):
    """Generate random integers."""
    if high is None:
        high = low
        low = 0
    
    if size is None:
        return python_random.randint(low, high - 1)
    elif isinstance(size, int):
        return MockArray([python_random.randint(low, high - 1) for _ in range(size)], dtype)
    elif isinstance(size, tuple):
        total_size = 1
        for dim in size:
            total_size *= dim
        result = MockArray([python_random.randint(low, high - 1) for _ in range(total_size)], dtype)
        result.shape = size
        return result
    return MockArray([python_random.randint(low, high - 1)], dtype)

def random_normal(loc=0.0, scale=1.0, size=None):
    """Generate random numbers from normal distribution."""
    if size is None:
        return python_random.gauss(loc, scale)
    elif isinstance(size, int):
        return MockArray([python_random.gauss(loc, scale) for _ in range(size)])
    elif isinstance(size, tuple):
        total_size = 1
        for dim in size:
            total_size *= dim
        result = MockArray([python_random.gauss(loc, scale) for _ in range(total_size)])
        result.shape = size
        return result
    return MockArray([python_random.gauss(loc, scale)])

def random_permutation(x):
    """Random permutation."""
    if isinstance(x, int):
        data = list(range(x))
        python_random.shuffle(data)
        return MockArray(data)
    elif isinstance(x, MockArray):
        data = x.data.copy()
        python_random.shuffle(data)
        result = MockArray(data)
        result.shape = x.shape
        return result
    return MockArray(x)

def argsort(arr, axis=-1):
    """Return indices that would sort array."""
    if isinstance(arr, MockArray):
        indexed_data = [(val, idx) for idx, val in enumerate(arr.data)]
        indexed_data.sort(key=lambda x: x[0])
        return MockArray([idx for _, idx in indexed_data])
    return MockArray(list(range(len(arr))))

def argmax(arr, axis=None):
    """Return indices of maximum values."""
    if isinstance(arr, MockArray):
        if axis is None:
            max_val = max(arr.flatten().data)
            return arr.flatten().data.index(max_val)
        else:
            # Simplified for axis case
            return MockArray([0])
    return 0

def maximum(arr1, arr2):
    """Element-wise maximum."""
    if isinstance(arr1, MockArray) and isinstance(arr2, MockArray):
        return MockArray([max(a, b) for a, b in zip(arr1.data, arr2.data)])
    elif isinstance(arr1, MockArray):
        return MockArray([max(x, arr2) for x in arr1.data])
    elif isinstance(arr2, MockArray):
        return MockArray([max(arr1, x) for x in arr2.data])
    return max(arr1, arr2)

def corrcoef(x, y=None):
    """Correlation coefficient."""
    if y is None:
        # Auto-correlation
        return MockArray([[1.0, 0.5], [0.5, 1.0]])
    
    # Simplified correlation calculation
    if isinstance(x, MockArray):
        x_data = x.flatten().data
    else:
        x_data = [x] if not hasattr(x, '__iter__') else list(x)
    
    if isinstance(y, MockArray):
        y_data = y.flatten().data
    else:
        y_data = [y] if not hasattr(y, '__iter__') else list(y)
    
    if len(x_data) != len(y_data) or len(x_data) == 0:
        return MockArray([[1.0, 0.0], [0.0, 1.0]])
    
    # Simple correlation calculation
    n = len(x_data)
    sum_x = sum(x_data)
    sum_y = sum(y_data)
    sum_xy = sum(a * b for a, b in zip(x_data, y_data))
    sum_x2 = sum(a * a for a in x_data)
    sum_y2 = sum(a * a for a in y_data)
    
    numerator = n * sum_xy - sum_x * sum_y
    denominator = math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
    
    if denominator == 0:
        corr = 0.0
    else:
        corr = numerator / denominator
    
    return MockArray([[1.0, corr], [corr, 1.0]])

def vstack(arrays):
    """Stack arrays vertically."""
    all_data = []
    for arr in arrays:
        if isinstance(arr, MockArray):
            all_data.extend(arr.data)
        else:
            all_data.extend(list(arr))
    return MockArray(all_data)

def hstack(arrays):
    """Stack arrays horizontally."""
    return vstack(arrays)  # Simplified

def stack(arrays, axis=0):
    """Stack arrays along new axis."""
    return vstack(arrays)  # Simplified

def exp(arr):
    """Element-wise exponential."""
    if isinstance(arr, MockArray):
        return MockArray([math.exp(x) for x in arr.data])
    return math.exp(arr)

def log(arr):
    """Element-wise natural logarithm."""
    if isinstance(arr, MockArray):
        return MockArray([math.log(x) if x > 0 else 0 for x in arr.data])
    return math.log(arr) if arr > 0 else 0

def sqrt(arr):
    """Element-wise square root."""
    if isinstance(arr, MockArray):
        return MockArray([math.sqrt(abs(x)) for x in arr.data])
    return math.sqrt(abs(arr))

def abs_func(arr):
    """Element-wise absolute value."""
    if isinstance(arr, MockArray):
        return MockArray([abs(float(x)) for x in arr.data])
    return abs(float(arr))

def isnan(arr):
    """Test for NaN values."""
    if isinstance(arr, MockArray):
        return MockArray([math.isnan(x) if isinstance(x, float) else False for x in arr.data])
    return math.isnan(arr) if isinstance(arr, float) else False

def histogram(arr, bins=10):
    """Compute histogram."""
    if isinstance(arr, MockArray):
        data = arr.flatten().data
    else:
        data = list(arr)
    
    if len(data) == 0:
        return MockArray([0] * bins), MockArray([0] * (bins + 1))
    
    min_val = min(data)
    max_val = max(data)
    
    if min_val == max_val:
        hist = [len(data)] + [0] * (bins - 1)
        bin_edges = [min_val] * (bins + 1)
    else:
        bin_width = (max_val - min_val) / bins
        hist = [0] * bins
        bin_edges = [min_val + i * bin_width for i in range(bins + 1)]
        
        for val in data:
            bin_idx = min(int((val - min_val) / bin_width), bins - 1)
            hist[bin_idx] += 1
    
    return MockArray(hist), MockArray(bin_edges)

# Math functions
def sin(arr):
    if isinstance(arr, MockArray):
        return MockArray([math.sin(x) for x in arr.data])
    return math.sin(arr)

def cos(arr):
    if isinstance(arr, MockArray):
        return MockArray([math.cos(x) for x in arr.data])
    return math.cos(arr)

def tanh(arr):
    if isinstance(arr, MockArray):
        return MockArray([math.tanh(x) for x in arr.data])
    return math.tanh(arr)

# FFT mock
class fft:
    @staticmethod
    def fft(arr):
        """Mock FFT - just return the input array."""
        return arr
    
    @staticmethod
    def ifft(arr):
        """Mock inverse FFT - just return the input array."""
        return arr
    
    @staticmethod
    def fftfreq(n, d=1.0):
        """Mock FFT frequencies."""
        return MockArray([i / (n * d) for i in range(n)])

# Random module
class random:
    random = random_random
    randn = random_randn
    randint = random_randint
    normal = random_normal
    permutation = random_permutation
    
    @staticmethod
    def uniform(low=0.0, high=1.0, size=None):
        """Generate uniform random numbers."""
        if size is None:
            return python_random.uniform(low, high)
        elif isinstance(size, int):
            return MockArray([python_random.uniform(low, high) for _ in range(size)])
        elif isinstance(size, tuple):
            total_size = 1
            for dim in size:
                total_size *= dim
            result = MockArray([python_random.uniform(low, high) for _ in range(total_size)])
            result.shape = size
            return result
        return MockArray([python_random.uniform(low, high)])
    
    @staticmethod
    def exponential(scale=1.0, size=None):
        """Generate exponential random numbers."""
        if size is None:
            return python_random.expovariate(1.0 / scale)
        elif isinstance(size, int):
            return MockArray([python_random.expovariate(1.0 / scale) for _ in range(size)])
        elif isinstance(size, tuple):
            total_size = 1
            for dim in size:
                total_size *= dim
            result = MockArray([python_random.expovariate(1.0 / scale) for _ in range(total_size)])
            result.shape = size
            return result
        return MockArray([python_random.expovariate(1.0 / scale)])

# Module-level constants
pi = math.pi
e = math.e
inf = float('inf')

# Export everything to sys.modules
numpy_mock = sys.modules[__name__]
numpy_mock.array = array
numpy_mock.ndarray = MockArray  # Add ndarray alias
numpy_mock.zeros = zeros
numpy_mock.ones = ones
numpy_mock.zeros_like = zeros_like
numpy_mock.ones_like = ones_like
numpy_mock.arange = arange
numpy_mock.linspace = linspace
numpy_mock.argsort = argsort
numpy_mock.argmax = argmax
numpy_mock.maximum = maximum
numpy_mock.corrcoef = corrcoef
numpy_mock.vstack = vstack
numpy_mock.hstack = hstack
numpy_mock.stack = stack
numpy_mock.exp = exp
numpy_mock.log = log
numpy_mock.sqrt = sqrt
numpy_mock.abs = abs_func
numpy_mock.isnan = isnan
numpy_mock.histogram = histogram
numpy_mock.sin = sin
numpy_mock.cos = cos
numpy_mock.tanh = tanh
numpy_mock.fft = fft
numpy_mock.random = random
numpy_mock.pi = pi
numpy_mock.e = e
numpy_mock.inf = inf
numpy_mock.nan = float('nan')
numpy_mock.uint8 = 'uint8'
numpy_mock.int32 = 'int32'
numpy_mock.int64 = 'int64'
numpy_mock.float32 = 'float32'
numpy_mock.float64 = 'float64'

# Add sum function at module level
def module_sum(arr, axis=None, dtype=None, keepdims=False):
    """Module-level sum function."""
    if isinstance(arr, MockArray):
        return arr.sum(axis)
    elif hasattr(arr, '__iter__'):
        if axis is None:
            # Use built-in sum to avoid recursion
            total = 0
            for item in arr:
                if hasattr(item, '__iter__'):
                    total += module_sum(item)
                else:
                    total += item
            return total
        else:
            # For axis-based sum, return MockArray
            if axis == 1 and len(arr) > 0 and hasattr(arr[0], '__iter__'):
                # Sum along axis 1 (rows)
                return MockArray([module_sum(row) for row in arr])
            else:
                total = 0
                for item in arr:
                    if hasattr(item, '__iter__'):
                        total += module_sum(item)
                    else:
                        total += item
                return MockArray([total])
    return arr

numpy_mock.sum = module_sum

# Add mean function at module level
def module_mean(arr, axis=None, dtype=None, keepdims=False):
    """Module-level mean function."""
    if isinstance(arr, MockArray):
        return arr.mean(axis)
    elif hasattr(arr, '__iter__'):
        total = module_sum(arr, axis)
        count = len(list(arr)) if hasattr(arr, '__len__') else 1
        return total / count if count > 0 else 0
    return arr

numpy_mock.mean = module_mean

# Add more missing functions
def round_func(arr, decimals=0):
    """Round array elements."""
    if isinstance(arr, MockArray):
        return MockArray([round(float(x), decimals) for x in arr.data])
    return round(float(arr), decimals)

def real_func(arr):
    """Get real part of complex array."""
    if isinstance(arr, MockArray):
        return MockArray([float(x.real if hasattr(x, 'real') else x) for x in arr.data])
    return float(arr.real if hasattr(arr, 'real') else arr)

numpy_mock.round = round_func
numpy_mock.real = real_func

# Register in sys.modules
sys.modules['numpy'] = numpy_mock
sys.modules['numpy.random'] = random
sys.modules['numpy.fft'] = fft

# Create np alias
np = numpy_mock