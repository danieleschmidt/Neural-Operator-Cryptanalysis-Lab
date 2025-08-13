"""Very simple typing mock that just provides the names."""

import sys

# Just create simple placeholder objects that can be used in type annotations
class MockType:
    def __getitem__(self, item):
        return self
    
    def __call__(self, *args, **kwargs):
        return self

# Create instances for all commonly used typing constructs
Any = MockType()
Union = MockType()
Optional = MockType()
List = MockType()
Dict = MockType()
Tuple = MockType()
Callable = MockType()
Set = MockType()
FrozenSet = MockType()
Sequence = MockType()
Mapping = MockType()
Iterable = MockType()
Iterator = MockType()
Generator = MockType()
TextIO = MockType()
BinaryIO = MockType()
IO = MockType()
ClassVar = MockType()
NoReturn = MockType()
Type = MockType()
TypeVar = MockType()
Generic = MockType()
_GenericAlias = MockType()

# Mock functions
def cast(typ, val):
    return val

def overload(func):
    return func

def no_type_check(func):
    return func

def no_type_check_decorator(decorator):
    return decorator

def get_type_hints(obj, globalns=None, localns=None):
    return {}

# Register in sys.modules
typing_mock = sys.modules[__name__]
sys.modules['typing'] = typing_mock