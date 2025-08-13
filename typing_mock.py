"""Simple mock implementation for typing module."""

import sys

# Mock typing classes and functions
class TypeVar:
    def __init__(self, name, *args, **kwargs):
        self.name = name

class Generic:
    pass

class Any:
    pass

class UnionType:
    def __init__(self, *args):
        self.args = args
    
    def __getitem__(self, item):
        return UnionType(*item) if isinstance(item, tuple) else UnionType(item)

class Union:
    @staticmethod
    def __getitem__(item):
        return UnionType(*item) if isinstance(item, tuple) else UnionType(item)

class OptionalType:
    def __init__(self, arg):
        self.arg = arg

class Optional:
    @staticmethod
    def __getitem__(item):
        return OptionalType(item)

class ListType:
    def __init__(self, item_type=None):
        self.item_type = item_type

class List:
    @staticmethod
    def __getitem__(item):
        return ListType(item)

class DictType:
    def __init__(self, key_type=None, value_type=None):
        self.key_type = key_type
        self.value_type = value_type

class Dict:
    @staticmethod
    def __getitem__(item):
        if isinstance(item, tuple) and len(item) == 2:
            return DictType(item[0], item[1])
        return DictType(item, None)

class TupleType:
    def __init__(self, *item_types):
        self.item_types = item_types

class Tuple:
    @staticmethod
    def __getitem__(item):
        if isinstance(item, tuple):
            return TupleType(*item)
        return TupleType(item)

class Callable:
    def __init__(self, *args):
        self.args = args
    
    def __getitem__(self, item):
        return Callable(item)

class Set:
    def __init__(self, item_type=None):
        self.item_type = item_type
    
    def __getitem__(self, item):
        return Set(item)

class FrozenSet:
    def __init__(self, item_type=None):
        self.item_type = item_type
    
    def __getitem__(self, item):
        return FrozenSet(item)

class Sequence:
    def __init__(self, item_type=None):
        self.item_type = item_type
    
    def __getitem__(self, item):
        return Sequence(item)

class Mapping:
    def __init__(self, key_type=None, value_type=None):
        self.key_type = key_type
        self.value_type = value_type
    
    def __getitem__(self, item):
        if isinstance(item, tuple) and len(item) == 2:
            return Mapping(item[0], item[1])
        return Mapping(item)

class Iterable:
    def __init__(self, item_type=None):
        self.item_type = item_type
    
    def __getitem__(self, item):
        return Iterable(item)

class Iterator:
    def __init__(self, item_type=None):
        self.item_type = item_type
    
    def __getitem__(self, item):
        return Iterator(item)

class Generator:
    def __init__(self, *args):
        self.args = args
    
    def __getitem__(self, item):
        return Generator(item)

class TextIO:
    pass

class BinaryIO:
    pass

class IO:
    def __init__(self, item_type=None):
        self.item_type = item_type
    
    def __getitem__(self, item):
        return IO(item)

# Create module-level instances
Any = Any()
Union = Union()
Optional = Optional()
List = List()
Dict = Dict()
Tuple = Tuple()
Callable = Callable()
Set = Set()
FrozenSet = FrozenSet()
Sequence = Sequence()
Mapping = Mapping()
Iterable = Iterable()
Iterator = Iterator()
Generator = Generator()
TextIO = TextIO()
BinaryIO = BinaryIO()
IO = IO()

# Additional typing constructs
class ClassVar:
    def __init__(self, item_type=None):
        self.item_type = item_type
    
    def __getitem__(self, item):
        return ClassVar(item)

class NoReturn:
    pass

class Type:
    def __init__(self, item_type=None):
        self.item_type = item_type
    
    def __getitem__(self, item):
        return Type(item)

ClassVar = ClassVar()
NoReturn = NoReturn()
Type = Type()

# Function for runtime type checking (no-op)
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

# Mock internal typing constructs
class _GenericAlias:
    def __init__(self, origin, args):
        self.origin = origin
        self.args = args

_GenericAlias = _GenericAlias

# Register in sys.modules
typing_mock = sys.modules[__name__]
sys.modules['typing'] = typing_mock