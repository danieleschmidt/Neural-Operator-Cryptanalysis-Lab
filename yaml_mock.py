"""Simple mock implementation for PyYAML functionality."""

import sys
import json
from typing import Any, Dict, Union, TextIO

def load(stream: Union[str, TextIO], Loader=None) -> Any:
    """Mock yaml.load function."""
    if isinstance(stream, str):
        # Simple parsing - try JSON format as fallback
        try:
            return json.loads(stream)
        except:
            # If JSON fails, return a simple dict
            return {'mock': True, 'data': stream}
    else:
        # If it's a file-like object, read and try to parse
        try:
            content = stream.read()
            return json.loads(content)
        except:
            return {'mock': True, 'content': str(content)}

def safe_load(stream: Union[str, TextIO]) -> Any:
    """Mock yaml.safe_load function."""
    return load(stream)

def dump(data: Any, stream=None, **kwargs) -> str:
    """Mock yaml.dump function."""
    if stream is None:
        # Return JSON string as a reasonable approximation
        try:
            return json.dumps(data, indent=2)
        except:
            return str(data)
    else:
        # Write to stream
        try:
            json.dump(data, stream, indent=2)
        except:
            stream.write(str(data))

def safe_dump(data: Any, stream=None, **kwargs) -> str:
    """Mock yaml.safe_dump function."""
    return dump(data, stream, **kwargs)

# Mock Loader classes
class SafeLoader:
    pass

class Loader:
    pass

# Register in sys.modules
yaml_mock = sys.modules[__name__]
yaml_mock.load = load
yaml_mock.safe_load = safe_load
yaml_mock.dump = dump
yaml_mock.safe_dump = safe_dump
yaml_mock.SafeLoader = SafeLoader
yaml_mock.Loader = Loader

sys.modules['yaml'] = yaml_mock