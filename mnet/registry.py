"""
Collection of all neural network models, activation functions and errors
"""

MODEL_MAP: dict[object: int] = {}
REVERSE_MODEL_MAP: dict[int: str] = {}
MODEL_DICT: dict[str: object] = {}

ACTIVATION_MAP: dict[function: int] = {}
REVERSE_ACTIVATION_MAP: dict[int: str] = {}
ACTIVATION_DICT: dict[str: function] = {}

class BaseNetworkError(Exception): pass
class ConfigError(BaseNetworkError): pass
class UpdateError(BaseNetworkError): pass

__all__ = [
    "MODEL_MAP",
    "REVERSE_MODEL_MAP",
    "MODEL_DICT",
    "ACTIVATION_MAP", 
    "REVERSE_ACTIVATION_MAP",
    "ACTIVATION_DICT",
    "ConfigError",
    "UpdateError"
]