"""Neural Operator Cryptanalysis Lab (noclab)."""

from .power_trace import PowerTraceSimulator
from .neural_operator import NeuralOperatorLeakageModel
from .cpa import CorrelationPowerAnalysis
from .adaptive_attacker import AdaptiveAttacker

__all__ = [
    "PowerTraceSimulator",
    "NeuralOperatorLeakageModel",
    "CorrelationPowerAnalysis",
    "AdaptiveAttacker",
]
