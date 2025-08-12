"""Side-channel analysis framework for neural operator cryptanalysis.

This module provides comprehensive tools for analyzing various side-channel
emanations from cryptographic implementations including power consumption,
electromagnetic radiation, acoustic, and optical leakage.

Components:
- Power Analysis: CPA, DPA, template attacks with neural operators
- EM Analysis: Near/far-field EM analysis with spatial modeling
- Acoustic Analysis: Sound-based cryptanalysis techniques
- Multi-channel Fusion: Combine multiple side-channel sources
- Preprocessing: Signal conditioning and feature extraction
"""

from .base import SideChannelAnalyzer, AnalysisConfig, LeakageModel, TraceData
from .power import PowerAnalyzer, CPAnalyzer, DPAnalyzer
from .electromagnetic import EMAnalyzer, NearFieldEM, FarFieldEM
from .acoustic import AcousticAnalyzer, SoundBasedAttack
from .fusion import MultiChannelFusion, SensorFusion
from .preprocessing import TracePreprocessor, FeatureExtractor
from .attacks import TemplateAttack, ProfilingAttack, AdaptiveAttack

__all__ = [
    "SideChannelAnalyzer",
    "AnalysisConfig", 
    "LeakageModel",
    "TraceData",
    "PowerAnalyzer",
    "CPAnalyzer",
    "DPAnalyzer", 
    "EMAnalyzer",
    "NearFieldEM",
    "FarFieldEM",
    "AcousticAnalyzer",
    "SoundBasedAttack",
    "MultiChannelFusion",
    "SensorFusion",
    "TracePreprocessor",
    "FeatureExtractor",
    "TemplateAttack",
    "ProfilingAttack", 
    "AdaptiveAttack",
]