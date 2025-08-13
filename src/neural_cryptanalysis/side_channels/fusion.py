"""Multi-channel fusion for side-channel analysis."""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from .base import SideChannelAnalyzer, AnalysisConfig, TraceData


class MultiChannelFusion(nn.Module):
    """Fusion of multiple side-channel sources."""
    
    def __init__(self, channels: List[str], fusion_method: str = 'concatenation'):
        super().__init__()
        self.channels = channels
        self.fusion_method = fusion_method
        
        # Simple fusion layers
        if fusion_method == 'attention':
            self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4)
        elif fusion_method == 'weighted':
            self.weights = nn.Parameter(torch.ones(len(channels)))
            
    def forward(self, channel_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse multiple channel data."""
        if self.fusion_method == 'concatenation':
            # Simple concatenation
            tensors = [channel_data[ch] for ch in self.channels if ch in channel_data]
            return torch.cat(tensors, dim=-1) if tensors else torch.empty(0)
        elif self.fusion_method == 'weighted':
            # Weighted sum
            result = None
            for i, ch in enumerate(self.channels):
                if ch in channel_data:
                    weighted = channel_data[ch] * self.weights[i]
                    result = weighted if result is None else result + weighted
            return result if result is not None else torch.empty(0)
        else:
            # Default to first channel
            for ch in self.channels:
                if ch in channel_data:
                    return channel_data[ch]
            return torch.empty(0)


class SensorFusion:
    """Sensor fusion for multiple measurement sources."""
    
    def __init__(self, sensor_types: List[str]):
        self.sensor_types = sensor_types
        self.calibration = {}
        
    def calibrate_sensor(self, sensor_type: str, calibration_data: np.ndarray):
        """Calibrate individual sensor."""
        self.calibration[sensor_type] = {
            'mean': np.mean(calibration_data),
            'std': np.std(calibration_data)
        }
    
    def fuse_measurements(self, measurements: Dict[str, np.ndarray]) -> np.ndarray:
        """Fuse measurements from multiple sensors."""
        # Simple fusion - just concatenate normalized measurements
        fused_data = []
        
        for sensor_type in self.sensor_types:
            if sensor_type in measurements:
                data = measurements[sensor_type]
                
                # Apply calibration if available
                if sensor_type in self.calibration:
                    cal = self.calibration[sensor_type]
                    data = (data - cal['mean']) / cal['std']
                
                fused_data.append(data)
        
        if fused_data:
            return np.concatenate(fused_data, axis=-1)
        else:
            return np.array([])