"""
Advanced Multi-Modal Sensor Fusion using Graph Neural Networks.

This module implements sophisticated fusion techniques for combining power, 
electromagnetic, acoustic, and optical side-channel measurements using 
graph-based neural architectures.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Set
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse, to_dense_batch
import logging
from dataclasses import dataclass, field
from pathlib import Path
import json
from collections import defaultdict

from .core import TraceData
from .utils.logging_utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class SensorConfig:
    """Configuration for individual sensor channels."""
    name: str
    sample_rate: float
    resolution: int
    noise_floor: float
    frequency_range: Tuple[float, float]
    spatial_position: Optional[Tuple[float, float, float]] = None
    calibration_factor: float = 1.0
    preprocessing: List[str] = field(default_factory=list)

@dataclass  
class MultiModalData:
    """Container for multi-modal side-channel measurements."""
    power_traces: Optional[np.ndarray] = None
    em_near_traces: Optional[np.ndarray] = None  
    em_far_traces: Optional[np.ndarray] = None
    acoustic_traces: Optional[np.ndarray] = None
    optical_traces: Optional[np.ndarray] = None
    timestamps: Optional[np.ndarray] = None
    sensor_configs: Dict[str, SensorConfig] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_available_modalities(self) -> List[str]:
        """Get list of available measurement modalities."""
        modalities = []
        if self.power_traces is not None:
            modalities.append('power')
        if self.em_near_traces is not None:
            modalities.append('em_near')
        if self.em_far_traces is not None:
            modalities.append('em_far')
        if self.acoustic_traces is not None:
            modalities.append('acoustic')
        if self.optical_traces is not None:
            modalities.append('optical')
        return modalities
    
    def get_trace_data(self, modality: str) -> Optional[np.ndarray]:
        """Get trace data for specific modality."""
        modality_map = {
            'power': self.power_traces,
            'em_near': self.em_near_traces,
            'em_far': self.em_far_traces,
            'acoustic': self.acoustic_traces,
            'optical': self.optical_traces
        }
        return modality_map.get(modality)
    
    def synchronize_traces(self, reference_modality: str = 'power') -> 'MultiModalData':
        """Synchronize all traces to reference modality timing."""
        if self.timestamps is None:
            logger.warning("No timestamp information available for synchronization")
            return self
        
        ref_traces = self.get_trace_data(reference_modality)
        if ref_traces is None:
            logger.error(f"Reference modality {reference_modality} not available")
            return self
        
        # Implement time-based synchronization
        # This is a simplified version - real implementation would use cross-correlation
        synchronized_data = MultiModalData()
        synchronized_data.sensor_configs = self.sensor_configs.copy()
        synchronized_data.metadata = self.metadata.copy()
        
        target_length = len(ref_traces[0]) if len(ref_traces) > 0 else 0
        
        for modality in self.get_available_modalities():
            traces = self.get_trace_data(modality)
            if traces is not None:
                # Simple length synchronization (in practice, use time-based alignment)
                if len(traces[0]) != target_length:
                    synchronized_traces = []
                    for trace in traces:
                        if len(trace) > target_length:
                            # Truncate
                            sync_trace = trace[:target_length]
                        else:
                            # Pad with zeros
                            sync_trace = np.pad(trace, (0, target_length - len(trace)))
                        synchronized_traces.append(sync_trace)
                    traces = np.array(synchronized_traces)
                
                setattr(synchronized_data, f"{modality}_traces", traces)
        
        return synchronized_data

class GraphTopologyBuilder:
    """Builds graph topologies for multi-modal sensor fusion."""
    
    def __init__(self):
        self.topology_cache = {}
    
    def build_spatial_graph(self, sensor_positions: Dict[str, Tuple[float, float, float]],
                          connection_threshold: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build graph based on spatial relationships between sensors."""
        sensors = list(sensor_positions.keys())
        n_sensors = len(sensors)
        
        if n_sensors == 0:
            return torch.empty((2, 0)), torch.empty(0)
        
        # Create adjacency matrix based on spatial distances
        adjacency = np.zeros((n_sensors, n_sensors))
        
        for i, sensor_i in enumerate(sensors):
            pos_i = np.array(sensor_positions[sensor_i])
            for j, sensor_j in enumerate(sensors):
                if i != j:
                    pos_j = np.array(sensor_positions[sensor_j])
                    distance = np.linalg.norm(pos_i - pos_j)
                    
                    # Connect sensors within threshold distance
                    if distance < connection_threshold:
                        # Weight inversely proportional to distance
                        weight = 1.0 / (1.0 + distance)
                        adjacency[i, j] = weight
        
        # Convert to edge format
        edge_indices = []
        edge_weights = []
        
        for i in range(n_sensors):
            for j in range(n_sensors):
                if adjacency[i, j] > 0:
                    edge_indices.append([i, j])
                    edge_weights.append(adjacency[i, j])
        
        if len(edge_indices) == 0:
            # Fully connected if no spatial connections
            edge_indices = [[i, j] for i in range(n_sensors) for j in range(n_sensors) if i != j]
            edge_weights = [1.0] * len(edge_indices)
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
        
        return edge_index, edge_weight
    
    def build_temporal_graph(self, trace_length: int, 
                           temporal_window: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build graph connecting temporal points within window."""
        edge_indices = []
        edge_weights = []
        
        for t in range(trace_length):
            for offset in range(1, temporal_window + 1):
                # Forward connections
                if t + offset < trace_length:
                    edge_indices.append([t, t + offset])
                    edge_weights.append(1.0 / offset)  # Closer points have higher weight
                
                # Backward connections
                if t - offset >= 0:
                    edge_indices.append([t, t - offset])
                    edge_weights.append(1.0 / offset)
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
        
        return edge_index, edge_weight
    
    def build_frequency_graph(self, frequencies: np.ndarray,
                            freq_bandwidth: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build graph connecting similar frequency components."""
        n_freq = len(frequencies)
        edge_indices = []
        edge_weights = []
        
        for i in range(n_freq):
            for j in range(i + 1, n_freq):
                freq_diff = abs(frequencies[i] - frequencies[j])
                if freq_diff < freq_bandwidth:
                    weight = 1.0 / (1.0 + freq_diff / freq_bandwidth)
                    edge_indices.extend([[i, j], [j, i]])
                    edge_weights.extend([weight, weight])
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
        
        return edge_index, edge_weight

class MultiModalGraphAttention(nn.Module):
    """Graph Attention Network for multi-modal fusion."""
    
    def __init__(self, 
                 input_dims: Dict[str, int],
                 hidden_dim: int = 128,
                 output_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        
        # Modality-specific encoders
        self.modality_encoders = nn.ModuleDict()
        for modality, input_dim in input_dims.items():
            self.modality_encoders[modality] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                in_dim = hidden_dim
            else:
                in_dim = hidden_dim * num_heads
            
            if layer == num_layers - 1:
                # Last layer
                self.gat_layers.append(
                    GATConv(in_dim, output_dim, heads=1, dropout=dropout, concat=False)
                )
            else:
                self.gat_layers.append(
                    GATConv(in_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True)
                )
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim * len(input_dims), output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, 
                batch_data: Dict[str, torch.Tensor],
                edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None,
                batch_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through multi-modal graph attention network."""
        
        # Encode each modality
        encoded_modalities = {}
        for modality, data in batch_data.items():
            if modality in self.modality_encoders:
                encoded = self.modality_encoders[modality](data)
                encoded_modalities[modality] = encoded
        
        # Create unified graph representation
        # Concatenate all modality features
        all_features = []
        modality_indices = {}
        current_idx = 0
        
        for modality, features in encoded_modalities.items():
            all_features.append(features)
            n_nodes = features.size(0)
            modality_indices[modality] = (current_idx, current_idx + n_nodes)
            current_idx += n_nodes
        
        if not all_features:
            return torch.zeros(1, self.output_dim)
        
        x = torch.cat(all_features, dim=0)
        
        # Apply graph attention layers
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index, edge_weight)
            x = F.relu(x)
        
        # Pool features by modality for cross-attention
        modality_features = []
        for modality, (start_idx, end_idx) in modality_indices.items():
            mod_features = x[start_idx:end_idx]
            # Global pooling for modality
            pooled = torch.mean(mod_features, dim=0, keepdim=True)
            modality_features.append(pooled)
        
        if len(modality_features) > 1:
            # Cross-modal attention
            stacked_features = torch.stack(modality_features, dim=0)  # (n_modalities, 1, hidden_dim)
            stacked_features = stacked_features.squeeze(1).unsqueeze(0)  # (1, n_modalities, hidden_dim)
            
            attended_features, _ = self.cross_attention(
                stacked_features, stacked_features, stacked_features
            )
            attended_features = attended_features.squeeze(0)  # (n_modalities, hidden_dim)
        else:
            attended_features = torch.stack(modality_features).squeeze(1)
        
        # Final fusion
        fused_features = torch.cat([feat for feat in attended_features], dim=-1)
        output = self.fusion_layer(fused_features.unsqueeze(0))
        
        return output.squeeze(0)

class AdaptiveMultiModalFusion(nn.Module):
    """Adaptive fusion network that learns optimal combination strategies."""
    
    def __init__(self,
                 input_dims: Dict[str, int],
                 hidden_dim: int = 128,
                 output_dim: int = 256,
                 num_fusion_heads: int = 4):
        super().__init__()
        
        self.input_dims = input_dims
        self.modalities = list(input_dims.keys())
        
        # Individual modality networks
        self.modality_networks = nn.ModuleDict()
        for modality, input_dim in input_dims.items():
            self.modality_networks[modality] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, output_dim)
            )
        
        # Attention weights for adaptive fusion
        self.fusion_attention = nn.Parameter(
            torch.ones(len(self.modalities)) / len(self.modalities)
        )
        
        # Quality assessment network
        self.quality_network = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Multi-head fusion
        self.fusion_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dim * len(self.modalities), output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim)
            )
            for _ in range(num_fusion_heads)
        ])
        
        # Final combination
        self.final_layer = nn.Linear(output_dim * num_fusion_heads, output_dim)
        
    def forward(self, modality_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass with adaptive fusion."""
        
        # Process each modality
        processed_modalities = {}
        quality_scores = {}
        
        for modality in self.modalities:
            if modality in modality_data:
                # Process modality
                processed = self.modality_networks[modality](modality_data[modality])
                processed_modalities[modality] = processed
                
                # Assess quality
                quality = self.quality_network(processed)
                quality_scores[modality] = quality
            else:
                # Handle missing modalities
                batch_size = list(modality_data.values())[0].size(0)
                processed_modalities[modality] = torch.zeros(
                    batch_size, self.input_dims[modality]
                ).to(list(modality_data.values())[0].device)
                quality_scores[modality] = torch.zeros(batch_size, 1).to(
                    list(modality_data.values())[0].device)
        
        # Adaptive attention based on quality
        attention_weights = []
        for i, modality in enumerate(self.modalities):
            base_weight = torch.softmax(self.fusion_attention, dim=0)[i]
            quality_weight = quality_scores[modality].mean()
            adaptive_weight = base_weight * quality_weight
            attention_weights.append(adaptive_weight)
        
        # Normalize attention weights
        total_weight = sum(attention_weights)
        if total_weight > 0:
            attention_weights = [w / total_weight for w in attention_weights]
        else:
            attention_weights = [1.0 / len(attention_weights)] * len(attention_weights)
        
        # Multi-head fusion
        head_outputs = []
        for head in self.fusion_heads:
            # Weighted combination
            weighted_features = []
            for i, modality in enumerate(self.modalities):
                weighted = processed_modalities[modality] * attention_weights[i]
                weighted_features.append(weighted)
            
            concatenated = torch.cat(weighted_features, dim=-1)
            head_output = head(concatenated)
            head_outputs.append(head_output)
        
        # Final combination
        all_heads = torch.cat(head_outputs, dim=-1)
        final_output = self.final_layer(all_heads)
        
        return {
            'fused_output': final_output,
            'attention_weights': dict(zip(self.modalities, attention_weights)),
            'quality_scores': quality_scores,
            'modality_outputs': processed_modalities
        }

class MultiModalSideChannelAnalyzer:
    """Complete multi-modal side-channel analysis framework."""
    
    def __init__(self,
                 fusion_method: str = 'graph_attention',
                 device: str = 'cpu'):
        self.fusion_method = fusion_method
        self.device = torch.device(device)
        self.topology_builder = GraphTopologyBuilder()
        
        # Initialize fusion network (will be set when first data is processed)
        self.fusion_network = None
        self.training_history = []
        
        logger.info(f"Initialized MultiModalSideChannelAnalyzer with {fusion_method} fusion")
    
    def _initialize_fusion_network(self, data: MultiModalData):
        """Initialize fusion network based on available modalities."""
        modalities = data.get_available_modalities()
        
        # Determine input dimensions for each modality
        input_dims = {}
        for modality in modalities:
            traces = data.get_trace_data(modality)
            if traces is not None and len(traces) > 0:
                input_dims[modality] = len(traces[0])
        
        if self.fusion_method == 'graph_attention':
            self.fusion_network = MultiModalGraphAttention(
                input_dims=input_dims,
                hidden_dim=128,
                output_dim=256,
                num_heads=8,
                num_layers=3
            ).to(self.device)
        elif self.fusion_method == 'adaptive':
            self.fusion_network = AdaptiveMultiModalFusion(
                input_dims=input_dims,
                hidden_dim=128,
                output_dim=256
            ).to(self.device)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        logger.info(f"Initialized {self.fusion_method} network with modalities: {modalities}")
    
    def prepare_graph_data(self, data: MultiModalData) -> Tuple[Dict[str, torch.Tensor], 
                                                              torch.Tensor, torch.Tensor]:
        """Prepare data for graph-based fusion."""
        modalities = data.get_available_modalities()
        
        # Convert traces to tensors
        batch_data = {}
        for modality in modalities:
            traces = data.get_trace_data(modality)
            if traces is not None:
                batch_data[modality] = torch.FloatTensor(traces).to(self.device)
        
        # Build graph topology
        if data.sensor_configs:
            # Use spatial positions if available
            positions = {}
            for modality, config in data.sensor_configs.items():
                if config.spatial_position:
                    positions[modality] = config.spatial_position
            
            if positions:
                edge_index, edge_weight = self.topology_builder.build_spatial_graph(
                    positions, connection_threshold=0.2
                )
            else:
                # Fully connected graph
                n_modalities = len(modalities)
                edges = [[i, j] for i in range(n_modalities) for j in range(n_modalities) if i != j]
                edge_index = torch.tensor(edges, dtype=torch.long).t().to(self.device)
                edge_weight = torch.ones(len(edges), dtype=torch.float).to(self.device)
        else:
            # Default fully connected
            n_modalities = len(modalities)
            edges = [[i, j] for i in range(n_modalities) for j in range(n_modalities) if i != j]
            edge_index = torch.tensor(edges, dtype=torch.long).t().to(self.device)
            edge_weight = torch.ones(len(edges), dtype=torch.float).to(self.device)
        
        return batch_data, edge_index, edge_weight
    
    def analyze_multi_modal(self, data: MultiModalData,
                           target_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Perform multi-modal side-channel analysis."""
        
        # Initialize fusion network if needed
        if self.fusion_network is None:
            self._initialize_fusion_network(data)
        
        # Synchronize traces
        sync_data = data.synchronize_traces()
        
        # Prepare data based on fusion method
        if self.fusion_method == 'graph_attention':
            batch_data, edge_index, edge_weight = self.prepare_graph_data(sync_data)
            
            # Forward pass
            with torch.no_grad():
                fused_output = self.fusion_network(batch_data, edge_index, edge_weight)
        
        elif self.fusion_method == 'adaptive':
            modality_tensors = {}
            for modality in sync_data.get_available_modalities():
                traces = sync_data.get_trace_data(modality)
                if traces is not None:
                    modality_tensors[modality] = torch.FloatTensor(traces).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                results = self.fusion_network(modality_tensors)
                fused_output = results['fused_output']
        
        # Analysis results
        analysis_results = {
            'fused_features': fused_output.cpu().numpy(),
            'n_traces': len(list(sync_data.get_trace_data(sync_data.get_available_modalities()[0]))),
            'modalities_used': sync_data.get_available_modalities(),
            'fusion_method': self.fusion_method
        }
        
        # Add modality-specific analysis
        if self.fusion_method == 'adaptive':
            analysis_results.update({
                'attention_weights': results['attention_weights'],
                'quality_scores': {k: v.cpu().numpy() for k, v in results['quality_scores'].items()},
                'modality_outputs': {k: v.cpu().numpy() for k, v in results['modality_outputs'].items()}
            })
        
        # Compute fusion quality metrics
        if len(sync_data.get_available_modalities()) > 1:
            fusion_quality = self._compute_fusion_quality(sync_data, fused_output.cpu().numpy())
            analysis_results['fusion_quality'] = fusion_quality
        
        logger.info(f"Multi-modal analysis complete: "
                   f"{len(sync_data.get_available_modalities())} modalities, "
                   f"{analysis_results['n_traces']} traces")
        
        return analysis_results
    
    def _compute_fusion_quality(self, data: MultiModalData, fused_features: np.ndarray) -> Dict[str, float]:
        """Compute quality metrics for fusion."""
        modalities = data.get_available_modalities()
        
        # Signal-to-noise ratio improvement
        individual_snrs = []
        for modality in modalities:
            traces = data.get_trace_data(modality)
            if traces is not None and len(traces) > 0:
                signal_var = np.var(np.mean(traces, axis=0))
                noise_var = np.mean(np.var(traces, axis=1))
                snr = signal_var / (noise_var + 1e-10)
                individual_snrs.append(snr)
        
        # Fused SNR estimate
        if len(fused_features) > 0:
            fused_signal_var = np.var(np.mean(fused_features, axis=0))
            fused_noise_var = np.mean(np.var(fused_features, axis=1))
            fused_snr = fused_signal_var / (fused_noise_var + 1e-10)
        else:
            fused_snr = 0.0
        
        avg_individual_snr = np.mean(individual_snrs) if individual_snrs else 0.0
        snr_improvement = (fused_snr / avg_individual_snr) if avg_individual_snr > 0 else 1.0
        
        # Correlation between modalities
        correlations = []
        if len(modalities) > 1:
            for i, mod1 in enumerate(modalities):
                for mod2 in modalities[i+1:]:
                    traces1 = data.get_trace_data(mod1)
                    traces2 = data.get_trace_data(mod2)
                    if traces1 is not None and traces2 is not None:
                        # Compute average correlation
                        min_len = min(len(traces1[0]), len(traces2[0]))
                        corr = np.corrcoef(
                            np.mean(traces1[:, :min_len], axis=0),
                            np.mean(traces2[:, :min_len], axis=0)
                        )[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
        
        avg_correlation = np.mean(correlations) if correlations else 0.0
        
        return {
            'snr_improvement': float(snr_improvement),
            'average_correlation': float(avg_correlation),
            'fusion_snr': float(fused_snr),
            'individual_snrs': [float(x) for x in individual_snrs]
        }
    
    def train_fusion_network(self, 
                           training_data: List[MultiModalData],
                           labels: List[np.ndarray],
                           epochs: int = 100,
                           learning_rate: float = 1e-3) -> Dict[str, List[float]]:
        """Train the multi-modal fusion network."""
        
        if self.fusion_network is None:
            self._initialize_fusion_network(training_data[0])
        
        optimizer = torch.optim.Adam(self.fusion_network.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        train_losses = []
        train_accuracies = []
        
        for epoch in range(epochs):
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            
            for i, (data, label_array) in enumerate(zip(training_data, labels)):
                # Prepare data
                if self.fusion_method == 'graph_attention':
                    batch_data, edge_index, edge_weight = self.prepare_graph_data(data)
                    fused_output = self.fusion_network(batch_data, edge_index, edge_weight)
                elif self.fusion_method == 'adaptive':
                    modality_tensors = {}
                    for modality in data.get_available_modalities():
                        traces = data.get_trace_data(modality)
                        if traces is not None:
                            modality_tensors[modality] = torch.FloatTensor(traces).to(self.device)
                    results = self.fusion_network(modality_tensors)
                    fused_output = results['fused_output']
                
                # Convert to classification logits
                logits = torch.mean(fused_output, dim=0, keepdim=True)  # Simplified
                labels_tensor = torch.LongTensor([label_array[0]]).to(self.device)  # Simplified
                
                # Compute loss
                loss = criterion(logits, labels_tensor)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Compute accuracy
                _, predicted = torch.max(logits.data, 1)
                total_samples += labels_tensor.size(0)
                total_correct += (predicted == labels_tensor).sum().item()
            
            avg_loss = total_loss / len(training_data)
            accuracy = total_correct / total_samples if total_samples > 0 else 0.0
            
            train_losses.append(avg_loss)
            train_accuracies.append(accuracy)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.3f}")
        
        self.training_history = {
            'losses': train_losses,
            'accuracies': train_accuracies
        }
        
        return self.training_history
    
    def save_model(self, path: str):
        """Save trained fusion model."""
        if self.fusion_network:
            torch.save({
                'model_state_dict': self.fusion_network.state_dict(),
                'fusion_method': self.fusion_method,
                'training_history': self.training_history
            }, path)
            logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained fusion model."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.fusion_method = checkpoint['fusion_method']
        self.training_history = checkpoint.get('training_history', [])
        
        # Model will be initialized when first used
        self._model_state_dict = checkpoint['model_state_dict']
        
        logger.info(f"Model loaded from {path}")

# Utility functions
def create_synthetic_multimodal_data(n_traces: int = 1000,
                                   trace_length: int = 5000,
                                   modalities: List[str] = None) -> MultiModalData:
    """Create synthetic multi-modal data for testing."""
    if modalities is None:
        modalities = ['power', 'em_near', 'acoustic']
    
    data = MultiModalData()
    
    for modality in modalities:
        # Generate realistic synthetic traces
        base_signal = np.random.randn(n_traces, trace_length)
        
        if modality == 'power':
            # Power traces: higher SNR, more structured
            signal = base_signal + 0.1 * np.sin(np.linspace(0, 10*np.pi, trace_length))
            noise = np.random.normal(0, 0.05, (n_traces, trace_length))
        elif modality == 'em_near':
            # EM near: medium SNR, some correlation with power
            signal = 0.7 * base_signal + 0.3 * np.random.randn(n_traces, trace_length)
            noise = np.random.normal(0, 0.1, (n_traces, trace_length))
        elif modality == 'acoustic':
            # Acoustic: lower SNR, different frequency content
            signal = base_signal + 0.05 * np.sin(np.linspace(0, 50*np.pi, trace_length))
            noise = np.random.normal(0, 0.2, (n_traces, trace_length))
        else:
            signal = base_signal
            noise = np.random.normal(0, 0.1, (n_traces, trace_length))
        
        traces = signal + noise
        setattr(data, f"{modality}_traces", traces)
        
        # Add sensor config
        data.sensor_configs[modality] = SensorConfig(
            name=modality,
            sample_rate=1e6,
            resolution=12,
            noise_floor=0.001,
            frequency_range=(1e3, 1e6),
            spatial_position=(hash(modality) % 10, 0, 0)  # Simple positioning
        )
    
    return data