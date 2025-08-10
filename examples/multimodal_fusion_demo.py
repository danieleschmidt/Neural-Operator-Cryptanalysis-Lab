#!/usr/bin/env python3
"""
Multi-Modal Sensor Fusion with Graph Neural Networks - Demonstration.

This example shows how to use advanced multi-modal fusion techniques
to combine power, electromagnetic, acoustic, and optical side-channel measurements.
"""

import sys
import os
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_cryptanalysis.multi_modal_fusion import (
    MultiModalData, SensorConfig, MultiModalSideChannelAnalyzer,
    create_synthetic_multimodal_data, GraphTopologyBuilder
)
from neural_cryptanalysis.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

def demonstrate_synthetic_data_generation():
    """Demonstrate synthetic multi-modal data generation."""
    print("\n" + "="*60)
    print("SYNTHETIC MULTI-MODAL DATA GENERATION")
    print("="*60)
    
    # Generate synthetic data with different modalities
    modalities = ['power', 'em_near', 'em_far', 'acoustic']
    data = create_synthetic_multimodal_data(
        n_traces=1000,
        trace_length=2000,
        modalities=modalities
    )
    
    print(f"Generated multi-modal dataset:")
    print(f"  Available modalities: {data.get_available_modalities()}")
    print(f"  Number of traces: {len(data.power_traces)}")
    print(f"  Trace length: {len(data.power_traces[0])}")
    
    # Display sensor configurations
    print(f"\nSensor Configurations:")
    for modality, config in data.sensor_configs.items():
        print(f"  {modality}:")
        print(f"    Sample Rate: {config.sample_rate:,.0f} Hz")
        print(f"    Resolution: {config.resolution} bits")
        print(f"    Noise Floor: {config.noise_floor}")
        print(f"    Position: {config.spatial_position}")
    
    # Basic statistics
    print(f"\nSignal Statistics:")
    for modality in data.get_available_modalities():
        traces = data.get_trace_data(modality)
        if traces is not None:
            mean_amplitude = np.mean(traces)
            std_amplitude = np.std(traces)
            snr_estimate = np.var(np.mean(traces, axis=0)) / np.mean(np.var(traces, axis=1))
            
            print(f"  {modality}:")
            print(f"    Mean Amplitude: {mean_amplitude:.6f}")
            print(f"    Std Amplitude: {std_amplitude:.6f}")
            print(f"    Estimated SNR: {snr_estimate:.4f}")
    
    return data

def demonstrate_graph_topology_building():
    """Demonstrate graph topology construction for sensor fusion."""
    print("\n" + "="*60)
    print("GRAPH TOPOLOGY CONSTRUCTION")
    print("="*60)
    
    builder = GraphTopologyBuilder()
    
    # Define sensor positions (x, y, z in mm)
    sensor_positions = {
        'power_probe': (0.0, 0.0, 0.0),      # At target
        'em_near': (5.0, 0.0, 2.0),         # 5mm away, 2mm up
        'em_far': (20.0, 0.0, 5.0),         # 20mm away, 5mm up
        'acoustic': (50.0, 0.0, 10.0),       # 50mm away, 10mm up
    }
    
    # Build spatial graph
    edge_index, edge_weight = builder.build_spatial_graph(
        sensor_positions, connection_threshold=25.0
    )
    
    print(f"Spatial Graph:")
    print(f"  Sensors: {list(sensor_positions.keys())}")
    print(f"  Edges: {edge_index.shape[1]}")
    print(f"  Edge connections:")
    
    for i in range(edge_index.shape[1]):
        src_idx = edge_index[0, i].item()
        dst_idx = edge_index[1, i].item()
        weight = edge_weight[i].item()
        
        sensors = list(sensor_positions.keys())
        src_sensor = sensors[src_idx] if src_idx < len(sensors) else f"sensor_{src_idx}"
        dst_sensor = sensors[dst_idx] if dst_idx < len(sensors) else f"sensor_{dst_idx}"
        
        print(f"    {src_sensor} -> {dst_sensor} (weight: {weight:.3f})")
    
    # Build temporal graph
    temporal_edge_index, temporal_edge_weight = builder.build_temporal_graph(
        trace_length=100, temporal_window=3
    )
    
    print(f"\nTemporal Graph (sample):")
    print(f"  Time points: 100")
    print(f"  Temporal window: 3")
    print(f"  Total edges: {temporal_edge_index.shape[1]}")
    print(f"  Sample connections (first 10):")
    
    for i in range(min(10, temporal_edge_index.shape[1])):
        src_time = temporal_edge_index[0, i].item()
        dst_time = temporal_edge_index[1, i].item()
        weight = temporal_edge_weight[i].item()
        print(f"    t={src_time} -> t={dst_time} (weight: {weight:.3f})")
    
    return edge_index, edge_weight

def demonstrate_adaptive_fusion():
    """Demonstrate adaptive multi-modal fusion."""
    print("\n" + "="*60)
    print("ADAPTIVE MULTI-MODAL FUSION")
    print("="*60)
    
    # Generate test data
    data = create_synthetic_multimodal_data(
        n_traces=500,
        trace_length=1000,
        modalities=['power', 'em_near', 'acoustic']
    )
    
    # Initialize analyzer with adaptive fusion
    analyzer = MultiModalSideChannelAnalyzer(
        fusion_method='adaptive',
        device='cpu'
    )
    
    print("Performing adaptive fusion analysis...")
    start_time = time.time()
    
    # Perform analysis
    results = analyzer.analyze_multi_modal(data)
    
    analysis_time = time.time() - start_time
    
    print(f"Analysis completed in {analysis_time:.2f} seconds")
    print(f"\nFusion Results:")
    print(f"  Traces analyzed: {results['n_traces']}")
    print(f"  Modalities used: {results['modalities_used']}")
    print(f"  Fusion method: {results['fusion_method']}")
    print(f"  Fused features shape: {results['fused_features'].shape}")
    
    # Display attention weights if available
    if 'attention_weights' in results:
        print(f"\nLearned Attention Weights:")
        for modality, weight in results['attention_weights'].items():
            if hasattr(weight, 'item'):
                weight_value = weight.item()
            else:
                weight_value = float(weight)
            print(f"  {modality}: {weight_value:.4f}")
    
    # Display quality scores
    if 'quality_scores' in results:
        print(f"\nModality Quality Scores:")
        for modality, scores in results['quality_scores'].items():
            if hasattr(scores, 'shape') and len(scores.shape) > 0:
                avg_quality = np.mean(scores)
                print(f"  {modality}: {avg_quality:.4f}")
    
    # Display fusion quality metrics
    if 'fusion_quality' in results:
        quality = results['fusion_quality']
        print(f"\nFusion Quality Metrics:")
        print(f"  SNR Improvement: {quality['snr_improvement']:.2f}x")
        print(f"  Average Correlation: {quality['average_correlation']:.4f}")
        print(f"  Fusion SNR: {quality['fusion_snr']:.4f}")
        print(f"  Individual SNRs: {[f'{snr:.4f}' for snr in quality['individual_snrs']]}")
    
    return results

def demonstrate_graph_attention_fusion():
    """Demonstrate graph attention network fusion."""
    print("\n" + "="*60)
    print("GRAPH ATTENTION NETWORK FUSION")
    print("="*60)
    
    # Generate test data with spatial correlation
    data = create_synthetic_multimodal_data(
        n_traces=300,
        trace_length=800,
        modalities=['power', 'em_near', 'em_far']
    )
    
    # Initialize analyzer with graph attention
    analyzer = MultiModalSideChannelAnalyzer(
        fusion_method='graph_attention',
        device='cpu'
    )
    
    print("Performing graph attention fusion...")
    start_time = time.time()
    
    # Perform analysis
    results = analyzer.analyze_multi_modal(data)
    
    analysis_time = time.time() - start_time
    
    print(f"Analysis completed in {analysis_time:.2f} seconds")
    print(f"\nGraph Fusion Results:")
    print(f"  Traces analyzed: {results['n_traces']}")
    print(f"  Modalities used: {results['modalities_used']}")
    print(f"  Fusion method: {results['fusion_method']}")
    print(f"  Fused features shape: {results['fused_features'].shape}")
    
    # Analyze fusion effectiveness
    if 'fusion_quality' in results:
        quality = results['fusion_quality']
        print(f"\nGraph Fusion Quality:")
        print(f"  SNR Improvement: {quality['snr_improvement']:.2f}x")
        print(f"  Inter-modal Correlation: {quality['average_correlation']:.4f}")
        
        # Compare individual vs fused performance
        individual_snrs = quality['individual_snrs']
        fusion_snr = quality['fusion_snr']
        
        print(f"\nSNR Comparison:")
        for i, modality in enumerate(results['modalities_used']):
            if i < len(individual_snrs):
                improvement = fusion_snr / individual_snrs[i] if individual_snrs[i] > 0 else 1.0
                print(f"  {modality}: {individual_snrs[i]:.4f} -> {fusion_snr:.4f} ({improvement:.2f}x)")
    
    return results

def demonstrate_fusion_training():
    """Demonstrate training of fusion networks."""
    print("\n" + "="*60)
    print("MULTI-MODAL FUSION NETWORK TRAINING")
    print("="*60)
    
    # Generate training datasets
    print("Generating training datasets...")
    
    training_datasets = []
    training_labels = []
    
    for i in range(5):  # Create 5 different datasets
        data = create_synthetic_multimodal_data(
            n_traces=200,
            trace_length=500,
            modalities=['power', 'em_near', 'acoustic']
        )
        
        # Create synthetic labels (key bytes for this example)
        labels = np.random.randint(0, 256, size=200)
        
        training_datasets.append(data)
        training_labels.append(labels)
    
    print(f"Created {len(training_datasets)} training datasets")
    
    # Initialize analyzer
    analyzer = MultiModalSideChannelAnalyzer(
        fusion_method='adaptive',
        device='cpu'
    )
    
    print("Training fusion network...")
    start_time = time.time()
    
    # Train fusion network
    training_history = analyzer.train_fusion_network(
        training_data=training_datasets,
        labels=training_labels,
        epochs=20,  # Reduced for demo
        learning_rate=1e-3
    )
    
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"\nTraining Results:")
    print(f"  Epochs: {len(training_history['losses'])}")
    print(f"  Final Loss: {training_history['losses'][-1]:.6f}")
    print(f"  Final Accuracy: {training_history['accuracies'][-1]:.3f}")
    
    # Show training progress
    print(f"\nTraining Progress (every 5 epochs):")
    for i in range(0, len(training_history['losses']), 5):
        loss = training_history['losses'][i]
        acc = training_history['accuracies'][i]
        print(f"  Epoch {i+1:2d}: Loss={loss:.6f}, Accuracy={acc:.3f}")
    
    return training_history

def demonstrate_real_world_scenario():
    """Demonstrate realistic multi-modal analysis scenario."""
    print("\n" + "="*60)
    print("REALISTIC MULTI-MODAL ATTACK SCENARIO")
    print("="*60)
    
    print("Scenario: AES implementation analysis with multiple sensors")
    print("  Target: ARM Cortex-M4 running masked AES")
    print("  Sensors: Power probe, Near-field EM probe, Acoustic sensor")
    print("  Countermeasures: 1st-order Boolean masking")
    
    # Generate realistic noisy data
    np.random.seed(42)  # For reproducible results
    
    # Power traces: structured signal with masking
    n_traces = 800
    trace_length = 1500
    
    power_traces = []
    em_traces = []
    acoustic_traces = []
    
    for i in range(n_traces):
        # Simulate AES S-box operation with masking
        key_byte = 0x2B  # Fixed key byte for demo
        plaintext_byte = np.random.randint(0, 256)
        
        # Masked operation: Split into 2 shares
        mask = np.random.randint(0, 256)
        share1 = mask
        share2 = (plaintext_byte ^ key_byte) ^ mask
        
        # Power trace: Leakage from both shares at different times
        power_trace = np.random.normal(0, 0.1, trace_length)
        
        # Share 1 leakage (early in trace)
        hw1 = bin(share1).count('1')
        leak_pos1 = 500 + np.random.randint(-20, 21)
        for j in range(max(0, leak_pos1-30), min(trace_length, leak_pos1+30)):
            power_trace[j] += 0.02 * hw1 * np.exp(-((j-leak_pos1)**2) / 200)
        
        # Share 2 leakage (later in trace)  
        hw2 = bin(share2).count('1')
        leak_pos2 = 1000 + np.random.randint(-20, 21)
        for j in range(max(0, leak_pos2-30), min(trace_length, leak_pos2+30)):
            power_trace[j] += 0.02 * hw2 * np.exp(-((j-leak_pos2)**2) / 200)
        
        power_traces.append(power_trace)
        
        # EM trace: Correlated but noisier
        em_trace = 0.6 * power_trace + np.random.normal(0, 0.15, trace_length)
        em_traces.append(em_trace)
        
        # Acoustic trace: Much noisier, different frequency content
        acoustic_base = np.random.normal(0, 0.2, trace_length)
        # Add some correlation at operation times
        acoustic_trace = acoustic_base
        if leak_pos1 < trace_length:
            acoustic_trace[leak_pos1] += 0.01 * hw1
        if leak_pos2 < trace_length:
            acoustic_trace[leak_pos2] += 0.01 * hw2
        
        acoustic_traces.append(acoustic_trace)
    
    # Create multi-modal data
    data = MultiModalData(
        power_traces=np.array(power_traces),
        em_near_traces=np.array(em_traces),
        acoustic_traces=np.array(acoustic_traces)
    )
    
    # Add sensor configs
    data.sensor_configs = {
        'power': SensorConfig(
            name='power_probe',
            sample_rate=1e6,
            resolution=12,
            noise_floor=0.001,
            frequency_range=(1e3, 500e3),
            spatial_position=(0, 0, 0)
        ),
        'em_near': SensorConfig(
            name='em_near_field',
            sample_rate=1e6,
            resolution=10,
            noise_floor=0.005,
            frequency_range=(10e3, 1e6),
            spatial_position=(5, 0, 2)
        ),
        'acoustic': SensorConfig(
            name='acoustic_sensor',
            sample_rate=48e3,
            resolution=16,
            noise_floor=0.01,
            frequency_range=(100, 20e3),
            spatial_position=(50, 0, 10)
        )
    }
    
    print(f"\nGenerated realistic dataset:")
    print(f"  Traces: {n_traces}")
    print(f"  Length: {trace_length} samples")
    print(f"  Modalities: {data.get_available_modalities()}")
    
    # Analyze with different fusion methods
    methods = ['adaptive', 'graph_attention']
    results = {}
    
    for method in methods:
        print(f"\nTesting {method} fusion...")
        
        analyzer = MultiModalSideChannelAnalyzer(
            fusion_method=method,
            device='cpu'
        )
        
        start_time = time.time()
        result = analyzer.analyze_multi_modal(data)
        analysis_time = time.time() - start_time
        
        results[method] = result
        
        print(f"  Analysis time: {analysis_time:.2f}s")
        if 'fusion_quality' in result:
            quality = result['fusion_quality']
            print(f"  SNR improvement: {quality['snr_improvement']:.2f}x")
            print(f"  Fusion SNR: {quality['fusion_snr']:.4f}")
    
    # Compare methods
    print(f"\nMethod Comparison:")
    print(f"{'Method':<20} {'SNR Improvement':<15} {'Fusion SNR':<12}")
    print("-" * 50)
    
    for method, result in results.items():
        if 'fusion_quality' in result:
            quality = result['fusion_quality']
            snr_imp = quality['snr_improvement']
            fusion_snr = quality['fusion_snr']
            print(f"{method:<20} {snr_imp:<15.2f} {fusion_snr:<12.4f}")
    
    return results

def main():
    """Run all multi-modal fusion demonstrations."""
    print("NEURAL OPERATOR CRYPTANALYSIS - MULTI-MODAL FUSION DEMONSTRATIONS")
    print("=" * 70)
    
    try:
        # Synthetic data generation
        synthetic_data = demonstrate_synthetic_data_generation()
        
        # Graph topology building
        edge_info = demonstrate_graph_topology_building()
        
        # Adaptive fusion
        adaptive_results = demonstrate_adaptive_fusion()
        
        # Graph attention fusion
        graph_results = demonstrate_graph_attention_fusion()
        
        # Fusion training
        training_results = demonstrate_fusion_training()
        
        # Real-world scenario
        scenario_results = demonstrate_real_world_scenario()
        
        print("\n" + "="*70)
        print("ALL MULTI-MODAL FUSION DEMONSTRATIONS COMPLETED")
        print("="*70)
        
        print(f"\nSummary:")
        print(f"  Synthetic data: Generated with {len(synthetic_data.get_available_modalities())} modalities")
        print(f"  Adaptive fusion: {adaptive_results['n_traces']} traces analyzed")
        print(f"  Graph attention: {graph_results['n_traces']} traces analyzed")
        print(f"  Network training: {len(training_results['losses'])} epochs completed")
        print(f"  Realistic scenario: {len(scenario_results)} methods compared")
        
        # Performance summary
        if 'fusion_quality' in adaptive_results:
            adaptive_snr = adaptive_results['fusion_quality']['snr_improvement']
            print(f"  Best SNR improvement: {adaptive_snr:.2f}x (adaptive fusion)")
        
    except Exception as e:
        print(f"Error during demonstrations: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())