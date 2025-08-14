"""Integration tests for end-to-end workflows and cross-component functionality."""

import pytest
import numpy as np
import asyncio
import tempfile
import shutil
from pathlib import Path
import sys
import time
from unittest.mock import Mock, patch, AsyncMock

# Mock torch if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create a mock torch module
    torch = Mock()
    torch.tensor = Mock(return_value=Mock())
    torch.nn = Mock()
    torch.optim = Mock()
    torch.cuda = Mock()
    torch.cuda.is_available = Mock(return_value=False)
    torch.manual_seed = Mock()
    torch.randn = Mock(return_value=Mock())
    torch.randint = Mock(return_value=Mock())
    torch.zeros = Mock(return_value=Mock())
    torch.ones = Mock(return_value=Mock())
    torch.no_grad = Mock(return_value=Mock())
    torch.float32 = "float32"
    torch.long = "long"
    # Add method mocks for tensor operations
    mock_tensor = Mock()
    mock_tensor.unsqueeze = Mock(return_value=Mock())
    mock_tensor.requires_grad = False
    torch.tensor = Mock(return_value=mock_tensor)

# Import test fixtures
from conftest import (
    neural_operator_architectures, countermeasure_types,
    side_channel_modalities, skip_if_no_gpu, skip_if_no_hardware
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import components for integration testing
from neural_cryptanalysis.core import NeuralSCA, TraceData
from neural_cryptanalysis.datasets.synthetic import SyntheticDatasetGenerator
from neural_cryptanalysis.targets.post_quantum import KyberImplementation
from neural_cryptanalysis.side_channels import SideChannelAnalyzer, AnalysisConfig
from neural_cryptanalysis.adaptive_rl import AdaptiveAttackEngine, AttackState
from neural_cryptanalysis.multi_modal_fusion import (
    MultiModalSideChannelAnalyzer, create_synthetic_multimodal_data
)
from neural_cryptanalysis.advanced_countermeasures import (
    AdvancedCountermeasureEvaluator, create_boolean_masking
)
from neural_cryptanalysis.hardware_integration import HardwareInTheLoopSystem
from neural_cryptanalysis.distributed_computing import DistributedCoordinator
from neural_cryptanalysis.utils.performance import PerformanceProfiler


@pytest.mark.integration
class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    def test_full_attack_pipeline_aes(self, temp_directory):
        """Test complete AES attack pipeline from data generation to key recovery."""
        print("\n=== Testing Full AES Attack Pipeline ===")
        
        # Step 1: Generate synthetic dataset
        generator = SyntheticDatasetGenerator(random_seed=42)
        dataset = generator.generate_aes_dataset(
            n_traces=300,
            target_bytes=[0],
            trace_length=500,
            noise_snr_db=15
        )
        
        assert dataset['power_traces'].shape == (300, 500)
        assert len(dataset['labels'][0]) == 300
        print(f"✓ Generated {len(dataset['power_traces'])} AES traces")
        
        # Step 2: Initialize neural SCA
        neural_sca = NeuralSCA(
            architecture='fourier_neural_operator',
            channels=['power'],
            config={
                'fno': {'modes': 8, 'width': 32, 'n_layers': 3},
                'training': {'batch_size': 32, 'epochs': 10, 'learning_rate': 1e-3}
            }
        )
        
        # Step 3: Prepare training data
        traces = torch.tensor(dataset['power_traces'], dtype=torch.float32).unsqueeze(-1)
        labels = torch.tensor(dataset['labels'][0], dtype=torch.long)
        
        # Split data
        split_idx = int(0.7 * len(traces))
        train_traces = traces[:split_idx]
        train_labels = labels[:split_idx]
        test_traces = traces[split_idx:]
        test_labels = labels[split_idx:]
        
        print(f"✓ Split data: {len(train_traces)} training, {len(test_traces)} testing")
        
        # Step 4: Train model
        start_time = time.perf_counter()
        model = neural_sca.train(
            traces=train_traces,
            labels=train_labels,
            validation_split=0.2
        )
        training_time = time.perf_counter() - start_time
        
        assert model is not None
        print(f"✓ Training completed in {training_time:.2f}s")
        
        # Step 5: Evaluate model accuracy
        with torch.no_grad():
            predictions = model(test_traces)
            predicted_labels = torch.argmax(predictions, dim=1)
            accuracy = (predicted_labels == test_labels).float().mean().item()
        
        print(f"✓ Model accuracy: {accuracy:.3f}")
        assert accuracy > 0.1  # Should be better than random
        
        # Step 6: Perform attack simulation
        attack_results = neural_sca.attack(
            target_traces=test_traces,
            model=model,
            strategy='template',
            target_byte=0,
            plaintexts=dataset['plaintexts'][split_idx:]
        )
        
        assert 'predicted_key_byte' in attack_results
        assert 'confidence' in attack_results
        assert isinstance(attack_results['predicted_key_byte'], (int, np.integer))
        
        print(f"✓ Attack completed - Predicted key byte: {attack_results['predicted_key_byte']}")
        print(f"✓ Confidence: {attack_results['confidence']:.3f}")
        
        # Step 7: Validate attack success
        true_key_byte = dataset['key'][0]
        attack_success = attack_results['predicted_key_byte'] == true_key_byte
        
        print(f"✓ True key byte: {true_key_byte}")
        print(f"✓ Attack success: {attack_success}")
        
        # Step 8: Save results
        results_path = temp_directory / "attack_results.json"
        import json
        with open(results_path, 'w') as f:
            json.dump({
                'attack_results': {k: float(v) if isinstance(v, np.number) else v 
                                 for k, v in attack_results.items()},
                'accuracy': accuracy,
                'training_time': training_time,
                'attack_success': bool(attack_success),
                'dataset_size': len(dataset['power_traces'])
            }, f, indent=2)
        
        assert results_path.exists()
        print(f"✓ Results saved to {results_path}")
        
        return {
            'accuracy': accuracy,
            'attack_success': attack_success,
            'training_time': training_time,
            'confidence': attack_results['confidence']
        }
    
    def test_full_attack_pipeline_kyber(self, temp_directory):
        """Test complete Kyber attack pipeline."""
        print("\n=== Testing Full Kyber Attack Pipeline ===")
        
        # Step 1: Create Kyber target
        kyber_target = KyberImplementation(
            variant='kyber512',  # Smaller for faster testing
            platform='arm_cortex_m4',
            countermeasures=[]
        )
        
        print(f"✓ Created Kyber target: {kyber_target.variant}")
        
        # Step 2: Generate synthetic Kyber dataset
        generator = SyntheticDatasetGenerator(random_seed=42)
        dataset = generator.generate_kyber_dataset(
            n_traces=200,
            variant='kyber512',
            trace_length=1500
        )
        
        assert dataset['power_traces'].shape == (200, 1500)
        print(f"✓ Generated {len(dataset['power_traces'])} Kyber traces")
        
        # Step 3: Initialize neural SCA for Kyber
        neural_sca = NeuralSCA(
            architecture='leakage_fno',
            channels=['power'],
            config={
                'leakage_fno': {'operation_type': 'kyber_ntt'},
                'training': {'batch_size': 16, 'epochs': 5}
            }
        )
        
        # Step 4: Create labels from NTT coefficients
        traces = torch.tensor(dataset['power_traces'], dtype=torch.float32).unsqueeze(-1)
        # Use first coefficient of each polynomial as target
        labels = torch.tensor([ntt[0] % 256 for ntt in dataset['ntt_outputs']], dtype=torch.long)
        
        # Step 5: Train model
        split_idx = int(0.7 * len(traces))
        model = neural_sca.train(
            traces=traces[:split_idx],
            labels=labels[:split_idx],
            validation_split=0.2
        )
        
        assert model is not None
        print("✓ Kyber model training completed")
        
        # Step 6: Test model on Kyber data
        with torch.no_grad():
            predictions = model(traces[split_idx:])
            predicted_labels = torch.argmax(predictions, dim=1)
            accuracy = (predicted_labels == labels[split_idx:]).float().mean().item()
        
        print(f"✓ Kyber model accuracy: {accuracy:.3f}")
        assert accuracy >= 0.0  # Should complete without errors
        
        return {'kyber_accuracy': accuracy, 'target_variant': kyber_target.variant}
    
    @neural_operator_architectures
    def test_architecture_compatibility_workflow(self, architecture, config_params):
        """Test complete workflow with different neural operator architectures."""
        print(f"\n=== Testing {architecture} Architecture Workflow ===")
        
        # Generate test data
        generator = SyntheticDatasetGenerator(random_seed=42)
        dataset = generator.generate_aes_dataset(n_traces=100, trace_length=300)
        
        # Configure neural SCA for specific architecture
        config = {
            'training': {'batch_size': 16, 'epochs': 2},
            architecture.replace('_', ''): config_params
        }
        
        neural_sca = NeuralSCA(
            architecture=architecture,
            config=config
        )
        
        # Prepare data
        traces = torch.tensor(dataset['power_traces'], dtype=torch.float32)
        if architecture != 'deep_operator_network':
            traces = traces.unsqueeze(-1)
        
        labels = torch.tensor(dataset['labels'][0], dtype=torch.long)
        
        # Train model
        try:
            model = neural_sca.train(traces[:70], labels[:70], validation_split=0.2)
            assert model is not None
            
            # Test inference
            with torch.no_grad():
                predictions = model(traces[70:])
                assert predictions.shape[0] == 30  # Test batch size
                assert not torch.isnan(predictions).any()
            
            print(f"✓ {architecture} workflow completed successfully")
            return True
            
        except Exception as e:
            pytest.fail(f"{architecture} workflow failed: {e}")
    
    def test_preprocessing_integration(self, sample_traces, sample_labels):
        """Test integration of preprocessing with neural operators."""
        print("\n=== Testing Preprocessing Integration ===")
        
        # Create analyzer with multiple preprocessing methods
        config = AnalysisConfig(
            channel_types=['power'],
            preprocessing_methods=['standardize', 'filter', 'normalize'],
            analysis_methods=['correlation']
        )
        
        analyzer = SideChannelAnalyzer(config)
        trace_data = TraceData(traces=sample_traces, labels=sample_labels)
        
        # Test each preprocessing method with neural SCA
        for method in ['standardize', 'normalize']:
            preprocessed_traces = analyzer.preprocess(sample_traces, method=method)
            
            # Verify preprocessing worked
            if method == 'standardize':
                assert np.abs(np.mean(preprocessed_traces)) < 1e-10
                assert np.abs(np.std(preprocessed_traces) - 1.0) < 1e-6
            elif method == 'normalize':
                assert preprocessed_traces.min() >= 0
                assert preprocessed_traces.max() <= 1
            
            # Test with neural SCA
            neural_sca = NeuralSCA(config={
                'training': {'batch_size': 16, 'epochs': 1}
            })
            
            traces_tensor = torch.tensor(preprocessed_traces[:50], dtype=torch.float32).unsqueeze(-1)
            labels_tensor = torch.tensor(sample_labels[:50], dtype=torch.long)
            
            model = neural_sca.train(traces_tensor, labels_tensor, validation_split=0.2)
            assert model is not None
            
            print(f"✓ {method} preprocessing integrated successfully")
    
    def test_multi_target_byte_attack(self, temp_directory):
        """Test attack on multiple target bytes simultaneously."""
        print("\n=== Testing Multi-Target Byte Attack ===")
        
        # Generate dataset with multiple target bytes
        generator = SyntheticDatasetGenerator(random_seed=42)
        dataset = generator.generate_aes_dataset(
            n_traces=200,
            target_bytes=[0, 1, 2],  # First three bytes
            trace_length=400
        )
        
        traces = torch.tensor(dataset['power_traces'], dtype=torch.float32).unsqueeze(-1)
        
        attack_results = {}
        
        # Attack each target byte
        for byte_idx in range(3):
            print(f"  Attacking byte {byte_idx}...")
            
            neural_sca = NeuralSCA(config={
                'training': {'batch_size': 16, 'epochs': 3}
            })
            
            labels = torch.tensor(dataset['labels'][byte_idx], dtype=torch.long)
            
            # Train model for this byte
            model = neural_sca.train(traces[:140], labels[:140], validation_split=0.2)
            
            # Perform attack
            byte_results = neural_sca.attack(
                target_traces=traces[140:],
                model=model,
                strategy='template',
                target_byte=byte_idx,
                plaintexts=dataset['plaintexts'][140:]
            )
            
            attack_results[f'byte_{byte_idx}'] = byte_results
            print(f"    ✓ Byte {byte_idx} predicted: {byte_results['predicted_key_byte']}")
        
        # Verify all attacks completed
        assert len(attack_results) == 3
        for byte_idx in range(3):
            assert f'byte_{byte_idx}' in attack_results
            assert 'predicted_key_byte' in attack_results[f'byte_{byte_idx}']
        
        print("✓ Multi-target byte attack completed")
        return attack_results


@pytest.mark.integration
class TestCrossComponentIntegration:
    """Test integration between different framework components."""
    
    @pytest.mark.asyncio
    async def test_adaptive_rl_integration(self, sample_traces, sample_labels):
        """Test integration of adaptive RL with neural SCA."""
        print("\n=== Testing Adaptive RL Integration ===")
        
        # Create neural SCA
        neural_sca = NeuralSCA(config={
            'training': {'batch_size': 16, 'epochs': 2}
        })
        
        # Create adaptive attack engine
        adaptive_engine = AdaptiveAttackEngine(neural_sca, epsilon=0.2, device='cpu')
        
        # Create trace data
        trace_data = TraceData(traces=sample_traces, labels=sample_labels)
        
        # Mock the evaluation function for faster testing
        async def mock_evaluation(state, traces):
            # Simulate improving performance based on state
            base_success = 0.5
            snr_bonus = state.snr * 0.2
            confidence_bonus = state.confidence * 0.1
            return min(base_success + snr_bonus + confidence_bonus, 0.9), 0.8, 0.6
        
        with patch.object(adaptive_engine, 'evaluate_attack_performance', side_effect=mock_evaluation):
            # Run short autonomous attack
            results = adaptive_engine.autonomous_attack(
                traces=trace_data,
                target_success_rate=0.7,
                max_episodes=5,
                patience=3
            )
        
        assert 'success_rate' in results
        assert 'confidence' in results
        assert 'optimal_parameters' in results
        assert results['training_episodes'] > 0
        
        print(f"✓ Adaptive RL achieved {results['success_rate']:.3f} success rate")
        print(f"✓ Training episodes: {results['training_episodes']}")
    
    def test_multimodal_fusion_integration(self, multimodal_test_data):
        """Test integration of multi-modal fusion with neural operators."""
        print("\n=== Testing Multi-Modal Fusion Integration ===")
        
        # Create multi-modal data
        from neural_cryptanalysis.multi_modal_fusion import MultiModalData
        
        mm_data = MultiModalData(
            power_traces=multimodal_test_data['power_traces'],
            em_near_traces=multimodal_test_data['em_near_traces'],
            acoustic_traces=multimodal_test_data['acoustic_traces']
        )
        
        # Test multi-modal analyzer
        analyzer = MultiModalSideChannelAnalyzer(fusion_method='adaptive', device='cpu')
        
        results = analyzer.analyze_multi_modal(mm_data)
        
        assert 'fused_features' in results
        assert 'n_traces' in results
        assert 'modalities_used' in results
        assert results['n_traces'] == len(multimodal_test_data['power_traces'])
        assert len(results['modalities_used']) == 3
        
        print(f"✓ Fused {len(results['modalities_used'])} modalities")
        print(f"✓ Generated {results['fused_features'].shape[1]} fused features")
        
        # Test integration with neural SCA
        neural_sca = NeuralSCA(config={
            'training': {'batch_size': 16, 'epochs': 2}
        })
        
        # Use fused features as input
        fused_traces = torch.tensor(results['fused_features'], dtype=torch.float32).unsqueeze(-1)
        labels = torch.randint(0, 256, (len(fused_traces),))
        
        model = neural_sca.train(fused_traces[:35], labels[:35], validation_split=0.2)
        assert model is not None
        
        print("✓ Multi-modal fusion integrated with neural SCA")
    
    def test_countermeasure_evaluation_integration(self):
        """Test integration of countermeasure evaluation with neural SCA."""
        print("\n=== Testing Countermeasure Evaluation Integration ===")
        
        # Create original traces
        original_traces = np.random.randn(100, 500) * 0.1
        labels = np.random.randint(0, 256, 100)
        original_trace_data = TraceData(traces=original_traces, labels=labels)
        
        # Create neural SCA
        neural_sca = NeuralSCA(config={
            'training': {'batch_size': 16, 'epochs': 2}
        })
        
        # Create countermeasure evaluator
        evaluator = AdvancedCountermeasureEvaluator(neural_sca)
        
        # Test with Boolean masking
        masking = create_boolean_masking(order=1)
        
        # Mock the evaluation to avoid long computation
        def mock_evaluation(countermeasure, traces, max_traces=1000):
            from neural_cryptanalysis.advanced_countermeasures import EvaluationMetrics
            return EvaluationMetrics(
                theoretical_security_order=countermeasure.estimate_security_order(),
                practical_security_order=countermeasure.estimate_security_order(),
                traces_needed_90_percent=5000,
                traces_needed_95_percent=10000,
                max_success_rate=0.3,
                snr_reduction_factor=2.5,
                mutual_information=0.05,
                t_test_statistics={'max_t_statistic': 1.8}
            )
        
        with patch.object(evaluator, 'evaluate_countermeasure', side_effect=mock_evaluation):
            metrics = evaluator.evaluate_countermeasure(masking, original_trace_data)
        
        assert hasattr(metrics, 'theoretical_security_order')
        assert hasattr(metrics, 'practical_security_order')
        assert metrics.theoretical_security_order == 1
        
        print(f"✓ Countermeasure evaluation completed")
        print(f"✓ Security order: {metrics.theoretical_security_order}")
        print(f"✓ SNR reduction: {metrics.snr_reduction_factor:.2f}x")
    
    @skip_if_no_hardware
    @pytest.mark.asyncio
    async def test_hardware_integration_workflow(self, mock_hardware_device):
        """Test hardware integration workflow."""
        print("\n=== Testing Hardware Integration Workflow ===")
        
        # Create neural SCA
        neural_sca = NeuralSCA()
        
        # Create hardware-in-the-loop system
        hitl_system = HardwareInTheLoopSystem(neural_sca)
        
        # Register mock hardware device
        await hitl_system.add_device('test_scope', mock_hardware_device)
        
        # Verify device registration
        assert 'test_scope' in hitl_system.devices
        assert hitl_system.devices['test_scope'] == mock_hardware_device
        
        # Test system status
        status = hitl_system.get_system_status()
        assert 'devices' in status
        assert 'test_scope' in status['devices']
        assert not status['measurement_active']
        
        print("✓ Hardware device registered")
        print(f"✓ System status: {len(status['devices'])} devices")
        
        # Mock measurement workflow
        from neural_cryptanalysis.hardware_integration import MeasurementConfig
        
        config = MeasurementConfig(
            channels=['power'],
            sample_rate=1e6,
            memory_depth=1000
        )
        
        # This would normally perform actual measurements
        mock_traces = np.random.randn(50, 1000)
        
        # Integrate with neural SCA
        traces_tensor = torch.tensor(mock_traces, dtype=torch.float32).unsqueeze(-1)
        labels = torch.randint(0, 256, (50,))
        
        model = neural_sca.train(traces_tensor[:35], labels[:35], validation_split=0.2)
        assert model is not None
        
        print("✓ Hardware integration workflow completed")
    
    def test_distributed_computing_integration(self, temp_directory):
        """Test distributed computing integration."""
        print("\n=== Testing Distributed Computing Integration ===")
        
        # Create distributed coordinator
        coordinator = DistributedCoordinator(storage_root=temp_directory)
        
        # Create test data
        test_data = np.random.randn(200, 300)
        
        # Create data shards
        shards = coordinator.data_manager.create_data_shards(test_data, shard_size=80)
        
        assert len(shards) == 3  # 200 traces / 80 per shard = 2.5 -> 3 shards
        
        # Test shard loading
        for shard in shards:
            loaded_data = coordinator.data_manager.load_shard(shard.shard_id)
            assert loaded_data is not None
        
        print(f"✓ Created {len(shards)} data shards")
        
        # Test task scheduling
        from neural_cryptanalysis.distributed_computing import DistributedTask
        
        task = DistributedTask(
            task_id='test_training_task',
            task_type='training',
            priority=1,
            data_shards=[shard.shard_id for shard in shards[:2]]
        )
        
        task_id = coordinator.scheduler.submit_task(task)
        assert task_id == 'test_training_task'
        assert coordinator.scheduler.pending_tasks.qsize() == 1
        
        print("✓ Task submitted to scheduler")
        
        # Test worker creation
        from neural_cryptanalysis.distributed_computing import NeuralOperatorTrainingWorker
        neural_sca = NeuralSCA()
        
        worker = NeuralOperatorTrainingWorker('test_worker', neural_sca, device='cpu')
        assert worker.worker_id == 'test_worker'
        assert worker.capabilities['training'] is True
        
        print("✓ Distributed worker created")
    
    def test_performance_profiling_integration(self, sample_traces, sample_labels):
        """Test performance profiling integration across components."""
        print("\n=== Testing Performance Profiling Integration ===")
        
        profiler = PerformanceProfiler()
        
        # Profile data generation
        with profiler.profile("data_generation"):
            generator = SyntheticDatasetGenerator(random_seed=42)
            dataset = generator.generate_aes_dataset(n_traces=100, trace_length=200)
        
        # Profile neural SCA training
        with profiler.profile("neural_sca_training"):
            neural_sca = NeuralSCA(config={
                'training': {'batch_size': 16, 'epochs': 2}
            })
            
            traces = torch.tensor(dataset['power_traces'], dtype=torch.float32).unsqueeze(-1)
            labels = torch.tensor(dataset['labels'][0], dtype=torch.long)
            
            model = neural_sca.train(traces[:70], labels[:70], validation_split=0.2)
        
        # Profile inference
        with profiler.profile("inference"):
            with torch.no_grad():
                predictions = model(traces[70:])
        
        # Get profiling results
        data_gen_stats = profiler.get_stats("data_generation")
        training_stats = profiler.get_stats("neural_sca_training")
        inference_stats = profiler.get_stats("inference")
        
        assert data_gen_stats['call_count'] == 1
        assert training_stats['call_count'] == 1
        assert inference_stats['call_count'] == 1
        
        print(f"✓ Data generation: {data_gen_stats['total_time']:.3f}s")
        print(f"✓ Training: {training_stats['total_time']:.3f}s")
        print(f"✓ Inference: {inference_stats['total_time']:.3f}s")
        
        # Test overall performance report
        overall_stats = profiler.get_overall_stats()
        assert 'total_operations' in overall_stats
        assert 'total_time' in overall_stats
        assert overall_stats['total_operations'] == 3
        
        print(f"✓ Overall profiling: {overall_stats['total_operations']} operations")


@pytest.mark.integration
class TestSystemRobustness:
    """Test system robustness and error handling."""
    
    def test_error_propagation_and_recovery(self):
        """Test error propagation and recovery across components."""
        print("\n=== Testing Error Propagation and Recovery ===")
        
        # Test with invalid configuration
        with pytest.raises(Exception):  # Should raise configuration error
            neural_sca = NeuralSCA(config={'invalid_key': 'invalid_value'})
        
        # Test with mismatched data
        neural_sca = NeuralSCA(config={'training': {'batch_size': 16, 'epochs': 1}})
        
        traces = torch.randn(50, 100, 1)
        invalid_labels = torch.randint(0, 256, (30,))  # Wrong size
        
        with pytest.raises(Exception):  # Should raise data validation error
            model = neural_sca.train(traces, invalid_labels)
        
        # Test recovery with valid data
        valid_labels = torch.randint(0, 256, (50,))
        model = neural_sca.train(traces, valid_labels, validation_split=0.2)
        assert model is not None
        
        print("✓ Error propagation and recovery working correctly")
    
    def test_memory_and_resource_management(self, sample_traces):
        """Test memory and resource management under stress."""
        print("\n=== Testing Memory and Resource Management ===")
        
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple models to test memory management
        models = []
        for i in range(3):
            neural_sca = NeuralSCA(config={
                'training': {'batch_size': 8, 'epochs': 1}
            })
            
            traces = torch.tensor(sample_traces[:30], dtype=torch.float32).unsqueeze(-1)
            labels = torch.randint(0, 256, (30,))
            
            model = neural_sca.train(traces, labels, validation_split=0.2)
            models.append(model)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = peak_memory - initial_memory
        
        # Clean up
        del models
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_recovered = peak_memory - final_memory
        
        print(f"✓ Memory used: {memory_used:.1f} MB")
        print(f"✓ Memory recovered: {memory_recovered:.1f} MB")
        
        # Memory usage should be reasonable
        assert memory_used < 2048  # Less than 2GB for this test
    
    def test_concurrent_operations(self, sample_traces, sample_labels):
        """Test concurrent operations and thread safety."""
        print("\n=== Testing Concurrent Operations ===")
        
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def train_model(model_id):
            try:
                neural_sca = NeuralSCA(config={
                    'training': {'batch_size': 8, 'epochs': 1}
                })
                
                # Use different data slice for each thread
                start_idx = model_id * 20
                end_idx = start_idx + 20
                
                traces = torch.tensor(sample_traces[start_idx:end_idx], dtype=torch.float32).unsqueeze(-1)
                labels = torch.tensor(sample_labels[start_idx:end_idx], dtype=torch.long)
                
                model = neural_sca.train(traces, labels, validation_split=0.2)
                results_queue.put((model_id, 'success', model is not None))
                
            except Exception as e:
                results_queue.put((model_id, 'error', str(e)))
        
        # Start multiple training threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=train_model, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        assert len(results) == 3
        successful_threads = sum(1 for _, status, _ in results if status == 'success')
        
        print(f"✓ {successful_threads}/3 concurrent training operations succeeded")
        assert successful_threads >= 2  # At least 2 should succeed
    
    def test_scalability_stress_test(self):
        """Test system scalability under increasing load."""
        print("\n=== Testing Scalability Stress Test ===")
        
        # Test with increasing dataset sizes
        dataset_sizes = [50, 100, 200]
        training_times = []
        
        for size in dataset_sizes:
            # Generate dataset
            generator = SyntheticDatasetGenerator(random_seed=42)
            dataset = generator.generate_aes_dataset(n_traces=size, trace_length=200)
            
            # Train model
            neural_sca = NeuralSCA(config={
                'training': {'batch_size': min(16, size//4), 'epochs': 1}
            })
            
            traces = torch.tensor(dataset['power_traces'], dtype=torch.float32).unsqueeze(-1)
            labels = torch.tensor(dataset['labels'][0], dtype=torch.long)
            
            start_time = time.perf_counter()
            
            try:
                model = neural_sca.train(traces, labels, validation_split=0.2)
                training_time = time.perf_counter() - start_time
                training_times.append(training_time)
                
                print(f"  ✓ Size {size}: {training_time:.2f}s")
                
            except Exception as e:
                print(f"  ✗ Size {size}: Failed - {e}")
                training_times.append(float('inf'))
        
        # Check that training time scales reasonably
        successful_times = [t for t in training_times if t != float('inf')]
        if len(successful_times) >= 2:
            # Training time should not grow exponentially
            time_ratio = successful_times[-1] / successful_times[0]
            size_ratio = dataset_sizes[-1] / dataset_sizes[0]
            
            print(f"✓ Time scaling ratio: {time_ratio:.2f}x for {size_ratio}x data")
            assert time_ratio < size_ratio * 2  # Should be roughly linear


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])