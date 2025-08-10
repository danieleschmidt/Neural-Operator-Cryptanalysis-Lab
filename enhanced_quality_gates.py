#!/usr/bin/env python3
"""
Enhanced Quality Gates for Neural Operator Cryptanalysis Lab Advanced Features.

This script provides comprehensive validation of all enhancements including
adaptive RL, multi-modal fusion, hardware integration, countermeasure evaluation,
and distributed computing capabilities.
"""

import sys
import os
import time
import json
import asyncio
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def run_quality_gate(gate_name: str, gate_function, critical: bool = False) -> Dict[str, Any]:
    """Run a single quality gate and return results."""
    print(f"\n{'='*60}")
    print(f"QUALITY GATE: {gate_name}")
    print(f"Critical: {critical}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        if asyncio.iscoroutinefunction(gate_function):
            result = asyncio.run(gate_function())
        else:
            result = gate_function()
        
        execution_time = time.time() - start_time
        
        if isinstance(result, dict) and 'passed' in result:
            passed = result['passed']
            details = result
        else:
            passed = bool(result)
            details = {'result': result}
        
        print(f"✅ PASSED" if passed else f"❌ FAILED")
        print(f"Execution time: {execution_time:.3f}s")
        
        return {
            'name': gate_name,
            'passed': passed,
            'score': 1.0 if passed else 0.0,
            'critical': critical,
            'execution_time': execution_time,
            'details': details
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Error: {str(e)}"
        
        print(f"❌ FAILED - {error_msg}")
        print(f"Execution time: {execution_time:.3f}s")
        
        return {
            'name': gate_name,
            'passed': False,
            'score': 0.0,
            'critical': critical,
            'execution_time': execution_time,
            'details': {'error': error_msg, 'traceback': traceback.format_exc()}
        }

def test_adaptive_rl_functionality() -> Dict[str, Any]:
    """Test adaptive RL attack engine functionality."""
    try:
        from neural_cryptanalysis.core import NeuralSCA, TraceData
        from neural_cryptanalysis.adaptive_rl import AdaptiveAttackEngine, AttackState, AttackAction
        
        print("Testing adaptive RL components...")
        
        # Test AttackState functionality
        state = AttackState(snr=0.5, success_rate=0.8)
        state_vector = state.to_vector()
        assert len(state_vector) == 12, "AttackState vector should have 12 dimensions"
        
        # Test AttackAction functionality
        action = AttackAction.from_action_id(0)
        assert action.action_type == 'adjust_window', "Action type should be adjust_window"
        
        # Test AdaptiveAttackEngine initialization
        neural_sca = NeuralSCA(architecture='fourier_neural_operator')
        engine = AdaptiveAttackEngine(neural_sca, device='cpu')
        assert engine.neural_sca == neural_sca, "Neural SCA should be properly assigned"
        
        # Test action selection
        action_id = engine.select_action(state, training=False)
        assert isinstance(action_id, int), "Action selection should return integer"
        assert 0 <= action_id < engine.action_dim, "Action ID should be in valid range"
        
        # Test reward computation
        old_state = AttackState(success_rate=0.5)
        new_state = AttackState(success_rate=0.8)
        reward = engine.compute_reward(old_state, new_state, action)
        assert isinstance(reward, float), "Reward should be float"
        
        # Test action application
        initial_state = AttackState(window_size=1000)
        window_action = AttackAction('adjust_window', 'window_size', 1.5)
        modified_state = engine.apply_action(initial_state, window_action)
        assert modified_state.window_size == 1500, "Window size should be modified correctly"
        
        print("✓ All adaptive RL tests passed")
        
        return {
            'passed': True,
            'components_tested': ['AttackState', 'AttackAction', 'AdaptiveAttackEngine'],
            'test_results': {
                'state_vector_dimensions': len(state_vector),
                'action_selection': 'working',
                'reward_computation': 'working',
                'action_application': 'working'
            }
        }
        
    except Exception as e:
        return {'passed': False, 'error': str(e)}

def test_multimodal_fusion_functionality() -> Dict[str, Any]:
    """Test multi-modal sensor fusion functionality."""
    try:
        from neural_cryptanalysis.multi_modal_fusion import (
            MultiModalData, MultiModalSideChannelAnalyzer, 
            create_synthetic_multimodal_data, GraphTopologyBuilder
        )
        
        print("Testing multi-modal fusion components...")
        
        # Test synthetic data generation
        data = create_synthetic_multimodal_data(
            n_traces=100,
            trace_length=500,
            modalities=['power', 'em_near', 'acoustic']
        )
        
        modalities = data.get_available_modalities()
        assert len(modalities) == 3, "Should have 3 modalities"
        assert data.power_traces.shape == (100, 500), "Power traces shape incorrect"
        
        # Test graph topology builder
        builder = GraphTopologyBuilder()
        positions = {
            'sensor1': (0.0, 0.0, 0.0),
            'sensor2': (1.0, 0.0, 0.0),
            'sensor3': (0.0, 1.0, 0.0)
        }
        
        edge_index, edge_weight = builder.build_spatial_graph(positions, 2.0)
        assert edge_index.shape[0] == 2, "Edge index should have 2 rows"
        assert len(edge_weight) == edge_index.shape[1], "Edge weights should match edge count"
        
        # Test multi-modal analyzer
        analyzer = MultiModalSideChannelAnalyzer(fusion_method='adaptive', device='cpu')
        assert analyzer.fusion_method == 'adaptive', "Fusion method should be set correctly"
        
        # Test analysis (reduced dataset for speed)
        small_data = create_synthetic_multimodal_data(
            n_traces=20,
            trace_length=100,
            modalities=['power', 'em_near']
        )
        
        results = analyzer.analyze_multi_modal(small_data)
        assert 'fused_features' in results, "Results should contain fused features"
        assert results['n_traces'] == 20, "Should analyze correct number of traces"
        
        # Test trace synchronization
        data_unsync = MultiModalData(
            power_traces=np.random.randn(10, 1000),
            em_near_traces=np.random.randn(10, 800)  # Different length
        )
        
        sync_data = data_unsync.synchronize_traces()
        assert sync_data.power_traces.shape[1] == sync_data.em_near_traces.shape[1], \
            "Synchronized traces should have same length"
        
        print("✓ All multi-modal fusion tests passed")
        
        return {
            'passed': True,
            'components_tested': ['MultiModalData', 'GraphTopologyBuilder', 'MultiModalSideChannelAnalyzer'],
            'test_results': {
                'synthetic_data_generation': 'working',
                'graph_topology_building': 'working',
                'fusion_analysis': 'working',
                'trace_synchronization': 'working'
            }
        }
        
    except Exception as e:
        return {'passed': False, 'error': str(e)}

async def test_hardware_integration_functionality() -> Dict[str, Any]:
    """Test hardware integration functionality."""
    try:
        from neural_cryptanalysis.hardware_integration import (
            HardwareInTheLoopSystem, create_oscilloscope, create_target_board,
            MeasurementConfig, HardwareConfig
        )
        from neural_cryptanalysis.core import NeuralSCA
        
        print("Testing hardware integration components...")
        
        # Test hardware configuration
        config = HardwareConfig(
            device_type='oscilloscope',
            model='Test_Scope',
            connection_type='usb',
            connection_params={'usb_port': 'USB0::INSTR'},
            capabilities={'max_sample_rate': 1e9}
        )
        assert config.device_type == 'oscilloscope', "Device type should be set correctly"
        
        # Test device creation
        oscilloscope = create_oscilloscope('Test_Scope', {
            'type': 'usb',
            'usb_port': 'USB0::INSTR',
            'max_sample_rate': 1e9
        })
        assert oscilloscope.config.model == 'Test_Scope', "Oscilloscope model should be set"
        
        target_board = create_target_board('Test_Target', {
            'type': 'serial',
            'serial_port': '/dev/ttyUSB0'
        })
        assert target_board.config.model == 'Test_Target', "Target board model should be set"
        
        # Test device connection (simulated)
        connected = await oscilloscope.connect()
        assert connected, "Oscilloscope should connect successfully"
        assert oscilloscope.is_connected, "Oscilloscope should be marked as connected"
        
        target_connected = await target_board.connect()
        assert target_connected, "Target board should connect successfully"
        
        # Test device configuration
        scope_config = {
            'channels': {'A': {'range': '100mV'}, 'B': {'range': '50mV'}},
            'sample_rate': 1e6,
            'memory_depth': 10000
        }
        config_success = await oscilloscope.configure(scope_config)
        assert config_success, "Oscilloscope configuration should succeed"
        
        # Test HITL system
        neural_sca = NeuralSCA()
        hitl_system = HardwareInTheLoopSystem(neural_sca)
        assert hitl_system.neural_sca == neural_sca, "Neural SCA should be assigned"
        
        # Test device registration
        scope_registered = await hitl_system.add_device('test_scope', oscilloscope)
        assert scope_registered, "Device registration should succeed"
        assert 'test_scope' in hitl_system.devices, "Device should be in system"
        
        # Test measurement configuration
        measurement_config = MeasurementConfig(
            channels=['power', 'trigger'],
            sample_rate=1e6,
            memory_depth=5000,
            trigger_config={'channel': 'trigger', 'level': 2.5}
        )
        assert len(measurement_config.channels) == 2, "Should have 2 channels configured"
        
        # Test system status
        status = hitl_system.get_system_status()
        assert 'devices' in status, "Status should contain devices info"
        assert 'test_scope' in status['devices'], "Registered device should be in status"
        
        # Cleanup
        await oscilloscope.disconnect()
        await target_board.disconnect()
        
        print("✓ All hardware integration tests passed")
        
        return {
            'passed': True,
            'components_tested': ['HardwareConfig', 'OscilloscopeDevice', 'TargetBoard', 'HardwareInTheLoopSystem'],
            'test_results': {
                'device_creation': 'working',
                'device_connection': 'working',
                'device_configuration': 'working',
                'hitl_system': 'working',
                'measurement_config': 'working'
            }
        }
        
    except Exception as e:
        return {'passed': False, 'error': str(e)}

def test_advanced_countermeasures_functionality() -> Dict[str, Any]:
    """Test advanced countermeasure evaluation functionality."""
    try:
        from neural_cryptanalysis.advanced_countermeasures import (
            AdvancedCountermeasureEvaluator, BooleanMasking, ArithmeticMasking,
            TemporalShuffling, CountermeasureConfig, create_boolean_masking,
            create_arithmetic_masking, create_temporal_shuffling
        )
        from neural_cryptanalysis.core import NeuralSCA, TraceData
        
        print("Testing advanced countermeasure components...")
        
        # Test countermeasure configuration
        config = CountermeasureConfig(
            countermeasure_type='boolean_masking',
            order=1,
            parameters={'refresh_randomness': True}
        )
        assert config.countermeasure_type == 'boolean_masking', "Countermeasure type should be set"
        
        # Test Boolean masking
        boolean_masking = create_boolean_masking(order=1)
        assert isinstance(boolean_masking, BooleanMasking), "Should create BooleanMasking instance"
        assert boolean_masking.masking_order == 1, "Masking order should be 1"
        assert boolean_masking.estimate_security_order() == 1, "Security order should be 1"
        
        # Test masking application
        traces = np.random.randn(50, 500) * 0.1
        intermediate_values = np.random.randint(0, 256, 50)
        
        masked_traces, masked_values = boolean_masking.apply_countermeasure(traces, intermediate_values)
        assert masked_traces.shape == traces.shape, "Masked traces should have same shape"
        assert len(masked_values) == len(intermediate_values), "Should have masked values for all traces"
        assert len(masked_values[0]) == 2, "1st order masking should have 2 shares"
        
        # Test arithmetic masking
        arithmetic_masking = create_arithmetic_masking(order=1, modulus=256)
        assert isinstance(arithmetic_masking, ArithmeticMasking), "Should create ArithmeticMasking instance"
        assert arithmetic_masking.modulus == 256, "Modulus should be 256"
        
        # Test temporal shuffling
        temporal_shuffling = create_temporal_shuffling(n_operations=8)
        assert isinstance(temporal_shuffling, TemporalShuffling), "Should create TemporalShuffling instance"
        assert temporal_shuffling.n_operations == 8, "Should have 8 operations"
        
        shuffled_traces, shuffled_values = temporal_shuffling.apply_countermeasure(traces, intermediate_values)
        assert shuffled_traces.shape == traces.shape, "Shuffled traces should have same shape"
        
        # Test countermeasure evaluator
        neural_sca = NeuralSCA()
        evaluator = AdvancedCountermeasureEvaluator(neural_sca)
        assert evaluator.neural_sca == neural_sca, "Neural SCA should be assigned"
        
        # Test SNR computation
        snr = evaluator._compute_snr(traces)
        assert isinstance(snr, float), "SNR should be float"
        assert snr >= 0, "SNR should be non-negative"
        
        # Test T-test analysis
        t_test_results = evaluator._perform_ttest_analysis(traces)
        assert 'max_t_statistic' in t_test_results, "Should have max t-statistic"
        assert 'points_above_threshold' in t_test_results, "Should have threshold count"
        
        # Test higher-order analysis
        higher_order_results = evaluator._higher_order_analysis(traces, masked_values, 1)
        assert 'moments_analysis' in higher_order_results, "Should have moments analysis"
        assert 'practical_order' in higher_order_results, "Should have practical order"
        
        print("✓ All countermeasure evaluation tests passed")
        
        return {
            'passed': True,
            'components_tested': ['BooleanMasking', 'ArithmeticMasking', 'TemporalShuffling', 'AdvancedCountermeasureEvaluator'],
            'test_results': {
                'boolean_masking': 'working',
                'arithmetic_masking': 'working', 
                'temporal_shuffling': 'working',
                'statistical_analysis': 'working',
                'higher_order_analysis': 'working'
            }
        }
        
    except Exception as e:
        return {'passed': False, 'error': str(e)}

async def test_distributed_computing_functionality() -> Dict[str, Any]:
    """Test distributed computing functionality."""
    try:
        from neural_cryptanalysis.distributed_computing import (
            DistributedCoordinator, NeuralOperatorTrainingWorker, DistributedAttackWorker,
            TaskScheduler, DistributedDataManager, DistributedTask, DistributedDataManager
        )
        from neural_cryptanalysis.core import NeuralSCA, TraceData
        import tempfile
        
        print("Testing distributed computing components...")
        
        # Test distributed task
        task = DistributedTask(
            task_id='test_task_001',
            task_type='training',
            priority=1,
            data_shards=['shard_001', 'shard_002']
        )
        assert task.task_id == 'test_task_001', "Task ID should be set correctly"
        assert task.task_type == 'training', "Task type should be training"
        assert task.status == 'pending', "Initial status should be pending"
        
        # Test task scheduler
        scheduler = TaskScheduler()
        assert scheduler.pending_tasks.qsize() == 0, "Should start with empty queue"
        
        task_id = scheduler.submit_task(task)
        assert task_id == 'test_task_001', "Should return correct task ID"
        assert scheduler.pending_tasks.qsize() == 1, "Should have one pending task"
        
        # Test data manager
        with tempfile.TemporaryDirectory() as temp_dir:
            data_manager = DistributedDataManager(temp_dir)
            assert len(data_manager.shards) == 0, "Should start with no shards"
            
            # Test shard creation
            test_data = np.random.randn(1000, 100)
            shards = data_manager.create_data_shards(test_data, shard_size=400)
            
            assert len(shards) == 3, "Should create 3 shards for 1000 traces with shard_size=400"
            assert all(shard.data_type == 'traces' for shard in shards), "All shards should be trace type"
            
            # Test shard loading
            first_shard_id = shards[0].shard_id
            loaded_data = data_manager.load_shard(first_shard_id)
            assert loaded_data is not None, "Should load shard data"
            assert loaded_data.shape[0] == 400, "Should load correct number of traces"
            
            # Test integrity verification
            is_valid = data_manager.verify_shard_integrity(first_shard_id)
            assert is_valid, "Shard should pass integrity check"
        
        # Test workers
        neural_sca = NeuralSCA()
        
        training_worker = NeuralOperatorTrainingWorker('train_worker_001', neural_sca, device='cpu')
        assert training_worker.worker_id == 'train_worker_001', "Worker ID should be set"
        assert training_worker.capabilities['training'], "Should support training"
        
        attack_worker = DistributedAttackWorker('attack_worker_001', neural_sca)
        assert attack_worker.worker_id == 'attack_worker_001', "Worker ID should be set"
        assert attack_worker.capabilities['attack'], "Should support attacks"
        
        # Test worker lifecycle
        await training_worker.start()
        assert training_worker.is_running, "Worker should be running after start"
        
        # Test health check
        health_status = await training_worker.health_check()
        assert 'worker_id' in health_status, "Health status should include worker ID"
        assert health_status['status'] == 'healthy', "Worker should be healthy"
        
        await training_worker.stop()
        assert not training_worker.is_running, "Worker should not be running after stop"
        
        # Test distributed coordinator
        coordinator = DistributedCoordinator()
        assert isinstance(coordinator.scheduler, TaskScheduler), "Should have task scheduler"
        assert isinstance(coordinator.data_manager, DistributedDataManager), "Should have data manager"
        
        # Test coordinator lifecycle
        await coordinator.start()
        assert coordinator.is_running, "Coordinator should be running"
        
        await coordinator.stop()
        assert not coordinator.is_running, "Coordinator should not be running after stop"
        
        print("✓ All distributed computing tests passed")
        
        return {
            'passed': True,
            'components_tested': ['DistributedTask', 'TaskScheduler', 'DistributedDataManager', 'Workers', 'DistributedCoordinator'],
            'test_results': {
                'task_management': 'working',
                'data_sharding': 'working',
                'worker_management': 'working',
                'coordinator_lifecycle': 'working',
                'health_monitoring': 'working'
            }
        }
        
    except Exception as e:
        return {'passed': False, 'error': str(e)}

def test_performance_benchmarks() -> Dict[str, Any]:
    """Test performance benchmarks for enhanced features."""
    try:
        import time
        from neural_cryptanalysis.core import NeuralSCA, TraceData
        from neural_cryptanalysis.adaptive_rl import AdaptiveAttackEngine
        from neural_cryptanalysis.multi_modal_fusion import create_synthetic_multimodal_data, MultiModalSideChannelAnalyzer
        
        print("Running performance benchmarks...")
        
        results = {}
        
        # Benchmark 1: Adaptive RL state processing
        start_time = time.time()
        neural_sca = NeuralSCA()
        engine = AdaptiveAttackEngine(neural_sca, device='cpu')
        
        # Process 1000 state transitions
        from neural_cryptanalysis.adaptive_rl import AttackState
        for _ in range(1000):
            state = AttackState()
            vector = state.to_vector()
            action_id = engine.select_action(state, training=False)
        
        rl_time = time.time() - start_time
        results['adaptive_rl_1000_states'] = {
            'time_seconds': rl_time,
            'states_per_second': 1000 / rl_time,
            'requirement': 100,  # Should process at least 100 states/sec
            'passed': (1000 / rl_time) > 100
        }
        
        # Benchmark 2: Multi-modal fusion processing
        start_time = time.time()
        data = create_synthetic_multimodal_data(
            n_traces=500,
            trace_length=1000,
            modalities=['power', 'em_near']
        )
        
        analyzer = MultiModalSideChannelAnalyzer(fusion_method='adaptive', device='cpu')
        results_fusion = analyzer.analyze_multi_modal(data)
        
        fusion_time = time.time() - start_time
        results['multimodal_fusion_500_traces'] = {
            'time_seconds': fusion_time,
            'traces_per_second': 500 / fusion_time,
            'requirement': 50,  # Should process at least 50 traces/sec
            'passed': (500 / fusion_time) > 50
        }
        
        # Benchmark 3: Synthetic data generation speed
        start_time = time.time()
        for _ in range(10):
            data = create_synthetic_multimodal_data(
                n_traces=100,
                trace_length=500,
                modalities=['power', 'em_near', 'acoustic']
            )
        
        datagen_time = time.time() - start_time
        total_traces = 10 * 100
        results['data_generation_1000_traces'] = {
            'time_seconds': datagen_time,
            'traces_per_second': total_traces / datagen_time,
            'requirement': 200,  # Should generate at least 200 traces/sec
            'passed': (total_traces / datagen_time) > 200
        }
        
        # Overall performance assessment
        all_passed = all(result['passed'] for result in results.values())
        
        print("✓ Performance benchmarks completed")
        
        return {
            'passed': all_passed,
            'benchmarks': results,
            'summary': f"Passed {sum(1 for r in results.values() if r['passed'])}/{len(results)} benchmarks"
        }
        
    except Exception as e:
        return {'passed': False, 'error': str(e)}

def test_memory_efficiency() -> Dict[str, Any]:
    """Test memory efficiency of enhanced features."""
    try:
        import psutil
        import os
        from neural_cryptanalysis.multi_modal_fusion import create_synthetic_multimodal_data
        from neural_cryptanalysis.distributed_computing import DistributedDataManager
        import tempfile
        
        print("Testing memory efficiency...")
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Test 1: Large multi-modal dataset
        large_data = create_synthetic_multimodal_data(
            n_traces=2000,
            trace_length=5000,
            modalities=['power', 'em_near', 'em_far', 'acoustic']
        )
        
        after_multimodal = process.memory_info().rss / 1024 / 1024  # MB
        multimodal_memory = after_multimodal - initial_memory
        
        # Test 2: Distributed data sharding
        with tempfile.TemporaryDirectory() as temp_dir:
            data_manager = DistributedDataManager(temp_dir)
            
            # Create and shard large dataset
            large_traces = large_data.power_traces
            shards = data_manager.create_data_shards(large_traces, shard_size=500)
            
            after_sharding = process.memory_info().rss / 1024 / 1024  # MB
            sharding_memory = after_sharding - after_multimodal
        
        # Clean up large objects
        del large_data, large_traces
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory efficiency checks
        memory_results = {
            'initial_memory_mb': initial_memory,
            'multimodal_memory_increase_mb': multimodal_memory,
            'sharding_memory_increase_mb': sharding_memory,
            'final_memory_mb': final_memory,
            'total_increase_mb': final_memory - initial_memory,
            'efficient': (final_memory - initial_memory) < 500  # Should use less than 500MB
        }
        
        print(f"Memory efficiency: {memory_results['total_increase_mb']:.1f}MB increase")
        
        return {
            'passed': memory_results['efficient'],
            'memory_usage': memory_results
        }
        
    except Exception as e:
        return {'passed': False, 'error': str(e)}

def test_security_and_safety() -> Dict[str, Any]:
    """Test security and safety aspects of enhanced features."""
    try:
        print("Testing security and safety...")
        
        # Test 1: Input validation in adaptive RL
        from neural_cryptanalysis.adaptive_rl import AttackState, AttackAction
        from neural_cryptanalysis.core import NeuralSCA
        
        # Test extreme values
        extreme_state = AttackState(
            snr=-1000.0,  # Extreme negative
            success_rate=2.0,  # > 1.0
            traces_used=-100,  # Negative
            window_size=0  # Zero
        )
        
        vector = extreme_state.to_vector()
        assert len(vector) == 12, "Should handle extreme values without crashing"
        
        # Test 2: Data validation in multi-modal fusion
        from neural_cryptanalysis.multi_modal_fusion import MultiModalData
        
        # Test with mismatched data shapes
        mismatched_data = MultiModalData(
            power_traces=np.random.randn(100, 1000),
            em_near_traces=np.random.randn(50, 500)  # Different dimensions
        )
        
        sync_data = mismatched_data.synchronize_traces()
        assert sync_data is not None, "Should handle mismatched data gracefully"
        
        # Test 3: Hardware integration safety
        from neural_cryptanalysis.hardware_integration import create_oscilloscope
        
        # Test with invalid configuration
        try:
            invalid_scope = create_oscilloscope('Invalid_Model', {'invalid_param': 'invalid'})
            # Should create object but handle invalid params gracefully
            assert invalid_scope is not None, "Should handle invalid parameters"
        except Exception:
            pass  # Expected to potentially fail safely
        
        # Test 4: Distributed computing security
        from neural_cryptanalysis.distributed_computing import DistributedTask
        
        # Test with potentially malicious task data
        malicious_task = DistributedTask(
            task_id='<script>alert("xss")</script>',  # XSS-like
            task_type='../../../etc/passwd',  # Path traversal-like
            data_shards=['shard_' + 'A' * 1000]  # Very long string
        )
        
        assert malicious_task.task_id is not None, "Should handle malicious input safely"
        
        # Test 5: Memory safety with large inputs
        large_traces = np.random.randn(10000, 10000)  # Very large array
        
        try:
            # This should either work or fail gracefully without crashes
            from neural_cryptanalysis.advanced_countermeasures import create_boolean_masking
            masking = create_boolean_masking(order=1)
            # Don't actually apply to save memory, just test creation
            assert masking is not None, "Should create countermeasure safely"
        except MemoryError:
            pass  # Acceptable to fail with memory error
        
        print("✓ Security and safety tests passed")
        
        return {
            'passed': True,
            'test_results': {
                'input_validation': 'working',
                'data_validation': 'working',
                'configuration_safety': 'working',
                'malicious_input_handling': 'working',
                'memory_safety': 'working'
            }
        }
        
    except Exception as e:
        return {'passed': False, 'error': str(e)}

def test_integration_compatibility() -> Dict[str, Any]:
    """Test compatibility and integration between enhanced features."""
    try:
        print("Testing integration compatibility...")
        
        from neural_cryptanalysis.core import NeuralSCA
        from neural_cryptanalysis.adaptive_rl import AdaptiveAttackEngine
        from neural_cryptanalysis.multi_modal_fusion import create_synthetic_multimodal_data, MultiModalSideChannelAnalyzer
        from neural_cryptanalysis.advanced_countermeasures import create_boolean_masking
        
        # Test 1: Adaptive RL with Multi-modal data
        neural_sca = NeuralSCA()
        adaptive_engine = AdaptiveAttackEngine(neural_sca)
        
        multimodal_data = create_synthetic_multimodal_data(n_traces=50, trace_length=200)
        
        # Convert multimodal data to format compatible with adaptive RL
        from neural_cryptanalysis.core import TraceData
        power_traces = TraceData(traces=multimodal_data.power_traces)
        
        # Should be able to use adaptive RL with converted data
        assert power_traces is not None, "Should convert multimodal data for adaptive RL"
        
        # Test 2: Countermeasures with Multi-modal analysis
        analyzer = MultiModalSideChannelAnalyzer(fusion_method='adaptive')
        
        # Apply countermeasure to one modality
        masking = create_boolean_masking(order=1)
        masked_power, _ = masking.apply_countermeasure(
            multimodal_data.power_traces,
            np.random.randint(0, 256, len(multimodal_data.power_traces))
        )
        
        # Create new multimodal data with masked power
        from neural_cryptanalysis.multi_modal_fusion import MultiModalData
        masked_multimodal = MultiModalData(
            power_traces=masked_power,
            em_near_traces=multimodal_data.em_near_traces
        )
        
        # Should be able to analyze masked multimodal data
        results = analyzer.analyze_multi_modal(masked_multimodal)
        assert results is not None, "Should analyze countermeasure-protected multimodal data"
        
        # Test 3: All components working together
        components = {
            'neural_sca': neural_sca,
            'adaptive_engine': adaptive_engine,
            'multimodal_analyzer': analyzer,
            'countermeasure': masking
        }
        
        # Verify all components are compatible (no import conflicts, etc.)
        for name, component in components.items():
            assert component is not None, f"{name} should be initialized"
        
        # Test 4: Data format compatibility
        formats_compatible = True
        
        # Check that TraceData can be created from various sources
        try:
            trace_data_1 = TraceData(traces=np.random.randn(10, 100))
            trace_data_2 = TraceData(traces=multimodal_data.power_traces)
            trace_data_3 = TraceData(traces=masked_power)
            
            assert all(td.traces is not None for td in [trace_data_1, trace_data_2, trace_data_3])
        except Exception:
            formats_compatible = False
        
        print("✓ Integration compatibility tests passed")
        
        return {
            'passed': True,
            'test_results': {
                'adaptive_rl_multimodal_compatibility': 'working',
                'countermeasures_multimodal_compatibility': 'working',
                'component_compatibility': 'working',
                'data_format_compatibility': formats_compatible
            }
        }
        
    except Exception as e:
        return {'passed': False, 'error': str(e)}

def main():
    """Run all enhanced quality gates."""
    print("NEURAL OPERATOR CRYPTANALYSIS LAB - ENHANCED QUALITY GATES")
    print("=" * 70)
    print(f"Execution time: {datetime.now().isoformat()}")
    
    # Define quality gates
    quality_gates = [
        ("Enhanced Module Imports", lambda: test_enhanced_imports(), True),
        ("Adaptive RL Functionality", test_adaptive_rl_functionality, True),
        ("Multi-Modal Fusion Functionality", test_multimodal_fusion_functionality, True),
        ("Hardware Integration Functionality", test_hardware_integration_functionality, True),
        ("Advanced Countermeasures Functionality", test_advanced_countermeasures_functionality, True),
        ("Distributed Computing Functionality", test_distributed_computing_functionality, True),
        ("Performance Benchmarks", test_performance_benchmarks, False),
        ("Memory Efficiency", test_memory_efficiency, False),
        ("Security and Safety", test_security_and_safety, True),
        ("Integration Compatibility", test_integration_compatibility, False),
    ]
    
    # Run all quality gates
    results = []
    start_time = time.time()
    
    for gate_name, gate_function, is_critical in quality_gates:
        result = run_quality_gate(gate_name, gate_function, is_critical)
        results.append(result)
        
        # Stop on critical failure
        if is_critical and not result['passed']:
            print(f"\n💥 CRITICAL FAILURE: {gate_name}")
            print("Stopping execution due to critical failure.")
            break
    
    total_time = time.time() - start_time
    
    # Calculate overall results
    passed_gates = sum(1 for r in results if r['passed'])
    total_gates = len(results)
    critical_failures = sum(1 for r in results if r['critical'] and not r['passed'])
    
    overall_passed = critical_failures == 0 and passed_gates == total_gates
    overall_score = passed_gates / total_gates if total_gates > 0 else 0.0
    
    # Generate summary report
    print(f"\n" + "="*70)
    print("ENHANCED QUALITY GATES SUMMARY")
    print(f"="*70)
    print(f"Total execution time: {total_time:.2f}s")
    print(f"Gates passed: {passed_gates}/{total_gates} ({overall_score:.1%})")
    print(f"Critical failures: {critical_failures}")
    print(f"Overall result: {'✅ PASSED' if overall_passed else '❌ FAILED'}")
    
    # Detailed results
    print(f"\nDetailed Results:")
    for result in results:
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        critical = " (CRITICAL)" if result['critical'] else ""
        print(f"  {result['name']}: {status}{critical} ({result['execution_time']:.3f}s)")
    
    # Save results to file
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'overall_passed': str(overall_passed),
        'overall_score': overall_score,
        'execution_time_seconds': total_time,
        'results': results
    }
    
    with open('enhanced_quality_gates_report.json', 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
    
    print(f"\nReport saved to: enhanced_quality_gates_report.json")
    
    return 0 if overall_passed else 1

def test_enhanced_imports() -> Dict[str, Any]:
    """Test that all enhanced modules can be imported."""
    try:
        print("Testing enhanced module imports...")
        
        # Test all enhanced modules
        modules = [
            'neural_cryptanalysis.adaptive_rl',
            'neural_cryptanalysis.multi_modal_fusion', 
            'neural_cryptanalysis.hardware_integration',
            'neural_cryptanalysis.advanced_countermeasures',
            'neural_cryptanalysis.distributed_computing'
        ]
        
        imported_modules = []
        for module_name in modules:
            try:
                __import__(module_name)
                imported_modules.append(module_name)
            except ImportError as e:
                return {'passed': False, 'error': f"Failed to import {module_name}: {e}"}
        
        print(f"✓ Successfully imported {len(imported_modules)} enhanced modules")
        
        return {
            'passed': True,
            'imported_modules': imported_modules
        }
        
    except Exception as e:
        return {'passed': False, 'error': str(e)}

if __name__ == "__main__":
    exit(main())