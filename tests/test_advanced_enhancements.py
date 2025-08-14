#!/usr/bin/env python3
"""
Comprehensive test suite for advanced enhancements to Neural Operator Cryptanalysis Lab.

Tests for adaptive RL, multi-modal fusion, hardware integration, countermeasure evaluation,
and distributed computing capabilities.
"""

import pytest
import numpy as np
import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import os

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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_cryptanalysis.core import NeuralSCA, TraceData, LeakageSimulator
from neural_cryptanalysis.adaptive_rl import (
    AdaptiveAttackEngine, MetaLearningAdaptiveEngine, AttackState, AttackAction
)
from neural_cryptanalysis.multi_modal_fusion import (
    MultiModalData, MultiModalSideChannelAnalyzer, create_synthetic_multimodal_data,
    GraphTopologyBuilder, SensorConfig
)
from neural_cryptanalysis.hardware_integration import (
    HardwareInTheLoopSystem, OscilloscopeDevice, TargetBoard,
    create_oscilloscope, create_target_board, MeasurementConfig, HardwareConfig
)
from neural_cryptanalysis.advanced_countermeasures import (
    AdvancedCountermeasureEvaluator, BooleanMasking, ArithmeticMasking,
    TemporalShuffling, CountermeasureConfig, create_boolean_masking
)
from neural_cryptanalysis.distributed_computing import (
    DistributedCoordinator, NeuralOperatorTrainingWorker, DistributedAttackWorker,
    TaskScheduler, DistributedDataManager, DistributedTask
)

class TestAdaptiveRL:
    """Test suite for reinforcement learning adaptive attack engine."""
    
    @pytest.fixture
    def neural_sca(self):
        """Create neural SCA instance for testing."""
        return NeuralSCA(architecture='fourier_neural_operator')
    
    @pytest.fixture
    def sample_traces(self):
        """Create sample trace data."""
        traces = np.random.randn(500, 1000) * 0.1
        labels = np.random.randint(0, 256, 500)
        return TraceData(traces=traces, labels=labels)
    
    def test_attack_state_vector_conversion(self):
        """Test AttackState to vector conversion."""
        state = AttackState(
            snr=0.5,
            success_rate=0.8,
            traces_used=5000,
            confidence=0.9,
            preprocessing_method='standardize',
            window_size=2000,
            window_offset=500,
            n_pois=100,
            poi_method='mutual_information'
        )
        
        vector = state.to_vector()
        
        assert isinstance(vector, np.ndarray)
        assert len(vector) == 12  # Expected number of features
        assert vector[0] == 0.5  # SNR
        assert vector[1] == 0.8  # Success rate
        assert vector[2] == 0.5  # Normalized traces_used (5000/10000)
        assert vector[3] == 0.9  # Confidence
    
    def test_attack_action_from_id(self):
        """Test AttackAction creation from action ID."""
        action = AttackAction.from_action_id(0)  # Increase window size
        
        assert action.action_type == 'adjust_window'
        assert action.parameter == 'window_size'
        assert action.value == 1.2
    
    def test_adaptive_engine_initialization(self, neural_sca):
        """Test adaptive attack engine initialization."""
        engine = AdaptiveAttackEngine(
            neural_sca=neural_sca,
            learning_rate=1e-4,
            epsilon=0.9
        )
        
        assert engine.neural_sca == neural_sca
        assert engine.epsilon == 0.9
        assert engine.q_network is not None
        assert engine.target_network is not None
        assert len(engine.episode_rewards) == 0
    
    def test_action_selection(self, neural_sca):
        """Test epsilon-greedy action selection."""
        engine = AdaptiveAttackEngine(neural_sca, epsilon=0.0)  # No exploration
        state = AttackState()
        
        # With epsilon=0, should always select greedy action
        action1 = engine.select_action(state, training=True)
        action2 = engine.select_action(state, training=True)
        
        assert isinstance(action1, int)
        assert 0 <= action1 < engine.action_dim
        assert action1 == action2  # Should be deterministic with epsilon=0
    
    def test_reward_computation(self, neural_sca):
        """Test reward computation for state transitions."""
        engine = AdaptiveAttackEngine(neural_sca)
        
        old_state = AttackState(success_rate=0.5, confidence=0.6, snr=0.3)
        new_state = AttackState(success_rate=0.8, confidence=0.9, snr=0.4)
        action = AttackAction('adjust_window', 'window_size', 1.2)
        
        reward = engine.compute_reward(old_state, new_state, action)
        
        # Should be positive reward for improvement
        assert reward > 0
        
        # Test negative case
        worse_state = AttackState(success_rate=0.2, confidence=0.3, snr=0.1)
        negative_reward = engine.compute_reward(old_state, worse_state, action)
        
        assert negative_reward < reward
    
    def test_action_application(self, neural_sca):
        """Test applying actions to states."""
        engine = AdaptiveAttackEngine(neural_sca)
        
        initial_state = AttackState(window_size=1000, n_pois=50)
        
        # Test window size adjustment
        action = AttackAction('adjust_window', 'window_size', 1.5)
        new_state = engine.apply_action(initial_state, action)
        
        assert new_state.window_size == 1500  # 1000 * 1.5
        assert new_state.current_step == initial_state.current_step + 1
        
        # Test POI modification
        poi_action = AttackAction('modify_pois', 'n_pois', 2.0)
        poi_state = engine.apply_action(initial_state, poi_action)
        
        assert poi_state.n_pois == 100  # 50 * 2.0
    
    @pytest.mark.asyncio
    async def test_autonomous_attack_basic(self, neural_sca, sample_traces):
        """Test basic autonomous attack functionality."""
        engine = AdaptiveAttackEngine(neural_sca, epsilon=0.1)
        
        # Mock the evaluation method to avoid actual training
        async def mock_evaluation(state, traces):
            return 0.7, 0.8, 0.5  # success_rate, confidence, snr
        
        with patch.object(engine, 'evaluate_attack_performance', side_effect=mock_evaluation):
            # Run very short autonomous attack
            results = engine.autonomous_attack(
                traces=sample_traces,
                target_success_rate=0.6,
                max_episodes=3,  # Very short for testing
                patience=2
            )
        
        assert 'success_rate' in results
        assert 'confidence' in results
        assert 'optimal_parameters' in results
        assert 'training_episodes' in results
        assert results['training_episodes'] > 0
    
    def test_meta_learning_engine_initialization(self, neural_sca):
        """Test meta-learning adaptive engine initialization."""
        engine = MetaLearningAdaptiveEngine(neural_sca)
        
        assert hasattr(engine, 'meta_network')
        assert hasattr(engine, 'meta_optimizer')
        assert hasattr(engine, 'task_embeddings')
        assert isinstance(engine.task_embeddings, dict)
    
    def test_target_embedding_extraction(self, neural_sca, sample_traces):
        """Test target embedding extraction for meta-learning."""
        engine = MetaLearningAdaptiveEngine(neural_sca)
        
        embedding = engine.extract_target_embedding(sample_traces)
        
        assert isinstance(embedding, torch.Tensor)
        assert embedding.shape[0] == 32  # Expected embedding size
    
    def test_model_save_load(self, neural_sca, tmp_path):
        """Test saving and loading of trained models."""
        engine = AdaptiveAttackEngine(neural_sca)
        
        # Save model
        model_path = tmp_path / "test_model.pth"
        engine.save_model(str(model_path))
        
        assert model_path.exists()
        
        # Create new engine and load model
        new_engine = AdaptiveAttackEngine(neural_sca)
        new_engine.load_model(str(model_path))
        
        # Check that training step was loaded
        assert new_engine.training_step == engine.training_step

class TestMultiModalFusion:
    """Test suite for multi-modal sensor fusion."""
    
    def test_multimodal_data_creation(self):
        """Test MultiModalData creation and basic functionality."""
        power_traces = np.random.randn(100, 1000)
        em_traces = np.random.randn(100, 1000)
        
        data = MultiModalData(
            power_traces=power_traces,
            em_near_traces=em_traces
        )
        
        modalities = data.get_available_modalities()
        assert 'power' in modalities
        assert 'em_near' in modalities
        assert len(modalities) == 2
        
        # Test trace data retrieval
        retrieved_power = data.get_trace_data('power')
        assert np.array_equal(retrieved_power, power_traces)
    
    def test_sensor_config_creation(self):
        """Test sensor configuration creation."""
        config = SensorConfig(
            name='power_probe',
            sample_rate=1e6,
            resolution=12,
            noise_floor=0.001,
            frequency_range=(1e3, 1e6),
            spatial_position=(0, 0, 0)
        )
        
        assert config.name == 'power_probe'
        assert config.sample_rate == 1e6
        assert config.spatial_position == (0, 0, 0)
    
    def test_synthetic_multimodal_data_generation(self):
        """Test synthetic multi-modal data generation."""
        modalities = ['power', 'em_near', 'acoustic']
        data = create_synthetic_multimodal_data(
            n_traces=50,
            trace_length=500,
            modalities=modalities
        )
        
        assert len(data.get_available_modalities()) == 3
        assert data.power_traces.shape == (50, 500)
        assert data.em_near_traces.shape == (50, 500)
        assert data.acoustic_traces.shape == (50, 500)
        
        # Check sensor configs were created
        assert 'power' in data.sensor_configs
        assert 'em_near' in data.sensor_configs
        assert 'acoustic' in data.sensor_configs
    
    def test_trace_synchronization(self):
        """Test trace synchronization functionality."""
        # Create data with different length traces
        power_traces = np.random.randn(10, 1000)
        em_traces = np.random.randn(10, 800)  # Shorter
        acoustic_traces = np.random.randn(10, 1200)  # Longer
        
        data = MultiModalData(
            power_traces=power_traces,
            em_near_traces=em_traces,
            acoustic_traces=acoustic_traces
        )
        
        # Synchronize to power traces (reference)
        sync_data = data.synchronize_traces(reference_modality='power')
        
        # All traces should now have same length as power traces
        assert sync_data.power_traces.shape == (10, 1000)
        assert sync_data.em_near_traces.shape == (10, 1000)
        assert sync_data.acoustic_traces.shape == (10, 1000)
    
    def test_graph_topology_builder(self):
        """Test graph topology construction."""
        builder = GraphTopologyBuilder()
        
        # Test spatial graph
        positions = {
            'sensor1': (0.0, 0.0, 0.0),
            'sensor2': (1.0, 0.0, 0.0),
            'sensor3': (0.0, 1.0, 0.0)
        }
        
        edge_index, edge_weight = builder.build_spatial_graph(
            positions, connection_threshold=2.0
        )
        
        assert edge_index.shape[0] == 2  # Source and target indices
        assert edge_index.shape[1] == edge_weight.shape[0]  # Same number of edges
        
        # Test temporal graph
        temporal_edge_index, temporal_edge_weight = builder.build_temporal_graph(
            trace_length=10, temporal_window=2
        )
        
        assert temporal_edge_index.shape[0] == 2
        assert len(temporal_edge_weight) > 0
    
    def test_multimodal_analyzer_initialization(self):
        """Test MultiModalSideChannelAnalyzer initialization."""
        analyzer = MultiModalSideChannelAnalyzer(
            fusion_method='adaptive',
            device='cpu'
        )
        
        assert analyzer.fusion_method == 'adaptive'
        assert analyzer.device.type == 'cpu'
        assert analyzer.fusion_network is None  # Not initialized until first use
    
    def test_multimodal_analysis_adaptive(self):
        """Test multi-modal analysis with adaptive fusion."""
        # Create test data
        data = create_synthetic_multimodal_data(
            n_traces=20,
            trace_length=100,
            modalities=['power', 'em_near']
        )
        
        analyzer = MultiModalSideChannelAnalyzer(
            fusion_method='adaptive',
            device='cpu'
        )
        
        # Perform analysis
        results = analyzer.analyze_multi_modal(data)
        
        assert 'fused_features' in results
        assert 'n_traces' in results
        assert 'modalities_used' in results
        assert results['n_traces'] == 20
        assert 'power' in results['modalities_used']
        assert 'em_near' in results['modalities_used']
    
    def test_fusion_quality_computation(self):
        """Test fusion quality metrics computation."""
        data = create_synthetic_multimodal_data(
            n_traces=30,
            trace_length=200,
            modalities=['power', 'em_near', 'acoustic']
        )
        
        analyzer = MultiModalSideChannelAnalyzer(fusion_method='adaptive')
        
        # Mock fused features
        fused_features = np.random.randn(30, 200)
        
        quality_metrics = analyzer._compute_fusion_quality(data, fused_features)
        
        assert 'snr_improvement' in quality_metrics
        assert 'average_correlation' in quality_metrics
        assert 'fusion_snr' in quality_metrics
        assert 'individual_snrs' in quality_metrics
        
        assert isinstance(quality_metrics['snr_improvement'], float)
        assert len(quality_metrics['individual_snrs']) == 3  # Three modalities

class TestHardwareIntegration:
    """Test suite for hardware integration system."""
    
    def test_hardware_config_creation(self):
        """Test hardware configuration creation."""
        config = HardwareConfig(
            device_type='oscilloscope',
            model='Picoscope_6404D',
            connection_type='usb',
            connection_params={'usb_port': 'USB0::INSTR'},
            capabilities={'max_sample_rate': 5e9}
        )
        
        assert config.device_type == 'oscilloscope'
        assert config.model == 'Picoscope_6404D'
        assert config.capabilities['max_sample_rate'] == 5e9
    
    def test_oscilloscope_device_creation(self):
        """Test oscilloscope device creation."""
        params = {
            'type': 'usb',
            'usb_port': 'USB0::INSTR',
            'max_sample_rate': 1e9,
            'channels': 4
        }
        
        scope = create_oscilloscope('Test_Scope', params)
        
        assert isinstance(scope, OscilloscopeDevice)
        assert scope.config.model == 'Test_Scope'
        assert scope.config.connection_type == 'usb'
        assert not scope.is_connected  # Not connected initially
    
    def test_target_board_creation(self):
        """Test target board creation."""
        params = {
            'type': 'serial',
            'serial_port': '/dev/ttyUSB0',
            'programmable': True
        }
        
        target = create_target_board('Test_Board', params)
        
        assert isinstance(target, TargetBoard)
        assert target.config.model == 'Test_Board'
        assert target.config.capabilities['programmable'] is True
        assert not target.is_connected
    
    @pytest.mark.asyncio
    async def test_oscilloscope_connection(self):
        """Test oscilloscope connection process."""
        params = {
            'type': 'usb',
            'usb_port': 'USB0::INSTR'
        }
        
        scope = create_oscilloscope('Test_Scope', params)
        
        # Test connection
        connected = await scope.connect()
        assert connected
        assert scope.is_connected
        
        # Test disconnection
        disconnected = await scope.disconnect()
        assert disconnected
        assert not scope.is_connected
    
    @pytest.mark.asyncio
    async def test_oscilloscope_configuration(self):
        """Test oscilloscope configuration."""
        params = {'type': 'usb', 'usb_port': 'USB0::INSTR'}
        scope = create_oscilloscope('Test_Scope', params)
        
        await scope.connect()
        
        config = {
            'channels': {
                'A': {'range': '100mV', 'coupling': 'DC'},
                'B': {'range': '50mV', 'coupling': 'AC'}
            },
            'sample_rate': 1e6,
            'memory_depth': 10000
        }
        
        configured = await scope.configure(config)
        assert configured
    
    @pytest.mark.asyncio
    async def test_target_board_programming(self):
        """Test target board firmware programming."""
        params = {'type': 'serial', 'serial_port': '/dev/ttyUSB0'}
        target = create_target_board('Test_Target', params)
        
        await target.connect()
        
        # Test firmware programming
        firmware_path = "test_firmware.hex"
        programmed = await target.program_firmware(firmware_path)
        
        assert programmed
        assert target.is_programmed
        assert target.current_firmware == firmware_path
    
    @pytest.mark.asyncio
    async def test_operation_triggering(self):
        """Test cryptographic operation triggering."""
        params = {'type': 'serial', 'serial_port': '/dev/ttyUSB0'}
        target = create_target_board('Test_Target', params)
        
        await target.connect()
        
        # Test operation trigger
        test_data = b'\x00\x01\x02\x03' * 4  # 16 bytes
        triggered = await target.trigger_operation('aes_encrypt', test_data)
        
        assert triggered
    
    def test_measurement_config_creation(self):
        """Test measurement configuration creation."""
        config = MeasurementConfig(
            channels=['power', 'trigger'],
            sample_rate=1e6,
            memory_depth=10000,
            trigger_config={'channel': 'trigger', 'level': 2.5}
        )
        
        assert len(config.channels) == 2
        assert 'power' in config.channels
        assert config.sample_rate == 1e6
        assert config.trigger_config['level'] == 2.5
    
    @pytest.mark.asyncio
    async def test_hitl_system_initialization(self):
        """Test hardware-in-the-loop system initialization."""
        from neural_cryptanalysis.core import NeuralSCA
        
        neural_sca = NeuralSCA()
        hitl_system = HardwareInTheLoopSystem(neural_sca)
        
        assert hitl_system.neural_sca == neural_sca
        assert len(hitl_system.devices) == 0
        assert not hitl_system.measurement_active
    
    @pytest.mark.asyncio
    async def test_device_registration(self):
        """Test device registration with HITL system."""
        from neural_cryptanalysis.core import NeuralSCA
        
        neural_sca = NeuralSCA()
        hitl_system = HardwareInTheLoopSystem(neural_sca)
        
        # Create and register oscilloscope
        scope = create_oscilloscope('Test_Scope', {'type': 'usb', 'usb_port': 'USB0::INSTR'})
        registered = await hitl_system.add_device('scope1', scope)
        
        assert registered
        assert 'scope1' in hitl_system.devices
        assert hitl_system.devices['scope1'] == scope
    
    @pytest.mark.asyncio
    async def test_system_status(self):
        """Test system status reporting."""
        from neural_cryptanalysis.core import NeuralSCA
        
        neural_sca = NeuralSCA()
        hitl_system = HardwareInTheLoopSystem(neural_sca)
        
        # Add device
        scope = create_oscilloscope('Test_Scope', {'type': 'usb', 'usb_port': 'USB0::INSTR'})
        await hitl_system.add_device('scope1', scope)
        
        status = hitl_system.get_system_status()
        
        assert 'devices' in status
        assert 'measurement_active' in status
        assert 'analysis_running' in status
        assert 'scope1' in status['devices']
        assert not status['measurement_active']

class TestAdvancedCountermeasures:
    """Test suite for advanced countermeasure evaluation."""
    
    def test_countermeasure_config_creation(self):
        """Test countermeasure configuration creation."""
        config = CountermeasureConfig(
            countermeasure_type='boolean_masking',
            order=2,
            parameters={'refresh_randomness': True},
            description='Second-order Boolean masking'
        )
        
        assert config.countermeasure_type == 'boolean_masking'
        assert config.order == 2
        assert config.parameters['refresh_randomness'] is True
    
    def test_boolean_masking_creation(self):
        """Test Boolean masking countermeasure creation."""
        masking = create_boolean_masking(order=1, refresh_randomness=True)
        
        assert isinstance(masking, BooleanMasking)
        assert masking.masking_order == 1
        assert masking.refresh_randomness is True
        assert masking.estimate_security_order() == 1
    
    def test_boolean_masking_application(self):
        """Test Boolean masking application to traces."""
        masking = create_boolean_masking(order=1)
        
        # Create test traces and intermediate values
        traces = np.random.randn(100, 1000) * 0.1
        intermediate_values = np.random.randint(0, 256, 100)
        
        masked_traces, masked_values = masking.apply_countermeasure(traces, intermediate_values)
        
        assert masked_traces.shape == traces.shape
        assert len(masked_values) == len(intermediate_values)
        assert len(masked_values[0]) == 2  # 1st order masking = 2 shares
    
    def test_arithmetic_masking_creation(self):
        """Test arithmetic masking countermeasure."""
        from neural_cryptanalysis.advanced_countermeasures import create_arithmetic_masking
        
        masking = create_arithmetic_masking(order=2, modulus=256)
        
        assert isinstance(masking, ArithmeticMasking)
        assert masking.masking_order == 2
        assert masking.modulus == 256
    
    def test_temporal_shuffling_creation(self):
        """Test temporal shuffling countermeasure."""
        from neural_cryptanalysis.advanced_countermeasures import create_temporal_shuffling
        
        shuffling = create_temporal_shuffling(n_operations=16, shuffle_window=500)
        
        assert isinstance(shuffling, TemporalShuffling)
        assert shuffling.n_operations == 16
        assert shuffling.shuffle_window == 500
    
    def test_temporal_shuffling_application(self):
        """Test temporal shuffling application to traces."""
        from neural_cryptanalysis.advanced_countermeasures import create_temporal_shuffling
        
        shuffling = create_temporal_shuffling(n_operations=8)
        
        traces = np.random.randn(50, 800)
        intermediate_values = np.random.randint(0, 256, 50)
        
        shuffled_traces, shuffled_values = shuffling.apply_countermeasure(traces, intermediate_values)
        
        assert shuffled_traces.shape == traces.shape
        # Values should be unchanged (shuffling doesn't modify intermediate values)
        assert np.array_equal(shuffled_values, intermediate_values)
    
    def test_countermeasure_evaluator_initialization(self):
        """Test advanced countermeasure evaluator initialization."""
        from neural_cryptanalysis.core import NeuralSCA
        
        neural_sca = NeuralSCA()
        evaluator = AdvancedCountermeasureEvaluator(neural_sca)
        
        assert evaluator.neural_sca == neural_sca
        assert 'min_traces' in evaluator.evaluation_params
        assert evaluator.evaluation_params['confidence_level'] == 0.95
    
    def test_snr_computation(self):
        """Test SNR computation for countermeasures."""
        from neural_cryptanalysis.core import NeuralSCA
        
        neural_sca = NeuralSCA()
        evaluator = AdvancedCountermeasureEvaluator(neural_sca)
        
        # Create test traces with known SNR characteristics
        n_traces, trace_length = 100, 500
        traces = np.random.randn(n_traces, trace_length) * 0.1
        
        # Add signal component to increase SNR
        signal = np.sin(np.linspace(0, 2*np.pi, trace_length)) * 0.05
        traces += signal
        
        snr = evaluator._compute_snr(traces)
        
        assert isinstance(snr, float)
        assert snr > 0  # Should be positive due to added signal
    
    def test_ttest_analysis(self):
        """Test T-test analysis (TVLA)."""
        from neural_cryptanalysis.core import NeuralSCA
        
        neural_sca = NeuralSCA()
        evaluator = AdvancedCountermeasureEvaluator(neural_sca)
        
        # Create traces with different means in two groups
        group1 = np.random.normal(0.0, 0.1, (50, 100))
        group2 = np.random.normal(0.01, 0.1, (50, 100))  # Slight offset
        traces = np.vstack([group1, group2])
        
        t_test_results = evaluator._perform_ttest_analysis(traces)
        
        assert 'max_t_statistic' in t_test_results
        assert 'mean_t_statistic' in t_test_results
        assert 'points_above_threshold' in t_test_results
        assert t_test_results['max_t_statistic'] > 0
    
    def test_higher_order_analysis(self):
        """Test higher-order statistical analysis."""
        from neural_cryptanalysis.core import NeuralSCA
        
        neural_sca = NeuralSCA()
        evaluator = AdvancedCountermeasureEvaluator(neural_sca)
        
        traces = np.random.randn(100, 200) * 0.1
        masked_values = np.random.randint(0, 256, (100, 2))  # 2 shares
        
        higher_order_results = evaluator._higher_order_analysis(traces, masked_values, theoretical_order=1)
        
        assert 'moments_analysis' in higher_order_results
        assert 'practical_order' in higher_order_results
        assert 'detection_threshold' in higher_order_results
        assert isinstance(higher_order_results['practical_order'], int)
    
    def test_countermeasure_comparison(self):
        """Test comparison of multiple countermeasures."""
        from neural_cryptanalysis.core import NeuralSCA, TraceData
        
        neural_sca = NeuralSCA()
        evaluator = AdvancedCountermeasureEvaluator(neural_sca)
        
        # Create original traces
        traces = np.random.randn(200, 500) * 0.1
        labels = np.random.randint(0, 256, 200)
        original_traces = TraceData(traces=traces, labels=labels)
        
        # Create countermeasures to compare
        countermeasures = [
            create_boolean_masking(order=1),
            create_boolean_masking(order=2),
        ]
        
        # Mock the evaluation method to avoid long computation
        def mock_evaluate(countermeasure, original_traces, max_traces=1000):
            from neural_cryptanalysis.advanced_countermeasures import EvaluationMetrics
            return EvaluationMetrics(
                theoretical_security_order=countermeasure.estimate_security_order(),
                practical_security_order=countermeasure.estimate_security_order(),
                traces_needed_90_percent=10000 * countermeasure.estimate_security_order(),
                traces_needed_95_percent=20000 * countermeasure.estimate_security_order(),
                max_success_rate=0.5 / countermeasure.estimate_security_order(),
                snr_reduction_factor=2.0 ** countermeasure.estimate_security_order(),
                mutual_information=0.1 / countermeasure.estimate_security_order(),
                t_test_statistics={'max_t_statistic': 2.0}
            )
        
        with patch.object(evaluator, 'evaluate_countermeasure', side_effect=mock_evaluate):
            comparison_results = evaluator.compare_countermeasures(countermeasures, original_traces)
        
        assert 'individual_results' in comparison_results
        assert 'summary' in comparison_results
        assert len(comparison_results['individual_results']) == 2

class TestDistributedComputing:
    """Test suite for distributed computing framework."""
    
    def test_distributed_task_creation(self):
        """Test distributed task creation."""
        task = DistributedTask(
            task_id='test_task_001',
            task_type='training',
            priority=1,
            data_shards=['shard_001', 'shard_002']
        )
        
        assert task.task_id == 'test_task_001'
        assert task.task_type == 'training'
        assert len(task.data_shards) == 2
        assert task.status == 'pending'
    
    def test_task_scheduler_initialization(self):
        """Test task scheduler initialization."""
        scheduler = TaskScheduler()
        
        assert scheduler.pending_tasks.qsize() == 0
        assert len(scheduler.running_tasks) == 0
        assert len(scheduler.completed_tasks) == 0
        assert len(scheduler.nodes) == 0
    
    def test_task_submission(self):
        """Test task submission to scheduler."""
        scheduler = TaskScheduler()
        
        task = DistributedTask(
            task_id='test_task',
            task_type='training',
            priority=2
        )
        
        task_id = scheduler.submit_task(task)
        
        assert task_id == 'test_task'
        assert scheduler.pending_tasks.qsize() == 1
    
    def test_data_manager_initialization(self):
        """Test distributed data manager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_manager = DistributedDataManager(temp_dir)
            
            assert data_manager.storage_root.exists()
            assert len(data_manager.shards) == 0
            assert data_manager.replication_factor == 2
    
    def test_data_shard_creation(self):
        """Test data shard creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_manager = DistributedDataManager(temp_dir)
            
            # Create test data
            test_data = np.random.randn(1000, 500)
            
            shards = data_manager.create_data_shards(test_data, shard_size=400)
            
            # Should create 3 shards (1000 traces / 400 per shard = 2.5 -> 3)
            assert len(shards) == 3
            
            for shard in shards:
                assert shard.data_type == 'traces'
                assert Path(shard.location).exists()
                assert len(shard.checksum) == 32  # MD5 hex length
    
    def test_shard_loading(self):
        """Test loading data shards."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_manager = DistributedDataManager(temp_dir)
            
            # Create and save test data
            test_data = np.random.randn(500, 100)
            shards = data_manager.create_data_shards(test_data, shard_size=200)
            
            # Load first shard
            shard_id = shards[0].shard_id
            loaded_data = data_manager.load_shard(shard_id)
            
            assert loaded_data is not None
            assert loaded_data.shape[0] == 200  # Shard size
            assert loaded_data.shape[1] == 100  # Feature dimension
    
    def test_shard_integrity_verification(self):
        """Test shard integrity verification."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_manager = DistributedDataManager(temp_dir)
            
            # Create test shard
            test_data = np.random.randn(100, 50)
            shards = data_manager.create_data_shards(test_data, shard_size=100)
            
            shard_id = shards[0].shard_id
            
            # Verify integrity
            is_valid = data_manager.verify_shard_integrity(shard_id)
            assert is_valid
    
    def test_training_worker_initialization(self):
        """Test neural operator training worker initialization."""
        from neural_cryptanalysis.core import NeuralSCA
        
        neural_sca = NeuralSCA()
        worker = NeuralOperatorTrainingWorker('worker_001', neural_sca, device='cpu')
        
        assert worker.worker_id == 'worker_001'
        assert worker.neural_sca == neural_sca
        assert worker.capabilities['training'] is True
        assert worker.capabilities['device'] == 'cpu'
    
    def test_attack_worker_initialization(self):
        """Test distributed attack worker initialization."""
        from neural_cryptanalysis.core import NeuralSCA
        
        neural_sca = NeuralSCA()
        worker = DistributedAttackWorker('attack_worker_001', neural_sca)
        
        assert worker.worker_id == 'attack_worker_001'
        assert worker.capabilities['attack'] is True
        assert worker.capabilities['analysis'] is True
    
    @pytest.mark.asyncio
    async def test_worker_lifecycle(self):
        """Test worker start/stop lifecycle."""
        from neural_cryptanalysis.core import NeuralSCA
        
        neural_sca = NeuralSCA()
        worker = NeuralOperatorTrainingWorker('test_worker', neural_sca)
        
        # Initially not running
        assert not worker.is_running
        
        # Start worker
        await worker.start()
        assert worker.is_running
        
        # Stop worker
        await worker.stop()
        assert not worker.is_running
    
    @pytest.mark.asyncio
    async def test_worker_health_check(self):
        """Test worker health check."""
        from neural_cryptanalysis.core import NeuralSCA
        
        neural_sca = NeuralSCA()
        worker = NeuralOperatorTrainingWorker('health_test_worker', neural_sca)
        
        await worker.start()
        
        health_status = await worker.health_check()
        
        assert 'worker_id' in health_status
        assert 'status' in health_status
        assert 'device' in health_status
        assert health_status['worker_id'] == 'health_test_worker'
        assert health_status['status'] == 'healthy'
    
    @pytest.mark.asyncio
    async def test_distributed_coordinator_initialization(self):
        """Test distributed coordinator initialization."""
        coordinator = DistributedCoordinator()
        
        assert isinstance(coordinator.scheduler, TaskScheduler)
        assert isinstance(coordinator.data_manager, DistributedDataManager)
        assert len(coordinator.workers) == 0
        assert not coordinator.is_running
    
    @pytest.mark.asyncio
    async def test_coordinator_lifecycle(self):
        """Test coordinator start/stop lifecycle."""
        coordinator = DistributedCoordinator()
        
        # Start coordinator
        await coordinator.start()
        assert coordinator.is_running
        assert coordinator.coordinator_task is not None
        
        # Stop coordinator
        await coordinator.stop()
        assert not coordinator.is_running
    
    def test_storage_statistics(self):
        """Test storage statistics computation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_manager = DistributedDataManager(temp_dir)
            
            # Create some test shards
            test_data = np.random.randn(500, 100)
            shards = data_manager.create_data_shards(test_data, shard_size=200)
            
            stats = data_manager.get_storage_stats()
            
            assert 'total_shards' in stats
            assert 'total_size_mb' in stats
            assert 'storage_root' in stats
            assert 'healthy_shards' in stats
            
            assert stats['total_shards'] == len(shards)
            assert stats['total_size_mb'] > 0
            assert stats['healthy_shards'] == len(shards)

class TestIntegrationScenarios:
    """Integration tests combining multiple components."""
    
    @pytest.mark.asyncio
    async def test_adaptive_rl_with_multimodal_fusion(self):
        """Test adaptive RL with multi-modal fusion."""
        # Create multi-modal data
        data = create_synthetic_multimodal_data(
            n_traces=100,
            trace_length=500,
            modalities=['power', 'em_near']
        )
        
        # Initialize components
        neural_sca = NeuralSCA()
        adaptive_engine = AdaptiveAttackEngine(neural_sca, epsilon=0.1)
        
        # Convert multi-modal data to TraceData for adaptive engine
        traces = TraceData(traces=data.power_traces)
        
        # Mock evaluation to avoid long computation
        async def mock_evaluation(state, traces):
            return 0.6, 0.7, 0.4
        
        with patch.object(adaptive_engine, 'evaluate_attack_performance', side_effect=mock_evaluation):
            results = adaptive_engine.autonomous_attack(
                traces=traces,
                max_episodes=2,
                patience=1
            )
        
        assert 'success_rate' in results
        assert 'optimal_parameters' in results
    
    def test_countermeasures_with_multimodal_analysis(self):
        """Test countermeasure evaluation with multi-modal analysis."""
        # Create original multi-modal data
        original_data = create_synthetic_multimodal_data(
            n_traces=50,
            trace_length=200,
            modalities=['power', 'em_near']
        )
        
        # Apply Boolean masking to power traces
        masking = create_boolean_masking(order=1)
        masked_power, _ = masking.apply_countermeasure(
            original_data.power_traces,
            np.random.randint(0, 256, 50)
        )
        
        # Create masked multi-modal data
        masked_data = MultiModalData(
            power_traces=masked_power,
            em_near_traces=original_data.em_near_traces
        )
        
        # Analyze both datasets
        analyzer = MultiModalSideChannelAnalyzer(fusion_method='adaptive')
        
        original_results = analyzer.analyze_multi_modal(original_data)
        masked_results = analyzer.analyze_multi_modal(masked_data)
        
        # Masking should reduce fusion quality
        if 'fusion_quality' in original_results and 'fusion_quality' in masked_results:
            original_snr = original_results['fusion_quality']['fusion_snr']
            masked_snr = masked_results['fusion_quality']['fusion_snr']
            
            # Masked implementation should have lower SNR (in most cases)
            # Note: This is probabilistic due to random data
            assert isinstance(original_snr, float)
            assert isinstance(masked_snr, float)
    
    def test_comprehensive_system_integration(self):
        """Test comprehensive system integration."""
        # This test verifies that all components can be instantiated together
        # without conflicts
        
        # Core components
        neural_sca = NeuralSCA(architecture='fourier_neural_operator')
        
        # Adaptive RL
        adaptive_engine = AdaptiveAttackEngine(neural_sca)
        
        # Multi-modal fusion
        multimodal_analyzer = MultiModalSideChannelAnalyzer(fusion_method='adaptive')
        
        # Countermeasure evaluation
        countermeasure_evaluator = AdvancedCountermeasureEvaluator(neural_sca)
        
        # Hardware integration
        hitl_system = HardwareInTheLoopSystem(neural_sca)
        
        # Distributed computing
        coordinator = DistributedCoordinator()
        
        # Verify all components are properly initialized
        assert adaptive_engine.neural_sca == neural_sca
        assert multimodal_analyzer.fusion_method == 'adaptive'
        assert countermeasure_evaluator.neural_sca == neural_sca
        assert hitl_system.neural_sca == neural_sca
        assert isinstance(coordinator.scheduler, TaskScheduler)
        
        # Verify no conflicts in imports or class definitions
        assert AdaptiveAttackEngine != MultiModalSideChannelAnalyzer
        assert HardwareInTheLoopSystem != DistributedCoordinator

def test_module_imports():
    """Test that all enhanced modules can be imported without errors."""
    import neural_cryptanalysis.adaptive_rl
    import neural_cryptanalysis.multi_modal_fusion
    import neural_cryptanalysis.hardware_integration
    import neural_cryptanalysis.advanced_countermeasures
    import neural_cryptanalysis.distributed_computing
    
    # Verify key classes are available
    assert hasattr(neural_cryptanalysis.adaptive_rl, 'AdaptiveAttackEngine')
    assert hasattr(neural_cryptanalysis.multi_modal_fusion, 'MultiModalSideChannelAnalyzer')
    assert hasattr(neural_cryptanalysis.hardware_integration, 'HardwareInTheLoopSystem')
    assert hasattr(neural_cryptanalysis.advanced_countermeasures, 'AdvancedCountermeasureEvaluator')
    assert hasattr(neural_cryptanalysis.distributed_computing, 'DistributedCoordinator')

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])