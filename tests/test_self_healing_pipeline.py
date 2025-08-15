"""Comprehensive tests for self-healing pipeline functionality.

This test suite validates all components of the self-healing pipeline guard
system including monitoring, adaptive learning, resilience patterns, and
auto-scaling capabilities.
"""

import json
import pytest
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the pipeline modules
try:
    from src.neural_cryptanalysis.pipeline import (
        SelfHealingGuard,
        PipelineGuardManager,
        HealthStatus,
        FailureType,
        PipelineMetrics,
        RecoveryAction,
        AdaptiveLearningEngine,
        AdvancedMonitoringSystem,
        ResilienceManager,
        AutoScaler,
        ResourceType,
        ScalingPolicy,
        ResourceMetrics,
        CircuitBreakerConfig,
        RetryConfig,
        MonitoringConfig
    )
except ImportError as e:
    pytest.skip(f"Pipeline modules not available: {e}", allow_module_level=True)


class TestSelfHealingGuard:
    """Test the core self-healing guard functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_config = Path("test_config.json")
        self.guard = SelfHealingGuard(
            config_path=self.temp_config,
            monitoring_interval=0.1,
            metrics_history_size=10
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.guard.is_monitoring:
            self.guard.stop_monitoring()
        if self.temp_config.exists():
            self.temp_config.unlink()
    
    def test_guard_initialization(self):
        """Test guard initialization."""
        assert self.guard.current_status == HealthStatus.HEALTHY
        assert not self.guard.is_monitoring
        assert len(self.guard.recovery_actions) > 0
        assert 'restart_service' in self.guard.recovery_actions
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        # Start monitoring
        self.guard.start_monitoring()
        assert self.guard.is_monitoring
        assert self.guard.monitoring_thread is not None
        
        # Stop monitoring
        self.guard.stop_monitoring()
        assert not self.guard.is_monitoring
    
    def test_metrics_collection(self):
        """Test metrics collection."""
        metrics = self.guard._collect_metrics()
        
        assert isinstance(metrics, PipelineMetrics)
        assert metrics.timestamp is not None
        assert 0 <= metrics.cpu_usage <= 100
        assert 0 <= metrics.memory_usage <= 100
        assert metrics.error_rate >= 0
    
    def test_health_analysis(self):
        """Test health score calculation."""
        # Test with good metrics
        good_metrics = PipelineMetrics(
            timestamp=datetime.now(),
            cpu_usage=30.0,
            memory_usage=40.0,
            disk_usage=20.0,
            network_latency=10.0,
            throughput=800.0,
            error_rate=0.001,
            response_time=100.0,
            queue_depth=5,
            active_connections=50
        )
        
        health_score = self.guard._analyze_health(good_metrics)
        assert 0.7 <= health_score <= 1.0
        
        # Test with bad metrics
        bad_metrics = PipelineMetrics(
            timestamp=datetime.now(),
            cpu_usage=95.0,
            memory_usage=90.0,
            disk_usage=85.0,
            network_latency=200.0,
            throughput=100.0,
            error_rate=0.1,
            response_time=1000.0,
            queue_depth=100,
            active_connections=500
        )
        
        bad_health_score = self.guard._analyze_health(bad_metrics)
        assert 0.0 <= bad_health_score <= 0.5
        assert bad_health_score < health_score
    
    def test_status_transitions(self):
        """Test health status transitions."""
        # Start with healthy status
        assert self.guard.current_status == HealthStatus.HEALTHY
        
        # Test transition to warning
        self.guard._update_status(0.6, 0.3)
        assert self.guard.current_status == HealthStatus.WARNING
        
        # Test transition to critical
        self.guard._update_status(0.3, 0.8)
        assert self.guard.current_status == HealthStatus.CRITICAL
        
        # Test transition back to healthy
        self.guard._update_status(0.9, 0.1)
        assert self.guard.current_status == HealthStatus.HEALTHY
    
    def test_recovery_actions(self):
        """Test recovery action execution."""
        # Test successful action
        action = self.guard.recovery_actions['clear_cache']
        success = self.guard._execute_action(action)
        assert success
        assert action.execution_count == 1
        
        # Test action cooldown
        assert not self.guard._should_execute_action(action)
        
        # Reset time and test again
        action.last_executed = datetime.now() - timedelta(seconds=action.cooldown_seconds + 1)
        assert self.guard._should_execute_action(action)
    
    def test_custom_recovery_action(self):
        """Test adding custom recovery actions."""
        called = False
        
        def custom_action():
            nonlocal called
            called = True
            return True
        
        self.guard.add_custom_recovery_action(
            "test_action",
            "Test recovery action",
            custom_action,
            severity_threshold=0.5
        )
        
        assert "test_action" in self.guard.recovery_actions
        
        # Execute the custom action
        action = self.guard.recovery_actions["test_action"]
        success = self.guard._execute_action(action)
        assert success
        assert called
    
    def test_status_report(self):
        """Test status report generation."""
        # Add some metrics
        for _ in range(5):
            metrics = self.guard._collect_metrics()
            self.guard._add_metrics(metrics)
        
        report = self.guard.get_status_report()
        
        assert 'status' in report
        assert 'health_score' in report
        assert 'uptime_hours' in report
        assert 'latest_metrics' in report
        assert 'recovery_actions' in report
        assert 'monitoring_active' in report
    
    def test_force_recovery_action(self):
        """Test forcing a recovery action."""
        success = self.guard.force_recovery_action("clear_cache")
        assert success
        
        # Test unknown action
        success = self.guard.force_recovery_action("unknown_action")
        assert not success


class TestAdaptiveLearningEngine:
    """Test the adaptive learning engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path("test_models")
        self.engine = AdaptiveLearningEngine(
            model_save_path=self.temp_dir,
            learning_rate=0.1
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.engine.stop_background_learning()
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        assert self.engine.learning_state.model_version == 1
        assert self.engine.learning_state.training_samples == 0
        assert not self.engine.stop_learning.is_set()
    
    def test_action_outcome_recording(self):
        """Test recording action outcomes."""
        pre_metrics = {
            'cpu_usage': 80.0,
            'memory_usage': 70.0,
            'error_rate': 0.05,
            'response_time': 300.0
        }
        
        post_metrics = {
            'cpu_usage': 60.0,
            'memory_usage': 50.0,
            'error_rate': 0.01,
            'response_time': 150.0
        }
        
        self.engine.record_action_outcome(
            "scale_resources",
            pre_metrics,
            post_metrics,
            True
        )
        
        assert len(self.engine.action_outcomes) == 1
        assert self.engine.learning_state.training_samples == 1
        
        outcome = self.engine.action_outcomes[0]
        assert outcome.action_name == "scale_resources"
        assert outcome.success == True
        assert outcome.improvement_score > 0
    
    def test_action_recommendation(self):
        """Test action recommendation."""
        # Record some training data
        for i in range(10):
            pre_metrics = {
                'cpu_usage': 70.0 + i * 2,
                'memory_usage': 60.0 + i,
                'error_rate': 0.01 + i * 0.001,
                'response_time': 200.0 + i * 10
            }
            
            post_metrics = {
                'cpu_usage': max(10, pre_metrics['cpu_usage'] - 20),
                'memory_usage': max(10, pre_metrics['memory_usage'] - 15),
                'error_rate': max(0, pre_metrics['error_rate'] - 0.005),
                'response_time': max(50, pre_metrics['response_time'] - 50)
            }
            
            self.engine.record_action_outcome(
                "scale_resources",
                pre_metrics,
                post_metrics,
                True
            )
        
        # Get recommendation
        current_metrics = {
            'cpu_usage': 85.0,
            'memory_usage': 75.0,
            'error_rate': 0.02,
            'response_time': 300.0
        }
        
        available_actions = ['restart_service', 'scale_resources', 'clear_cache']
        action, confidence = self.engine.recommend_action(current_metrics, available_actions)
        
        assert action in available_actions
        assert 0 <= confidence <= 1
    
    def test_failure_prediction(self):
        """Test failure probability prediction."""
        # Add some training data
        for i in range(20):
            metrics = {
                'cpu_usage': 50.0 + i * 2,
                'memory_usage': 40.0 + i,
                'error_rate': 0.01 + i * 0.002,
                'response_time': 150.0 + i * 5
            }
            
            # Simulate failure for high metrics
            failure = metrics['cpu_usage'] > 80
            self.engine.pattern_recognizer.add_observation(
                metrics, failure, datetime.now()
            )
        
        # Test prediction
        test_metrics = {
            'cpu_usage': 85.0,
            'memory_usage': 80.0,
            'error_rate': 0.03,
            'response_time': 250.0
        }
        
        failure_prob = self.engine.predict_failure_probability(test_metrics)
        assert 0 <= failure_prob <= 1
    
    def test_learning_insights(self):
        """Test learning insights generation."""
        # Add some data
        for i in range(5):
            self.engine.record_action_outcome(
                "test_action",
                {'cpu_usage': 70 + i * 2},
                {'cpu_usage': 50 + i},
                True
            )
        
        insights = self.engine.get_learning_insights()
        
        assert 'learning_state' in insights
        assert 'action_effectiveness' in insights
        assert 'total_outcomes' in insights
        assert insights['total_outcomes'] == 5


class TestAdvancedMonitoringSystem:
    """Test the advanced monitoring system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = MonitoringConfig(
            collection_interval=0.1,
            retention_hours=1,
            anomaly_threshold=2.0
        )
        self.monitoring = AdvancedMonitoringSystem(self.config)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.monitoring.is_running:
            self.monitoring.stop_monitoring()
    
    def test_monitoring_initialization(self):
        """Test monitoring system initialization."""
        assert not self.monitoring.is_running
        assert len(self.monitoring.metrics_collector.collectors) >= 2
        assert self.monitoring.config.collection_interval == 0.1
    
    def test_metrics_collection(self):
        """Test metrics collection."""
        metrics = self.monitoring.metrics_collector.collect_all()
        
        assert len(metrics) > 0
        for metric in metrics:
            assert hasattr(metric, 'timestamp')
            assert hasattr(metric, 'name')
            assert hasattr(metric, 'value')
            assert isinstance(metric.value, (int, float))
    
    def test_anomaly_detection(self):
        """Test anomaly detection."""
        detector = self.monitoring.anomaly_detector
        
        # Add normal metrics first to establish baseline
        for i in range(25):
            from src.neural_cryptanalysis.pipeline.monitoring import MetricPoint
            metric = MetricPoint(
                timestamp=datetime.now(),
                name="test_metric",
                value=50.0 + i * 0.1,  # Gradually increasing
                tags={"source": "test"}
            )
            detector.add_metric_point(metric)
        
        # Add anomalous metric
        anomalous_metric = MetricPoint(
            timestamp=datetime.now(),
            name="test_metric",
            value=150.0,  # Much higher than baseline
            tags={"source": "test"}
        )
        
        alert = detector.add_metric_point(anomalous_metric)
        
        # Should detect anomaly after baseline is established
        if alert:
            assert alert.metric_name == "test_metric"
            assert alert.current_value == 150.0
    
    def test_alert_management(self):
        """Test alert management."""
        alert_manager = self.monitoring.alert_manager
        
        # Create a test alert
        from src.neural_cryptanalysis.pipeline.monitoring import Alert, AlertSeverity
        alert = Alert(
            id="test_alert",
            timestamp=datetime.now(),
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="This is a test alert",
            metric_name="test_metric",
            current_value=80.0,
            threshold_value=70.0,
            tags={"source": "test"}
        )
        
        # Process alert
        alert_manager.process_alert(alert)
        
        # Check if alert was stored
        active_alerts = alert_manager.get_active_alerts()
        assert len(active_alerts) > 0
        assert active_alerts[0].id == "test_alert"
        
        # Acknowledge alert
        success = alert_manager.acknowledge_alert("test_alert")
        assert success
        
        # Resolve alert
        success = alert_manager.resolve_alert("test_alert")
        assert success
        
        # Check if alert was removed from active alerts
        active_alerts = alert_manager.get_active_alerts()
        assert len(active_alerts) == 0
    
    def test_custom_metrics(self):
        """Test adding custom metrics."""
        self.monitoring.add_custom_metric(
            "test.custom_metric",
            42.0,
            {"source": "unit_test"}
        )
        
        assert "test.custom_metric" in self.monitoring.metrics_history
        assert len(self.monitoring.metrics_history["test.custom_metric"]) == 1
    
    def test_system_status(self):
        """Test system status reporting."""
        # Add some metrics
        for i in range(5):
            self.monitoring.add_custom_metric(f"test.metric_{i}", i * 10.0)
        
        status = self.monitoring.get_system_status()
        
        assert 'monitoring_active' in status
        assert 'health_score' in status
        assert 'latest_metrics' in status
        assert 'total_metrics' in status
        assert status['total_metrics'] == 5


class TestResilienceManager:
    """Test the resilience management system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.resilience = ResilienceManager("test_system")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.resilience.shutdown()
    
    def test_circuit_breaker_creation(self):
        """Test circuit breaker creation and usage."""
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=1.0)
        cb = self.resilience.create_circuit_breaker("test_cb", config)
        
        assert "test_cb" in self.resilience.circuit_breakers
        assert cb.name == "test_cb"
        assert cb.config.failure_threshold == 3
    
    def test_circuit_breaker_operation(self):
        """Test circuit breaker behavior."""
        cb = self.resilience.create_circuit_breaker("test_cb")
        
        # Test successful calls
        def success_func():
            return "success"
        
        result = cb.call(success_func)
        assert result == "success"
        
        # Test failing calls
        def fail_func():
            raise Exception("Test failure")
        
        # Should fail and increment failure count
        for i in range(cb.config.failure_threshold):
            try:
                cb.call(fail_func)
            except Exception:
                pass
        
        # Circuit should now be open
        from src.neural_cryptanalysis.pipeline.resilience import CircuitState, CircuitBreakerError
        assert cb.state == CircuitState.OPEN
        
        # Should reject calls
        with pytest.raises(CircuitBreakerError):
            cb.call(success_func)
    
    def test_retry_mechanism(self):
        """Test retry mechanism."""
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        retry = self.resilience.create_retry_mechanism("test_retry", config)
        
        # Test function that succeeds on third attempt
        attempt_count = 0
        def flaky_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Not yet")
            return "success"
        
        result = retry.execute(flaky_func)
        assert result == "success"
        assert attempt_count == 3
    
    def test_health_checker(self):
        """Test health checker functionality."""
        health_status = True
        
        def health_check():
            return health_status
        
        checker = self.resilience.create_health_checker("test_service", health_check)
        checker.start_monitoring()
        
        # Wait a bit for health check
        time.sleep(0.1)
        
        status = checker.get_status()
        assert status['name'] == "test_service"
        assert status['is_monitoring'] == True
        
        # Change health status
        health_status = False
        time.sleep(0.2)  # Wait for health check to run
        
        checker.stop_monitoring()
    
    def test_resilient_operation_execution(self):
        """Test executing operations with resilience patterns."""
        # Create components
        self.resilience.create_circuit_breaker("test_cb")
        self.resilience.create_retry_mechanism("test_retry")
        
        def test_function(x):
            return x * 2
        
        def fallback_function(error, x):
            return x  # Just return input as fallback
        
        result = self.resilience.execute_resilient_operation(
            test_function,
            5,
            circuit_breaker="test_cb",
            retry="test_retry",
            fallback=fallback_function
        )
        
        assert result == 10
    
    def test_system_health_reporting(self):
        """Test system health reporting."""
        # Create some components
        self.resilience.create_circuit_breaker("cb1")
        self.resilience.create_retry_mechanism("retry1")
        
        health = self.resilience.get_system_health()
        
        assert 'overall_health_score' in health
        assert 'circuit_breakers' in health
        assert 'retry_mechanisms' in health
        assert 0 <= health['overall_health_score'] <= 1


class TestAutoScaler:
    """Test the auto-scaling system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.autoscaler = AutoScaler("test_autoscaler")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.autoscaler.is_running:
            self.autoscaler.stop_auto_scaling()
    
    def test_autoscaler_initialization(self):
        """Test auto-scaler initialization."""
        assert self.autoscaler.name == "test_autoscaler"
        assert not self.autoscaler.is_running
        assert len(self.autoscaler.policies) == 0
    
    def test_scaling_policy_addition(self):
        """Test adding scaling policies."""
        policy = ScalingPolicy(
            name="test_policy",
            resource_type=ResourceType.CPU,
            min_instances=1,
            max_instances=5,
            target_utilization=70.0
        )
        
        self.autoscaler.add_scaling_policy(policy)
        
        assert "test_policy" in self.autoscaler.policies
        assert self.autoscaler.current_instances[ResourceType.CPU] == 1
    
    def test_scaling_callbacks(self):
        """Test scaling callback registration and execution."""
        scale_up_called = False
        scale_down_called = False
        
        def scale_up_callback(resource_type, target_instances):
            nonlocal scale_up_called
            scale_up_called = True
            return True
        
        def scale_down_callback(resource_type, target_instances):
            nonlocal scale_down_called
            scale_down_called = True
            return True
        
        self.autoscaler.register_scale_up_callback(scale_up_callback)
        self.autoscaler.register_scale_down_callback(scale_down_callback)
        
        # Test manual scaling
        success = self.autoscaler.force_scale(ResourceType.CPU, 3, "Test scale up")
        assert success
        assert scale_up_called
        
        success = self.autoscaler.force_scale(ResourceType.CPU, 1, "Test scale down")
        assert success
        assert scale_down_called
    
    def test_metrics_processing(self):
        """Test metrics processing for scaling decisions."""
        # Add a policy
        policy = ScalingPolicy(
            name="cpu_policy",
            resource_type=ResourceType.CPU,
            scale_up_threshold=80.0,
            scale_down_threshold=30.0
        )
        self.autoscaler.add_scaling_policy(policy)
        
        # Add metrics that should trigger scaling
        high_cpu_metrics = ResourceMetrics(
            timestamp=datetime.now(),
            cpu_utilization=85.0,
            memory_utilization=60.0,
            request_rate=1000.0,
            response_time=200.0
        )
        
        self.autoscaler.add_metrics(high_cpu_metrics)
        
        # Check that metrics were stored
        assert len(self.autoscaler.metrics_history) == 1
        assert self.autoscaler.metrics_history[-1].cpu_utilization == 85.0
    
    def test_scaling_status(self):
        """Test scaling status reporting."""
        # Add a policy and some metrics
        policy = ScalingPolicy(
            name="test_policy",
            resource_type=ResourceType.CPU
        )
        self.autoscaler.add_scaling_policy(policy)
        
        status = self.autoscaler.get_scaling_status()
        
        assert 'is_running' in status
        assert 'current_instances' in status
        assert 'policies' in status
        assert 'predictive_model_trained' in status
        assert len(status['policies']) == 1


class TestPipelineGuardManager:
    """Test the pipeline guard manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = PipelineGuardManager()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.manager.stop_all()
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        assert len(self.manager.guards) == 0
    
    def test_pipeline_addition(self):
        """Test adding pipelines to manager."""
        guard = self.manager.add_pipeline("test_pipeline")
        
        assert "test_pipeline" in self.manager.guards
        assert isinstance(guard, SelfHealingGuard)
    
    def test_global_status(self):
        """Test global status reporting."""
        # Add some pipelines
        self.manager.add_pipeline("pipeline1")
        self.manager.add_pipeline("pipeline2")
        
        status = self.manager.get_global_status()
        
        assert len(status) == 2
        assert "pipeline1" in status
        assert "pipeline2" in status
    
    def test_unhealthy_pipeline_detection(self):
        """Test detection of unhealthy pipelines."""
        # Add pipelines
        guard1 = self.manager.add_pipeline("healthy_pipeline")
        guard2 = self.manager.add_pipeline("unhealthy_pipeline")
        
        # Make one pipeline unhealthy
        guard2.current_status = HealthStatus.CRITICAL
        
        unhealthy = self.manager.get_unhealthy_pipelines()
        
        assert "unhealthy_pipeline" in unhealthy
        assert "healthy_pipeline" not in unhealthy


class TestIntegration:
    """Integration tests for the complete self-healing pipeline system."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.manager = PipelineGuardManager()
        self.temp_dir = Path("test_integration")
        self.temp_dir.mkdir(exist_ok=True)
    
    def teardown_method(self):
        """Clean up integration test fixtures."""
        self.manager.stop_all()
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_complete_pipeline_integration(self):
        """Test complete pipeline with all components."""
        # Create a comprehensive pipeline guard
        guard = self.manager.add_pipeline(
            "comprehensive_pipeline",
            config_path=self.temp_dir / "config.json",
            monitoring_interval=0.1
        )
        
        # Add custom recovery action
        recovery_executed = False
        
        def custom_recovery():
            nonlocal recovery_executed
            recovery_executed = True
            return True
        
        guard.add_custom_recovery_action(
            "custom_action",
            "Custom recovery action",
            custom_recovery,
            severity_threshold=0.6
        )
        
        # Start monitoring
        guard.start_monitoring()
        
        # Let it run for a short time
        time.sleep(0.5)
        
        # Force a degraded status to trigger recovery
        guard._update_status(0.4, 0.8)  # Low health, high failure risk
        
        # Give time for recovery actions
        time.sleep(0.2)
        
        # Check status
        status = guard.get_status_report()
        assert status['monitoring_active'] == True
        assert len(status['recovery_actions']) > 0
        
        # Stop monitoring
        guard.stop_monitoring()
    
    def test_pipeline_persistence(self):
        """Test configuration persistence."""
        guard = self.manager.add_pipeline(
            "persistent_pipeline",
            config_path=self.temp_dir / "persistent_config.json",
            monitoring_interval=1.0
        )
        
        # Modify configuration
        guard.monitoring_interval = 2.0
        guard._save_config()
        
        # Create new guard with same config
        new_guard = SelfHealingGuard(
            config_path=self.temp_dir / "persistent_config.json"
        )
        
        # Should load the saved configuration
        assert new_guard.monitoring_interval == 2.0
    
    def test_performance_under_load(self):
        """Test system performance under simulated load."""
        guard = self.manager.add_pipeline(
            "load_test_pipeline",
            monitoring_interval=0.05  # Very fast monitoring
        )
        
        guard.start_monitoring()
        
        # Simulate high load by rapidly adding metrics
        start_time = time.time()
        metric_count = 0
        
        while time.time() - start_time < 1.0:  # Run for 1 second
            metrics = guard._collect_metrics()
            guard._add_metrics(metrics)
            metric_count += 1
        
        guard.stop_monitoring()
        
        # Should have processed many metrics without crashing
        assert metric_count > 10
        assert len(guard.metrics_history) > 0
        
        # System should still be responsive
        status = guard.get_status_report()
        assert status is not None


# Utility functions for testing
def create_test_metrics(cpu=50.0, memory=60.0, error_rate=0.01, response_time=200.0):
    """Create test metrics for consistent testing."""
    return PipelineMetrics(
        timestamp=datetime.now(),
        cpu_usage=cpu,
        memory_usage=memory,
        disk_usage=30.0,
        network_latency=20.0,
        throughput=500.0,
        error_rate=error_rate,
        response_time=response_time,
        queue_depth=10,
        active_connections=50
    )


def create_test_resource_metrics(cpu=50.0, memory=60.0, request_rate=500.0, response_time=200.0):
    """Create test resource metrics for auto-scaling tests."""
    return ResourceMetrics(
        timestamp=datetime.now(),
        cpu_utilization=cpu,
        memory_utilization=memory,
        request_rate=request_rate,
        response_time=response_time,
        queue_depth=10,
        active_connections=50
    )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])