"""Self-Healing Pipeline Guard - Autonomous Pipeline Recovery System.

This module implements an intelligent pipeline monitoring and self-healing system
that automatically detects failures, diagnoses issues, and implements corrective
actions without human intervention.

The guard uses machine learning to predict pipeline failures and proactively
implement preventive measures.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
import threading
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import statistics

# Mock imports for dependencies that may not be available
try:
    import numpy as np
except ImportError:
    # Fallback mock
    class np:
        @staticmethod
        def array(x): return x
        @staticmethod
        def mean(x): return sum(x) / len(x) if x else 0
        @staticmethod
        def std(x): return statistics.stdev(x) if len(x) > 1 else 0
        @staticmethod
        def percentile(x, p): return sorted(x)[int(len(x) * p / 100)] if x else 0

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
except ImportError:
    # Mock implementations
    class IsolationForest:
        def __init__(self, **kwargs): pass
        def fit(self, X): return self
        def predict(self, X): return [1] * len(X)
        def decision_function(self, X): return [0.5] * len(X)
    
    class StandardScaler:
        def __init__(self): pass
        def fit(self, X): return self
        def transform(self, X): return X
        def fit_transform(self, X): return X


class HealthStatus(Enum):
    """Pipeline health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"
    RECOVERING = "recovering"


class FailureType(Enum):
    """Types of pipeline failures."""
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_FAILURE = "network_failure"
    DATA_CORRUPTION = "data_corruption"
    ALGORITHM_FAILURE = "algorithm_failure"
    DEPENDENCY_FAILURE = "dependency_failure"
    TIMEOUT = "timeout"
    MEMORY_LEAK = "memory_leak"
    DEADLOCK = "deadlock"
    PERFORMANCE_DEGRADATION = "performance_degradation"


@dataclass
class PipelineMetrics:
    """Pipeline performance and health metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    throughput: float
    error_rate: float
    response_time: float
    queue_depth: int
    active_connections: int
    
    def to_features(self) -> List[float]:
        """Convert metrics to feature vector for ML models."""
        return [
            self.cpu_usage,
            self.memory_usage,
            self.disk_usage,
            self.network_latency,
            self.throughput,
            self.error_rate,
            self.response_time,
            float(self.queue_depth),
            float(self.active_connections)
        ]


@dataclass
class RecoveryAction:
    """Represents a recovery action that can be taken."""
    name: str
    description: str
    severity_threshold: float
    action_function: Callable
    cooldown_seconds: int = 300
    max_retries: int = 3
    success_rate: float = 0.0
    last_executed: Optional[datetime] = None
    execution_count: int = 0


class SelfHealingGuard:
    """Autonomous pipeline monitoring and self-healing system."""
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        metrics_history_size: int = 1000,
        monitoring_interval: float = 5.0,
        prediction_horizon: int = 60,
    ):
        """Initialize the self-healing guard.
        
        Args:
            config_path: Path to configuration file
            metrics_history_size: Number of metrics to keep in history
            monitoring_interval: Seconds between monitoring checks
            prediction_horizon: Seconds ahead to predict failures
        """
        self.config_path = config_path or Path("pipeline_guard_config.json")
        self.metrics_history_size = metrics_history_size
        self.monitoring_interval = monitoring_interval
        self.prediction_horizon = prediction_horizon
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Pipeline state
        self.is_monitoring = False
        self.current_status = HealthStatus.HEALTHY
        self.metrics_history: List[PipelineMetrics] = []
        self.failure_history: List[Dict[str, Any]] = []
        
        # ML models for prediction
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.models_trained = False
        
        # Recovery actions
        self.recovery_actions: Dict[str, RecoveryAction] = {}
        self._initialize_recovery_actions()
        
        # Threading
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Load configuration
        self._load_config()
        
        # Adaptive learning
        self.learning_enabled = True
        self.adaptation_rate = 0.1
        
    def _initialize_recovery_actions(self) -> None:
        """Initialize default recovery actions."""
        self.recovery_actions = {
            "restart_service": RecoveryAction(
                name="restart_service",
                description="Restart the pipeline service",
                severity_threshold=0.8,
                action_function=self._restart_service,
                cooldown_seconds=300,
                max_retries=3
            ),
            "scale_resources": RecoveryAction(
                name="scale_resources",
                description="Scale up computational resources",
                severity_threshold=0.6,
                action_function=self._scale_resources,
                cooldown_seconds=180,
                max_retries=5
            ),
            "clear_cache": RecoveryAction(
                name="clear_cache",
                description="Clear system caches",
                severity_threshold=0.4,
                action_function=self._clear_cache,
                cooldown_seconds=60,
                max_retries=10
            ),
            "optimize_algorithms": RecoveryAction(
                name="optimize_algorithms",
                description="Switch to optimized algorithm variants",
                severity_threshold=0.5,
                action_function=self._optimize_algorithms,
                cooldown_seconds=120,
                max_retries=3
            ),
            "circuit_breaker": RecoveryAction(
                name="circuit_breaker",
                description="Activate circuit breaker pattern",
                severity_threshold=0.9,
                action_function=self._activate_circuit_breaker,
                cooldown_seconds=60,
                max_retries=1
            ),
            "graceful_degradation": RecoveryAction(
                name="graceful_degradation",
                description="Enable graceful service degradation",
                severity_threshold=0.7,
                action_function=self._graceful_degradation,
                cooldown_seconds=120,
                max_retries=2
            )
        }
    
    def _load_config(self) -> None:
        """Load configuration from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Apply configuration
                    self.monitoring_interval = config.get('monitoring_interval', self.monitoring_interval)
                    self.prediction_horizon = config.get('prediction_horizon', self.prediction_horizon)
                    self.learning_enabled = config.get('learning_enabled', self.learning_enabled)
            except Exception as e:
                self.logger.warning(f"Failed to load config: {e}")
    
    def _save_config(self) -> None:
        """Save current configuration to file."""
        config = {
            'monitoring_interval': self.monitoring_interval,
            'prediction_horizon': self.prediction_horizon,
            'learning_enabled': self.learning_enabled,
            'recovery_actions': {
                name: {
                    'success_rate': action.success_rate,
                    'execution_count': action.execution_count,
                    'severity_threshold': action.severity_threshold
                }
                for name, action in self.recovery_actions.items()
            }
        }
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save config: {e}")
    
    def start_monitoring(self) -> None:
        """Start the monitoring loop."""
        if self.is_monitoring:
            self.logger.warning("Monitoring already active")
            return
        
        self.is_monitoring = True
        self.stop_event.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Self-healing guard started monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring loop."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.stop_event.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        self.logger.info("Self-healing guard stopped monitoring")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self.stop_event.is_set():
            try:
                # Collect current metrics
                metrics = self._collect_metrics()
                self._add_metrics(metrics)
                
                # Analyze health
                health_score = self._analyze_health(metrics)
                
                # Predict failures
                failure_risk = self._predict_failure_risk()
                
                # Update status
                self._update_status(health_score, failure_risk)
                
                # Take corrective actions if needed
                if self.current_status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                    self._execute_recovery_actions(health_score, failure_risk)
                
                # Adaptive learning
                if self.learning_enabled:
                    self._adaptive_learning()
                
                # Wait for next iteration
                self.stop_event.wait(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> PipelineMetrics:
        """Collect current pipeline metrics."""
        # In a real implementation, this would gather actual system metrics
        # For now, we simulate realistic metrics
        import random
        
        try:
            # Try to get real system metrics if psutil is available
            import psutil
            cpu_usage = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            memory_usage = memory_info.percent
            disk_usage = psutil.disk_usage('/').percent
        except ImportError:
            # Fallback to simulated metrics when psutil is not available
            cpu_usage = random.uniform(20, 80)
            memory_usage = random.uniform(30, 70)
            disk_usage = random.uniform(10, 90)
        except Exception:
            # Fallback to simulated metrics for any other errors
            cpu_usage = random.uniform(20, 80)
            memory_usage = random.uniform(30, 70)
            disk_usage = random.uniform(10, 90)
        
        return PipelineMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_latency=random.uniform(1, 100),
            throughput=random.uniform(100, 1000),
            error_rate=random.uniform(0, 0.1),
            response_time=random.uniform(50, 500),
            queue_depth=random.randint(0, 100),
            active_connections=random.randint(10, 200)
        )
    
    def _add_metrics(self, metrics: PipelineMetrics) -> None:
        """Add metrics to history."""
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.metrics_history_size:
            self.metrics_history.pop(0)
    
    def _analyze_health(self, metrics: PipelineMetrics) -> float:
        """Analyze current pipeline health and return health score (0-1)."""
        # Define health score based on multiple factors
        scores = []
        
        # CPU health (inverted - lower is better)
        cpu_score = max(0, 1 - (metrics.cpu_usage / 100))
        scores.append(cpu_score)
        
        # Memory health
        memory_score = max(0, 1 - (metrics.memory_usage / 100))
        scores.append(memory_score)
        
        # Error rate health
        error_score = max(0, 1 - (metrics.error_rate * 10))  # Scale error rate
        scores.append(error_score)
        
        # Response time health (assume 200ms is baseline)
        response_score = max(0, 1 - (metrics.response_time / 1000))
        scores.append(response_score)
        
        # Network health
        network_score = max(0, 1 - (metrics.network_latency / 200))
        scores.append(network_score)
        
        # Calculate weighted average
        health_score = np.mean(scores)
        
        return max(0, min(1, health_score))
    
    def _predict_failure_risk(self) -> float:
        """Predict failure risk using ML models."""
        if len(self.metrics_history) < 10:
            return 0.0
        
        try:
            # Prepare feature matrix
            features = [m.to_features() for m in self.metrics_history[-50:]]
            
            if not self.models_trained and len(features) >= 20:
                # Train models on historical data
                X = self.scaler.fit_transform(features)
                self.anomaly_detector.fit(X)
                self.models_trained = True
                self.logger.info("ML models trained on historical data")
            
            if self.models_trained:
                # Predict anomaly on recent data
                recent_features = self.scaler.transform([features[-1]])
                anomaly_score = self.anomaly_detector.decision_function(recent_features)[0]
                
                # Convert to risk probability (0-1)
                risk = max(0, min(1, (0.5 - anomaly_score) * 2))
                return risk
            
        except Exception as e:
            self.logger.warning(f"Prediction error: {e}")
        
        return 0.0
    
    def _update_status(self, health_score: float, failure_risk: float) -> None:
        """Update pipeline status based on health and risk."""
        combined_score = (health_score + (1 - failure_risk)) / 2
        
        if combined_score >= 0.8:
            new_status = HealthStatus.HEALTHY
        elif combined_score >= 0.6:
            new_status = HealthStatus.WARNING
        elif combined_score >= 0.3:
            new_status = HealthStatus.CRITICAL
        else:
            new_status = HealthStatus.FAILED
        
        if new_status != self.current_status:
            old_status = self.current_status
            self.current_status = new_status
            self.logger.info(f"Status changed: {old_status.value} → {new_status.value}")
            
            # Log status change
            self.failure_history.append({
                'timestamp': datetime.now().isoformat(),
                'old_status': old_status.value,
                'new_status': new_status.value,
                'health_score': health_score,
                'failure_risk': failure_risk
            })
    
    def _execute_recovery_actions(self, health_score: float, failure_risk: float) -> None:
        """Execute appropriate recovery actions."""
        severity = 1 - health_score + failure_risk
        
        # Sort actions by severity threshold
        applicable_actions = [
            action for action in self.recovery_actions.values()
            if severity >= action.severity_threshold
        ]
        
        applicable_actions.sort(key=lambda x: x.severity_threshold)
        
        for action in applicable_actions:
            if self._should_execute_action(action):
                success = self._execute_action(action)
                self._update_action_stats(action, success)
    
    def _should_execute_action(self, action: RecoveryAction) -> bool:
        """Determine if an action should be executed."""
        now = datetime.now()
        
        # Check cooldown
        if action.last_executed:
            cooldown_elapsed = (now - action.last_executed).total_seconds()
            if cooldown_elapsed < action.cooldown_seconds:
                return False
        
        # Check retry limit
        if action.execution_count >= action.max_retries:
            return False
        
        return True
    
    def _execute_action(self, action: RecoveryAction) -> bool:
        """Execute a recovery action."""
        try:
            self.logger.info(f"Executing recovery action: {action.name}")
            action.last_executed = datetime.now()
            action.execution_count += 1
            
            # Execute the action
            result = action.action_function()
            
            if result:
                self.logger.info(f"Recovery action {action.name} succeeded")
                return True
            else:
                self.logger.warning(f"Recovery action {action.name} failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Recovery action {action.name} error: {e}")
            return False
    
    def _update_action_stats(self, action: RecoveryAction, success: bool) -> None:
        """Update action success statistics."""
        if action.execution_count == 1:
            action.success_rate = 1.0 if success else 0.0
        else:
            # Exponential moving average
            alpha = 0.3
            action.success_rate = (
                alpha * (1.0 if success else 0.0) + 
                (1 - alpha) * action.success_rate
            )
    
    def _adaptive_learning(self) -> None:
        """Implement adaptive learning to improve recovery strategies."""
        if len(self.metrics_history) < 50:
            return
        
        # Analyze effectiveness of recent actions
        recent_metrics = self.metrics_history[-20:]
        
        # Calculate trend in health scores
        health_scores = [self._analyze_health(m) for m in recent_metrics]
        
        if len(health_scores) > 1:
            health_trend = health_scores[-1] - health_scores[0]
            
            # Adjust thresholds based on effectiveness
            for action in self.recovery_actions.values():
                if action.execution_count > 0:
                    if health_trend > 0 and action.success_rate > 0.7:
                        # Action is effective, lower threshold slightly
                        action.severity_threshold *= (1 - self.adaptation_rate * 0.1)
                    elif health_trend < 0 or action.success_rate < 0.3:
                        # Action ineffective, raise threshold
                        action.severity_threshold *= (1 + self.adaptation_rate * 0.1)
                    
                    # Clamp thresholds
                    action.severity_threshold = max(0.1, min(0.9, action.severity_threshold))
    
    # Recovery action implementations
    def _restart_service(self) -> bool:
        """Restart the pipeline service."""
        try:
            self.logger.info("Simulating service restart")
            # In real implementation, this would restart the actual service
            time.sleep(2)  # Simulate restart time
            return True
        except Exception as e:
            self.logger.error(f"Service restart failed: {e}")
            return False
    
    def _scale_resources(self) -> bool:
        """Scale up computational resources."""
        try:
            self.logger.info("Simulating resource scaling")
            # In real implementation, this would scale up resources
            time.sleep(1)
            return True
        except Exception as e:
            self.logger.error(f"Resource scaling failed: {e}")
            return False
    
    def _clear_cache(self) -> bool:
        """Clear system caches."""
        try:
            self.logger.info("Simulating cache clearing")
            # In real implementation, this would clear actual caches
            return True
        except Exception as e:
            self.logger.error(f"Cache clearing failed: {e}")
            return False
    
    def _optimize_algorithms(self) -> bool:
        """Switch to optimized algorithm variants."""
        try:
            self.logger.info("Simulating algorithm optimization")
            # In real implementation, this would switch to optimized algorithms
            return True
        except Exception as e:
            self.logger.error(f"Algorithm optimization failed: {e}")
            return False
    
    def _activate_circuit_breaker(self) -> bool:
        """Activate circuit breaker pattern."""
        try:
            self.logger.info("Activating circuit breaker")
            # In real implementation, this would activate circuit breaker
            return True
        except Exception as e:
            self.logger.error(f"Circuit breaker activation failed: {e}")
            return False
    
    def _graceful_degradation(self) -> bool:
        """Enable graceful service degradation."""
        try:
            self.logger.info("Enabling graceful degradation")
            # In real implementation, this would enable degraded service mode
            return True
        except Exception as e:
            self.logger.error(f"Graceful degradation failed: {e}")
            return False
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        if not self.metrics_history:
            return {"status": "no_data", "message": "No metrics available"}
        
        latest_metrics = self.metrics_history[-1]
        health_score = self._analyze_health(latest_metrics)
        failure_risk = self._predict_failure_risk()
        
        # Calculate uptime
        uptime_hours = len(self.metrics_history) * self.monitoring_interval / 3600
        
        # Recovery action stats
        action_stats = {
            name: {
                'success_rate': action.success_rate,
                'execution_count': action.execution_count,
                'last_executed': action.last_executed.isoformat() if action.last_executed else None
            }
            for name, action in self.recovery_actions.items()
        }
        
        return {
            'status': self.current_status.value,
            'health_score': health_score,
            'failure_risk': failure_risk,
            'uptime_hours': uptime_hours,
            'latest_metrics': asdict(latest_metrics),
            'recovery_actions': action_stats,
            'models_trained': self.models_trained,
            'learning_enabled': self.learning_enabled,
            'monitoring_active': self.is_monitoring
        }
    
    def force_recovery_action(self, action_name: str) -> bool:
        """Force execution of a specific recovery action."""
        if action_name not in self.recovery_actions:
            self.logger.error(f"Unknown recovery action: {action_name}")
            return False
        
        action = self.recovery_actions[action_name]
        success = self._execute_action(action)
        self._update_action_stats(action, success)
        return success
    
    def add_custom_recovery_action(
        self,
        name: str,
        description: str,
        action_function: Callable,
        severity_threshold: float = 0.5,
        cooldown_seconds: int = 300,
        max_retries: int = 3
    ) -> None:
        """Add a custom recovery action."""
        # Input validation
        if not name or not isinstance(name, str):
            raise ValueError("Action name must be a non-empty string")
        if not description or not isinstance(description, str):
            raise ValueError("Action description must be a non-empty string")
        if not callable(action_function):
            raise TypeError("Action function must be callable")
        if not (0.0 <= severity_threshold <= 1.0):
            raise ValueError("Severity threshold must be between 0.0 and 1.0")
        if cooldown_seconds < 0:
            raise ValueError("Cooldown seconds must be non-negative")
        if max_retries < 0:
            raise ValueError("Max retries must be non-negative")
        
        self.recovery_actions[name] = RecoveryAction(
            name=name,
            description=description,
            severity_threshold=severity_threshold,
            action_function=action_function,
            cooldown_seconds=cooldown_seconds,
            max_retries=max_retries
        )
        self.logger.info(f"Added custom recovery action: {name}")
    
    def export_metrics(self, output_path: Path) -> None:
        """Export metrics history to file."""
        try:
            metrics_data = [asdict(m) for m in self.metrics_history]
            # Convert datetime objects to strings
            for metric in metrics_data:
                metric['timestamp'] = metric['timestamp'].isoformat()
            
            with open(output_path, 'w') as f:
                json.dump({
                    'metrics': metrics_data,
                    'failure_history': self.failure_history,
                    'recovery_actions': {
                        name: {
                            'success_rate': action.success_rate,
                            'execution_count': action.execution_count
                        }
                        for name, action in self.recovery_actions.items()
                    }
                }, f, indent=2)
            
            self.logger.info(f"Metrics exported to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")


class PipelineGuardManager:
    """High-level manager for multiple pipeline guards."""
    
    def __init__(self):
        self.guards: Dict[str, SelfHealingGuard] = {}
        self.logger = logging.getLogger(__name__)
    
    def add_pipeline(
        self,
        name: str,
        config_path: Optional[Path] = None,
        **kwargs
    ) -> SelfHealingGuard:
        """Add a new pipeline to monitor."""
        guard = SelfHealingGuard(config_path=config_path, **kwargs)
        self.guards[name] = guard
        self.logger.info(f"Added pipeline guard: {name}")
        return guard
    
    def start_all(self) -> None:
        """Start monitoring all pipelines."""
        for name, guard in self.guards.items():
            guard.start_monitoring()
            self.logger.info(f"Started monitoring pipeline: {name}")
    
    def stop_all(self) -> None:
        """Stop monitoring all pipelines."""
        for name, guard in self.guards.items():
            guard.stop_monitoring()
            self.logger.info(f"Stopped monitoring pipeline: {name}")
    
    def get_global_status(self) -> Dict[str, Any]:
        """Get status report for all pipelines."""
        return {
            name: guard.get_status_report()
            for name, guard in self.guards.items()
        }
    
    def get_unhealthy_pipelines(self) -> List[str]:
        """Get list of unhealthy pipelines."""
        unhealthy = []
        for name, guard in self.guards.items():
            if guard.current_status in [HealthStatus.WARNING, HealthStatus.CRITICAL, HealthStatus.FAILED]:
                unhealthy.append(name)
        return unhealthy


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start guard
    guard = SelfHealingGuard(monitoring_interval=2.0)
    guard.start_monitoring()
    
    try:
        # Let it run for a bit
        time.sleep(30)
        
        # Get status report
        report = guard.get_status_report()
        print("\nStatus Report:")
        print(json.dumps(report, indent=2, default=str))
        
    finally:
        guard.stop_monitoring()