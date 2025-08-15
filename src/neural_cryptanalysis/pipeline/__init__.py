"""Pipeline monitoring and self-healing modules.

This package provides autonomous pipeline monitoring, failure detection,
and self-healing capabilities for neural cryptanalysis pipelines.

Key Components:
- SelfHealingGuard: Main monitoring and recovery system
- PipelineGuardManager: Manager for multiple pipeline instances
- AdaptiveLearningEngine: ML-driven continuous improvement
- AdvancedMonitoringSystem: Comprehensive monitoring and alerting
- ResilienceManager: Fault tolerance and recovery patterns
- AutoScaler: Intelligent auto-scaling with ML predictions
"""

from .self_healing_guard import (
    SelfHealingGuard,
    PipelineGuardManager,
    HealthStatus,
    FailureType,
    PipelineMetrics,
    RecoveryAction
)

# Try to import optional components with graceful fallback
try:
    from .adaptive_learning import (
        AdaptiveLearningEngine,
        ReinforcementLearner,
        OnlineLearner,
        PatternRecognizer,
        LearningState,
        ActionOutcome
    )
    _ADAPTIVE_LEARNING_AVAILABLE = True
except ImportError:
    _ADAPTIVE_LEARNING_AVAILABLE = False

try:
    from .monitoring import (
        AdvancedMonitoringSystem,
        AlertManager,
        MetricsCollector,
        AnomalyDetector,
        DashboardGenerator,
        Alert,
        AlertSeverity,
        MonitoringConfig
    )
    _MONITORING_AVAILABLE = True
except ImportError:
    _MONITORING_AVAILABLE = False

try:
    from .resilience import (
        ResilienceManager,
        CircuitBreaker,
        Bulkhead,
        RetryMechanism,
        HealthChecker,
        GracefulDegradation,
        DisasterRecovery,
        CircuitBreakerConfig,
        RetryConfig,
        BulkheadConfig
    )
    _RESILIENCE_AVAILABLE = True
except ImportError:
    _RESILIENCE_AVAILABLE = False

try:
    from .auto_scaling import (
        AutoScaler,
        PredictiveScaler,
        LoadBalancer,
        ScalingPolicy,
        ResourceMetrics,
        ScalingDecision,
        ResourceType,
        ScalingDirection,
        LoadBalancingConfig
    )
    _AUTO_SCALING_AVAILABLE = True
except ImportError:
    _AUTO_SCALING_AVAILABLE = False

# Build dynamic __all__ based on available components
__all__ = [
    # Core self-healing (always available)
    "SelfHealingGuard",
    "PipelineGuardManager", 
    "HealthStatus",
    "FailureType",
    "PipelineMetrics",
    "RecoveryAction",
]

# Add optional components if available
if _ADAPTIVE_LEARNING_AVAILABLE:
    __all__.extend([
        "AdaptiveLearningEngine",
        "ReinforcementLearner",
        "OnlineLearner",
        "PatternRecognizer",
        "LearningState",
        "ActionOutcome",
    ])

if _MONITORING_AVAILABLE:
    __all__.extend([
        "AdvancedMonitoringSystem",
        "AlertManager",
        "MetricsCollector",
        "AnomalyDetector",
        "DashboardGenerator",
        "Alert",
        "AlertSeverity",
        "MonitoringConfig",
    ])

if _RESILIENCE_AVAILABLE:
    __all__.extend([
        "ResilienceManager",
        "CircuitBreaker",
        "Bulkhead",
        "RetryMechanism",
        "HealthChecker",
        "GracefulDegradation",
        "DisasterRecovery",
        "CircuitBreakerConfig",
        "RetryConfig",
        "BulkheadConfig",
    ])

if _AUTO_SCALING_AVAILABLE:
    __all__.extend([
        "AutoScaler",
        "PredictiveScaler",
        "LoadBalancer",
        "ScalingPolicy",
        "ResourceMetrics",
        "ScalingDecision",
        "ResourceType",
        "ScalingDirection",
        "LoadBalancingConfig"
    ])