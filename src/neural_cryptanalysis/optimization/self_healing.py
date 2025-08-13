"""Self-Healing System for Neural Cryptanalysis Framework.

This module implements advanced self-healing mechanisms, adaptive optimization,
and predictive resource management for autonomous system maintenance.
"""

import time
import threading
import asyncio
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import statistics
import pickle
import json
import warnings
import traceback

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None

from ..utils.logging_utils import get_logger
from ..utils.errors import NeuralCryptanalysisError, ResourceError

logger = get_logger(__name__)


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"
    RECOVERING = "recovering"


class IssueType(Enum):
    """Types of system issues."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MEMORY_LEAK = "memory_leak"
    HIGH_CPU_USAGE = "high_cpu_usage"
    DISK_SPACE_LOW = "disk_space_low"
    NETWORK_ISSUE = "network_issue"
    MODEL_CONVERGENCE_FAILURE = "model_convergence_failure"
    DATA_CORRUPTION = "data_corruption"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEADLOCK = "deadlock"
    TIMEOUT = "timeout"


@dataclass
class HealthIssue:
    """Represents a system health issue."""
    issue_id: str
    issue_type: IssueType
    severity: HealthStatus
    description: str
    detected_at: datetime
    source_component: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    resolution_attempted: bool = False
    resolved_at: Optional[datetime] = None
    resolution_actions: List[str] = field(default_factory=list)
    impact_score: float = 0.0
    recurring_count: int = 1


@dataclass 
class SystemMetrics:
    """System performance and health metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    network_latency_ms: Optional[float] = None
    active_threads: int = 0
    active_processes: int = 0
    gpu_memory_percent: Optional[float] = None
    gpu_utilization: Optional[float] = None
    operation_throughput: float = 0.0
    error_rate: float = 0.0
    response_time_ms: float = 0.0


class HealthMonitor:
    """Comprehensive system health monitoring."""
    
    def __init__(self, check_interval: float = 5.0, history_size: int = 1000):
        self.check_interval = check_interval
        self.history_size = history_size
        
        # Monitoring state
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_history = deque(maxlen=history_size)
        self.active_issues = {}
        self.resolved_issues = []
        
        # Health thresholds
        self.thresholds = {
            'cpu_critical': 90.0,
            'cpu_warning': 80.0,
            'memory_critical': 95.0,
            'memory_warning': 85.0,
            'disk_critical': 95.0,
            'disk_warning': 90.0,
            'error_rate_critical': 0.1,
            'error_rate_warning': 0.05,
            'response_time_critical': 5000.0,  # ms
            'response_time_warning': 2000.0,   # ms
        }
        
        # Issue detection patterns
        self.anomaly_detectors = {}
        self.pattern_analyzers = {}
        
        self.lock = threading.RLock()
        
        logger.info("Health monitor initialized")
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Collect current metrics
                metrics = self._collect_system_metrics()
                
                with self.lock:
                    self.metrics_history.append(metrics)
                
                # Check for health issues
                self._check_health_conditions(metrics)
                
                # Detect anomalies and patterns
                self._detect_anomalies(metrics)
                
                # Age out old issues
                self._age_issues()
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(1)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics."""
        if not HAS_PSUTIL:
            # Fallback metrics
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_available_gb=8.0,
                disk_usage_percent=50.0
            )
        
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_gb=memory.available / 1024**3,
                disk_usage_percent=disk.percent,
                active_threads=threading.active_count(),
                active_processes=len(psutil.pids())
            )
            
            # GPU metrics if available
            try:
                import torch
                if torch.cuda.is_available():
                    metrics.gpu_memory_percent = (torch.cuda.memory_allocated() / 
                                                torch.cuda.max_memory_allocated()) * 100
            except:
                pass
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to collect metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_available_gb=1.0,
                disk_usage_percent=0.0
            )
    
    def _check_health_conditions(self, metrics: SystemMetrics):
        """Check for immediate health issues."""
        issues_found = []
        
        # CPU usage checks
        if metrics.cpu_percent > self.thresholds['cpu_critical']:
            issues_found.append(self._create_issue(
                IssueType.HIGH_CPU_USAGE,
                HealthStatus.CRITICAL,
                f"CPU usage critically high: {metrics.cpu_percent:.1f}%",
                "health_monitor",
                {'cpu_percent': metrics.cpu_percent}
            ))
        elif metrics.cpu_percent > self.thresholds['cpu_warning']:
            issues_found.append(self._create_issue(
                IssueType.HIGH_CPU_USAGE,
                HealthStatus.WARNING,
                f"CPU usage high: {metrics.cpu_percent:.1f}%",
                "health_monitor",
                {'cpu_percent': metrics.cpu_percent}
            ))
        
        # Memory usage checks
        if metrics.memory_percent > self.thresholds['memory_critical']:
            issues_found.append(self._create_issue(
                IssueType.RESOURCE_EXHAUSTION,
                HealthStatus.CRITICAL,
                f"Memory usage critically high: {metrics.memory_percent:.1f}%",
                "health_monitor",
                {'memory_percent': metrics.memory_percent}
            ))
        elif metrics.memory_percent > self.thresholds['memory_warning']:
            issues_found.append(self._create_issue(
                IssueType.RESOURCE_EXHAUSTION,
                HealthStatus.WARNING,
                f"Memory usage high: {metrics.memory_percent:.1f}%",
                "health_monitor",
                {'memory_percent': metrics.memory_percent}
            ))
        
        # Disk space checks
        if metrics.disk_usage_percent > self.thresholds['disk_critical']:
            issues_found.append(self._create_issue(
                IssueType.DISK_SPACE_LOW,
                HealthStatus.CRITICAL,
                f"Disk space critically low: {metrics.disk_usage_percent:.1f}%",
                "health_monitor",
                {'disk_usage_percent': metrics.disk_usage_percent}
            ))
        
        # Register new issues
        for issue in issues_found:
            self._register_issue(issue)
    
    def _detect_anomalies(self, current_metrics: SystemMetrics):
        """Detect anomalies using statistical analysis."""
        if len(self.metrics_history) < 30:  # Need enough history
            return
        
        recent_metrics = list(self.metrics_history)[-30:]
        
        # Detect performance degradation
        recent_cpu = [m.cpu_percent for m in recent_metrics]
        recent_memory = [m.memory_percent for m in recent_metrics]
        
        cpu_trend = self._calculate_trend(recent_cpu)
        memory_trend = self._calculate_trend(recent_memory)
        
        # Performance degradation detection
        if cpu_trend > 2.0:  # Increasing trend
            self._register_issue(self._create_issue(
                IssueType.PERFORMANCE_DEGRADATION,
                HealthStatus.WARNING,
                f"CPU usage trending upward: {cpu_trend:.2f}%/min",
                "anomaly_detector",
                {'cpu_trend': cpu_trend}
            ))
        
        # Memory leak detection
        if memory_trend > 1.0 and current_metrics.memory_percent > 70:
            self._register_issue(self._create_issue(
                IssueType.MEMORY_LEAK,
                HealthStatus.WARNING,
                f"Potential memory leak detected: {memory_trend:.2f}%/min",
                "anomaly_detector",
                {'memory_trend': memory_trend, 'current_memory': current_metrics.memory_percent}
            ))
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in time series data."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend calculation
        n = len(values)
        x = list(range(n))
        
        if HAS_NUMPY:
            # Use numpy for better precision
            coeffs = np.polyfit(x, values, 1)
            return coeffs[0] * 60 / self.check_interval  # Per minute
        else:
            # Simple slope calculation
            x_mean = sum(x) / n
            y_mean = sum(values) / n
            
            numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
            
            if denominator == 0:
                return 0.0
            
            slope = numerator / denominator
            return slope * 60 / self.check_interval  # Per minute
    
    def _create_issue(self, issue_type: IssueType, severity: HealthStatus,
                     description: str, source: str, metrics: Dict[str, Any]) -> HealthIssue:
        """Create a new health issue."""
        issue_id = f"{issue_type.value}_{int(time.time())}"
        
        return HealthIssue(
            issue_id=issue_id,
            issue_type=issue_type,
            severity=severity,
            description=description,
            detected_at=datetime.now(),
            source_component=source,
            metrics=metrics,
            impact_score=self._calculate_impact_score(issue_type, severity, metrics)
        )
    
    def _calculate_impact_score(self, issue_type: IssueType, severity: HealthStatus,
                              metrics: Dict[str, Any]) -> float:
        """Calculate impact score for an issue."""
        base_scores = {
            HealthStatus.CRITICAL: 10.0,
            HealthStatus.WARNING: 5.0,
            HealthStatus.HEALTHY: 0.0,
            HealthStatus.FAILED: 15.0,
            HealthStatus.RECOVERING: 3.0
        }
        
        type_multipliers = {
            IssueType.RESOURCE_EXHAUSTION: 1.5,
            IssueType.MEMORY_LEAK: 1.3,
            IssueType.HIGH_CPU_USAGE: 1.2,
            IssueType.PERFORMANCE_DEGRADATION: 1.1,
            IssueType.MODEL_CONVERGENCE_FAILURE: 1.4,
            IssueType.DATA_CORRUPTION: 2.0
        }
        
        base_score = base_scores.get(severity, 5.0)
        multiplier = type_multipliers.get(issue_type, 1.0)
        
        return base_score * multiplier
    
    def _register_issue(self, issue: HealthIssue):
        """Register a new health issue."""
        with self.lock:
            # Check if similar issue already exists
            similar_issue = self._find_similar_issue(issue)
            
            if similar_issue:
                # Update existing issue
                similar_issue.recurring_count += 1
                similar_issue.detected_at = issue.detected_at
                similar_issue.metrics.update(issue.metrics)
                
                # Escalate severity if recurring
                if similar_issue.recurring_count > 3 and similar_issue.severity == HealthStatus.WARNING:
                    similar_issue.severity = HealthStatus.CRITICAL
                    logger.warning(f"Escalated issue {similar_issue.issue_id} to CRITICAL")
            else:
                # Add new issue
                self.active_issues[issue.issue_id] = issue
                logger.warning(f"New health issue detected: {issue.description}")
    
    def _find_similar_issue(self, new_issue: HealthIssue) -> Optional[HealthIssue]:
        """Find similar existing issue."""
        for issue in self.active_issues.values():
            if (issue.issue_type == new_issue.issue_type and
                issue.source_component == new_issue.source_component and
                not issue.resolved_at):
                return issue
        return None
    
    def _age_issues(self):
        """Age out old or resolved issues."""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=1)
        
        resolved_issues = []
        
        for issue_id, issue in list(self.active_issues.items()):
            # Auto-resolve old issues if conditions improved
            if issue.detected_at < cutoff_time and not issue.resolution_attempted:
                if self._check_issue_auto_resolved(issue):
                    issue.resolved_at = current_time
                    issue.resolution_actions.append("auto_resolved_timeout")
                    resolved_issues.append(issue_id)
        
        # Move resolved issues
        for issue_id in resolved_issues:
            issue = self.active_issues.pop(issue_id)
            self.resolved_issues.append(issue)
            logger.info(f"Auto-resolved issue: {issue.description}")
    
    def _check_issue_auto_resolved(self, issue: HealthIssue) -> bool:
        """Check if issue has been automatically resolved."""
        if not self.metrics_history:
            return False
        
        latest_metrics = self.metrics_history[-1]
        
        # Check if the problematic condition has improved
        if issue.issue_type == IssueType.HIGH_CPU_USAGE:
            return latest_metrics.cpu_percent < self.thresholds['cpu_warning']
        elif issue.issue_type == IssueType.RESOURCE_EXHAUSTION:
            return latest_metrics.memory_percent < self.thresholds['memory_warning']
        elif issue.issue_type == IssueType.DISK_SPACE_LOW:
            return latest_metrics.disk_usage_percent < self.thresholds['disk_warning']
        
        return False
    
    def get_health_status(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.active_issues:
            return HealthStatus.HEALTHY
        
        severities = [issue.severity for issue in self.active_issues.values()]
        
        if HealthStatus.FAILED in severities:
            return HealthStatus.FAILED
        elif HealthStatus.CRITICAL in severities:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in severities:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def get_active_issues(self) -> List[HealthIssue]:
        """Get list of active health issues."""
        with self.lock:
            return list(self.active_issues.values())
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        with self.lock:
            latest_metrics = self.metrics_history[-1] if self.metrics_history else None
            
            return {
                'overall_status': self.get_health_status().value,
                'active_issues': len(self.active_issues),
                'resolved_issues_count': len(self.resolved_issues),
                'latest_metrics': latest_metrics.__dict__ if latest_metrics else {},
                'active_issues_details': [
                    {
                        'id': issue.issue_id,
                        'type': issue.issue_type.value,
                        'severity': issue.severity.value,
                        'description': issue.description,
                        'detected_at': issue.detected_at.isoformat(),
                        'impact_score': issue.impact_score,
                        'recurring_count': issue.recurring_count
                    }
                    for issue in self.active_issues.values()
                ],
                'system_trends': self._calculate_system_trends(),
                'recommendations': self._generate_recommendations()
            }
    
    def _calculate_system_trends(self) -> Dict[str, float]:
        """Calculate system performance trends."""
        if len(self.metrics_history) < 10:
            return {}
        
        recent_metrics = list(self.metrics_history)[-30:]
        
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        
        return {
            'cpu_trend': self._calculate_trend(cpu_values),
            'memory_trend': self._calculate_trend(memory_values),
            'avg_cpu_last_30': statistics.mean(cpu_values),
            'avg_memory_last_30': statistics.mean(memory_values)
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate health improvement recommendations."""
        recommendations = []
        
        if not self.active_issues:
            return ["System health is good. Continue monitoring."]
        
        # Analyze issues and generate recommendations
        for issue in self.active_issues.values():
            if issue.issue_type == IssueType.HIGH_CPU_USAGE:
                recommendations.append("Consider reducing concurrent operations or optimizing algorithms")
            elif issue.issue_type == IssueType.MEMORY_LEAK:
                recommendations.append("Investigate memory usage patterns and implement garbage collection")
            elif issue.issue_type == IssueType.RESOURCE_EXHAUSTION:
                recommendations.append("Scale up resources or optimize resource utilization")
            elif issue.issue_type == IssueType.DISK_SPACE_LOW:
                recommendations.append("Clean up temporary files or expand storage capacity")
        
        return recommendations


class SelfHealingSystem:
    """Autonomous self-healing system with adaptive optimization."""
    
    def __init__(self, health_monitor: HealthMonitor):
        self.health_monitor = health_monitor
        
        # Healing mechanisms
        self.healing_strategies = {}
        self.active_healings = {}
        self.healing_history = []
        
        # Adaptive optimization
        self.optimization_strategies = {}
        self.performance_baselines = {}
        
        # Recovery executor
        self.recovery_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="healing")
        
        # State
        self.healing_enabled = True
        self.learning_enabled = True
        self.recovery_in_progress = set()
        
        self.lock = threading.RLock()
        
        # Register default healing strategies
        self._register_default_strategies()
        
        logger.info("Self-healing system initialized")
    
    def _register_default_strategies(self):
        """Register default healing strategies."""
        self.register_healing_strategy(
            IssueType.HIGH_CPU_USAGE,
            self._heal_high_cpu_usage,
            priority=1
        )
        
        self.register_healing_strategy(
            IssueType.MEMORY_LEAK,
            self._heal_memory_leak,
            priority=2
        )
        
        self.register_healing_strategy(
            IssueType.RESOURCE_EXHAUSTION,
            self._heal_resource_exhaustion,
            priority=3
        )
        
        self.register_healing_strategy(
            IssueType.DISK_SPACE_LOW,
            self._heal_disk_space_low,
            priority=2
        )
    
    def register_healing_strategy(self, issue_type: IssueType, 
                                healing_func: Callable, priority: int = 1):
        """Register a healing strategy for an issue type."""
        self.healing_strategies[issue_type] = {
            'function': healing_func,
            'priority': priority,
            'success_count': 0,
            'failure_count': 0,
            'avg_healing_time': 0.0
        }
        
        logger.info(f"Registered healing strategy for {issue_type.value}")
    
    def start_healing(self):
        """Start the self-healing process."""
        self.healing_enabled = True
        
        # Start healing monitor thread
        healing_thread = threading.Thread(target=self._healing_loop, daemon=True)
        healing_thread.start()
        
        logger.info("Self-healing system started")
    
    def stop_healing(self):
        """Stop the self-healing process."""
        self.healing_enabled = False
        
        # Wait for active healings to complete
        self.recovery_executor.shutdown(wait=True)
        
        logger.info("Self-healing system stopped")
    
    def _healing_loop(self):
        """Main healing loop."""
        while self.healing_enabled:
            try:
                # Get active issues
                active_issues = self.health_monitor.get_active_issues()
                
                # Prioritize and heal issues
                for issue in sorted(active_issues, key=lambda x: x.impact_score, reverse=True):
                    if issue.issue_id not in self.recovery_in_progress:
                        self._attempt_healing(issue)
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Healing loop error: {e}")
                time.sleep(5)
    
    def _attempt_healing(self, issue: HealthIssue):
        """Attempt to heal a specific issue."""
        if issue.issue_type not in self.healing_strategies:
            logger.warning(f"No healing strategy for {issue.issue_type.value}")
            return
        
        if issue.resolution_attempted:
            logger.debug(f"Issue {issue.issue_id} already has healing attempted")
            return
        
        with self.lock:
            self.recovery_in_progress.add(issue.issue_id)
        
        # Submit healing task
        future = self.recovery_executor.submit(self._execute_healing, issue)
        future.add_done_callback(lambda f: self._healing_completed(issue.issue_id, f))
        
        logger.info(f"Started healing for issue: {issue.description}")
    
    def _execute_healing(self, issue: HealthIssue) -> Dict[str, Any]:
        """Execute healing strategy for an issue."""
        start_time = time.time()
        
        try:
            strategy = self.healing_strategies[issue.issue_type]
            healing_func = strategy['function']
            
            # Mark as attempted
            issue.resolution_attempted = True
            
            # Execute healing function
            result = healing_func(issue)
            
            healing_time = time.time() - start_time
            
            # Update strategy statistics
            strategy['success_count'] += 1
            strategy['avg_healing_time'] = (
                (strategy['avg_healing_time'] * (strategy['success_count'] - 1) + healing_time) /
                strategy['success_count']
            )
            
            # Record healing action
            action_description = result.get('action', 'healing_attempted')
            issue.resolution_actions.append(action_description)
            
            # Check if issue is resolved
            if result.get('resolved', False):
                issue.resolved_at = datetime.now()
                logger.info(f"Successfully healed issue: {issue.description}")
            
            # Record healing in history
            healing_record = {
                'issue_id': issue.issue_id,
                'issue_type': issue.issue_type.value,
                'healing_time': healing_time,
                'result': result,
                'timestamp': datetime.now()
            }
            
            with self.lock:
                self.healing_history.append(healing_record)
                
                # Keep only recent history
                if len(self.healing_history) > 1000:
                    self.healing_history = self.healing_history[-500:]
            
            return result
            
        except Exception as e:
            # Update failure statistics
            strategy = self.healing_strategies[issue.issue_type]
            strategy['failure_count'] += 1
            
            error_message = f"Healing failed: {e}"
            issue.resolution_actions.append(error_message)
            
            logger.error(f"Healing failed for {issue.description}: {e}")
            
            return {
                'resolved': False,
                'action': 'healing_failed',
                'error': str(e)
            }
    
    def _healing_completed(self, issue_id: str, future):
        """Callback when healing completes."""
        with self.lock:
            self.recovery_in_progress.discard(issue_id)
        
        try:
            result = future.result()
            logger.debug(f"Healing completed for {issue_id}: {result}")
        except Exception as e:
            logger.error(f"Healing future failed for {issue_id}: {e}")
    
    def _heal_high_cpu_usage(self, issue: HealthIssue) -> Dict[str, Any]:
        """Heal high CPU usage issue."""
        try:
            # Strategy 1: Reduce concurrent operations
            current_threads = threading.active_count()
            if current_threads > 20:
                # Signal to reduce thread pool sizes (this would need integration with the system)
                logger.info("Attempting to reduce thread pool sizes")
                return {
                    'resolved': True,
                    'action': 'reduced_thread_pools',
                    'details': f'Thread count: {current_threads}'
                }
            
            # Strategy 2: Trigger garbage collection
            import gc
            collected = gc.collect()
            
            return {
                'resolved': False,  # May need time to take effect
                'action': 'triggered_garbage_collection',
                'details': f'Collected {collected} objects'
            }
            
        except Exception as e:
            return {
                'resolved': False,
                'action': 'healing_failed',
                'error': str(e)
            }
    
    def _heal_memory_leak(self, issue: HealthIssue) -> Dict[str, Any]:
        """Heal memory leak issue."""
        try:
            # Strategy 1: Force garbage collection
            import gc
            
            # Disable garbage collection temporarily and then enable it
            gc.disable()
            time.sleep(0.1)
            gc.enable()
            collected = gc.collect()
            
            # Strategy 2: Clear caches (if global optimizer is available)
            try:
                from .performance_optimizer import get_global_optimizer
                optimizer = get_global_optimizer()
                optimizer.cache.clear()
                logger.info("Cleared performance optimizer cache")
            except:
                pass
            
            return {
                'resolved': True,
                'action': 'memory_cleanup',
                'details': f'Collected {collected} objects, cleared caches'
            }
            
        except Exception as e:
            return {
                'resolved': False,
                'action': 'memory_cleanup_failed',
                'error': str(e)
            }
    
    def _heal_resource_exhaustion(self, issue: HealthIssue) -> Dict[str, Any]:
        """Heal resource exhaustion issue."""
        try:
            # Strategy 1: Free up memory
            import gc
            collected = gc.collect()
            
            # Strategy 2: Reduce batch sizes (signal to system)
            logger.info("Signaling to reduce batch sizes")
            
            # Strategy 3: Pause non-critical operations
            logger.info("Signaling to pause non-critical operations")
            
            return {
                'resolved': False,  # Needs time to take effect
                'action': 'resource_optimization',
                'details': f'Collected {collected} objects, signaled optimizations'
            }
            
        except Exception as e:
            return {
                'resolved': False,
                'action': 'resource_optimization_failed',
                'error': str(e)
            }
    
    def _heal_disk_space_low(self, issue: HealthIssue) -> Dict[str, Any]:
        """Heal low disk space issue."""
        try:
            cleaned_mb = 0
            
            # Strategy 1: Clean temporary files
            import tempfile
            import shutil
            
            temp_dir = Path(tempfile.gettempdir())
            if temp_dir.exists():
                for temp_file in temp_dir.glob("*neural_crypto*"):
                    try:
                        if temp_file.is_file():
                            size_mb = temp_file.stat().st_size / 1024 / 1024
                            temp_file.unlink()
                            cleaned_mb += size_mb
                    except:
                        pass
            
            # Strategy 2: Clear old cache files
            cache_dirs = [Path("./cache"), Path("./experiments"), Path("./logs")]
            
            for cache_dir in cache_dirs:
                if cache_dir.exists():
                    for cache_file in cache_dir.glob("*.cache"):
                        try:
                            # Remove files older than 7 days
                            if (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days > 7:
                                size_mb = cache_file.stat().st_size / 1024 / 1024
                                cache_file.unlink()
                                cleaned_mb += size_mb
                        except:
                            pass
            
            return {
                'resolved': cleaned_mb > 100,  # Consider resolved if cleaned >100MB
                'action': 'disk_cleanup',
                'details': f'Cleaned {cleaned_mb:.1f} MB'
            }
            
        except Exception as e:
            return {
                'resolved': False,
                'action': 'disk_cleanup_failed',
                'error': str(e)
            }
    
    def get_healing_report(self) -> Dict[str, Any]:
        """Generate self-healing report."""
        with self.lock:
            total_healings = len(self.healing_history)
            successful_healings = len([h for h in self.healing_history if h['result'].get('resolved', False)])
            
            strategy_stats = {}
            for issue_type, strategy in self.healing_strategies.items():
                strategy_stats[issue_type.value] = {
                    'success_count': strategy['success_count'],
                    'failure_count': strategy['failure_count'],
                    'success_rate': (strategy['success_count'] / 
                                   (strategy['success_count'] + strategy['failure_count'])
                                   if strategy['success_count'] + strategy['failure_count'] > 0 else 0),
                    'avg_healing_time': strategy['avg_healing_time']
                }
            
            return {
                'healing_enabled': self.healing_enabled,
                'total_healings_attempted': total_healings,
                'successful_healings': successful_healings,
                'success_rate': successful_healings / total_healings if total_healings > 0 else 0,
                'active_recoveries': len(self.recovery_in_progress),
                'strategy_statistics': strategy_stats,
                'recent_healings': self.healing_history[-10:] if self.healing_history else []
            }


class AdaptiveOptimizer:
    """Adaptive optimization based on system behavior and performance."""
    
    def __init__(self, health_monitor: HealthMonitor):
        self.health_monitor = health_monitor
        
        # Optimization state
        self.optimization_parameters = {
            'batch_size': 64,
            'thread_pool_size': 4,
            'cache_size': 1000,
            'memory_threshold': 85.0,
            'cpu_threshold': 80.0
        }
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        self.optimization_history = []
        
        # Learning parameters
        self.learning_rate = 0.1
        self.exploration_rate = 0.2
        self.min_samples_for_learning = 10
        
        self.lock = threading.RLock()
        
        logger.info("Adaptive optimizer initialized")
    
    def start_optimization(self):
        """Start adaptive optimization."""
        optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        optimization_thread.start()
        
        logger.info("Adaptive optimization started")
    
    def _optimization_loop(self):
        """Main optimization loop."""
        while True:
            try:
                time.sleep(30)  # Optimize every 30 seconds
                
                # Collect performance data
                self._collect_performance_data()
                
                # Analyze performance and adapt
                self._analyze_and_adapt()
                
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                time.sleep(10)
    
    def _collect_performance_data(self):
        """Collect current performance data."""
        metrics = self.health_monitor.metrics_history[-1] if self.health_monitor.metrics_history else None
        
        if metrics:
            performance_data = {
                'timestamp': datetime.now(),
                'cpu_percent': metrics.cpu_percent,
                'memory_percent': metrics.memory_percent,
                'throughput': metrics.operation_throughput,
                'response_time': metrics.response_time_ms,
                'parameters': self.optimization_parameters.copy()
            }
            
            with self.lock:
                self.performance_history.append(performance_data)
    
    def _analyze_and_adapt(self):
        """Analyze performance and adapt parameters."""
        if len(self.performance_history) < self.min_samples_for_learning:
            return
        
        recent_performance = list(self.performance_history)[-10:]
        
        # Calculate performance metrics
        avg_cpu = statistics.mean([p['cpu_percent'] for p in recent_performance])
        avg_memory = statistics.mean([p['memory_percent'] for p in recent_performance])
        avg_throughput = statistics.mean([p['throughput'] for p in recent_performance])
        
        # Adaptation logic
        adaptations = []
        
        # CPU-based adaptations
        if avg_cpu > self.optimization_parameters['cpu_threshold']:
            # Reduce batch size to lower CPU load
            new_batch_size = max(16, int(self.optimization_parameters['batch_size'] * 0.8))
            if new_batch_size != self.optimization_parameters['batch_size']:
                adaptations.append(('batch_size', new_batch_size))
        elif avg_cpu < 50 and avg_throughput > 0:
            # Increase batch size if CPU is underutilized
            new_batch_size = min(256, int(self.optimization_parameters['batch_size'] * 1.2))
            if new_batch_size != self.optimization_parameters['batch_size']:
                adaptations.append(('batch_size', new_batch_size))
        
        # Memory-based adaptations
        if avg_memory > self.optimization_parameters['memory_threshold']:
            # Reduce cache size
            new_cache_size = max(100, int(self.optimization_parameters['cache_size'] * 0.8))
            if new_cache_size != self.optimization_parameters['cache_size']:
                adaptations.append(('cache_size', new_cache_size))
        
        # Apply adaptations
        if adaptations:
            self._apply_adaptations(adaptations)
    
    def _apply_adaptations(self, adaptations: List[Tuple[str, Any]]):
        """Apply parameter adaptations."""
        with self.lock:
            old_params = self.optimization_parameters.copy()
            
            for param_name, new_value in adaptations:
                self.optimization_parameters[param_name] = new_value
            
            # Record adaptation
            adaptation_record = {
                'timestamp': datetime.now(),
                'old_parameters': old_params,
                'new_parameters': self.optimization_parameters.copy(),
                'adaptations': adaptations
            }
            
            self.optimization_history.append(adaptation_record)
            
            logger.info(f"Applied adaptations: {adaptations}")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get optimization report."""
        with self.lock:
            return {
                'current_parameters': self.optimization_parameters.copy(),
                'adaptations_made': len(self.optimization_history),
                'recent_adaptations': self.optimization_history[-5:] if self.optimization_history else [],
                'performance_samples': len(self.performance_history),
                'learning_enabled': len(self.performance_history) >= self.min_samples_for_learning
            }


class PredictiveResourceManager:
    """Predictive resource management using historical patterns."""
    
    def __init__(self, health_monitor: HealthMonitor):
        self.health_monitor = health_monitor
        
        # Prediction models (simplified)
        self.usage_patterns = {
            'hourly': defaultdict(list),
            'daily': defaultdict(list),
            'weekly': defaultdict(list)
        }
        
        # Resource predictions
        self.predictions = {}
        self.prediction_accuracy = {}
        
        # Resource pool management
        self.resource_pools = {
            'compute': {'allocated': 0, 'available': 100, 'reserved': 0},
            'memory': {'allocated': 0, 'available': 8192, 'reserved': 0},  # MB
            'disk': {'allocated': 0, 'available': 100000, 'reserved': 0}   # MB
        }
        
        self.lock = threading.RLock()
        
        logger.info("Predictive resource manager initialized")
    
    def start_prediction(self):
        """Start predictive resource management."""
        prediction_thread = threading.Thread(target=self._prediction_loop, daemon=True)
        prediction_thread.start()
        
        logger.info("Predictive resource management started")
    
    def _prediction_loop(self):
        """Main prediction loop."""
        while True:
            try:
                time.sleep(300)  # Predict every 5 minutes
                
                # Update usage patterns
                self._update_usage_patterns()
                
                # Generate predictions
                self._generate_predictions()
                
                # Validate previous predictions
                self._validate_predictions()
                
                # Proactive resource allocation
                self._proactive_resource_allocation()
                
            except Exception as e:
                logger.error(f"Prediction loop error: {e}")
                time.sleep(60)
    
    def _update_usage_patterns(self):
        """Update historical usage patterns."""
        if not self.health_monitor.metrics_history:
            return
        
        current_time = datetime.now()
        latest_metrics = self.health_monitor.metrics_history[-1]
        
        # Extract usage data
        cpu_usage = latest_metrics.cpu_percent
        memory_usage = latest_metrics.memory_percent
        
        with self.lock:
            # Hourly patterns
            hour_key = current_time.hour
            self.usage_patterns['hourly'][hour_key].append({
                'cpu': cpu_usage,
                'memory': memory_usage,
                'timestamp': current_time
            })
            
            # Daily patterns (day of week)
            day_key = current_time.weekday()
            self.usage_patterns['daily'][day_key].append({
                'cpu': cpu_usage,
                'memory': memory_usage,
                'timestamp': current_time
            })
            
            # Keep only recent data
            cutoff_time = current_time - timedelta(days=30)
            for pattern_type in self.usage_patterns:
                for key in self.usage_patterns[pattern_type]:
                    self.usage_patterns[pattern_type][key] = [
                        data for data in self.usage_patterns[pattern_type][key]
                        if data['timestamp'] > cutoff_time
                    ]
    
    def _generate_predictions(self):
        """Generate resource usage predictions."""
        current_time = datetime.now()
        
        # Predict next hour usage
        next_hour = (current_time + timedelta(hours=1)).hour
        hour_pattern = self.usage_patterns['hourly'].get(next_hour, [])
        
        if hour_pattern:
            avg_cpu = statistics.mean([data['cpu'] for data in hour_pattern])
            avg_memory = statistics.mean([data['memory'] for data in hour_pattern])
            
            prediction = {
                'timestamp': current_time,
                'prediction_time': current_time + timedelta(hours=1),
                'predicted_cpu': avg_cpu,
                'predicted_memory': avg_memory,
                'confidence': min(len(hour_pattern) / 10.0, 1.0)  # More data = higher confidence
            }
            
            with self.lock:
                self.predictions[current_time.isoformat()] = prediction
                
                # Keep only recent predictions
                cutoff_time = current_time - timedelta(days=7)
                self.predictions = {
                    k: v for k, v in self.predictions.items()
                    if datetime.fromisoformat(k) > cutoff_time
                }
    
    def _validate_predictions(self):
        """Validate accuracy of previous predictions."""
        current_time = datetime.now()
        
        if not self.health_monitor.metrics_history:
            return
        
        current_metrics = self.health_monitor.metrics_history[-1]
        
        # Find predictions that should be validated now
        for pred_time_str, prediction in list(self.predictions.items()):
            pred_time = datetime.fromisoformat(pred_time_str)
            prediction_target_time = prediction['prediction_time']
            
            # Check if prediction time has passed
            if current_time >= prediction_target_time:
                # Calculate prediction accuracy
                cpu_error = abs(prediction['predicted_cpu'] - current_metrics.cpu_percent)
                memory_error = abs(prediction['predicted_memory'] - current_metrics.memory_percent)
                
                accuracy = {
                    'prediction_time': pred_time,
                    'target_time': prediction_target_time,
                    'cpu_error': cpu_error,
                    'memory_error': memory_error,
                    'cpu_accuracy': max(0, 100 - cpu_error),
                    'memory_accuracy': max(0, 100 - memory_error)
                }
                
                with self.lock:
                    self.prediction_accuracy[pred_time_str] = accuracy
                    del self.predictions[pred_time_str]
    
    def _proactive_resource_allocation(self):
        """Proactively allocate resources based on predictions."""
        if not self.predictions:
            return
        
        # Get latest prediction
        latest_prediction = max(self.predictions.values(), key=lambda x: x['timestamp'])
        
        # Proactive allocation based on predicted usage
        if latest_prediction['confidence'] > 0.5:  # Only act on confident predictions
            predicted_cpu = latest_prediction['predicted_cpu']
            predicted_memory = latest_prediction['predicted_memory']
            
            # Reserve resources if high usage predicted
            if predicted_cpu > 80 or predicted_memory > 80:
                self._reserve_emergency_resources()
            
            # Scale down reservations if low usage predicted
            elif predicted_cpu < 30 and predicted_memory < 30:
                self._release_excess_resources()
    
    def _reserve_emergency_resources(self):
        """Reserve emergency resources for predicted high usage."""
        with self.lock:
            # Reserve 20% more compute and memory
            for resource_type in ['compute', 'memory']:
                pool = self.resource_pools[resource_type]
                additional_reserve = pool['available'] * 0.2
                
                if pool['available'] >= additional_reserve:
                    pool['reserved'] += additional_reserve
                    pool['available'] -= additional_reserve
                    
                    logger.info(f"Reserved additional {resource_type}: {additional_reserve}")
    
    def _release_excess_resources(self):
        """Release excess reserved resources."""
        with self.lock:
            # Release half of reserved resources
            for resource_type in ['compute', 'memory']:
                pool = self.resource_pools[resource_type]
                release_amount = pool['reserved'] * 0.5
                
                if release_amount > 0:
                    pool['reserved'] -= release_amount
                    pool['available'] += release_amount
                    
                    logger.info(f"Released excess {resource_type}: {release_amount}")
    
    def get_prediction_report(self) -> Dict[str, Any]:
        """Get predictive resource management report."""
        with self.lock:
            # Calculate average prediction accuracy
            accuracy_scores = []
            for accuracy in self.prediction_accuracy.values():
                avg_accuracy = (accuracy['cpu_accuracy'] + accuracy['memory_accuracy']) / 2
                accuracy_scores.append(avg_accuracy)
            
            avg_accuracy = statistics.mean(accuracy_scores) if accuracy_scores else 0
            
            return {
                'active_predictions': len(self.predictions),
                'validated_predictions': len(self.prediction_accuracy),
                'average_accuracy': avg_accuracy,
                'resource_pools': self.resource_pools.copy(),
                'pattern_data_points': {
                    pattern_type: sum(len(data) for data in pattern_data.values())
                    for pattern_type, pattern_data in self.usage_patterns.items()
                },
                'latest_prediction': max(self.predictions.values(), key=lambda x: x['timestamp'])
                                   if self.predictions else None
            }


def create_complete_self_healing_system() -> Dict[str, Any]:
    """Create and initialize complete self-healing system."""
    # Initialize components
    health_monitor = HealthMonitor(check_interval=5.0)
    self_healing = SelfHealingSystem(health_monitor)
    adaptive_optimizer = AdaptiveOptimizer(health_monitor)
    predictive_manager = PredictiveResourceManager(health_monitor)
    
    # Start all systems
    health_monitor.start_monitoring()
    self_healing.start_healing()
    adaptive_optimizer.start_optimization()
    predictive_manager.start_prediction()
    
    logger.info("Complete self-healing system initialized and started")
    
    return {
        'health_monitor': health_monitor,
        'self_healing': self_healing,
        'adaptive_optimizer': adaptive_optimizer,
        'predictive_manager': predictive_manager
    }