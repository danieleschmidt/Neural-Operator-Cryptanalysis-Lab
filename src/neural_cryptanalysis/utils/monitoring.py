"""Monitoring and metrics collection for neural cryptanalysis experiments."""

import time
import json
import threading
import functools
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class MetricValue:
    """Container for a single metric value."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = None
    unit: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


class MetricsCollector:
    """Centralized metrics collection system."""
    
    def __init__(self, buffer_size: int = 10000, flush_interval: int = 60):
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        self.metrics_buffer = deque(maxlen=buffer_size)
        self.metrics_aggregates = defaultdict(list)
        self.custom_metrics = {}
        
        self.lock = threading.RLock()
        self.last_flush = time.time()
        
        # Start background flush thread
        self.flush_thread = threading.Thread(target=self._background_flush, daemon=True)
        self.flush_thread.start()
        
        logger.info("Metrics collector initialized")
    
    def record(self, name: str, value: float, tags: Dict[str, str] = None, 
              unit: str = ""):
        """Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for metric
            unit: Unit of measurement
        """
        metric = MetricValue(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            unit=unit
        )
        
        with self.lock:
            self.metrics_buffer.append(metric)
            self.metrics_aggregates[name].append(value)
            
            # Limit aggregate buffer size
            if len(self.metrics_aggregates[name]) > 1000:
                self.metrics_aggregates[name] = self.metrics_aggregates[name][-500:]
    
    def increment(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Increment a counter metric.
        
        Args:
            name: Counter name
            value: Increment value
            tags: Optional tags
        """
        counter_name = f"{name}_total"
        
        if counter_name not in self.custom_metrics:
            self.custom_metrics[counter_name] = 0
        
        self.custom_metrics[counter_name] += value
        self.record(counter_name, self.custom_metrics[counter_name], tags, "count")
    
    def gauge(self, name: str, value: float, tags: Dict[str, str] = None, 
             unit: str = ""):
        """Record a gauge metric (instantaneous value).
        
        Args:
            name: Gauge name
            value: Current value
            tags: Optional tags
            unit: Unit of measurement
        """
        gauge_name = f"{name}_gauge"
        self.custom_metrics[gauge_name] = value
        self.record(gauge_name, value, tags, unit)
    
    def histogram(self, name: str, value: float, tags: Dict[str, str] = None,
                 unit: str = ""):
        """Record a histogram metric.
        
        Args:
            name: Histogram name
            value: Value to record
            tags: Optional tags
            unit: Unit of measurement
        """
        hist_name = f"{name}_histogram"
        self.record(hist_name, value, tags, unit)
    
    def timing(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Record a timing metric.
        
        Args:
            name: Operation name
            duration: Duration in seconds
            tags: Optional tags
        """
        timing_name = f"{name}_duration"
        self.record(timing_name, duration, tags, "seconds")
    
    def get_metric_summary(self, name: str, window_minutes: int = 10) -> Dict[str, float]:
        """Get summary statistics for a metric.
        
        Args:
            name: Metric name
            window_minutes: Time window in minutes
            
        Returns:
            Summary statistics
        """
        with self.lock:
            if name not in self.metrics_aggregates:
                return {}
            
            # Filter to time window
            cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
            recent_metrics = [
                metric for metric in self.metrics_buffer
                if metric.name == name and metric.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                return {}
            
            values = [m.value for m in recent_metrics]
            
            return {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                'latest': values[-1],
                'unit': recent_metrics[-1].unit
            }
    
    def get_all_metrics(self, window_minutes: int = 10) -> Dict[str, Dict[str, float]]:
        """Get summary for all metrics.
        
        Args:
            window_minutes: Time window in minutes
            
        Returns:
            Dictionary of metric summaries
        """
        metric_names = set(m.name for m in self.metrics_buffer)
        return {
            name: self.get_metric_summary(name, window_minutes)
            for name in metric_names
        }
    
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format.
        
        Returns:
            Prometheus-formatted metrics string
        """
        lines = []
        
        with self.lock:
            # Group metrics by name
            metric_groups = defaultdict(list)
            for metric in self.metrics_buffer:
                metric_groups[metric.name].append(metric)
            
            for name, metrics in metric_groups.items():
                if not metrics:
                    continue
                
                latest_metric = metrics[-1]
                
                # Add help text
                lines.append(f"# HELP {name} {name} metric")
                lines.append(f"# TYPE {name} gauge")
                
                # Add metric value with tags
                tag_str = ""
                if latest_metric.tags:
                    tag_pairs = [f'{k}="{v}"' for k, v in latest_metric.tags.items()]
                    tag_str = "{" + ",".join(tag_pairs) + "}"
                
                lines.append(f"{name}{tag_str} {latest_metric.value}")
        
        return "\n".join(lines)
    
    def flush_to_file(self, file_path: Path):
        """Flush metrics to file.
        
        Args:
            file_path: Path to write metrics
        """
        with self.lock:
            metrics_data = []
            
            for metric in self.metrics_buffer:
                metrics_data.append({
                    'name': metric.name,
                    'value': metric.value,
                    'timestamp': metric.timestamp.isoformat(),
                    'tags': metric.tags,
                    'unit': metric.unit
                })
            
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            logger.debug(f"Flushed {len(metrics_data)} metrics to {file_path}")
    
    def _background_flush(self):
        """Background thread for periodic metric flushing."""
        while True:
            time.sleep(self.flush_interval)
            
            current_time = time.time()
            if current_time - self.last_flush >= self.flush_interval:
                self._periodic_cleanup()
                self.last_flush = current_time
    
    def _periodic_cleanup(self):
        """Periodic cleanup of old metrics."""
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        with self.lock:
            # Remove old metrics from buffer
            recent_metrics = deque()
            for metric in self.metrics_buffer:
                if metric.timestamp >= cutoff_time:
                    recent_metrics.append(metric)
            
            self.metrics_buffer = recent_metrics
            
            # Clean up aggregates
            for name in list(self.metrics_aggregates.keys()):
                if len(self.metrics_aggregates[name]) > 1000:
                    self.metrics_aggregates[name] = self.metrics_aggregates[name][-500:]


def timed_metric(metric_name: str, collector: MetricsCollector, 
                tags: Dict[str, str] = None):
    """Decorator to time function execution and record as metric.
    
    Args:
        metric_name: Name of the timing metric
        collector: Metrics collector instance
        tags: Optional tags for the metric
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                metric_tags = (tags or {}).copy()
                metric_tags['function'] = func.__name__
                metric_tags['success'] = str(success)
                
                collector.timing(metric_name, duration, metric_tags)
            
            return result
        return wrapper
    return decorator


class ExperimentMonitor:
    """Monitor for tracking experiment progress and performance."""
    
    def __init__(self, experiment_id: str, metrics_collector: MetricsCollector):
        self.experiment_id = experiment_id
        self.metrics = metrics_collector
        
        self.start_time = datetime.now()
        self.phase_start_times = {}
        self.current_phase = None
        
        # Experiment-specific metrics
        self.experiment_tags = {'experiment_id': experiment_id}
        
        logger.info(f"Experiment monitor started: {experiment_id}")
        self.metrics.increment('experiments_started', tags=self.experiment_tags)
    
    def start_phase(self, phase_name: str):
        """Start a new experiment phase.
        
        Args:
            phase_name: Name of the phase (e.g., 'training', 'attack')
        """
        if self.current_phase:
            self.end_phase()
        
        self.current_phase = phase_name
        self.phase_start_times[phase_name] = time.time()
        
        phase_tags = self.experiment_tags.copy()
        phase_tags['phase'] = phase_name
        
        logger.info(f"Started phase: {phase_name}")
        self.metrics.increment('phase_starts', tags=phase_tags)
    
    def end_phase(self, results: Dict[str, Any] = None):
        """End current experiment phase.
        
        Args:
            results: Optional phase results to record
        """
        if not self.current_phase:
            return
        
        phase_name = self.current_phase
        start_time = self.phase_start_times.get(phase_name, time.time())
        duration = time.time() - start_time
        
        phase_tags = self.experiment_tags.copy()
        phase_tags['phase'] = phase_name
        
        # Record phase duration
        self.metrics.timing(f'phase_duration', duration, phase_tags)
        
        # Record phase results
        if results:
            for metric_name, value in results.items():
                if isinstance(value, (int, float)):
                    self.metrics.record(f'phase_{metric_name}', value, phase_tags)
        
        logger.info(f"Completed phase: {phase_name} ({duration:.2f}s)")
        self.metrics.increment('phase_completions', tags=phase_tags)
        
        self.current_phase = None
    
    def record_training_metrics(self, epoch: int, loss: float, accuracy: float,
                              learning_rate: float = None):
        """Record training metrics.
        
        Args:
            epoch: Current epoch
            loss: Training loss
            accuracy: Training accuracy
            learning_rate: Current learning rate
        """
        training_tags = self.experiment_tags.copy()
        training_tags.update({'phase': 'training', 'epoch': str(epoch)})
        
        self.metrics.record('training_loss', loss, training_tags)
        self.metrics.record('training_accuracy', accuracy, training_tags, '%')
        
        if learning_rate is not None:
            self.metrics.record('learning_rate', learning_rate, training_tags)
    
    def record_attack_metrics(self, traces_used: int, success_rate: float,
                            confidence: float, attack_type: str = None):
        """Record attack metrics.
        
        Args:
            traces_used: Number of traces used
            success_rate: Attack success rate
            confidence: Average confidence
            attack_type: Type of attack
        """
        attack_tags = self.experiment_tags.copy()
        attack_tags['phase'] = 'attack'
        
        if attack_type:
            attack_tags['attack_type'] = attack_type
        
        self.metrics.record('traces_used', traces_used, attack_tags, 'count')
        self.metrics.record('success_rate', success_rate, attack_tags, '%')
        self.metrics.record('confidence', confidence, attack_tags, '%')
    
    def record_resource_usage(self, memory_mb: float, cpu_percent: float,
                            gpu_memory_mb: float = None):
        """Record resource usage metrics.
        
        Args:
            memory_mb: Memory usage in MB
            cpu_percent: CPU usage percentage
            gpu_memory_mb: GPU memory usage in MB
        """
        resource_tags = self.experiment_tags.copy()
        
        self.metrics.gauge('memory_usage', memory_mb, resource_tags, 'MB')
        self.metrics.gauge('cpu_usage', cpu_percent, resource_tags, '%')
        
        if gpu_memory_mb is not None:
            self.metrics.gauge('gpu_memory_usage', gpu_memory_mb, resource_tags, 'MB')
    
    def finalize_experiment(self, final_results: Dict[str, Any] = None):
        """Finalize experiment monitoring.
        
        Args:
            final_results: Final experiment results
        """
        if self.current_phase:
            self.end_phase()
        
        total_duration = (datetime.now() - self.start_time).total_seconds()
        
        # Record total duration
        self.metrics.timing('experiment_duration', total_duration, self.experiment_tags)
        
        # Record final results
        if final_results:
            for metric_name, value in final_results.items():
                if isinstance(value, (int, float)):
                    self.metrics.record(f'final_{metric_name}', value, self.experiment_tags)
        
        logger.info(f"Experiment completed: {self.experiment_id} ({total_duration:.2f}s)")
        self.metrics.increment('experiments_completed', tags=self.experiment_tags)
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get experiment summary statistics.
        
        Returns:
            Experiment summary
        """
        current_duration = (datetime.now() - self.start_time).total_seconds()
        
        summary = {
            'experiment_id': self.experiment_id,
            'start_time': self.start_time.isoformat(),
            'current_duration': current_duration,
            'current_phase': self.current_phase,
            'completed_phases': list(self.phase_start_times.keys())
        }
        
        # Add recent metrics
        recent_metrics = {}
        for metric_name in ['training_loss', 'training_accuracy', 'success_rate', 'confidence']:
            metric_summary = self.metrics.get_metric_summary(metric_name, window_minutes=5)
            if metric_summary:
                recent_metrics[metric_name] = metric_summary.get('latest')
        
        summary['recent_metrics'] = recent_metrics
        
        return summary


class HealthMonitor:
    """System health monitoring for neural cryptanalysis operations."""
    
    def __init__(self, metrics_collector: MetricsCollector, 
                 check_interval: int = 30):
        self.metrics = metrics_collector
        self.check_interval = check_interval
        
        self.health_checks = {}
        self.alerts = []
        
        # System thresholds
        self.thresholds = {
            'memory_usage_percent': 80.0,
            'cpu_usage_percent': 90.0,
            'disk_usage_percent': 85.0,
            'gpu_memory_percent': 90.0
        }
        
        # Start health monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Health monitor initialized")
    
    def register_health_check(self, name: str, check_func: Callable[[], bool]):
        """Register a custom health check.
        
        Args:
            name: Health check name
            check_func: Function that returns True if healthy
        """
        self.health_checks[name] = check_func
        logger.debug(f"Registered health check: {name}")
    
    def add_alert(self, severity: str, message: str, component: str = None):
        """Add a health alert.
        
        Args:
            severity: Alert severity (info, warning, error, critical)
            message: Alert message
            component: Optional component name
        """
        alert = {
            'timestamp': datetime.now().isoformat(),
            'severity': severity,
            'message': message,
            'component': component or 'system'
        }
        
        self.alerts.append(alert)
        
        # Keep only recent alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-50:]
        
        logger.warning(f"Health alert [{severity}]: {message}")
        
        # Record as metric
        alert_tags = {'severity': severity, 'component': alert['component']}
        self.metrics.increment('health_alerts', tags=alert_tags)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current system health status.
        
        Returns:
            Health status summary
        """
        status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'system_metrics': self._get_system_metrics(),
            'custom_checks': {},
            'recent_alerts': self.alerts[-10:],  # Last 10 alerts
            'alert_count': len(self.alerts)
        }
        
        # Run custom health checks
        for name, check_func in self.health_checks.items():
            try:
                is_healthy = check_func()
                status['custom_checks'][name] = {
                    'status': 'healthy' if is_healthy else 'unhealthy',
                    'timestamp': datetime.now().isoformat()
                }
                
                if not is_healthy:
                    status['overall_status'] = 'degraded'
                    
            except Exception as e:
                logger.warning(f"Health check failed: {name}: {e}")
                status['custom_checks'][name] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                status['overall_status'] = 'degraded'
        
        # Check system thresholds
        sys_metrics = status['system_metrics']
        
        if sys_metrics.get('memory_percent', 0) > self.thresholds['memory_usage_percent']:
            status['overall_status'] = 'degraded'
        
        if sys_metrics.get('cpu_percent', 0) > self.thresholds['cpu_usage_percent']:
            status['overall_status'] = 'critical'
        
        return status
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while True:
            try:
                self._check_system_health()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(self.check_interval)
    
    def _check_system_health(self):
        """Check system health metrics."""
        import psutil
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        self.metrics.gauge('system_memory_percent', memory_percent, unit='%')
        
        if memory_percent > self.thresholds['memory_usage_percent']:
            self.add_alert('warning', f'High memory usage: {memory_percent:.1f}%')
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics.gauge('system_cpu_percent', cpu_percent, unit='%')
        
        if cpu_percent > self.thresholds['cpu_usage_percent']:
            self.add_alert('critical', f'High CPU usage: {cpu_percent:.1f}%')
        
        # Disk usage
        try:
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.metrics.gauge('system_disk_percent', disk_percent, unit='%')
            
            if disk_percent > self.thresholds['disk_usage_percent']:
                self.add_alert('warning', f'High disk usage: {disk_percent:.1f}%')
        except Exception:
            pass
    
    def _get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics.
        
        Returns:
            System metrics dictionary
        """
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / 1024**3,
                'cpu_percent': psutil.cpu_percent(),
                'disk_percent': (disk.used / disk.total) * 100,
                'disk_free_gb': disk.free / 1024**3
            }
        except ImportError:
            return {}


class StatusEndpoint:
    """HTTP-like status endpoint for monitoring."""
    
    def __init__(self, metrics_collector: MetricsCollector, health_monitor: HealthMonitor):
        self.metrics = metrics_collector
        self.health = health_monitor
        
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        health_status = self.health.get_health_status()
        metrics_summary = self.metrics.get_all_metrics(window_minutes=5)
        
        return {
            'status': health_status['overall_status'],
            'timestamp': datetime.now().isoformat(),
            'version': '2.0.0',  # Neural cryptanalysis framework version
            'uptime_seconds': time.time() - getattr(self, '_start_time', time.time()),
            'health': health_status,
            'metrics': {
                'recent_metrics': metrics_summary,
                'metric_count': len(self.metrics.metrics_buffer),
                'buffer_utilization': len(self.metrics.metrics_buffer) / self.metrics.buffer_size
            },
            'system': self.health._get_system_metrics()
        }
    
    def get_metrics(self, format_type: str = 'json') -> Union[str, Dict[str, Any]]:
        """Get metrics in specified format."""
        if format_type == 'prometheus':
            return self.metrics.export_prometheus_format()
        else:
            return self.metrics.get_all_metrics(window_minutes=60)
    
    def get_health(self) -> Dict[str, Any]:
        """Get health check results."""
        return self.health.get_health_status()


class ResourceTracker:
    """Advanced resource usage tracking."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.baseline_resources = None
        self.peak_resources = {}
        self.resource_history = deque(maxlen=1000)
        
        # Initialize baseline
        self._record_baseline()
        
    def _record_baseline(self):
        """Record baseline resource usage."""
        try:
            import psutil
            
            self.baseline_resources = {
                'memory_mb': psutil.virtual_memory().used / 1024**2,
                'cpu_percent': psutil.cpu_percent(),
                'num_threads': psutil.Process().num_threads(),
                'open_files': len(psutil.Process().open_files())
            }
            
            logger.info(f"Baseline resources recorded: {self.baseline_resources}")
            
        except ImportError:
            self.baseline_resources = {}
    
    def track_operation(self, operation_name: str):
        """Context manager to track resource usage during operation."""
        class OperationTracker:
            def __init__(self, tracker, name):
                self.tracker = tracker
                self.name = name
                self.start_resources = None
                self.start_time = None
                
            def __enter__(self):
                self.start_time = time.time()
                self.start_resources = self.tracker._get_current_resources()
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                end_time = time.time()
                end_resources = self.tracker._get_current_resources()
                duration = end_time - self.start_time
                
                # Calculate resource deltas
                resource_delta = {}
                if self.start_resources and end_resources:
                    for key in self.start_resources:
                        if key in end_resources:
                            resource_delta[key] = end_resources[key] - self.start_resources[key]
                
                # Record metrics
                operation_tags = {'operation': self.name}
                self.tracker.metrics.timing('operation_duration', duration, operation_tags)
                
                for resource, delta in resource_delta.items():
                    self.tracker.metrics.record(f'resource_delta_{resource}', delta, operation_tags)
                
                # Update peaks
                for resource, value in end_resources.items():
                    current_peak = self.tracker.peak_resources.get(resource, 0)
                    if value > current_peak:
                        self.tracker.peak_resources[resource] = value
                        self.tracker.metrics.gauge(f'peak_{resource}', value)
                
                # Store in history
                self.tracker.resource_history.append({
                    'timestamp': end_time,
                    'operation': self.name,
                    'duration': duration,
                    'resources': end_resources,
                    'deltas': resource_delta
                })
                
                logger.debug(f"Operation '{self.name}' completed in {duration:.3f}s, "
                           f"resource deltas: {resource_delta}")
        
        return OperationTracker(self, operation_name)
    
    def _get_current_resources(self) -> Dict[str, float]:
        """Get current resource usage."""
        try:
            import psutil
            process = psutil.Process()
            
            return {
                'memory_mb': process.memory_info().rss / 1024**2,
                'cpu_percent': process.cpu_percent(),
                'num_threads': process.num_threads(),
                'open_files': len(process.open_files()),
                'memory_percent': psutil.virtual_memory().percent
            }
        except ImportError:
            return {}
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource usage summary."""
        current = self._get_current_resources()
        
        summary = {
            'current': current,
            'baseline': self.baseline_resources,
            'peaks': self.peak_resources,
            'deltas_from_baseline': {}
        }
        
        # Calculate deltas from baseline
        if self.baseline_resources and current:
            for key in self.baseline_resources:
                if key in current:
                    summary['deltas_from_baseline'][key] = current[key] - self.baseline_resources[key]
        
        # Recent history statistics
        if self.resource_history:
            recent_operations = list(self.resource_history)[-100:]  # Last 100 operations
            
            durations = [op['duration'] for op in recent_operations]
            memory_usage = [op['resources'].get('memory_mb', 0) for op in recent_operations]
            
            summary['recent_stats'] = {
                'avg_operation_duration': statistics.mean(durations) if durations else 0,
                'avg_memory_usage': statistics.mean(memory_usage) if memory_usage else 0,
                'operation_count': len(recent_operations)
            }
        
        return summary


class PerformanceProfiler:
    """Performance profiling and analysis."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.profiles = {}
        self.active_profiles = {}
        
    def profile_function(self, name: str = None):
        """Decorator to profile function performance."""
        def decorator(func):
            profile_name = name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self.profile_execution(profile_name, func, *args, **kwargs)
            
            return wrapper
        return decorator
    
    def profile_execution(self, name: str, func: Callable, *args, **kwargs) -> Any:
        """Profile function execution."""
        import cProfile
        import pstats
        from io import StringIO
        
        # Start profiling
        profiler = cProfile.Profile()
        profiler.enable()
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            success = False
            raise
        finally:
            # Stop profiling
            profiler.disable()
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory if start_memory and end_memory else 0
            
            # Generate profile stats
            stats_stream = StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream)
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # Top 20 functions
            
            profile_data = {
                'name': name,
                'timestamp': end_time,
                'duration': duration,
                'memory_delta_mb': memory_delta,
                'success': success,
                'stats': stats_stream.getvalue()
            }
            
            # Store profile
            if name not in self.profiles:
                self.profiles[name] = deque(maxlen=100)
            self.profiles[name].append(profile_data)
            
            # Record metrics
            profile_tags = {'function': name, 'success': str(success)}
            self.metrics.timing('function_duration', duration, profile_tags)
            self.metrics.record('function_memory_delta', memory_delta, profile_tags, 'MB')
            
            logger.debug(f"Profiled {name}: {duration:.3f}s, memory delta: {memory_delta:.1f}MB")
        
        return result
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024**2
        except ImportError:
            return None
    
    def get_profile_summary(self, name: str = None) -> Dict[str, Any]:
        """Get profiling summary."""
        if name:
            if name not in self.profiles:
                return {}
            
            profiles = list(self.profiles[name])
            
            durations = [p['duration'] for p in profiles]
            memory_deltas = [p['memory_delta_mb'] for p in profiles]
            success_rate = sum(1 for p in profiles if p['success']) / len(profiles)
            
            return {
                'function': name,
                'execution_count': len(profiles),
                'avg_duration': statistics.mean(durations) if durations else 0,
                'min_duration': min(durations) if durations else 0,
                'max_duration': max(durations) if durations else 0,
                'avg_memory_delta': statistics.mean(memory_deltas) if memory_deltas else 0,
                'success_rate': success_rate,
                'recent_executions': profiles[-10:] if profiles else []
            }
        else:
            # Summary for all profiles
            summary = {}
            for profile_name in self.profiles:
                summary[profile_name] = self.get_profile_summary(profile_name)
            return summary


# Enhanced health checks
def register_neural_operator_health_checks(health_monitor: HealthMonitor,
                                         neural_operator_instance=None):
    """Register health checks specific to neural operator components."""
    
    def check_model_loaded():
        """Check if neural operator model is loaded."""
        if neural_operator_instance is None:
            return True  # No model to check
        
        try:
            # Check if model has parameters
            if hasattr(neural_operator_instance, 'parameters'):
                param_count = sum(p.numel() for p in neural_operator_instance.parameters())
                return param_count > 0
            return True
        except Exception:
            return False
    
    def check_gpu_availability():
        """Check GPU availability for PyTorch."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return True  # Not using PyTorch
    
    def check_data_integrity():
        """Check data directory integrity."""
        # This could check for corrupted data files, missing datasets, etc.
        return True  # Simplified for now
    
    def check_dependencies():
        """Check critical dependencies."""
        try:
            import numpy
            import torch
            return True
        except ImportError:
            return False
    
    # Register health checks
    health_monitor.register_health_check('model_loaded', check_model_loaded)
    health_monitor.register_health_check('gpu_available', check_gpu_availability)
    health_monitor.register_health_check('data_integrity', check_data_integrity)
    health_monitor.register_health_check('dependencies', check_dependencies)


# Global instances for enhanced monitoring
enhanced_metrics = MetricsCollector(buffer_size=50000, flush_interval=30)
enhanced_health = HealthMonitor(enhanced_metrics, check_interval=15)
resource_tracker = ResourceTracker(enhanced_metrics)
performance_profiler = PerformanceProfiler(enhanced_metrics)
status_endpoint = StatusEndpoint(enhanced_metrics, enhanced_health)