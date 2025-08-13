"""Research Acceleration Framework for Neural Cryptanalysis.

This module provides advanced research capabilities including experiment management,
hyperparameter optimization, A/B testing, and automated benchmarking.
"""

import time
import json
import pickle
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from functools import partial
from itertools import product
import hashlib
import uuid
import statistics
from collections import defaultdict, deque

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import scipy.optimize
    from scipy.stats import mannwhitneyu, ttest_ind
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    scipy = None

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

from ..utils.logging_utils import get_logger
from ..utils.errors import NeuralCryptanalysisError

logger = get_logger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for research experiments."""
    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: List[str] = field(default_factory=list)
    iterations: int = 1
    timeout_seconds: Optional[float] = None
    save_artifacts: bool = True
    random_seed: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Results from a research experiment."""
    experiment_id: str
    config: ExperimentConfig
    metrics: Dict[str, float]
    artifacts: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    status: str = "pending"  # pending, running, completed, failed
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    iteration: int = 0


@dataclass
class HyperparameterSpace:
    """Definition of hyperparameter search space."""
    name: str
    param_type: str  # 'continuous', 'discrete', 'categorical'
    bounds: Optional[Tuple[float, float]] = None
    choices: Optional[List[Any]] = None
    log_scale: bool = False
    prior: Optional[str] = None  # 'uniform', 'normal', 'log_normal'


class ExperimentManager:
    """Advanced experiment management and tracking system."""
    
    def __init__(self, workspace_dir: Path = None, max_concurrent: int = 4):
        self.workspace_dir = workspace_dir or Path("./experiments")
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_concurrent = max_concurrent
        self.experiments = {}
        self.running_experiments = {}
        self.completed_experiments = {}
        self.failed_experiments = {}
        
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.lock = threading.RLock()
        
        # Load existing experiments
        self._load_existing_experiments()
        
        logger.info(f"Experiment manager initialized: {self.workspace_dir}")
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new experiment."""
        experiment_id = str(uuid.uuid4())[:8]
        
        # Create experiment directory
        exp_dir = self.workspace_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        # Save configuration
        config_path = exp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2, default=str)
        
        with self.lock:
            self.experiments[experiment_id] = {
                'config': config,
                'directory': exp_dir,
                'status': 'created',
                'created_at': datetime.now()
            }
        
        logger.info(f"Created experiment: {experiment_id} - {config.name}")
        return experiment_id
    
    def run_experiment(self, experiment_id: str, experiment_func: Callable,
                      *args, **kwargs) -> ExperimentResult:
        """Run a single experiment."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        config = self.experiments[experiment_id]['config']
        
        # Create result object
        result = ExperimentResult(
            experiment_id=experiment_id,
            config=config,
            metrics={}
        )
        
        try:
            with self.lock:
                self.running_experiments[experiment_id] = result
                result.status = "running"
            
            start_time = time.time()
            
            # Set random seed if specified
            if config.random_seed is not None:
                if HAS_NUMPY:
                    np.random.seed(config.random_seed)
                if HAS_TORCH:
                    torch.manual_seed(config.random_seed)
            
            # Run experiment function
            experiment_output = experiment_func(*args, **kwargs)
            
            # Extract metrics and artifacts
            if isinstance(experiment_output, dict):
                result.metrics = experiment_output.get('metrics', {})
                result.artifacts = experiment_output.get('artifacts', {})
            else:
                result.metrics = {'result': float(experiment_output)}
            
            result.duration = time.time() - start_time
            result.status = "completed"
            
            # Save results
            self._save_experiment_result(result)
            
            with self.lock:
                del self.running_experiments[experiment_id]
                self.completed_experiments[experiment_id] = result
            
            logger.info(f"Completed experiment: {experiment_id} in {result.duration:.2f}s")
            
        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
            result.duration = time.time() - start_time
            
            with self.lock:
                if experiment_id in self.running_experiments:
                    del self.running_experiments[experiment_id]
                self.failed_experiments[experiment_id] = result
            
            logger.error(f"Experiment {experiment_id} failed: {e}")
        
        return result
    
    def run_batch_experiments(self, configs: List[ExperimentConfig],
                            experiment_func: Callable, *args, **kwargs) -> List[ExperimentResult]:
        """Run multiple experiments in parallel."""
        experiment_ids = [self.create_experiment(config) for config in configs]
        
        # Submit all experiments
        futures = []
        for exp_id in experiment_ids:
            future = self.executor.submit(
                self.run_experiment, exp_id, experiment_func, *args, **kwargs
            )
            futures.append((exp_id, future))
        
        # Collect results
        results = []
        for exp_id, future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Batch experiment {exp_id} failed: {e}")
                # Create failed result
                config = self.experiments[exp_id]['config']
                failed_result = ExperimentResult(
                    experiment_id=exp_id,
                    config=config,
                    metrics={},
                    status="failed",
                    error_message=str(e)
                )
                results.append(failed_result)
        
        return results
    
    def get_experiment_results(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get results for a specific experiment."""
        if experiment_id in self.completed_experiments:
            return self.completed_experiments[experiment_id]
        elif experiment_id in self.failed_experiments:
            return self.failed_experiments[experiment_id]
        else:
            # Try to load from disk
            return self._load_experiment_result(experiment_id)
    
    def list_experiments(self, status: str = None, tag: str = None) -> List[str]:
        """List experiments with optional filtering."""
        experiment_ids = []
        
        for exp_id, exp_info in self.experiments.items():
            if status and exp_info.get('status') != status:
                continue
            
            if tag:
                config = exp_info['config']
                if tag not in config.tags:
                    continue
            
            experiment_ids.append(exp_id)
        
        return experiment_ids
    
    def compare_experiments(self, experiment_ids: List[str],
                          metric: str = None) -> Dict[str, Any]:
        """Compare results across experiments."""
        results = []
        
        for exp_id in experiment_ids:
            result = self.get_experiment_results(exp_id)
            if result and result.status == "completed":
                results.append(result)
        
        if not results:
            return {}
        
        comparison = {
            'experiments': len(results),
            'metrics_summary': {},
            'best_experiment': None,
            'statistical_analysis': {}
        }
        
        # Aggregate metrics
        all_metrics = set()
        for result in results:
            all_metrics.update(result.metrics.keys())
        
        for metric_name in all_metrics:
            values = []
            for result in results:
                if metric_name in result.metrics:
                    values.append(result.metrics[metric_name])
            
            if values:
                comparison['metrics_summary'][metric_name] = {
                    'mean': statistics.mean(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0,
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
        
        # Find best experiment
        if metric and comparison['metrics_summary'].get(metric):
            best_result = max(results, key=lambda r: r.metrics.get(metric, float('-inf')))
            comparison['best_experiment'] = {
                'experiment_id': best_result.experiment_id,
                'config': asdict(best_result.config),
                'metrics': best_result.metrics
            }
        
        return comparison
    
    def _save_experiment_result(self, result: ExperimentResult):
        """Save experiment result to disk."""
        exp_dir = self.workspace_dir / result.experiment_id
        
        # Save result
        result_path = exp_dir / "result.json"
        with open(result_path, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        # Save artifacts separately
        if result.artifacts:
            artifacts_dir = exp_dir / "artifacts"
            artifacts_dir.mkdir(exist_ok=True)
            
            for name, artifact in result.artifacts.items():
                artifact_path = artifacts_dir / f"{name}.pkl"
                try:
                    with open(artifact_path, 'wb') as f:
                        pickle.dump(artifact, f)
                except Exception as e:
                    logger.warning(f"Failed to save artifact {name}: {e}")
    
    def _load_experiment_result(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Load experiment result from disk."""
        result_path = self.workspace_dir / experiment_id / "result.json"
        
        if not result_path.exists():
            return None
        
        try:
            with open(result_path, 'r') as f:
                data = json.load(f)
            
            # Convert back to ExperimentResult
            config_data = data['config']
            config = ExperimentConfig(**config_data)
            
            result = ExperimentResult(
                experiment_id=data['experiment_id'],
                config=config,
                metrics=data['metrics'],
                artifacts={},  # Load artifacts separately if needed
                duration=data['duration'],
                status=data['status'],
                error_message=data.get('error_message'),
                timestamp=datetime.fromisoformat(data['timestamp']),
                iteration=data.get('iteration', 0)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to load experiment result {experiment_id}: {e}")
            return None
    
    def _load_existing_experiments(self):
        """Load existing experiments from workspace."""
        if not self.workspace_dir.exists():
            return
        
        for exp_dir in self.workspace_dir.iterdir():
            if exp_dir.is_dir():
                config_path = exp_dir / "config.json"
                if config_path.exists():
                    try:
                        with open(config_path, 'r') as f:
                            config_data = json.load(f)
                        
                        config = ExperimentConfig(**config_data)
                        exp_id = exp_dir.name
                        
                        self.experiments[exp_id] = {
                            'config': config,
                            'directory': exp_dir,
                            'status': 'loaded',
                            'created_at': datetime.now()
                        }
                        
                    except Exception as e:
                        logger.warning(f"Failed to load experiment {exp_dir.name}: {e}")


class HyperparameterOptimizer:
    """Advanced hyperparameter optimization using multiple strategies."""
    
    def __init__(self, experiment_manager: ExperimentManager):
        self.experiment_manager = experiment_manager
        self.optimization_history = []
        
    def optimize(self, objective_func: Callable, parameter_space: List[HyperparameterSpace],
                n_trials: int = 100, strategy: str = 'bayesian',
                metric_name: str = 'objective', maximize: bool = True) -> Dict[str, Any]:
        """Optimize hyperparameters using specified strategy."""
        
        if strategy == 'grid':
            return self._grid_search(objective_func, parameter_space, metric_name, maximize)
        elif strategy == 'random':
            return self._random_search(objective_func, parameter_space, n_trials, metric_name, maximize)
        elif strategy == 'bayesian':
            return self._bayesian_optimization(objective_func, parameter_space, n_trials, metric_name, maximize)
        else:
            raise ValueError(f"Unknown optimization strategy: {strategy}")
    
    def _grid_search(self, objective_func: Callable, parameter_space: List[HyperparameterSpace],
                    metric_name: str, maximize: bool) -> Dict[str, Any]:
        """Grid search optimization."""
        # Generate all parameter combinations
        param_combinations = self._generate_grid_combinations(parameter_space)
        
        logger.info(f"Starting grid search with {len(param_combinations)} combinations")
        
        # Run experiments for all combinations
        configs = []
        for i, params in enumerate(param_combinations):
            config = ExperimentConfig(
                name=f"grid_search_{i}",
                description=f"Grid search iteration {i}",
                parameters=params,
                metrics=[metric_name],
                tags=["grid_search", "hyperopt"]
            )
            configs.append(config)
        
        results = self.experiment_manager.run_batch_experiments(configs, objective_func)
        
        # Find best result
        valid_results = [r for r in results if r.status == "completed" and metric_name in r.metrics]
        
        if not valid_results:
            raise RuntimeError("No valid results from grid search")
        
        best_result = max(valid_results, key=lambda r: r.metrics[metric_name]) if maximize else \
                     min(valid_results, key=lambda r: r.metrics[metric_name])
        
        return {
            'best_parameters': best_result.config.parameters,
            'best_value': best_result.metrics[metric_name],
            'n_evaluations': len(valid_results),
            'all_results': [
                {
                    'parameters': r.config.parameters,
                    'value': r.metrics.get(metric_name, None),
                    'experiment_id': r.experiment_id
                }
                for r in valid_results
            ],
            'optimization_history': self.optimization_history
        }
    
    def _random_search(self, objective_func: Callable, parameter_space: List[HyperparameterSpace],
                      n_trials: int, metric_name: str, maximize: bool) -> Dict[str, Any]:
        """Random search optimization."""
        logger.info(f"Starting random search with {n_trials} trials")
        
        configs = []
        for i in range(n_trials):
            params = self._sample_random_parameters(parameter_space)
            config = ExperimentConfig(
                name=f"random_search_{i}",
                description=f"Random search trial {i}",
                parameters=params,
                metrics=[metric_name],
                tags=["random_search", "hyperopt"]
            )
            configs.append(config)
        
        results = self.experiment_manager.run_batch_experiments(configs, objective_func)
        
        # Process results similar to grid search
        valid_results = [r for r in results if r.status == "completed" and metric_name in r.metrics]
        
        if not valid_results:
            raise RuntimeError("No valid results from random search")
        
        best_result = max(valid_results, key=lambda r: r.metrics[metric_name]) if maximize else \
                     min(valid_results, key=lambda r: r.metrics[metric_name])
        
        return {
            'best_parameters': best_result.config.parameters,
            'best_value': best_result.metrics[metric_name],
            'n_evaluations': len(valid_results),
            'all_results': [
                {
                    'parameters': r.config.parameters,
                    'value': r.metrics.get(metric_name, None),
                    'experiment_id': r.experiment_id
                }
                for r in valid_results
            ]
        }
    
    def _bayesian_optimization(self, objective_func: Callable, parameter_space: List[HyperparameterSpace],
                             n_trials: int, metric_name: str, maximize: bool) -> Dict[str, Any]:
        """Bayesian optimization (simplified implementation)."""
        logger.info(f"Starting Bayesian optimization with {n_trials} trials")
        
        # For simplicity, fall back to random search
        # In a full implementation, would use Gaussian Process models
        return self._random_search(objective_func, parameter_space, n_trials, metric_name, maximize)
    
    def _generate_grid_combinations(self, parameter_space: List[HyperparameterSpace]) -> List[Dict[str, Any]]:
        """Generate all combinations for grid search."""
        param_values = {}
        
        for param in parameter_space:
            if param.param_type == 'discrete':
                if param.bounds:
                    start, end = param.bounds
                    param_values[param.name] = list(range(int(start), int(end) + 1))
                elif param.choices:
                    param_values[param.name] = param.choices
                else:
                    raise ValueError(f"Discrete parameter {param.name} needs bounds or choices")
            
            elif param.param_type == 'categorical':
                if param.choices:
                    param_values[param.name] = param.choices
                else:
                    raise ValueError(f"Categorical parameter {param.name} needs choices")
            
            elif param.param_type == 'continuous':
                if param.bounds:
                    start, end = param.bounds
                    # For grid search, discretize continuous parameters
                    n_points = 5  # Adjust as needed
                    param_values[param.name] = [start + i * (end - start) / (n_points - 1) 
                                              for i in range(n_points)]
                else:
                    raise ValueError(f"Continuous parameter {param.name} needs bounds")
        
        # Generate all combinations
        param_names = list(param_values.keys())
        param_combinations = []
        
        for values in product(*param_values.values()):
            combination = dict(zip(param_names, values))
            param_combinations.append(combination)
        
        return param_combinations
    
    def _sample_random_parameters(self, parameter_space: List[HyperparameterSpace]) -> Dict[str, Any]:
        """Sample random parameters from the space."""
        params = {}
        
        for param in parameter_space:
            if param.param_type == 'discrete':
                if param.bounds:
                    start, end = param.bounds
                    params[param.name] = np.random.randint(int(start), int(end) + 1) if HAS_NUMPY else int(start)
                elif param.choices:
                    params[param.name] = np.random.choice(param.choices) if HAS_NUMPY else param.choices[0]
            
            elif param.param_type == 'categorical':
                if param.choices:
                    params[param.name] = np.random.choice(param.choices) if HAS_NUMPY else param.choices[0]
            
            elif param.param_type == 'continuous':
                if param.bounds:
                    start, end = param.bounds
                    if param.log_scale:
                        log_start, log_end = np.log(start), np.log(end)
                        log_value = np.random.uniform(log_start, log_end) if HAS_NUMPY else log_start
                        params[param.name] = np.exp(log_value) if HAS_NUMPY else start
                    else:
                        params[param.name] = np.random.uniform(start, end) if HAS_NUMPY else start
        
        return params


class ABTestFramework:
    """A/B testing framework for comparing approaches."""
    
    def __init__(self, experiment_manager: ExperimentManager):
        self.experiment_manager = experiment_manager
        self.active_tests = {}
        
    def create_ab_test(self, test_name: str, variants: Dict[str, Dict[str, Any]],
                      sample_size: int = 100, significance_level: float = 0.05) -> str:
        """Create a new A/B test."""
        test_id = str(uuid.uuid4())[:8]
        
        test_config = {
            'test_name': test_name,
            'variants': variants,
            'sample_size': sample_size,
            'significance_level': significance_level,
            'created_at': datetime.now(),
            'status': 'created',
            'results': {}
        }
        
        self.active_tests[test_id] = test_config
        logger.info(f"Created A/B test: {test_id} - {test_name}")
        
        return test_id
    
    def run_ab_test(self, test_id: str, experiment_func: Callable,
                   metric_name: str = 'success_rate') -> Dict[str, Any]:
        """Run A/B test with statistical analysis."""
        if test_id not in self.active_tests:
            raise ValueError(f"A/B test {test_id} not found")
        
        test_config = self.active_tests[test_id]
        test_config['status'] = 'running'
        
        # Run experiments for each variant
        variant_results = {}
        
        for variant_name, variant_config in test_config['variants'].items():
            logger.info(f"Running variant {variant_name}")
            
            # Create experiments for this variant
            configs = []
            for i in range(test_config['sample_size']):
                config = ExperimentConfig(
                    name=f"ab_test_{test_id}_{variant_name}_{i}",
                    description=f"A/B test {test_config['test_name']} - {variant_name}",
                    parameters=variant_config,
                    metrics=[metric_name],
                    tags=["ab_test", variant_name, test_id]
                )
                configs.append(config)
            
            results = self.experiment_manager.run_batch_experiments(configs, experiment_func)
            
            # Extract metric values
            values = []
            for result in results:
                if result.status == "completed" and metric_name in result.metrics:
                    values.append(result.metrics[metric_name])
            
            variant_results[variant_name] = {
                'values': values,
                'mean': statistics.mean(values) if values else 0,
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'count': len(values),
                'success_rate': len([v for v in values if v > 0]) / len(values) if values else 0
            }
        
        # Statistical analysis
        statistical_results = self._analyze_ab_results(variant_results, test_config['significance_level'])
        
        test_results = {
            'test_id': test_id,
            'test_name': test_config['test_name'],
            'variant_results': variant_results,
            'statistical_analysis': statistical_results,
            'conclusion': self._generate_ab_conclusion(variant_results, statistical_results),
            'completed_at': datetime.now()
        }
        
        test_config['results'] = test_results
        test_config['status'] = 'completed'
        
        return test_results
    
    def _analyze_ab_results(self, variant_results: Dict[str, Dict], 
                           significance_level: float) -> Dict[str, Any]:
        """Perform statistical analysis on A/B test results."""
        if not HAS_SCIPY or len(variant_results) < 2:
            return {'error': 'Statistical analysis requires scipy and at least 2 variants'}
        
        variants = list(variant_results.keys())
        variant1, variant2 = variants[0], variants[1]
        
        values1 = variant_results[variant1]['values']
        values2 = variant_results[variant2]['values']
        
        if not values1 or not values2:
            return {'error': 'Insufficient data for statistical analysis'}
        
        # Perform t-test
        try:
            statistic, p_value = ttest_ind(values1, values2)
            
            is_significant = p_value < significance_level
            effect_size = abs(variant_results[variant1]['mean'] - variant_results[variant2]['mean'])
            
            return {
                'test_type': 't-test',
                'statistic': statistic,
                'p_value': p_value,
                'significance_level': significance_level,
                'is_significant': is_significant,
                'effect_size': effect_size,
                'better_variant': variant1 if variant_results[variant1]['mean'] > variant_results[variant2]['mean'] else variant2
            }
        
        except Exception as e:
            return {'error': f'Statistical test failed: {e}'}
    
    def _generate_ab_conclusion(self, variant_results: Dict[str, Dict],
                              statistical_results: Dict[str, Any]) -> str:
        """Generate human-readable conclusion."""
        if 'error' in statistical_results:
            return f"Could not determine statistical significance: {statistical_results['error']}"
        
        if statistical_results['is_significant']:
            better_variant = statistical_results['better_variant']
            return f"Variant '{better_variant}' is statistically significantly better " \
                   f"(p={statistical_results['p_value']:.4f})"
        else:
            return f"No statistically significant difference found " \
                   f"(p={statistical_results['p_value']:.4f})"


class BenchmarkRunner:
    """Automated benchmark runner for performance evaluation."""
    
    def __init__(self, experiment_manager: ExperimentManager):
        self.experiment_manager = experiment_manager
        self.benchmark_suites = {}
        
    def register_benchmark(self, benchmark_name: str, benchmark_func: Callable,
                         configurations: List[Dict[str, Any]], 
                         metrics: List[str] = None):
        """Register a benchmark suite."""
        self.benchmark_suites[benchmark_name] = {
            'function': benchmark_func,
            'configurations': configurations,
            'metrics': metrics or ['performance', 'accuracy'],
            'registered_at': datetime.now()
        }
        
        logger.info(f"Registered benchmark: {benchmark_name} with {len(configurations)} configurations")
    
    def run_benchmark(self, benchmark_name: str, iterations: int = 3) -> Dict[str, Any]:
        """Run a registered benchmark suite."""
        if benchmark_name not in self.benchmark_suites:
            raise ValueError(f"Benchmark {benchmark_name} not registered")
        
        benchmark = self.benchmark_suites[benchmark_name]
        
        all_results = []
        
        # Run each configuration
        for config_idx, config in enumerate(benchmark['configurations']):
            config_results = []
            
            # Run multiple iterations
            for iteration in range(iterations):
                exp_config = ExperimentConfig(
                    name=f"benchmark_{benchmark_name}_{config_idx}_{iteration}",
                    description=f"Benchmark {benchmark_name} config {config_idx} iteration {iteration}",
                    parameters=config,
                    metrics=benchmark['metrics'],
                    tags=["benchmark", benchmark_name],
                    iteration=iteration
                )
                
                exp_id = self.experiment_manager.create_experiment(exp_config)
                result = self.experiment_manager.run_experiment(
                    exp_id, benchmark['function'], **config
                )
                
                config_results.append(result)
            
            all_results.append({
                'configuration': config,
                'results': config_results
            })
        
        # Analyze results
        benchmark_report = self._analyze_benchmark_results(benchmark_name, all_results)
        
        return benchmark_report
    
    def _analyze_benchmark_results(self, benchmark_name: str, 
                                 all_results: List[Dict]) -> Dict[str, Any]:
        """Analyze benchmark results and generate report."""
        report = {
            'benchmark_name': benchmark_name,
            'configurations_tested': len(all_results),
            'total_experiments': sum(len(config['results']) for config in all_results),
            'configuration_analysis': [],
            'best_configuration': None,
            'performance_ranking': []
        }
        
        # Analyze each configuration
        for config_idx, config_data in enumerate(all_results):
            config = config_data['configuration']
            results = config_data['results']
            
            # Aggregate metrics across iterations
            successful_results = [r for r in results if r.status == "completed"]
            
            if not successful_results:
                continue
            
            config_analysis = {
                'configuration_id': config_idx,
                'configuration': config,
                'iterations': len(results),
                'successful_iterations': len(successful_results),
                'success_rate': len(successful_results) / len(results),
                'metrics': {}
            }
            
            # Calculate metric statistics
            all_metrics = set()
            for result in successful_results:
                all_metrics.update(result.metrics.keys())
            
            for metric in all_metrics:
                values = [r.metrics[metric] for r in successful_results if metric in r.metrics]
                
                if values:
                    config_analysis['metrics'][metric] = {
                        'mean': statistics.mean(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0,
                        'min': min(values),
                        'max': max(values),
                        'median': statistics.median(values)
                    }
            
            report['configuration_analysis'].append(config_analysis)
        
        # Find best configuration (by average performance)
        if report['configuration_analysis']:
            # Assuming 'performance' is a key metric (higher is better)
            best_config = max(
                report['configuration_analysis'],
                key=lambda x: x['metrics'].get('performance', {}).get('mean', 0)
            )
            report['best_configuration'] = best_config
            
            # Create performance ranking
            report['performance_ranking'] = sorted(
                report['configuration_analysis'],
                key=lambda x: x['metrics'].get('performance', {}).get('mean', 0),
                reverse=True
            )
        
        return report


class ResearchPipeline:
    """End-to-end research pipeline orchestrator."""
    
    def __init__(self, workspace_dir: Path = None):
        self.workspace_dir = workspace_dir or Path("./research_pipeline")
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.experiment_manager = ExperimentManager(self.workspace_dir / "experiments")
        self.hyperopt = HyperparameterOptimizer(self.experiment_manager)
        self.ab_test = ABTestFramework(self.experiment_manager)
        self.benchmark = BenchmarkRunner(self.experiment_manager)
        
        # Pipeline state
        self.pipeline_history = []
        self.current_pipeline = None
        
        logger.info(f"Research pipeline initialized: {self.workspace_dir}")
    
    def create_research_pipeline(self, pipeline_name: str, stages: List[Dict[str, Any]]) -> str:
        """Create a new research pipeline."""
        pipeline_id = str(uuid.uuid4())[:8]
        
        pipeline_config = {
            'pipeline_id': pipeline_id,
            'name': pipeline_name,
            'stages': stages,
            'created_at': datetime.now(),
            'status': 'created',
            'current_stage': 0,
            'results': {}
        }
        
        self.current_pipeline = pipeline_config
        
        # Save pipeline configuration
        pipeline_path = self.workspace_dir / f"pipeline_{pipeline_id}.json"
        with open(pipeline_path, 'w') as f:
            json.dump(pipeline_config, f, indent=2, default=str)
        
        logger.info(f"Created research pipeline: {pipeline_id} - {pipeline_name}")
        return pipeline_id
    
    def run_pipeline(self, pipeline_id: str = None) -> Dict[str, Any]:
        """Run a research pipeline."""
        if pipeline_id:
            # Load pipeline from file
            pipeline_path = self.workspace_dir / f"pipeline_{pipeline_id}.json"
            with open(pipeline_path, 'r') as f:
                pipeline_config = json.load(f)
            self.current_pipeline = pipeline_config
        
        if not self.current_pipeline:
            raise ValueError("No pipeline specified")
        
        pipeline = self.current_pipeline
        pipeline['status'] = 'running'
        
        logger.info(f"Running pipeline: {pipeline['name']}")
        
        # Execute each stage
        for stage_idx, stage in enumerate(pipeline['stages']):
            pipeline['current_stage'] = stage_idx
            
            logger.info(f"Executing stage {stage_idx + 1}/{len(pipeline['stages'])}: {stage['name']}")
            
            try:
                stage_result = self._execute_pipeline_stage(stage)
                pipeline['results'][f"stage_{stage_idx}"] = stage_result
                
            except Exception as e:
                logger.error(f"Pipeline stage {stage_idx} failed: {e}")
                pipeline['status'] = 'failed'
                pipeline['error'] = str(e)
                break
        else:
            pipeline['status'] = 'completed'
        
        pipeline['completed_at'] = datetime.now()
        
        # Save final pipeline state
        pipeline_path = self.workspace_dir / f"pipeline_{pipeline['pipeline_id']}.json"
        with open(pipeline_path, 'w') as f:
            json.dump(pipeline, f, indent=2, default=str)
        
        self.pipeline_history.append(pipeline)
        
        return pipeline
    
    def _execute_pipeline_stage(self, stage: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single pipeline stage."""
        stage_type = stage['type']
        stage_config = stage.get('config', {})
        
        if stage_type == 'hyperparameter_optimization':
            return self._execute_hyperopt_stage(stage_config)
        elif stage_type == 'ab_test':
            return self._execute_ab_test_stage(stage_config)
        elif stage_type == 'benchmark':
            return self._execute_benchmark_stage(stage_config)
        elif stage_type == 'experiment_batch':
            return self._execute_experiment_batch_stage(stage_config)
        else:
            raise ValueError(f"Unknown pipeline stage type: {stage_type}")
    
    def _execute_hyperopt_stage(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute hyperparameter optimization stage."""
        # This would need to be customized based on the specific objective function
        logger.info("Hyperparameter optimization stage - placeholder implementation")
        return {'status': 'completed', 'message': 'Hyperopt stage completed'}
    
    def _execute_ab_test_stage(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute A/B test stage."""
        logger.info("A/B test stage - placeholder implementation")
        return {'status': 'completed', 'message': 'A/B test stage completed'}
    
    def _execute_benchmark_stage(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute benchmark stage."""
        logger.info("Benchmark stage - placeholder implementation")
        return {'status': 'completed', 'message': 'Benchmark stage completed'}
    
    def _execute_experiment_batch_stage(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute experiment batch stage."""
        logger.info("Experiment batch stage - placeholder implementation")
        return {'status': 'completed', 'message': 'Experiment batch stage completed'}
    
    def get_pipeline_status(self, pipeline_id: str = None) -> Dict[str, Any]:
        """Get current pipeline status."""
        if pipeline_id:
            pipeline_path = self.workspace_dir / f"pipeline_{pipeline_id}.json"
            if pipeline_path.exists():
                with open(pipeline_path, 'r') as f:
                    pipeline = json.load(f)
                return pipeline
        
        return self.current_pipeline or {}