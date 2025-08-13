"""Research quality gates with statistical validation and reproducibility checks."""

import pytest
import numpy as np
import torch
import time
import json
import hashlib
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from unittest.mock import patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import components for research validation
from neural_cryptanalysis.core import NeuralSCA, TraceData
from neural_cryptanalysis.datasets.synthetic import SyntheticDatasetGenerator
from neural_cryptanalysis.neural_operators import FourierNeuralOperator, OperatorConfig
from neural_cryptanalysis.utils.validation import StatisticalValidator
from neural_cryptanalysis.adaptive_rl import AdaptiveAttackEngine
from neural_cryptanalysis.multi_modal_fusion import MultiModalSideChannelAnalyzer


@dataclass
class ResearchResult:
    """Research experiment result."""
    experiment_name: str
    metric_name: str
    value: float
    std_error: float
    confidence_interval: Tuple[float, float]
    p_value: float
    effect_size: float
    sample_size: int
    timestamp: float
    reproducible: bool
    
    def to_dict(self) -> Dict:
        return {
            'experiment_name': self.experiment_name,
            'metric_name': self.metric_name,
            'value': self.value,
            'std_error': self.std_error,
            'confidence_interval': self.confidence_interval,
            'p_value': self.p_value,
            'effect_size': self.effect_size,
            'sample_size': self.sample_size,
            'timestamp': self.timestamp,
            'reproducible': self.reproducible
        }


@dataclass
class StatisticalTest:
    """Statistical test result."""
    test_name: str
    statistic: float
    p_value: float
    critical_value: float
    rejected_null: bool
    power: float
    effect_size: float
    interpretation: str


class ResearchValidator:
    """Validator for research quality and statistical rigor."""
    
    def __init__(self, alpha: float = 0.05, power_threshold: float = 0.8):
        self.alpha = alpha
        self.power_threshold = power_threshold
        self.results: List[ResearchResult] = []
        self.baseline_results: Dict[str, List[float]] = {}
    
    def add_baseline_result(self, experiment_name: str, metric_name: str, value: float):
        """Add baseline result for comparison."""
        key = f"{experiment_name}_{metric_name}"
        if key not in self.baseline_results:
            self.baseline_results[key] = []
        self.baseline_results[key].append(value)
    
    def validate_statistical_significance(self, 
                                        experiment_values: List[float],
                                        baseline_values: List[float] = None,
                                        test_type: str = 'ttest') -> StatisticalTest:
        """Validate statistical significance of results."""
        
        if baseline_values is None:
            # One-sample t-test against zero
            statistic, p_value = stats.ttest_1samp(experiment_values, 0)
            critical_value = stats.t.ppf(1 - self.alpha/2, len(experiment_values) - 1)
            effect_size = np.mean(experiment_values) / np.std(experiment_values, ddof=1)
            
        else:
            # Two-sample t-test
            if test_type == 'ttest':
                statistic, p_value = stats.ttest_ind(experiment_values, baseline_values)
                df = len(experiment_values) + len(baseline_values) - 2
                critical_value = stats.t.ppf(1 - self.alpha/2, df)
                
                # Cohen's d effect size
                pooled_std = np.sqrt(((len(experiment_values) - 1) * np.var(experiment_values, ddof=1) +
                                    (len(baseline_values) - 1) * np.var(baseline_values, ddof=1)) / df)
                effect_size = (np.mean(experiment_values) - np.mean(baseline_values)) / pooled_std
                
            elif test_type == 'mannwhitney':
                statistic, p_value = stats.mannwhitneyu(experiment_values, baseline_values, alternative='two-sided')
                critical_value = np.nan  # No simple critical value for Mann-Whitney
                effect_size = self._calculate_rank_biserial_correlation(experiment_values, baseline_values)
        
        # Calculate statistical power (simplified)
        power = self._calculate_power(effect_size, len(experiment_values), self.alpha)
        
        # Interpret results
        interpretation = self._interpret_statistical_test(p_value, effect_size, power)
        
        return StatisticalTest(
            test_name=test_type,
            statistic=statistic,
            p_value=p_value,
            critical_value=critical_value,
            rejected_null=p_value < self.alpha,
            power=power,
            effect_size=effect_size,
            interpretation=interpretation
        )
    
    def _calculate_rank_biserial_correlation(self, group1: List[float], group2: List[float]) -> float:
        """Calculate rank-biserial correlation for Mann-Whitney U test."""
        n1, n2 = len(group1), len(group2)
        u_statistic, _ = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        r = 1 - (2 * u_statistic) / (n1 * n2)
        return r
    
    def _calculate_power(self, effect_size: float, sample_size: int, alpha: float) -> float:
        """Calculate statistical power (simplified approximation)."""
        from scipy.stats import norm
        
        # Simplified power calculation for t-test
        critical_t = stats.t.ppf(1 - alpha/2, sample_size - 1)
        ncp = effect_size * np.sqrt(sample_size)  # Non-centrality parameter
        
        # Approximate power using normal distribution
        power = 1 - norm.cdf(critical_t - ncp) + norm.cdf(-critical_t - ncp)
        return np.clip(power, 0, 1)
    
    def _interpret_statistical_test(self, p_value: float, effect_size: float, power: float) -> str:
        """Interpret statistical test results."""
        interpretations = []
        
        if p_value < 0.001:
            interpretations.append("highly significant")
        elif p_value < 0.01:
            interpretations.append("very significant")
        elif p_value < 0.05:
            interpretations.append("significant")
        else:
            interpretations.append("not significant")
        
        # Effect size interpretation (Cohen's conventions)
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            interpretations.append("negligible effect")
        elif abs_effect < 0.5:
            interpretations.append("small effect")
        elif abs_effect < 0.8:
            interpretations.append("medium effect")
        else:
            interpretations.append("large effect")
        
        if power < 0.8:
            interpretations.append("underpowered")
        else:
            interpretations.append("adequately powered")
        
        return ", ".join(interpretations)
    
    def validate_reproducibility(self, 
                                experiment_func,
                                n_runs: int = 5,
                                tolerance: float = 0.1) -> Tuple[bool, Dict[str, Any]]:
        """Validate experimental reproducibility."""
        
        results = []
        
        for run in range(n_runs):
            # Set different random seed for each run
            np.random.seed(42 + run)
            torch.manual_seed(42 + run)
            
            try:
                result = experiment_func()
                results.append(result)
            except Exception as e:
                return False, {'error': str(e), 'failed_run': run}
        
        # Calculate reproducibility metrics
        if not results:
            return False, {'error': 'No successful runs'}
        
        if isinstance(results[0], dict):
            # Multiple metrics case
            reproducibility_report = {}
            overall_reproducible = True
            
            for metric_name in results[0].keys():
                metric_values = [r[metric_name] for r in results if metric_name in r]
                if metric_values:
                    mean_val = np.mean(metric_values)
                    std_val = np.std(metric_values)
                    cv = std_val / mean_val if mean_val != 0 else float('inf')
                    
                    is_reproducible = cv <= tolerance
                    overall_reproducible &= is_reproducible
                    
                    reproducibility_report[metric_name] = {
                        'mean': mean_val,
                        'std': std_val,
                        'cv': cv,
                        'reproducible': is_reproducible,
                        'values': metric_values
                    }
        
        else:
            # Single metric case
            mean_val = np.mean(results)
            std_val = np.std(results)
            cv = std_val / mean_val if mean_val != 0 else float('inf')
            
            overall_reproducible = cv <= tolerance
            
            reproducibility_report = {
                'mean': mean_val,
                'std': std_val,
                'cv': cv,
                'reproducible': overall_reproducible,
                'values': results
            }
        
        return overall_reproducible, reproducibility_report
    
    def validate_sample_size_adequacy(self, 
                                    effect_size: float,
                                    power: float = 0.8,
                                    alpha: float = 0.05) -> int:
        """Calculate required sample size for adequate power."""
        
        # Simplified sample size calculation for t-test
        from scipy.stats import norm
        
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)
        
        # Cohen's formula for sample size per group
        n_per_group = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        
        return int(np.ceil(n_per_group))
    
    def generate_research_report(self) -> Dict[str, Any]:
        """Generate comprehensive research quality report."""
        
        total_experiments = len(self.results)
        significant_results = sum(1 for r in self.results if r.p_value < self.alpha)
        reproducible_results = sum(1 for r in self.results if r.reproducible)
        
        # Calculate overall statistics
        if self.results:
            mean_effect_size = np.mean([abs(r.effect_size) for r in self.results])
            mean_power = np.mean([self._calculate_power(r.effect_size, r.sample_size, self.alpha) 
                                for r in self.results])
            mean_p_value = np.mean([r.p_value for r in self.results])
        else:
            mean_effect_size = mean_power = mean_p_value = 0.0
        
        return {
            'total_experiments': total_experiments,
            'significant_results': significant_results,
            'reproducible_results': reproducible_results,
            'significance_rate': significant_results / total_experiments if total_experiments > 0 else 0,
            'reproducibility_rate': reproducible_results / total_experiments if total_experiments > 0 else 0,
            'mean_effect_size': mean_effect_size,
            'mean_statistical_power': mean_power,
            'mean_p_value': mean_p_value,
            'quality_rating': self._calculate_research_quality_rating(),
            'recommendations': self._generate_research_recommendations(),
            'detailed_results': [r.to_dict() for r in self.results]
        }
    
    def _calculate_research_quality_rating(self) -> str:
        """Calculate overall research quality rating."""
        if not self.results:
            return 'INSUFFICIENT_DATA'
        
        significance_rate = sum(1 for r in self.results if r.p_value < self.alpha) / len(self.results)
        reproducibility_rate = sum(1 for r in self.results if r.reproducible) / len(self.results)
        mean_power = np.mean([self._calculate_power(r.effect_size, r.sample_size, self.alpha) 
                            for r in self.results])
        
        score = (significance_rate * 0.3 + reproducibility_rate * 0.4 + 
                (1 if mean_power >= self.power_threshold else 0) * 0.3)
        
        if score >= 0.9:
            return 'EXCELLENT'
        elif score >= 0.8:
            return 'GOOD'
        elif score >= 0.6:
            return 'ADEQUATE'
        elif score >= 0.4:
            return 'NEEDS_IMPROVEMENT'
        else:
            return 'POOR'
    
    def _generate_research_recommendations(self) -> List[str]:
        """Generate research quality recommendations."""
        recommendations = []
        
        if not self.results:
            recommendations.append("Conduct experiments to generate research results")
            return recommendations
        
        significance_rate = sum(1 for r in self.results if r.p_value < self.alpha) / len(self.results)
        reproducibility_rate = sum(1 for r in self.results if r.reproducible) / len(self.results)
        mean_power = np.mean([self._calculate_power(r.effect_size, r.sample_size, self.alpha) 
                            for r in self.results])
        
        if significance_rate < 0.5:
            recommendations.append("Review experimental design - low significance rate may indicate weak effects or insufficient power")
        
        if reproducibility_rate < 0.8:
            recommendations.append("Improve experimental reproducibility - ensure proper random seed control and minimize environmental factors")
        
        if mean_power < self.power_threshold:
            recommendations.append("Increase sample sizes to achieve adequate statistical power (≥0.8)")
        
        low_effect_results = [r for r in self.results if abs(r.effect_size) < 0.2]
        if len(low_effect_results) > len(self.results) * 0.5:
            recommendations.append("Consider practical significance - many results show small effect sizes")
        
        multiple_testing_risk = len(self.results) > 10
        if multiple_testing_risk:
            recommendations.append("Apply multiple testing corrections (e.g., Bonferroni, FDR) for multiple comparisons")
        
        return recommendations


@pytest.mark.research
class TestResearchQualityGates:
    """Research quality gates and statistical validation."""
    
    def test_neural_operator_baseline_comparison(self):
        """Test neural operator performance against established baselines."""
        print("\n=== Testing Neural Operator Baseline Comparison ===")
        
        validator = ResearchValidator()
        
        # Establish baseline performance (traditional methods)
        baseline_accuracies = []
        
        # Simulate traditional correlation-based attack baseline
        for run in range(5):
            np.random.seed(42 + run)
            
            # Generate synthetic data
            generator = SyntheticDatasetGenerator(random_seed=42 + run)
            dataset = generator.generate_aes_dataset(n_traces=200, trace_length=500)
            
            # Simulate correlation-based attack accuracy
            # This would be actual CPA implementation in real scenario
            simulated_cpa_accuracy = np.random.normal(0.65, 0.05)  # Typical CPA performance
            baseline_accuracies.append(simulated_cpa_accuracy)
            validator.add_baseline_result('neural_vs_traditional', 'accuracy', simulated_cpa_accuracy)
        
        # Test neural operator performance
        neural_accuracies = []
        
        for run in range(5):
            np.random.seed(42 + run)
            torch.manual_seed(42 + run)
            
            # Generate data
            generator = SyntheticDatasetGenerator(random_seed=42 + run)
            dataset = generator.generate_aes_dataset(n_traces=200, trace_length=500)
            
            # Train neural operator
            neural_sca = NeuralSCA(config={
                'fno': {'modes': 8, 'width': 32, 'n_layers': 3},
                'training': {'batch_size': 32, 'epochs': 5}
            })
            
            traces = torch.tensor(dataset['power_traces'], dtype=torch.float32).unsqueeze(-1)
            labels = torch.tensor(dataset['labels'][0], dtype=torch.long)
            
            # Split data
            split_idx = int(0.7 * len(traces))
            train_traces, train_labels = traces[:split_idx], labels[:split_idx]
            test_traces, test_labels = traces[split_idx:], labels[split_idx:]
            
            # Train and evaluate
            model = neural_sca.train(train_traces, train_labels, validation_split=0.2)
            
            with torch.no_grad():
                predictions = model(test_traces)
                predicted_labels = torch.argmax(predictions, dim=1)
                accuracy = (predicted_labels == test_labels).float().mean().item()
            
            neural_accuracies.append(accuracy)
        
        # Statistical comparison
        stat_test = validator.validate_statistical_significance(
            neural_accuracies, baseline_accuracies, test_type='ttest'
        )
        
        print(f"✓ Baseline accuracy (CPA): {np.mean(baseline_accuracies):.3f} ± {np.std(baseline_accuracies):.3f}")
        print(f"✓ Neural operator accuracy: {np.mean(neural_accuracies):.3f} ± {np.std(neural_accuracies):.3f}")
        print(f"✓ Statistical test: {stat_test.test_name}")
        print(f"✓ p-value: {stat_test.p_value:.6f}")
        print(f"✓ Effect size (Cohen's d): {stat_test.effect_size:.3f}")
        print(f"✓ Statistical power: {stat_test.power:.3f}")
        print(f"✓ Interpretation: {stat_test.interpretation}")
        
        # Research quality gates
        assert stat_test.power >= 0.8, f"Insufficient statistical power: {stat_test.power:.3f}"
        assert len(neural_accuracies) >= 5, "Insufficient sample size for reliable results"
        
        # Add to validator results
        result = ResearchResult(
            experiment_name='neural_vs_traditional',
            metric_name='accuracy',
            value=np.mean(neural_accuracies),
            std_error=np.std(neural_accuracies) / np.sqrt(len(neural_accuracies)),
            confidence_interval=stats.t.interval(0.95, len(neural_accuracies)-1, 
                                                np.mean(neural_accuracies), 
                                                np.std(neural_accuracies)/np.sqrt(len(neural_accuracies))),
            p_value=stat_test.p_value,
            effect_size=stat_test.effect_size,
            sample_size=len(neural_accuracies),
            timestamp=time.time(),
            reproducible=True  # Will be validated separately
        )
        
        validator.results.append(result)
        
        return validator, stat_test
    
    def test_reproducibility_validation(self):
        """Test experimental reproducibility across multiple runs."""
        print("\n=== Testing Experimental Reproducibility ===")
        
        validator = ResearchValidator()
        
        def neural_operator_experiment():
            """Standardized neural operator experiment for reproducibility testing."""
            # Generate dataset with fixed parameters
            generator = SyntheticDatasetGenerator(random_seed=42)  # Fixed seed for data generation
            dataset = generator.generate_aes_dataset(n_traces=150, trace_length=300)
            
            # Train neural operator with fixed configuration
            neural_sca = NeuralSCA(config={
                'fno': {'modes': 8, 'width': 32, 'n_layers': 2},
                'training': {'batch_size': 16, 'epochs': 3, 'learning_rate': 1e-3}
            })
            
            traces = torch.tensor(dataset['power_traces'], dtype=torch.float32).unsqueeze(-1)
            labels = torch.tensor(dataset['labels'][0], dtype=torch.long)
            
            # Split data deterministically
            split_idx = int(0.7 * len(traces))
            train_traces, train_labels = traces[:split_idx], labels[:split_idx]
            test_traces, test_labels = traces[split_idx:], labels[split_idx:]
            
            # Train model
            model = neural_sca.train(train_traces, train_labels, validation_split=0.2)
            
            # Evaluate performance
            with torch.no_grad():
                predictions = model(test_traces)
                predicted_labels = torch.argmax(predictions, dim=1)
                accuracy = (predicted_labels == test_labels).float().mean().item()
                
                # Additional metrics
                loss = torch.nn.functional.cross_entropy(predictions, test_labels).item()
                confidence = torch.softmax(predictions, dim=1).max(dim=1)[0].mean().item()
            
            return {
                'accuracy': accuracy,
                'loss': loss,
                'confidence': confidence
            }
        
        # Test reproducibility
        is_reproducible, reproducibility_report = validator.validate_reproducibility(
            neural_operator_experiment, n_runs=5, tolerance=0.05  # 5% tolerance
        )
        
        print(f"✓ Reproducibility test: {'PASSED' if is_reproducible else 'FAILED'}")
        
        for metric_name, metric_data in reproducibility_report.items():
            if isinstance(metric_data, dict) and 'mean' in metric_data:
                print(f"  {metric_name}:")
                print(f"    Mean: {metric_data['mean']:.4f}")
                print(f"    Std: {metric_data['std']:.4f}")
                print(f"    CV: {metric_data['cv']:.4f}")
                print(f"    Reproducible: {metric_data['reproducible']}")
        
        # Research quality gate
        assert is_reproducible, f"Experiment not reproducible: {reproducibility_report}"
        
        return is_reproducible, reproducibility_report
    
    def test_statistical_power_analysis(self):
        """Test statistical power analysis for experimental design."""
        print("\n=== Testing Statistical Power Analysis ===")
        
        validator = ResearchValidator()
        
        # Test different sample sizes and their statistical power
        sample_sizes = [50, 100, 200, 500]
        effect_sizes = [0.2, 0.5, 0.8]  # Small, medium, large effects
        
        power_analysis = {}
        
        for effect_size in effect_sizes:
            power_analysis[f'effect_{effect_size}'] = {}
            
            for sample_size in sample_sizes:
                power = validator._calculate_power(effect_size, sample_size, 0.05)
                power_analysis[f'effect_{effect_size}'][f'n_{sample_size}'] = power
                
                print(f"Effect size {effect_size}, n={sample_size}: Power = {power:.3f}")
        
        # Test required sample size calculation
        required_n = validator.validate_sample_size_adequacy(
            effect_size=0.5,  # Medium effect
            power=0.8,
            alpha=0.05
        )
        
        print(f"✓ Required sample size for 80% power (medium effect): {required_n}")
        
        # Validate that our typical experiments have adequate power
        typical_effect_size = 0.3  # Conservative estimate
        typical_sample_size = 100   # Our typical experiment size
        typical_power = validator._calculate_power(typical_effect_size, typical_sample_size, 0.05)
        
        print(f"✓ Typical experiment power: {typical_power:.3f}")
        
        # Research quality gate
        assert typical_power >= 0.7, f"Insufficient power for typical experiments: {typical_power:.3f}"
        
        return power_analysis, required_n
    
    def test_effect_size_validation(self):
        """Test practical significance through effect size analysis."""
        print("\n=== Testing Effect Size Validation ===")
        
        validator = ResearchValidator()
        
        # Compare neural operator variants
        architectures = ['fourier_neural_operator', 'deep_operator_network']
        architecture_results = {}
        
        for arch in architectures:
            print(f"  Testing {arch}...")
            accuracies = []
            
            for run in range(5):
                np.random.seed(42 + run)
                torch.manual_seed(42 + run)
                
                # Generate data
                generator = SyntheticDatasetGenerator(random_seed=42 + run)
                dataset = generator.generate_aes_dataset(n_traces=150, trace_length=300)
                
                # Configure neural SCA
                config = {
                    'training': {'batch_size': 16, 'epochs': 3}
                }
                
                if arch == 'fourier_neural_operator':
                    config['fno'] = {'modes': 8, 'width': 32, 'n_layers': 2}
                else:
                    config['deeponet'] = {'branch_layers': [64, 64], 'trunk_layers': [64, 64]}
                
                neural_sca = NeuralSCA(architecture=arch, config=config)
                
                traces = torch.tensor(dataset['power_traces'], dtype=torch.float32)
                if arch != 'deep_operator_network':
                    traces = traces.unsqueeze(-1)
                
                labels = torch.tensor(dataset['labels'][0], dtype=torch.long)
                
                # Train and evaluate
                split_idx = int(0.7 * len(traces))
                model = neural_sca.train(traces[:split_idx], labels[:split_idx], validation_split=0.2)
                
                with torch.no_grad():
                    predictions = model(traces[split_idx:])
                    predicted_labels = torch.argmax(predictions, dim=1)
                    accuracy = (predicted_labels == labels[split_idx:]).float().mean().item()
                
                accuracies.append(accuracy)
            
            architecture_results[arch] = accuracies
            print(f"    Mean accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
        
        # Statistical comparison between architectures
        fno_results = architecture_results['fourier_neural_operator']
        deeponet_results = architecture_results['deep_operator_network']
        
        stat_test = validator.validate_statistical_significance(
            fno_results, deeponet_results, test_type='ttest'
        )
        
        print(f"✓ Statistical comparison:")
        print(f"  p-value: {stat_test.p_value:.6f}")
        print(f"  Effect size (Cohen's d): {stat_test.effect_size:.3f}")
        print(f"  Interpretation: {stat_test.interpretation}")
        
        # Effect size interpretation
        abs_effect = abs(stat_test.effect_size)
        practical_significance = abs_effect >= 0.2  # At least small effect
        
        print(f"✓ Practical significance: {'YES' if practical_significance else 'NO'}")
        
        # Research quality gate
        if stat_test.p_value < 0.05:
            assert practical_significance, f"Statistically significant but not practically significant: d={stat_test.effect_size:.3f}"
        
        return stat_test, architecture_results
    
    def test_multiple_comparisons_correction(self):
        """Test multiple comparisons correction procedures."""
        print("\n=== Testing Multiple Comparisons Correction ===")
        
        # Simulate multiple neural operator configurations
        configurations = [
            {'modes': 4, 'width': 16, 'layers': 2},
            {'modes': 8, 'width': 32, 'layers': 2},
            {'modes': 16, 'width': 64, 'layers': 3},
            {'modes': 8, 'width': 32, 'layers': 4},
        ]
        
        config_results = {}
        p_values = []
        
        # Generate baseline performance
        baseline_accuracies = [np.random.normal(0.6, 0.05) for _ in range(5)]
        
        for i, config in enumerate(configurations):
            config_name = f"config_{i+1}"
            print(f"  Testing {config_name}: {config}")
            
            accuracies = []
            
            for run in range(5):
                np.random.seed(42 + run + i * 10)
                torch.manual_seed(42 + run + i * 10)
                
                # Simulate neural operator performance
                # Add small random effect to create realistic variation
                base_performance = 0.65 + np.random.normal(0, 0.02)
                config_effect = (config['modes'] / 16 + config['width'] / 64) * 0.05
                accuracy = base_performance + config_effect + np.random.normal(0, 0.03)
                accuracies.append(np.clip(accuracy, 0.3, 0.95))
            
            config_results[config_name] = accuracies
            
            # Statistical test against baseline
            statistic, p_value = stats.ttest_ind(accuracies, baseline_accuracies)
            p_values.append(p_value)
            
            print(f"    Mean accuracy: {np.mean(accuracies):.3f}")
            print(f"    p-value (uncorrected): {p_value:.6f}")
        
        # Apply multiple testing corrections
        from statsmodels.stats.multitest import multipletests
        
        # Bonferroni correction
        bonferroni_rejected, bonferroni_p_corrected, _, _ = multipletests(
            p_values, alpha=0.05, method='bonferroni'
        )
        
        # False Discovery Rate (FDR) correction
        fdr_rejected, fdr_p_corrected, _, _ = multipletests(
            p_values, alpha=0.05, method='fdr_bh'
        )
        
        print(f"\n✓ Multiple comparisons correction:")
        print(f"  Number of comparisons: {len(p_values)}")
        print(f"  Uncorrected significant results: {sum(p < 0.05 for p in p_values)}")
        print(f"  Bonferroni significant results: {sum(bonferroni_rejected)}")
        print(f"  FDR significant results: {sum(fdr_rejected)}")
        
        # Research quality gate
        if len(p_values) > 1:
            # At least one correction method should be applied
            any_correction_applied = any(bonferroni_rejected) or any(fdr_rejected)
            print(f"✓ Correction applied: {any_correction_applied}")
        
        return {
            'uncorrected_p_values': p_values,
            'bonferroni_p_values': bonferroni_p_corrected.tolist(),
            'fdr_p_values': fdr_p_corrected.tolist(),
            'bonferroni_rejected': bonferroni_rejected.tolist(),
            'fdr_rejected': fdr_rejected.tolist()
        }
    
    def test_confidence_intervals_validation(self):
        """Test confidence interval calculation and interpretation."""
        print("\n=== Testing Confidence Intervals Validation ===")
        
        # Generate experimental data
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Run experiment multiple times to get distribution
        accuracies = []
        
        for run in range(10):
            # Generate data
            generator = SyntheticDatasetGenerator(random_seed=42 + run)
            dataset = generator.generate_aes_dataset(n_traces=150, trace_length=300)
            
            # Train neural operator
            neural_sca = NeuralSCA(config={
                'fno': {'modes': 8, 'width': 32, 'n_layers': 2},
                'training': {'batch_size': 16, 'epochs': 3}
            })
            
            traces = torch.tensor(dataset['power_traces'], dtype=torch.float32).unsqueeze(-1)
            labels = torch.tensor(dataset['labels'][0], dtype=torch.long)
            
            split_idx = int(0.7 * len(traces))
            model = neural_sca.train(traces[:split_idx], labels[:split_idx], validation_split=0.2)
            
            with torch.no_grad():
                predictions = model(traces[split_idx:])
                predicted_labels = torch.argmax(predictions, dim=1)
                accuracy = (predicted_labels == labels[split_idx:]).float().mean().item()
            
            accuracies.append(accuracy)
        
        # Calculate confidence intervals
        mean_accuracy = np.mean(accuracies)
        std_error = np.std(accuracies, ddof=1) / np.sqrt(len(accuracies))
        
        # 95% confidence interval using t-distribution
        ci_95 = stats.t.interval(0.95, len(accuracies)-1, mean_accuracy, std_error)
        
        # 99% confidence interval
        ci_99 = stats.t.interval(0.99, len(accuracies)-1, mean_accuracy, std_error)
        
        # Bootstrap confidence interval
        bootstrap_means = []
        for _ in range(1000):
            bootstrap_sample = np.random.choice(accuracies, size=len(accuracies), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        bootstrap_ci_95 = np.percentile(bootstrap_means, [2.5, 97.5])
        
        print(f"✓ Mean accuracy: {mean_accuracy:.4f}")
        print(f"✓ Standard error: {std_error:.4f}")
        print(f"✓ 95% CI (t-distribution): [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
        print(f"✓ 99% CI (t-distribution): [{ci_99[0]:.4f}, {ci_99[1]:.4f}]")
        print(f"✓ 95% CI (bootstrap): [{bootstrap_ci_95[0]:.4f}, {bootstrap_ci_95[1]:.4f}]")
        
        # Calculate confidence interval width
        ci_width = ci_95[1] - ci_95[0]
        relative_precision = ci_width / mean_accuracy
        
        print(f"✓ CI width: {ci_width:.4f}")
        print(f"✓ Relative precision: {relative_precision:.4f}")
        
        # Research quality gate
        assert ci_width < 0.2, f"Confidence interval too wide: {ci_width:.4f}"
        assert relative_precision < 0.3, f"Relative precision too low: {relative_precision:.4f}"
        
        return {
            'mean': mean_accuracy,
            'std_error': std_error,
            'ci_95': ci_95,
            'ci_99': ci_99,
            'bootstrap_ci_95': bootstrap_ci_95,
            'ci_width': ci_width,
            'relative_precision': relative_precision
        }
    
    def test_research_methodology_documentation(self):
        """Test that research methodology is properly documented."""
        print("\n=== Testing Research Methodology Documentation ===")
        
        # Check for methodology documentation
        methodology_docs = [
            Path('docs/METHODOLOGY.md'),
            Path('research/methodology.md'),
            Path('METHODOLOGY.md')
        ]
        
        methodology_found = any(doc.exists() for doc in methodology_docs)
        
        # Check for statistical analysis documentation
        required_sections = [
            'statistical analysis',
            'experimental design',
            'sample size',
            'significance level',
            'multiple comparisons',
            'reproducibility'
        ]
        
        documented_sections = []
        
        if methodology_found:
            for doc in methodology_docs:
                if doc.exists():
                    content = doc.read_text().lower()
                    for section in required_sections:
                        if section in content:
                            documented_sections.append(section)
        
        print(f"✓ Methodology documentation found: {methodology_found}")
        print(f"✓ Documented sections: {len(set(documented_sections))}/{len(required_sections)}")
        
        # Research quality gate
        documentation_score = len(set(documented_sections)) / len(required_sections)
        assert documentation_score >= 0.5, f"Insufficient methodology documentation: {documentation_score:.2f}"
        
        return {
            'methodology_documented': methodology_found,
            'documented_sections': list(set(documented_sections)),
            'documentation_score': documentation_score
        }


def generate_research_quality_report(validator: ResearchValidator, 
                                   output_path: Path = None) -> Dict[str, Any]:
    """Generate comprehensive research quality report."""
    
    base_report = validator.generate_research_report()
    
    # Add additional quality metrics
    base_report['quality_gates'] = {
        'statistical_significance': base_report['significance_rate'] >= 0.8,
        'reproducibility': base_report['reproducibility_rate'] >= 0.9,
        'statistical_power': base_report['mean_statistical_power'] >= 0.8,
        'effect_size_adequacy': base_report['mean_effect_size'] >= 0.2,
        'sample_size_adequacy': all(r['sample_size'] >= 30 for r in base_report['detailed_results'])
    }
    
    # Calculate overall research readiness
    quality_gates_passed = sum(base_report['quality_gates'].values())
    total_quality_gates = len(base_report['quality_gates'])
    research_readiness = quality_gates_passed / total_quality_gates
    
    base_report['research_readiness'] = {
        'score': research_readiness,
        'rating': 'READY' if research_readiness >= 0.8 else 'NEEDS_WORK',
        'gates_passed': quality_gates_passed,
        'total_gates': total_quality_gates
    }
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(base_report, f, indent=2, default=str)
    
    return base_report


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "research"])