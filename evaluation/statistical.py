"""Statistical significance testing framework for compression experiments."""

import numpy as np
from scipy import stats
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import logging
from collections import defaultdict
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StatisticalTestResult:
    """Result from a statistical test."""
    test_name: str
    p_value: float
    statistic: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    significant: bool
    interpretation: str


class StatisticalTester:
    """Framework for statistical significance testing."""
    
    def __init__(
        self,
        alpha: float = 0.05,
        multiple_comparison_correction: str = "bonferroni"
    ):
        """
        Initialize statistical tester.
        
        Args:
            alpha: Significance level
            multiple_comparison_correction: Method for multiple comparison correction
        """
        self.alpha = alpha
        self.correction_method = multiple_comparison_correction
    
    def compare_compression_methods(
        self,
        method_results: Dict[str, List[float]],
        metric: str = "compression_ratio"
    ) -> Dict[str, StatisticalTestResult]:
        """
        Compare multiple compression methods.
        
        Args:
            method_results: Dictionary mapping method names to lists of results
            metric: Metric being compared
            
        Returns:
            Dictionary of pairwise comparison results
        """
        methods = list(method_results.keys())
        n_comparisons = len(methods) * (len(methods) - 1) // 2
        
        # Adjust alpha for multiple comparisons
        if self.correction_method == "bonferroni":
            adjusted_alpha = self.alpha / n_comparisons
        else:
            adjusted_alpha = self.alpha
        
        results = {}
        
        # Pairwise comparisons
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                method1, method2 = methods[i], methods[j]
                data1 = np.array(method_results[method1])
                data2 = np.array(method_results[method2])
                
                # Paired t-test if same samples
                if len(data1) == len(data2):
                    result = self.paired_t_test(
                        data1, data2,
                        f"{method1}_vs_{method2}",
                        adjusted_alpha
                    )
                else:
                    # Independent t-test
                    result = self.independent_t_test(
                        data1, data2,
                        f"{method1}_vs_{method2}",
                        adjusted_alpha
                    )
                
                results[f"{method1}_vs_{method2}"] = result
        
        # Overall ANOVA if more than 2 groups
        if len(methods) > 2:
            f_stat, p_value = stats.f_oneway(*[method_results[m] for m in methods])
            results['overall_anova'] = StatisticalTestResult(
                test_name="One-way ANOVA",
                p_value=p_value,
                statistic=f_stat,
                effect_size=self._calculate_eta_squared(method_results),
                confidence_interval=(0, 0),  # Not applicable for ANOVA
                significant=p_value < self.alpha,
                interpretation=self._interpret_anova(p_value, self.alpha)
            )
        
        return results
    
    def paired_t_test(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        name: str,
        alpha: Optional[float] = None
    ) -> StatisticalTestResult:
        """
        Perform paired t-test.
        
        Args:
            data1: First dataset
            data2: Second dataset
            name: Test name
            alpha: Significance level
            
        Returns:
            Test result
        """
        if alpha is None:
            alpha = self.alpha
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(data1, data2)
        
        # Effect size (Cohen's d)
        diff = data1 - data2
        effect_size = np.mean(diff) / np.std(diff, ddof=1)
        
        # Confidence interval for mean difference
        n = len(diff)
        se = stats.sem(diff)
        ci = stats.t.interval(1 - alpha, n - 1, loc=np.mean(diff), scale=se)
        
        return StatisticalTestResult(
            test_name=f"Paired t-test: {name}",
            p_value=p_value,
            statistic=t_stat,
            effect_size=effect_size,
            confidence_interval=ci,
            significant=p_value < alpha,
            interpretation=self._interpret_t_test(p_value, effect_size, alpha)
        )
    
    def independent_t_test(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        name: str,
        alpha: Optional[float] = None
    ) -> StatisticalTestResult:
        """
        Perform independent samples t-test.
        
        Args:
            data1: First dataset
            data2: Second dataset
            name: Test name
            alpha: Significance level
            
        Returns:
            Test result
        """
        if alpha is None:
            alpha = self.alpha
        
        # Check for equal variances
        _, p_levene = stats.levene(data1, data2)
        equal_var = p_levene > 0.05
        
        # Independent t-test
        t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(data1) - 1) * np.var(data1, ddof=1) + 
             (len(data2) - 1) * np.var(data2, ddof=1)) /
            (len(data1) + len(data2) - 2)
        )
        effect_size = (np.mean(data1) - np.mean(data2)) / pooled_std
        
        # Confidence interval for mean difference
        mean_diff = np.mean(data1) - np.mean(data2)
        se_diff = pooled_std * np.sqrt(1/len(data1) + 1/len(data2))
        df = len(data1) + len(data2) - 2
        ci = stats.t.interval(1 - alpha, df, loc=mean_diff, scale=se_diff)
        
        return StatisticalTestResult(
            test_name=f"Independent t-test: {name}",
            p_value=p_value,
            statistic=t_stat,
            effect_size=effect_size,
            confidence_interval=ci,
            significant=p_value < alpha,
            interpretation=self._interpret_t_test(p_value, effect_size, alpha)
        )
    
    def wilcoxon_test(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        name: str,
        alpha: Optional[float] = None
    ) -> StatisticalTestResult:
        """
        Perform Wilcoxon signed-rank test (non-parametric).
        
        Args:
            data1: First dataset
            data2: Second dataset
            name: Test name
            alpha: Significance level
            
        Returns:
            Test result
        """
        if alpha is None:
            alpha = self.alpha
        
        # Wilcoxon test
        statistic, p_value = stats.wilcoxon(data1, data2)
        
        # Effect size (r = Z / sqrt(N))
        n = len(data1)
        z_score = stats.norm.ppf(1 - p_value/2)  # Approximate Z-score
        effect_size = z_score / np.sqrt(n)
        
        # Hodges-Lehmann estimator for confidence interval
        differences = []
        for i in range(n):
            for j in range(i, n):
                differences.append((data1[i] - data2[i] + data1[j] - data2[j]) / 2)
        
        differences = sorted(differences)
        ci_lower = np.percentile(differences, (alpha/2) * 100)
        ci_upper = np.percentile(differences, (1 - alpha/2) * 100)
        
        return StatisticalTestResult(
            test_name=f"Wilcoxon test: {name}",
            p_value=p_value,
            statistic=statistic,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            significant=p_value < alpha,
            interpretation=self._interpret_wilcoxon(p_value, effect_size, alpha)
        )
    
    def bootstrap_confidence_interval(
        self,
        data: np.ndarray,
        statistic_func: callable,
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Calculate bootstrap confidence interval.
        
        Args:
            data: Input data
            statistic_func: Function to calculate statistic
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level
            
        Returns:
            (statistic, (ci_lower, ci_upper))
        """
        # Original statistic
        original_stat = statistic_func(data)
        
        # Bootstrap samples
        bootstrap_stats = []
        n = len(data)
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(bootstrap_stat)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_stats, (alpha/2) * 100)
        ci_upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)
        
        return original_stat, (ci_lower, ci_upper)
    
    def compare_accuracy_retention(
        self,
        baseline_accuracies: List[float],
        compressed_accuracies: List[float],
        model_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare accuracy retention across models.
        
        Args:
            baseline_accuracies: Original model accuracies
            compressed_accuracies: Compressed model accuracies
            model_names: Optional model names
            
        Returns:
            Statistical comparison results
        """
        # Calculate retention percentages
        retentions = [
            (comp / base) * 100 
            for base, comp in zip(baseline_accuracies, compressed_accuracies)
        ]
        
        results = {
            'mean_retention': np.mean(retentions),
            'std_retention': np.std(retentions, ddof=1),
            'min_retention': np.min(retentions),
            'max_retention': np.max(retentions),
            'retention_ci': self.bootstrap_confidence_interval(
                np.array(retentions),
                np.mean
            )[1]
        }
        
        # Test if retention is significantly different from 100%
        t_stat, p_value = stats.ttest_1samp(retentions, 100)
        
        results['retention_test'] = StatisticalTestResult(
            test_name="One-sample t-test (retention vs 100%)",
            p_value=p_value,
            statistic=t_stat,
            effect_size=(np.mean(retentions) - 100) / np.std(retentions, ddof=1),
            confidence_interval=results['retention_ci'],
            significant=p_value < self.alpha,
            interpretation=self._interpret_retention_test(
                np.mean(retentions), p_value, self.alpha
            )
        )
        
        # Per-model analysis if names provided
        if model_names:
            results['per_model'] = {}
            for name, base, comp, ret in zip(
                model_names, baseline_accuracies, compressed_accuracies, retentions
            ):
                results['per_model'][name] = {
                    'baseline': base,
                    'compressed': comp,
                    'retention': ret,
                    'significant_drop': (base - comp) > 1.0  # >1% drop
                }
        
        return results
    
    def effect_size_interpretation(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _calculate_eta_squared(self, group_data: Dict[str, List[float]]) -> float:
        """Calculate eta squared effect size for ANOVA."""
        # Flatten all data
        all_data = []
        group_labels = []
        
        for group, data in group_data.items():
            all_data.extend(data)
            group_labels.extend([group] * len(data))
        
        # Calculate sum of squares
        grand_mean = np.mean(all_data)
        
        ss_between = 0
        for group, data in group_data.items():
            group_mean = np.mean(data)
            ss_between += len(data) * (group_mean - grand_mean) ** 2
        
        ss_total = np.sum([(x - grand_mean) ** 2 for x in all_data])
        
        # Eta squared
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        return eta_squared
    
    def _interpret_t_test(self, p_value: float, effect_size: float, alpha: float) -> str:
        """Interpret t-test results."""
        sig_str = "significant" if p_value < alpha else "not significant"
        effect_str = self.effect_size_interpretation(effect_size)
        
        return f"The difference is {sig_str} (p={p_value:.4f}) with {effect_str} effect size (d={effect_size:.3f})"
    
    def _interpret_anova(self, p_value: float, alpha: float) -> str:
        """Interpret ANOVA results."""
        if p_value < alpha:
            return f"Significant differences exist between groups (p={p_value:.4f})"
        else:
            return f"No significant differences between groups (p={p_value:.4f})"
    
    def _interpret_wilcoxon(self, p_value: float, effect_size: float, alpha: float) -> str:
        """Interpret Wilcoxon test results."""
        sig_str = "significant" if p_value < alpha else "not significant"
        
        # Effect size interpretation for r
        if abs(effect_size) < 0.1:
            effect_str = "negligible"
        elif abs(effect_size) < 0.3:
            effect_str = "small"
        elif abs(effect_size) < 0.5:
            effect_str = "medium"
        else:
            effect_str = "large"
        
        return f"The difference is {sig_str} (p={p_value:.4f}) with {effect_str} effect size (r={effect_size:.3f})"
    
    def _interpret_retention_test(self, mean_retention: float, p_value: float, alpha: float) -> str:
        """Interpret retention test results."""
        if p_value < alpha:
            if mean_retention < 100:
                return f"Significant accuracy degradation detected (mean retention: {mean_retention:.2f}%)"
            else:
                return f"Significant accuracy improvement detected (mean retention: {mean_retention:.2f}%)"
        else:
            return f"No significant change in accuracy (mean retention: {mean_retention:.2f}%)"


def run_significance_tests(
    experiment_results: Dict[str, Any],
    baseline_results: Dict[str, Any],
    metrics: List[str] = ["compression_ratio", "accuracy", "inference_time"],
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run comprehensive significance tests on experiment results.
    
    Args:
        experiment_results: Results from compression experiments
        baseline_results: Results from baseline methods
        metrics: Metrics to test
        output_file: Optional file to save results
        
    Returns:
        Dictionary of test results
    """
    tester = StatisticalTester()
    all_results = {}
    
    for metric in metrics:
        logger.info(f"Running significance tests for {metric}")
        
        # Extract data for each method
        method_data = defaultdict(list)
        
        # Add INWF results
        if metric in experiment_results:
            method_data['INWF'] = experiment_results[metric]
        
        # Add baseline results
        for baseline_name, baseline_data in baseline_results.items():
            if metric in baseline_data:
                method_data[baseline_name] = baseline_data[metric]
        
        if len(method_data) < 2:
            logger.warning(f"Insufficient data for {metric} comparison")
            continue
        
        # Run comparisons
        test_results = tester.compare_compression_methods(
            dict(method_data),
            metric
        )
        
        all_results[metric] = test_results
    
    # Create summary report
    summary = create_significance_summary(all_results)
    
    # Save results if requested
    if output_file:
        import json
        
        # Convert results to serializable format
        serializable = {}
        for metric, tests in all_results.items():
            serializable[metric] = {}
            for test_name, result in tests.items():
                serializable[metric][test_name] = {
                    'p_value': result.p_value,
                    'statistic': result.statistic,
                    'effect_size': result.effect_size,
                    'significant': result.significant,
                    'interpretation': result.interpretation
                }
        
        with open(output_file, 'w') as f:
            json.dump({
                'test_results': serializable,
                'summary': summary
            }, f, indent=2)
    
    return all_results


def create_significance_summary(test_results: Dict[str, Dict[str, StatisticalTestResult]]) -> Dict[str, Any]:
    """Create a summary of significance test results."""
    summary = {
        'total_tests': 0,
        'significant_results': 0,
        'metrics': {}
    }
    
    for metric, tests in test_results.items():
        metric_summary = {
            'n_tests': len(tests),
            'n_significant': sum(1 for t in tests.values() if t.significant),
            'significant_comparisons': []
        }
        
        for test_name, result in tests.items():
            summary['total_tests'] += 1
            if result.significant:
                summary['significant_results'] += 1
                metric_summary['significant_comparisons'].append({
                    'comparison': test_name,
                    'p_value': result.p_value,
                    'effect_size': result.effect_size,
                    'interpretation': result.interpretation
                })
        
        summary['metrics'][metric] = metric_summary
    
    summary['significance_rate'] = (
        summary['significant_results'] / summary['total_tests'] * 100
        if summary['total_tests'] > 0 else 0
    )
    
    return summary