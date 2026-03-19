# ==============================================================================
# METRICS CALCULATION MODULE
# ==============================================================================
"""
Module tính toán các metrics đánh giá:
- Classification metrics (F1, Precision, Recall, etc.)
- Regression metrics (MAE, RMSE, R2)
- Clustering metrics (Silhouette)
- Custom metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.metrics import (
    # Classification
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, cohen_kappa_score,
    roc_auc_score, log_loss,
    # Regression
    mean_absolute_error, mean_squared_error, r2_score,
    # Clustering
    silhouette_score, silhouette_samples,
    # Utilities
    auc, roc_curve
)
import warnings
warnings.filterwarnings('ignore')


class MetricsCalculator:
    """
    Class tính toán và so sánh các metrics.
    """
    
    def __init__(self):
        """Initialize MetricsCalculator."""
        self.results = {}
    
    # ==========================================================================
    # CLASSIFICATION METRICS
    # ==========================================================================
    
    def classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = 'macro'
    ) -> Dict[str, float]:
        """
        Tính các classification metrics cơ bản.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging method for multiclass
            
        Returns:
            Dictionary chứa các metrics
        """
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, average=average, zero_division=0)),
            'kappa': float(cohen_kappa_score(y_true, y_pred))
        }
    
    def detailed_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Tạo detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: List of class labels
            
        Returns:
            Dictionary chứa detailed report
        """
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        return {
            'metrics': report,
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': self._extract_per_class_metrics(report)
        }
    
    def _extract_per_class_metrics(self, report: Dict) -> Dict[str, Dict]:
        """Extract per-class metrics from sklearn report."""
        classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
        
        return {
            cls: {
                'precision': report[cls]['precision'],
                'recall': report[cls]['recall'],
                'f1_score': report[cls]['f1-score'],
                'support': report[cls]['support']
            }
            for cls in classes
        }
    
    # ==========================================================================
    # REGRESSION METRICS
    # ==========================================================================
    
    def regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Tính các regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary chứa các metrics
        """
        return {
            'MAE': float(mean_absolute_error(y_true, y_pred)),
            'MSE': float(mean_squared_error(y_true, y_pred)),
            'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'R2': float(r2_score(y_true, y_pred))
        }
    
    # ==========================================================================
    # CLUSTERING METRICS
    # ==========================================================================
    
    def clustering_metrics(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Tính các clustering metrics.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            sample_size: Size for sampling (for large datasets)
            
        Returns:
            Dictionary chứa các metrics
        """
        # Sample if needed
        if sample_size and X.shape[0] > sample_size:
            indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[indices]
            labels_sample = labels[indices]
        else:
            X_sample = X
            labels_sample = labels
        
        # Calculate metrics
        n_clusters = len(set(labels_sample))
        
        metrics = {
            'n_clusters': n_clusters,
            'silhouette_score': float(silhouette_score(X_sample, labels_sample))
        }
        
        # Cluster sizes
        unique, counts = np.unique(labels_sample, return_counts=True)
        metrics['cluster_sizes'] = dict(zip(unique.tolist(), counts.tolist()))
        metrics['avg_cluster_size'] = float(np.mean(counts))
        
        return metrics
    
    # ==========================================================================
    # COMPARISON METHODS
    # ==========================================================================
    
    def compare_classification_models(
        self,
        results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> pd.DataFrame:
        """
        So sánh nhiều classification models.
        
        Args:
            results_dict: Dictionary mapping model name to (y_true, y_pred)
            
        Returns:
            DataFrame so sánh
        """
        comparison_data = []
        
        for model_name, (y_true, y_pred) in results_dict.items():
            metrics = self.classification_metrics(y_true, y_pred)
            metrics['Model'] = model_name
            comparison_data.append(metrics)
        
        df = pd.DataFrame(comparison_data)
        df = df[['Model', 'accuracy', 'precision', 'recall', 'f1_score', 'kappa']]
        df = df.sort_values('f1_score', ascending=False)
        
        return df
    
    def compare_regression_models(
        self,
        results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> pd.DataFrame:
        """
        So sánh nhiều regression models.
        
        Args:
            results_dict: Dictionary mapping model name to (y_true, y_pred)
            
        Returns:
            DataFrame so sánh
        """
        comparison_data = []
        
        for model_name, (y_true, y_pred) in results_dict.items():
            metrics = self.regression_metrics(y_true, y_pred)
            metrics['Model'] = model_name
            comparison_data.append(metrics)
        
        df = pd.DataFrame(comparison_data)
        df = df[['Model', 'MAE', 'RMSE', 'R2']]
        df = df.sort_values('RMSE', ascending=True)
        
        return df
    
    # ==========================================================================
    # STATISTICAL TESTS
    # ==========================================================================
    
    def statistical_significance_test(
        self,
        scores1: np.ndarray,
        scores2: np.ndarray,
        test_type: str = 'paired_t'
    ) -> Dict[str, float]:
        """
        Thực hiện statistical significance test.
        
        Args:
            scores1: Scores from first model
            scores2: Scores from second model
            test_type: Type of test ('paired_t', 'wilcoxon')
            
        Returns:
            Dictionary chứa test results
        """
        try:
            from scipy import stats
            
            if test_type == 'paired_t':
                statistic, p_value = stats.ttest_rel(scores1, scores2)
            elif test_type == 'wilcoxon':
                statistic, p_value = stats.wilcoxon(scores1, scores2)
            else:
                return {'error': 'Unknown test type'}
            
            return {
                'test': test_type,
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant_05': p_value < 0.05,
                'significant_01': p_value < 0.01
            }
        except ImportError:
            return {'error': 'scipy not available'}
    
    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================
    
    def format_metrics_table(
        self,
        metrics_dict: Dict[str, float],
        precision: int = 4
    ) -> str:
        """Format metrics dictionary as a nice table string."""
        lines = []
        lines.append("-" * 40)
        lines.append("METRICS SUMMARY")
        lines.append("-" * 40)
        
        for key, value in metrics_dict.items():
            if isinstance(value, float):
                lines.append(f"{key:20s}: {value:.{precision}f}")
            else:
                lines.append(f"{key:20s}: {value}")
        
        lines.append("-" * 40)
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Test metrics calculator
    print("Testing MetricsCalculator...")
    
    calc = MetricsCalculator()
    
    # Classification test
    y_true = np.array(['a', 'b', 'a', 'b', 'a', 'b'])
    y_pred = np.array(['a', 'b', 'b', 'a', 'a', 'b'])
    
    metrics = calc.classification_metrics(y_true, y_pred)
    print("\nClassification Metrics:")
    print(calc.format_metrics_table(metrics))
    
    # Regression test
    y_true_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred_reg = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    
    reg_metrics = calc.regression_metrics(y_true_reg, y_pred_reg)
    print("\nRegression Metrics:")
    print(calc.format_metrics_table(reg_metrics))
