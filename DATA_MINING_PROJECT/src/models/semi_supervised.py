# ==============================================================================
# SEMI-SUPERVISED LEARNING MODULE
# ==============================================================================
"""
Module Bán giám sát (Semi-supervised Learning):
- Label Spreading
- Self-Training
- So sánh: Supervised-only vs Semi-supervised
- Learning curve theo % nhãn
- Phân tích pseudo-label sai ở review ngắn
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


class SemiSupervisedLearner:
    """
    Class học bán giám sát cho hotel reviews.
    Giả lập kịch bản thiếu nhãn aspect/sentiment theo domain.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SemiSupervisedLearner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.semi_config = self.config.get('semi_supervised', {})
        
        # Label percentages to test
        self.label_percentages = self.semi_config.get('label_percentages', [5, 10, 20, 50])
        
        # Models
        self.label_spreading_model = None
        self.self_training_model = None
        self.supervised_baseline = None
        
        # Results
        self.results = {}
        self.learning_curve_data = []
        
    def simulate_labeled_unlabeled(
        self,
        X: np.ndarray,
        y: np.ndarray,
        label_percentage: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate labeled and unlabeled data.
        
        Args:
            X: Feature matrix
            y: Labels
            label_percentage: Percentage of labels to keep
            
        Returns:
            Tuple of (X_labeled, y_labeled, X_unlabeled, X_test, y_test)
        """
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Further split train into labeled and unlabeled
        n_labeled = int(len(X_train) * (label_percentage / 100))
        
        if n_labeled < 1:
            n_labeled = 1
        
        # Sample labeled data
        labeled_indices = np.random.choice(len(X_train), n_labeled, replace=False)
        all_indices = np.arange(len(X_train))
        unlabeled_indices = np.setdiff1d(all_indices, labeled_indices)
        
        X_labeled = X_train[labeled_indices]
        y_labeled = y_train[labeled_indices]
        X_unlabeled = X_train[unlabeled_indices]
        
        return X_labeled, y_labeled, X_unlabeled, X_test, y_test
    
    def train_label_spreading(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
        params: Optional[Dict] = None
    ) -> Any:
        """
        Train Label Spreading model.
        
        Args:
            X_labeled: Labeled features
            y_labeled: Labeled labels
            X_unlabeled: Unlabeled features
            params: Model parameters
            
        Returns:
            Trained model
        """
        if params is None:
            params = {
                'kernel': 'knn',
                'n_neighbors': 7,
                'alpha': 0.2,
                'max_iter': 30
            }
        
        # Prepare data: combine labeled and unlabeled
        X_combined = np.vstack([X_labeled, X_unlabeled])
        
        # Create labels: -1 for unlabeled
        n_labeled = len(y_labeled)
        n_unlabeled = len(X_unlabeled)
        y_combined = np.concatenate([y_labeled, np.full(n_unlabeled, -1)])
        
        # Initialize model
        model = LabelSpreading(
            kernel=params.get('kernel', 'knn'),
            n_neighbors=params.get('n_neighbors', 7),
            alpha=params.get('alpha', 0.2),
            max_iter=params.get('max_iter', 30)
        )
        
        # Train
        model.fit(X_combined, y_combined)
        
        self.label_spreading_model = model
        
        return model
    
    def train_self_training(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
        params: Optional[Dict] = None
    ) -> Any:
        """
        Train Self-Training model.
        
        Args:
            X_labeled: Labeled features
            y_labeled: Labeled labels
            X_unlabeled: Unlabeled features
            params: Model parameters
            
        Returns:
            Trained model
        """
        if params is None:
            threshold = 0.9
            max_iter = 100
        else:
            threshold = params.get('threshold', 0.9)
            max_iter = params.get('max_iter', 100)
        
        # Initialize base estimator
        base_estimator = LogisticRegression(max_iter=1000)
        
        # Initialize self-training model
        # Note: sklearn >= 1.0 uses 'estimator' instead of 'base_estimator'
        try:
            model = SelfTrainingClassifier(
                estimator=base_estimator,
                threshold=threshold,
                max_iter=max_iter
            )
        except TypeError:
            # Fallback for older sklearn versions
            model = SelfTrainingClassifier(
                base_estimator=base_estimator,
                threshold=threshold,
                max_iter=max_iter
            )
        
        # Prepare data
        X_combined = np.vstack([X_labeled, X_unlabeled])
        y_combined = np.concatenate([y_labeled, np.full(len(X_unlabeled), -1)])
        
        # Train
        model.fit(X_combined, y_combined)
        
        self.self_training_model = model
        
        return model
    
    def train_supervised_baseline(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray
    ) -> Any:
        """
        Train supervised baseline (only labeled data).
        
        Args:
            X_labeled: Labeled features
            y_labeled: Labeled labels
            
        Returns:
            Trained model
        """
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
        model.fit(X_labeled, y_labeled)
        
        self.supervised_baseline = model
        
        return model
    
    def run_experiment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        label_percentage: float
    ) -> Dict[str, Any]:
        """
        Chạy experiment với một label percentage.
        
        Args:
            X: Feature matrix
            y: Labels
            label_percentage: Percentage of labels to keep
            
        Returns:
            Dictionary chứa kết quả
        """
        # Simulate labeled/unlabeled split
        X_labeled, y_labeled, X_unlabeled, X_test, y_test = self.simulate_labeled_unlabeled(
            X, y, label_percentage
        )
        
        print(f"\n[INFO] Running experiment with {label_percentage}% labels")
        print(f"  Labeled: {len(X_labeled)}, Unlabeled: {len(X_unlabeled)}, Test: {len(X_test)}")
        
        results = {
            'label_percentage': label_percentage,
            'n_labeled': len(X_labeled),
            'n_unlabeled': len(X_unlabeled),
            'n_test': len(X_test)
        }
        
        # Train supervised baseline
        self.train_supervised_baseline(X_labeled, y_labeled)
        y_pred_sup = self.supervised_baseline.predict(X_test)
        results['supervised'] = {
            'accuracy': float(accuracy_score(y_test, y_pred_sup)),
            'f1_macro': float(f1_score(y_test, y_pred_sup, average='macro'))
        }
        
        # Train Label Spreading
        try:
            self.train_label_spreading(X_labeled, y_labeled, X_unlabeled)
            y_pred_ls = self.label_spreading_model.predict(X_test)
            results['label_spreading'] = {
                'accuracy': float(accuracy_score(y_test, y_pred_ls)),
                'f1_macro': float(f1_score(y_test, y_pred_ls, average='macro'))
            }
        except Exception as e:
            print(f"[WARNING] Label Spreading failed: {e}")
            results['label_spreading'] = {'accuracy': 0, 'f1_macro': 0}
        
        # Train Self-Training
        try:
            self.train_self_training(X_labeled, y_labeled, X_unlabeled)
            y_pred_st = self.self_training_model.predict(X_test)
            results['self_training'] = {
                'accuracy': float(accuracy_score(y_test, y_pred_st)),
                'f1_macro': float(f1_score(y_test, y_pred_st, average='macro'))
            }
        except Exception as e:
            print(f"[WARNING] Self-Training failed: {e}")
            results['self_training'] = {'accuracy': 0, 'f1_macro': 0}
        
        # Calculate improvements
        if results['label_spreading']['f1_macro'] > 0:
            results['label_spreading']['improvement'] = (
                results['label_spreading']['f1_macro'] - results['supervised']['f1_macro']
            ) / max(results['supervised']['f1_macro'], 0.001)
        
        if results['self_training']['f1_macro'] > 0:
            results['self_training']['improvement'] = (
                results['self_training']['f1_macro'] - results['supervised']['f1_macro']
            ) / max(results['supervised']['f1_macro'], 0.001)
        
        return results
    
    def run_learning_curve_experiment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 3
    ) -> pd.DataFrame:
        """
        Chạy learning curve experiment với nhiều label percentages.
        
        Args:
            X: Feature matrix
            y: Labels
            n_repeats: Số lần lặp lại để lấy trung bình
            
        Returns:
            DataFrame chứa learning curve data
        """
        print("\n" + "=" * 60)
        print("LEARNING CURVE EXPERIMENT")
        print("=" * 60)
        
        all_results = []
        
        for label_pct in self.label_percentages:
            print(f"\n>>> Testing with {label_pct}% labels...")
            
            for repeat in range(n_repeats):
                np.random.seed(42 + repeat)
                
                results = self.run_experiment(X, y, label_pct)
                results['repeat'] = repeat
                all_results.append(results)
        
        # Convert to DataFrame
        df_results = pd.DataFrame(all_results)
        
        # Extract F1 scores from nested dictionaries
        df_results['supervised_f1'] = df_results['supervised'].apply(lambda x: x.get('f1_macro', 0) if isinstance(x, dict) else 0)
        df_results['label_spreading_f1'] = df_results['label_spreading'].apply(lambda x: x.get('f1_macro', 0) if isinstance(x, dict) else 0)
        df_results['self_training_f1'] = df_results['self_training'].apply(lambda x: x.get('f1_macro', 0) if isinstance(x, dict) else 0)
        
        # Aggregate by label percentage
        agg_results = df_results.groupby('label_percentage').agg({
            'supervised_f1': ['mean', 'std'],
            'label_spreading_f1': ['mean', 'std'],
            'self_training_f1': ['mean', 'std'],
        }).reset_index()
        
        # Flatten column names
        agg_results.columns = ['Label_%', 'Supervised_F1_mean', 'Supervised_F1_std',
                             'LabelSpreading_F1_mean', 'LabelSpreading_F1_std',
                             'SelfTraining_F1_mean', 'SelfTraining_F1_std']
        
        self.learning_curve_data = df_results
        
        print("\n" + "=" * 60)
        print("LEARNING CURVE SUMMARY")
        print("=" * 60)
        
        for _, row in agg_results.iterrows():
            label_pct = row['Label_%']
            print(f"\n{label_pct}% Labels:")
            print(f"  Supervised F1-macro: {row['Supervised_F1_mean']:.4f} (+/- {row['Supervised_F1_std']:.4f})")
            print(f"  Label Spreading F1-macro: {row['LabelSpreading_F1_mean']:.4f} (+/- {row['LabelSpreading_F1_std']:.4f})")
            print(f"  Self-Training F1-macro: {row['SelfTraining_F1_mean']:.4f} (+/- {row['SelfTraining_F1_std']:.4f})")
        
        return df_results
    
    def analyze_pseudo_label_errors(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
        y_true_unlabeled: Optional[np.ndarray] = None,
        review_lengths: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Phân tích pseudo-label errors, đặc biệt cho review ngắn.
        
        Args:
            X_labeled: Labeled features
            y_labeled: Labeled labels
            X_unlabeled: Unlabeled features
            y_true_unlabeled: True labels for unlabeled (for evaluation)
            review_lengths: Review lengths for analysis
            
        Returns:
            Dictionary chứa error analysis
        """
        print("\n" + "=" * 60)
        print("PSEUDO-LABEL ERROR ANALYSIS")
        print("=" * 60)
        
        # Train self-training to get pseudo labels
        self.train_self_training(X_labeled, y_labeled, X_unlabeled)
        
        # Get pseudo labels for unlabeled data
        pseudo_labels = self.self_training_model.predict(X_unlabeled)
        label_probs = self.self_training_model.predict_proba(X_unlabeled)
        max_probs = np.max(label_probs, axis=1)
        
        analysis = {
            'total_pseudo_labels': len(pseudo_labels),
            'confidence_stats': {
                'mean': float(np.mean(max_probs)),
                'std': float(np.std(max_probs)),
                'min': float(np.min(max_probs)),
                'max': float(np.max(max_probs))
            },
            'labels_distribution': {},
            'short_review_errors': {}
        }
        
        # Label distribution
        for label in np.unique(pseudo_labels):
            count = (pseudo_labels == label).sum()
            analysis['labels_distribution'][str(label)] = int(count)
        
        # If we have true labels, analyze errors
        if y_true_unlabeled is not None:
            errors = pseudo_labels != y_true_unlabeled
            error_rate = errors.sum() / len(errors)
            
            analysis['error_analysis'] = {
                'total_errors': int(errors.sum()),
                'error_rate': float(error_rate)
            }
            
            # Error analysis by confidence
            high_conf_mask = max_probs >= 0.9
            low_conf_mask = max_probs < 0.7
            
            if high_conf_mask.sum() > 0:
                high_conf_error_rate = errors[high_conf_mask].sum() / high_conf_mask.sum()
                analysis['error_analysis']['high_confidence_error_rate'] = float(high_conf_error_rate)
            
            if low_conf_mask.sum() > 0:
                low_conf_error_rate = errors[low_conf_mask].sum() / low_conf_mask.sum()
                analysis['error_analysis']['low_confidence_error_rate'] = float(low_conf_error_rate)
        
        # Short review analysis
        if review_lengths is not None:
            short_mask = review_lengths < 50  # Short reviews
            long_mask = review_lengths >= 50
            
            analysis['short_review_errors'] = {
                'short_reviews': int(short_mask.sum()),
                'long_reviews': int(long_mask.sum())
            }
            
            if short_mask.sum() > 0:
                analysis['short_review_errors']['avg_length_short'] = float(np.mean(review_lengths[short_mask]))
            
            if long_mask.sum() > 0:
                analysis['short_review_errors']['avg_length_long'] = float(np.mean(review_lengths[long_mask]))
        
        print(f"\nPseudo-label distribution: {analysis['labels_distribution']}")
        print(f"Confidence - Mean: {analysis['confidence_stats']['mean']:.4f}, Std: {analysis['confidence_stats']['std']:.4f}")
        
        return analysis
    
    def get_learning_curve_summary(self) -> pd.DataFrame:
        """
        Lấy tóm tắt learning curve.
        
        Returns:
            DataFrame summary
        """
        if self.learning_curve_data is None or len(self.learning_curve_data) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.learning_curve_data)
        
        # Flatten nested dict columns
        for col in ['supervised', 'label_spreading', 'self_training']:
            if col in df.columns:
                df[f'{col}_f1'] = df[col].apply(lambda x: x.get('f1_macro', 0) if isinstance(x, dict) else 0)
                df[f'{col}_acc'] = df[col].apply(lambda x: x.get('accuracy', 0) if isinstance(x, dict) else 0)
        
        summary = df.groupby('label_percentage').agg({
            'supervised_f1': ['mean', 'std'],
            'label_spreading_f1': ['mean', 'std'],
            'self_training_f1': ['mean', 'std']
        }).reset_index()
        
        summary.columns = ['Label_%', 'Supervised_F1_mean', 'Supervised_F1_std',
                         'LabelSpreading_F1_mean', 'LabelSpreading_F1_std',
                         'SelfTraining_F1_mean', 'SelfTraining_F1_std']
        
        return summary


if __name__ == "__main__":
    print("Testing SemiSupervisedLearner...")
    
    # Sample data
    np.random.seed(42)
    n_samples = 500
    n_features = 100
    
    X = np.random.rand(n_samples, n_features)
    y = np.random.choice(['positive', 'neutral', 'negative'], n_samples, p=[0.5, 0.25, 0.25])
    
    # Test experiment
    learner = SemiSupervisedLearner(
        config={'semi_supervised': {'label_percentages': [10, 20, 50]}}
    )
    
    results = learner.run_learning_curve_experiment(X, y, n_repeats=2)
    
    summary = learner.get_learning_curve_summary()
    print("\nLearning Curve Summary:")
    print(summary)
