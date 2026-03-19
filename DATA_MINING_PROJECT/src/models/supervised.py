# ==============================================================================
# SUPERVISED CLASSIFICATION MODULE
# ==============================================================================
"""
Module phân lớp Sentiment/Aspect Classification:
- Baseline models: Logistic Regression, Naive Bayes
- Strong model: Random Forest
- Metric: F1-macro, Confusion Matrix
- Error Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    accuracy_score, precision_score, recall_score,
    cohen_kappa_score
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib
import warnings
warnings.filterwarnings('ignore')


class SentimentClassifier:
    """
    Class phân lớp sentiment cho hotel reviews.
    Supports baseline models (LogisticRegression, NaiveBayes) and strong model (RandomForest).
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SentimentClassifier.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.clf_config = self.config.get('classification', {})
        
        # Model parameters
        self.cv_folds = self.clf_config.get('cv_folds', 5)
        self.test_size = self.clf_config.get('test_size', 0.2)
        self.random_state = self.clf_config.get('random_state', 42)
        
        # Models
        self.baselines = {}
        self.baseline_nb_scaler = None  # Scaler for NaiveBayes (requires non-negative)
        self.strong_model = None
        self.best_model = None
        
        # Data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoder = LabelEncoder()
        
        # Results
        self.results = {}
        self.error_analysis = {}
        
    def prepare_labels(
        self,
        df: pd.DataFrame,
        rating_column: str = 'rating',
        mapping: Optional[Dict[str, List[int]]] = None
    ) -> pd.Series:
        """
        Chuẩn bị labels từ ratings.
        
        Args:
            df: DataFrame
            rating_column: Tên cột rating
            mapping: Dictionary mapping sentiment -> list of ratings
            
        Returns:
            Series chứa sentiment labels
        """
        if mapping is None:
            mapping = {
                'negative': [1, 2],
                'neutral': [3],
                'positive': [4, 5]
            }
        
        def map_rating(rating):
            for sentiment, ratings in mapping.items():
                if rating in ratings:
                    return sentiment
            return 'neutral'
        
        return df[rating_column].apply(map_rating)
    
    def train_baselines(
        self,
        X_train,
        y_train,
        models_config: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Train baseline models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            models_config: List of model configurations
            
        Returns:
            Dictionary chứa trained models và results
        """
        if models_config is None:
            models_config = [
                {
                    'name': 'LogisticRegression',
                    'params': {'max_iter': 1000, 'class_weight': 'balanced', 'random_state': self.random_state}
                },
                {
                    'name': 'NaiveBayes',
                    'params': {'alpha': 1.0}
                }
            ]
        
        print("\n" + "=" * 60)
        print("TRAINING BASELINE MODELS")
        print("=" * 60)
        
        for model_config in models_config:
            model_name = model_config['name']
            model_params = model_config.get('params', {})
            
            print(f"\n[INFO] Training {model_name}...")
            
            # Create model
            if model_name == 'LogisticRegression':
                model = LogisticRegression(**model_params)
            elif model_name == 'NaiveBayes':
                # MultinomialNB requires non-negative features
                # Apply MinMaxScaler to ensure non-negative values
                # Check if data has negative values (works with both dense and sparse)
                X_train_arr = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
                try:
                    X_min = X_train_arr.min()
                except:
                    X_min = 0
                    
                if X_min < 0:
                    scaler_nb = MinMaxScaler()
                    X_train_nb = scaler_nb.fit_transform(X_train_arr)
                else:
                    X_train_nb = X_train_arr
                    scaler_nb = None
                    
                model = MultinomialNB(**model_params)
                model.fit(X_train_nb, y_train)
                
                # Store scaler for later use in prediction
                self.baseline_nb_scaler = scaler_nb
                
                # Store
                self.baselines[model_name] = model
                
                # Evaluate with cross-validation (on scaled data)
                cv_scores = cross_val_score(model, X_train_nb, y_train, cv=self.cv_folds, scoring='f1_macro')
                
                self.results[model_name] = {
                    'cv_f1_macro_mean': float(cv_scores.mean()),
                    'cv_f1_macro_std': float(cv_scores.std()),
                    'cv_scores': cv_scores.tolist()
                }
                
                print(f"[INFO] {model_name} CV F1-macro: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
                continue
            
            # Store
            self.baselines[model_name] = model
            
            # Evaluate with cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring='f1_macro')
            
            self.results[model_name] = {
                'cv_f1_macro_mean': float(cv_scores.mean()),
                'cv_f1_macro_std': float(cv_scores.std()),
                'cv_scores': cv_scores.tolist()
            }
            
            print(f"[INFO] {model_name} CV F1-macro: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return self.results
    
    def train_strong_model(
        self,
        X_train,
        y_train,
        model_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Train strong model (RandomForest).
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_config: Model configuration
            
        Returns:
            Dictionary chứa trained model và results
        """
        if model_config is None:
            model_config = {
                'name': 'RandomForest',
                'params': {
                    'n_estimators': 200,
                    'max_depth': 20,
                    'class_weight': 'balanced',
                    'random_state': self.random_state
                }
            }
        
        print("\n" + "=" * 60)
        print("TRAINING STRONG MODEL")
        print("=" * 60)
        
        model_name = model_config['name']
        model_params = model_config.get('params', {})
        
        print(f"\n[INFO] Training {model_name}...")
        
        if model_name == 'RandomForest':
            model = RandomForestClassifier(**model_params)
        else:
            print(f"[WARNING] Unknown strong model: {model_name}")
            return {}
        
        # Train
        model.fit(X_train, y_train)
        
        # Store
        self.strong_model = model
        self.best_model = model
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring='f1_macro')
        
        self.results[model_name] = {
            'cv_f1_macro_mean': float(cv_scores.mean()),
            'cv_f1_macro_std': float(cv_scores.std()),
            'cv_scores': cv_scores.tolist()
        }
        
        print(f"[INFO] {model_name} CV F1-macro: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return self.results.get(model_name, {})
    
    def evaluate(
        self,
        X_test,
        y_test,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate models on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            model_name: Specific model to evaluate (None = all)
            
        Returns:
            Dictionary chứa evaluation metrics
        """
        if model_name:
            models_to_eval = {model_name: getattr(self, 'baselines', {}).get(model_name) or 
                            (self.strong_model if model_name == 'RandomForest' else None)}
        else:
            models_to_eval = {**self.baselines, 'RandomForest': self.strong_model}
        
        print("\n" + "=" * 60)
        print("MODEL EVALUATION ON TEST SET")
        print("=" * 60)
        
        evaluation_results = {}
        
        for name, model in models_to_eval.items():
            if model is None:
                continue
            
            try:
                y_pred = model.predict(X_test)
            except Exception as e:
                print(f"[WARNING] Error predicting {name}: {str(e)[:80]}")
                continue
            
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'f1_macro': float(f1_score(y_test, y_pred, average='macro')),
                'f1_weighted': float(f1_score(y_test, y_pred, average='weighted')),
                'precision_macro': float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
                'recall_macro': float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
                'kappa': float(cohen_kappa_score(y_test, y_pred)),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            evaluation_results[name] = metrics
            
            print(f"\n{name}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1-macro: {metrics['f1_macro']:.4f}")
            print(f"  Precision-macro: {metrics['precision_macro']:.4f}")
            print(f"  Recall-macro: {metrics['recall_macro']:.4f}")
            print(f"  Cohen's Kappa: {metrics['kappa']:.4f}")
        
        self.results['evaluation'] = evaluation_results
        
        return evaluation_results
    
    def compare_models(self) -> pd.DataFrame:
        """
        So sánh tất cả models.
        
        Returns:
            DataFrame so sánh các models
        """
        if 'evaluation' not in self.results:
            print("[WARNING] No evaluation results. Run evaluate() first.")
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, metrics in self.results['evaluation'].items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'F1-macro': metrics['f1_macro'],
                'F1-weighted': metrics['f1_weighted'],
                'Precision-macro': metrics['precision_macro'],
                'Recall-macro': metrics['recall_macro'],
                'Kappa': metrics['kappa']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1-macro', ascending=False)
        
        return comparison_df
    
    def error_analysis(
        self,
        X_test,
        y_test,
        df: Optional[pd.DataFrame] = None,
        text_column: str = 'review_text'
    ) -> Dict[str, Any]:
        """
        Phân tích lỗi của mô hình tốt nhất.
        Đặc biệt chú ý review đa chủ đề (multi-aspect).
        
        Args:
            X_test: Test features
            y_test: Test labels
            df: Optional DataFrame để lấy original text
            text_column: Tên cột văn bản
            
        Returns:
            Dictionary chứa error analysis
        """
        if self.best_model is None:
            print("[WARNING] No trained model. Train models first.")
            return {}
        
        print("\n" + "=" * 60)
        print("ERROR ANALYSIS")
        print("=" * 60)
        
        # Handle NaiveBayes case (requires non-negative features)
        X_test_ea = X_test
        if hasattr(self, 'baseline_nb_scaler') and self.baseline_nb_scaler is not None:
            # Check if best_model is NaiveBayes by checking the class name
            if self.best_model.__class__.__name__ == 'MultinomialNB':
                X_test_arr = X_test.toarray() if hasattr(X_test, 'toarray') else X_test
                X_test_ea = self.baseline_nb_scaler.transform(X_test_arr)
        
        y_pred = self.best_model.predict(X_test_ea)
        
        # Find misclassifications
        mask = y_pred != y_test
        misclassified_idx = np.where(mask)[0]
        
        error_analysis = {
            'total_errors': len(misclassified_idx),
            'error_rate': float(len(misclassified_idx) / len(y_test)),
            'error_by_true_class': {},
            'error_by_pred_class': {},
            'misclassified_reviews': []
        }
        
        # Analyze by true class
        for true_class in np.unique(y_test):
            true_mask = y_test == true_class
            n_true = true_mask.sum()
            n_errors = (mask & true_mask).sum()
            error_analysis['error_by_true_class'][true_class] = {
                'total': int(n_true),
                'errors': int(n_errors),
                'error_rate': float(n_errors / n_true) if n_true > 0 else 0
            }
        
        # Analyze by predicted class
        for pred_class in np.unique(y_pred):
            pred_mask = y_pred == pred_class
            n_pred = pred_mask.sum()
            n_errors = (mask & pred_mask).sum()
            error_analysis['error_by_pred_class'][pred_class] = {
                'total': int(n_pred),
                'errors': int(n_errors),
                'error_rate': float(n_errors / n_pred) if n_pred > 0 else 0
            }
        
        # Get misclassified reviews (if df provided)
        if df is not None:
            test_start_idx = len(df) - len(y_test)
            
            for idx in misclassified_idx[:20]:  # Limit to 20 examples
                actual_idx = test_start_idx + idx
                if actual_idx < len(df):
                    review_info = {
                        'review': df.iloc[actual_idx][text_column][:200] if text_column in df.columns else "N/A",
                        'true_label': str(y_test[idx]),
                        'predicted_label': str(y_pred[idx]),
                        'rating': df.iloc[actual_idx]['rating'] if 'rating' in df.columns else 'N/A'
                    }
                    
                    # Check for multi-aspect indicators
                    review_text = review_info['review'].lower()
                    n_aspects = sum([
                        any(kw in review_text for kw in ['room', 'bed', 'bathroom']),
                        any(kw in review_text for kw in ['service', 'staff', 'friendly']),
                        any(kw in review_text for kw in ['location', 'central', 'near']),
                        any(kw in review_text for kw in ['food', 'breakfast', 'restaurant']),
                        any(kw in review_text for kw in ['price', 'expensive', 'value'])
                    ])
                    review_info['n_aspects_detected'] = n_aspects
                    review_info['is_multi_aspect'] = n_aspects >= 2
                    
                    error_analysis['misclassified_reviews'].append(review_info)
        
        # Print summary
        print(f"\nTotal misclassifications: {error_analysis['total_errors']} ({error_analysis['error_rate']*100:.1f}%)")
        
        print("\nError rate by true class:")
        for cls, stats in error_analysis['error_by_true_class'].items():
            print(f"  {cls}: {stats['error_rate']*100:.1f}% ({stats['errors']}/{stats['total']})")
        
        # Multi-aspect analysis
        multi_aspect_errors = [r for r in error_analysis['misclassified_reviews'] if r.get('is_multi_aspect', False)]
        print(f"\nMulti-aspect reviews in errors: {len(multi_aspect_errors)}/{len(error_analysis['misclassified_reviews'])}")
        
        self.error_analysis = error_analysis
        
        return error_analysis
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Lấy feature importance từ Random Forest.
        
        Args:
            top_n: Số features quan trọng nhất
            
        Returns:
            Dictionary chứa top features
        """
        if self.strong_model is None or not hasattr(self.strong_model, 'feature_importances_'):
            return {}
        
        importances = self.strong_model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        return {
            'top_features': [(f'feature_{i}', importances[i]) for i in indices],
            'all_importances': importances.tolist()
        }
    
    def save_model(self, path: str) -> None:
        """Lưu trained models."""
        import pickle
        from pathlib import Path
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self.best_model:
            joblib.dump(self.best_model, path / 'best_model.pkl')
        
        if self.baselines:
            joblib.dump(self.baselines, path / 'baseline_models.pkl')
        
        if self.label_encoder:
            joblib.dump(self.label_encoder, path / 'label_encoder.pkl')
        
        print(f"[INFO] Models saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load trained models."""
        from pathlib import Path
        
        path = Path(path)
        
        if (path / 'best_model.pkl').exists():
            self.best_model = joblib.load(path / 'best_model.pkl')
        
        if (path / 'baseline_models.pkl').exists():
            self.baselines = joblib.load(path / 'baseline_models.pkl')
        
        if (path / 'label_encoder.pkl').exists():
            self.label_encoder = joblib.load(path / 'label_encoder.pkl')
        
        print(f"[INFO] Models loaded from {path}")


if __name__ == "__main__":
    print("Testing SentimentClassifier...")
    
    # Sample data
    np.random.seed(42)
    n_samples = 500
    
    X_train = np.random.rand(n_samples, 100)
    X_test = np.random.rand(n_samples // 5, 100)
    
    y_train = np.random.choice(['positive', 'neutral', 'negative'], n_samples, p=[0.5, 0.25, 0.25])
    y_test = np.random.choice(['positive', 'neutral', 'negative'], n_samples // 5, p=[0.5, 0.25, 0.25])
    
    # Train
    classifier = SentimentClassifier()
    
    # Train baselines
    classifier.train_baselines(X_train, y_train)
    
    # Train strong model
    classifier.train_strong_model(X_train, y_train)
    
    # Evaluate
    results = classifier.evaluate(X_test, y_test)
    
    # Compare
    comparison = classifier.compare_models()
    print("\nModel Comparison:")
    print(comparison)
