# ==============================================================================
# REGRESSION MODULE
# ==============================================================================
"""
Module Hồi quy Rating (Regression):
- Baseline: Ridge / Linear Regression
- Strong model: SVR / XGBoost
- Metric: MAE, RMSE
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARNING] XGBoost not available. Install with: pip install xgboost")


class RatingPredictor:
    """
    Class dự đoán rating từ nội dung review.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize RatingPredictor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.reg_config = self.config.get('regression', {})
        
        # Parameters
        self.cv_folds = self.reg_config.get('evaluation', {}).get('cv_folds', 5)
        self.test_size = self.reg_config.get('evaluation', {}).get('test_size', 0.2)
        self.random_state = 42
        
        # Models
        self.baselines = {}
        self.strong_model = None
        self.best_model = None
        
        # Results
        self.results = {}
        
    def evaluate_regression(
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
            Dictionary chứa metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R2': float(r2)
        }
    
    def train_baselines(
        self,
        X_train,
        y_train,
        models_config: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Train baseline regression models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            models_config: List of model configurations
            
        Returns:
            Dictionary chứa trained models và results
        """
        if models_config is None:
            models_config = [
                {
                    'name': 'Ridge',
                    'params': {'alpha': 1.0, 'random_state': self.random_state}
                },
                {
                    'name': 'LinearRegression',
                    'params': {}
                }
            ]
        
        print("\n" + "=" * 60)
        print("TRAINING REGRESSION BASELINES")
        print("=" * 60)
        
        for model_config in models_config:
            model_name = model_config['name']
            model_params = model_config.get('params', {})
            
            print(f"\n[INFO] Training {model_name}...")
            
            # Create model
            if model_name == 'Ridge':
                model = Ridge(**model_params)
            elif model_name == 'LinearRegression':
                model = LinearRegression(**model_params)
            elif model_name == 'Lasso':
                model = Lasso(**model_params)
            else:
                print(f"[WARNING] Unknown baseline model: {model_name}")
                continue
            
            # Train
            model.fit(X_train, y_train)
            
            # Store
            self.baselines[model_name] = model
            
            # Cross-validation
            cv_rmse = -cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring='neg_root_mean_squared_error')
            cv_r2 = cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring='r2')
            
            self.results[model_name] = {
                'cv_rmse_mean': float(cv_rmse.mean()),
                'cv_rmse_std': float(cv_rmse.std()),
                'cv_r2_mean': float(cv_r2.mean()),
                'cv_r2_std': float(cv_r2.std())
            }
            
            print(f"[INFO] {model_name} CV RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std():.4f})")
            print(f"[INFO] {model_name} CV R2: {cv_r2.mean():.4f} (+/- {cv_r2.std():.4f})")
        
        return self.results
    
    def train_strong_model(
        self,
        X_train,
        y_train,
        model_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Train strong regression model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            model_config: Model configuration
            
        Returns:
            Dictionary chứa trained model và results
        """
        if model_config is None:
            if XGBOOST_AVAILABLE:
                model_config = {
                    'name': 'XGBoost',
                    'params': {
                        'n_estimators': 100,
                        'max_depth': 6,
                        'learning_rate': 0.1,
                        'random_state': self.random_state
                    }
                }
            else:
                model_config = {
                    'name': 'GradientBoosting',
                    'params': {
                        'n_estimators': 100,
                        'max_depth': 6,
                        'learning_rate': 0.1,
                        'random_state': self.random_state
                    }
                }
        
        print("\n" + "=" * 60)
        print("TRAINING STRONG REGRESSION MODEL")
        print("=" * 60)
        
        model_name = model_config['name']
        model_params = model_config.get('params', {})
        
        print(f"\n[INFO] Training {model_name}...")
        
        # Create model
        if model_name == 'XGBoost' and XGBOOST_AVAILABLE:
            model = xgb.XGBRegressor(**model_params)
        elif model_name == 'SVR':
            model = SVR(**model_params)
        elif model_name == 'GradientBoosting':
            model = GradientBoostingRegressor(**model_params)
        else:
            print(f"[WARNING] Unknown strong model: {model_name}")
            return {}
        
        # Train
        model.fit(X_train, y_train)
        
        # Store
        self.strong_model = model
        self.best_model = model
        
        # Cross-validation
        cv_rmse = -cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring='neg_root_mean_squared_error')
        cv_r2 = cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring='r2')
        
        self.results[model_name] = {
            'cv_rmse_mean': float(cv_rmse.mean()),
            'cv_rmse_std': float(cv_rmse.std()),
            'cv_r2_mean': float(cv_r2.mean()),
            'cv_r2_std': float(cv_r2.std())
        }
        
        print(f"[INFO] {model_name} CV RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std():.4f})")
        print(f"[INFO] {model_name} CV R2: {cv_r2.mean():.4f} (+/- {cv_r2.std():.4f})")
        
        return self.results.get(model_name, {})
    
    def evaluate(
        self,
        X_test,
        y_test,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate regression models on test set.
        
        Args:
            X_test: Test features
            y_test: Test targets
            model_name: Specific model to evaluate (None = all)
            
        Returns:
            Dictionary chứa evaluation metrics
        """
        if model_name:
            models_to_eval = {model_name: getattr(self, 'baselines', {}).get(model_name) or 
                            (self.strong_model if model_name == 'XGBoost' else None)}
        else:
            models_to_eval = {**self.baselines, **({'XGBoost': self.strong_model} if XGBOOST_AVAILABLE else {})}
        
        print("\n" + "=" * 60)
        print("REGRESSION EVALUATION ON TEST SET")
        print("=" * 60)
        
        evaluation_results = {}
        
        for name, model in models_to_eval.items():
            if model is None:
                continue
            
            y_pred = model.predict(X_test)
            
            # Clip predictions to valid range
            y_pred = np.clip(y_pred, 1, 5)
            
            metrics = self.evaluate_regression(y_test, y_pred)
            evaluation_results[name] = metrics
            
            print(f"\n{name}:")
            print(f"  MAE: {metrics['MAE']:.4f}")
            print(f"  RMSE: {metrics['RMSE']:.4f}")
            print(f"  R2: {metrics['R2']:.4f}")
        
        self.results['evaluation'] = evaluation_results
        
        return evaluation_results
    
    def compare_models(self) -> pd.DataFrame:
        """
        So sánh tất cả regression models.
        
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
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'R2': metrics['R2']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('RMSE', ascending=True)
        
        return comparison_df
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Lấy feature importance.
        
        Args:
            top_n: Số features quan trọng nhất
            
        Returns:
            Dictionary chứa top features
        """
        if self.strong_model is None:
            return {}
        
        # XGBoost
        if hasattr(self.strong_model, 'feature_importances_'):
            importances = self.strong_model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            return {
                'top_features': [(f'feature_{i}', importances[i]) for i in indices],
                'all_importances': importances.tolist()
            }
        
        # Linear models
        elif hasattr(self.strong_model, 'coef_'):
            importances = np.abs(self.strong_model.coef_)
            indices = np.argsort(importances)[::-1][:top_n]
            
            return {
                'top_features': [(f'feature_{i}', importances[i]) for i in indices],
                'all_importances': importances.tolist()
            }
        
        return {}
    
    def save_model(self, path: str) -> None:
        """Lưu trained models."""
        from pathlib import Path
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self.best_model:
            joblib.dump(self.best_model, path / 'regression_model.pkl')
        
        if self.baselines:
            joblib.dump(self.baselines, path / 'baseline_models.pkl')
        
        print(f"[INFO] Models saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load trained models."""
        from pathlib import Path
        
        path = Path(path)
        
        if (path / 'regression_model.pkl').exists():
            self.best_model = joblib.load(path / 'regression_model.pkl')
        
        if (path / 'baseline_models.pkl').exists():
            self.baselines = joblib.load(path / 'baseline_models.pkl')
        
        print(f"[INFO] Models loaded from {path}")


if __name__ == "__main__":
    print("Testing RatingPredictor...")
    
    # Sample data
    np.random.seed(42)
    n_samples = 500
    n_features = 100
    
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(1, 6, n_samples).astype(float)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    predictor = RatingPredictor()
    
    # Train baselines
    predictor.train_baselines(X_train, y_train)
    
    # Train strong model
    predictor.train_strong_model(X_train, y_train)
    
    # Evaluate
    results = predictor.evaluate(X_test, y_test)
    
    # Compare
    comparison = predictor.compare_models()
    print("\nModel Comparison:")
    print(comparison)
