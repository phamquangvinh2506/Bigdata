# ==============================================================================
# FEATURE BUILDER MODULE
# ==============================================================================
"""
Module trích xuất và xây dựng đặc trưng từ dữ liệu văn bản:
- TF-IDF Vectorization
- Statistical features
- Aspect-based features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore')


class FeatureBuilder:
    """
    Class xây dựng đặc trưng cho mô hình.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize FeatureBuilder.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.features_config = self.config.get('features', {})
        self.tfidf_config = self.features_config.get('tfidf', {})
        self.aspect_keywords = self.features_config.get('aspect_keywords', {})
        
        # Components
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.scaler = None
        self.label_encoder = None
        self.svd = None
        
        # Feature matrices
        self.tfidf_matrix = None
        self.statistical_features = None
        self.aspect_features = None
        
    def build_tfidf_features(
        self,
        texts: List[str],
        fit: bool = True,
        max_features: Optional[int] = None,
        min_df: Optional[int] = None,
        max_df: Optional[float] = None,
        ngram_range: Tuple[int, int] = (1, 2)
    ) -> np.ndarray:
        """
        Xây dựng TF-IDF features.
        
        Args:
            texts: List of text documents
            fit: Fit vectorizer (True for training)
            max_features: Maximum number of features
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            ngram_range: N-gram range
            
        Returns:
            TF-IDF feature matrix
        """
        # Get parameters
        max_features = max_features or self.tfidf_config.get('max_features', 5000)
        min_df = min_df or self.tfidf_config.get('min_df', 2)
        max_df = max_df or self.tfidf_config.get('max_df', 0.95)
        ngram_range = tuple(ngram_range) or tuple(self.tfidf_config.get('ngram_range', (1, 2)))
        
        if fit or self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
                ngram_range=ngram_range,
                sublinear_tf=True,
                stop_words='english'
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            print(f"[INFO] TF-IDF fitted with {max_features} features, shape: {self.tfidf_matrix.shape}")
        else:
            self.tfidf_matrix = self.tfidf_vectorizer.transform(texts)
            
        return self.tfidf_matrix
    
    def build_statistical_features(
        self,
        df: pd.DataFrame,
        text_column: str = 'review_text'
    ) -> np.ndarray:
        """
        Xây dựng các đặc trưng thống kê.
        
        Args:
            df: DataFrame chứa dữ liệu
            text_column: Tên cột văn bản
            
        Returns:
            Statistical feature matrix
        """
        texts = df[text_column].astype(str)
        
        features = pd.DataFrame()
        
        # Basic length features
        features['review_length'] = texts.str.len()
        features['word_count'] = texts.str.split().str.len()
        features['avg_word_length'] = features['review_length'] / (features['word_count'] + 1)
        features['char_per_word'] = features['review_length'] / (features['word_count'] + 1)
        
        # Sentence features
        features['sentence_count'] = texts.str.count(r'[.!?]') + 1
        features['avg_words_per_sentence'] = features['word_count'] / (features['sentence_count'] + 1)
        
        # Punctuation features
        features['exclamation_count'] = texts.str.count('!')
        features['question_count'] = texts.str.count(r'\?')
        features['comma_count'] = texts.str.count(',')
        features['period_count'] = texts.str.count(r'\.')
        
        # Capitalization features
        features['uppercase_count'] = texts.str.count(r'[A-Z]')
        features['uppercase_ratio'] = features['uppercase_count'] / (features['review_length'] + 1)
        features['capital_words'] = texts.str.count(r'\b[A-Z]{2,}\b')
        
        # Special patterns
        features['digit_count'] = texts.str.count(r'\d')
        features['digit_ratio'] = features['digit_count'] / (features['review_length'] + 1)
        features['special_char_count'] = texts.str.count(r'[^a-zA-Z0-9\s]')
        
        # Word-level features
        features['unique_words'] = texts.apply(lambda x: len(set(x.lower().split())))
        features['word_diversity'] = features['unique_words'] / (features['word_count'] + 1)
        
        # Long words ratio
        features['long_words'] = texts.apply(lambda x: sum(1 for w in x.split() if len(w) > 6))
        features['long_words_ratio'] = features['long_words'] / (features['word_count'] + 1)
        
        self.statistical_features = features.values
        
        return self.statistical_features
    
    def build_aspect_features(
        self,
        df: pd.DataFrame,
        text_column: str = 'review_text',
        keywords_dict: Optional[Dict[str, List[str]]] = None
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Xây dựng aspect-based features dựa trên keywords.
        
        Args:
            df: DataFrame
            text_column: Tên cột văn bản
            keywords_dict: Dictionary of aspect keywords
            
        Returns:
            Tuple (aspect feature matrix, aspect counts)
        """
        if keywords_dict is None:
            keywords_dict = {
                'room': ['room', 'clean', 'dirty', 'bed', 'bathroom', 'towel', 'shower', 'bath', 'sleep', 'mattress', 'pillow'],
                'service': ['service', 'staff', 'friendly', 'helpful', 'professional', 'attentive', 'manager', 'reception', 'concierge'],
                'location': ['location', 'central', 'near', 'close', 'walking', 'distance', 'convenient', 'downtown', 'beach', 'view'],
                'food': ['food', 'breakfast', 'dinner', 'restaurant', 'buffet', 'meal', 'delicious', 'coffee', 'dish'],
                'price': ['price', 'expensive', 'cheap', 'value', 'worth', 'money', 'affordable', 'cost', 'budget', 'overpriced'],
                'amenities': ['wifi', 'pool', 'parking', 'gym', 'spa', 'facility', 'amenities', 'comfort', 'ac', 'internet'],
                'cleanliness': ['clean', 'hygiene', 'tidy', 'spotless', 'dust', 'dirty', 'maintenance', 'sanitary'],
                'noise': ['noise', 'quiet', 'noisy', 'loud', 'peaceful', 'sound', 'disturb', 'traffic']
            }
        
        texts = df[text_column].astype(str).str.lower()
        features = pd.DataFrame()
        aspect_counts = {}
        
        for aspect, keywords in keywords_dict.items():
            pattern = '|'.join([r'\b' + kw + r'\b' for kw in keywords])
            features[f'{aspect}_mentioned'] = texts.str.contains(pattern, regex=True, na=False).astype(int)
            features[f'{aspect}_count'] = texts.str.count(pattern)
            features[f'{aspect}_ratio'] = features[f'{aspect}_count'] / (texts.str.split().str.len() + 1)
            aspect_counts[aspect] = features[f'{aspect}_mentioned'].sum()
        
        # Overall
        features['total_aspects'] = features[[f'{a}_mentioned' for a in keywords_dict.keys()]].sum(axis=1)
        
        # Sentiment indicators in text
        positive_words = ['excellent', 'amazing', 'wonderful', 'perfect', 'great', 'best', 'fantastic', 'outstanding', 'superb']
        negative_words = ['terrible', 'awful', 'horrible', 'bad', 'worst', 'disgusting', 'poor', 'disappointed', 'dirty']
        
        pos_pattern = '|'.join(positive_words)
        neg_pattern = '|'.join(negative_words)
        
        features['positive_words_count'] = texts.str.count(pos_pattern)
        features['negative_words_count'] = texts.str.count(neg_pattern)
        features['sentiment_ratio'] = features['positive_words_count'] - features['negative_words_count']
        
        self.aspect_features = features.values
        self.feature_names = list(features.columns)
        
        return self.aspect_features, aspect_counts
    
    def build_all_features(
        self,
        df: pd.DataFrame,
        text_column: str = 'review_text'
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Xây dựng tất cả features.
        
        Args:
            df: DataFrame
            text_column: Tên cột văn bản
            
        Returns:
            Tuple (combined feature matrix, feature info)
        """
        print("[INFO] Building all features...")
        
        # TF-IDF features
        cleaned_texts = df.get('cleaned_text', df[text_column]).astype(str).tolist()
        tfidf_matrix = self.build_tfidf_features(cleaned_texts, fit=True)
        
        # Statistical features
        stat_features = self.build_statistical_features(df, text_column)
        
        # Scale statistical features
        self.scaler = StandardScaler()
        stat_features_scaled = self.scaler.fit_transform(stat_features)
        
        # Aspect features
        aspect_features, aspect_counts = self.build_aspect_features(df, text_column)
        
        # Combine all features
        X = np.hstack([tfidf_matrix.toarray(), stat_features_scaled, aspect_features])
        
        feature_info = {
            'tfidf_shape': tfidf_matrix.shape,
            'stat_features_shape': stat_features.shape,
            'aspect_features_shape': aspect_features.shape,
            'total_features': X.shape[1],
            'aspect_counts': aspect_counts,
            'tfidf_vocabulary_size': len(self.tfidf_vectorizer.vocabulary_) if self.tfidf_vectorizer else 0,
            'stat_feature_names': ['review_length', 'word_count', 'avg_word_length', 'char_per_word',
                                   'sentence_count', 'avg_words_per_sentence', 'exclamation_count',
                                   'question_count', 'comma_count', 'period_count', 'uppercase_count',
                                   'uppercase_ratio', 'capital_words', 'digit_count', 'digit_ratio',
                                   'special_char_count', 'unique_words', 'word_diversity', 'long_words',
                                   'long_words_ratio']
        }
        
        print(f"[INFO] Combined feature matrix shape: {X.shape}")
        
        return X, feature_info
    
    def get_top_tfidf_terms(self, n_terms: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Lấy top TF-IDF terms cho mỗi class/rating.
        
        Args:
            n_terms: Số terms cần lấy
            
        Returns:
            Dictionary mapping rating/sentiment to top terms
        """
        if self.tfidf_vectorizer is None:
            return {}
        
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        top_terms = {}
        
        # Get mean TF-IDF scores
        if self.tfidf_matrix is not None:
            mean_tfidf = np.array(self.tfidf_matrix.mean(axis=0)).flatten()
            sorted_indices = np.argsort(mean_tfidf)[::-1]
            
            top_terms['overall'] = [
                (feature_names[i], mean_tfidf[i]) 
                for i in sorted_indices[:n_terms]
            ]
        
        return top_terms
    
    def save_vectorizers(self, path: str) -> None:
        """Lưu các vectorizers."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self.tfidf_vectorizer:
            joblib.dump(self.tfidf_vectorizer, path / 'tfidf_vectorizer.pkl')
        if self.scaler:
            joblib.dump(self.scaler, path / 'scaler.pkl')
        if self.svd:
            joblib.dump(self.svd, path / 'svd.pkl')
            
        print(f"[INFO] Vectorizers saved to {path}")
    
    def load_vectorizers(self, path: str) -> None:
        """Load các vectorizers."""
        path = Path(path)
        
        if (path / 'tfidf_vectorizer.pkl').exists():
            self.tfidf_vectorizer = joblib.load(path / 'tfidf_vectorizer.pkl')
        if (path / 'scaler.pkl').exists():
            self.scaler = joblib.load(path / 'scaler.pkl')
        if (path / 'svd.pkl').exists():
            self.svd = joblib.load(path / 'svd.pkl')
            
        print(f"[INFO] Vectorizers loaded from {path}")


class AspectExtractor:
    """Extract aspects from review text."""
    
    def __init__(self, keywords_dict: Optional[Dict[str, List[str]]] = None):
        self.keywords_dict = keywords_dict or {
            'room': ['room', 'bed', 'bathroom', 'towel', 'shower', 'bath', 'sleep'],
            'service': ['service', 'staff', 'friendly', 'helpful', 'professional'],
            'location': ['location', 'central', 'near', 'close', 'walking', 'distance'],
            'food': ['food', 'breakfast', 'dinner', 'restaurant', 'buffet', 'meal'],
            'price': ['price', 'expensive', 'cheap', 'value', 'worth', 'money'],
            'amenities': ['wifi', 'pool', 'parking', 'gym', 'spa', 'facility'],
            'cleanliness': ['clean', 'hygiene', 'tidy', 'spotless', 'dust'],
            'noise': ['noise', 'quiet', 'noisy', 'loud', 'peaceful']
        }
    
    def extract_aspects(self, text: str) -> Dict[str, List[str]]:
        """Extract aspects and their related keywords from text."""
        text_lower = text.lower()
        results = {}
        
        for aspect, keywords in self.keywords_dict.items():
            found_keywords = [kw for kw in keywords if kw in text_lower]
            if found_keywords:
                results[aspect] = found_keywords
        
        return results
    
    def extract_aspects_batch(self, texts: List[str]) -> List[Dict[str, List[str]]]:
        """Extract aspects for multiple texts."""
        return [self.extract_aspects(text) for text in texts]


if __name__ == "__main__":
    print("Testing FeatureBuilder...")
    
    # Sample data
    sample_data = {
        'review_text': [
            "The hotel room was clean and the bed was comfortable. Great service!",
            "Terrible service, dirty bathroom, awful breakfast. Not recommended.",
            "The location is perfect, staff is friendly, but the room is small.",
            "Amazing hotel with great amenities and excellent breakfast buffet.",
            "Noisy location, poor wifi, but the staff was helpful."
        ] * 20,
        'cleaned_text': [
            "hotel room clean bed comfort great servic",
            "terribl servic dirt bathroom awfull breakfast not recommend",
            "the location perfect staff friendli but the room small",
            "amaz hotel great ameniti excel breakfast buffet",
            "noisi location poor wifi but the staff helpfull"
        ] * 20
    }
    
    df = pd.DataFrame(sample_data)
    
    # Build features
    builder = FeatureBuilder()
    X, info = builder.build_all_features(df)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"\nFeature info: {info}")
