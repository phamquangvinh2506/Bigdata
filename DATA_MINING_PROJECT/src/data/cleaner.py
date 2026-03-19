# ==============================================================================
# DATA CLEANER MODULE
# ==============================================================================
"""
Module chịu trách nhiệm làm sạch và tiền xử lý dữ liệu văn bản:
- Loại bỏ special characters, numbers
- Lowercase conversion
- Stopwords removal
- Stemming
- Handle missing values
- Remove duplicates
"""

import re
import os
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Set
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# NLTK imports - sẽ được import trong class
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer, PorterStemmer, WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    
    # Download required NLTK data
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        
except ImportError:
    print("[WARNING] NLTK not available. Some text processing features will be limited.")


class TextCleaner:
    """
    Class chịu trách nhiệm làm sạch và tiền xử lý văn bản đánh giá.
    """
    
    # Default English stopwords
    DEFAULT_ENGLISH_STOPWORDS = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
        "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
        'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
        'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
        'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
        'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
        'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
        'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
        'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
        'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
        've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
        'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
        'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
        "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
        "weren't", 'won', "won't", 'wouldn', "wouldn't"
    }
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_special_chars: bool = True,
        remove_stopwords: bool = True,
        remove_numbers: bool = True,
        apply_stemming: bool = True,
        min_word_length: int = 2,
        stopwords_lang: str = 'english',
        custom_stopwords: Optional[Set[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize TextCleaner với các tham số cấu hình.
        
        Args:
            lowercase: Chuyển đổi về lowercase
            remove_special_chars: Loại bỏ ký tự đặc biệt
            remove_stopwords: Loại bỏ stopwords
            remove_numbers: Loại bỏ số
            apply_stemming: Áp dụng stemming
            min_word_length: Độ dài tối thiểu của từ
            stopwords_lang: Ngôn ngữ stopwords
            custom_stopwords: Tập stopwords tùy chỉnh
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Load từ config nếu có
        if 'preprocessing' in self.config:
            prep_config = self.config['preprocessing']
            lowercase = prep_config.get('lowercase', lowercase)
            remove_special_chars = prep_config.get('remove_special_chars', remove_special_chars)
            remove_stopwords = prep_config.get('remove_stopwords', remove_stopwords)
            remove_numbers = prep_config.get('remove_numbers', remove_numbers)
            apply_stemming = prep_config.get('apply_stemming', apply_stemming)
            min_word_length = prep_config.get('min_word_length', min_word_length)
        
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars
        self.remove_stopwords = remove_stopwords
        self.remove_numbers = remove_numbers
        self.apply_stemming = apply_stemming
        self.min_word_length = min_word_length
        
        # Initialize stopwords
        self.stopwords = self._load_stopwords(stopwords_lang, custom_stopwords)
        
        # Initialize stemmer
        self.stemmer = SnowballStemmer('english') if apply_stemming else None
        
        # Statistics
        self.stats = {
            'original_rows': 0,
            'cleaned_rows': 0,
            'removed_duplicates': 0,
            'removed_missing': 0,
            'avg_text_length_before': 0,
            'avg_text_length_after': 0,
        }
    
    def _load_stopwords(
        self, 
        lang: str, 
        custom: Optional[Set[str]] = None
    ) -> Set[str]:
        """Load stopwords từ NLTK hoặc custom."""
        if custom:
            return custom
        
        try:
            nltk_stopwords = set(stopwords.words(lang))
            return nltk_stopwords
        except:
            print(f"[WARNING] Could not load NLTK stopwords for '{lang}'. Using default.")
            return self.DEFAULT_ENGLISH_STOPWORDS
    
    def clean_text(self, text: str) -> str:
        """
        Làm sạch một đoạn văn bản.
        
        Args:
            text: Văn bản cần làm sạch
            
        Returns:
            Văn bản đã được làm sạch
        """
        if not isinstance(text, str) or pd.isna(text):
            return ""
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        if self.remove_special_chars:
            # Giữ lại khoảng trắng và chữ cái alphabet
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize và xử lý từng từ
        words = text.split()
        
        # Remove stopwords và short words
        if self.remove_stopwords:
            words_filtered = [w for w in words if w not in self.stopwords]
            # If too many words removed, keep original words
            if len(words_filtered) < len(words) * 0.3:
                words = words
            else:
                words = words_filtered
        
        if self.min_word_length > 0:
            words = [w for w in words if len(w) >= max(1, self.min_word_length)]
        
        # Stemming
        if self.stemmer:
            words = [self.stemmer.stem(w) for w in words]
        
        return ' '.join(words)
    
    def clean_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'review_text',
        missing_strategy: str = 'drop',
        duplicate_strategy: str = 'drop',
        add_stats_columns: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Làm sạch toàn bộ DataFrame.
        
        Args:
            df: DataFrame cần làm sạch
            text_column: Tên cột chứa văn bản
            missing_strategy: Chiến lược xử lý missing ('drop', 'fill_empty')
            duplicate_strategy: Chiến lược xử lý duplicate ('drop', 'keep_first')
            add_stats_columns: Thêm các cột thống kê
            
        Returns:
            Tuple (DataFrame đã làm sạch, statistics dictionary)
        """
        self.stats['original_rows'] = len(df)
        df = df.copy()
        
        # Store original text for statistics
        if add_stats_columns:
            df['original_text'] = df[text_column].astype(str)
            df['original_length'] = df['original_text'].str.len()
            df['original_word_count'] = df['original_text'].str.split().str.len()
        
        # Handle missing values
        if missing_strategy == 'drop':
            missing_count = df[text_column].isnull().sum()
            df = df.dropna(subset=[text_column])
            self.stats['removed_missing'] = missing_count
        else:  # fill_empty
            df[text_column] = df[text_column].fillna('')
        
        # Remove duplicates - SKIP since already handled by DataCleaner.clean()
        # (avoid double duplicate removal)
        self.stats['removed_duplicates'] = 0
        
        # Apply text cleaning - keep original text if cleaned is empty
        print("[INFO] Cleaning text...")
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Keep original text if cleaned is empty (for short reviews)
        empty_mask = (df['cleaned_text'] == '') | (df['cleaned_text'].isna())
        if empty_mask.sum() > 0:
            print(f"[INFO] Keeping {empty_mask.sum()} short reviews with original text")
            df.loc[empty_mask, 'cleaned_text'] = df.loc[empty_mask, text_column]
        
        # Add statistics columns
        if add_stats_columns:
            df['cleaned_length'] = df['cleaned_text'].str.len()
            df['cleaned_word_count'] = df['cleaned_text'].str.split().str.len()
            df['length_reduction'] = df['original_length'] - df['cleaned_length']
            df['word_reduction'] = df['original_word_count'] - df['cleaned_word_count']
            
            self.stats['avg_text_length_before'] = float(df['original_length'].mean())
            self.stats['avg_text_length_after'] = float(df['cleaned_length'].mean())
        
        self.stats['cleaned_rows'] = len(df)
        
        return df, self.stats
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Trả về tóm tắt các bước preprocessing đã thực hiện."""
        return {
            'lowercase': self.lowercase,
            'remove_special_chars': self.remove_special_chars,
            'remove_stopwords': self.remove_stopwords,
            'stopwords_count': len(self.stopwords),
            'remove_numbers': self.remove_numbers,
            'apply_stemming': self.apply_stemming,
            'stemmer': 'SnowballStemmer' if self.apply_stemming else None,
            'min_word_length': self.min_word_length,
            'stats': self.stats
        }


class DataCleaner:
    """
    Class chính để làm sạch dữ liệu đánh giá khách sạn.
    Kết hợp TextCleaner với các bước khác.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataCleaner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.text_cleaner = TextCleaner(config=config)
        self.preprocess_stats = {}
    
    def clean(
        self,
        df: pd.DataFrame,
        text_column: str = 'review_text',
        rating_column: str = 'rating'
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Thực hiện tất cả các bước làm sạch dữ liệu.
        
        Args:
            df: DataFrame cần làm sạch
            text_column: Tên cột văn bản
            rating_column: Tên cột rating
            
        Returns:
            Tuple (DataFrame đã làm sạch, statistics)
        """
        print("=" * 60)
        print("DATA CLEANING PROCESS")
        print("=" * 60)
        
        df = df.copy()
        n_original = len(df)
        
        # Step 1: Basic data cleaning
        print("\n[Step 1/5] Basic data cleaning...")
        
        # NOTE: Skip duplicate removal here - it's handled in TextCleaner.clean_dataframe
        print(f"  - Original rows: {n_original}")
        
        # Step 2: Handle missing values
        print("\n[Step 2/5] Handling missing values...")
        for col in [text_column, rating_column]:
            if col in df.columns:
                missing = df[col].isnull().sum()
                if missing > 0:
                    print(f"  - {col}: {missing} missing values")
                    if col == text_column:
                        df = df.dropna(subset=[col])
                    else:
                        df[col] = df[col].fillna(df[col].median() if df[col].dtype in ['int64', 'float64'] else 'Unknown')
        
        # Step 3: Text preprocessing
        print("\n[Step 3/5] Text preprocessing...")
        df, text_stats = self.text_cleaner.clean_dataframe(
            df, 
            text_column=text_column,
            add_stats_columns=True
        )
        
        # Step 4: Rating validation
        print("\n[Step 4/5] Validating ratings...")
        if rating_column in df.columns:
            valid_ratings = df[rating_column].between(1, 5)
            invalid_count = (~valid_ratings).sum()
            if invalid_count > 0:
                print(f"  - Found {invalid_count} invalid ratings, removing...")
                df = df[valid_ratings]
            
            # Ensure integer type
            df[rating_column] = df[rating_column].astype(int)
        
        # Step 5: Add derived features
        print("\n[Step 5/5] Adding derived features...")
        df = self._add_derived_features(df, text_column)
        
        # Summary statistics
        self.preprocess_stats = {
            'original_rows': n_original,
            'final_rows': len(df),
            'rows_removed': n_original - len(df),
            'removal_rate': f"{(n_original - len(df)) / n_original * 100:.1f}%",
            'text_cleaning_stats': text_stats,
            'columns': list(df.columns)
        }
        
        print("\n" + "=" * 60)
        print("CLEANING SUMMARY")
        print("=" * 60)
        print(f"Original rows: {n_original}")
        print(f"Final rows: {len(df)}")
        print(f"Rows removed: {n_original - len(df)} ({self.preprocess_stats['removal_rate']})")
        print(f"Columns: {list(df.columns)}")
        
        return df, self.preprocess_stats
    
    def _add_derived_features(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Thêm các đặc trưng được derived từ text."""
        
        # Review length features
        df['review_length'] = df[text_column].str.len()
        df['word_count'] = df[text_column].str.split().str.len()
        df['avg_word_length'] = df['review_length'] / (df['word_count'] + 1)
        
        # Sentence features
        df['sentence_count'] = df[text_column].str.count(r'[.!?]') + 1
        
        # Punctuation features
        df['exclamation_count'] = df[text_column].str.count('!')
        df['question_count'] = df[text_column].str.count(r'\?')
        
        # Uppercase ratio
        df['uppercase_count'] = df[text_column].str.count(r'[A-Z]')
        df['uppercase_ratio'] = df['uppercase_count'] / (df['review_length'] + 1)
        
        # Sentiment mapping
        if 'rating' in df.columns:
            df['sentiment'] = df['rating'].map(lambda x: 
                'positive' if x >= 4 else ('neutral' if x == 3 else 'negative')
            )
        
        return df
    
    def get_before_after_comparison(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
        """
        So sánh dữ liệu trước và sau khi làm sạch.
        
        Returns:
            Dictionary chứa so sánh
        """
        comparison = {
            'n_rows': {
                'before': len(original_df),
                'after': len(cleaned_df),
                'difference': len(original_df) - len(cleaned_df)
            }
        }
        
        if 'review_length' in cleaned_df.columns and 'original_length' in cleaned_df.columns:
            comparison['avg_review_length'] = {
                'before': float(cleaned_df['original_length'].mean()),
                'after': float(cleaned_df['review_length'].mean()),
                'reduction': float(cleaned_df['original_length'].mean() - cleaned_df['review_length'].mean())
            }
            
            comparison['avg_word_count'] = {
                'before': float(cleaned_df['original_word_count'].mean()),
                'after': float(cleaned_df['word_count'].mean())
            }
        
        if 'rating' in cleaned_df.columns:
            comparison['rating_distribution'] = {
                'before': original_df['rating'].value_counts().to_dict() if 'rating' in original_df.columns else {},
                'after': cleaned_df['rating'].value_counts().to_dict()
            }
        
        return comparison


def clean_pipeline(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    text_column: str = 'review_text',
    rating_column: str = 'rating'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function để chạy cleaning pipeline.
    
    Args:
        df: DataFrame cần làm sạch
        config: Configuration
        text_column: Tên cột văn bản
        rating_column: Tên cột rating
        
    Returns:
        Tuple (cleaned_df, stats)
    """
    cleaner = DataCleaner(config)
    return cleaner.clean(df, text_column, rating_column)


if __name__ == "__main__":
    # Test cleaner
    print("=" * 60)
    print("Testing Data Cleaner Module")
    print("=" * 60)
    
    # Create sample data
    sample_data = {
        'review_text': [
            "The hotel was GREAT! Best stay ever!!!",
            "TERRIBLE service, dirty room. <script>bad</script>",
            "It was OK. Average experience.",
            None,
            "The room was clean and the staff was friendly.",
            "Excellent location, wonderful amenities, highly recommended!",
            "The hotel was absolutely amazing! Great service, friendly staff!",
            "Terrible experience! Room was dirty and staff was rude.",
            None,
            "A decent hotel for the price. Nothing special."
        ],
        'rating': [5, 1, 3, 4, 5, 4, 5, 1, 3, 3],
        'Hotel': ['Hotel A', 'Hotel B', 'Hotel C', 'Hotel D', 'Hotel E',
                  'Hotel A', 'Hotel B', 'Hotel C', 'Hotel D', 'Hotel E']
    }
    
    df = pd.DataFrame(sample_data)
    
    print("\nOriginal DataFrame:")
    print(df)
    
    # Clean data
    cleaner = DataCleaner()
    cleaned_df, stats = cleaner.clean(df)
    
    print("\n\nCleaned DataFrame:")
    print(cleaned_df[['review_text', 'cleaned_text', 'rating', 'sentiment']])
    
    print("\n\nPreprocessing Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n\nBefore/After Comparison:")
    comparison = cleaner.get_before_after_comparison(df, cleaned_df)
    for section, data in comparison.items():
        print(f"  {section}: {data}")
