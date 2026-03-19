# ==============================================================================
# DATA LOADER MODULE
# ==============================================================================
"""
Module chịu trách nhiệm load dữ liệu từ các nguồn khác nhau
- Load từ CSV file
- Load từ URL (Kaggle)
- Validate data format
- Generate sample data nếu không có dữ liệu thực
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union
import yaml
import warnings

warnings.filterwarnings('ignore')


def load_config(config_path: str = "configs/params.yaml") -> Dict[str, Any]:
    """
    Load configuration từ YAML file.
    
    Args:
        config_path: Đường dẫn đến file cấu hình
        
    Returns:
        Dictionary chứa các tham số cấu hình
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"[WARNING] Config file not found at {config_path}")
        print("Using default configuration...")
        return get_default_config()
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def get_default_config() -> Dict[str, Any]:
    """Trả về default configuration nếu không tìm thấy file config."""
    return {
        'data': {
            'raw_path': 'data/raw/hotel_reviews.csv',
            'processed_path': 'data/processed/processed_reviews.csv',
            'source': 'Kaggle Hotel Reviews Dataset',
            'n_rows_sample': 10000,
            'columns': {
                'review_text': 'Review',
                'rating': 'Rating',
                'reviewer_name': 'Reviewer',
                'date': 'Date',
                'hotel_name': 'Hotel'
            }
        },
        'preprocessing': {
            'lowercase': True,
            'remove_special_chars': True,
            'remove_stopwords': True,
            'remove_numbers': True,
            'apply_stemming': True,
            'min_word_length': 2
        },
        'features': {
            'tfidf': {
                'max_features': 5000,
                'min_df': 2,
                'max_df': 0.95,
                'ngram_range': [1, 2]
            }
        }
    }


def load_data(
    path: Optional[str] = None,
    n_rows: Optional[int] = None,
    config_path: str = "configs/params.yaml"
) -> pd.DataFrame:
    """
    Load dữ liệu từ CSV file hoặc tạo sample data.
    
    Args:
        path: Đường dẫn đến file CSV (None = auto detect)
        n_rows: Số dòng muốn load (None = load all)
        config_path: Đường dẫn config file
        
    Returns:
        DataFrame chứa dữ liệu đánh giá khách sạn
        
    Raises:
        FileNotFoundError: Nếu không tìm thấy file và không tạo được sample
    """
    config = load_config(config_path)
    
    # Auto detect path
    if path is None:
        path = config['data']['raw_path']
    
    path = Path(path)
    
    # Try to load from file
    if path.exists():
        print(f"[INFO] Loading data from: {path}")
        
        try:
            # Thử load với các encoding khác nhau
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    df = pd.read_csv(path, encoding=encoding, nrows=n_rows)
                    print(f"[INFO] Successfully loaded with encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                # Nếu không decode được, dùng errors='ignore'
                df = pd.read_csv(path, encoding='utf-8', errors='ignore', nrows=n_rows)
            
            # Validate columns
            df = validate_and_standardize_columns(df, config)
            
            print(f"[INFO] Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            print(f"[WARNING] Error loading file: {e}")
            print("[INFO] Generating sample data instead...")
            return generate_sample_data(n_rows or config['data']['n_rows_sample'], config)
    
    else:
        print(f"[INFO] File not found at {path}")
        print("[INFO] Generating sample hotel reviews data...")
        return generate_sample_data(n_rows or config['data']['n_rows_sample'], config)


def validate_and_standardize_columns(
    df: pd.DataFrame, 
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Validate và standardize column names theo config.
    
    Args:
        df: DataFrame cần validate
        config: Configuration dictionary
        
    Returns:
        DataFrame đã được standardize
    """
    col_map = config['data']['columns']
    
    # Rename columns nếu cần
    rename_map = {}
    for target_col, source_col in col_map.items():
        if source_col in df.columns:
            rename_map[source_col] = target_col
    
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"[INFO] Renamed columns: {rename_map}")
    
    # Check required columns
    required_cols = ['review_text', 'rating']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df


def generate_sample_data(n_rows: int, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Tạo sample data mô phỏng Hotel Reviews Dataset.
    
    Args:
        n_rows: Số lượng sample cần tạo
        config: Configuration dictionary
        
    Returns:
        DataFrame chứa sample data
    """
    if config is None:
        config = get_default_config()
    
    print(f"[INFO] Generating {n_rows} sample hotel reviews...")
    
    np.random.seed(42)
    
    # Template reviews cho các khía cạnh khác nhau
    positive_reviews = [
        "The hotel was absolutely amazing! Great service, friendly staff, and perfect location.",
        "Excellent stay! The room was clean, comfortable, and well-equipped. Will definitely come back.",
        "Outstanding experience! The staff went above and beyond. Great breakfast and amenities.",
        "Perfect location, beautiful views, and exceptional service. Highly recommended!",
        "Wonderful hotel with great facilities. The pool and gym were excellent.",
        "Amazing experience! Clean rooms, helpful staff, and delicious breakfast.",
        "The best hotel I've ever stayed at. Everything was perfect from check-in to check-out.",
        "Fantastic service and beautiful rooms. The location was perfect for sightseeing.",
        "Great value for money! The room was spacious and modern. Staff were very friendly.",
        "Perfect getaway! The hotel exceeded all expectations. Will recommend to friends.",
        "The room was spotless, the bed was comfortable, and the staff was incredibly helpful.",
        "Excellent hotel with great amenities. The spa treatment was wonderful.",
        "Very clean and well-maintained hotel. The location was ideal for our trip.",
        "The service was impeccable. The staff remembered our preferences from our last stay.",
        "Beautiful hotel in a great location. The rooftop bar had amazing views.",
    ]
    
    negative_reviews = [
        "Terrible experience! The room was dirty, staff was rude, and the location was noisy.",
        "Very disappointed. The room was not as described and the bathroom was filthy.",
        "Poor service and dirty facilities. The AC didn't work and no one fixed it.",
        "The worst hotel experience ever. Mold in bathroom, broken furniture, awful smell.",
        "Not recommended at all. The staff was unhelpful and the room was uncomfortable.",
        "Disgusting! Hair in the bed, stained sheets, and rude management.",
        "The hotel was dirty and outdated. The photos looked nothing like reality.",
        "Horrible stay. Noisy, dirty, and the staff was extremely rude.",
        "The room smelled bad and was not cleaned properly. Very disappointed.",
        "Awful experience. The pool was dirty, the gym was broken, and no refunds.",
        "Terrible location, noisy throughout the night. Would not stay here again.",
        "The bathroom was disgusting and the room had bedbugs. Avoid at all costs!",
        "Overpriced and underwhelming. The service was poor and the facilities were outdated.",
        "The hotel was not as advertised. Misleading photos and rude staff.",
        "Complete waste of money. The room was tiny, dirty, and the breakfast was awful.",
    ]
    
    neutral_reviews = [
        "Decent hotel in a convenient location. The room was okay but could be cleaner.",
        "Average experience. The service was fine but nothing special.",
        "The hotel was okay for a one-night stay. Nothing remarkable.",
        "Standard hotel with basic amenities. Good for budget travelers.",
        "The room was clean but small. The location was convenient.",
        "Reasonable stay for the price. The breakfast was average.",
        "The hotel met our basic needs. The staff was polite but not memorable.",
        "An okay experience. The room was as expected for the price range.",
        "The hotel was fine for the price. Some areas need updating.",
        "Average hotel with basic facilities. The location was good.",
        "Decent place to stay. The room was comfortable enough.",
        "The hotel was okay. Nothing special but also nothing terrible.",
        "Standard accommodation. The wifi was slow but the bed was comfortable.",
        "The hotel served its purpose. Basic but clean rooms.",
        "An average experience. The price was fair for what we got.",
    ]
    
    # Mixed/aspect-specific reviews
    mixed_reviews = [
        "Great location and friendly staff, but the room was dirty and the bed was uncomfortable.",
        "Beautiful hotel but terrible service. The staff was rude and unhelpful.",
        "The room was nice but the noise from the street was unbearable.",
        "Excellent breakfast but the room was overpriced for what it offered.",
        "The staff was helpful but the hotel was far from the city center.",
        "Modern room but the bathroom was not clean. Disappointing.",
        "Great amenities but the room was small and cramped.",
        "The location was perfect but the AC was broken and noisy.",
        "Friendly staff and good breakfast, but the bed was uncomfortable.",
        "The hotel was clean but the decor was outdated and dull.",
        "Amazing pool and facilities but the room was not soundproofed.",
        "The service was excellent but the hotel lacked basic amenities.",
        "Beautiful views but the room was too hot and the AC didn't work well.",
        "The staff was professional but the room had a strange smell.",
        "Great for families but not ideal for business travelers. Slow wifi.",
    ]
    
    # Hotel names
    hotels = [
        "Grand Plaza Hotel", "Seaside Resort", "Mountain View Inn", "City Center Suites",
        "Royal Palace Hotel", "Ocean Breeze Resort", "Park View Hotel", "Luxury Stay Hotel",
        "Budget Inn Express", "Sunset Beach Hotel", "Downtown Lodge", "Royal Gardens Hotel",
        "The Metropolitan", "Harbor View Hotel", "Golden Star Hotel", "Crystal Palace Hotel"
    ]
    
    # Reviewer names
    reviewers = [
        "John D.", "Sarah M.", "Michael B.", "Emily R.", "David K.", "Lisa P.", "James W.",
        "Jennifer L.", "Robert H.", "Amanda C.", "Chris T.", "Nicole S.", "Kevin J.",
        "Rachel G.", "Brian M.", "Ashley D.", "Daniel F.", "Megan R.", "Eric S.", "Laura B."
    ]
    
    # Generate data
    data = []
    
    # Distribution: 60% positive, 25% neutral, 10% negative, 5% mixed
    n_positive = int(n_rows * 0.50)
    n_neutral = int(n_rows * 0.25)
    n_negative = int(n_rows * 0.15)
    n_mixed = n_rows - n_positive - n_neutral - n_negative
    
    for i in range(n_positive):
        rating = np.random.choice([4, 5], p=[0.4, 0.6])
        data.append({
            'review_text': np.random.choice(positive_reviews),
            'rating': rating,
            'sentiment': 'positive',
            'reviewer_name': np.random.choice(reviewers),
            'date': generate_random_date(),
            'hotel_name': np.random.choice(hotels),
            'review_length': len(np.random.choice(positive_reviews)),
            'word_count': len(np.random.choice(positive_reviews).split())
        })
    
    for i in range(n_neutral):
        rating = np.random.choice([3, 3], p=[0.5, 0.5])
        data.append({
            'review_text': np.random.choice(neutral_reviews),
            'rating': rating,
            'sentiment': 'neutral',
            'reviewer_name': np.random.choice(reviewers),
            'date': generate_random_date(),
            'hotel_name': np.random.choice(hotels),
            'review_length': len(np.random.choice(neutral_reviews)),
            'word_count': len(np.random.choice(neutral_reviews).split())
        })
    
    for i in range(n_negative):
        rating = np.random.choice([1, 2], p=[0.5, 0.5])
        data.append({
            'review_text': np.random.choice(negative_reviews),
            'rating': rating,
            'sentiment': 'negative',
            'reviewer_name': np.random.choice(reviewers),
            'date': generate_random_date(),
            'hotel_name': np.random.choice(hotels),
            'review_length': len(np.random.choice(negative_reviews)),
            'word_count': len(np.random.choice(negative_reviews).split())
        })
    
    for i in range(n_mixed):
        rating = np.random.choice([2, 3, 4], p=[0.3, 0.4, 0.3])
        data.append({
            'review_text': np.random.choice(mixed_reviews),
            'rating': rating,
            'sentiment': 'mixed',
            'reviewer_name': np.random.choice(reviewers),
            'date': generate_random_date(),
            'hotel_name': np.random.choice(hotels),
            'review_length': len(np.random.choice(mixed_reviews)),
            'word_count': len(np.random.choice(mixed_reviews).split())
        })
    
    df = pd.DataFrame(data)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Drop temporary columns
    df = df.drop(columns=['review_length', 'word_count'])
    
    # Add more variation to reviews
    df = add_review_variation(df)
    
    print(f"[INFO] Generated {len(df)} sample reviews")
    print(f"[INFO] Rating distribution:\n{df['rating'].value_counts().sort_index()}")
    
    return df


def generate_random_date() -> str:
    """Generate random date string."""
    year = np.random.randint(2020, 2025)
    month = np.random.randint(1, 13)
    day = np.random.randint(1, 29)
    return f"{year}-{month:02d}-{day:02d}"


def add_review_variation(df: pd.DataFrame) -> pd.DataFrame:
    """Add variation to reviews by combining templates and adding noise."""
    variations = [
        " Overall, ",
        " I would say ",
        " In my opinion, ",
        " To be honest, ",
        " Honestly, ",
        " I must say ",
        " All things considered, ",
        " Considering everything, ",
    ]
    
    endings = [
        " Would visit again.",
        " Not bad.",
        " Could be better.",
        " Would recommend.",
        " Will come back.",
        " Good value.",
        " A bit pricey.",
        " Worth the money.",
    ]
    
    # Add variation to some reviews
    n_to_modify = len(df) // 3
    indices = np.random.choice(len(df), n_to_modify, replace=False)
    
    for idx in indices:
        if np.random.random() > 0.5:
            prefix = np.random.choice(variations)
            df.at[idx, 'review_text'] = prefix + df.at[idx, 'review_text'].lower()
        else:
            suffix = np.random.choice(endings)
            df.at[idx, 'review_text'] = df.at[idx, 'review_text'] + suffix
    
    return df


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary statistics of the dataset.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Dictionary containing summary statistics
    """
    summary = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
    }
    
    if 'rating' in df.columns:
        summary['rating_stats'] = {
            'mean': float(df['rating'].mean()),
            'std': float(df['rating'].std()),
            'min': int(df['rating'].min()),
            'max': int(df['rating'].max()),
            'distribution': df['rating'].value_counts().sort_index().to_dict()
        }
    
    if 'review_text' in df.columns:
        df_copy = df.copy()
        df_copy['review_length'] = df_copy['review_text'].astype(str).str.len()
        df_copy['word_count'] = df_copy['review_text'].astype(str).str.split().str.len()
        
        summary['text_stats'] = {
            'avg_length': float(df_copy['review_length'].mean()),
            'avg_words': float(df_copy['word_count'].mean()),
            'max_length': int(df_copy['review_length'].max()),
            'min_length': int(df_copy['review_length'].min())
        }
    
    return summary


def save_data(df: pd.DataFrame, path: str) -> None:
    """
    Save DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(path, index=False, encoding='utf-8')
    print(f"[INFO] Data saved to: {path}")


if __name__ == "__main__":
    # Test loader
    print("=" * 60)
    print("Testing Data Loader Module")
    print("=" * 60)
    
    # Load or generate data
    df = load_data(n_rows=100)
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nData summary:")
    summary = get_data_summary(df)
    for key, value in summary.items():
        print(f"  {key}: {value}")
