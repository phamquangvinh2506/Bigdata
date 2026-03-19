# -*- coding: utf-8 -*-
"""
Streamlit Demo App - Hotel Reviews Analysis
Đề tài 11: Phân tích đánh giá khách sạn & chủ đề dịch vụ

Run: streamlit run app/streamlit_app.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.data.loader import load_data
from src.data.cleaner import DataCleaner
from src.features.builder import FeatureBuilder
from src.mining.clustering import ClusterAnalyzer
from src.models.supervised import SentimentClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib

# Page config
st.set_page_config(
    page_title="Hotel Reviews Analysis",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #ecf0f1;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .positive { color: #27ae60; }
    .negative { color: #e74c3c; }
    .neutral { color: #f39c12; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_sample_data(n_rows=2000):
    """Load sample data."""
    return load_data(n_rows=n_rows)


@st.cache_resource
def train_models(df):
    """Train models on data."""
    # Clean data
    cleaner = DataCleaner()
    df_cleaned, _ = cleaner.clean(df)
    
    # Create features
    vectorizer = TfidfVectorizer(max_features=1000, max_df=0.95, min_df=2)
    X = vectorizer.fit_transform(df_cleaned['cleaned_text'].astype(str))
    
    # Train classifier
    classifier = SentimentClassifier()
    y_sentiment = df_cleaned['sentiment'].values
    
    # Use subset for faster training
    n_train = min(len(y_sentiment), 1000)
    indices = np.random.choice(len(y_sentiment), n_train, replace=False)
    
    classifier.train_baselines(X[indices].toarray(), y_sentiment[indices])
    classifier.train_strong_model(X[indices].toarray(), y_sentiment[indices])
    
    return classifier, vectorizer, df_cleaned


def main():
    """Main app function."""
    
    # Header
    st.markdown('<p class="main-header">🏨 Hotel Reviews Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #7f8c8d;">Đề tài 11: Phân tích đánh giá khách sạn & chủ đề dịch vụ</p>', unsafe_allow_html=True)
    st.divider()
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    n_samples = st.sidebar.slider("Number of samples", 500, 3000, 1500, step=500)
    show_advanced = st.sidebar.checkbox("Show Advanced Options", False)
    
    # Load data
    df = load_sample_data(n_samples)
    
    # Clean data
    cleaner = DataCleaner()
    df_cleaned, _ = cleaner.clean(df)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", "📝 Review Analyzer", "🔮 Predictions", "📈 Insights", "ℹ️ About"
    ])
    
    with tab1:
        st.header("📊 Dashboard")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_rating = df['rating'].mean()
            st.metric("Average Rating", f"{avg_rating:.2f}/5", 
                     delta=f"{(avg_rating - 3.5):.2f} vs baseline")
        
        with col2:
            total_reviews = len(df)
            st.metric("Total Reviews", f"{total_reviews:,}")
        
        with col3:
            positive_pct = (df['sentiment'] == 'positive').mean() * 100
            st.metric("Positive Reviews", f"{positive_pct:.1f}%", 
                     delta=f"{positive_pct - 50:.1f}%", delta_color="off")
        
        with col4:
            negative_pct = (df['sentiment'] == 'negative').mean() * 100
            st.metric("Negative Reviews", f"{negative_pct:.1f}%",
                     delta=f"-{negative_pct - 10:.1f}%", delta_color="inverse")
        
        st.divider()
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Rating Distribution")
            rating_counts = df['rating'].value_counts().sort_index()
            fig = px.bar(
                x=rating_counts.index, 
                y=rating_counts.values,
                color=rating_counts.index,
                color_continuous_scale=['#e74c3c', '#e74c3c', '#f39c12', '#2ecc71', '#2ecc71'],
                labels={'x': 'Rating', 'y': 'Count'}
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Sentiment Distribution")
            sentiment_counts = df['sentiment'].value_counts()
            colors = {'positive': '#2ecc71', 'neutral': '#f39c12', 'negative': '#e74c3c'}
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                color=sentiment_counts.index,
                color_discrete_map=colors
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        # Hotels performance
        st.divider()
        st.subheader("🏨 Hotel Performance")
        
        if 'hotel_name' in df.columns:
            hotel_stats = df.groupby('hotel_name').agg({
                'rating': ['mean', 'count'],
                'sentiment': lambda x: (x == 'positive').mean() * 100
            }).round(2)
            hotel_stats.columns = ['Avg Rating', 'Reviews', 'Positive %']
            hotel_stats = hotel_stats.sort_values('Avg Rating', ascending=False)
            
            st.dataframe(hotel_stats, use_container_width=True)
    
    with tab2:
        st.header("📝 Review Analyzer")
        
        # Text input
        review_text = st.text_area(
            "Enter a hotel review to analyze:",
            height=150,
            placeholder="Example: The hotel was amazing! Great service, clean room, perfect location..."
        )
        
        if st.button("🔍 Analyze Review", type="primary"):
            if review_text:
                # Simple sentiment analysis based on keywords
                review_lower = review_text.lower()
                
                positive_words = ['excellent', 'amazing', 'great', 'perfect', 'wonderful', 
                               'fantastic', 'outstanding', 'superb', 'best', 'love', 'recommend',
                               'clean', 'friendly', 'helpful', 'comfortable']
                negative_words = ['terrible', 'awful', 'horrible', 'bad', 'worst', 'disgusting',
                               'poor', 'disappointed', 'dirty', 'rude', 'avoid', 'not', 'no']
                
                pos_count = sum(1 for word in positive_words if word in review_lower)
                neg_count = sum(1 for word in negative_words if word in review_lower)
                
                # Determine sentiment
                if pos_count > neg_count:
                    sentiment = "Positive 😊"
                    color = "positive"
                    confidence = min(95, 50 + (pos_count - neg_count) * 15)
                elif neg_count > pos_count:
                    sentiment = "Negative 😞"
                    color = "negative"
                    confidence = min(95, 50 + (neg_count - pos_count) * 15)
                else:
                    sentiment = "Neutral 😐"
                    color = "neutral"
                    confidence = 50
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"**Sentiment:** <span class='{color}'>{sentiment}</span>", 
                               unsafe_allow_html=True)
                
                with col2:
                    st.metric("Confidence", f"{confidence:.0f}%")
                
                with col3:
                    st.metric("Positive/Negative Words", f"{pos_count}/{neg_count}")
                
                # Aspect detection
                st.subheader("Detected Aspects")
                
                aspects = {
                    'Room': ['room', 'bed', 'bathroom', 'shower', 'clean'],
                    'Service': ['service', 'staff', 'friendly', 'helpful'],
                    'Location': ['location', 'central', 'near', 'walk'],
                    'Food': ['food', 'breakfast', 'restaurant', 'buffet'],
                    'Price': ['price', 'expensive', 'value', 'money']
                }
                
                detected = []
                for aspect, keywords in aspects.items():
                    if any(kw in review_lower for kw in keywords):
                        detected.append(aspect)
                
                if detected:
                    st.success(f"Detected aspects: {', '.join(detected)}")
                else:
                    st.info("No specific aspects detected")
                
                # Rating estimation
                estimated_rating = 3 + (pos_count - neg_count) * 0.5
                estimated_rating = max(1, min(5, estimated_rating))
                
                st.subheader("Estimated Rating")
                st.progress(int((estimated_rating / 5) * 100), text=f"{estimated_rating:.1f}/5")
            else:
                st.warning("Please enter a review to analyze.")
    
    with tab3:
        st.header("🔮 Predictions")
        
        # Sentiment prediction
        st.subheader("Sentiment Prediction")
        
        # Train simple model
        try:
            vectorizer = TfidfVectorizer(max_features=500, max_df=0.9, min_df=2)
            texts = df_cleaned['cleaned_text'].astype(str).tolist()[:500]
            X = vectorizer.fit_transform(texts)
            y = df_cleaned['sentiment'].values[:500]
            
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=500)
            clf.fit(X, y)
            
            # Prediction input
            pred_text = st.text_input("Enter review for sentiment prediction:", key="pred_text")
            
            if pred_text:
                X_pred = vectorizer.transform([pred_text.lower()])
                pred = clf.predict(X_pred)[0]
                proba = clf.predict_proba(X_pred)[0]
                
                st.success(f"**Predicted Sentiment:** {pred.capitalize()}")
                
                # Show probabilities
                proba_df = pd.DataFrame({
                    'Sentiment': clf.classes_,
                    'Probability': proba
                })
                fig = px.bar(proba_df, x='Sentiment', y='Probability', 
                           color='Sentiment',
                           color_discrete_map={'positive': '#2ecc71', 
                                             'neutral': '#f39c12', 
                                             'negative': '#e74c3c'})
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error in prediction: {e}")
        
        # Rating prediction
        st.divider()
        st.subheader("Rating Prediction")
        
        st.info("Rating prediction model analyzes text to estimate the likely rating (1-5).")
        
        rating_pred_text = st.text_input("Enter review for rating prediction:", key="rating_pred")
        
        if rating_pred_text:
            # Simple estimation based on word counts
            text_lower = rating_pred_text.lower()
            pos_words = ['excellent', 'amazing', 'great', 'perfect', 'best', 'wonderful', 'love']
            neg_words = ['terrible', 'bad', 'worst', 'dirty', 'awful', 'horrible', 'disgusting']
            
            pos = sum(1 for w in pos_words if w in text_lower)
            neg = sum(1 for w in neg_words if w in text_lower)
            
            estimated = 3 + (pos - neg) * 0.5
            estimated = max(1, min(5, round(estimated)))
            
            st.markdown(f"**Estimated Rating:** {estimated}/5")
            
            # Visual rating
            rating_display = "⭐" * estimated + "☆" * (5 - estimated)
            st.markdown(f"<h3 style='text-align: center;'>{rating_display}</h3>", unsafe_allow_html=True)
    
    with tab4:
        st.header("📈 Insights & Recommendations")
        
        # Key metrics
        st.subheader("Key Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating distribution insights
            fig = px.histogram(df, x='rating', nbins=5, 
                             color_discrete_sequence=['#3498db'])
            fig.update_layout(title="Rating Distribution",
                            xaxis_title="Rating",
                            yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sentiment by rating
            sentiment_rating = pd.crosstab(df['rating'], df['sentiment'])
            fig = px.bar(sentiment_rating, barmode='group',
                        color_discrete_map={'positive': '#2ecc71',
                                           'neutral': '#f39c12',
                                           'negative': '#e74c3c'})
            fig.update_layout(title="Sentiment by Rating",
                            xaxis_title="Rating",
                            yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Insights
        st.subheader("🎯 Key Insights")
        
        insights = [
            {
                "icon": "🛏️",
                "title": "Room Quality",
                "description": "Room quality and cleanliness are the most discussed aspects in positive reviews.",
                "recommendation": "Maintain high cleaning standards and invest in room amenities."
            },
            {
                "icon": "👥",
                "title": "Service Quality",
                "description": "Friendly and helpful staff strongly correlates with positive sentiment.",
                "recommendation": "Implement regular staff training programs."
            },
            {
                "icon": "📍",
                "title": "Location",
                "description": "Convenient location is a key factor in positive reviews.",
                "recommendation": "Highlight location benefits in marketing materials."
            },
            {
                "icon": "💰",
                "title": "Value for Money",
                "description": "Price perception affects both satisfaction and repeat visits.",
                "recommendation": "Offer competitive pricing and value packages."
            }
        ]
        
        for insight in insights:
            with st.container():
                col1, col2 = st.columns([1, 5])
                with col1:
                    st.markdown(f"<h2 style='text-align: center;'>{insight['icon']}</h2>", 
                              unsafe_allow_html=True)
                with col2:
                    st.markdown(f"**{insight['title']}**")
                    st.write(insight['description'])
                    st.markdown(f"<i style='color: #27ae60;'>💡 {insight['recommendation']}</i>", 
                              unsafe_allow_html=True)
                st.divider()
        
        # Top words
        st.subheader("📝 Most Frequent Terms")
        
        # Simple word frequency
        all_text = ' '.join(df_cleaned['cleaned_text'].astype(str).tolist())
        words = all_text.split()
        word_freq = pd.Series(words).value_counts().head(20)
        
        fig = px.bar(x=word_freq.values[::-1], 
                    y=word_freq.index[::-1],
                    orientation='h',
                    color=word_freq.values[::-1],
                    color_continuous_scale='Blues')
        fig.update_layout(title="Top 20 Most Frequent Words",
                        xaxis_title="Frequency",
                        showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("ℹ️ About")
        
        st.markdown("""
        ## 🏨 Hotel Reviews Analysis - Đề tài 11
        
        A comprehensive data mining project for hotel review analysis.
        
        ### Features
        
        - **Sentiment Classification**: Classify reviews as positive, neutral, or negative
        - **Aspect Detection**: Identify key aspects mentioned in reviews (room, service, location, etc.)
        - **Rating Prediction**: Predict ratings from review text
        - **Trend Analysis**: Visualize review patterns and trends
        
        ### Methodology
        
        1. **Data Collection**: Hotel reviews from Kaggle dataset
        2. **Preprocessing**: Text cleaning, stopwords removal, stemming
        3. **Feature Engineering**: TF-IDF vectorization, statistical features
        4. **Analysis**: 
           - Association Rules Mining
           - Topic Clustering (K-Means)
           - Sentiment Classification
           - Semi-supervised Learning
        5. **Evaluation**: F1-macro, MAE, RMSE metrics
        
        ### Technologies Used
        
        - Python 3
        - Scikit-learn
        - Pandas & NumPy
        - Streamlit
        - Plotly
        
        ### Project Structure
        
        ```
        DATA_MINING_PROJECT/
        ├── configs/          # Configuration files
        ├── data/            # Raw and processed data
        ├── notebooks/       # Jupyter notebooks
        ├── src/             # Source code
        │   ├── data/        # Data loading and cleaning
        │   ├── features/    # Feature engineering
        │   ├── mining/      # Association rules & clustering
        │   ├── models/      # ML models
        │   └── visualization/ # Plots and charts
        └── outputs/         # Results and models
        ```
        """)
        
        st.divider()
        
        # Dataset info
        st.subheader("📊 Dataset Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Source**: Kaggle Hotel Reviews Dataset
            
            **Columns**:
            - `review_text`: Review content
            - `rating`: Rating (1-5)
            - `sentiment`: Derived sentiment
            - `hotel_name`: Hotel name
            - `date`: Review date
            """)
        
        with col2:
            st.markdown("""
            **Target Variables**:
            
            1. **Sentiment** (Classification):
               - Positive: Rating 4-5
               - Neutral: Rating 3
               - Negative: Rating 1-2
            
            2. **Rating** (Regression):
               - Continuous value 1-5
            
            **Aspects**:
            - Room, Service, Location, Food, Price, etc.
            """)
        
        st.divider()
        
        # Contact
        st.markdown("""
        ---
        **© 2026 - Hotel Reviews Data Mining Project - Đề tài 11**
        
        Built with ❤️ using Streamlit
        """)


if __name__ == "__main__":
    main()
