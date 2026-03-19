# -*- coding: utf-8 -*-
"""
Pipeline script chạy toàn bộ data mining workflow.
Chạy: python scripts/run_pipeline.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set UTF-8 encoding for Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.data.loader import load_data, load_config
from src.data.cleaner import DataCleaner
from src.features.builder import FeatureBuilder
from src.mining.association import AssociationMiner
from src.mining.clustering import ClusterAnalyzer
from src.models.supervised import SentimentClassifier
from src.models.semi_supervised import SemiSupervisedLearner
from src.models.regression import RatingPredictor
from src.evaluation.report import ReportGenerator
from src.visualization.plots import PlotGenerator


def run_pipeline(config_path: str = "configs/params.yaml", n_samples: int = 2000):
    """
    Chạy toàn bộ pipeline.
    
    Args:
        config_path: Đường dẫn đến config file
        n_samples: Số lượng samples để xử lý
    """
    print("=" * 80)
    print("HOTEL REVIEWS DATA MINING PIPELINE")
    print("Đề tài 11: Phân tích đánh giá khách sạn & chủ đề dịch vụ")
    print("=" * 80)
    
    # Load configuration
    config = load_config(config_path)
    print(f"\n[INFO] Loaded configuration from {config_path}")
    
    # Create output directories
    output_dirs = [
        "data/processed",
        "outputs/figures",
        "outputs/tables",
        "outputs/models",
        "outputs/reports"
    ]
    for d in output_dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    # STEP 1: Load Data
    print("\n" + "=" * 80)
    print("STEP 1: DATA LOADING")
    print("=" * 80)
    df = load_data(n_rows=n_samples, config_path=config_path)
    print(f"\n[INFO] Loaded {len(df)} reviews")
    
    # STEP 2: Preprocessing
    print("\n" + "=" * 80)
    print("STEP 2: TEXT PREPROCESSING")
    print("=" * 80)
    cleaner = DataCleaner(config)
    df_cleaned, preprocess_stats = cleaner.clean(df, text_column='review_text', rating_column='rating')
    print(f"\n[INFO] Preprocessed data shape: {df_cleaned.shape}")
    
    # STEP 3: Feature Engineering
    print("\n" + "=" * 80)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 80)
    feature_builder = FeatureBuilder(config)
    cleaned_texts = df_cleaned['cleaned_text'].astype(str).tolist()
    tfidf_matrix = feature_builder.build_tfidf_features(cleaned_texts, fit=True)
    
    # Rebuild stat and aspect features on CLEANED data
    stat_features = feature_builder.build_statistical_features(df_cleaned, text_column='review_text')
    aspect_features, aspect_counts = feature_builder.build_aspect_features(df_cleaned, text_column='cleaned_text')
    
    print(f"\n[INFO] TF-IDF shape: {tfidf_matrix.shape}")
    print(f"[INFO] Aspect counts: {aspect_counts}")
    
    # STEP 4: Mining - Association Rules
    print("\n" + "=" * 80)
    print("STEP 4: ASSOCIATION RULES MINING")
    print("=" * 80)
    miner = AssociationMiner(config)
    rules = miner.mine_association_rules(df_cleaned, text_column='cleaned_text')
    if len(rules) > 0:
        top_rules = miner.get_top_rules(10)
        print(f"\n[INFO] Found {len(rules)} rules")
        print("\nTop 10 Rules:")
        print(top_rules)
    else:
        print("\n[WARNING] No association rules found")
    
    # STEP 5: Clustering
    print("\n" + "=" * 80)
    print("STEP 5: TOPIC CLUSTERING")
    print("=" * 80)
    cluster_analyzer = ClusterAnalyzer(config)
    labels, coords_2d = cluster_analyzer.fit_predict(tfidf_matrix.toarray())
    cluster_names = cluster_analyzer.assign_cluster_names(tfidf_matrix, feature_builder.tfidf_vectorizer)
    cluster_stats = cluster_analyzer.get_cluster_statistics(df_cleaned)
    print(f"\n[INFO] Silhouette Score: {cluster_analyzer.silhouette_avg:.4f}")
    print("\nCluster Statistics:")
    print(cluster_stats)
    
    # STEP 6: Classification
    print("\n" + "=" * 80)
    print("STEP 6: SENTIMENT CLASSIFICATION")
    print("=" * 80)
    from sklearn.model_selection import train_test_split
    # Prepare data for classification
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    scaler = MinMaxScaler()
    stat_scaled = scaler.fit_transform(stat_features)
    
    # Ensure non-negative values for Naive Bayes
    X_combined = np.hstack([tfidf_matrix.toarray(), stat_scaled, aspect_features])
    X_combined = np.clip(X_combined, 0, None)  # Ensure non-negative
    y_sentiment = df_cleaned['sentiment'].values
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_sentiment, test_size=0.2, random_state=42)
    
    # Train models
    classifier = SentimentClassifier(config)
    classifier.train_baselines(X_train, y_train)
    classifier.train_strong_model(X_train, y_train)
    clf_results = classifier.evaluate(X_test, y_test)
    clf_comparison = classifier.compare_models()
    print("\nModel Comparison:")
    print(clf_comparison)
    
    # STEP 7: Regression
    print("\n" + "=" * 80)
    print("STEP 7: RATING REGRESSION")
    print("=" * 80)
    y_rating = df_cleaned['rating'].values.astype(float)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_combined, y_rating, test_size=0.2, random_state=42)
    
    predictor = RatingPredictor(config)
    predictor.train_baselines(X_train_reg, y_train_reg)
    predictor.train_strong_model(X_train_reg, y_train_reg)
    reg_results = predictor.evaluate(X_test_reg, y_test_reg)
    reg_comparison = predictor.compare_models()
    print("\nRegression Model Comparison:")
    print(reg_comparison)
    
    # STEP 8: Semi-supervised Learning
    print("\n" + "=" * 80)
    print("STEP 8: SEMI-SUPERVISED LEARNING")
    print("=" * 80)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_sentiment)
    
    learner = SemiSupervisedLearner(config)
    lc_results = learner.run_learning_curve_experiment(X_combined, y_encoded, n_repeats=2)
    lc_summary = learner.get_learning_curve_summary()
    print("\nLearning Curve Summary:")
    print(lc_summary)
    
    # STEP 9: Visualizations
    print("\n" + "=" * 80)
    print("STEP 9: VISUALIZATIONS")
    print("=" * 80)
    plot_gen = PlotGenerator(config)
    
    fig1 = plot_gen.plot_rating_distribution(df_cleaned)
    print("[INFO] Created rating distribution plot")
    
    fig2 = plot_gen.plot_sentiment_distribution(df_cleaned)
    print("[INFO] Created sentiment distribution plot")
    
    fig3 = plot_gen.plot_text_statistics(df_cleaned)
    print("[INFO] Created text statistics plot")
    
    if coords_2d is not None:
        fig4 = plot_gen.plot_cluster_visualization(coords_2d, labels, cluster_names)
        print("[INFO] Created cluster visualization")
    
    # Additional visualizations
    print("\n[INFO] Creating additional visualizations...")
    
    # Model comparison plot - fix column names
    clf_df = clf_comparison.copy()
    if 'F1-macro' in clf_df.columns:
        fig5 = plot_gen.plot_model_comparison(clf_df, metric='F1-macro', title="Sentiment Classification Model Comparison")
        print("[INFO] Created model comparison plot")
    
    # Regression comparison
    if len(reg_comparison) > 0:
        reg_df = reg_comparison.copy()
        if 'RMSE' in reg_df.columns:
            fig6 = plot_gen.plot_model_comparison(reg_df, metric='RMSE', title="Rating Regression Model Comparison")
            print("[INFO] Created regression comparison plot")
    
    # Top terms per cluster
    top_terms = cluster_analyzer.get_top_terms_per_cluster(tfidf_matrix, feature_builder.tfidf_vectorizer)
    fig7 = plot_gen.plot_top_terms(top_terms, n_terms=10)
    print("[INFO] Created top terms plot")
    
    # Association rules plot
    if len(rules) > 0:
        fig8 = plot_gen.plot_association_rules(top_rules, n_rules=10)
        print("[INFO] Created association rules plot")
    
    # Regression predictions plot
    y_pred_reg = predictor.best_model.predict(X_test_reg) if predictor.best_model else None
    if y_pred_reg is not None:
        fig9 = plot_gen.plot_regression_predictions(y_test_reg, y_pred_reg)
        print("[INFO] Created regression predictions plot")
    
    # STEP 10: Generate Report
    print("\n" + "=" * 80)
    print("STEP 10: GENERATING REPORT")
    print("=" * 80)
    generator = ReportGenerator("Hotel Reviews Data Mining - Đề tài 11")
    
    exec_summary = generator.create_executive_summary(
        df_cleaned,
        {'classification': clf_results, 'regression': reg_results}
    )
    generator.add_section("Executive Summary", exec_summary)
    
    insights = [
        {'title': 'Room Quality', 'description': 'Most discussed aspect in reviews', 'action': 'Maintain high standards'},
        {'title': 'Service', 'description': 'Strong correlation with satisfaction', 'action': 'Staff training'},
        {'title': 'Location', 'description': 'Key factor in positive reviews', 'action': 'Marketing focus'},
        {'title': 'Cleanliness', 'description': 'Critical for guest satisfaction', 'action': 'Quality control'},
        {'title': 'Price/Value', 'description': 'Important for repeat customers', 'action': 'Competitive pricing'}
    ]
    generator.add_insights(insights)
    
    generator.save_report("outputs/reports/final_report.md")
    print("[INFO] Report saved to outputs/reports/final_report.md")
    
    # STEP 11: Save artifacts
    print("\n" + "=" * 80)
    print("STEP 11: SAVING ARTIFACTS")
    print("=" * 80)
    
    # Save cleaned data
    df_cleaned.to_csv("data/processed/cleaned_data.csv", index=False)
    print("[INFO] Saved cleaned data")
    
    # Save models
    classifier.save_model("outputs/models")
    predictor.save_model("outputs/models")
    feature_builder.save_vectorizers("outputs/models")
    print("[INFO] Saved trained models")
    
    # Save clustering results
    df_clustered = cluster_analyzer.add_clusters_to_dataframe(df_cleaned)
    df_clustered.to_csv("data/processed/clustered_data.csv", index=False)
    if len(rules) > 0:
        rules.to_csv("data/processed/association_rules.csv", index=False)
    print("[INFO] Saved clustering and rules results")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nOutputs saved to:")
    print("  - outputs/figures/")
    print("  - outputs/models/")
    print("  - outputs/reports/")
    print("  - data/processed/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Hotel Reviews Data Mining Pipeline")
    parser.add_argument("--config", type=str, default="configs/params.yaml", help="Config file path")
    parser.add_argument("--samples", type=int, default=2000, help="Number of samples to process")
    
    args = parser.parse_args()
    
    run_pipeline(config_path=args.config, n_samples=args.samples)
