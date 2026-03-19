# ==============================================================================
# SRC PACKAGE INITIALIZATION
# Hotel Reviews Data Mining Project - Đề tài 11
# ==============================================================================

"""
DATA_MINING_PROJECT - Hotel Reviews Analysis
============================================

A comprehensive data mining project for hotel review analysis including:
- Data loading and cleaning
- Feature engineering (TF-IDF, statistical features)
- Association rules mining
- Clustering (K-Means, HDBSCAN)
- Sentiment classification
- Semi-supervised learning
- Rating regression
"""

__version__ = "1.0.0"
__author__ = "Data Mining Team"
__project__ = "Hotel Reviews Analysis - Đề tài 11"

# Import main functions for easy access
from .data.loader import load_data, load_config
from .data.cleaner import TextCleaner
from .features.builder import FeatureBuilder
from .mining.association import AssociationMiner
from .mining.clustering import ClusterAnalyzer
from .models.supervised import SentimentClassifier
from .models.semi_supervised import SemiSupervisedLearner
from .models.regression import RatingPredictor
from .evaluation.metrics import MetricsCalculator
from .evaluation.report import ReportGenerator
from .visualization.plots import PlotGenerator

__all__ = [
    "load_data",
    "load_config",
    "TextCleaner",
    "FeatureBuilder",
    "AssociationMiner",
    "ClusterAnalyzer",
    "SentimentClassifier",
    "SemiSupervisedLearner",
    "RatingPredictor",
    "MetricsCalculator",
    "ReportGenerator",
    "PlotGenerator",
]
