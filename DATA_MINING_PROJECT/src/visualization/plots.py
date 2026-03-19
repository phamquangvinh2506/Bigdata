# ==============================================================================
# VISUALIZATION PLOTS MODULE
# ==============================================================================
"""
Module tạo biểu đồ trực quan:
- EDA plots
- Model evaluation plots
- Clustering visualization
- Learning curves
- Confusion matrices
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class PlotGenerator:
    """
    Class tạo các biểu đồ trực quan cho data mining project.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        output_dir: str = "outputs/figures"
    ):
        """
        Initialize PlotGenerator.
        
        Args:
            config: Configuration dictionary
            output_dir: Directory to save plots
        """
        self.config = config or {}
        self.viz_config = self.config.get('visualization', {})
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Style settings
        self.figure_size = tuple(self.viz_config.get('figure_size', [12, 8]))
        self.dpi = self.viz_config.get('dpi', 150)
        self.colors = self.viz_config.get('colors', {
            'positive': '#2ecc71',
            'neutral': '#f39c12',
            'negative': '#e74c3c',
            'primary': '#3498db',
            'secondary': '#9b59b6'
        })
        
    def _save_plot(self, filename: str, fig: plt.Figure) -> str:
        """Save plot to file."""
        save_format = self.viz_config.get('save_format', 'png')
        filepath = self.output_dir / f"{filename}.{save_format}"
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        return str(filepath)
    
    # ==========================================================================
    # EDA PLOTS
    # ==========================================================================
    
    def plot_rating_distribution(
        self,
        df: pd.DataFrame,
        rating_column: str = 'rating',
        save: bool = True
    ) -> plt.Figure:
        """
        Plot rating distribution.
        
        Args:
            df: DataFrame
            rating_column: Tên cột rating
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figure_size)
        
        # Count plot
        rating_counts = df[rating_column].value_counts().sort_index()
        colors = [self.colors.get('negative'), self.colors.get('negative'), 
                 self.colors.get('neutral'), self.colors.get('positive'), 
                 self.colors.get('positive')]
        
        axes[0].bar(rating_counts.index, rating_counts.values, color=colors)
        axes[0].set_xlabel('Rating')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Rating Distribution (Count)')
        
        # Pie chart
        axes[1].pie(rating_counts.values, labels=rating_counts.index, autopct='%1.1f%%',
                   colors=colors, explode=[0.02]*5)
        axes[1].set_title('Rating Distribution (Percentage)')
        
        plt.suptitle('Hotel Reviews Rating Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            self._save_plot('01_rating_distribution', fig)
        
        return fig
    
    def plot_sentiment_distribution(
        self,
        df: pd.DataFrame,
        sentiment_column: str = 'sentiment',
        save: bool = True
    ) -> plt.Figure:
        """
        Plot sentiment distribution.
        
        Args:
            df: DataFrame
            sentiment_column: Tên cột sentiment
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figure_size)
        
        sentiment_counts = df[sentiment_column].value_counts()
        colors = [self.colors.get('positive'), self.colors.get('neutral'), 
                 self.colors.get('negative')]
        
        # Bar chart
        axes[0].bar(sentiment_counts.index, sentiment_counts.values, color=colors)
        axes[0].set_xlabel('Sentiment')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Sentiment Distribution')
        
        for i, (idx, v) in enumerate(sentiment_counts.items()):
            axes[0].text(i, v + 10, f'{v}', ha='center')
        
        # Pie chart
        axes[1].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
                   colors=colors, explode=[0.02]*len(sentiment_counts))
        axes[1].set_title('Sentiment Distribution (Percentage)')
        
        plt.suptitle('Hotel Reviews Sentiment Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            self._save_plot('02_sentiment_distribution', fig)
        
        return fig
    
    def plot_text_statistics(
        self,
        df: pd.DataFrame,
        text_column: str = 'review_text',
        save: bool = True
    ) -> plt.Figure:
        """
        Plot text statistics.
        
        Args:
            df: DataFrame
            text_column: Tên cột văn bản
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        df_copy = df.copy()
        df_copy['text_length'] = df_copy[text_column].astype(str).str.len()
        df_copy['word_count'] = df_copy[text_column].astype(str).str.split().str.len()
        
        fig, axes = plt.subplots(2, 2, figsize=self.figure_size)
        
        # Text length distribution
        axes[0, 0].hist(df_copy['text_length'], bins=50, color=self.colors['primary'], edgecolor='white')
        axes[0, 0].set_xlabel('Character Count')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Review Length Distribution (Characters)')
        axes[0, 0].axvline(df_copy['text_length'].mean(), color='red', linestyle='--', label=f'Mean: {df_copy["text_length"].mean():.0f}')
        axes[0, 0].legend()
        
        # Word count distribution
        axes[0, 1].hist(df_copy['word_count'], bins=50, color=self.colors['secondary'], edgecolor='white')
        axes[0, 1].set_xlabel('Word Count')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Review Length Distribution (Words)')
        axes[0, 1].axvline(df_copy['word_count'].mean(), color='red', linestyle='--', label=f'Mean: {df_copy["word_count"].mean():.0f}')
        axes[0, 1].legend()
        
        # Text length by rating
        df_copy.boxplot(column='text_length', by='rating', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Rating')
        axes[1, 0].set_ylabel('Character Count')
        axes[1, 0].set_title('Review Length by Rating')
        plt.suptitle('')  # Remove automatic title
        
        # Text length by sentiment
        df_copy.boxplot(column='text_length', by='sentiment', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Sentiment')
        axes[1, 1].set_ylabel('Character Count')
        axes[1, 1].set_title('Review Length by Sentiment')
        plt.suptitle('')
        
        plt.tight_layout()
        
        if save:
            self._save_plot('03_text_statistics', fig)
        
        return fig
    
    def plot_wordcloud(
        self,
        texts: List[str],
        title: str = "Word Cloud",
        save: bool = True
    ) -> plt.Figure:
        """
        Plot word cloud.
        
        Args:
            texts: List of text documents
            title: Plot title
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        try:
            from wordcloud import WordCloud
            
            # Combine all texts
            text = ' '.join(texts)
            
            # Create word cloud
            wordcloud = WordCloud(
                width=800, height=400,
                background_color='white',
                max_words=100,
                colormap='viridis'
            ).generate(text)
            
            fig, ax = plt.subplots(figsize=self.figure_size)
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(title, fontsize=14, fontweight='bold')
            
            if save:
                self._save_plot('04_wordcloud', fig)
            
            return fig
        except ImportError:
            print("[WARNING] wordcloud not available. Skipping word cloud plot.")
            return None
    
    # ==========================================================================
    # MODEL EVALUATION PLOTS
    # ==========================================================================
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        labels: List[str],
        title: str = "Confusion Matrix",
        save: bool = True
    ) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            labels: Class labels
            title: Plot title
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save:
            self._save_plot('05_confusion_matrix', fig)
        
        return fig
    
    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        metric: str = 'f1_macro',
        title: str = "Model Comparison",
        save: bool = True
    ) -> plt.Figure:
        """
        Plot model comparison bar chart.
        
        Args:
            comparison_df: DataFrame with model comparison
            metric: Metric column to plot
            title: Plot title
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by metric
        df_sorted = comparison_df.sort_values(metric, ascending=False)
        
        # Plot
        bars = ax.barh(df_sorted['Model'], df_sorted[metric], color=self.colors['primary'])
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{width:.4f}', va='center')
        
        ax.set_xlabel(metric)
        ax.set_ylabel('Model')
        ax.set_title(title)
        ax.set_xlim(0, max(df_sorted[metric]) * 1.15)
        
        plt.tight_layout()
        
        if save:
            self._save_plot('06_model_comparison', fig)
        
        return fig
    
    def plot_learning_curve(
        self,
        results_df: pd.DataFrame,
        x_col: str = 'label_percentage',
        y_col: str = 'f1_macro',
        title: str = "Learning Curve",
        save: bool = True
    ) -> plt.Figure:
        """
        Plot learning curve.
        
        Args:
            results_df: DataFrame with learning curve data
            x_col: Column for x-axis (e.g., label percentage)
            y_col: Column for y-axis (e.g., F1 score)
            title: Plot title
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group by x_col and calculate mean/std
        grouped = results_df.groupby(x_col)[y_col].agg(['mean', 'std']).reset_index()
        
        # Plot
        ax.errorbar(grouped[x_col], grouped['mean'], yerr=grouped['std'],
                   marker='o', capsize=5, linewidth=2, markersize=8,
                   color=self.colors['primary'])
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)
        
        # Add labels
        for _, row in grouped.iterrows():
            ax.annotate(f'{row["mean"]:.3f}', 
                       (row[x_col], row['mean'] + row['std'] + 0.01),
                       ha='center', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            self._save_plot('07_learning_curve', fig)
        
        return fig
    
    def plot_cluster_visualization(
        self,
        coords_2d: np.ndarray,
        labels: np.ndarray,
        cluster_names: Optional[Dict[int, str]] = None,
        title: str = "Cluster Visualization",
        save: bool = True
    ) -> plt.Figure:
        """
        Plot 2D cluster visualization.
        
        Args:
            coords_2d: 2D coordinates
            labels: Cluster labels
            cluster_names: Optional cluster names
            title: Plot title
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create colormap
        n_clusters = len(set(labels))
        cmap = plt.cm.get_cmap('tab10', n_clusters)
        
        # Plot points
        scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                           c=labels, cmap=cmap, alpha=0.6, s=50)
        
        # Add legend
        handles = []
        for cluster_id in sorted(set(labels)):
            if cluster_names:
                name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
            else:
                name = f"Cluster {cluster_id}"
            handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor=cmap(cluster_id), 
                                     markersize=10, label=name))
        
        ax.legend(handles=handles, loc='best')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save:
            self._save_plot('08_cluster_visualization', fig)
        
        return fig
    
    def plot_top_terms(
        self,
        top_terms: Dict[int, List[Tuple[str, float]]],
        n_terms: int = 10,
        title: str = "Top Terms per Cluster",
        save: bool = True
    ) -> plt.Figure:
        """
        Plot top terms for each cluster.
        
        Args:
            top_terms: Dictionary mapping cluster_id to list of (term, score)
            n_terms: Number of top terms to show
            title: Plot title
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        n_clusters = len(top_terms)
        fig, axes = plt.subplots(1, n_clusters, figsize=(4*n_clusters, 6))
        
        if n_clusters == 1:
            axes = [axes]
        
        for i, (cluster_id, terms) in enumerate(top_terms.items()):
            terms = terms[:n_terms]
            term_names = [t[0] for t in terms]
            term_scores = [t[1] for t in terms]
            
            axes[i].barh(term_names[::-1], term_scores[::-1], color=self.colors['primary'])
            axes[i].set_xlabel('TF-IDF Score')
            axes[i].set_title(f'Cluster {cluster_id}')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            self._save_plot('09_top_terms', fig)
        
        return fig
    
    def plot_association_rules(
        self,
        rules_df: pd.DataFrame,
        n_rules: int = 15,
        title: str = "Top Association Rules",
        save: bool = True
    ) -> plt.Figure:
        """
        Plot top association rules.
        
        Args:
            rules_df: DataFrame with rules
            n_rules: Number of top rules to show
            title: Plot title
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        top_rules = rules_df.head(n_rules)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        
        # Support
        axes[0].barh(range(len(top_rules)), top_rules['support'].values)
        axes[0].set_yticks(range(len(top_rules)))
        axes[0].set_yticklabels([f"Rule {i+1}" for i in range(len(top_rules))])
        axes[0].set_xlabel('Support')
        axes[0].set_title('Support')
        
        # Confidence
        axes[1].barh(range(len(top_rules)), top_rules['confidence'].values, color=self.colors['positive'])
        axes[1].set_yticks(range(len(top_rules)))
        axes[1].set_yticklabels([f"Rule {i+1}" for i in range(len(top_rules))])
        axes[1].set_xlabel('Confidence')
        axes[1].set_title('Confidence')
        
        # Lift
        axes[2].barh(range(len(top_rules)), top_rules['lift'].values, color=self.colors['secondary'])
        axes[2].set_yticks(range(len(top_rules)))
        axes[2].set_yticklabels([f"Rule {i+1}" for i in range(len(top_rules))])
        axes[2].set_xlabel('Lift')
        axes[2].set_title('Lift')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            self._save_plot('10_association_rules', fig)
        
        return fig
    
    def plot_regression_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Regression Predictions vs Actual",
        save: bool = True
    ) -> plt.Figure:
        """
        Plot regression predictions vs actual values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figure_size)
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5, color=self.colors['primary'])
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                    'r--', linewidth=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Rating')
        axes[0].set_ylabel('Predicted Rating')
        axes[0].set_title('Predictions vs Actual')
        axes[0].legend()
        
        # Residuals
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, color=self.colors['secondary'])
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicted Rating')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            self._save_plot('11_regression_predictions', fig)
        
        return fig


if __name__ == "__main__":
    print("Testing PlotGenerator...")
    
    # Create sample data
    np.random.seed(42)
    
    # Sample DataFrame
    df = pd.DataFrame({
        'rating': np.random.choice([1, 2, 3, 4, 5], 500, p=[0.05, 0.1, 0.15, 0.4, 0.3]),
        'sentiment': np.random.choice(['positive', 'neutral', 'negative'], 500, p=[0.6, 0.25, 0.15]),
        'review_text': ['Sample review text ' + str(i) for i in range(500)]
    })
    
    # Create plots
    generator = PlotGenerator()
    
    # Test plots
    fig1 = generator.plot_rating_distribution(df)
    print("Rating distribution plot created")
    
    fig2 = generator.plot_sentiment_distribution(df)
    print("Sentiment distribution plot created")
    
    fig3 = generator.plot_text_statistics(df)
    print("Text statistics plot created")
    
    print(f"\nPlots saved to: {generator.output_dir}")
