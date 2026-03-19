# ==============================================================================
# CLUSTERING MODULE
# ==============================================================================
"""
Module phân cụm chủ đề (Topic Clustering):
- K-Means clustering
- HDBSCAN clustering
- Silhouette Score evaluation
- Cluster naming và representative reviews
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("[WARNING] HDBSCAN not available. Install with: pip install hdbscan")


class ClusterAnalyzer:
    """
    Class phân tích clustering cho hotel reviews.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ClusterAnalyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.clustering_config = self.config.get('clustering', {})
        
        # K-Means params
        self.n_clusters = self.clustering_config.get('kmeans', {}).get('n_clusters', 5)
        self.random_state = self.clustering_config.get('kmeans', {}).get('random_state', 42)
        
        # HDBSCAN params
        self.min_cluster_size = self.clustering_config.get('hdbscan', {}).get('min_cluster_size', 50)
        self.min_samples = self.clustering_config.get('hdbscan', {}).get('min_samples', 10)
        
        self.method = self.clustering_config.get('method', 'kmeans')
        
        # Components
        self.model = None
        self.cluster_labels = None
        self.silhouette_avg = None
        self.cluster_names = {}
        
    def fit_predict(
        self,
        X: np.ndarray,
        method: Optional[str] = None,
        return_reduced: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit và predict clusters.
        
        Args:
            X: Feature matrix
            method: Clustering method ('kmeans' or 'hdbscan')
            return_reduced: Return 2D coordinates for visualization
            
        Returns:
            Tuple (cluster_labels, 2D_coordinates)
        """
        method = method or self.method
        
        print(f"[INFO] Clustering with {method}...")
        
        if method == 'kmeans':
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
                max_iter=300
            )
            self.cluster_labels = self.model.fit_predict(X)
            
        elif method == 'hdbscan':
            if HDBSCAN_AVAILABLE:
                self.model = hdbscan.HDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=self.min_samples,
                    metric='euclidean'
                )
                self.cluster_labels = self.model.fit_predict(X)
                
                # Update n_clusters for HDBSCAN
                self.n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
            else:
                print("[WARNING] HDBSCAN not available, falling back to K-Means")
                self.fit_predict(X, method='kmeans', return_reduced=False)
                return self.cluster_labels, None
        
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Calculate silhouette score
        if len(set(self.cluster_labels)) > 1:
            self.silhouette_avg = silhouette_score(X, self.cluster_labels)
            print(f"[INFO] Silhouette Score: {self.silhouette_avg:.4f}")
        
        # Reduce dimensions for visualization
        coords_2d = None
        if return_reduced:
            coords_2d = self._reduce_dimensions(X)
        
        return self.cluster_labels, coords_2d
    
    def _reduce_dimensions(self, X: np.ndarray, method: str = 'pca') -> np.ndarray:
        """
        Reduce dimensions to 2D for visualization.
        
        Args:
            X: High-dimensional feature matrix
            method: 'pca' or 'tsne'
            
        Returns:
            2D coordinates
        """
        n_samples = min(5000, X.shape[0])
        X_sample = X[:n_samples] if X.shape[0] > n_samples else X
        
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else:
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        
        coords = reducer.fit_transform(X_sample)
        return coords
    
    def assign_cluster_names(
        self,
        feature_matrix: np.ndarray,
        vectorizer,
        top_n: int = 5
    ) -> Dict[int, str]:
        """
        Đặt tên cho mỗi cluster dựa trên top terms.
        
        Args:
            feature_matrix: TF-IDF feature matrix
            vectorizer: TF-IDF vectorizer
            top_n: Số terms cao nhất để đặt tên
            
        Returns:
            Dictionary mapping cluster_id to cluster_name
        """
        if self.cluster_labels is None:
            raise ValueError("Must run fit_predict first")
        
        feature_names = vectorizer.get_feature_names_out()
        cluster_names = {}
        
        for cluster_id in range(self.n_clusters):
            # Get indices of documents in this cluster
            mask = self.cluster_labels == cluster_id
            
            if mask.sum() == 0:
                cluster_names[cluster_id] = f"Cluster {cluster_id}"
                continue
            
            # Get mean TF-IDF scores for this cluster
            if hasattr(feature_matrix, 'toarray'):
                cluster_tfidf = feature_matrix[mask].mean(axis=0).A1
            else:
                cluster_tfidf = feature_matrix[mask].mean(axis=0)
            
            # Get top terms
            top_indices = np.argsort(cluster_tfidf)[::-1][:top_n]
            top_terms = [feature_names[i] for i in top_indices]
            
            # Generate cluster name
            cluster_names[cluster_id] = self._generate_cluster_name(top_terms, cluster_id)
            self.cluster_names[cluster_id] = cluster_names[cluster_id]
        
        return cluster_names
    
    def _generate_cluster_name(self, top_terms: List[str], cluster_id: int) -> str:
        """Generate descriptive name for cluster based on top terms."""
        terms_str = ', '.join(top_terms[:3])
        
        # Simple heuristic naming
        if any(term in ['clean', 'dirty', 'tidy', 'hygiene'] for term in top_terms):
            return f"Cleanliness & Room Quality"
        elif any(term in ['service', 'staff', 'friendly', 'helpful'] for term in top_terms):
            return f"Service & Staff Quality"
        elif any(term in ['location', 'central', 'near', 'walk'] for term in top_terms):
            return f"Location & Accessibility"
        elif any(term in ['food', 'breakfast', 'restaurant', 'buffet'] for term in top_terms):
            return f"Food & Dining"
        elif any(term in ['price', 'expensive', 'value', 'money'] for term in top_terms):
            return f"Value & Pricing"
        elif any(term in ['pool', 'gym', 'amenities', 'facilities'] for term in top_terms):
            return f"Facilities & Amenities"
        elif any(term in ['room', 'bed', 'bathroom', 'shower'] for term in top_terms):
            return f"Room & Comfort"
        elif any(term in ['noise', 'quiet', 'noisy'] for term in top_terms):
            return f"Noise & Peacefulness"
        else:
            return f"Topic: {terms_str}"
    
    def get_cluster_statistics(
        self,
        df: pd.DataFrame,
        text_column: str = 'review_text'
    ) -> pd.DataFrame:
        """
        Tính statistics cho mỗi cluster.
        
        Args:
            df: DataFrame với dữ liệu
            text_column: Tên cột văn bản
            
        Returns:
            DataFrame chứa cluster statistics
        """
        if self.cluster_labels is None:
            raise ValueError("Must run fit_predict first")
        
        df = df.copy()
        df['cluster'] = self.cluster_labels
        
        stats = []
        
        for cluster_id in range(self.n_clusters):
            mask = df['cluster'] == cluster_id
            cluster_df = df[mask]
            
            stat = {
                'cluster_id': cluster_id,
                'cluster_name': self.cluster_names.get(cluster_id, f"Cluster {cluster_id}"),
                'n_reviews': len(cluster_df),
                'percentage': len(cluster_df) / len(df) * 100
            }
            
            if 'rating' in df.columns:
                stat['avg_rating'] = cluster_df['rating'].mean()
                stat['rating_std'] = cluster_df['rating'].std()
                stat['rating_distribution'] = cluster_df['rating'].value_counts().to_dict()
            
            if text_column in df.columns:
                stat['avg_review_length'] = cluster_df[text_column].str.len().mean()
            
            # Dominant sentiment
            if 'sentiment' in df.columns:
                dominant_sentiment = cluster_df['sentiment'].mode().iloc[0] if len(cluster_df['sentiment'].mode()) > 0 else 'unknown'
                stat['dominant_sentiment'] = dominant_sentiment
            
            stats.append(stat)
        
        return pd.DataFrame(stats)
    
    def get_representative_reviews(
        self,
        df: pd.DataFrame,
        text_column: str = 'review_text',
        n_representatives: int = 3
    ) -> Dict[int, List[str]]:
        """
        Lấy representative reviews cho mỗi cluster.
        
        Args:
            df: DataFrame
            text_column: Tên cột văn bản
            n_representatives: Số reviews đại diện cho mỗi cluster
            
        Returns:
            Dictionary mapping cluster_id to list of representative reviews
        """
        if self.cluster_labels is None:
            raise ValueError("Must run fit_predict first")
        
        df = df.copy()
        df['cluster'] = self.cluster_labels
        
        representatives = {}
        
        for cluster_id in range(self.n_clusters):
            mask = df['cluster'] == cluster_id
            cluster_df = df[mask]
            
            if len(cluster_df) == 0:
                representatives[cluster_id] = []
                continue
            
            # Get reviews closest to cluster centroid (simple heuristic: median length)
            if text_column in cluster_df.columns:
                median_length = cluster_df[text_column].str.len().median()
                cluster_df = cluster_df.copy()
                cluster_df['length_diff'] = abs(cluster_df[text_column].str.len() - median_length)
                cluster_df = cluster_df.sort_values('length_diff')
            
            # Take top n
            reviews = cluster_df[text_column].head(n_representatives).tolist()
            representatives[cluster_id] = reviews
        
        return representatives
    
    def get_top_terms_per_cluster(
        self,
        feature_matrix: np.ndarray,
        vectorizer,
        n_terms: int = 10
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Lấy top terms cho mỗi cluster.
        
        Args:
            feature_matrix: TF-IDF feature matrix
            vectorizer: TF-IDF vectorizer
            n_terms: Số terms cao nhất
            
        Returns:
            Dictionary mapping cluster_id to list of (term, score) tuples
        """
        if self.cluster_labels is None:
            raise ValueError("Must run fit_predict first")
        
        feature_names = vectorizer.get_feature_names_out()
        top_terms = {}
        
        for cluster_id in range(self.n_clusters):
            mask = self.cluster_labels == cluster_id
            
            if mask.sum() == 0:
                top_terms[cluster_id] = []
                continue
            
            cluster_tfidf = feature_matrix[mask].mean(axis=0).A1
            top_indices = np.argsort(cluster_tfidf)[::-1][:n_terms]
            
            top_terms[cluster_id] = [
                (feature_names[i], cluster_tfidf[i]) 
                for i in top_indices
            ]
        
        return top_terms
    
    def get_silhouette_analysis(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Phân tích silhouette chi tiết.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary chứa silhouette analysis
        """
        if self.cluster_labels is None:
            raise ValueError("Must run fit_predict first")
        
        n_samples = min(5000, X.shape[0])
        X_sample = X[:n_samples]
        labels_sample = self.cluster_labels[:n_samples]
        
        silhouette_vals = silhouette_samples(X_sample, labels_sample)
        
        analysis = {
            'overall_silhouette': float(silhouette_score(X_sample, labels_sample)),
            'per_cluster_silhouette': {},
            'worst_cluster': None,
            'best_cluster': None
        }
        
        for cluster_id in range(self.n_clusters):
            mask = labels_sample == cluster_id
            if mask.sum() > 0:
                cluster_sil = silhouette_vals[mask].mean()
                analysis['per_cluster_silhouette'][cluster_id] = float(cluster_sil)
        
        if analysis['per_cluster_silhouette']:
            worst_id = min(analysis['per_cluster_silhouette'], 
                          key=analysis['per_cluster_silhouette'].get)
            best_id = max(analysis['per_cluster_silhouette'],
                         key=analysis['per_cluster_silhouette'].get)
            analysis['worst_cluster'] = worst_id
            analysis['best_cluster'] = best_id
        
        return analysis
    
    def evaluate_clustering(self, X: np.ndarray) -> Dict[str, float]:
        """
        Đánh giá clustering quality.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.cluster_labels is None:
            raise ValueError("Must run fit_predict first")
        
        metrics = {}
        
        # Silhouette Score
        metrics['silhouette_score'] = float(silhouette_score(X, self.cluster_labels))
        
        # Inertia (for K-Means)
        if hasattr(self.model, 'inertia_'):
            metrics['inertia'] = float(self.model.inertia_)
        
        # Cluster sizes
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        metrics['n_clusters'] = len(unique)
        metrics['cluster_sizes'] = dict(zip(unique.tolist(), counts.tolist()))
        metrics['largest_cluster'] = int(max(counts))
        metrics['smallest_cluster'] = int(min(counts))
        metrics['avg_cluster_size'] = float(np.mean(counts))
        metrics['size_variance'] = float(np.var(counts))
        
        return metrics
    
    def add_clusters_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Thêm cluster labels vào DataFrame."""
        if self.cluster_labels is None:
            raise ValueError("Must run fit_predict first")
        
        df = df.copy()
        df['cluster'] = self.cluster_labels
        
        if self.cluster_names:
            df['cluster_name'] = df['cluster'].map(self.cluster_names)
        
        return df


if __name__ == "__main__":
    print("Testing Cluster Analyzer...")
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Sample texts
    texts = [
        "The hotel room was clean and comfortable, great service",
        "Terrible service, dirty room, awful breakfast",
        "Perfect location, friendly staff, excellent amenities",
        "Great breakfast buffet, clean room, amazing service",
        "Noisy location, poor wifi, small room",
        "Amazing hotel with wonderful amenities and great location",
        "The bathroom was dirty and the bed was uncomfortable",
        "Excellent location, the staff was very helpful"
    ] * 30
    
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(texts)
    
    # Cluster
    analyzer = ClusterAnalyzer(config={'clustering': {'method': 'kmeans', 'kmeans': {'n_clusters': 4}}})
    labels, coords_2d = analyzer.fit_predict(X.toarray())
    
    print(f"\nCluster labels: {np.unique(labels)}")
    print(f"Silhouette Score: {analyzer.silhouette_avg:.4f}")
    
    # Name clusters
    cluster_names = analyzer.assign_cluster_names(X.toarray(), vectorizer)
    print(f"\nCluster names: {cluster_names}")
