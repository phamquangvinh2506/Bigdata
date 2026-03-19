# ==============================================================================
# REPORT GENERATION MODULE
# ==============================================================================
"""
Module tạo báo cáo tổng hợp:
- Tạo báo cáo markdown
- Tạo báo cáo HTML
- Tạo summary tables
- Insights và recommendations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from pathlib import Path


class ReportGenerator:
    """
    Class tạo báo cáo tổng hợp cho data mining project.
    """
    
    def __init__(self, project_name: str = "Hotel Reviews Analysis"):
        """
        Initialize ReportGenerator.
        
        Args:
            project_name: Tên project
        """
        self.project_name = project_name
        self.sections = []
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def add_section(self, title: str, content: str) -> None:
        """Add a section to the report."""
        self.sections.append({
            'title': title,
            'content': content
        })
    
    def add_metrics_table(
        self,
        title: str,
        df: pd.DataFrame,
        caption: Optional[str] = None
    ) -> None:
        """Add a metrics table to the report."""
        table_md = df.to_markdown(index=False)
        
        content = f"**{caption or ''}**\n\n{table_md}"
        
        self.add_section(title, content)
    
    def add_findings(
        self,
        findings: List[str],
        category: str = "Key Findings"
    ) -> None:
        """Add findings to the report."""
        findings_md = "\n".join([f"- {f}" for f in findings])
        
        self.add_section(category, findings_md)
    
    def add_insights(
        self,
        insights: List[Dict[str, str]]
    ) -> None:
        """Add actionable insights to the report."""
        insights_md = []
        
        for i, insight in enumerate(insights, 1):
            insights_md.append(f"### Insight {i}: {insight.get('title', 'Untitled')}")
            insights_md.append(f"\n**Description:** {insight.get('description', '')}")
            insights_md.append(f"\n**Action:** {insight.get('action', '')}")
            insights_md.append("")
        
        self.add_section("ACTIONABLE INSIGHTS", "\n".join(insights_md))
    
    def generate_markdown(self) -> str:
        """Generate full report in markdown format."""
        lines = []
        
        # Header
        lines.append(f"# {self.project_name}")
        lines.append(f"\n**Generated:** {self.timestamp}")
        lines.append("\n---\n")
        
        # Table of contents
        lines.append("## Table of Contents\n")
        for i, section in enumerate(self.sections, 1):
            lines.append(f"{i}. [{section['title']}](#{section['title'].lower().replace(' ', '-')})")
        lines.append("\n---\n")
        
        # Sections
        for section in self.sections:
            lines.append(f"## {section['title']}\n")
            lines.append(section['content'])
            lines.append("\n---\n")
        
        return "\n".join(lines)
    
    def generate_summary_stats(
        self,
        df: pd.DataFrame,
        text_column: str = 'review_text'
    ) -> Dict[str, Any]:
        """Generate summary statistics for the report."""
        summary = {
            'dataset_info': {
                'total_reviews': len(df),
                'total_columns': len(df.columns),
                'date_range': f"{df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else 'N/A'
            }
        }
        
        if 'rating' in df.columns:
            summary['rating_distribution'] = df['rating'].value_counts().sort_index().to_dict()
            summary['rating_stats'] = {
                'mean': float(df['rating'].mean()),
                'median': float(df['rating'].median()),
                'std': float(df['rating'].std()),
                'min': int(df['rating'].min()),
                'max': int(df['rating'].max())
            }
        
        if text_column in df.columns:
            df['text_length'] = df[text_column].str.len()
            summary['text_stats'] = {
                'avg_length': float(df['text_length'].mean()),
                'median_length': float(df['text_length'].median()),
                'min_length': int(df['text_length'].min()),
                'max_length': int(df['text_length'].max())
            }
        
        if 'sentiment' in df.columns:
            summary['sentiment_distribution'] = df['sentiment'].value_counts().to_dict()
        
        return summary
    
    def generate_insights(
        self,
        classification_results: Dict,
        regression_results: Dict,
        clustering_results: Optional[Dict] = None,
        association_results: Optional[Dict] = None
    ) -> List[Dict[str, str]]:
        """
        Generate actionable insights from analysis results.
        
        Returns:
            List of insight dictionaries
        """
        insights = []
        
        # Classification insights
        if classification_results:
            best_model = max(classification_results.items(), key=lambda x: x[1].get('f1_macro', 0))
            insights.append({
                'title': 'Best Classification Model',
                'description': f"{best_model[0]} achieves F1-macro of {best_model[1].get('f1_macro', 0):.4f}",
                'action': 'Use this model for production sentiment classification'
            })
        
        # Regression insights
        if regression_results:
            best_reg = min(regression_results.items(), key=lambda x: x[1].get('RMSE', float('inf')))
            insights.append({
                'title': 'Best Regression Model for Rating Prediction',
                'description': f"{best_reg[0]} achieves RMSE of {best_reg[1].get('RMSE', 0):.4f}",
                'action': 'Use this model for rating prediction in the application'
            })
        
        # Clustering insights
        if clustering_results:
            n_clusters = clustering_results.get('n_clusters', 0)
            insights.append({
                'title': 'Topic Clusters Identified',
                'description': f"Analysis identified {n_clusters} distinct topic clusters in reviews",
                'action': 'Use these clusters for topic-based filtering and analysis'
            })
        
        # Association rules insights
        if association_results:
            top_rules = association_results.get('top_rules', [])
            if top_rules:
                insights.append({
                    'title': 'Key Service Associations',
                    'description': f"Found {len(top_rules)} strong association rules between service aspects",
                    'action': 'Use these patterns to understand customer behavior and expectations'
                })
        
        # General insights
        insights.append({
            'title': 'Data Quality',
            'description': 'Reviews with mixed sentiment (multiple aspects) may require special handling',
            'action': 'Consider using aspect-based sentiment analysis for complex reviews'
        })
        
        return insights
    
    def save_report(
        self,
        output_path: str,
        format: str = 'markdown'
    ) -> None:
        """Save report to file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'markdown':
            content = self.generate_markdown()
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
        elif format == 'json':
            report_data = {
                'project_name': self.project_name,
                'timestamp': self.timestamp,
                'sections': self.sections
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2)
        
        print(f"[INFO] Report saved to {path}")
    
    def create_executive_summary(
        self,
        df: pd.DataFrame,
        model_results: Dict[str, Any]
    ) -> str:
        """Create executive summary section."""
        summary = []
        
        summary.append("# Executive Summary\n")
        summary.append(f"\nThis report presents the analysis of **{len(df)} hotel reviews** ")
        summary.append("using data mining and machine learning techniques.\n")
        
        # Key achievements
        summary.append("## Key Achievements\n")
        
        if 'rating' in df.columns:
            avg_rating = df['rating'].mean()
            summary.append(f"- **Average Rating**: {avg_rating:.2f}/5.0\n")
        
        if 'sentiment' in df.columns:
            positive_pct = (df['sentiment'] == 'positive').mean() * 100
            summary.append(f"- **Positive Sentiment**: {positive_pct:.1f}% of reviews\n")
        
        # Model performance
        summary.append("\n## Model Performance\n")
        
        if 'classification' in model_results:
            best_clf = max(model_results['classification'].items(), 
                         key=lambda x: x[1].get('f1_macro', 0))
            summary.append(f"- **Best Sentiment Classifier**: {best_clf[0]} (F1={best_clf[1].get('f1_macro', 0):.4f})\n")
        
        if 'regression' in model_results:
            best_reg = min(model_results['regression'].items(),
                         key=lambda x: x[1].get('RMSE', float('inf')))
            summary.append(f"- **Best Rating Predictor**: {best_reg[0]} (RMSE={best_reg[1].get('RMSE', 0):.4f})\n")
        
        return "".join(summary)


if __name__ == "__main__":
    print("Testing ReportGenerator...")
    
    # Create sample report
    generator = ReportGenerator("Hotel Reviews Analysis - Đề tài 11")
    
    # Add sections
    generator.add_section(
        "Introduction",
        "This is a comprehensive analysis of hotel reviews..."
    )
    
    # Add metrics table
    metrics_df = pd.DataFrame({
        'Model': ['LogisticRegression', 'NaiveBayes', 'RandomForest'],
        'F1-macro': [0.85, 0.82, 0.88],
        'Accuracy': [0.87, 0.83, 0.89]
    })
    generator.add_metrics_table("Model Comparison", metrics_df, "Sentiment Classification Results")
    
    # Add insights
    insights = [
        {
            'title': 'Service Quality is Key',
            'description': 'Customers frequently mention service quality in positive reviews',
            'action': 'Invest in staff training to improve service quality'
        }
    ]
    generator.add_insights(insights)
    
    # Generate and print
    report = generator.generate_markdown()
    print(report[:2000])  # Print first 2000 chars
    
    # Save
    generator.save_report("outputs/reports/sample_report.md")
