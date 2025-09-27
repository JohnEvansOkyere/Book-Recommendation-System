#!/usr/bin/env python3
"""
Visualization Analysis for Book Recommendation System
==================================================

This script creates comprehensive visualizations for the book recommendation system
including performance charts, data distribution plots, and recommendation quality graphs.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BookRecommendationVisualizer:
    """
    Visualization class for the book recommendation system
    """
    
    def __init__(self, artifacts_dir="artifacts"):
        """Initialize the visualizer"""
        self.artifacts_dir = artifacts_dir
        self.book_pivot_path = os.path.join(artifacts_dir, "serialized_objects", "book_pivot.pkl")
        self.clean_data_path = os.path.join(artifacts_dir, "dataset", "clean_data", "clean_data.csv")
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load necessary data files"""
        try:
            print("Loading data for visualization...")
            
            # Load clean data
            self.df = pd.read_csv(self.clean_data_path)
            print(f"✓ Loaded clean data: {self.df.shape}")
            
            # Load pivot table
            self.book_pivot = pickle.load(open(self.book_pivot_path, 'rb'))
            print(f"✓ Loaded pivot table: {self.book_pivot.shape}")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            sys.exit(1)
    
    def plot_rating_distribution(self):
        """Plot rating distribution"""
        print("📊 Creating rating distribution plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Rating count plot
        rating_counts = self.df['rating'].value_counts().sort_index()
        ax1.bar(rating_counts.index, rating_counts.values, color='skyblue', alpha=0.7)
        ax1.set_xlabel('Rating')
        ax1.set_ylabel('Count')
        ax1.set_title('Distribution of Ratings')
        ax1.grid(True, alpha=0.3)
        
        # Rating percentage plot
        rating_pct = (rating_counts / len(self.df)) * 100
        ax2.bar(rating_pct.index, rating_pct.values, color='lightcoral', alpha=0.7)
        ax2.set_xlabel('Rating')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('Rating Distribution (Percentage)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('rating_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Rating distribution plot saved as 'rating_distribution.png'")
    
    def plot_user_activity(self):
        """Plot user activity patterns"""
        print("📊 Creating user activity plots...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # User rating counts
        user_rating_counts = self.df['user_id'].value_counts()
        ax1.hist(user_rating_counts, bins=50, color='lightgreen', alpha=0.7)
        ax1.set_xlabel('Number of Ratings per User')
        ax1.set_ylabel('Number of Users')
        ax1.set_title('Distribution of User Activity')
        ax1.grid(True, alpha=0.3)
        
        # User rating counts (log scale)
        ax2.hist(np.log10(user_rating_counts), bins=50, color='orange', alpha=0.7)
        ax2.set_xlabel('Log10(Number of Ratings per User)')
        ax2.set_ylabel('Number of Users')
        ax2.set_title('User Activity Distribution (Log Scale)')
        ax2.grid(True, alpha=0.3)
        
        # Top active users
        top_users = user_rating_counts.head(20)
        ax3.barh(range(len(top_users)), top_users.values, color='purple', alpha=0.7)
        ax3.set_xlabel('Number of Ratings')
        ax3.set_ylabel('User ID')
        ax3.set_title('Top 20 Most Active Users')
        ax3.grid(True, alpha=0.3)
        
        # User activity statistics
        stats = user_rating_counts.describe()
        ax4.bar(['Mean', 'Median', 'Max', 'Min'], 
                [stats['mean'], stats['50%'], stats['max'], stats['min']], 
                color='red', alpha=0.7)
        ax4.set_ylabel('Number of Ratings')
        ax4.set_title('User Activity Statistics')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('user_activity.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ User activity plots saved as 'user_activity.png'")
    
    def plot_book_popularity(self):
        """Plot book popularity patterns"""
        print("📊 Creating book popularity plots...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Book rating counts
        book_rating_counts = self.df['title'].value_counts()
        ax1.hist(book_rating_counts, bins=50, color='lightblue', alpha=0.7)
        ax1.set_xlabel('Number of Ratings per Book')
        ax1.set_ylabel('Number of Books')
        ax1.set_title('Distribution of Book Popularity')
        ax1.grid(True, alpha=0.3)
        
        # Book rating counts (log scale)
        ax2.hist(np.log10(book_rating_counts), bins=50, color='pink', alpha=0.7)
        ax2.set_xlabel('Log10(Number of Ratings per Book)')
        ax2.set_ylabel('Number of Books')
        ax2.set_title('Book Popularity Distribution (Log Scale)')
        ax2.grid(True, alpha=0.3)
        
        # Top popular books
        top_books = book_rating_counts.head(20)
        ax3.barh(range(len(top_books)), top_books.values, color='green', alpha=0.7)
        ax3.set_xlabel('Number of Ratings')
        ax3.set_ylabel('Book Title')
        ax3.set_title('Top 20 Most Popular Books')
        ax3.grid(True, alpha=0.3)
        
        # Book popularity statistics
        stats = book_rating_counts.describe()
        ax4.bar(['Mean', 'Median', 'Max', 'Min'], 
                [stats['mean'], stats['50%'], stats['max'], stats['min']], 
                color='brown', alpha=0.7)
        ax4.set_ylabel('Number of Ratings')
        ax4.set_title('Book Popularity Statistics')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('book_popularity.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Book popularity plots saved as 'book_popularity.png'")
    
    def plot_rating_heatmap(self):
        """Plot rating heatmap for sample users and books"""
        print("📊 Creating rating heatmap...")
        
        # Sample users and books for heatmap
        sample_users = self.df['user_id'].value_counts().head(20).index
        sample_books = self.df['title'].value_counts().head(20).index
        
        # Create pivot table for heatmap
        heatmap_data = self.df[
            (self.df['user_id'].isin(sample_users)) & 
             (self.df['title'].isin(sample_books))
        ].pivot_table(
            index='title', 
            columns='user_id', 
            values='rating', 
            fill_value=0
        )
        
        plt.figure(figsize=(15, 10))
        sns.heatmap(heatmap_data, cmap='YlOrRd', cbar=True, 
                   cbar_kws={'label': 'Rating'})
        plt.title('Rating Heatmap: Top Users vs Top Books')
        plt.xlabel('User ID')
        plt.ylabel('Book Title')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('rating_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Rating heatmap saved as 'rating_heatmap.png'")
    
    def plot_performance_metrics(self):
        """Plot performance metrics comparison"""
        print("📊 Creating performance metrics visualization...")
        
        # Performance metrics from evaluation
        metrics = {
            'Precision@3': 0.0367,
            'Precision@5': 0.0300,
            'Precision@10': 0.0200,
            'Recall@3': 0.0073,
            'Recall@5': 0.0048,
            'Recall@10': 0.0098,
            'MAP@5': 0.3708,
            'NDCG@5': 0.0425,
            'Diversity': 0.2900,
            'Coverage': 0.0876,
            'Cold Start': 0.1000
        }
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Precision metrics
        precision_metrics = {k: v for k, v in metrics.items() if 'Precision' in k}
        ax1.bar(precision_metrics.keys(), precision_metrics.values(), 
                color='skyblue', alpha=0.7)
        ax1.set_ylabel('Precision Score')
        ax1.set_title('Precision@K Metrics')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Recall metrics
        recall_metrics = {k: v for k, v in metrics.items() if 'Recall' in k}
        ax2.bar(recall_metrics.keys(), recall_metrics.values(), 
                color='lightcoral', alpha=0.7)
        ax2.set_ylabel('Recall Score')
        ax2.set_title('Recall@K Metrics')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Advanced metrics
        advanced_metrics = {k: v for k, v in metrics.items() if k in ['MAP@5', 'NDCG@5', 'Diversity']}
        ax3.bar(advanced_metrics.keys(), advanced_metrics.values(), 
                color='lightgreen', alpha=0.7)
        ax3.set_ylabel('Score')
        ax3.set_title('Advanced Metrics')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Coverage and cold start
        coverage_metrics = {k: v for k, v in metrics.items() if k in ['Coverage', 'Cold Start']}
        ax4.bar(coverage_metrics.keys(), coverage_metrics.values(), 
                color='orange', alpha=0.7)
        ax4.set_ylabel('Score')
        ax4.set_title('Coverage and Cold Start Metrics')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Performance metrics plot saved as 'performance_metrics.png'")
    
    def plot_data_sparsity(self):
        """Plot data sparsity analysis"""
        print("📊 Creating data sparsity visualization...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Sparsity in pivot table
        sparsity_matrix = (self.book_pivot == 0).astype(int)
        
        # Sample for visualization (too large otherwise)
        sample_size = 100
        sample_sparsity = sparsity_matrix.iloc[:sample_size, :sample_size]
        
        im1 = ax1.imshow(sample_sparsity, cmap='RdYlBu_r', aspect='auto')
        ax1.set_title(f'Data Sparsity Pattern (Sample {sample_size}x{sample_size})')
        ax1.set_xlabel('Users')
        ax1.set_ylabel('Books')
        plt.colorbar(im1, ax=ax1, label='Missing (1) / Present (0)')
        
        # Sparsity statistics
        total_cells = self.book_pivot.shape[0] * self.book_pivot.shape[1]
        missing_cells = (self.book_pivot == 0).sum().sum()
        sparsity_pct = (missing_cells / total_cells) * 100
        
        ax2.pie([missing_cells, total_cells - missing_cells], 
                labels=['Missing Ratings', 'Present Ratings'],
                colors=['red', 'green'], 
                autopct='%1.1f%%',
                startangle=90)
        ax2.set_title(f'Data Sparsity: {sparsity_pct:.1f}%')
        
        plt.tight_layout()
        plt.savefig('data_sparsity.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Data sparsity plot saved as 'data_sparsity.png'")
    
    def create_comprehensive_dashboard(self):
        """Create a comprehensive dashboard with all visualizations"""
        print("📊 Creating comprehensive visualization dashboard...")
        
        # Create all plots
        self.plot_rating_distribution()
        self.plot_user_activity()
        self.plot_book_popularity()
        self.plot_rating_heatmap()
        self.plot_performance_metrics()
        self.plot_data_sparsity()
        
        print("\n✅ All visualizations created successfully!")
        print("Generated files:")
        print("  • rating_distribution.png")
        print("  • user_activity.png")
        print("  • book_popularity.png")
        print("  • rating_heatmap.png")
        print("  • performance_metrics.png")
        print("  • data_sparsity.png")


def main():
    """Main function to run the visualization analysis"""
    print("🎨 Starting Book Recommendation System Visualization Analysis")
    print("="*70)
    
    try:
        # Initialize visualizer
        visualizer = BookRecommendationVisualizer()
        
        # Create comprehensive dashboard
        visualizer.create_comprehensive_dashboard()
        
        print("\n✅ Visualization analysis completed successfully!")
        print("Check the generated PNG files for detailed visualizations.")
        
    except Exception as e:
        print(f"❌ Visualization analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
