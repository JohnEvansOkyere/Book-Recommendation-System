#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script for Book Recommendation System
================================================================

This script evaluates the performance of the collaborative filtering book recommendation system
using various metrics including accuracy, precision, recall, AUC, and recommendation quality metrics.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    mean_squared_error, mean_absolute_error,
    roc_auc_score, average_precision_score
)
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BookRecommendationEvaluator:
    """
    Comprehensive evaluation class for the book recommendation system
    """
    
    def __init__(self, artifacts_dir="artifacts"):
        """
        Initialize the evaluator with paths to model artifacts
        
        Args:
            artifacts_dir (str): Path to the artifacts directory
        """
        self.artifacts_dir = artifacts_dir
        self.model_path = os.path.join(artifacts_dir, "trained_model", "model.pkl")
        self.book_pivot_path = os.path.join(artifacts_dir, "serialized_objects", "book_pivot.pkl")
        self.final_rating_path = os.path.join(artifacts_dir, "serialized_objects", "final_rating.pkl")
        self.clean_data_path = os.path.join(artifacts_dir, "dataset", "clean_data", "clean_data.csv")
        
        # Load data and model
        self._load_data()
        self._load_model()
        
    def _load_data(self):
        """Load all necessary data files"""
        try:
            print("Loading data files...")
            
            # Load clean data
            self.df = pd.read_csv(self.clean_data_path)
            print(f"✓ Loaded clean data: {self.df.shape}")
            
            # Load pivot table
            self.book_pivot = pickle.load(open(self.book_pivot_path, 'rb'))
            print(f"✓ Loaded pivot table: {self.book_pivot.shape}")
            
            # Load final rating data
            self.final_rating = pickle.load(open(self.final_rating_path, 'rb'))
            print(f"✓ Loaded final rating: {self.final_rating.shape}")
            
            # Create sparse matrix for model
            self.book_sparse = csr_matrix(self.book_pivot)
            print(f"✓ Created sparse matrix: {self.book_sparse.shape}")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            sys.exit(1)
    
    def _load_model(self):
        """Load the trained model"""
        try:
            self.model = pickle.load(open(self.model_path, 'rb'))
            print(f"✓ Loaded model: {type(self.model)}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
    
    def basic_data_statistics(self):
        """Generate basic statistics about the dataset"""
        print("\n" + "="*60)
        print("📊 DATASET STATISTICS")
        print("="*60)
        
        stats = {
            "Total Ratings": len(self.df),
            "Unique Users": self.df['user_id'].nunique(),
            "Unique Books": self.df['title'].nunique(),
            "Average Rating": round(self.df['rating'].mean(), 2),
            "Rating Range": f"{self.df['rating'].min()} - {self.df['rating'].max()}",
            "Pivot Table Shape": f"{self.book_pivot.shape[0]} books × {self.book_pivot.shape[1]} users",
            "Sparsity": f"{(self.book_pivot == 0).sum().sum() / (self.book_pivot.shape[0] * self.book_pivot.shape[1]) * 100:.2f}%"
        }
        
        for key, value in stats.items():
            print(f"{key:20}: {value}")
        
        return stats
    
    def rating_distribution_analysis(self):
        """Analyze rating distribution"""
        print("\n" + "="*60)
        print("📈 RATING DISTRIBUTION ANALYSIS")
        print("="*60)
        
        rating_counts = self.df['rating'].value_counts().sort_index()
        print("Rating Distribution:")
        for rating, count in rating_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"Rating {rating:2}: {count:6} ({percentage:5.1f}%)")
        
        # High ratings (7-10) vs Low ratings (0-3)
        high_ratings = len(self.df[self.df['rating'] >= 7])
        low_ratings = len(self.df[self.df['rating'] <= 3])
        zero_ratings = len(self.df[self.df['rating'] == 0])
        
        print(f"\nHigh Ratings (7-10): {high_ratings} ({(high_ratings/len(self.df)*100):.1f}%)")
        print(f"Low Ratings (0-3):   {low_ratings} ({(low_ratings/len(self.df)*100):.1f}%)")
        print(f"Zero Ratings:        {zero_ratings} ({(zero_ratings/len(self.df)*100):.1f}%)")
        
        return {
            'rating_distribution': rating_counts.to_dict(),
            'high_ratings_pct': high_ratings/len(self.df)*100,
            'low_ratings_pct': low_ratings/len(self.df)*100,
            'zero_ratings_pct': zero_ratings/len(self.df)*100
        }
    
    def user_activity_analysis(self):
        """Analyze user activity patterns"""
        print("\n" + "="*60)
        print("👥 USER ACTIVITY ANALYSIS")
        print("="*60)
        
        user_stats = self.df.groupby('user_id').agg({
            'rating': ['count', 'mean', 'std'],
            'title': 'nunique'
        }).round(2)
        
        user_stats.columns = ['Total_Ratings', 'Avg_Rating', 'Rating_Std', 'Unique_Books']
        
        print("User Activity Statistics:")
        print(f"Average ratings per user: {user_stats['Total_Ratings'].mean():.1f}")
        print(f"Median ratings per user:  {user_stats['Total_Ratings'].median():.1f}")
        print(f"Max ratings by a user:    {user_stats['Total_Ratings'].max()}")
        print(f"Min ratings by a user:    {user_stats['Total_Ratings'].min()}")
        print(f"Average unique books per user: {user_stats['Unique_Books'].mean():.1f}")
        
        return user_stats
    
    def book_popularity_analysis(self):
        """Analyze book popularity patterns"""
        print("\n" + "="*60)
        print("📚 BOOK POPULARITY ANALYSIS")
        print("="*60)
        
        book_stats = self.df.groupby('title').agg({
            'rating': ['count', 'mean', 'std'],
            'user_id': 'nunique'
        }).round(2)
        
        book_stats.columns = ['Total_Ratings', 'Avg_Rating', 'Rating_Std', 'Unique_Users']
        
        print("Book Popularity Statistics:")
        print(f"Average ratings per book: {book_stats['Total_Ratings'].mean():.1f}")
        print(f"Median ratings per book:  {book_stats['Total_Ratings'].median():.1f}")
        print(f"Max ratings for a book:   {book_stats['Total_Ratings'].max()}")
        print(f"Min ratings for a book:   {book_stats['Total_Ratings'].min()}")
        print(f"Average unique users per book: {book_stats['Unique_Users'].mean():.1f}")
        
        # Most and least popular books
        most_popular = book_stats.nlargest(5, 'Total_Ratings')
        least_popular = book_stats.nsmallest(5, 'Total_Ratings')
        
        print("\nTop 5 Most Popular Books:")
        for idx, (book, stats) in enumerate(most_popular.iterrows(), 1):
            print(f"{idx}. {book[:50]}... - {stats['Total_Ratings']} ratings")
        
        return book_stats
    
    def evaluate_recommendation_quality(self, test_size=0.2, random_state=42):
        """
        Evaluate recommendation quality using holdout validation
        
        Args:
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
        """
        print("\n" + "="*60)
        print("🎯 RECOMMENDATION QUALITY EVALUATION")
        print("="*60)
        
        # Create train-test split
        train_data, test_data = train_test_split(
            self.df, test_size=test_size, random_state=random_state, stratify=None
        )
        
        print(f"Training set: {len(train_data)} ratings")
        print(f"Test set: {len(test_data)} ratings")
        
        # Create pivot table for training
        train_pivot = train_data.pivot_table(
            columns='user_id', index='title', values='rating'
        ).fillna(0)
        
        # Train model on training data
        train_sparse = csr_matrix(train_pivot)
        train_model = NearestNeighbors(algorithm='brute')
        train_model.fit(train_sparse)
        
        # Evaluate on test set
        precision_scores = []
        recall_scores = []
        f1_scores = []
        hit_rates = []
        
        test_users = test_data['user_id'].unique()[:50]  # Sample for efficiency
        
        print(f"Evaluating on {len(test_users)} test users...")
        
        for user_id in test_users:
            user_test_books = test_data[test_data['user_id'] == user_id]['title'].tolist()
            if len(user_test_books) < 2:
                continue
                
            # Get user's training books
            user_train_books = train_data[train_data['user_id'] == user_id]['title'].tolist()
            if len(user_train_books) == 0:
                continue
            
            # Get recommendations for a random training book
            seed_book = np.random.choice(user_train_books)
            if seed_book not in train_pivot.index:
                continue
                
            book_idx = train_pivot.index.get_loc(seed_book)
            distances, indices = train_model.kneighbors(
                train_pivot.iloc[book_idx, :].values.reshape(1, -1), n_neighbors=6
            )
            
            # Get recommended books (excluding the seed book)
            recommended_books = [train_pivot.index[idx] for idx in indices[0][1:]]
            
            # Calculate metrics
            relevant_books = set(user_test_books)
            recommended_set = set(recommended_books)
            
            if len(recommended_set) > 0:
                precision = len(relevant_books & recommended_set) / len(recommended_set)
                recall = len(relevant_books & recommended_set) / len(relevant_books) if len(relevant_books) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                hit_rate = 1 if len(relevant_books & recommended_set) > 0 else 0
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)
                hit_rate = 1 if len(relevant_books & recommended_set) > 0 else 0
                hit_rates.append(hit_rate)
        
        # Calculate average metrics
        avg_precision = np.mean(precision_scores) if precision_scores else 0
        avg_recall = np.mean(recall_scores) if recall_scores else 0
        avg_f1 = np.mean(f1_scores) if f1_scores else 0
        hit_rate = np.mean(hit_rates) if hit_rates else 0
        
        print(f"\nRecommendation Quality Metrics:")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall:    {avg_recall:.4f}")
        print(f"Average F1-Score:  {avg_f1:.4f}")
        print(f"Hit Rate:          {hit_rate:.4f}")
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': avg_f1,
            'hit_rate': hit_rate,
            'num_evaluations': len(precision_scores)
        }
    
    def evaluate_model_performance(self):
        """Evaluate the overall model performance"""
        print("\n" + "="*60)
        print("🤖 MODEL PERFORMANCE EVALUATION")
        print("="*60)
        
        # Model information
        print(f"Model Type: {type(self.model).__name__}")
        print(f"Algorithm: {self.model.algorithm}")
        print(f"Number of Neighbors: {self.model.n_neighbors}")
        print(f"Model Parameters: {self.model.get_params()}")
        
        # Test model on sample data
        sample_books = self.book_pivot.index[:10]
        total_recommendations = 0
        successful_recommendations = 0
        
        print(f"\nTesting model on {len(sample_books)} sample books...")
        
        for book in sample_books:
            try:
                book_idx = self.book_pivot.index.get_loc(book)
                distances, indices = self.model.kneighbors(
                    self.book_pivot.iloc[book_idx, :].values.reshape(1, -1), 
                    n_neighbors=6
                )
                
                total_recommendations += 1
                if len(indices[0]) > 1:  # If we got recommendations
                    successful_recommendations += 1
                    
            except Exception as e:
                print(f"Error with book '{book}': {e}")
        
        success_rate = successful_recommendations / total_recommendations if total_recommendations > 0 else 0
        
        print(f"Model Success Rate: {success_rate:.4f}")
        print(f"Successful Recommendations: {successful_recommendations}/{total_recommendations}")
        
        return {
            'success_rate': success_rate,
            'successful_recommendations': successful_recommendations,
            'total_attempts': total_recommendations
        }
    
    def calculate_coverage_metrics(self):
        """Calculate coverage and diversity metrics"""
        print("\n" + "="*60)
        print("📊 COVERAGE AND DIVERSITY METRICS")
        print("="*60)
        
        # Catalog coverage
        total_books = len(self.book_pivot.index)
        books_with_ratings = len(self.df['title'].unique())
        catalog_coverage = (books_with_ratings / total_books) * 100
        
        print(f"Total Books in System: {total_books}")
        print(f"Books with Ratings: {books_with_ratings}")
        print(f"Catalog Coverage: {catalog_coverage:.2f}%")
        
        # User coverage
        total_users = len(self.book_pivot.columns)
        active_users = len(self.df['user_id'].unique())
        user_coverage = (active_users / total_users) * 100
        
        print(f"Total Users in System: {total_users}")
        print(f"Active Users: {active_users}")
        print(f"User Coverage: {user_coverage:.2f}%")
        
        # Sparsity analysis
        total_ratings = len(self.df)
        total_possible_ratings = total_books * total_users
        sparsity = (1 - (total_ratings / total_possible_ratings)) * 100
        
        print(f"Data Sparsity: {sparsity:.2f}%")
        
        return {
            'catalog_coverage': catalog_coverage,
            'user_coverage': user_coverage,
            'sparsity': sparsity,
            'total_books': total_books,
            'active_books': books_with_ratings,
            'total_users': total_users,
            'active_users': active_users
        }
    
    def generate_performance_report(self):
        """Generate a comprehensive performance report"""
        print("\n" + "="*80)
        print("📋 COMPREHENSIVE PERFORMANCE REPORT")
        print("="*80)
        print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Run all evaluations
        basic_stats = self.basic_data_statistics()
        rating_analysis = self.rating_distribution_analysis()
        user_analysis = self.user_activity_analysis()
        book_analysis = self.book_popularity_analysis()
        model_performance = self.evaluate_model_performance()
        coverage_metrics = self.calculate_coverage_metrics()
        rec_quality = self.evaluate_recommendation_quality()
        
        # Summary
        print("\n" + "="*60)
        print("📈 PERFORMANCE SUMMARY")
        print("="*60)
        
        print(f"Dataset Quality:")
        print(f"  • Data Sparsity: {coverage_metrics['sparsity']:.1f}%")
        print(f"  • User Coverage: {coverage_metrics['user_coverage']:.1f}%")
        print(f"  • Catalog Coverage: {coverage_metrics['catalog_coverage']:.1f}%")
        
        print(f"\nRecommendation Quality:")
        print(f"  • Precision: {rec_quality['precision']:.4f}")
        print(f"  • Recall: {rec_quality['recall']:.4f}")
        print(f"  • F1-Score: {rec_quality['f1_score']:.4f}")
        print(f"  • Hit Rate: {rec_quality['hit_rate']:.4f}")
        
        print(f"\nModel Performance:")
        print(f"  • Success Rate: {model_performance['success_rate']:.4f}")
        print(f"  • Algorithm: {self.model.algorithm}")
        print(f"  • Neighbors: {self.model.n_neighbors}")
        
        # Save report to file
        self._save_report_to_file({
            'basic_stats': basic_stats,
            'rating_analysis': rating_analysis,
            'user_analysis': user_analysis.describe().to_dict(),
            'book_analysis': book_analysis.describe().to_dict(),
            'model_performance': model_performance,
            'coverage_metrics': coverage_metrics,
            'recommendation_quality': rec_quality
        })
        
        return {
            'basic_stats': basic_stats,
            'rating_analysis': rating_analysis,
            'user_analysis': user_analysis,
            'book_analysis': book_analysis,
            'model_performance': model_performance,
            'coverage_metrics': coverage_metrics,
            'recommendation_quality': rec_quality
        }
    
    def _save_report_to_file(self, report_data):
        """Save the performance report to a file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"model_performance_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("BOOK RECOMMENDATION SYSTEM PERFORMANCE REPORT\n")
            f.write("="*50 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("DATASET STATISTICS:\n")
            f.write("-" * 20 + "\n")
            for key, value in report_data['basic_stats'].items():
                f.write(f"{key}: {value}\n")
            
            f.write(f"\nRECOMMENDATION QUALITY METRICS:\n")
            f.write("-" * 30 + "\n")
            for key, value in report_data['recommendation_quality'].items():
                f.write(f"{key}: {value:.4f}\n")
            
            f.write(f"\nMODEL PERFORMANCE:\n")
            f.write("-" * 20 + "\n")
            for key, value in report_data['model_performance'].items():
                f.write(f"{key}: {value:.4f}\n")
        
        print(f"\n📄 Performance report saved to: {report_file}")


def main():
    """Main function to run the evaluation"""
    print("🚀 Starting Book Recommendation System Evaluation")
    print("="*60)
    
    try:
        # Initialize evaluator
        evaluator = BookRecommendationEvaluator()
        
        # Generate comprehensive report
        results = evaluator.generate_performance_report()
        
        print("\n✅ Evaluation completed successfully!")
        print("Check the generated report file for detailed results.")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
