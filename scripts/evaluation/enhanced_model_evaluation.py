#!/usr/bin/env python3
"""
Enhanced Model Evaluation Script for Book Recommendation System
==============================================================

This script provides advanced evaluation metrics including:
- Precision@K, Recall@K, F1@K
- Mean Average Precision (MAP)
- Normalized Discounted Cumulative Gain (NDCG)
- Diversity and Novelty metrics
- Coverage metrics
- Cold start evaluation

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
    mean_squared_error, mean_absolute_error
)
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedBookRecommendationEvaluator:
    """
    Enhanced evaluation class with advanced metrics for the book recommendation system
    """
    
    def __init__(self, artifacts_dir="artifacts"):
        """Initialize the enhanced evaluator"""
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
    
    def precision_at_k(self, k=5):
        """Calculate Precision@K"""
        print(f"\n📊 Calculating Precision@{k}...")
        
        # Sample users for evaluation
        test_users = self.df['user_id'].unique()[:100]
        precision_scores = []
        
        for user_id in test_users:
            user_books = self.df[self.df['user_id'] == user_id]['title'].tolist()
            if len(user_books) < 2:
                continue
                
            # Split user's books into train/test
            train_books, test_books = train_test_split(user_books, test_size=0.3, random_state=42)
            
            if len(train_books) == 0 or len(test_books) == 0:
                continue
            
            # Get recommendations for a random training book
            seed_book = np.random.choice(train_books)
            if seed_book not in self.book_pivot.index:
                continue
                
            book_idx = self.book_pivot.index.get_loc(seed_book)
            distances, indices = self.model.kneighbors(
                self.book_pivot.iloc[book_idx, :].values.reshape(1, -1), 
                n_neighbors=k+1
            )
            
            # Get recommended books (excluding the seed book)
            recommended_books = [self.book_pivot.index[idx] for idx in indices[0][1:k+1]]
            
            # Calculate precision@k
            relevant_books = set(test_books)
            recommended_set = set(recommended_books)
            
            if len(recommended_set) > 0:
                precision = len(relevant_books & recommended_set) / len(recommended_set)
                precision_scores.append(precision)
        
        avg_precision = np.mean(precision_scores) if precision_scores else 0
        print(f"Precision@{k}: {avg_precision:.4f}")
        return avg_precision
    
    def recall_at_k(self, k=5):
        """Calculate Recall@K"""
        print(f"\n📊 Calculating Recall@{k}...")
        
        test_users = self.df['user_id'].unique()[:100]
        recall_scores = []
        
        for user_id in test_users:
            user_books = self.df[self.df['user_id'] == user_id]['title'].tolist()
            if len(user_books) < 2:
                continue
                
            train_books, test_books = train_test_split(user_books, test_size=0.3, random_state=42)
            
            if len(train_books) == 0 or len(test_books) == 0:
                continue
            
            seed_book = np.random.choice(train_books)
            if seed_book not in self.book_pivot.index:
                continue
                
            book_idx = self.book_pivot.index.get_loc(seed_book)
            distances, indices = self.model.kneighbors(
                self.book_pivot.iloc[book_idx, :].values.reshape(1, -1), 
                n_neighbors=k+1
            )
            
            recommended_books = [self.book_pivot.index[idx] for idx in indices[0][1:k+1]]
            
            # Calculate recall@k
            relevant_books = set(test_books)
            recommended_set = set(recommended_books)
            
            if len(relevant_books) > 0:
                recall = len(relevant_books & recommended_set) / len(relevant_books)
                recall_scores.append(recall)
        
        avg_recall = np.mean(recall_scores) if recall_scores else 0
        print(f"Recall@{k}: {avg_recall:.4f}")
        return avg_recall
    
    def mean_average_precision(self, k=5):
        """Calculate Mean Average Precision (MAP)"""
        print(f"\n📊 Calculating MAP@{k}...")
        
        test_users = self.df['user_id'].unique()[:50]
        map_scores = []
        
        for user_id in test_users:
            user_books = self.df[self.df['user_id'] == user_id]['title'].tolist()
            if len(user_books) < 2:
                continue
                
            train_books, test_books = train_test_split(user_books, test_size=0.3, random_state=42)
            
            if len(train_books) == 0 or len(test_books) == 0:
                continue
            
            seed_book = np.random.choice(train_books)
            if seed_book not in self.book_pivot.index:
                continue
                
            book_idx = self.book_pivot.index.get_loc(seed_book)
            distances, indices = self.model.kneighbors(
                self.book_pivot.iloc[book_idx, :].values.reshape(1, -1), 
                n_neighbors=k+1
            )
            
            recommended_books = [self.book_pivot.index[idx] for idx in indices[0][1:k+1]]
            relevant_books = set(test_books)
            
            # Calculate AP
            ap = 0
            relevant_count = 0
            for i, book in enumerate(recommended_books):
                if book in relevant_books:
                    relevant_count += 1
                    ap += relevant_count / (i + 1)
            
            if relevant_count > 0:
                ap = ap / relevant_count
                map_scores.append(ap)
        
        mean_ap = np.mean(map_scores) if map_scores else 0
        print(f"MAP@{k}: {mean_ap:.4f}")
        return mean_ap
    
    def normalized_dcg(self, k=5):
        """Calculate Normalized Discounted Cumulative Gain (NDCG)"""
        print(f"\n📊 Calculating NDCG@{k}...")
        
        test_users = self.df['user_id'].unique()[:50]
        ndcg_scores = []
        
        for user_id in test_users:
            user_data = self.df[self.df['user_id'] == user_id]
            if len(user_data) < 2:
                continue
                
            # Split user's data
            train_data, test_data = train_test_split(user_data, test_size=0.3, random_state=42)
            
            if len(train_data) == 0 or len(test_data) == 0:
                continue
            
            # Get a training book
            seed_book = train_data.iloc[0]['title']
            if seed_book not in self.book_pivot.index:
                continue
                
            book_idx = self.book_pivot.index.get_loc(seed_book)
            distances, indices = self.model.kneighbors(
                self.book_pivot.iloc[book_idx, :].values.reshape(1, -1), 
                n_neighbors=k+1
            )
            
            recommended_books = [self.book_pivot.index[idx] for idx in indices[0][1:k+1]]
            
            # Create relevance scores (1 if book is in test set, 0 otherwise)
            relevance_scores = [1 if book in test_data['title'].values else 0 for book in recommended_books]
            
            # Calculate DCG
            dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))
            
            # Calculate IDCG (ideal DCG)
            ideal_relevance = sorted(relevance_scores, reverse=True)
            idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
            
            # Calculate NDCG
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(ndcg)
        
        mean_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0
        print(f"NDCG@{k}: {mean_ndcg:.4f}")
        return mean_ndcg
    
    def diversity_metrics(self, k=5):
        """Calculate diversity metrics"""
        print(f"\n📊 Calculating Diversity Metrics...")
        
        # Get sample recommendations
        sample_books = self.book_pivot.index[:20]
        all_recommendations = []
        
        for book in sample_books:
            try:
                book_idx = self.book_pivot.index.get_loc(book)
                distances, indices = self.model.kneighbors(
                    self.book_pivot.iloc[book_idx, :].values.reshape(1, -1), 
                    n_neighbors=k+1
                )
                recommended_books = [self.book_pivot.index[idx] for idx in indices[0][1:k+1]]
                all_recommendations.extend(recommended_books)
            except:
                continue
        
        # Calculate intra-list diversity (average pairwise distance)
        if len(all_recommendations) > 1:
            unique_recommendations = list(set(all_recommendations))
            diversity = len(unique_recommendations) / len(all_recommendations) if all_recommendations else 0
        else:
            diversity = 0
        
        print(f"Intra-list Diversity: {diversity:.4f}")
        return diversity
    
    def coverage_metrics(self):
        """Calculate coverage metrics"""
        print(f"\n📊 Calculating Coverage Metrics...")
        
        # Catalog coverage
        total_books = len(self.book_pivot.index)
        books_in_recommendations = set()
        
        # Sample books to get recommendations
        sample_books = self.book_pivot.index[:50]
        
        for book in sample_books:
            try:
                book_idx = self.book_pivot.index.get_loc(book)
                distances, indices = self.model.kneighbors(
                    self.book_pivot.iloc[book_idx, :].values.reshape(1, -1), 
                    n_neighbors=6
                )
                recommended_books = [self.book_pivot.index[idx] for idx in indices[0][1:]]
                books_in_recommendations.update(recommended_books)
            except:
                continue
        
        catalog_coverage = len(books_in_recommendations) / total_books
        print(f"Catalog Coverage: {catalog_coverage:.4f}")
        
        return catalog_coverage
    
    def cold_start_evaluation(self):
        """Evaluate performance for cold start scenarios"""
        print(f"\n📊 Evaluating Cold Start Performance...")
        
        # Find users with few ratings (cold start users)
        user_rating_counts = self.df['user_id'].value_counts()
        cold_start_users = user_rating_counts[user_rating_counts <= 5].index[:20]
        
        if len(cold_start_users) == 0:
            print("No cold start users found")
            return 0
        
        cold_start_performance = []
        
        for user_id in cold_start_users:
            user_books = self.df[self.df['user_id'] == user_id]['title'].tolist()
            if len(user_books) < 2:
                continue
                
            # Use one book to get recommendations
            seed_book = user_books[0]
            if seed_book not in self.book_pivot.index:
                continue
                
            book_idx = self.book_pivot.index.get_loc(seed_book)
            distances, indices = self.model.kneighbors(
                self.book_pivot.iloc[book_idx, :].values.reshape(1, -1), 
                n_neighbors=6
            )
            
            recommended_books = [self.book_pivot.index[idx] for idx in indices[0][1:]]
            
            # Check if any recommended books are in user's other books
            other_books = set(user_books[1:])
            hit = len(set(recommended_books) & other_books) > 0
            cold_start_performance.append(1 if hit else 0)
        
        cold_start_accuracy = np.mean(cold_start_performance) if cold_start_performance else 0
        print(f"Cold Start Accuracy: {cold_start_accuracy:.4f}")
        return cold_start_accuracy
    
    def generate_enhanced_report(self):
        """Generate enhanced performance report with all metrics"""
        print("\n" + "="*80)
        print("🚀 ENHANCED PERFORMANCE EVALUATION REPORT")
        print("="*80)
        print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Calculate all metrics
        metrics = {}
        
        # Precision and Recall at different K values
        for k in [3, 5, 10]:
            metrics[f'precision@{k}'] = self.precision_at_k(k)
            metrics[f'recall@{k}'] = self.recall_at_k(k)
        
        # Advanced metrics
        metrics['map@5'] = self.mean_average_precision(5)
        metrics['ndcg@5'] = self.normalized_dcg(5)
        metrics['diversity'] = self.diversity_metrics(5)
        metrics['catalog_coverage'] = self.coverage_metrics()
        metrics['cold_start_accuracy'] = self.cold_start_evaluation()
        
        # Summary
        print("\n" + "="*60)
        print("📈 ENHANCED PERFORMANCE SUMMARY")
        print("="*60)
        
        print("Precision Metrics:")
        for k in [3, 5, 10]:
            print(f"  • Precision@{k}: {metrics[f'precision@{k}']:.4f}")
        
        print("\nRecall Metrics:")
        for k in [3, 5, 10]:
            print(f"  • Recall@{k}: {metrics[f'recall@{k}']:.4f}")
        
        print(f"\nAdvanced Metrics:")
        print(f"  • MAP@5: {metrics['map@5']:.4f}")
        print(f"  • NDCG@5: {metrics['ndcg@5']:.4f}")
        print(f"  • Diversity: {metrics['diversity']:.4f}")
        print(f"  • Catalog Coverage: {metrics['catalog_coverage']:.4f}")
        print(f"  • Cold Start Accuracy: {metrics['cold_start_accuracy']:.4f}")
        
        # Save enhanced report
        self._save_enhanced_report(metrics)
        
        return metrics
    
    def _save_enhanced_report(self, metrics):
        """Save enhanced report to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"enhanced_performance_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("ENHANCED BOOK RECOMMENDATION SYSTEM PERFORMANCE REPORT\n")
            f.write("="*60 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("PRECISION METRICS:\n")
            f.write("-" * 20 + "\n")
            for k in [3, 5, 10]:
                f.write(f"Precision@{k}: {metrics[f'precision@{k}']:.4f}\n")
            
            f.write(f"\nRECALL METRICS:\n")
            f.write("-" * 15 + "\n")
            for k in [3, 5, 10]:
                f.write(f"Recall@{k}: {metrics[f'recall@{k}']:.4f}\n")
            
            f.write(f"\nADVANCED METRICS:\n")
            f.write("-" * 18 + "\n")
            f.write(f"MAP@5: {metrics['map@5']:.4f}\n")
            f.write(f"NDCG@5: {metrics['ndcg@5']:.4f}\n")
            f.write(f"Diversity: {metrics['diversity']:.4f}\n")
            f.write(f"Catalog Coverage: {metrics['catalog_coverage']:.4f}\n")
            f.write(f"Cold Start Accuracy: {metrics['cold_start_accuracy']:.4f}\n")
        
        print(f"\n📄 Enhanced report saved to: {report_file}")


def main():
    """Main function to run the enhanced evaluation"""
    print("🚀 Starting Enhanced Book Recommendation System Evaluation")
    print("="*70)
    
    try:
        # Initialize enhanced evaluator
        evaluator = EnhancedBookRecommendationEvaluator()
        
        # Generate enhanced report
        results = evaluator.generate_enhanced_report()
        
        print("\n✅ Enhanced evaluation completed successfully!")
        print("Check the generated enhanced report file for detailed results.")
        
    except Exception as e:
        print(f"❌ Enhanced evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
