import os
import sys
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from books_recommender.logger.log import logging
from books_recommender.config.configuration import AppConfiguration
from books_recommender.exception.exception_handler import AppException


class ModelTrainer:
    def __init__(self, app_config = AppConfiguration()):
        try:
            self.model_trainer_config = app_config.get_model_trainer_config()
            self.data_validation_config = app_config.get_data_validation_config()
        except Exception as e:
            raise AppException(e, sys) from e

    
    def train_original_knn(self):
        """Train original KNN model for backward compatibility"""
        try:
            logging.info("Training original KNN model...")
            
            # Load original pivot data
            book_pivot = pickle.load(open(self.model_trainer_config.transformed_data_file_dir,'rb'))
            book_sparse = csr_matrix(book_pivot)
            
            # Training original model
            model = NearestNeighbors(algorithm='brute')
            model.fit(book_sparse)

            # Save original model
            os.makedirs(self.model_trainer_config.trained_model_dir, exist_ok=True)
            file_name = os.path.join(self.model_trainer_config.trained_model_dir, self.model_trainer_config.trained_model_name)
            pickle.dump(model, open(file_name,'wb'))
            logging.info(f"Saved original KNN model to {file_name}")

        except Exception as e:
            logging.error(f"Error training original KNN model: {e}")
            raise AppException(e, sys) from e

    def train_improved_knn(self):
        """Train improved KNN model with better parameters"""
        try:
            logging.info("Training improved KNN model...")
            
            # Load binary pivot data for better performance
            binary_pivot_path = os.path.join(self.model_trainer_config.transformed_data_file_dir.replace('transformed_data.pkl', 'transformed_data_binary.pkl'))
            
            if os.path.exists(binary_pivot_path):
                book_pivot = pickle.load(open(binary_pivot_path, 'rb'))
                logging.info(f"Using binary pivot table: {book_pivot.shape}")
            else:
                # Fallback to original
                book_pivot = pickle.load(open(self.model_trainer_config.transformed_data_file_dir, 'rb'))
                logging.info(f"Using original pivot table: {book_pivot.shape}")
            
            book_sparse = csr_matrix(book_pivot)
            
            # Improved KNN with cosine similarity
            model = NearestNeighbors(
                algorithm='brute',
                metric='cosine',
                n_neighbors=15  # Increased from default 5
            )
            model.fit(book_sparse)

            # Save improved model
            improved_model_data = {
                'model': model,
                'algorithm': 'Improved_KNN',
                'metric': 'cosine',
                'n_neighbors': 15,
                'book_names': book_pivot.index,
                'user_ids': book_pivot.columns
            }
            
            file_name = os.path.join(self.model_trainer_config.trained_model_dir, 'improved_knn_model.pkl')
            pickle.dump(improved_model_data, open(file_name, 'wb'))
            logging.info(f"Saved improved KNN model to {file_name}")

        except Exception as e:
            logging.error(f"Error training improved KNN model: {e}")
            raise AppException(e, sys) from e

    def train_svd_model(self):
        """Train SVD-based collaborative filtering model"""
        try:
            logging.info("Training SVD model...")
            
            # Load explicit ratings pivot
            explicit_pivot_path = os.path.join(self.model_trainer_config.transformed_data_file_dir.replace('transformed_data.pkl', 'transformed_data_explicit.pkl'))
            
            if os.path.exists(explicit_pivot_path):
                book_pivot = pickle.load(open(explicit_pivot_path, 'rb'))
                logging.info(f"Using explicit pivot table: {book_pivot.shape}")
            else:
                # Fallback to original
                book_pivot = pickle.load(open(self.model_trainer_config.transformed_data_file_dir, 'rb'))
                logging.info(f"Using original pivot table: {book_pivot.shape}")
            
            # Train SVD model
            n_components = min(50, min(book_pivot.shape) - 1)  # Ensure valid components
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            
            user_factors = svd.fit_transform(book_pivot)
            item_factors = svd.components_
            
            # Save SVD model
            svd_model_data = {
                'algorithm': 'SVD',
                'n_components': n_components,
                'user_factors': user_factors,
                'item_factors': item_factors,
                'explained_variance_ratio': svd.explained_variance_ratio_,
                'book_names': book_pivot.index,
                'user_ids': book_pivot.columns
            }
            
            file_name = os.path.join(self.model_trainer_config.trained_model_dir, 'svd_model.pkl')
            pickle.dump(svd_model_data, open(file_name, 'wb'))
            logging.info(f"Saved SVD model to {file_name} (explained variance: {svd.explained_variance_ratio_.sum():.3f})")

        except Exception as e:
            logging.error(f"Error training SVD model: {e}")
            raise AppException(e, sys) from e

    def train_nmf_model(self):
        """Train NMF-based collaborative filtering model"""
        try:
            logging.info("Training NMF model...")
            
            # Load implicit feedback pivot
            implicit_pivot_path = os.path.join(self.model_trainer_config.transformed_data_file_dir.replace('transformed_data.pkl', 'transformed_data_implicit.pkl'))
            
            if os.path.exists(implicit_pivot_path):
                book_pivot = pickle.load(open(implicit_pivot_path, 'rb'))
                logging.info(f"Using implicit pivot table: {book_pivot.shape}")
            else:
                # Fallback to original
                book_pivot = pickle.load(open(self.model_trainer_config.transformed_data_file_dir, 'rb'))
                logging.info(f"Using original pivot table: {book_pivot.shape}")
            
            # Train NMF model
            n_components = min(50, min(book_pivot.shape) - 1)  # Ensure valid components
            nmf = NMF(n_components=n_components, random_state=42, max_iter=200)
            
            user_factors = nmf.fit_transform(book_pivot)
            item_factors = nmf.components_
            
            # Save NMF model
            nmf_model_data = {
                'algorithm': 'NMF',
                'n_components': n_components,
                'user_factors': user_factors,
                'item_factors': item_factors,
                'reconstruction_err': nmf.reconstruction_err_,
                'book_names': book_pivot.index,
                'user_ids': book_pivot.columns
            }
            
            file_name = os.path.join(self.model_trainer_config.trained_model_dir, 'nmf_model.pkl')
            pickle.dump(nmf_model_data, open(file_name, 'wb'))
            logging.info(f"Saved NMF model to {file_name} (reconstruction error: {nmf.reconstruction_err_:.3f})")

        except Exception as e:
            logging.error(f"Error training NMF model: {e}")
            raise AppException(e, sys) from e

    def train_hybrid_model(self):
        """Train hybrid model combining SVD and NMF"""
        try:
            logging.info("Training hybrid model...")
            
            # Load both SVD and NMF models
            svd_path = os.path.join(self.model_trainer_config.trained_model_dir, 'svd_model.pkl')
            nmf_path = os.path.join(self.model_trainer_config.trained_model_dir, 'nmf_model.pkl')
            
            if os.path.exists(svd_path) and os.path.exists(nmf_path):
                with open(svd_path, 'rb') as f:
                    svd_model = pickle.load(f)
                with open(nmf_path, 'rb') as f:
                    nmf_model = pickle.load(f)
                
                # Create hybrid model
                hybrid_model_data = {
                    'algorithm': 'Hybrid_SVD_NMF',
                    'svd_model': svd_model,
                    'nmf_model': nmf_model,
                    'book_names': svd_model['book_names'],
                    'user_ids': svd_model['user_ids']
                }
                
                file_name = os.path.join(self.model_trainer_config.trained_model_dir, 'hybrid_model.pkl')
                pickle.dump(hybrid_model_data, open(file_name, 'wb'))
                logging.info(f"Saved hybrid model to {file_name}")
            else:
                logging.warning("SVD or NMF model not found, skipping hybrid model")

        except Exception as e:
            logging.error(f"Error training hybrid model: {e}")
            raise AppException(e, sys) from e

    def train(self):
        """Train all models"""
        try:
            logging.info("Starting comprehensive model training...")
            
            # Train all models
            self.train_original_knn()
            self.train_improved_knn()
            self.train_svd_model()
            self.train_nmf_model()
            self.train_hybrid_model()
            
            logging.info("All models trained successfully!")

        except Exception as e:
            raise AppException(e, sys) from e

    def initiate_model_trainer(self):
        try:
            logging.info(f"{'='*20}Model Trainer log started.{'='*20} ")
            self.train()
            logging.info(f"{'='*20}Model Trainer log completed.{'='*20} \n\n")
        except Exception as e:
            raise AppException(e, sys) from e