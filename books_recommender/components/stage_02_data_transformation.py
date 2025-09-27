import os
import sys
import pickle
import pandas as pd
from books_recommender.logger.log import logging
from books_recommender.config.configuration import AppConfiguration
from books_recommender.exception.exception_handler import AppException



class DataTransformation:
    def __init__(self, app_config = AppConfiguration()):
        try:
            self.data_transformation_config = app_config.get_data_transformation_config()
            self.data_validation_config= app_config.get_data_validation_config()
        except Exception as e:
            raise AppException(e, sys) from e


    
    def get_data_transformer(self):
        try:
            # IMPROVEMENT: Create multiple pivot tables for different algorithms
            logging.info("Creating multiple pivot tables for different algorithms...")
            
            # Load all datasets
            df_original = pd.read_csv(self.data_transformation_config.clean_data_file_path)
            df_explicit = pd.read_csv(os.path.join(self.data_validation_config.clean_data_dir, 'clean_data_explicit.csv'))
            df_implicit = pd.read_csv(os.path.join(self.data_validation_config.clean_data_dir, 'clean_data_implicit.csv'))
            df_binary = pd.read_csv(os.path.join(self.data_validation_config.clean_data_dir, 'clean_data_binary.csv'))
            
            logging.info(f"Original dataset shape: {df_original.shape}")
            logging.info(f"Explicit dataset shape: {df_explicit.shape}")
            logging.info(f"Implicit dataset shape: {df_implicit.shape}")
            logging.info(f"Binary dataset shape: {df_binary.shape}")
            
            # Create pivot tables for different algorithms
            pivot_tables = {}
            
            # 1. Original pivot table (for KNN)
            book_pivot_original = df_original.pivot_table(columns='user_id', index='title', values='rating')
            book_pivot_original.fillna(0, inplace=True)
            pivot_tables['original'] = book_pivot_original
            logging.info(f"Original pivot table shape: {book_pivot_original.shape}")
            
            # 2. Explicit ratings pivot (for SVD)
            if len(df_explicit) > 0:
                book_pivot_explicit = df_explicit.pivot_table(columns='user_id', index='title', values='rating')
                book_pivot_explicit.fillna(0, inplace=True)
                pivot_tables['explicit'] = book_pivot_explicit
                logging.info(f"Explicit pivot table shape: {book_pivot_explicit.shape}")
            
            # 3. Implicit feedback pivot (for NMF)
            if len(df_implicit) > 0:
                book_pivot_implicit = df_implicit.pivot_table(columns='user_id', index='title', values='implicit_rating')
                book_pivot_implicit.fillna(0, inplace=True)
                pivot_tables['implicit'] = book_pivot_implicit
                logging.info(f"Implicit pivot table shape: {book_pivot_implicit.shape}")
            
            # 4. Binary ratings pivot (for improved KNN)
            if len(df_binary) > 0:
                book_pivot_binary = df_binary.pivot_table(columns='user_id', index='title', values='binary_rating')
                book_pivot_binary.fillna(0, inplace=True)
                pivot_tables['binary'] = book_pivot_binary
                logging.info(f"Binary pivot table shape: {book_pivot_binary.shape}")

            # Save all pivot tables
            os.makedirs(self.data_transformation_config.transformed_data_dir, exist_ok=True)
            
            for pivot_name, pivot_table in pivot_tables.items():
                pickle.dump(pivot_table, open(os.path.join(self.data_transformation_config.transformed_data_dir, f"transformed_data_{pivot_name}.pkl"), 'wb'))
                logging.info(f"Saved {pivot_name} pivot table data to {self.data_transformation_config.transformed_data_dir}")

            # Keep original for backward compatibility
            pickle.dump(book_pivot_original, open(os.path.join(self.data_transformation_config.transformed_data_dir, "transformed_data.pkl"), 'wb'))
            logging.info(f"Saved original pivot table data to {self.data_transformation_config.transformed_data_dir}")

            # Save book names and pivot tables for web app
            os.makedirs(self.data_validation_config.serialized_objects_dir, exist_ok=True)
            
            # Save book names
            book_names = book_pivot_original.index
            pickle.dump(book_names, open(os.path.join(self.data_validation_config.serialized_objects_dir, "book_names.pkl"), 'wb'))
            logging.info(f"Saved book_names serialization object to {self.data_validation_config.serialized_objects_dir}")

            # Save all pivot tables for web app
            for pivot_name, pivot_table in pivot_tables.items():
                pickle.dump(pivot_table, open(os.path.join(self.data_validation_config.serialized_objects_dir, f"book_pivot_{pivot_name}.pkl"), 'wb'))
                logging.info(f"Saved book_pivot_{pivot_name} serialization object to {self.data_validation_config.serialized_objects_dir}")
            
            # Keep original for backward compatibility
            pickle.dump(book_pivot_original, open(os.path.join(self.data_validation_config.serialized_objects_dir, "book_pivot.pkl"), 'wb'))
            logging.info(f"Saved book_pivot serialization object to {self.data_validation_config.serialized_objects_dir}")

        except Exception as e:
            raise AppException(e, sys) from e

    

    def initiate_data_transformation(self):
        try:
            logging.info(f"{'='*20}Data Transformation log started.{'='*20} ")
            self.get_data_transformer()
            logging.info(f"{'='*20}Data Transformation log completed.{'='*20} \n\n")
        except Exception as e:
            raise AppException(e, sys) from e