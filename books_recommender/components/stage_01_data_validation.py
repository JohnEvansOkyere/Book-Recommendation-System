import os
import sys
import ast 
import pandas as pd
import pickle
from books_recommender.logger.log import logging
from books_recommender.config.configuration import AppConfiguration
from books_recommender.exception.exception_handler import AppException



class DataValidation:
    def __init__(self, app_config = AppConfiguration()):
        try:
            self.data_validation_config= app_config.get_data_validation_config()
        except Exception as e:
            raise AppException(e, sys) from e


    
    def preprocess_data(self):
        try:
            
            ratings = pd.read_csv(self.data_validation_config.ratings_csv_file, sep=";", on_bad_lines='skip', encoding='latin-1')
            books = pd.read_csv(self.data_validation_config.books_csv_file, sep=";", on_bad_lines='skip', encoding='latin-1')
            
            logging.info(f" Shape of ratings data file: {ratings.shape}")
            logging.info(f" Shape of books data file: {books.shape}")

            #Here Image URL columns is important for the poster. So, we will keep it
            books = books[['ISBN','Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher','Image-URL-L']]
            # Lets remane some wierd columns name in books
            books.rename(columns={"Book-Title":'title',
                                'Book-Author':'author',
                                "Year-Of-Publication":'year',
                                "Publisher":"publisher",
                                "Image-URL-L":"image_url"},inplace=True)

            
            # Lets remane some wierd columns name in ratings
            ratings.rename(columns={"User-ID":'user_id',
                                'Book-Rating':'rating'},inplace=True)

            # IMPROVEMENT 1: Better user filtering - reduce threshold for more users
            logging.info("Applying improved user filtering...")
            user_rating_counts = ratings['user_id'].value_counts()
            logging.info(f"Original user count: {len(user_rating_counts)}")
            
            # Reduce threshold from 200 to 50 to get more users
            active_users = user_rating_counts[user_rating_counts >= 50].index
            ratings = ratings[ratings['user_id'].isin(active_users)]
            logging.info(f"Users after filtering (>=50 ratings): {len(active_users)}")

            # Now join ratings with books
            ratings_with_books = ratings.merge(books, on='ISBN')
            number_rating = ratings_with_books.groupby('title')['rating'].count().reset_index()
            number_rating.rename(columns={'rating':'num_of_rating'},inplace=True)
            final_rating = ratings_with_books.merge(number_rating, on='title')

            # IMPROVEMENT 2: Better book filtering - reduce threshold for more books
            logging.info("Applying improved book filtering...")
            book_rating_counts = final_rating['title'].value_counts()
            logging.info(f"Original book count: {len(book_rating_counts)}")
            
            # Reduce threshold from 50 to 25 to get more books
            popular_books = book_rating_counts[book_rating_counts >= 25].index
            final_rating = final_rating[final_rating['title'].isin(popular_books)]
            logging.info(f"Books after filtering (>=25 ratings): {len(popular_books)}")

            # IMPROVEMENT 3: Handle zero ratings better
            logging.info("Handling zero ratings...")
            original_size = len(final_rating)
            
            # Create explicit ratings dataset (remove zeros)
            final_rating_explicit = final_rating[final_rating['rating'] > 0].copy()
            logging.info(f"Explicit ratings (rating > 0): {len(final_rating_explicit)}")
            
            # Create implicit feedback dataset (0 = not interested, >0 = interested)
            final_rating_implicit = final_rating.copy()
            final_rating_implicit['implicit_rating'] = (final_rating_implicit['rating'] > 0).astype(int)
            logging.info(f"Implicit feedback created: {len(final_rating_implicit)}")
            
            # Create binary ratings (high vs low)
            final_rating_binary = final_rating[final_rating['rating'] > 0].copy()
            final_rating_binary['binary_rating'] = (final_rating_binary['rating'] >= 7).astype(int)
            logging.info(f"Binary ratings created: {len(final_rating_binary)}")

            # lets drop the duplicates
            final_rating.drop_duplicates(['user_id','title'],inplace=True)
            final_rating_explicit.drop_duplicates(['user_id','title'],inplace=True)
            final_rating_implicit.drop_duplicates(['user_id','title'],inplace=True)
            final_rating_binary.drop_duplicates(['user_id','title'],inplace=True)
            
            logging.info(f" Shape of the final clean dataset: {final_rating.shape}")
            logging.info(f" Shape of explicit ratings dataset: {final_rating_explicit.shape}")
            logging.info(f" Shape of implicit ratings dataset: {final_rating_implicit.shape}")
            logging.info(f" Shape of binary ratings dataset: {final_rating_binary.shape}")
                        
            # Saving the cleaned data for transformation
            os.makedirs(self.data_validation_config.clean_data_dir, exist_ok=True)
            final_rating.to_csv(os.path.join(self.data_validation_config.clean_data_dir,'clean_data.csv'), index = False)
            final_rating_explicit.to_csv(os.path.join(self.data_validation_config.clean_data_dir,'clean_data_explicit.csv'), index = False)
            final_rating_implicit.to_csv(os.path.join(self.data_validation_config.clean_data_dir,'clean_data_implicit.csv'), index = False)
            final_rating_binary.to_csv(os.path.join(self.data_validation_config.clean_data_dir,'clean_data_binary.csv'), index = False)
            
            logging.info(f"Saved cleaned data to {self.data_validation_config.clean_data_dir}")

            #saving final_rating objects for web app
            os.makedirs(self.data_validation_config.serialized_objects_dir, exist_ok=True)
            pickle.dump(final_rating,open(os.path.join(self.data_validation_config.serialized_objects_dir, "final_rating.pkl"),'wb'))
            pickle.dump(final_rating_explicit,open(os.path.join(self.data_validation_config.serialized_objects_dir, "final_rating_explicit.pkl"),'wb'))
            pickle.dump(final_rating_implicit,open(os.path.join(self.data_validation_config.serialized_objects_dir, "final_rating_implicit.pkl"),'wb'))
            pickle.dump(final_rating_binary,open(os.path.join(self.data_validation_config.serialized_objects_dir, "final_rating_binary.pkl"),'wb'))
            
            logging.info(f"Saved final_rating serialization objects to {self.data_validation_config.serialized_objects_dir}")

        except Exception as e:
            raise AppException(e, sys) from e

    
    def initiate_data_validation(self):
        try:
            logging.info(f"{'='*20}Data Validation log started.{'='*20} ")
            self.preprocess_data()
            logging.info(f"{'='*20}Data Validation log completed.{'='*20} \n\n")
        except Exception as e:
            raise AppException(e, sys) from e



    