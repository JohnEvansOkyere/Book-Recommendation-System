from collections import namedtuple

DataIngestionConfig = namedtuple("DatasetConfig", ["dataset_download_url",
                                                   "raw_data_dir",
                                                   "ingested_dir"])



DataValidationConfig = namedtuple("DatavalidationConfig", ["clean_data_dir",
                                                         "serialized_objects_dir",
                                                         "books_csv_file",
                                                         "ratings_csv_file"])


DataTransformationConfig = namedtuple("DataTransformationConfig", ["transformed_data_dir",
                                                                 "clean_data_file_path"])   



ModelTrainerConfig = namedtuple("ModelTrainerConfig", ["transformed_data_file_dir",
                                                      "trained_model_dir",
                                                      "trained_model_name"])



# ModelRecommendationConfig = namedtuple("ModelRecommendationConfig", ["book_name_serialized_objects",
#                                                       "book_pivot_serialized_objects",
#                                                       "final_rating_serialized_objects",
#                                                       "trained_model_path"])