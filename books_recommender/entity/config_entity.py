from collections import namedtuple

DataIngestionConfig = namedtuple("DatasetConfig", ["dataset_download_url",
                                                   "raw_data_dir",
                                                   "ingested_dir"])



DataValidationConfig = namedtuple("DatavalidationConfig", ["clean_data_dir",
                                                         "serialized_objects_dir",
                                                         "books_csv_file",
                                                         "ratings_csv_file"])