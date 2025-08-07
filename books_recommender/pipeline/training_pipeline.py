from books_recommender.components.stage_00_data_ingestion import DataIngestion 
from books_recommender.components.stage_01_data_validation import DataValidation
from books_recommender.components.stage_02_data_transformation import DataTransformation


class TrainingPipeline:
    """
    TrainingPipeline class to manage the training process
    """
    
    def __init__(self):
        """
        Initialize the TrainingPipeline with a DataIngestion instance
        """
        self.data_ingestion = DataIngestion()
        self.data_validation = DataValidation()
        self.data_transformation = DataTransformation()

    def start_training_pipeline(self):
        """
        Start the training pipeline by downloading and extracting data
        returns None
        """
        self.data_ingestion.initiate_data_ingestion()
        self.data_validation.initiate_data_validation()
        self.data_transformation.initiate_data_transformation()

   