from Chest_Cancer_Classification.config.configuration import ConfigurationManager
from Chest_Cancer_Classification.components.data_ingestion import DataIngestion
from Chest_Cancer_Classification import logger


Stage_name = "Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __inti__(self):
        