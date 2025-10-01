from Chest_Cancer_Classification.config.configuration import ConfigurationManager
from Chest_Cancer_Classification.components.data_ingestion import DataIngestion
from Chest_Cancer_Classification import logger


Stage_name = "Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __inti__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()