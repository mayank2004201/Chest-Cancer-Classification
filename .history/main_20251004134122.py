from src.Chest_Cancer_Classification import logger
from src.Chest_Cancer_Classification.pipeline.stage_1_data_ingestion import DataIngestionTrainingPipeline
from src.Chest_Cancer_Classification.pipeline.stage_2_prepare_base_model import PrepareBaseModelTrainingPipeline

Stage_name = "Data Ingestion Stage"


try:
    logger.info(f">>>>>>> Stage {Stage_name} started <<<<<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>> Stage {Stage_name} completed <<<<<<<<\n\nx================x")
except Exception as e:
    logger.exception(e)
    raise e

Stage_name = "Prepare base model"

class PrepareBaseModelTrainingPipeline: 
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config=config.get_prepare_base_model_config()
        prepare_base_model=PrepareBaseModel (config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()
        
if __name__ == '__main__':
    try:
        logger.info(f"*****************")
        logger.info(f">>>>>>> Stage {Stage_name} started <<<<<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>> Stage {Stage_name} completed <<<<<<<<\n\nx================x")
    except Exception as e:
        logger.exception(e)
        raise e
    