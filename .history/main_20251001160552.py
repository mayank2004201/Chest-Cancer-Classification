from Chest_Cancer_Classification import logger
from Chest_Cancer_Classification.pipeline.stage_1_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "Data Ingetion Stage"


try:
    logger.info(f">>>>>>> Stage {Stage_name} started <<<<<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>> Stage {Stage_name} completed <<<<<<<<\n\nx================x")
except Exception as e:
    logger.exception(e)
    raise e
