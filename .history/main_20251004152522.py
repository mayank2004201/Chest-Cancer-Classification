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

try:
    logger.info(f"*****************")
    logger.info(f">>>>>>> Stage {Stage_name} started <<<<<<<<<")
    prepare_base_model = PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()
    logger.info(f">>>>>>> Stage {Stage_name} completed <<<<<<<<\n\nx================x")
except Exception as e:
    logger.exception(e)
    raise e


Stage_name = "Prepare base model"

try:
    logger.info(f"*****************")
    logger.info(f">>>>>>> Stage {Stage_name} started <<<<<<<<<")
    prepare_base_model = PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()
    logger.info(f">>>>>>> Stage {Stage_name} completed <<<<<<<<\n\nx================x")
except Exception as e:
    logger.exception(e)
    raise e
