from src.Chest_Cancer_Classification import logger
from src.Chest_Cancer_Classification.pipeline.stage_1_data_ingestion import DataIngestionTrainingPipeline
from src.Chest_Cancer_Classification.pipeline.stage_2_prepare_base_model import PrepareBaseModelTrainingPipeline
from src.Chest_Cancer_Classification.pipeline.stage_3_model_trainer import ModelTrainingPipeline
from src.Chest_Cancer_Classification.pipeline.stage_4_model_evaluation import EvaluationPipeline

import os
os.environ["MLFLOW_DISABLE_GIT_COMMIT_TRACKING"] = "true"

from dotenv import load_dotenv
load_dotenv()

print("MLFLOW_TRACKING_URI from env:", os.getenv("MLFLOW_TRACKING_URI"))
import mlflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
os.chdir(r"C:\Users\Mayank Goel\OneDrive\Desktop\Chest Cancer Classification Using ML Flow")

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


Stage_name = "Training"

try:
    logger.info(f"*****************")
    logger.info(f">>>>>>> Stage {Stage_name} started <<<<<<<<<")
    model_trainer = ModelTrainingPipeline()
    model_trainer.main()
    logger.info(f">>>>>>> Stage {Stage_name} completed <<<<<<<<\n\nx================x")
except Exception as e:
    logger.exception(e)
    raise e


Stage_name = "Evaluation"

try:
    logger.info(f"*****************")
    logger.info(f">>>>>>> Stage {Stage_name} started <<<<<<<<<")
    model_evaluation = EvaluationPipeline()
    model_evaluation.main()
    logger.info(f">>>>>>> Stage {Stage_name} completed <<<<<<<<\n\nx================x")
except Exception as e:
    logger.exception(e)
    raise e
