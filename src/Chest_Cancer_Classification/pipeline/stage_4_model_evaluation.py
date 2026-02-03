from Chest_Cancer_Classification.config.configuration import ConfigurationManager
from Chest_Cancer_Classification.components.model_evaluation_mlflow import Evaluation
from Chest_Cancer_Classification import logger


Stage_name = 'Evaluation'



class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        evaluation_config = config.get_evaluation_config()
        evaluation = Evaluation(evaluation_config)
        evaluation.evaluation()
        evaluation.save_score()
        evaluation.log_into_mlflow()
        

if __name__ == '__main__':
    try:
        logger.info(f"*****************")
        logger.info(f">>>>>>> Stage {Stage_name} started <<<<<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>>> Stage {Stage_name} completed <<<<<<<<\n\nx================x")
    except Exception as e:
        logger.exception(e)
        raise e
    
