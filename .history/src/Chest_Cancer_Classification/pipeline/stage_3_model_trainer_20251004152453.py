from src.Chest_Cancer_Classification.config.configuration import ConfigurationManager
from src.Chest_Cancer_Classification.components.model_trainer import Training
from src.Chest_Cancer_Classification import logger

Stage_name = "Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()
        
