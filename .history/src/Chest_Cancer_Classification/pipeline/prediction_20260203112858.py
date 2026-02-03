import numpy as np 
from tensorflow.keras.models import load_model
from tensorflow.preprocessing.image import image
import os 

class prediction:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = load_model(self.model_path)
        
    def predict(self, image_path: str):
        img = image.load_img(image_path, target_size=(224,224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        
        return predicted_class