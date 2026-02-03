import numpy as np 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os 

class predictionPipeline:
    def __init__(self,filename):
        self.filename = filename
        
    def predict(self):
        model = load_model(os.path.join("model","model.h5"))
        
        image_name = self.filename
        test_image = image.load_img(image_name,target_size=(224,224))
        test_image = image.img_to_array(test_image) 
        test_image = np.expand_dims(test_image,axis=0)
    
        predictions = model.predict(test_image)         
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        
        class_names = [
            "adenocarcinoma",          
            "large cell carcinoma",    
            "normal",                  
            "squamous cell carcinoma"  
        ]

        
        prediction = class_names[predicted_class_idx]

        return [{
            "image": prediction,
        }]
        
        
        
        