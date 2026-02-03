import numpy as np 
from tensorflow.keras.models import load_model
from tensorflow.preprocessing.image import image
import os 

class predictionPipeline:
    def __init__(self,filename):
        self.filename = filename
        
    def predict(self):
        model = load_model(os.path.join("artifacts","training","model.h5"))
        
        image_name = self.filename
        test_image = image.load_img(image_name,target_size=(224,224))
        test_image = image.img_to_array(test_image) 
        test_image = np.expand_dims(test_image,axis=0)
        # result = np.argmax(model.predict(test_image),axis=1)
        # print(result)        
        predictions = model.predict(test_image)         
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        

        # Define class names – IMPORTANT: must match the order used during training!
        class_names = [
            "adenocarcinoma",          # class 0
            "large cell carcinoma",    # class 1
            "normal",                  # class 2
            "squamous cell carcinoma"  # class 3
        ]

        # You can also use a more readable / clinical style:
        # class_names = ["Adenocarcinoma", "Large Cell Carcinoma", "Normal", "Squamous Cell Carcinoma"]

        prediction = class_names[predicted_class_idx]


        # Return format similar to your original code
        return [{
            "image": prediction,
            "predicted_class_index": int(predicted_class_idx)
        }]
        
        
        
        
        
        
        
        
        
        # if result[0] == 1:
        #     prediction = 'Normal'
        #     return [{"image" : prediction}]
        
        # else:
        #     prediction = 'Adino Carcinoma'
        #     return [{"image" : prediction}]
        
        
        
        
        #     prediction = 'Normal'
        #     return [{"image" : prediction}]
        # elif result[0] == 2:
        #     prediction = 'Large Cell Carcinoma'
        #     return [{"image" : prediction}]
        # elif result[0] == 3:
        #     prediction = 'Squamous Cell Carcinoma'  
        #     return [{"image" : prediction}]
    