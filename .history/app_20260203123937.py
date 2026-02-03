from flask import Flask, jsonify,render_template,request
from flask_cors import CORS,cross_origin
import os 
from src.Chest_Cancer_Classification.pipeline.prediction import predictionPipeline
from src.Chest_Cancer_Classification.utils.common import decodeImage

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = 'input_image.jpg'
        self.classifier = predictionPipeline(self.filename)
        

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/train",methods=['GET','POST'])
@cross_origin()
def train():
    os.system("python main.py")
    return "Training done successfully!!"

@app.route("/predict", methods=['POST'])

    
