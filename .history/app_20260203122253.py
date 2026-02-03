from flask import Flask, jsonify,render_template,request
from flask_cors import CORS
import os 
from src.Chest_Cancer_Classification.pipeline.prediction import predictionPipeline
from src.Chest_Cancer_Classification.utils.common import decodeImage

os.putenv('LANG',)

app = Flask(__name__)
CORS(app)