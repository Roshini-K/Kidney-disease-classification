"""Flask application for kidney disease classification using CNN."""

from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline



os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class ClientApp:
    """Client wrapper for prediction pipeline."""

    def __init__(self):
        """Initialize client app with default input image and prediction pipeline."""
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

# Initialize at import time so Gunicorn workers have it
clApp = ClientApp()

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    """Render the home page."""
    return render_template('index.html')




@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    """
    Trigger model training pipeline.
    Uses `main.py` to execute all stages.
    """
    os.system("python main.py")
    # os.system("dvc repro")
    return "Training done successfully!"



@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    """
    Handle prediction requests:
    - Decode the base64 image
    - Run prediction pipeline
    - Return results as JSON
    """
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predict()
    return jsonify(result)


if __name__ == "__main__":
    clApp = ClientApp()

    app.run(host='0.0.0.0', port=8080) #for AWS


