"""Prediction pipeline for kidney disease classification using CNN."""
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


class PredictionPipeline:
    """Pipeline for running inference on kidney CT scan images."""

    def __init__(self,filename):
        """
        Initialize prediction pipeline.

        Args:
            filename (str): Path to the input image file.
        """
        self.filename =filename


    
    def predict(self):
        """
        Run prediction on the input image using the trained CNN model.

        Returns:
            List[Dict[str, str]]: Prediction result with label ('Tumor' or 'Normal').
        """
        # load model
        model = load_model(os.path.join("model", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        if result[0] == 1:
            prediction = 'Tumor'
            return [{ "image" : prediction}]
        else:
            prediction = 'Normal'
            return [{ "image" : prediction}]