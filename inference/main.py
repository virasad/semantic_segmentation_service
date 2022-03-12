from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
import train as tr
from inference.predict import InferenceSeg
import base64
import cv2
import numpy as np
import pickle




app = FastAPI()
detector = None
detector = InferenceSeg()

class Train(BaseModel):
    images: str
    annotation: str
    save_name: Optional[str] = None
    batch_size: Optional[int] = None
    num_dataloader_workers: Optional[int] = None
    epochs: Optional[int] = None

class Predict(BaseModel):
    image_path: str


def im2json(image):
    """Convert a Numpy array to JSON string"""
    imdata = pickle.dumps(image)
    return base64.b64encode(imdata).decode('ascii')

@app.post('/set_model/')
def set_model(model_path: str = None):
    try:
        detector.load_model(model_path)
        return {'status': 'success'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


@app.post('/predict/')
def predict(predict: Predict = None):
    try:
        result = detector.detect_add_to_image(predict.image_path)
        img_str = im2json(result)
        print(img_str)
        return {"result": "success", 'image': img_str}

    except Exception as e:
        return {"result": "failed", 'error': str(e)}
