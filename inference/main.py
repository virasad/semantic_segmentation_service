import json

import cv2
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from inference.predict import InferenceSeg

app = FastAPI()
detector = InferenceSeg(100)


class Predict(BaseModel):
    image_path: str


class modelPath(BaseModel):
    modelPath: str


@app.post('/set_model/')
def set_model(model_path: modelPath):
    try:
        detector.load_model(model_path.modelPath)
        return {'status': 'success'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


@app.post('/predict/')
def predict(predict: Predict = None):
    try:
        image = cv2.imread(predict.image_path, 1)
        image = np.moveaxis(image, 2, 0)
        result = detector.predict(image, batch_size=1)
        result = detector.result_to_polygon(result)
        df = pd.DataFrame(result)
        parsed = json.loads(df.to_json())
        return parsed

    except Exception as e:
        return {"result": "failed", 'error': str(e)}



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
