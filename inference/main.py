import base64
import json

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from predict import InferenceSeg
import os

app = FastAPI()
detector = InferenceSeg(100)


class Predict(BaseModel):
    image: str
    labeled_image: bool = False

class ModelPath(BaseModel):
    model_path: str


@app.post('/set_model/')
def set_model(model_path: ModelPath):
    try:
        detector.load_model(model_path.model_path)
        return {'status': 'success'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


@app.post('/predict/')
def predict(predict: Predict):
    print('Predicting...')
    try:
        image = cv2.imdecode(np.frombuffer(base64.b64decode(predict.image.encode('utf-8')), dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        image_ma = np.moveaxis(image, 2, 0)
        result = detector.predict(image_ma, batch_size=1)
        result_poly = detector.result_to_polygon(result)
        if predict.labeled_image:
            image = detector.predict_image_path_add_image(image, mask=result)
            _, img_encoded = cv2.imencode('.jpg', image)
            result_labeled = {}
            result_labeled['labeled_image'] = base64.b64encode(img_encoded).decode('utf-8')
            result_labeled['result'] = result_poly
            return json.dumps(result_labeled)

        result = json.dumps(result_poly)
        return result

    except Exception as e:
        return {"result": "failed", 'error': str(e)}

# if __name__ == '__main__':
#     uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', '5556')))
