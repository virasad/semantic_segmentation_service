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


@app.post('/set_model/')
def set_model(model_path: str = None):
    try:
        detector.load_model(model_path)
        return {'status': 'success'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


@app.post("/train/")
def read_train(train: Train = None):
    try:
        result = tr.train_from_coco(train.images, train.annotation, train.save_name, train.batch_size,
                                    train.num_dataloader_workers, train.epochs)
        return result

    except Exception as e:
        return {"result": "failed", 'error': str(e)}
