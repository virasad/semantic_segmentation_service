from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
import train as tr
import os
import requests
import uvicorn

app = FastAPI()


class Train(BaseModel):
    images: str
    annotation: str
    save_name: Optional[str] = None
    batch_size: Optional[int] = None
    num_dataloader_workers: Optional[int] = None
    epochs: Optional[int] = None


@app.post("/train/")
def read_train(train: Train = None):
    try:
        result = tr.train_from_coco(train.images, train.annotation, train.save_name, train.batch_size,
                                    train.num_dataloader_workers, train.epochs)

        response_url = os.environ.get('RESPONSE_URL')
        requests.post(response_url, json={'result': result})
        return result

    except Exception as e:
        return {"result": "failed", 'error': str(e)}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5000)