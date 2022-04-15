import os
from typing import Optional

import requests
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

import trainer as tr

app = FastAPI()


class Train(BaseModel):
    images: str
    annotation: str
    save_name: Optional[str] = None
    batch_size: Optional[str] = None
    extra_kwargs: Optional[dict] = None
    num_dataloader_workers: Optional[str] = None
    epochs: Optional[str] = None
    num_classes: str = 2
    validation_split: str = 0.2


@app.post("/train/")
def read_train(train: Train = None):
    try:
        result = tr.train_from_coco(train.images, train.annotation, train.save_name, int(train.batch_size),
                                    int(train.num_dataloader_workers), int(train.epochs), int(train.num_classes),
                                    float(train.validation_split))
        response_url = os.environ.get('RESPONSE_URL','http://127.0.0.1:8000/api/v1/train/done')
        a = requests.post(response_url, data={**result, **train.extra_kwargs, 'save_name': train.save_name +'_model.pt'})
        print(a.text)
        return result

    except Exception as e:
        return {"result": "failed", 'error': str(e)}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', '5554')))
