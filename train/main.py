import os
from typing import Optional

import requests
from fastapi import FastAPI
from pydantic import BaseModel
import shutil
import trainer as tr

app = FastAPI()

train_models = {
    "backbone": "mobilenet_v2",
    "head": "deeplabv3plus"
}

class Train(BaseModel):
    images: str
    annotation: str
    data_type : str
    labelmap : Optional[str] = None
    save_name: Optional[str] = None
    batch_size: Optional[str] = None
    extra_kwargs: Optional[dict] = None
    num_dataloader_workers: Optional[str] = None
    epochs: Optional[str] = None
    num_classes: str = 2
    validation_split: str = 0.2
    pretrained_path: Optional[str] = None
    is_augment: Optional[bool] = None
    augment_params: Optional[dict] = None
    logger: Optional[str] = None

class SetModel(BaseModel):
    backbone: str
    head: str



@app.post("/train")
def read_train(train: Train = None):
    try:
        trainer = tr.SemanticSegmentTrainer(backbone = train_models["backbone"],
                             head = train_models["head"],
                             data_type=train.data_type,
                             pre_trained_path = train.pretrained_path,
                             is_augment = train.is_augment,
                             augment_params = train.augment_params,
                             label_map=train.labelmap,
                             logger=train.logger
                             )
        result = trainer.train(train.images, train.annotation, train.save_name, int(train.batch_size),
                                    int(train.num_dataloader_workers), int(train.epochs), int(train.num_classes),
                                    float(train.validation_split))
        response_url = os.environ.get('RESPONSE_URL', 'http://127.0.0.1:8000/api/v1/train/done')
        a = requests.post(response_url, data={**result, **train.extra_kwargs, 'save_name': train.save_name +'_model.pt'})
        print(a.text)
        shutil.rmtree("/dataset/temp")
        return result

    except Exception as e:
        shutil.rmtree("/dataset/temp")
        return {"result": "failed", 'error': str(e)}


# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=int(os.environ.get('PORT', '5554')))
