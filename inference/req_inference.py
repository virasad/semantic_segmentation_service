import base64

import cv2
import os
import requests
import numpy as np
from tqdm import tqdm
import shutil
import json


def set_model(url, dir):
    data = {"model_path": dir}
    r = requests.post(url, json=data)
    return r.text


def predict_image(url, image, return_image=False):
    data = {"image": image, 'labeled_image': return_image}
    r = requests.post(url, json=data)
    return r.json()


def get_image(image_path):
    image = cv2.imread(image_path)
    _, img_encoded = cv2.imencode('.jpg', image)
    return base64.b64encode(img_encoded).decode('utf-8')


if __name__ == '__main__':
    url = 'http://0.0.0.0:5553'
    model_dir = '/weights/burrsegment200_model.pt'

    set_model(url=url + '/set_model/', dir=model_dir)

    images_dir = '/dataset/images'
    images_list = os.listdir(images_dir)
    images_list.sort()
    dst_dir = '/dataset/out_images'
    try:
        os.mkdir(dst_dir)
    except FileExistsError:
        shutil.rmtree(dst_dir)
        os.mkdir(dst_dir)

    for img_dir in tqdm(images_list):
        img_str = get_image(os.path.join(images_dir, img_dir))
        r = predict_image(url=url + '/predict/', image=img_str, return_image=True)
        d = json.loads(r)
        image = cv2.imdecode(np.frombuffer(base64.b64decode(d['labeled_image'].encode('utf-8')), dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        cv2.imwrite(os.path.join(dst_dir, img_dir), image)
