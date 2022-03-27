import base64

import cv2
import requests


def set_model(url, dir):
    data = {"model_path": dir}
    r = requests.post(url, json=data)
    return r.text


def predict_image(url, image):
    data = {"image": image}
    r = requests.post(url, json=data)
    return r.json()

def get_image(image_path):
    image = cv2.imread(image_path)
    _, img_encoded = cv2.imencode('.jpg', image)
    return base64.b64encode(img_encoded).decode('utf-8')

if __name__ == '__main__':
    url = 'http://127.0.0.1:8000'
    model_dir = 'https://flash-weights.s3.amazonaws.com/0.7.0/semantic_segmentation_model.pt'

    img_str = get_image('road.png')

    set_model(url=url + '/set_model/', dir=model_dir)
    r = predict_image(url=url + '/predict/', image=img_str)
    print(r)
