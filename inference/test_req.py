import requests
import json
import pprint
import cv2
from pathlib import Path
import base64
import flash


TEST_IMAGE='/home/amir/anaconda3/envs/deep-gpu/lib/python3.9/site-packages/flash/assets/road.png'
DETECTION_URL='http://127.0.0.1:8000/predict'
def send_with_image(image_path,detection_url):
    image_data = open(image_path, "rb").read()
    response = requests.post(detection_url, files={"image": image_data}).json()
    pprint.pprint(response)
    return response

def send_cv2_image(cv2image, detection_url):
    cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
    img_bytes = cv2.imencode('.jpg', cv2image)[1].tobytes()
    response = requests.post(detection_url, files={"image": img_bytes}).json()
    pprint.pprint(response)

def send_for_flash(image_path= None,detection_url=None):
    print(flash.ASSETS_ROOT)
    # cv2image = cv2.imread()

    with (Path(flash.ASSETS_ROOT) / "road.png").open("rb") as f:
        # Show
        imgstr = base64.b64encode(f.read()).decode("UTF-8")
    # print(imgstr)
    body = {"session": "UUID", "payload": {"inputs": {"data": imgstr}}}
    resp = requests.post("http://127.0.0.1:8000/predict", json=body)
    print(resp.json())
    # resp = requests.post("http://127.0.0.1:8000/predict", json=body)
if __name__ == '__main__':
    send_for_flash()
    # cv2image = cv2.imread(TEST_IMAGE)
    # cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
    # response = send_with_image(TEST_IMAGE, DETECTION_URL)
    # response = send_cv2_image(cv2image, DETECTION_URL)