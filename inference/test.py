import cv2
import requests
import numpy as np
import json
import base64
import pickle



def json2im(jstr):
    """Convert a JSON string back to a Numpy array"""
    imdata = base64.b64decode(jstr['image'])
    im = pickle.loads(imdata)
    return im

status = requests.post('http://127.0.0.1:8000/set_model/',
              json={'model_path': '/home/amir/Projects/rutilea_singularity_gear_inspection/backbone/coco_train_model.pt'})
print(status.json())

# status = requests.post('http://127.0.0.1:8000/set_model/',
#               json={'model_path': '/home/amir/Projects/rutilea_singularity_gear_inspection/backbone/coco_train_model.pt'},headers={'accept: application/json'})
# print(status.json())
# response = requests.post ('http://127.0.0.1:8000/predict/', 
#                 json={'image_path': '/home/amir/Projects/rutilea_singularity_gear_inspection/backbone/dataset/coco/pngimages/0acfd711-WIN_20211218_14_25_49_Pro.png'})
# print(response.json()['image'])
# image = json2im(response.json())
# cv2.imshow('test',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# response = response.json()
# print(response)
# 
# nparr = np.fromstring(response.data, np.uint8)
#     # decode image
# img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)