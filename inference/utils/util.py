import base64
import pickle
import cv2
import numpy as np
from flash.core.data.io.input import DataKeys
import torch

def im2json(image):
    """Convert a Numpy array to JSON string"""
    imdata = pickle.dumps(image)
    return base64.b64encode(imdata).decode('ascii')


def str_to_numpy_image(image: str):
    """Convert base64 image to Numpy array"""
    image = cv2.imdecode(np.frombuffer(base64.b64decode(image.encode('utf-8')), dtype=np.uint8), cv2.IMREAD_COLOR)
    return image

def make_masked_image_from_labelmap(result, image, label_map_path):
    label_map = open(label_map_path, "r")
    labelmaps = label_map.readlines()
    label_map.close()
    labelmaps = [x.strip() for x in labelmaps]
    class_color = []
    for idx, labelmap in enumerate(labelmaps[1:]):
        class_color.append(labelmap.split(":")[1])

    preds = result[DataKeys.PREDS]
    labels = torch.argmax(preds, dim=-3)  # HxW
    labels = np.array(labels)


    mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for idx, color in enumerate(class_color):
        color = color.split(",")
        color = [int(x) for x in color]
        mask[np.where(labels == idx)] = color

    # put mask on images
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    merge_img = cv2.addWeighted(image, 0.5, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR), 0.5, 0, image)
    merge_img = cv2.cvtColor(merge_img, cv2.COLOR_BGR2RGB)
    return merge_img


