import base64
import pickle
import cv2
import numpy as np

def im2json(image):
    """Convert a Numpy array to JSON string"""
    imdata = pickle.dumps(image)
    return base64.b64encode(imdata).decode('ascii')


def str_to_numpy_image(image: str):
    """Convert base64 image to Numpy array"""
    image = cv2.imdecode(np.frombuffer(base64.b64decode(image.encode('utf-8')), dtype=np.uint8), cv2.IMREAD_COLOR)
    return image
