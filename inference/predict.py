import warnings

import cv2
import imantics
import numpy as np
from PIL import Image
from flash import Trainer
from flash.image import SemanticSegmentation, SemanticSegmentationData

from utils.postprocess import SegmentationLabelsOutput
import torch

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*Default upsampling behavior when*")


class InferenceSeg:
    def __init__(self, num_classes):
        self.trainer = Trainer(gpus=torch.cuda.device_count())
        self.num_classes = num_classes
        self.label_map = dict(zip(range(2), [(0, 0, 0), (0, 0, 255)]))

    def set_label_map(self, label_map):
        self.label_map = label_map

    def load_model(self, model_path):
        print('label map : : ', dict(zip(range(2), [(0, 0, 0), (0, 0, 255)])))
        self.model = SemanticSegmentation.load_from_checkpoint(model_path)
        # self.model.output = SegmentationLabelsOutput(visualize=True, labels_map=self.label_map)

    def predict(self, images, batch_size):
        """
        :param images: It should be a list of image paths or a list of numpy arrays
        :param batch_size:
        :return:

        """
        # check type of images
        if not isinstance(images, list):
            images = [images]

        # check type of images
        if isinstance(images[0], str):
            images = [np.array(Image.open(image)) for image in images]
            # change color channel to first dimension
            images = [np.moveaxis(image, 2, 0) for image in images]
            # images = [np.array(image) for image in images]
            # images = [cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB) for image in images]
        datamodule = SemanticSegmentationData.from_numpy(predict_data=images,
                                                         batch_size=batch_size,
                                                         num_classes=self.num_classes)
        predictions = self.trainer.predict(self.model, datamodule=datamodule)
        return predictions[0][0]

    def predict_image_path_add_image(self, image_path):
        """
        :param image_path:
        :return: BGR Image
        """
        mask = self.predict(image_path, 1)
        label_map = SegmentationLabelsOutput.create_random_labels_map(num_classes=100)
        label_output = SegmentationLabelsOutput(visualize=True, labels_map=label_map)
        image = cv2.imread(image_path)
        merge_img = cv2.addWeighted(image, 0.5, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR), 0.5, 0, image)
        return merge_img

    @staticmethod
    def result_to_polygon(result, num_classes: int = 100):
        label_map = SegmentationLabelsOutput.create_random_labels_map(num_classes=num_classes)
        label_output = SegmentationLabelsOutput(visualize=False, labels_map=label_map, return_mask_as_image=False,
                                                labels_class=True)
        m, classes = label_output.transform(result)

        re = []

        for i in range(len(m)):
            ia = imantics.Mask(m[i])
            po = list(ia.polygons())
            re.append({"class_id": classes[i], "polygon": po[0].tolist()})

        return re


if __name__ == '__main__':
    '''model_path = "https://flash-weights.s3.amazonaws.com/0.7.0/semantic_segmentation_model.pt"
    images_path = 'download.jpg'
    image = cv2.imread(images_path, 1)
    images_path = [
        images_path,
    ]
    detector = InferenceSeg(100)
    detector.load_model(model_path)
    mask = detector.predict(images_path, batch_size=1)
    label_map = SegmentationLabelsOutput.create_random_labels_map(num_classes=100)
    label_output = SegmentationLabelsOutput(visualize=False, labels_map=label_map, return_mask_as_image=False,
                                            labels_class=True)
    m, classes = label_output.transform(mask)
    print(classes)
    print(len(m))
    kp = []

    for mask in m:
        ia = imantics.Mask(mask)
        po = list(ia.polygons())
        for j in range(len(po)):
            for i in range(0, len(po[j]), 2):
                kp.append((po[j][i], po[j][i + 1]))

    print(kp)
    for point in kp:
        image = cv2.circle(image, (point[0], point[1]), radius=0, color=(0, 0, 255), thickness=-1)

    # image = cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), -1)
    cv2.imshow('out', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''