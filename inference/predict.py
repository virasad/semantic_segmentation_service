import matplotlib
from flash import Trainer
from flash.image import SemanticSegmentation, SemanticSegmentationData
from utils.postprocess import SegmentationLabelsOutput
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

class InferenceSeg:
    def __init__(self, num_classes):
        matplotlib.use('TkAgg')
        self.trainer = Trainer()
        self.num_classes = num_classes
        self.label_map = dict(zip(range(2), [(0, 0, 0), (0, 0, 255)]))

    def set_label_map(self, label_map):
        self.label_map = label_map


    def load_model(self, model_path):
        print('label map : : ',dict(zip(range(2), [(0, 0, 0), (0, 0, 255)])))
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
            print("images are paths")
            images = [np.array(Image.open(image)) for image in images]
            # change color channel to first dimension
            images = [np.moveaxis(image, 2, 0) for image in images]
            print(images[0].shape)
            # images = [np.array(image) for image in images]
            # images = [cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB) for image in images]

        datamodule = SemanticSegmentationData.from_numpy(predict_data=images,
                                            batch_size=batch_size,
                                            num_classes=100,)
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


if __name__ == '__main__':
    model_path = "https://flash-weights.s3.amazonaws.com/0.7.0/semantic_segmentation_model.pt"
    images_path = '/home/amir/Projects/semantic_segmentation_service/data/CameraRGB/F61-1.png'
    # image = cv2.imread(images_path)
    images_path = [
        images_path,
    ]
    detector = InferenceSeg(model_path)
    detector.load_model(model_path)
    mask = detector.predict(images_path, batch_size=1)
    print(mask)
    label_map = SegmentationLabelsOutput.create_random_labels_map(num_classes=100)
    label_output = SegmentationLabelsOutput(visualize=True, labels_map=label_map)

    label_output.transform(mask)
    print(type(mask))
