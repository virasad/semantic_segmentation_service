import os
import shutil

import flash
import torch
from flash.image import SemanticSegmentation
from torchmetrics import IoU, F1, Accuracy, Precision, Recall
from utils.augment import Augmentor

from utils import dataloader, logger, dataset
from glob import glob

data_path = '/home/amir/Projects/rutilea_singularity_gear_inspection/backbone/dataset/coco/images'
labels_path = "/dataset/coco/masks/"
labels_json_path = "/home/amir/Projects/rutilea_singularity_gear_inspection/backbone/dataset/coco/result.json"


class SemanticSegmentTrainer:
    def __init__(self, backbone, head, pre_trained_path=None, is_augment=False, augment_params=None):
        self.head = head
        self.backbone = backbone
        self.pre_trained_path = pre_trained_path
        self.is_augment = is_augment
        self.augment_params = augment_params

    def augment(self, images_path, masks_path, augment_params):
        try:
            os.mkdir('/dataset/temp')
        except FileExistsError:
            shutil.rmtree('/dataset/temp')
            os.mkdir('/dataset/temp')

        os.makedirs('/dataset/temp/images')
        os.makedirs('/dataset/temp/masks')

        for image_path in glob(os.path.join(images_path, '*')):
            shutil.copy(image_path, '/dataset/temp/images')
        for mask_path in glob(os.path.join(masks_path, '*')):
            shutil.copy(mask_path, '/dataset/temp/masks')

        os.makedirs('/dataset/temp/augmented')

        aug = Augmentor('/dataset/temp/images', '/dataset/temp/masks', '/dataset/temp/augmented')
        if augment_params:
            images_path, masks_path = aug.auto_augment(**augment_params)
        else:
            images_path, masks_path = aug.auto_augment()
        return images_path, masks_path

    def train_from_images_mask(self, images_path, masks_path, save_name, batch_size=4, num_dataloader_workers=8, epochs=100,
                               num_classes=2, validation_split=0.2):
        """
        :param images_path: images should be in png format
        :param masks_path: mask path should be raw image and in png format
        :return:
        """

        datamodule = dataloader.get_dataset_for_flash(images_path, masks_path, batch_size,
                                                      num_workers=num_dataloader_workers, num_classes=num_classes,
                                                      validation_split=validation_split)
        # 2. Build the task
        if self.pre_trained_path != None:
            model = SemanticSegmentation.load_from_checkpoint(
                self.pre_trained_path)

        else:
            model = SemanticSegmentation(
                backbone=self.backbone,
                head=self.head,
                num_classes=datamodule.num_classes,
                metrics=[IoU(num_classes=datamodule.num_classes),
                         F1(num_classes=datamodule.num_classes,
                            mdmc_average='samplewise'),
                         Accuracy(num_classes=datamodule.num_classes,
                                  mdmc_average='samplewise'),
                         Precision(num_classes=datamodule.num_classes,
                                   mdmc_average='samplewise'),
                         Recall(num_classes=datamodule.num_classes,
                                mdmc_average='samplewise')],
            )
        # 3. Create the trainer and finetune the model
        trainer = flash.Trainer(
            max_epochs=epochs, logger=logger.ClientLogger(), gpus=torch.cuda.device_count())
        trainer.finetune(model, datamodule=datamodule, strategy="no_freeze")
        trainer.save_checkpoint(os.path.join(os.environ.get(
            'WEIGHTS_DIR', '/weights'), "{}_model.pt".format(save_name)))
        result = trainer.validate(model, datamodule=datamodule)

        return result[0]

    def train_from_coco(self, images_path, json_annotation_path, save_name, batch_size=4, num_dataloader_workers=8, epochs=100,
                        num_classes=2,
                        validation_split=0.2):
        """

        :param images_path: jpg or png images path
        :param json_annotation_path: coco dataset annotation path
        :param save_name: save weight name ( you can add time to it)
        :param batch_size: batch size for train
        :param num_dataloader_workers: depends on your cpu
        :param epochs: max number of epochs
        :return: {"result": "staus", "error": "error message"}
        """

        # check files exist

        # list files in dir
        if not os.path.exists(images_path):
            raise FileExistsError("images path not found")
        if not os.path.exists(json_annotation_path):
            raise FileExistsError("json annotation path not found")

        png_images_path = images_path.replace("images", "pngimages")
        try:
            os.mkdir(png_images_path)
        except FileExistsError:
            shutil.rmtree(png_images_path)
            os.mkdir(png_images_path)

        dataset.batch_jpg_to_png(images_path)
        pngmasks_path = images_path.replace("images", "pngmasks")
        try:
            os.mkdir(pngmasks_path)

        except FileExistsError:
            shutil.rmtree(pngmasks_path)
            os.mkdir(pngmasks_path)

        dataset.CocoHandler(json_annotation_path,
                            images_path).convert_dataset_to_masks(pngmasks_path)
        print('dataset convert to masks')

        if self.is_augment:
            png_images_path, pngmasks_path = self.augment(
                png_images_path, pngmasks_path, self.augment_params)

        result = self.train_from_images_mask(png_images_path, pngmasks_path, save_name, batch_size, num_dataloader_workers,
                                             epochs, num_classes, validation_split)
        return result


if __name__ == '__main__':
    #train_from_coco(data_path, labels_json_path, "coco_train")
    print()
