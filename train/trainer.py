import os
import shutil

import flash
import torch
from flash.image import SemanticSegmentation
from torchmetrics import IoU, F1, Accuracy, Precision, Recall
from utils.augment import Augmentor
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import WandbLogger

from utils import dataloader, logger, utils, datahandler

data_path = '/home/amir/Projects/rutilea_singularity_gear_inspection/backbone/dataset/coco/images'
labels_path = "/dataset/coco/masks/"
labels_json_path = "/home/amir/Projects/rutilea_singularity_gear_inspection/backbone/dataset/coco/result.json"


class SemanticSegmentTrainer:
    def __init__(self, backbone, head, data_type, pre_trained_path=None, is_augment=False, augment_params=None,
                 label_map=None, logger=None):
        self.head = head
        self.backbone = backbone
        self.pre_trained_path = pre_trained_path
        self.is_augment = is_augment
        self.augment_params = augment_params
        self.labelmap = label_map
        self.data_type = data_type
        self.logger_name = logger

    def augment(self, train_images, train_masks, test_images, test_masks, augment_params):

        os.makedirs('/dataset/temp/augmented')

        os.makedirs('/dataset/temp/augmented/train')
        os.makedirs('/dataset/temp/augmented/validation')

        aug_train = Augmentor(train_images, train_masks, '/dataset/temp/augmented/train')
        aug_validation = Augmentor(test_images, test_masks, '/dataset/temp/augmented/validation')

        if augment_params:
            train_images_path, train_masks_path = aug_train.auto_augment(**augment_params)
            validation_images_path, validation_masks_path = aug_validation.auto_augment(**augment_params)
        else:
            train_images_path, train_masks_path = aug_train.auto_augment()
            validation_images_path, validation_masks_path = aug_validation.auto_augment()

        return train_images_path, train_masks_path, validation_images_path, validation_masks_path

    def train_validation_split(self, images_path, masks_path, validation_split):

        all_images_list = os.listdir(images_path)
        all_images_list.sort()

        all_masks_list = os.listdir(masks_path)
        all_masks_list.sort()

        min_length = min(len(all_images_list), len(all_masks_list))

        all_masks_list = all_masks_list[:min_length]
        all_images_list = all_images_list[:min_length]

        all_images_list = [os.path.join(images_path, x) for x in all_images_list]
        all_masks_list = [os.path.join(masks_path, x) for x in all_masks_list]

        train_images, test_images, train_masks, test_masks = train_test_split(all_images_list, all_masks_list,
                                                                              test_size=validation_split, shuffle=True,
                                                                              random_state=42)

        return train_images, train_masks, test_images, test_masks

    def train_from_images_mask(self, images_path, masks_path, save_name, batch_size=4, num_dataloader_workers=8,
                               epochs=100,
                               num_classes=2, validation_split=0.2):
        """
        :param images_path: images should be in png format
        :param masks_path: mask path should be raw image and in png format
        :return:
        """

        utils.remove_overuse_image_in_path(images_path, masks_path)
        utils.check_mask_with_cv(images_path, masks_path)

        train_images_paths, train_masks_paths, validation_images_paths, validation_masks_paths = self.train_validation_split(
            images_path, masks_path, validation_split)

        train_images_path = '/dataset/temp/train_images'
        train_masks_path = '/dataset/temp/train_masks'
        validation_images_path = '/dataset/temp/test_images'
        validation_masks_path = '/dataset/temp/test_masks'

        os.makedirs(train_images_path)
        os.makedirs(train_masks_path)
        os.makedirs(validation_images_path)
        os.makedirs(validation_masks_path)

        for f in train_images_paths:
            shutil.move(f, os.path.join(train_images_path, os.path.basename(f)))
        for f in train_masks_paths:
            shutil.move(f, os.path.join(train_masks_path, os.path.basename(f)))
        for f in validation_images_paths:
            shutil.move(f, os.path.join(validation_images_path, os.path.basename(f)))
        for f in validation_masks_paths:
            shutil.move(f, os.path.join(validation_masks_path, os.path.basename(f)))

        if self.is_augment:
            train_images_path, train_masks_path, validation_images_path, validation_masks_path = self.augment(
                train_images=train_images_path,
                train_masks=train_masks_path,
                test_images=validation_images_path,
                test_masks=validation_masks_path,
                augment_params=self.augment_params
            )

        utils.remove_overuse_image_in_path(train_images_path, train_masks_path)
        utils.check_mask_with_cv(train_images_path, train_masks_path)

        utils.remove_overuse_image_in_path(validation_images_path, validation_masks_path)
        utils.check_mask_with_cv(validation_images_path, validation_masks_path)

        datamodule = dataloader.get_dataset_for_flash(train_folder=train_images_path,
                                                      train_target_folder=train_masks_path,
                                                      batch_size=batch_size,
                                                      num_workers=num_dataloader_workers,
                                                      num_classes=num_classes,
                                                      val_folder=validation_images_path,
                                                      val_target_folder=validation_masks_path)
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
        if self.logger_name == 'wandb':
            loggerm = WandbLogger()
        else:
            loggerm = logger.ClientLogger()

        trainer = flash.Trainer(
            max_epochs=epochs, logger=loggerm, gpus=torch.cuda.device_count())
        trainer.finetune(model, datamodule=datamodule, strategy="no_freeze")
        trainer.save_checkpoint(os.path.join(os.environ.get(
            'WEIGHTS_DIR', '/weights'), "{}_model.pt".format(save_name)))
        result = trainer.validate(model, datamodule=datamodule)

        return result[0]

    def train(self, images_path, annotation_path, save_name, batch_size=4, num_dataloader_workers=8, epochs=100,
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

        try:
            os.mkdir('/dataset/temp')
        except FileExistsError:
            shutil.rmtree('/dataset/temp')
            os.mkdir('/dataset/temp')

        if self.data_type in ["coco", "COCO"]:
            images_path, masks_path = datahandler.coco_data(images_path, annotation_path)
        elif self.data_type in ['pascal', 'pascal_voc', 'pascal-voc']:
            images_path, masks_path, num_classes = datahandler.pascal_voc_data(images_path, annotation_path,
                                                                               self.labelmap)
        else:
            raise ValueError("Data type not supported")

        result = self.train_from_images_mask(images_path, masks_path, save_name, batch_size, num_dataloader_workers,
                                             epochs, num_classes, validation_split)
        return result


if __name__ == '__main__':
    # train_from_coco(data_path, labels_json_path, "coco_train")
    print()
