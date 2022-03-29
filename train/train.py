import os
import shutil

import flash
import torch
from flash.image import SemanticSegmentation
from torchmetrics import IoU, F1, Accuracy, Precision, Recall

from utils import dataloader, logger, dataset

data_path = '/home/amir/Projects/rutilea_singularity_gear_inspection/backbone/dataset/coco/images'
labels_path = "/dataset/coco/masks/"
labels_json_path = "/home/amir/Projects/rutilea_singularity_gear_inspection/backbone/dataset/coco/result.json"


def train_from_images_mask(images_path, masks_path, save_name, batch_size=4, num_dataloader_workers=8, epochs=100):
    """
    :param images_path: images should be in png format
    :param masks_path: mask path should be raw image and in png format
    :return:
    """
    datamodule = dataloader.get_dataset_for_flash(images_path, masks_path, batch_size,
                                                  num_workers=num_dataloader_workers)
    # 2. Build the task
    model = SemanticSegmentation(
        backbone="deeplabv3plus",
        head='deeplabv3',
        num_classes=datamodule.num_classes,
        metrics=[IoU(num_classes=datamodule.num_classes),
                 F1(num_classes=datamodule.num_classes, mdmc_average='samplewise'),
                 Accuracy(num_classes=datamodule.num_classes, mdmc_average='samplewise'),
                 Precision(num_classes=datamodule.num_classes, mdmc_average='samplewise'),
                 Recall(num_classes=datamodule.num_classes, mdmc_average='samplewise')],
    )

    # 3. Create the trainer and finetune the model
    trainer = flash.Trainer(max_epochs=epochs, logger=logger.ClientLogger(), gpus=torch.cuda.device_count())
    trainer.finetune(model, datamodule=datamodule, strategy="no_freeze")
    trainer.save_checkpoint("{}_model.pt".format(save_name))
    result = trainer.validate(model, datamodule=datamodule)

    return result[0]


def train_from_coco(images_path, json_annotation_path, save_name, batch_size=4, num_dataloader_workers=8, epochs=100):
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

    dataset.CocoHandler(json_annotation_path, images_path).convert_dataset_to_masks(pngmasks_path)
    result = train_from_images_mask(png_images_path, pngmasks_path, save_name, batch_size, num_dataloader_workers, epochs)
    return result


if __name__ == '__main__':
    train_from_coco(data_path, labels_json_path, "coco_train")
