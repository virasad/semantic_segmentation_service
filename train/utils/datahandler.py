from . import dataset
import os
import shutil
from tqdm import tqdm
import cv2
import numpy as np


def coco_data(images_path, json_annotation_path):
    # list files in dir
    if not os.path.exists(images_path):
        raise FileExistsError("images path not found")
    if not os.path.exists(json_annotation_path):
        raise FileExistsError("json annotation path not found")

    png_images_path = "/dataset/temp/pngimages"
    try:
        os.mkdir(png_images_path)
    except FileExistsError:
        shutil.rmtree(png_images_path)
        os.mkdir(png_images_path)

    dataset.batch_jpg_to_png(images_path, png_images_path)

    pngmasks_path = "/dataset/temp/pngmasks"
    try:
        os.mkdir(pngmasks_path)
    except FileExistsError:
        shutil.rmtree(pngmasks_path)
        os.mkdir(pngmasks_path)

    dataset.CocoHandler(json_annotation_path,
                        images_path).convert_dataset_to_masks(pngmasks_path)
    return png_images_path, pngmasks_path


def pascal_voc_data(images_path, annotation_path, labelmap_path):
    dataset_path = os.path.dirname(images_path)
    converted_mask_p =os.path.join(dataset_path, "temp/converted_masks")
    try:
        os.makedirs(converted_mask_p)
    except FileExistsError:
        shutil.rmtree(converted_mask_p)
        os.makedirs(converted_mask_p)

    png_images_path = os.path.join(dataset_path, "temp/pngimages")
    try:
        os.mkdir(png_images_path)
    except FileExistsError:
        shutil.rmtree(png_images_path)
        os.mkdir(png_images_path)

    dataset.batch_jpg_to_png(images_path, png_images_path)

    pngmasks_path = os.path.join(dataset_path,"temp/pngmasks")

    try:
        os.mkdir(pngmasks_path)
    except FileExistsError:
        shutil.rmtree(pngmasks_path)
        os.mkdir(pngmasks_path)

    dataset.batch_jpg_to_png(annotation_path, pngmasks_path)

    images_path = png_images_path
    annotation_path = pngmasks_path

    label_map = open(labelmap_path, "r")
    labelmaps = label_map.readlines()
    label_map.close()

    labelmaps = [x.strip() for x in labelmaps]

    class_names = []
    class_index = []
    class_color = []

    for idx, labelmap in enumerate(labelmaps):
        class_names.append(labelmap.split(":")[0])
        class_index.append(idx)
        class_color.append(labelmap.split(":")[1])

    mask_paths = os.listdir(annotation_path)
    mask_paths = [os.path.join(annotation_path, x) for x in mask_paths]

    for mask_path in tqdm(mask_paths):
        mask = cv2.imread(mask_path, 1)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        converted_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        # converted_mask = cv2.cvtColor(converted_mask, cv2.COLOR_BGR2GRAY)

        for idx, color in enumerate(class_color):
            color = color.split(",")
            color = [int(x) for x in color]
            converted_mask[np.where((mask == color).all(axis=2))] = class_index[idx]

        cv2.imwrite(os.path.join(converted_mask_p, os.path.basename(mask_path)), converted_mask)

    return images_path, converted_mask_p, len(class_names)
