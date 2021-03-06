import os
from glob import glob

import cv2
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

dataset_path = '/dataset/coco'


class CocoHandler:
    def __init__(self, annotation_path, images_path):
        self.annotation_path = annotation_path
        self.images_path = images_path
        self.coco = COCO(annotation_path)
        self.ids = self.coco.getImgIds()

    def coco_to_mask(self, image_id):
        image_info = self.coco.loadImgs(image_id)[0]
        cat_ids = self.coco.getCatIds()
        anns_ids = self.coco.getAnnIds(imgIds=image_info['id'], catIds=cat_ids, iscrowd=None)
        anns = self.coco.loadAnns(anns_ids)
        anns_img = np.zeros((image_info['height'], image_info['width']))
        for ann in anns:
            anns_img = np.maximum(anns_img, self.coco.annToMask(ann) * (ann['category_id']))
        return anns_img, image_info['file_name']

    def convert_dataset_to_masks(self, dst_mask_path):
        for image_id in tqdm(self.ids, desc='Converting coco annotation to masks'):
            mask, file_name = self.coco_to_mask(image_id)
            mask = mask.astype(np.uint8)
            cv2.imwrite(os.path.join(dst_mask_path, file_name).replace('.jpg', '.png'), mask)

    def generate_label_map(self):
        label_map = {}
        for cat in self.coco.loadCats(self.coco.getCatIds()):
            label_map[cat['id']] = cat['name']


def check_dataset(input_data, target_data):
    input_files = os.listdir(input_data)
    target_files = os.listdir(target_data)
    all_files = set(input_files).intersection(set(target_files))

    if len(all_files) != len(input_files) or len(all_files) != len(target_files):
        print('not equal')
        print(all_files, len(input_files), len(target_files), len(all_files))


def batch_jpg_to_png(images_path):
    for img_path in tqdm(glob(images_path + '/*.jpg'), desc='Converting jpg to png'):
        dst_path = img_path.replace('images', 'pngimages').replace('.jpg', '.png')
        jpg_to_png(img_path, dst_path)


def jpg_to_png(path, dst_path):
    img = Image.open(path)
    img.save(dst_path)


def main():
    pass


if __name__ == '__main__':
    pass
