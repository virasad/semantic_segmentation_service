import os
from glob import glob
import cv2


def mkdir_p(dirname: str):
    if not os.path.isdir(dirname):
        os.mkdir(dirname)


def remove_overuse_image_in_path(images_dir, mask_dir):
    for image_file in glob(os.path.join(images_dir, '*.png')):
        image_name = os.path.basename(image_file)
        mask_file = os.path.join(mask_dir, image_name)
        if not os.path.exists(mask_file):
            os.remove(image_file)

def check_mask_with_cv(images_dir, mask_dir):
    print('check mask')
    for mask_file in glob(os.path.join(mask_dir, '*.png')):
        mask_name = os.path.basename(mask_file)
        image_file = os.path.join(images_dir, mask_name)
        # image = cv2.imread(image_file)

        print('open image', image_file)
        image = cv2.imread(image_file)
        mask = cv2.imread(mask_file)
        if not image and not mask:
            print(mask_file)
            os.remove(mask_file)
            os.remove(image_file)
            print('remove')
    print('done')