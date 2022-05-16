import albumentations as A
import cv2
import os
from tqdm import tqdm

def get_filters():
    filters_of_aug = [
        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0,
                           shift_limit=0.1, p=1, border_mode=0),
        A.RandomRotate90(p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomShadow(p=0.1),
        A.RandomSnow(snow_point_lower=0.1,
                     snow_point_upper=0.15, p=0.1),
        A.RGBShift(p=0.2),
        A.CLAHE(p=0.2),

        A.HueSaturationValue(
            p=0.1, hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10),

        A.MotionBlur(p=0.1),
        A.MedianBlur(p=0.2),
        A.ISONoise(p=0.2),
        A.Posterize(p=0.2),
        A.Perspective(p=0.1),
        A.PiecewiseAffine(
            p=0.1, scale=(0.01, 0.02)),
        A.Emboss(p=0.2),
    ]
    return filters_of_aug

class Augmentor:

    _file_counter = 0

    def __init__(self, images_path, masks_path, save_path):
        self.filters = get_filters()
        self.images_path = images_path
        self.masks_path = masks_path
        if os.path.isdir(os.path.join(save_path, 'images')):
            self.new_images_path = os.path.join(save_path, 'images')
        else:
            os.mkdir(os.path.join(save_path, 'images'))
            self.new_images_path = os.path.join(save_path, 'images')

        if os.path.isdir(os.path.join(save_path, 'masks')):
            self.new_masks_path = os.path.join(save_path, 'masks')
        else:
            os.mkdir(os.path.join(save_path, 'masks'))
            self.new_masks_path = os.path.join(save_path, 'masks')

    def new_augment(self, image_path, mask_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        augmented = self.aug(image=image, mask=mask)
        return augmented['image'], augmented['mask']

    @staticmethod
    def save_images(image_path, mask_path, image, mask):
        cv2.imwrite(image_path, image)
        cv2.imwrite(mask_path, mask)



    def auto_augment(self, quantity = 100, resize=False, width=0, height=0):
        print('augment started')
        images_list = os.listdir(self.images_path)
        images_list.sort()

        if resize:
            self.filters.insert(0, A.Resize(width=width, height=height, p=1))

        self.aug = A.Compose(self.filters)

        for image_name in tqdm(images_list):

            image_path = os.path.join(self.images_path, image_name)
            mask_path = os.path.join(self.masks_path, image_name)

            for _ in tqdm(range(quantity)):
                image, mask = self.new_augment(image_path, mask_path)
                new_image_path = os.path.join(self.new_images_path, f'{self._file_counter}.png')
                new_mask_path = os.path.join(self.new_masks_path, f'{self._file_counter}.png')
                self.save_images(new_image_path, new_mask_path, image, mask)
                self._file_counter += 1

        return self.new_images_path, self.new_masks_path