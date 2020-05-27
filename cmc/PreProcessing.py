import cv2
import numpy as np
from cmc.Configuration import Configuration


class PreProcessing(object):
    def __init__(self, config_instance: Configuration):
        print("create instance of pre-processing ... ")
        self.config_instance = config_instance

    def applyTransformOnImg(self, image: np.ndarray) -> np.ndarray:
        image_trans = image

        # convert to grayscale image
        if(int(self.config_instance.flag_convert2Gray) == 1):
            image_trans = self.convertRGB2Gray(image_trans)

        # crop image
        if (int(self.config_instance.center_crop_flag) == 1):
            image_trans = self.crop(image_trans, (image_trans.shape[0], image_trans.shape[0]))

        # resize image
        if(self.config_instance.flag_downscale == 1):
            dim = self.config_instance.resize_dim
            image_trans = self.resize(image_trans, dim)

        return image_trans

    def applyTransformOnImgSeq(self, img_seq: np.ndarray) -> np.ndarray:
        #printCustom("NOT IMPLEMENTED YET", STDOUT_TYPE.INFO);
        img_seq_trans_l = []
        for i in range(0, len(img_seq)):
            img_seq_trans_l.append(self.applyTransformOnImg(img_seq[i]))
        img_seq_trans = np.array(img_seq_trans_l)
        return img_seq_trans

    def convertRGB2Gray(self, img: np.ndarray):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = np.expand_dims(img_gray, axis=-1)
        return img_gray

    def crop(self, img: np.ndarray, dim: tuple):
        crop_w, crop_h = dim

        crop_h_1 = 0
        crop_h_2 = 0
        crop_w_1 = 0
        crop_w_2 = 0

        img_h = img.shape[0]
        img_w = img.shape[1]

        crop_w_1 = int(img_w / 2) - int(crop_w / 2)
        if (crop_w_1 < 0):
            crop_w_1 = 0

        crop_w_2 = int(img_w / 2) + int(crop_w / 2)
        if (crop_w_2 >= img_w):
            crop_w_2 = img_w

        crop_h_1 = int(img_h / 2) - int(crop_h / 2)
        if (crop_h_1 < 0):
            crop_h_1 = 0

        crop_h_2 = int(img_h / 2) + int(crop_h / 2)
        if (crop_h_2 >= img_h):
            crop_h_2 = img_h

        img_crop = img[crop_h_1:crop_h_2, crop_w_1:crop_w_2]
        return img_crop

    def resize(self, img: np.ndarray, dim: tuple):
        img_resized = cv2.resize(img, dim)
        return img_resized


