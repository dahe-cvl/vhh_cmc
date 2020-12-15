import cv2
import numpy as np
from vhh_cmc.Configuration import Configuration


class PreProcessing(object):
    """
    This class is used to pre-process frames.
    """

    def __init__(self, config_instance: Configuration):
        """
        Constructor

        :param config_file: [required] path to configuration file (e.g. PATH_TO/config.yaml)
                            must be with extension ".yaml"
        """
        print("create instance of pre-processing ... ")
        self.config_instance = config_instance

    def applyTransformOnImg(self, image: np.ndarray) -> np.ndarray:
        """
        This method is used to apply the configured pre-processing methods on a numpy frame.

        :param image: This parameter must hold a valid numpy image (WxHxC).
        :return: This methods returns the preprocessed numpy image.
        """
        image_trans = image

        # convert to grayscale image
        if(int(self.config_instance.flag_convert2Gray) == 1):
            image_trans = self.convertRGB2Gray(image_trans)

        # crop image
        if (int(self.config_instance.center_crop_flag) == 1):
            image_trans = self.crop(image_trans, (image_trans.shape[0], image_trans.shape[0]))
            #image_trans = self.crop(image_trans, (700, 512))

        # resize image
        if(self.config_instance.flag_downscale == 1):
            dim = self.config_instance.resize_dim
            image_trans = self.resize(image_trans, dim)

        return image_trans

    def convertRGB2Gray(self, img: np.ndarray):
        """
        This method is used to convert a RBG numpy image to a grayscale image.

        :param img: This parameter must hold a valid numpy image.
        :return: This method returns a grayscale image (WxHx1).
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img_gray = np.expand_dims(img_gray, axis=-1)
        return img_gray

    def crop(self, img: np.ndarray, dim: tuple):
        """
        This method is used to crop a specified region of interest from a given image.

        :param img: This parameter must hold a valid numpy image.
        :param dim: This parameter must hold a valid tuple including the crop dimensions.
        :return: This method returns the cropped image.
        """
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
        """
        This method is used to resize a image.

        :param img: This parameter must hold a valid numpy image.
        :param dim: This parameter must hold a valid tuple including the resize dimensions.
        :return: This method returns the resized image.
        """
        img_resized = cv2.resize(img, dim)
        if(int(self.config_instance.flag_convert2Gray) == 1):
            img_resized = np.expand_dims(img_resized, axis=-1)
        return img_resized


