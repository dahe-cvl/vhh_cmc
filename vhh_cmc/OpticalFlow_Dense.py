import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt


class OpticalFlow_Dense(object):
    def __init__(self):
        print("dense of")

    def getFlow(self, frame1, frame2):
        flow = cv2.calcOpticalFlowFarneback(frame1,
                                            frame2,
                                            None,
                                            pyr_scale=0.5,
                                            levels=3,
                                            winsize=15,
                                            iterations=3,  # 3
                                            poly_n=5,  # 5
                                            poly_sigma=1.2,
                                            flags=0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1], angleInDegrees=True)

        '''
        hsv = np.zeros((frame1.shape[0],frame1.shape[1], 3), dtype=np.uint8)
        print(hsv.shape)
        print(frame1.shape)
        hsv[..., 1] = 255

        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("colored flow", bgr)
        cv2.imshow("orig", frame1)
        cv2.waitKey(10)
        '''
        return mag, ang, flow[...,0], flow[...,1]