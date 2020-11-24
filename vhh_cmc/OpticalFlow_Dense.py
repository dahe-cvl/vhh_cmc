import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt


class OpticalFlow_Dense(object):
    def __init__(self):
        print("dense of")


    def getFlow(self, frame1, frame2):
        kp_curr_list = []
        kp_prev_list = []

        flow = cv2.calcOpticalFlowFarneback(frame1, 
                                            frame2, 
                                            None, 
                                            pyr_scale=0.5, 
                                            levels=6, 
                                            winsize=35, 
                                            iterations=3, 
                                            poly_n=5, 
                                            poly_sigma=1.2, 
                                            flags=1)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        return mag, ang, flow[...,0], flow[...,1]