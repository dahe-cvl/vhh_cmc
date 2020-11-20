import numpy as np
import cv2
import datetime
from PIL import Image
import torch


class Video(object):
    """
    This class is representing a video. Each instance of this class is holding the properties of one Video.
    """

    def __init__(self):
        """
        Constructor
        """

        #printCustom("create instance of video class ... ", STDOUT_TYPE.INFO);
        self.vidFile = ''
        self.vidName = ""
        self.frame_rate = 0
        self.channels = 0
        self.height = 0
        self.width = 0
        self.format = ''
        self.length = 0
        self.number_of_frames = 0
        self.vid = None
        self.convert_to_gray = False
        self.convert_to_hsv = False

    def load(self, vidFile: str):
        """
        Method to load video file.

        :param vidFile: [required] string representing path to video file
        """

        #print(vidFile)
        #printCustom("load video information ... ", STDOUT_TYPE.INFO);
        self.vidFile = vidFile;
        if(self.vidFile == ""):
            #print("A")
            print("ERROR: you must add a video file path!");
            exit(1);
        self.vidName = self.vidFile.split('/')[-1]
        self.vid = cv2.VideoCapture(self.vidFile);

        if(self.vid.isOpened() == False):
            #print("B")
            print("ERROR: not able to open video file!");
            exit(1);

        status, frm = self.vid.read();

        self.channels = frm.shape[2];
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT);
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH);
        self.frame_rate = self.vid.get(cv2.CAP_PROP_FPS);
        self.format = self.vid.get(cv2.CAP_PROP_FORMAT);
        self.number_of_frames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT);

        self.vid.release();

    def printVIDInfo(self):
        """
        Method to a print summary of video properties.
        """

        print("---------------------------------");
        print("Video information");
        print("filename: " + str(self.vidFile));
        print("format: " + str(self.format));
        print("fps: " + str(self.frame_rate));
        print("channels: " + str(self.channels));
        print("width: " + str(self.width));
        print("height: " + str(self.height));
        print("nFrames: " + str(self.number_of_frames));
        print("---------------------------------");


    def getFrame(self, frame_id):
        """
        Method to get one frame of a video on a specified position.

        :param frame_id: [required] integer value with valid frame index
        :return: numpy frame (WxHx3)
        """

        self.vid.open(self.vidFile)
        if(frame_id >= self.number_of_frames):
            print("ERROR: frame idx out of range!")
            return []

        #print("Read frame with id: " + str(frame_id));
        time_stamp_start = datetime.datetime.now().timestamp()

        self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        status, frame_np = self.vid.read()
        self.vid.release()

        if(status == True):
            if(self.convert_to_gray == True):
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY)
                #print(frame_gray_np.shape);
            if (self.convert_to_hsv == True):
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(frame_np)

        time_stamp_end = datetime.datetime.now().timestamp()
        time_diff = time_stamp_end - time_stamp_start
        #print("time: " + str(round(time_diff, 4)) + " sec");

        return frame_np

    def getRangeOfFrames(self, start_id=-1, stop_id=-1, pre_process=None):
        time_stamp_start = datetime.datetime.now().timestamp()

        self.vid.open(self.vidFile)
        if(start_id >= self.number_of_frames or stop_id >= self.number_of_frames):
            print("ERROR: frame index out of range!")
            return []

        if(start_id > stop_id):
            print("ERROR: start_id must be smaller (or equal) then stop_id!")
            return []

        if(start_id < 0 or stop_id < 0):
            print("ERROR: frame index out of range!")
            return []

        frame_number = 0
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_number) # optional
        success, image = self.vid.read()

        frames_l = []
        for i in range(start_id, stop_id+1):
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, image = self.vid.read()

            if(pre_process != None):
                image = pre_process.applyTransformOnImg(image)
                frames_l.append(image)
            else:    
                frames_l.append(image)
        self.vid.release()

        time_stamp_end = datetime.datetime.now().timestamp()
        time_diff = time_stamp_end - time_stamp_start

        if(pre_process != None):
            frames_np = np.array(frames_l)
            return frames_np, time_diff
        else:    
            frames_np = np.array(frames_l)
            return frames_np, time_diff
