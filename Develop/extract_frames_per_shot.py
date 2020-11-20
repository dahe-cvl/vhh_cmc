from cmc.CMC import CMC
import os

#####################################################################
##           CONFIGURATION 
#####################################################################

video_path = "/data/share/maxrecall_vhh_mmsi/develop/videos/downloaded/"  
dstPath = "/data/share/datasets/cmc_v1/extracted_frames/"  

#####################################################################

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

        '''
        frame_number = 0
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_number) # optional
        success, image = self.vid.read()

        frames_l = []
        for i in range(start_id, stop_id+1):
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, image = self.vid.read()

            if(pre_process != None):
                image = pre_process(image)
                frames_l.append(image)
            else:    
                frames_l.append(image)
        '''

        success = self.vid.grab() # get the next frame
        frame_number = 0

        frames_l = []
        while success:
            if (frame_number >= start_id and frame_number <= stop_id ):
                #print(frame_number)
                _, image = self.vid.retrieve()
                if(pre_process != None):
                    image = pre_process(image)
                    frames_l.append(image)
                else:    
                    frames_l.append(image)

            if (frame_number > stop_id):
                break
            # read next frame
            success = self.vid.grab()	
            frame_number = frame_number + 1 

        self.vid.release()

        time_stamp_end = datetime.datetime.now().timestamp()
        time_diff = time_stamp_end - time_stamp_start
 
        if(pre_process != None):
            frame_tensors = torch.stack(frames_l)
            return frame_tensors, time_diff
        else:    
            frames_np = np.array(frames_l)
            return frames_np, time_diff


def loadStcResults(stc_results_path):
        """
        Method for loading shot boundary detection results as numpy array

        .. note::
            Only used in debug_mode.

        :param sbd_results_path: [required] path to results file of shot boundary detection module (vhh_sbd)
        :return: numpy array holding list of detected shots.
        """

        # open sbd results
        fp = open(stc_results_path, 'r')
        lines = fp.readlines()
        lines = lines[1:]

        lines_n = []
        for i in range(0, len(lines)):
            line = lines[i].replace('\n', '')
            line_split = line.split(';')
            lines_n.append([line_split[0], os.path.join(line_split[1]), line_split[2], line_split[3], line_split[4]])
        lines_np = np.array(lines_n)
        # print(lines_np.shape)

        return lines_np


config_file = "/caa/Homes01/dhelm/working/vhh/develop/vhh_cmc/config/config_cmc.yaml"
cmc_instance = CMC(config_file)

results_path = "/data/share/datasets/vhh_mmsi_test_db_v2/annotations/stc/"
results_file_list = os.listdir(results_path)
print(results_file_list)

cnt = 0
for file in results_file_list:
    #print(file)
    shots_np = loadStcResults(results_path + file)
    
    # read all frames of video
    vid_name = shots_np[0][0]

    vid_instance = Video()
    vid_instance.load(video_path + vid_name)
    #vid_instance.printVIDInfo()

    for i in range(0, len(shots_np)):
        #print(shots_np[i])
        vid_name = str(shots_np[i][0])
        start = int(shots_np[i][2])
        stop = int(shots_np[i][3])
        stc_class = str(shots_np[i][4])
        center_frame_idx = int(abs(stop - start) / 2)

        if(stc_class == "LS" or stc_class == "ELS" or stc_class == "MS"):
            print(shots_np[i])
            print("center_idx: " + str(center_frame_idx))
            cnt = cnt + 1
            
            frames_np, time_diff = vid_instance.getRangeOfFrames(start_id=start, stop_id=stop, pre_process=None)
            print(time_diff)
            #print(frames_np.shape)
            #print(frames_np[center_frame_idx].shape)
            
            cv2.imwrite(dstPath + "/" + str(vid_name.split('.')[0]) + "_" + str(center_frame_idx) + ".png", frames_np[center_frame_idx])
            #k = cv2.waitKey(1)


print(cnt)