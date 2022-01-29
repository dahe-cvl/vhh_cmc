import numpy as np
import cv2
import datetime
from PIL import Image


class Video(object):
    """
    This class is representing a video. Each instance of this class is holding the properties of one Video.
    """

    def __init__(self):
        """
        Constructor
        """

        # printCustom("create instance of video class ... ", STDOUT_TYPE.INFO);
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

        # print(vidFile)
        # printCustom("load video information ... ", STDOUT_TYPE.INFO);
        self.vidFile = vidFile;
        if (self.vidFile == ""):
            # print("A")
            print("ERROR: you must add a video file path!");
            exit(1);
        self.vidName = self.vidFile.split('/')[-1]

        print(self.vidFile)
        self.vid = cv2.VideoCapture(self.vidFile);

        if (self.vid.isOpened() == False):
            # print("B")
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
        if (frame_id >= self.number_of_frames):
            print("ERROR: frame idx out of range!")
            return []

        # print("Read frame with id: " + str(frame_id));
        time_stamp_start = datetime.datetime.now().timestamp()

        self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        status, frame_np = self.vid.read()
        self.vid.release()

        if (status == True):
            if (self.convert_to_gray == True):
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY)
                # print(frame_gray_np.shape);
            if (self.convert_to_hsv == True):
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(frame_np)

        time_stamp_end = datetime.datetime.now().timestamp()
        time_diff = time_stamp_end - time_stamp_start
        # print("time: " + str(round(time_diff, 4)) + " sec");

        return frame_np

    # Returns Video Shot by Shot
    def getFramesByShots(self, shots_np, preprocess=None):

        # initialize video capture
        cap = cv2.VideoCapture(self.vidFile)

        frame_number = 0
        for i in range(0, len(shots_np)):
            shot = shots_np[i]

            frame_l = []
            frames_orig = []

            sid = int(shot[1])
            start_idx = int(shot[2])
            stop_idx = int(shot[3])
            #cmc_class = shot[4]

            # print(f"Retrieving Frames for Shot {sid} (frames {frame_number} to {stop_idx})...")
            while frame_number <= stop_idx:
                # read next frame
                success, image = cap.read()
                frame_number = frame_number + 1
                #print(frame_number)

                # if(start_idx == stop_idx):
                #    cv2.imshow("frame", image)
                #    k = cv2.waitKey()

                # skip to start position (for gradual cuts)
                if frame_number < start_idx:
                    # print(frame_number)
                    continue

                if success == True:
                    # if ( (frame_number >= start_idx and frame_number <= stop_idx) or (start_idx == stop_idx) ):
                    if (preprocess != None):
                        image_pre = preprocess(image)
                        frame_l.append(image_pre)
                    else:
                        frame_l.append(image)
                else:
                    break

            yield {"Images": np.array(frame_l),
                   "sid": sid,
                   "video_name": self.vidName,
                   "start": start_idx,
                   "end": stop_idx,
                   #"cmc_class": cmc_class
                   }
        cap.release()