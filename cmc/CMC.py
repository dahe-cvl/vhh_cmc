import cv2
import numpy as np
from enum import IntEnum
from matplotlib import pyplot as plt
from cmc import cmc_io
from cmc.Configuration import Configuration
import os
from cmc.OpticalFlow import OpticalFlow


class CMC(object):
    """
    Main class of shot type classification (stc) package.
    """
    def __init__(self, config_file: str):
        """
        Constructor

        :param config_file: [required] path to configuration file (e.g. PATH_TO/config.yaml)
                                       must be with extension ".yaml"
        """
        print("create instance of cmc ... ")

        if (config_file == ""):
            print("No configuration file specified!")
            exit()

        self.config_instance = Configuration(config_file)
        self.config_instance.loadConfig()

        if (self.config_instance.debug_flag == True):
            print("DEBUG MODE activated!")
            self.debug_results = "/data/share/maxrecall_vhh_mmsi/videos/results/cmc/develop/"

    def runOnSingleVideo(self, shots_per_vid_np=None, max_recall_id=-1):
        """
        Method to run cmc classification on specified video.

        :param shots_per_vid_np: [required] numpy array representing all detected shots in a video
                                 (e.g. sid | movie_name | start | end )
        :param max_recall_id: [required] integer value holding unique video id from VHH MMSI system
        """

        print("run cmc classifier on single video ... ")

        if (type(shots_per_vid_np) == None):
            print("ERROR: you have to set the parameter shots_per_vid_np!")
            exit()

        if (max_recall_id == -1 or max_recall_id == 0):
            print("ERROR: you have to set a valid max_recall_id [1-n]!")
            exit()

        if (self.config_instance.debug_flag == True):
            # load shot list from result file
            shots_np = self.loadSbdResults(self.config_instance.sbd_results_path)
        else:
            shots_np = shots_per_vid_np

        if (len(shots_np) == 0):
            print("ERROR: there must be at least one shot in the list!")
            exit()

        if (self.config_instance.debug_flag == True):
            num_shots = 2
        else:
            num_shots = len(shots_np)

        # read all frames of video
        vid_name = shots_np[0][1]
        cap = cv2.VideoCapture(self.config_instance.path_videos + "/" + vid_name)
        frame_l = []
        cnt = 0

        while(True):
            cnt = cnt + 1
            ret, frame = cap.read()
            # print(cnt)
            # print(ret)
            # print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if (ret == True):
                frame = self.pre_process(frame)
                frame_l.append(frame)
            else:
                break

        all_frames_np = np.array(frame_l)
        print(all_frames_np.shape)

        results_cmc_l = []
        for idx in range(0, num_shots):
            # print(shots_np[idx])
            shot_id = int(shots_np[idx][0])
            vid_name = str(shots_np[idx][1])
            start = int(shots_np[idx][2])
            stop = int(shots_np[idx][3])
            shot_frames_np = all_frames_np[start:stop + 1, :, :, :]
            print(shot_frames_np.shape)

            # run optical flow process
            optical_flow_instance = OpticalFlow(video_frames=shot_frames_np,
                                                fPath=self.config_instance.path_videos + "/" + vid_name,
                                                sf=0,
                                                ef=len(shot_frames_np),
                                                mode=self.config_instance.mode,
                                                sensitivity=self.config_instance.sensitivity,
                                                specificity=self.config_instance.specificity,
                                                border=self.config_instance.border,
                                                number_of_features=self.config_instance.number_of_features,
                                                angle_diff_limit=self.config_instance.angle_diff_limit,
                                                config=None)
            pan_list, tilt_list = optical_flow_instance.run()

            number_of_pans = len(pan_list)
            number_of_tilts = len(tilt_list)

            print("---------------")
            print(start)
            print(stop)

            class_name = "NA"
            if (number_of_pans >= 1) and (number_of_pans > number_of_tilts):
                print("PAN")
                class_name = self.config_instance.class_names[0]
            elif(number_of_tilts >= 1) and (number_of_tilts > number_of_pans):
                print("TILT")
                class_name = self.config_instance.class_names[1]
            elif(number_of_tilts == number_of_pans):
                print("NA")
                class_name = self.config_instance.class_names[2]

            # prepare results
            print(str(vid_name) + ";" + str(shot_id) + ";" + str(start) + ";" + str(stop) + ";" + str(class_name))
            results_cmc_l.append([str(vid_name) + ";" + str(shot_id) + ";" + str(start) + ";" + str(stop) + ";" + str(class_name)])

        results_cmc_np = np.array(results_cmc_l)

        # export results
        self.exportCmcResults(str(max_recall_id), results_cmc_np)

    def pre_process(self, frame):
        frame = cv2.resize(frame, self.config_instance.resize_dim)
        return frame

    def loadSbdResults(self, sbd_results_path):
        """
        Method for loading shot boundary detection results as numpy array

        .. note::
            Only used in debug_mode.

        :param sbd_results_path: [required] path to results file of shot boundary detection module (vhh_sbd)
        :return: numpy array holding list of detected shots.
        """

        # open sbd results
        fp = open(sbd_results_path, 'r')
        lines = fp.readlines()
        lines = lines[1:]

        lines_n = []
        for i in range(0, len(lines)):
            line = lines[i].replace('\n', '')
            line_split = line.split(';')
            lines_n.append([line_split[0], os.path.join(line_split[1]), line_split[2], line_split[3]])
        lines_np = np.array(lines_n)
        # print(lines_np.shape)

        return lines_np

    def exportCmcResults(self, fName, cmc_results_np: np.ndarray):
        """
        Method to export cmc results as csv file.

        :param fName: [required] name of result file.
        :param cmc_results_np: numpy array holding the camera movements classification predictions for each shot of a movie.
        """

        print("export results to csv!")

        if (len(cmc_results_np) == 0):
            print("ERROR: numpy is empty")
            exit()

        # open stc resutls file
        if (self.config_instance.debug_flag == True):
            fp = open(self.debug_results + "/" + fName + ".csv", 'w')
        else:
            fp = open(self.config_instance.path_final_results + "/" + fName + ".csv", 'w')
        header = "vid_name;shot_id;start;end;cmc"
        fp.write(header + "\n")

        for i in range(0, len(cmc_results_np)):
            tmp_line = str(cmc_results_np[i][0])
            for c in range(1, len(cmc_results_np[i])):
                tmp_line = tmp_line + ";" + cmc_results_np[i][c]
            fp.write(tmp_line + "\n")

