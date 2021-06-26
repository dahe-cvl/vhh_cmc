import cv2
import numpy as np
from vhh_cmc.Configuration import Configuration
from vhh_cmc.PreProcessing import PreProcessing
import os
from vhh_cmc.OpticalFlow import OpticalFlow
from vhh_cmc.Video import Video

#import matplotlib
#matplotlib.use('Qt5Agg')

class CMC(object):
    """
    Main class of camera movements classification (cmc) package.
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
            self.debug_results = self.config_instance.path_debug_results
            print("save debug results to: " + str(self.debug_results))

        self.pre_processing_instance = PreProcessing(config_instance=self.config_instance)

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
            debug_sid = 104 #tilt ids: 86  #104 73 108  pan ids: 5 7 21 13 28 na ids: 3 24 77
        else:
            shots_np = shots_per_vid_np
            debug_sid = -1

        if (len(shots_np) == 0):
            print("ERROR: there must be at least one shot in the list!")
            exit()

        # read all frames of video
        vid_name = shots_np[0][0]

        if(self.config_instance.save_eval_results == 1):
            print("Evaluation mode is activated ...")
            print(self.config_instance.path_videos + "/" + vid_name)
            #cap = cv2.VideoCapture(self.config_instance.path_videos + "/" + vid_name)

            vid_instance = Video()
            vid_instance.load(self.config_instance.path_videos + "/" + vid_name)
        else:
            print(self.config_instance.path_videos + "/" + vid_name)
            #cap = cv2.VideoCapture(self.config_instance.path_videos + "/" + vid_name)

            vid_instance = Video()
            vid_instance.load(self.config_instance.path_videos + "/" + vid_name)

            #vid_name = "20_3776.mp4"  #training_data/pan/69_24604.mp4;0;0;142  training_data/pan/77_66460.mp4;0;0;124 training_data/tilt/20_3776.mp4;1;0;125
            #vid_instance.load("/data/share/datasets/cmc_final_dataset_v2/training_data/tilt" + "/" + vid_name)
            #shots_np = np.array([["999", "1", "0", "125"]])

        print(shots_np)
        results_cmc_l = []
        for data in vid_instance.getFramesByShots(shots_np, preprocess=self.pre_processing_instance.applyTransformOnImg):
            frames_per_shots_np = data['Images']
            shot_id = data['sid']
            vid_name = data['video_name']
            start = data['start']
            stop = data['end']

            if(shot_id != debug_sid and self.config_instance.debug_flag == True):
                continue

            print(f'sid: {shot_id}')
            print(f'vid_name: {vid_name}')
            print(f'frames_per_shot: {frames_per_shots_np.shape}')
            print(f'start: {start}, end: {stop}')

            shot_len = stop - start
            MIN_NUMBER_OF_FRAMES_PER_SHOT = 40
            if(shot_len <= MIN_NUMBER_OF_FRAMES_PER_SHOT ):
                #print("shot length is too small!")
                class_name = "NA"
            else:
                # add new optical flow version
                optical_flow_instance = OpticalFlow(video_frames=frames_per_shots_np,
                                                    algorithm=None,
                                                    config_instance=self.config_instance)
                class_name = optical_flow_instance.runDense()

            # prepare results
            print(str(vid_name) + ";" + str(shot_id) + ";" + str(start) + ";" + str(stop) + ";" + str(class_name))
            results_cmc_l.append([str(vid_name) + ";" + str(shot_id) + ";" + str(start) + ";" + str(stop) + ";" + str(class_name)])

            if (shot_id == debug_sid and self.config_instance.debug_flag == True):
                break

        results_cmc_np = np.array(results_cmc_l)

        if(self.config_instance.debug_flag == True):
            print(results_cmc_np)
            exit()
        # export results
        if (self.config_instance.save_eval_results == 1):
            self.exportCmcResults(vid_name, results_cmc_np)
        else:
            self.exportCmcResults(str(max_recall_id), results_cmc_np)

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

    def loadCmcResults(self, cmc_results_path):
        """
        Method for loading camera movments results as numpy array

        .. note::
            Only used in debug_mode.

        :param cmc_results_path: [required] path to results file of shot boundary detection module (vhh_sbd)
        :return: numpy array holding list of detected shots.
        """

        # open sbd results
        fp = open(cmc_results_path, 'r')
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
        elif (self.config_instance.save_eval_results == 1):

            print(self.config_instance.path_eval_results)
            fp = open(self.config_instance.path_eval_results + "/" + fName.split('/')[-1].split('.')[0] + ".csv", 'w')
        else:
            fp = open(self.config_instance.path_final_results + "/" + fName + ".csv", 'w')



        header = "vid_name;shot_id;start;end;cmc"
        fp.write(header + "\n")

        for i in range(0, len(cmc_results_np)):
            tmp_line = str(cmc_results_np[i][0])
            for c in range(1, len(cmc_results_np[i])):
                tmp_line = tmp_line + ";" + cmc_results_np[i][c]
            fp.write(tmp_line + "\n")

        fp.close()

    def export_shots_as_file(self, shots_np, dst_path="./vhh_mmsi_eval_db_tiny/shots/"):
        print("export shot as video")

        print(shots_np.shape)

        vid_name = shots_np[0][0]
        vid_instance = Video()
        vid_instance.load(self.config_instance.path_videos + "/" + vid_name)

        h = int(vid_instance.height)
        w = int(vid_instance.width)
        fps = int(vid_instance.frame_rate)

        print(h)
        print(w)
        print(fps)

        for i, data in enumerate(vid_instance.getFramesByShots(shots_np, preprocess=None)):
            frames_per_shots_np = data['Images']
            shot_id = data['sid']
            vid_name = data['video_name']
            start = data['start']
            stop = data['end']
            camera_movement_class = data["cmc_class"]

            print("######################")
            print(i)
            print(i % 32 == 0)
            print(f'sid: {shot_id}')
            print(f'vid_name: {vid_name}')
            print(f'frames_per_shot: {frames_per_shots_np.shape}')
            print(f'start: {start}, end: {stop}')
            print(f'camera_movement_class: {camera_movement_class}')

            #if (camera_movement_class == "PAN" or camera_movement_class == "TILT" or camera_movement_class == "pan" or camera_movement_class == "tilt"):
            if (camera_movement_class == "NA" or camera_movement_class == "na"):
                print("save video! ")

                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out = cv2.VideoWriter(dst_path + "/" + camera_movement_class + "_" + str(i) + ".avi", fourcc, 12, (w, h))

                for j in range(0, len(frames_per_shots_np)):
                    frame = frames_per_shots_np[j]
                    out.write(frame)

                out.release()




