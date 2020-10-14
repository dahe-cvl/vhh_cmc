import cv2
import numpy as np
from vhh_cmc.Configuration import Configuration
from vhh_cmc.PreProcessing import PreProcessing
import os
from vhh_cmc.OpticalFlow import OpticalFlow


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
        else:
            shots_np = shots_per_vid_np

        if (len(shots_np) == 0):
            print("ERROR: there must be at least one shot in the list!")
            exit()

        if (self.config_instance.debug_flag == True):
            num_shots = 1
        else:
            num_shots = len(shots_np)

        # read all frames of video
        vid_name = shots_np[0][1]

        if(self.config_instance.save_eval_results == 1):
            print("Evaluation mode is activated ...")
            cap = cv2.VideoCapture(vid_name)
        else:
            print(self.config_instance.path_videos + "/" + vid_name)
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
                frame = self.pre_processing_instance.applyTransformOnImg(frame)
                frame_l.append(frame)
                #cv2.imshow("asd", frame)
                #cv2.waitKey(1)
                #if(cnt == 100):
                #    exit()
            else:
                break

        all_frames_np = np.array(frame_l)
        #print(all_frames_np.shape)

        #all_frames_np = all_frames_np[1240:1478,:,:,:]
        shot_start_idx = 0  # used in debugging mode - to select specific shot
        results_cmc_l = []
        for idx in range(shot_start_idx, shot_start_idx + num_shots):
            #print(shots_np[idx])
            shot_id = int(shots_np[idx][0])
            vid_name = str(shots_np[idx][1])
            start = int(shots_np[idx][2])
            stop = int(shots_np[idx][3])
            shot_frames_np = all_frames_np[start:stop + 1, :, :, :]
            shot_len = stop - start

            MIN_NUMBER_OF_FRAMES_PER_SHOT = 10
            if(shot_len <= MIN_NUMBER_OF_FRAMES_PER_SHOT ):
                #print("shot length is too small!")
                class_name = "NA"
            else:
                # add new optical flow version
                optical_flow_instance = OpticalFlow(video_frames=shot_frames_np,
                                                    algorithm="orb",
                                                    config_instance=self.config_instance)
                mag_l, angles_l = optical_flow_instance.run()
                class_name = optical_flow_instance.predict_final_result(mag_l,
                                                                        angles_l,
                                                                        self.config_instance.class_names)


            '''
            # run optical flow process
            optical_flow_instance = OpticalFlow(video_frames=shot_frames_np,
                                                fPath=self.config_instance.path_videos + "/" + vid_name,
                                                debug_path=self.config_instance.path_raw_results,
                                                sf=start,
                                                ef=stop,
                                                mode=self.config_instance.mode,
                                                sensitivity=self.config_instance.sensitivity,
                                                specificity=self.config_instance.specificity,
                                                border=self.config_instance.border,
                                                number_of_features=self.config_instance.number_of_features,
                                                angle_diff_limit=self.config_instance.angle_diff_limit,
                                                config=None)
            pan_list, tilt_list = optical_flow_instance.run()

            print(pan_list)
            print(tilt_list)

            number_of_all_frames = abs(start - stop)
            if number_of_all_frames == 0:
                number_of_all_frames = 0.000000000001

            number_of_pan_frames = 0
            for sf, ef in pan_list:
                diff = abs(sf - ef)
                number_of_pan_frames = number_of_pan_frames + diff
            pans_score = int((number_of_pan_frames * 100) / number_of_all_frames)
            print(pans_score)

            number_of_tilt_frames = 0
            for sf, ef in tilt_list:
                diff = abs(sf - ef)
                number_of_tilt_frames = number_of_tilt_frames + diff
            tilts_score = int((number_of_tilt_frames * 100) / number_of_all_frames)
            print(tilts_score)

            #if(self.config_instance.save_eval_results == 1):
            #    if (pans_score >= threshold):
            #        class_name = self.config_instance.class_names[0]
            #    else:
            #        class_name = self.config_instance.class_names[1]
            #else:
            threshold = 60
            if (pans_score >= threshold):
                class_name = self.config_instance.class_names[0]
            elif(tilts_score >= threshold):
                class_name = self.config_instance.class_names[1]
            elif (pans_score >= threshold) and (tilts_score >= threshold):
                class_name = self.config_instance.class_names[2]
            else:
                class_name = self.config_instance.class_names[2]
            '''

            # prepare results
            print(str(vid_name) + ";" + str(shot_id) + ";" + str(start) + ";" + str(stop) + ";" + str(class_name))
            results_cmc_l.append([str(vid_name) + ";" + str(shot_id) + ";" + str(start) + ";" + str(stop) + ";" + str(class_name)])
            #exit()

            '''
            # save raw results
            if(self.config_instance.save_raw_results == 1):
                optical_flow_instance.to_csv(
                    "/".join([self.config_instance.path_raw_results, self.config_instance.path_prefix_raw_results + str(vid_name.split('/')[-1]) + ".csv"]))
                optical_flow_instance.to_png(
                    "/".join([self.config_instance.path_raw_results, self.config_instance.path_prefix_raw_results + str(vid_name) + ".png"]))
                optical_flow_instance.to_avi(
                    "/".join([self.config_instance.path_raw_results, self.config_instance.path_prefix_raw_results + str(vid_name) + ".avi"]))
            '''
        results_cmc_np = np.array(results_cmc_l)
        print(results_cmc_np)
        #exit()

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
