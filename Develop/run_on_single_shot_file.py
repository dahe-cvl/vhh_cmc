import cv2
import numpy as np
from vhh_cmc.Configuration import Configuration
from vhh_cmc.PreProcessing import PreProcessing
import os
from vhh_cmc.OpticalFlow import OpticalFlow
from vhh_cmc.Video import Video

#db_path = "/data/share/datasets/cmc_final_dataset_v2/"

# na
#path_shot_file = db_path + "training_data/na/EF-NS_063_OeFM_2.mp4"
#path_shot_file = db_path + "training_data/na/EF-NS_063_OeFM_3.mp4"
#path_shot_file = db_path + "training_data/na/EF-NS_063_OeFM_4.mp4"
#path_shot_file = db_path + "training_data/na/EF-NS_063_OeFM_5.mp4"

# tilt
#path_shot_file = db_path + "training_data/tilt/eb0d74d4-7e4b-4d1c-9539-89402ebc1334_27.mp4"  # --> fp --> KOMISCH
#path_shot_file = db_path + "training_data/tilt/eb0d74d4-7e4b-4d1c-9539-89402ebc1334_1.mp4" # --> tp --> KOMISCH
#path_shot_file = db_path + "training_data/tilt/d984c426-428d-4e04-adcd-33ea2514ff00_102.mp4" # --> fp  sehr schwierig mix tilt/pan
#path_shot_file = db_path + "training_data/tilt/tilt_30_21196.mp4" # --> fp  sehr sehr langsam
#path_shot_file = "./WIN_20210621_08_34_11_Pro.mp4"
#path_shot_file = "./WIN_20210621_08_48_15_Pro.mp4"

# pan
#path_shot_file = db_path + "training_data/pan/1371a561-6b19-4d69-8210-1347ca75eb90_12.mp4"  # --> fp  sehr langsam
#path_shot_file = db_path + "training_data/pan/1371a561-6b19-4d69-8210-1347ca75eb90_17.mp4"  # --> fp  sehr langsam
#path_shot_file = db_path + "training_data/pan/pan_a1dc1433-b00e-4b62-a5ec-75db765ad34d_70.mp4"  # --> fp  moving objects
#path_shot_file = db_path + "training_data/pan/ae9bb5fe-5da9-49cb-97ec-3b0b54e50816_15.mp4"  # --> tp aber sehr knapp moving objects
#path_shot_file = "./WIN_20210621_08_23_51_Pro.mp4"


#db_path = "/data/share/datasets/vhh_mmsi_eval_db_tiny/"
#path_shot_file = db_path + "training_data/pan/pan_3.avi"
#path_shot_file = db_path + "training_data/pan/pan_4.avi"
#path_shot_file = db_path + "training_data/pan/pan_9.avi"
#path_shot_file = db_path + "training_data/pan/PAN_12.avi"
#path_shot_file = db_path + "training_data/pan/pan_17.avi"
#path_shot_file = db_path + "training_data/pan/pan_18.avi"
#path_shot_file = db_path + "training_data/pan/PAN_27.avi"
#path_shot_file = db_path + "training_data/pan/PAN_31.avi"
#path_shot_file = db_path + "training_data/pan/pan_39.avi"
#path_shot_file = db_path + "training_data/pan/pan_51.avi"
#path_shot_file = db_path + "training_data/pan/pan_52.avi"
#path_shot_file = db_path + "training_data/pan/pan_88.avi"
#path_shot_file = db_path + "training_data/pan/pan_108.avi"
#path_shot_file = db_path + "training_data/pan/pan_110.avi"
#path_shot_file = db_path + "training_data/pan/pan_112.avi"
#path_shot_file = db_path + "training_data/pan/pan_114.avi"
#path_shot_file = db_path + "training_data/pan/PAN_121.avi"
#path_shot_file = db_path + "training_data/pan/PAN_146.avi"
#path_shot_file = db_path + "training_data/pan/pan_147.avi"


#path_shot_file = db_path + "training_data/tilt/tilt_73.avi"  # lange static scene
#path_shot_file = db_path + "training_data/tilt/tilt_69.avi"  # lange static scene
#path_shot_file = db_path + "training_data/tilt/tilt_42.avi"    # lange static scene
#path_shot_file = db_path + "training_data/tilt/tilt_34.avi"  # lange static scene
#path_shot_file = db_path + "training_data/tilt/tilt_24.avi"
#path_shot_file = db_path + "training_data/tilt/tilt_22.avi"
#path_shot_file = db_path + "training_data/tilt/tilt_13.avi"
#path_shot_file = db_path + "training_data/tilt/tilt_11.avi"
#path_shot_file = db_path + "training_data/tilt/TILT_85.avi"

#path_shot_file = db_path + "training_data/na/NA_9.avi" # tilt
#path_shot_file = db_path + "training_data/na/NA_4.avi" # tilt
#path_shot_file = db_path + "training_data/na/NA_29.avi" # pan
#path_shot_file = db_path + "training_data/na/NA_17.avi" # pan
#path_shot_file = db_path + "training_data/na/NA_23.avi" # yes
#path_shot_file = db_path + "training_data/na/NA_10.avi" # pan
#path_shot_file = db_path + "training_data/na/NA_19.avi"  # tilt --> problem

db_path = "/data/vhh_release/release_v1_3_0/vhh_core/videos/"
path_shot_file = db_path + "8256.m4v"

print(path_shot_file)

# load video
vid_instance = Video()
vid_instance.load(vidFile=path_shot_file)
vid_instance.printVIDInfo()

shots_np = np.array([[0, 0, 0, vid_instance.number_of_frames]])

config_file = "./config/config_cmc_debug.yaml"
config_instance = Configuration(config_file)
config_instance.loadConfig()

pre_processing_instance = PreProcessing(config_instance=config_instance)

for data in vid_instance.getFramesByShots(shots_np, preprocess=pre_processing_instance.applyTransformOnImg):
    frames_per_shots_np = data['Images']
    shot_id = data['sid']
    vid_name = data['video_name']
    start = data['start']
    stop = data['end']

    print(f'start: {start}, end: {stop}')
    print(frames_per_shots_np.shape)

    # add new optical flow version
    optical_flow_instance = OpticalFlow(video_frames=frames_per_shots_np,
                                        algorithm="orb",
                                        config_instance=config_instance)

    class_name = optical_flow_instance.runDense()
    print(class_name)
