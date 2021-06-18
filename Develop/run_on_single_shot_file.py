import cv2
import numpy as np
from vhh_cmc.Configuration import Configuration
from vhh_cmc.PreProcessing import PreProcessing
import os
from vhh_cmc.OpticalFlow import OpticalFlow
from vhh_cmc.Video import Video

db_path = "/data/share/datasets/cmc_final_dataset_v2/"

'''
path_videos = db_path + "training_data/"
class_name = "pan"
final_video_path = path_videos + class_name + "/"

all_shot_files = os.listdir(final_video_path)
print(all_shot_files)
'''

# na
path_shot_file = db_path + "training_data/na/EF-NS_063_OeFM_2.mp4"
#path_shot_file = db_path + "training_data/na/EF-NS_063_OeFM_3.mp4"
#path_shot_file = db_path + "training_data/na/EF-NS_063_OeFM_4.mp4"
#path_shot_file = db_path + "training_data/na/EF-NS_063_OeFM_5.mp4"

# tilt
#path_shot_file = db_path + "training_data/tilt/eb0d74d4-7e4b-4d1c-9539-89402ebc1334_27.mp4"  # --> fp
#path_shot_file = db_path + "training_data/tilt/eb0d74d4-7e4b-4d1c-9539-89402ebc1334_1.mp4" # --> tp
#path_shot_file = db_path + "training_data/tilt/d984c426-428d-4e04-adcd-33ea2514ff00_102.mp4" # --> fp  sehr schwierig mix tilt/pan
#path_shot_file = db_path + "training_data/tilt/tilt_30_21196.mp4" # --> fp  sehr sehr langsam

# pan
#path_shot_file = db_path + "training_data/pan/1371a561-6b19-4d69-8210-1347ca75eb90_12.mp4"  # --> fp  sehr langsam
#path_shot_file = db_path + "training_data/pan/1371a561-6b19-4d69-8210-1347ca75eb90_17.mp4"  # --> fp  sehr langsam
#path_shot_file = db_path + "training_data/pan/pan_a1dc1433-b00e-4b62-a5ec-75db765ad34d_70.mp4"  # --> fp  moving objects
#path_shot_file = db_path + "training_data/pan/ae9bb5fe-5da9-49cb-97ec-3b0b54e50816_15.mp4"  # --> tp aber sehr knapp moving objects


'''
path_shot_file = final_video_path + "/" + all_shot_files[25]  ## pan with moving objects
#path_shot_file = final_video_path + "/" + all_shot_files[15]  ## pan right
#path_shot_file = final_video_path + "/" + all_shot_files[45]  ##

#path_shot_file = final_video_path + "/" + all_shot_files[4]  ## na with moving objects
#path_shot_file = final_video_path + "/" + all_shot_files[1]  ## na
#path_shot_file = final_video_path + "/" +"3ab4dd5e-1323-4535-a023-b6c6cedbee53_17.mp4"

#path_shot_file = "./WIN_20210614_20_57_41_Pro.mp4"
#path_shot_file = "./WIN_20210614_20_58_15_Pro.mp4"
#path_shot_file = "./WIN_20210615_13_37_57_Pro.mp4"
'''

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

    class_name = optical_flow_instance.runDense_v3()
    print(class_name)

    exit()