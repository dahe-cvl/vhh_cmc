import cv2
import numpy as np
from vhh_cmc.Configuration import Configuration
from vhh_cmc.PreProcessing import PreProcessing
import os
from vhh_cmc.OpticalFlow import OpticalFlow
from vhh_cmc.Video import Video

db_path = "/data/share/datasets/cmc_final_dataset_v2/"
path_videos = db_path + "training_data/"
class_name = "pan"
final_video_path = path_videos + class_name + "/"

all_shot_files = os.listdir(final_video_path)
print(all_shot_files)

path_shot_file = final_video_path + "/" + all_shot_files[25]  ## pan with moving objects
#path_shot_file = final_video_path + "/" + all_shot_files[15]  ## pan right
#path_shot_file = final_video_path + "/" + all_shot_files[12]  ## na

path_shot_file = "./WIN_20210614_20_57_41_Pro.mp4"
#path_shot_file = "./WIN_20210614_20_58_15_Pro.mp4"
#path_shot_file = "./WIN_20210615_13_37_57_Pro.mp4"

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

    x_filtered_mag_np, x_filtered_ang_np, filtered_u_np, filtered_v_np = optical_flow_instance.runDense_NEW()
    # class_name = optical_flow_instance.runDense()
    class_name = optical_flow_instance.predict_final_result_NEW(x_filtered_mag_np,
                                                                x_filtered_ang_np,
                                                                filtered_u_np,
                                                                filtered_v_np)

    print(class_name)

    exit()