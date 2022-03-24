import os
import numpy as np
from vhh_cmc.Video import Video

def csvWriter(dst_folder="", name="metrics_history.log", entries_list=None):
    if (entries_list == None):
        print("ERROR: entries_list must have a valid entry!")

    # prepare entry_line
    entry_line = ""
    for i in range(0, len(entries_list)):
        tmp = entries_list[i]
        entry_line = entry_line + ";" + str(tmp)

    fp = open(dst_folder + "/" + str(name), 'a')
    fp.write(entry_line + "\n")
    fp.close()


video_path = "/data/ext/VHH/datasets/HistShotDS_V2/eval_cmc/training_data/"
dst_folder= "/data/ext/VHH/datasets/HistShotDS_V2/eval_cmc/annotation_new/"

class_name_list = os.listdir(video_path)

print(class_name_list)
#class_name_list.remove("track")
print(class_name_list)

samples_l = []
for i, class_name in enumerate(class_name_list):
    samples_list = os.listdir(video_path + class_name)
    print(samples_list)

    if("pan" == class_name): label = 0
    if ("tilt" == class_name): label = 1
    if ("na" == class_name): label = 2
    #if ("track" == class_name): label = 3

    for sample in samples_list:
        path = os.path.join(video_path, class_name, sample)
        vid_instance = Video()
        vid_instance.load(vidFile=path)
        path_split = path.split('/')[-3:]
        final_path = os.path.join(path_split[0], path_split[1], path_split[2])
        samples_l.append([final_path, label, 0, int(vid_instance.number_of_frames)])
samples_np = np.array(samples_l)

print(samples_np)

for entry in samples_l:
    csvWriter(dst_folder=dst_folder, name="annotations_NNNNN.csv", entries_list=entry)