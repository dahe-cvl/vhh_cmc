from cmc.CMC import CMC
import os

config_file = "/home/dhelm/VHH_Develop/pycharm_vhh_cmc/config/config_cmc.yaml"
cmc_instance = CMC(config_file)

results_path = "/data/share/maxrecall_vhh_mmsi/develop/videos/results/sbd/final_results/"
results_file_list = os.listdir(results_path)
print(results_file_list)

for file in results_file_list:
    print(file)
    shots_np = cmc_instance.loadSbdResults(results_path + file)
    print(shots_np)
    max_recall_id = int(file.split('.')[0])
    cmc_instance.runOnSingleVideo(shots_per_vid_np=shots_np, max_recall_id=max_recall_id)


