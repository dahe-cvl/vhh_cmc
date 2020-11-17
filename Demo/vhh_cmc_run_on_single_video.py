from cmc.CMC import CMC
import os

config_file = "/caa/Homes01/dhelm/working/vhh/releases/vhh_cmc/config/config_cmc.yaml"
cmc_instance = CMC(config_file)

results_path = "/data/share/datasets/vhh_mmsi_test_db_v2/annotations/sbd//"
results_file_list = os.listdir(results_path)
print(results_file_list)

for file in results_file_list[2:3]:
    print(file)
    shots_np = cmc_instance.loadSbdResults(results_path + file)
    #print(shots_np)
    max_recall_id = (file.split('.')[0])
    cmc_instance.runOnSingleVideo(shots_per_vid_np=shots_np, max_recall_id=max_recall_id)



