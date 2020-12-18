from vhh_cmc.CMC import CMC
import os

config_file = "./config/config_cmc.yaml"
cmc_instance = CMC(config_file)

if(cmc_instance.config_instance.debug_flag == True):
    print("DEBUG MODE activated!")
    sbd_results_file = cmc_instance.config_instance.sbd_results_path
    shots_np = cmc_instance.loadSbdResults(sbd_results_file)
    max_recall_id = int(shots_np[0][0].split('.')[0])
    cmc_instance.runOnSingleVideo(shots_per_vid_np=shots_np, max_recall_id=max_recall_id)
else:
    results_path = "/data/share/datasets/vhh_mmsi_test_db_v3/annotations/sbd/"
    results_file_list = os.listdir(results_path)
    results_file_list.sort()
    print(results_file_list)

    for file in results_file_list[1:2]:
        shots_np = cmc_instance.loadSbdResults(results_path + file)
        max_recall_id = int(file.split('.')[0])
        cmc_instance.runOnSingleVideo(shots_per_vid_np=shots_np, max_recall_id=max_recall_id)
