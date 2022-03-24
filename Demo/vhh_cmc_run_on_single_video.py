from vhh_cmc.CMC import CMC
import os

config_file = "./config/config_cmc.yaml"
cmc_instance = CMC(config_file)

if(cmc_instance.config_instance.debug_flag == True):
    print("DEBUG MODE activated!")
    sbd_results_file = cmc_instance.config_instance.sbd_results_path
    shots_np = cmc_instance.loadSbdResults(sbd_results_file)
    max_recall_id = int(shots_np[0][0].split('.')[0])
    #max_recall_id = "efilms_test_pan"
    #shots_np = []
    cmc_instance.runOnSingleVideo(shots_per_vid_np=shots_np, max_recall_id=max_recall_id)
else:
    #results_path = "/data/share/datasets/vhh_mmsi_test_db_v3/annotations/sbd/"
    results_path = "/data/ext/VHH/datasets/HistShotDS_V2/annotations/automatic/"

    import glob
    results_file_list = glob.glob(os.path.join(results_path, "*-shot_annotations.json"))
    #results_file_list = os.listdir(results_path)
    results_file_list.sort()
    print(results_file_list)

    for file in results_file_list:
        #shots_np = cmc_instance.loadSbdResults(results_path + file)
        #shots_np = cmc_instance.loadStcResultsFromJson(results_path + file)
        shots_np = cmc_instance.loadStcResultsFromJson(file)

        max_recall_id = file.split('/')[-1]
        #print(max_recall_id)
        max_recall_id = max_recall_id.split('.')[0]
        #print(max_recall_id)
        max_recall_id = max_recall_id.split('-')[0]
        #print(max_recall_id)
        cmc_instance.runOnSingleVideo(shots_per_vid_np=shots_np, max_recall_id=max_recall_id)
