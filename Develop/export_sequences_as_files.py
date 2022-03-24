from vhh_cmc.CMC import CMC
import os
import numpy as np

config_file = "./config/config_cmc.yaml"
cmc_instance = CMC(config_file)

results_path = "/data/ext/VHH/datasets/HistShotDS_V2/eval_cmc/annotations_per_vid_12022022/"
#results_path = "./debug/"
dst_path = "/data/ext/VHH/datasets/HistShotDS_V2/eval_cmc/training_data_12022022/"

results_file_list = os.listdir(results_path)
results_file_list.sort()
print(np.array(results_file_list))
#exit()


for file in results_file_list:
    print(results_path + file)
    #shots_np = cmc_instance.loadCmcResultsFromJson(results_path + file)
    shots_np = cmc_instance.loadCmcResults(results_path + file)
    print(shots_np)
    
    if(len(shots_np) > 0):    
        max_recall_id = file.split('.')[0]
        #print(max_recall_id)
        #print(shots_np)
        cmc_instance.export_cmc_sequences_as_file(seq_np=shots_np, dst_path=dst_path)