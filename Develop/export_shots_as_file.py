from vhh_cmc.CMC import CMC
import os

config_file = "./config/config_cmc_debug.yaml"
cmc_instance = CMC(config_file)

results_path = "/data/share/datasets/vhh_mmsi_test_db_v3/annotations/cmc/"
results_file_list = os.listdir(results_path)
results_file_list.sort()
print(results_file_list)

for file in results_file_list:
    shots_np = cmc_instance.loadCmcResults(results_path + file)
    max_recall_id = int(file.split('.')[0])
    cmc_instance.export_shots_as_file(shots_np)
