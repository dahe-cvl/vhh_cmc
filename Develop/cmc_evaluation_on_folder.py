from vhh_cmc.CMC import CMC
from vhh_cmc.Evaluation import Evaluation
from vhh_cmc.Configuration import Configuration
import numpy as np

eval_instance = None
exp_results = []

'''
exp_file_list = [#"./config/config_cmc_exp_1_vhh_mmsi_test_db_v2.yaml",
                 #"./config/config_cmc_exp_2_vhh_mmsi_test_db_v2.yaml",
                 #"./config/config_cmc_exp_3_vhh_mmsi_test_db_v2.yaml",
                 #"./config/config_cmc_exp_4_vhh_mmsi_test_db_v2.yaml",
                 #"./config/config_cmc_exp_6_vhh_mmsi_test_db_v2.yaml",
                 #"./config/config_cmc_exp_5_vhh_mmsi_test_db_v2.yaml",
                 "./config/config_cmc.yaml",
                 #"./config/config_cmc_exp_7_vhh_mmsi_test_db_v2.yaml",
                 #"./config/config_cmc_exp_8_vhh_mmsi_test_db_v2.yaml"
                 ] #vhh_mmsi_test_db_v2_final_results_mag_th_2
'''
exp_file_list = [#"./config/config_cmc_exp_1_cmc_final_db_v2.yaml",
                 #"./config/config_cmc_exp_2_cmc_final_db_v2.yaml",
                 #"./config/config_cmc_exp_3_cmc_final_db_v2.yaml",
                 #"./config/config_cmc_exp_4_cmc_final_db_v2.yaml",
                 #"./config/config_cmc_exp_6_cmc_final_db_v2.yaml",
                 #"./config/config_cmc_exp_5_cmc_final_db_v2.yaml",
                 #"./config/config_cmc_exp_7_cmc_final_db_v2.yaml",
                 #"./config/config_cmc_exp_8_cmc_final_db_v2.yaml",
                 "./config/config_cmc_exp_9_cmc_final_db_v2.yaml",
                 "./config/config_cmc_exp_10_cmc_final_db_v2.yaml",
                 #"./config/config_cmc.yaml",
                 ] #vhh_mmsi_test_db_v2_final_results_mag_th_2

for i, exp_file in enumerate(exp_file_list):
    cmc_instance = CMC(config_file=exp_file)

    # load video shots from specified dataset folder as numpy
    config_instance = Configuration(config_file=exp_file)
    config_instance.loadConfig()

    eval_instance = Evaluation(config_instance=config_instance)
    eval_instance.load_cmc_eval_db_v2()
    #eval_instance.load_vhhmmsi_GT_V2_db()

    # run cmc classification process
    # (e.g. sid | movie_name | start | end )
    # max_recall_id can be ignored in eval mode
    # shots_per_vid_np must be a numpy with the follwoing shape (Nx4 --> N >= 1)
    #
    ACTIVE_FLAG = True
    if(ACTIVE_FLAG == True):
        all_shots_np = eval_instance.final_dataset_np
        vids_idx = np.unique(all_shots_np[:, :1])   
          
        for s, idx in enumerate(vids_idx.tolist()):    
            shot_idx = np.where(all_shots_np[:, :1] == idx)[0]
            shot_np = all_shots_np[shot_idx]
            shots_final = shot_np[:, :4]
            cmc_instance.runOnSingleVideo(shots_per_vid_np=shots_final, max_recall_id=s+1)

    # run evaluation process
    accuracy, precision, recall, f1_score = eval_instance.run_evaluation(idx=None)

    # add exp results to list
    exp_results.append(["exp_" + str(i+1),
                        config_instance.min_magnitude_threshold,
                        config_instance.distance_threshold,
                        accuracy,
                        precision,
                        recall,
                        f1_score
                        ])
exp_results_np = np.array(exp_results)

# export overall results
eval_instance.exportExperimentResults("Experiments", exp_results_np)







