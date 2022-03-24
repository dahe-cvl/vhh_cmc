from vhh_cmc.CMC import CMC
from vhh_cmc.Evaluation import Evaluation
from vhh_cmc.Configuration import Configuration
import numpy as np

eval_instance = None
exp_results = []


exp_file_list = [
                 "./config/config_cmc_evaluation_historian_1.yaml",
                 "./config/config_cmc_evaluation_historian_2.yaml",
                 "./config/config_cmc_evaluation_historian_3.yaml",
                 "./config/config_cmc_evaluation_historian_4.yaml",
                 "./config/config_cmc_evaluation_historian_5.yaml",
                 ]

'''
exp_file_list = [
                 #"./config/config_cmc_evaluation_vhh_mmsi_eval_db_tiny_XX.yaml",
                 "./config/config_cmc_evaluation_vhh_mmsi_eval_db_tiny_1.yaml",
                 "./config/config_cmc_evaluation_vhh_mmsi_eval_db_tiny_2.yaml",
                 "./config/config_cmc_evaluation_vhh_mmsi_eval_db_tiny_3.yaml",
                 "./config/config_cmc_evaluation_vhh_mmsi_eval_db_tiny_4.yaml",
                 "./config/config_cmc_evaluation_vhh_mmsi_eval_db_tiny_5.yaml",
                 "./config/config_cmc_evaluation_vhh_mmsi_eval_db_tiny_6.yaml",
                 "./config/config_cmc_evaluation_vhh_mmsi_eval_db_tiny_7.yaml",
                 "./config/config_cmc_evaluation_vhh_mmsi_eval_db_tiny_8.yaml",
                 ]

'''

'''
exp_file_list = [
                 "./config/config_cmc_evaluation_cmc_v2_1.yaml",
                 "./config/config_cmc_evaluation_cmc_v2_2.yaml",
                 "./config/config_cmc_evaluation_cmc_v2_3.yaml",
                 "./config/config_cmc_evaluation_cmc_v2_4.yaml",
                 "./config/config_cmc_evaluation_cmc_v2_5.yaml",
                 "./config/config_cmc_evaluation_cmc_v2_6.yaml",
                 "./config/config_cmc_evaluation_cmc_v2_7.yaml",
                 "./config/config_cmc_evaluation_cmc_v2_8.yaml",
                 ]              
'''

'''
exp_file_list = [
                 "./config/config_cmc_evaluation_cmc_v3_test.yaml",
                 #"./config/config_cmc_evaluation_cmc_v3_1.yaml",
                 #"./config/config_cmc_evaluation_cmc_v3_2.yaml",
                 #"./config/config_cmc_evaluation_cmc_v3_3.yaml",
                 #"./config/config_cmc_evaluation_cmc_v3_4.yaml",
                 #"./config/config_cmc_evaluation_cmc_v3_5.yaml",
                 #"./config/config_cmc_evaluation_cmc_v3_6.yaml",
                 #"./config/config_cmc_evaluation_cmc_v3_7.yaml",
                 #"./config/config_cmc_evaluation_cmc_v3_8.yaml",
                 ]
'''

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
        print(all_shots_np)
        vids_idx = np.unique(all_shots_np[:, :1])
        #print(all_shots_np)
        #print("/training_data/pan/vid_8242_sid_10_start_8560_stop_8645_classname_pan.avi" in all_shots_np)
        #exit()
        #i = 132
        #vids_idx = vids_idx[i:i+1]
        #print(vids_idx)
        #exit()

        for s, idx in enumerate(vids_idx.tolist()):    
            shot_idx = np.where(all_shots_np[:, :1] == idx)[0]
            shot_np = all_shots_np[shot_idx]
            shots_final = shot_np[:, :4]
            #print(shots_final)
            #continue
            cmc_instance.runOnSingleVideo(shots_per_vid_np=shots_final, max_recall_id=s+1)
        #exit()

    # run evaluation process
    accuracy, precision, recall, f1_score = eval_instance.run_evaluation(idx=None)

    # add exp results to list
    exp_results.append(["exp_" + str(i+1),
                        config_instance.mvi_mv_ratio,
                        config_instance.threshold_significance,
                        config_instance.threshold_consistency,
                        config_instance.mvi_window_size,
                        config_instance.region_window_size,
                        config_instance.active_threshold,
                        accuracy,
                        precision,
                        recall,
                        f1_score
                        ])
exp_results_np = np.array(exp_results)

# export overall results
eval_instance.exportExperimentResults("Experiments", exp_results_np)







