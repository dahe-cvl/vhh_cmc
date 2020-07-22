from cmc.CMC import CMC
from cmc.Evaluation import Evaluation
from cmc.Configuration import Configuration
import numpy as np

eval_instance = None
exp_results = []

exp_file_list = ["/home/dhelm/VHH_Develop/pycharm_vhh_cmc/config/config_cmc_exp1.yaml",
                 "/home/dhelm/VHH_Develop/pycharm_vhh_cmc/config/config_cmc_exp2.yaml",
                 "/home/dhelm/VHH_Develop/pycharm_vhh_cmc/config/config_cmc_exp3.yaml"]

for i, exp_file in enumerate(exp_file_list):
    cmc_instance = CMC(config_file=exp_file)

    # load video shots from specified dataset folder as numpy
    config_instance = Configuration(config_file=exp_file)
    config_instance.loadConfig()

    eval_instance = Evaluation(config_instance=config_instance)
    eval_instance.load_dataset_V2()

    # run cmc classification process
    # (e.g. sid | movie_name | start | end )
    # max_recall_id can be ignored in eval mode
    # shots_per_vid_np must be a numpy with the follwoing shape (Nx4 --> N >= 1)
    #
    ACTIVE_FLAG = True
    if(ACTIVE_FLAG == True):
        shots_l = eval_instance.final_dataset_np
        #print(shots_l)
        for shots in shots_l:
            cmc_instance.runOnSingleVideo(shots_per_vid_np=shots[:, :4], max_recall_id=8)

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







