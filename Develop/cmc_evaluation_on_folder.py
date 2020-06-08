from cmc.CMC import CMC
from cmc.Evaluation import Evaluation
from cmc.Configuration import Configuration
import numpy as np

cmc_instance = CMC(config_file="../config/config_cmc_evaluation.yaml")

# load video shots from specified dataset folder as numpy
config_instance = Configuration(config_file="../config/config_cmc_evaluation.yaml")
config_instance.loadConfig()
eval_instance = Evaluation(config_instance=config_instance)
eval_instance.load_dataset()

# run cmc classification process
# (e.g. sid | movie_name | start | end )
# max_recall_id can be ignored in eval mode
# shots_per_vid_np must be a numpy with the follwoing shape (Nx4 --> N >= 1)
#
ACTIVE_FLAG = True
if(ACTIVE_FLAG == True):
    shots_np = eval_instance.final_dataset_np[:, :4]
    for i in range(0, len(shots_np)):
        shots_per_video = np.expand_dims((shots_np[i]), axis=0)
        cmc_instance.runOnSingleVideo(shots_per_vid_np=shots_per_video, max_recall_id=8)

# run evaluation process
eval_instance.run_evaluation()


