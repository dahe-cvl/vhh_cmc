import os
import wandb


class WandB_Logger(object):
    def __init__(self):
        print("create instance of wandb logger ...")

        print("set needed environment variables ... ")
        os.environ["WANDB_API_KEY"] = "7ab633461569be4b899d278d59e518ad4ad26361"
        #os.environ["WANDB_MODE"] = "dryrun"

        self.exp_config = None
        self.experiment_dir = None
        self.project_name = None
        self.experiment_name = None

        print("set needed environment variables ... ")

    def set_config_params(self, config_dict):
        print("set (optional) model configuration params for logging ... ")
        '''
        {
        "epochs": n_epochs, 
        "batch_size": batch_size, 
        "learning_rate": lr, 
        "wDecay": wDecay
        }
        '''
        self.exp_config = config_dict

    def set_general_experiment_params(self, exp_config, experiment_dir, project_name, experiment_name):
        print("set general experiment params ... ")
        self.exp_config = exp_config
        self.experiment_dir = experiment_dir
        self.project_name = project_name
        self.experiment_name = experiment_name

    def initialize_logger(self):
        print("initialize looger ... ")
        if (self.experiment_dir == None or 
            self.project_name == None or 
            self.experiment_name == None or 
            self.exp_config == None):
            print("ERROR: you have to set valid experiment settings. (call method set_general_experiment_params(...)!")
            exit()

        wandb.init(dir=self.experiment_dir, 
                   project=self.project_name, 
                   name=self.experiment_name, 
                   config=self.exp_config)

    def log_batch_sequence(self, inputs, fps=4, format_type='mp4'):
        #####################################
        # Log wand: sequences 
        # input_shape: channels, time, width, height
        # convert to:
        # time, channels, width, height
        #####################################
        for b in range(0, len(inputs)):
            seq_np = inputs[b].detach().cpu().numpy()
            seq_final_np = np.transpose(seq_np, (1, 0, 2, 3))
            wandb.log({"video_" + str(b): wandb.Video(seq_final_np, fps=fps, format=format_type)})

    def log_metrics(self, metrics_dict):
        '''
        example: 
        {
        "train_loss": tLoss_sum / len(trainloader),
        "train_acc": tAcc_sum / len(trainloader),
        "val_loss": vLoss_sum / len(validloader),
        "val_acc": vAcc_sum / len(validloader)
        }
        '''
        wandb.log(metrics_dict)

    def log_model(self, model):
         wandb.watch(model)



    