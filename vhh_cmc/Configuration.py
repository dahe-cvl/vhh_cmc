import yaml


class Configuration:
    """
    This class is needed to read the configuration parameters specified in the configuration.yaml file.
    The instance of the class is holding all parameters during runtime.

    .. note::
       e.g. ./config/config_vhh_test.yaml

        the yaml file is separated in multiple sections
        config['Development']
        config['PreProcessing']
        config['CmcCore']
        config['Evaluation']

        whereas each section should hold related and meaningful parameters.
    """

    def __init__(self, config_file: str):
        """
        Constructor

        :param config_file: [required] path to configuration file (e.g. PATH_TO/config.yaml)
                                       must be with extension ".yaml"
        """
        print("create instance of configuration ... ")

        if(config_file.split('.')[-1] != "yaml"):
            print("Configuration file must have the extension .yaml!")

        self.config_file = config_file

        # developer_config section
        self.debug_flag = -1
        self.sbd_results_path = None
        self.save_debug_pkg_flag = -1
        self.path_debug_results = None

        # pre-processing section
        self.flag_convert2Gray = -1
        self.center_crop_flag = -1
        self.flag_downscale = -1
        self.resize_dim = None

        # optical flow section
        self.mvi_mv_ratio = -1
        self.threshold_significance = -1
        self.threshold_consistency = -1
        self.mvi_window_size = -1
        self.region_window_size = -1
        self.active_threshold = -1

        # stc_core_config section
        self.class_names = None

        self.save_raw_results = -1
        self.path_postfix_raw_results = None
        self.path_prefix_raw_results = None
        self.path_raw_results = None

        self.save_final_results = -1
        self.path_prefix_final_results = None
        self.path_postfix_final_results = None
        self.path_final_results = None

        self.path_videos = None

        # evaluation section
        self.path_eval_dataset = None
        self.path_eval_results = ""
        self.save_eval_results = -1
        self.path_gt_data = None

    def loadConfig(self):
        """
        Method to load configurables from the specified configuration file
        """

        fp = open(self.config_file, 'r')
        config = yaml.load(fp, Loader=yaml.BaseLoader)

        developer_config = config['Development']
        pre_processing_config = config['PreProcessing']
        optical_flow_config = config['OpticalFlow']
        cmc_core_config = config['CmcCore']
        evaluation_config = config['Evaluation']

        # developer_config section
        self.debug_flag = int(developer_config['DEBUG_FLAG'])
        self.sbd_results_path = developer_config['SBD_RESULTS_PATH']
        self.save_debug_pkg_flag = int(developer_config['SAVE_DEBUG_PKG'])
        self.path_debug_results = developer_config['PATH_DEBUG_RESULTS']

        # pre-processing section
        self.flag_convert2Gray = int(pre_processing_config['CONVERT2GRAY_FLAG'])
        self.center_crop_flag = int(pre_processing_config['CENTER_CROP_FLAG'])
        self.flag_downscale = int(pre_processing_config['DOWNSCALE_FLAG'])
        self.resize_dim = (int(pre_processing_config['RESIZE_DIM'].split(',')[0]),
                           int(pre_processing_config['RESIZE_DIM'].split(',')[1]))

        # optical flow section
        self.mvi_mv_ratio = float(optical_flow_config['MVI_MV_RATIO'])
        self.threshold_significance = float(optical_flow_config['THRESHOLD_SIGNIFICANCE'])
        self.threshold_consistency = float(optical_flow_config['THRESHOLD_CONSISTENCY'])
        self.mvi_window_size = int(optical_flow_config['MVI_WINDOW_SIZE'])
        self.region_window_size = int(optical_flow_config['REGION_WINDOW_SIZE'])
        self.active_threshold = float(optical_flow_config['ACTIVE_THRESHOLD'])

        # cmc_core_config section
        self.class_names = cmc_core_config['CLASS_NAMES']

        self.save_raw_results = int(cmc_core_config['SAVE_RAW_RESULTS'])
        self.path_postfix_raw_results = cmc_core_config['POSTFIX_RAW_RESULTS']
        self.path_prefix_raw_results = cmc_core_config['PREFIX_RAW_RESULTS']
        self.path_raw_results = cmc_core_config['PATH_RAW_RESULTS']

        self.save_final_results = int(cmc_core_config['SAVE_FINAL_RESULTS'])
        self.path_prefix_final_results = cmc_core_config['PREFIX_FINAL_RESULTS']
        self.path_postfix_final_results = cmc_core_config['POSTFIX_FINAL_RESULTS']
        self.path_final_results = cmc_core_config['PATH_FINAL_RESULTS']

        self.path_videos = cmc_core_config['PATH_VIDEOS']

        # evaluation section
        self.path_eval_dataset = evaluation_config['PATH_EVAL_DATASET']
        self.path_eval_results = evaluation_config['PATH_EVAL_RESULTS']
        self.save_eval_results = int(evaluation_config['SAVE_EVAL_RESULTS'])
        self.path_gt_data = evaluation_config['PATH_GT_ANNOTATIONS']
