from vhh_cmc.CMC import CMC

cmc_instance = CMC(config_file="/caa/Homes01/dhelm/working/vhh_release_test/vhh_cmc/config/config_cmc_debug.yaml")
cmc_instance.runOnSingleVideo(shots_per_vid_np=None, max_recall_id=1)


