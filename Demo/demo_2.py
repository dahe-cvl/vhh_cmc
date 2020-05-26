import sys
from CMC_pkg import cmc_io, cmc_ofc

def run(config):

    ofc = cmc_ofc.OFCMClassifier(config=config)
    ofc.run()

    ofc.to_csv("/".join([config["OUTPUT_PATH"], config["OUTPUT_CSV"]]))
    ofc.to_png("/".join([config["OUTPUT_PATH"], config["OUTPUT_PLOT"]]))
    ofc.to_avi("/".join([config["OUTPUT_PATH"], config["OUTPUT_VIDEO"]]))


if __name__ == "__main__":

    # parse the yaml file
    print("parsing config file ...")
    config = cmc_io.parse_config()

    print(config)

    if sys.argv[1:].__len__() > 0:
        print("parsing system arguments ...")
        config = cmc_io.parse_sys_arg(sys.argv[1:], config)
        print(config)

    # add current working directory to input/output path.
    from os import getcwd

    wd = getcwd()
    config["INPUT_PATH"] = wd + config["INPUT_PATH"]
    config["OUTPUT_PATH"] = wd + config["OUTPUT_PATH"]

    # update output paths if they are not set
    name = config["INPUT_VIDEO"].split('.')[0]
    if config["OUTPUT_VIDEO"] == "":
        config["OUTPUT_VIDEO"] = "_".join([name, str(config["BEGIN_FRAME"]), str(config["END_FRAME"]), "ann.avi"])
    if config["OUTPUT_PLOT"] == "":
        config["OUTPUT_PLOT"] = "_".join([name, str(config["BEGIN_FRAME"]), str(config["END_FRAME"]), "movements.png"])

    run(config)
