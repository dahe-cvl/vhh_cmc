import sys
from cmc import cmc_io, CMC

def run(config_man):
   # ofc_man = cmc_ofc.OFCMClassifier(config=config_man)
   # ofc_man.run_manual_evaluation()
   # ofc_man.to_csv("/".join([config_man["OUTPUT_PATH"], config_man["OUTPUT_CSV"]]))
   # ofc_man.to_png("/".join([config_man["OUTPUT_PATH"], config_man["OUTPUT_PLOT"]]))

    ofc = CMC.OFCMClassifier(config=config)
    ofc.run()

    ofc.to_csv("/".join([config["OUTPUT_PATH"], config["OUTPUT_CSV"]]))
    ofc.to_png("/".join([config["OUTPUT_PATH"], config["OUTPUT_PLOT"]]))

# loading from csv
import pandas

def load(ifile):
    try:
        df = pandas.read_excel(ifile, header=0)
    except:
        df = pandas.read_csv(ifile, header=0, delimiter=";")
    print(df)
    return df

if __name__ == "__main__":
    # parse the yaml file
    print("parsing config file ...")
    #config_man = cmc_io.parse_config("config_manually.yaml")
    config = cmc_io.parse_config("config_manually.yaml")

    # add current working directory to input/output path.
    from os import getcwd

    wd = getcwd()
    config["INPUT_PATH"] = wd + config["INPUT_PATH"]
    config["OUTPUT_PATH"] = wd + config["OUTPUT_PATH"]

    ofc = CMC.OFCMClassifier(config=config)
    ofc.run()

    name = config["INPUT_VIDEO"].split('.')[0]
    config["OUTPUT_VIDEO"] = "_".join([name, str(config["BEGIN_FRAME"]), str(config["END_FRAME"]), "ann.avi"])
    config["OUTPUT_PLOT"] = "_".join([name, str(config["BEGIN_FRAME"]), str(config["END_FRAME"]), "movements.png"])

    ofc.to_avi("/".join([config["OUTPUT_PATH"], config["OUTPUT_VIDEO"]]))
    ofc.to_csv("/".join([config["OUTPUT_PATH"], config["OUTPUT_CSV"]]))
    ofc.to_png("/".join([config["OUTPUT_PATH"], config["OUTPUT_PLOT"]]))

    ofc.fpath = "/".join([config["OUTPUT_PATH"], config["OUTPUT_VIDEO"]])

    input("Start manually ... press any key")

    ofc.run_manual_evaluation()

    config["OUTPUT_VIDEO"] = "_".join([name, str(config["BEGIN_FRAME"]), str(config["END_FRAME"]), "ann_manual.avi"])
    config["OUTPUT_PLOT"] = "_".join([name, str(config["BEGIN_FRAME"]), str(config["END_FRAME"]), "movements_manual.png"])
    ofc.fpath = "/".join(config["OUTPUT_PATH", config["OUTPUT_VIDEO"]])

    ofc.to_avi("/".join([config["OUTPUT_PATH"], config["OUTPUT_VIDEO"]]))
    ofc.to_csv("/".join([config["OUTPUT_PATH"], config["OUTPUT_CSV"]]))
    ofc.to_png("/".join([config["OUTPUT_PATH"], config["OUTPUT_PLOT"]]))

#   config["OUTPUT_CSV"] = "/result.csv"
    #config_man["INPUT_PATH"] = wd + config_man["INPUT_PATH"]
    #config_man["OUTPUT_PATH"] = wd + config_man["OUTPUT_PATH"]

    # update output paths if they are not set
 #   name = config["INPUT_VIDEO"].split('.')[0]
 #   if config["OUTPUT_VIDEO"] == "":
#        config["OUTPUT_VIDEO"] = "_".join([name, str(config["BEGIN_FRAME"]), str(config["END_FRAME"]), "ann.avi"])
#   if config["OUTPUT_PLOT"] == "":
#        config["OUTPUT_PLOT"] = "_".join([name, str(config["BEGIN_FRAME"]), str(config["END_FRAME"]), "movements.png"])
#    name = config_man["INPUT_VIDEO"].split('.')[0]
#    if config_man["OUTPUT_VIDEO"] == "":
#        config_man["OUTPUT_VIDEO"] = "_".join([name, str(config_man["BEGIN_FRAME"]), str(config_man["END_FRAME"]), "manually_ann.avi"])
#    if config_man["OUTPUT_PLOT"] == "":
#        config_man["OUTPUT_PLOT"] = "_".join([name, str(config_man["BEGIN_FRAME"]), str(config_man["END_FRAME"]), "manually_movements.png"])

 #   inputfile = "E:/TU/9.Semester/Computer Vision Systems Programming/CVSP/data/annotations/efilms/train_cmd_efilms_annotations_20191017.csv"

 #   df = load(inputfile)
 #   print(df)

 #   i = 0 # bis i=75
 #   load_new = True
 #   while load_new:
 #       fpath = df["eF_FILM_ID"].values[i]
 #       sf = df["startTime"].values[i]
 #       ef = df["endTime"].values[i]

 #       config["INPUT_VIDEO"] = fpath + ".mp4"
 #       config["BEGIN_FRAME"] = sf
 #       config["END_FRAME"] = ef

  #      run(config)

#        load_new = bool(int(input('Another one?')))
#        i = i + 1


