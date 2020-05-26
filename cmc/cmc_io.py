from sys import exit


# config_path ... the path to the config file
def parse_config(config_path="config.yaml"):
    from yaml import load, CLoader

    try:
        with open(config_path) as stream:
            config = load(stream=stream, Loader=CLoader)
            return config
    except ImportError:
        print("failing to load config.yaml.")
        exit(2)


# argv ... the system arguments
# defaults ... a dictionary with default options
def parse_sys_arg(argv, defaults):
    from getopt import getopt, GetoptError

    short_arguments = "hs:p:b:n:a:m:i:o:g:c:d:t:f:l:"

    long_arguments = ['SENSITIVITY=', 'SPECIFICITY=', 'BORDER=', 'NUMBER_OF_FEATURES=', 'ANGLE_DIFF_LIMIT=', 'MODE=',
                      'INPUT_VIDEO=', 'OUTPUT_VIDEO=', 'OUTPUT_PLOT=', 'OUTPUT_CSV=', 'INPUT_PATH=', 'OUTPUT_PATH=',
                      'BEGIN_FRAME=', 'END_FRAME=']

    helper_string = "\n".join(["<your_script_name.py>", "-s <SENSITIVITY>", "-p <SPECIFICITY>", "-b <BORDER>",
                               "-n <NUMBER_OF_FEATURES>", "-a <ANGLE_DIFF_LIMIT>", "-m <MODE>", "-i <INPUT_VIDEO>",
                               "-o <OUTPUT_VIDEO>", "-g <OUTPUT_PLOT>", "-c <OUTPUT_CSV>", "-d <INPUT_PATH>",
                               "-t <OUTPUT_PATH>", "-f <BEGIN_FRAME>", "-l <END_FRAME>"])

    try:
        opts, args = getopt(argv, short_arguments, long_arguments)

        for opt, arg in opts:

            if opt == "-h":
                print(helper_string)
                exit()

            for i in range(0, len(long_arguments)):

                short_argument = short_arguments[1 + i * 2]
                long_argument = long_arguments[i].split("=")[0]

                if opt in ("-" + short_argument, "--" + long_argument):
                    defaults[long_argument] = arg
                    print("updating default ", long_argument, " to ", arg)

        return defaults

    except GetoptError:
        print(helper_string)
        exit(2)

class KeyHandler:

    # funcs is a dictionary { keycode: function, keycode: function, ... }
    def __init__(self, delay, funcs):
        self.delay = delay
        self.funcs = funcs

    def wait_and_handle_key(self):
        import cv2
        c = cv2.waitKey(self.delay) & 0xFF
        for k, v in self.funcs.items():
            if c == k:
                v()