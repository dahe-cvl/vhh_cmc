import cv2
import numpy as np
from enum import IntEnum
from matplotlib import pyplot as plt
from vhh_cmc import cmc_io
from vhh_cmc.Configuration import Configuration
import os


class ClassificationType(IntEnum):
    """
    Class holds enum values to specify PAN and TILT.
    """
    PAN = 0,
    TILT = 1


class AngleClassifier(object):
    """
    This class is used to classify different camera movements (e.g. Tilt and Pan).
    """

    # r ... range of angles allowed
    def __init__(self, classification_type, r=np.array([-25, 25])):
        """
        Constructor

        :param classification_type:
        :param r: This parameter specifies the range of valid angles (e.g. [-25, 25])
        """

        self.r = r
        self.classification_type = classification_type
        if self.classification_type == ClassificationType.PAN:
            self.main_angles = np.array([180, 0])
        elif self.classification_type == ClassificationType.TILT:
            self.main_angles = np.array([90, -90])

    def classify(self, angle):
        """
        This method is used to classify a given angle to one out of the classes PAN or TILT.

        :param angle: This parameter represents a angle (e.g. value in range [0, 360])
        :return: This method returns a boolean flag which is set to TRUE if the angle corresponds to a specified range of TILT or PAN.
        """

        print("Checking for " + self.classification_type.name + ": " + str(angle) + " degrees.")
        for i, a in enumerate(self.main_angles):
            print("Classifying: distance " + str(self.r) + " to angle " + str(a))
            if int(angle) in range(self.r[0]+a, self.r[1]+a):
                print("Angle contributes to " + self.classification_type.name + " movement.")
                return True
        print("Angle does not contribute to " + self.classification_type.name + "movement.")
        return False

    def __str__(self):
        """
        This method is used to get the camera movement type.

        :return: This method returns the name (string) of the camera movement (e.g. PAN or TILT).
        """
        return self.classification_type.name


class ClassificationCounter(object):
    # classification = [ frame index, is movement]
    # classification[i] = [ i, is movement ]
    # classification[:, 0] = [ ... , -i, ..., 0, ..., i, ... ]
    # classification[:, 1] = [ is movement, is movement, ..., is movement ... ]

    def inc(self, i):
        self.counter[i] = self.counter[i] + 1

    def reset(self, i):
        self.counter[i] = 0

    def fill_totals(self, i, v, o):
        self.totals[i - 1] = np.array([i + o, v])

    def oversteps(self, i, v):
        return True if self.counter[i] > v else False

    def equals(self, i, v):
        return True if self.counter[i] == v else False

    def get(self, i):
        return self.counter[i]

    def add_movement(self, i):
        self.movements.append(np.array([i, -1]))
        print("creating movement at start frame ", i)
        self.movement_created = True

    def end_movement(self, i):
        if self.movement_created:
            self.movements[self.movements.__len__() - 1][1] = i
            print("movement [" + str(self.movements[self.movements.__len__() - 1][0]) + ", " +
                  str(self.movements[self.movements.__len__() - 1][0]) + "]")
            self.movement_created = False

    def resize(self, size, i):
        self.totals = np.resize(self.totals, [size, 2])
        if self.movement_created:
            self.end_movement(i)

    def is_movement(self, frame_idx):
        is_movement = np.array([(sf <= frame_idx <= ef) for (sf, ef) in self.movements])
        return np.any(is_movement)

    def draw_movement(self, xy, o, c='red', c2='darkred', l='movement'):
        points = xy[:, self.totals[:, 1].astype(bool)]
        plt.scatter(points[0], points[1], color=c, label=l)

        xmin = np.array([sf for (sf, ef) in self.movements])
        xmax = np.array([ef for (sf, ef) in self.movements])

        if xmin.__len__() == 0:
            return
        ymin = xy[1, xmin - o]
        ymax = xy[1, xmax - o]

        yavg = ymin# + ymax) / 2.
        plt.hlines(yavg, xmin, xmax, colors=c2, linewidth=5)
        plt.vlines(xmin, yavg - 10, yavg + 10, colors=c2, linewidth=5)
        plt.vlines(xmax, yavg - 10, yavg + 10, colors=c2, linewidth=5)

    def __init__(self, totals):
        self.counter = np.zeros(2).astype(int)
        self.totals = totals
        # assert at least one fake movement.
        self.movements = []
        self.movement_created = False


class Runmodi(IntEnum):
    """
    This class holds enum values to specify runtime mode.
    """
    NORMAL_MODE = 0
    DEBUG_MODE = 1
    SAVE_MODE = 2
    DEBUG_AND_SAVE_MODE = 3


class OpticalFlow_Pesc(object):
    """
    This class is used for optical flow calculation.
    """

    def __init__(self,
                 video_frames=None,
                 fPath="",
                 debug_path="",
                 sf=0,
                 ef=1,
                 mode=0,
                 pan_classifier=AngleClassifier(ClassificationType.PAN),
                 tilt_classifier=AngleClassifier(ClassificationType.TILT),
                 sensitivity=20,
                 specificity=3,
                 border=50,
                 number_of_features=100,
                 angle_diff_limit=20,
                 config=None):
        """
        Constructor.

        :param video_frames: This parameter holds a valid numpy array representing a range of frames (e.g. NxWxHxchannels).
        :param fPath: This parameter holds a valid path consisting of the absolute path and the correct video name.
        :param debug_path: This parameter specifies a valid path to store results in debug mode.
        :param sf: This parameter represents the starting frame index.
        :param ef: This parameter represents the ending frame index.
        :param mode: This parameter represents runtime mode (e.g. DEBUG_MODE=1 or SAVE_MODE=2).
        :param pan_classifier: This parameter holds a valid object of class type AngleClassifier.
        :param tilt_classifier: This parameter holds a valid object of class type AngleClassifier.
        :param sensitivity: This parameter is used to configure the optical flow algorithm.
        :param specificity: This parameter is used to configure the optical flow algorithm.
        :param border: This parameter is used to configure the optical flow algorithm.
        :param number_of_features: This parameter is used to configure the optical flow algorithm.
        :param angle_diff_limit: This parameter is used to configure the optical flow algorithm.
        :param config:
        """
        self.debug_path = debug_path
        self.video_frames = video_frames
        self.fpath = fPath
        self.sf = sf
        self.ef = ef
        self.mode = Runmodi(mode)
        self.sensitivity = sensitivity
        self.specificity = specificity
        self.border = border
        self.number_of_features = number_of_features
        self.angle_diff_limit = angle_diff_limit

        if config is not None:
            self.configure(config)

        self.i = 1
        self.end = self.ef - self.sf
        if(self.end == 0): self.end = 1

        self.most_common_angles = np.zeros([self.end - 1, 1])
        self.weights = np.zeros([self.end - 1, 1])
        self.frame_size = (0, 0)

        self.cap = None

        self.out = {}

        self.pan_classifier = pan_classifier
        self.tilt_classifier = tilt_classifier
        self.pan_counter = ClassificationCounter(np.zeros([self.end, 2]).astype(int))
        self.tilt_counter = ClassificationCounter(np.zeros([self.end, 2]).astype(int))
        # lists containing all movements: [ start frame, end frame ]
        self.pans = []
        self.tilts = []

        # key handling
        self.key_handler = cmc_io.KeyHandler(50, {
            27: self.interrupt,
            ord('s'): self.init_savemode,
            ord('d'): self.init_debugmode,
            ord('b'): self.init_savedebugmode,
            ord('n'): self.init_normalmode,
            ord('p'): self.annotate_pan,
            ord('t'): self.annotate_tilt,
            ord('e'): self.annotate_nomove
        })

        print("Creating new OpticalFlowCameraMovementClassifier " + str(self))

    # run optical flow computation
    def run(self):
        """
        This method is used to run the optical flow calculation process.

        :return: This method returns a separate list for each movement class and holds the predicted frame ranges of both.
        """
        curr_frame = self.video_frames[0]
        h, w, _ = curr_frame.shape
        self.frame_size = (w, h)

        if self.mode > Runmodi.DEBUG_MODE:
            # need to compute output path
            print("Running program in save mode: configuring outputs.")
            self.out = self.init_outputs(self.fpath, self.sf, self.ef, (w, h))

        # TODO: replace with meaningful feature extractor
        #curr_feat = self.create_random_features(self.number_of_features, self.range_of_frame(curr_frame))
        #print(len(curr_feat))

        # params for ShiTomasi corner detection
        feature_params = dict(maxCorners=500,
                              qualityLevel=0.000003,
                              minDistance=5,
                              blockSize=128)

        print(curr_frame.shape)
        curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        curr_feat = cv2.goodFeaturesToTrack(curr_frame_gray, mask=None, **feature_params)

        #prev_feat, curr_feat = self.getORBMatches(prev_frame_gray, curr_frame_gray)

        #print(len(curr_feat))
        #exit()
        raw_angles_l = []
        raw_mag_l = []
        while self.i < self.end:
            prev_frame, prev_feat, curr_frame, curr_feat = self.step(curr_frame, curr_feat)

            print("prev: ", prev_feat.__len__())
            print("curr: ", curr_feat.__len__())

            prev_feat, curr_feat = self.optical_flow(prev_frame, prev_feat, curr_frame)
            mag, angle = self.compute_magnitude_angle(prev_feat, curr_feat)

            mag_mean = np.mean(mag)
            angle_mean = np.mean(angle)
            raw_angles_l.append(angle_mean)
            raw_mag_l.append(mag_mean)


            most_common_angle = None
            weight = 1
            if self.i >= 2:
                most_common_angle = self.most_common_angles[self.i - 2]
                weight = self.weights[self.i - 2]

                # classifiy movement
                #self.check_for_movement(most_common_angle, self.pan_counter, self.pan_classifier)
                #self.check_for_movement(most_common_angle, self.tilt_counter, self.tilt_classifier)
                print("----")
                print(most_common_angle)
                print(angle_mean)
                self.check_for_movement(angle_mean, self.pan_counter, self.pan_classifier)
                self.check_for_movement(angle_mean, self.tilt_counter, self.tilt_classifier)

            print("prev: ", prev_feat.__len__())
            print("curr: ", curr_feat.__len__())

            if prev_feat.__len__() == 0:
                self.most_common_angles[self.i - 1] = self.most_common_angles[self.i - 2]
                self.weights[self.i - 1] = self.weights[self.i - 2]
            else:
                self.most_common_angles[self.i - 1], curr_feat, self.weights[self.i - 1] = \
                    self.estimate_background(prev_feat, curr_feat, most_common_angle, weight, curr_frame)
            self.i = self.i + 1

        self.pans = self.pan_counter.movements
        self.tilts = self.tilt_counter.movements

        for (sf, ef) in self.pan_counter.movements:
            print(sf)
            print(ef)
        for (sf, ef) in self.tilt_counter.movements:
            print(sf)
            print(ef)
        ''''''
        print(self.fpath)
        print("#############")
        print(raw_angles_l)
        print(raw_mag_l)

        '''
        # plot angles over time
        plt.figure()
        plt.plot(np.arange(len(raw_angles_l)), raw_angles_l)
        # plt.plot(np.arange(len(mag_l)), mag_l)
        plt.ylim(ymax=190, ymin=-190)
        plt.grid(True)
        # plt.show()
        plt.savefig("./raw_angles_" + str(self.fpath.split('/')[-1].split('.')[0]) + ".png")
        '''
        self.clear()
        return self.pans, self.tilts

    # run optical flow computation
    def run_eval(self):
        """
        This method is used to run the optical flow calculation to evaluate specified videos.
        """
        self.cap = cv2.VideoCapture(self.fpath)
        if self.ef == 1:
            self.re_init_ef(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.cap.set(1, self.sf)

        print("\n".join(["Video path: " + self.fpath,
                         "first frame to be captured: " + str(self.sf),
                         "last frame to be captured: " + str(self.ef)]))

        ret, curr_frame = self.cap.read()

        if not ret:
            raise Exception("Failing to open video file.")

        h, w, _ = curr_frame.shape
        self.frame_size = (w, h)

        if self.mode > Runmodi.DEBUG_MODE:
            # need to compute output path
            print("Running program in save mode: configuring outputs.")
            self.out = self.init_outputs(self.fpath, self.sf, self.ef, (w, h))

        curr_feat = self.create_random_features(self.number_of_features, self.range_of_frame(curr_frame))

        while self.i < self.end and self.cap.isOpened():
            prev_frame, prev_feat, curr_frame, curr_feat = self.step_eval(curr_frame, curr_feat)

            print("prev: ", prev_feat.__len__())
            print("curr: ", curr_feat.__len__())

            prev_feat, curr_feat = self.optical_flow(prev_frame, prev_feat, curr_frame)

            most_common_angle = None
            weight = 1
            if self.i >= 2:
                most_common_angle = self.most_common_angles[self.i - 2]
                weight = self.weights[self.i - 2]

                # classifiy movement
                self.check_for_movement(most_common_angle, self.pan_counter, self.pan_classifier)
                self.check_for_movement(most_common_angle, self.tilt_counter, self.tilt_classifier)

            print("prev: ", prev_feat.__len__())
            print("curr: ", curr_feat.__len__())

            if prev_feat.__len__() == 0:
                self.most_common_angles[self.i - 1] = self.most_common_angles[self.i - 2]
                self.weights[self.i - 1] = self.weights[self.i - 2]
            else:
                self.most_common_angles[self.i - 1], curr_feat, self.weights[self.i - 1] = \
                    self.estimate_background(prev_feat, curr_feat, most_common_angle, weight, curr_frame)
            self.i = self.i + 1

        self.pans = self.pan_counter.movements
        self.tilts = self.tilt_counter.movements

        self.clear()

    def clear(self):
        """
        This method is used to reset all internal counters and destroy all cv2 windows.
        """
        print("Clear and release everything. Reset frame index.")

        if self.mode > Runmodi.DEBUG_MODE:
            self.clear_outputs()

        if self.pan_counter.movement_created:
            self.pan_counter.movements[self.pan_counter.movements.__len__() - 1][1] = self.i + self.sf
        if self.tilt_counter.movement_created:
            self.tilt_counter.movements[self.tilt_counter.movements.__len__() - 1][1] = self.i + self.sf
        # reset i
        self.i = 1
        #self.cap.release()
        cv2.destroyAllWindows()

    def configure(self, config):
        """
        This method is used for reading configuration parameters (just needed during developing phase).

        :param config: This parameter must hold a valid dictionary including the mandatory parameters.
        """
        self.sensitivity = int(config["SENSITIVITY"])
        self.specificity = int(config["SPECIFICITY"])
        self.border = int(config["BORDER"])
        self.number_of_features = int(config["NUMBER_OF_FEATURES"])
        self.angle_diff_limit = int(config["ANGLE_DIFF_LIMIT"])
        self.mode = Runmodi(int(config["MODE"]))
        self.fpath = "/".join([config["INPUT_PATH"], config["INPUT_VIDEO"]])
        self.sf = int(config["BEGIN_FRAME"])
        self.ef = int(config["END_FRAME"])

    def __str__(self):
        """
        This method is used to create a summary string including key information related to the optical flow process.
        :return: This method returns a string summary of internal calculated key information.
        """
        return "\n".join(
            [
                "OFCMClassifier:",
                "- Video path: " + self.fpath,
                "- first frame to be captured: " + str(self.sf),
                "- last frame to be captured: " + str(self.ef),
                "Parameters: ",
                "- sensitivity: " + str(self.sensitivity) + " consecutive frames"
                "- border: " + str(self.border) + " for random features"
                "- number of features: " + str(self.number_of_features) + " to be tracked",
                "- angle diff limit: " + str(self.angle_diff_limit) + " degrees for background consideration.",
                "running mode " + self.mode.name
            ]
        )

    def re_init_ef(self, ef):
        """
        This method is used to re-initialize the end frame index.

        :param ef: This parameter must hold a valid frame index.
        """
        self.ef = ef
        self.end = self.ef - self.sf
        self.most_common_angles = np.zeros([self.end - 1, 1])
        self.weights = np.zeros([self.end - 1, 1])

        self.pan_counter = ClassificationCounter(np.zeros([self.end, 2]).astype(int))
        self.tilt_counter = ClassificationCounter(np.zeros([self.end, 2]).astype(int))

    def annotate_pan(self):
        """
        This method is used to annotate a frame as PAN.
        """
        if self.tilt_counter.movement_created:
            self.tilt_counter.end_movement(self.i + self.sf)
        if self.pan_counter.movement_created:
            self.pan_counter.end_movement(self.i + self.sf)
        else:
            self.pan_counter.add_movement(self.i + self.sf)

    def annotate_tilt(self):
        """
        This method is used to annotate a frame as TILT.
        """

        if self.pan_counter.movement_created:
            self.pan_counter.end_movement(self.i + self.sf)
        if self.tilt_counter.movement_created:
            self.tilt_counter.end_movement(self.i + self.sf)
        else:
            self.tilt_counter.add_movement(self.i + self.sf)

    def annotate_nomove(self):
        """
        This method is used to annotate a frame as NA.
        """

        if self.pan_counter.movement_created:
            self.pan_counter.end_movement(self.i + self.sf)
        if self.tilt_counter.movement_created:
            self.tilt_counter.end_movement(self.i + self.sf)

    def run_manual_evaluation(self):
        """
        This method is used to run optical flow process in DEBUG mode. A valid X-Server is needed to visualize the frame player.
        """
        self.cap = cv2.VideoCapture(self.fpath)

        if self.ef == 1:
            self.re_init_ef(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.cap.set(1, self.sf)

        if self.mode > Runmodi.DEBUG_MODE:
            print("\n".join(["Video path: " + self.fpath,
                            "first frame to be captured: " + str(self.sf),
                            "last frame to be captured: " + str(self.ef)]))

        ret, curr_frame = self.cap.read()

        if not ret:
            raise Exception("Failing to open video file.")

        h, w, _ = curr_frame.shape
        self.frame_size = (w, h)

        print("configuring outputs.")
        self.out = self.init_outputs(self.fpath, self.sf, self.ef, (w, h))

        while self.i < self.end and self.cap.isOpened():

            ret, curr_frame = self.cap.read()

            if not ret:
                raise Exception("Failing to retrieve frame.")

            img = curr_frame.copy()
            img2 = curr_frame.copy()
            h, w, d = curr_frame.shape
            self.add_text(img, self.str_movement_info_manually(), h)
            self.add_text(img2, self.str_movement_info_manually(), int(h/2))
            title = str("Manual annotation")
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            f = self.frame_size[1] / self.frame_size[0]
            cv2.resizeWindow(title, 500, np.round(500 * f).astype(int))
            cv2.imshow(title, img2)
            self.key_handler.wait_and_handle_key()

            self.out["manually"].write(img)
            self.i = self.i + 1

        if self.pan_counter.movement_created:
            self.pan_counter.end_movement(self.i + self.sf)
        if self.tilt_counter.movement_created:
            self.tilt_counter.end_movement(self.i + self.sf)

        self.pans = self.pan_counter.movements
        self.tilts = self.tilt_counter.movements

        self.clear()

    @staticmethod
    def compute_output_path(name, sf, ef, info):
        return "_".join([
            name,
            str(sf),
            str(ef),
            info,
            ".avi"
        ])

    # size=(width, height) of frame
    def init_outputs(self, fpath, sf, ef, size):
        s = fpath.split('/')
        s = s[s.__len__() - 1].split('.')

        import os
        try:
            opath = self.debug_path + "/" + s[0]
            os.mkdir(opath)
            print("successfully created output path ", opath)
        except OSError:
            print("failed to create directory. saving to working directory.")

        out_original = cv2.VideoWriter(opath + "/" + self.compute_output_path(s[0], sf, ef, "original"), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, size)
        out_of = cv2.VideoWriter(opath + "/" + self.compute_output_path(s[0], sf, ef, "opticalFlow"), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, size)
        out_mca = cv2.VideoWriter(opath + "/" + self.compute_output_path(s[0], sf, ef, "mostCommonAngle"), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, size)
        out_manually = cv2.VideoWriter(opath + "/" + self.compute_output_path(s[0], sf, ef, "mannually"), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, size)

        return {"original": out_original, "of": out_of, "mca": out_mca, "manually" : out_manually}

    def clear_outputs(self):
        for k, v in self.out.items():
            print("Releasing output element: " + k)
            v.release()

    # creates n random features within specific range r
    @staticmethod
    def create_random_features(n, r):
        feat = np.random.randint(low=r[0], high=r[1], size=(n, 1, 2)).astype(np.float32)
        return feat

    # returns minimum of height and width of frame minus self.border
    def range_of_frame(self, frame):
        h, w, _ = frame.shape
        return np.array([self.border, min(h, w)])

    def str_step_info(self):
        return str(self.i + self.sf) + "/" + str(self.ef)

    def getORBMatches(self, frame1, frame2):
        kp_curr_list = []
        kp_prev_list = []

        orb = cv2.ORB_create()

        kp_prev, descriptor_prev = orb.detectAndCompute(frame1, None)
        kp_curr, descriptor_curr = orb.detectAndCompute(frame2, None)

        out_frame1 = frame1.copy()
        # img = cv2.drawKeypoints(old_frame, kp, out_img)
        # out_img_prev = cv2.drawKeypoints(frames_np[i-1],
        #                                 kp_prev,
        #                                 out_img_prev,
        #                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        out_frame2 = frame2.copy()
        # img = cv2.drawKeypoints(old_frame, kp, out_img)
        # out_img_curr = cv2.drawKeypoints(frames_np[i - 1],
        #                                 kp_curr,
        #                                 out_img_curr,
        #                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # print(out_img_prev.shape)
        # print(out_img_curr.shape)

        # Create a Brute Force Matcher object.
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Perform the matching between the ORB descriptors of the training image and the test image
        matches = bf.match(descriptor_prev, descriptor_curr)

        # The matches with shorter distance are the ones we want.
        matches = sorted(matches, key=lambda x: x.distance)

        for match in matches:
            kp_curr_list.append(kp_curr[match.trainIdx].pt)
            kp_prev_list.append(kp_prev[match.queryIdx].pt)

        '''
        result = cv2.drawMatches(out_img_curr, kp_curr, out_img_prev, kp_prev, matches[:1], out_img_prev, flags=2)
        # Display the best matching points
        plt.title('Best Matching Points')
        plt.imshow(result)
        plt.draw()
        plt.pause(0.05)
        '''
        return np.array(kp_prev_list).astype(np.float32).reshape(-1, 1, 2), np.array(kp_curr_list).astype(np.float32).reshape(-1, 1, 2)

    # Store previous image, I_{i-1} and get current image I_i
    def step(self, curr_frame, curr_feat):
        print(self.i)

        print(curr_feat)
        if curr_feat == None:
            curr_feat = []

        prev_frame = curr_frame
        curr_frame = self.video_frames[self.i]

        print("Update step: frame " + self.str_step_info())

        prev_feat = curr_feat

        if curr_feat.__len__() < self.number_of_features:
            print("Not enough features. adding " +
                  str(self.number_of_features - curr_feat.__len__()) +
                  " new features.")

            # TODO: replace with meaningful feature extractor
            prev_feat = self.create_random_features(self.number_of_features, self.range_of_frame(curr_frame))

            '''
            #prev_feat, curr_feat = self.getORBMatches(curr_frame_gray, curr_frame_gray)

            # params for ShiTomasi corner detection
            feature_params = dict(maxCorners=100,
                                  qualityLevel=0.0003,
                                  minDistance=7,
                                  blockSize=7
                                  )

            print(curr_frame.shape)
            curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            prev_feat = cv2.goodFeaturesToTrack(curr_frame_gray, mask=None, **feature_params)
            print(type(prev_feat))
            '''
            ''''''

            #if(prev_feat == None):
            #    prev_feat = self.create_random_features(self.number_of_features, self.range_of_frame(curr_frame))
            print(len(curr_feat))
            print(len(prev_feat))


            prev_feat[0:curr_feat.__len__()] = curr_feat[0:prev_feat.__len__()]

        if self.mode > Runmodi.DEBUG_MODE:
            self.out["original"].write(curr_frame)

        return prev_frame, prev_feat, curr_frame, curr_feat

        # Store previous image, I_{i-1} and get current image I_i

    def step_eval(self, curr_frame, curr_feat):
        print(self.i)

        prev_frame = curr_frame
        ret, curr_frame = self.cap.read()

        print("Update step: frame " + self.str_step_info())

        prev_feat = curr_feat
        if curr_feat.__len__() < self.number_of_features:
            print("Not enough features. adding " +
                  str(self.number_of_features - curr_feat.__len__()) +
                  " new features.")
            prev_feat = self.create_random_features(self.number_of_features, self.range_of_frame(curr_frame))
            prev_feat[0:curr_feat.__len__()] = curr_feat

        if self.mode > Runmodi.DEBUG_MODE:
            self.out["original"].write(curr_frame)

        return prev_frame, prev_feat, curr_frame, curr_feat

    def optical_flow(self, prev_frame, prev_feat, curr_frame):
        """
        This method is used to calculate the optical flow between two consecutive frames.

        :param prev_frame: This parameter must hold a valid numpy frame (previous).
        :param prev_feat:  This parameter must hold valid features of the previous frame.
        :param curr_frame: This parameter must hold a valid numpy frame (current).
        :return: This method returns two arrays including the features of the previous frame as well as of the current frame.
        """
        print("Calculating optical flow: frame " + self.str_step_info())
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        #print(prev_feat)

        curr_feat, st, err = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
                                                      cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY), prev_feat, None,
                                                      **lk_params)

        # Select good points
        curr_feat = curr_feat[st == 1].reshape(-1, 1, 2)
        prev_feat = prev_feat[st == 1].reshape(-1, 1, 2)

        if self.mode > Runmodi.NORMAL_MODE:
            # debug or saving mode.
            img = curr_frame.copy()
            for i in range(curr_feat.__len__()):
                img = cv2.arrowedLine(img,(prev_feat[i, 0, 0], prev_feat[i, 0, 1]),
                                      (curr_feat[i, 0, 0], curr_feat[i, 0, 1]), [0, 0, 0], 2, 0)

            if self.mode > Runmodi.DEBUG_MODE:
                # saving mode
                self.out["of"].write(img)
            if self.mode == Runmodi.DEBUG_MODE or self.mode == Runmodi.DEBUG_AND_SAVE_MODE:
                # is debug mode too

                f = self.frame_size[1] / self.frame_size[0]

                title = str("Total Optical Flow per Frame.")
                cv2.namedWindow(title, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(title, 500, np.round(500 * f).astype(int))
                cv2.imshow(title, img)
                self.key_handler.wait_and_handle_key()

        return prev_feat, curr_feat

    def compute_magnitude_angle(self, prev_feat, curr_feat):
        """
        This method is used to calculate the magnitude and angle between two consecutive feature frames.

        :param prev_feat: This parameter must hold a valid feature vector.
        :param curr_feat: This parameter must hold a valid feature vector.
        :return: This method returns the magnitude and angle between the given feature vectors.
        """
        print("Determine magnitude and angle (degrees) of features: frame " + self.str_step_info())

        if prev_feat.__len__() <= 0:
            print("no previous features... returning")
            assert(prev_feat.__len__() > 0)
        if prev_feat.__len__() != curr_feat.__len__():
            print("length is not correct")
            assert (prev_feat.__len__() == curr_feat.__len__())
        d = curr_feat - prev_feat
        mag = np.hypot(d[:, 0, 0], d[:, 0, 1])
        ang = np.round(np.degrees(np.arctan2(d[:, 0, 1], d[:, 0, 0])))

        return mag, ang

    def str_movement_info(self):
        """
        This method is used to visualize camera movement information.

        :return: This method returns a string including internal movement information.
        """
        return "PAN" if self.pan_counter.oversteps(1, self.sensitivity) else \
            ("TILT" if self.tilt_counter.oversteps(1, self.sensitivity) else "NO MOVEMENT")

    def str_movement_info_manually(self):
        """
        This method is used to visualize camera movement information (for manual debugging mode only).

        :return: This method returns a string including internal movement information.
        """
        return "PAN" if self.pan_counter.movement_created else \
            ("TILT" if self.tilt_counter.movement_created else "NO MOVEMENT")

    def add_text(self, img, text, h):
        """
        This method is used to add text to a given frame.

        :param img: This parameter must hold a valid numpy image.
        :param text: This parameter must hold a valid text string.
        :param h: This parameter must hold an integer which represents the text height.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner = (20, h - 20)
        font_scale = 1
        font_color = (0, 0, 255)
        line_type = 2

        cv2.putText(img, text,
                    bottom_left_corner,
                    font,
                    font_scale,
                    font_color,
                    line_type)

    def estimate_background(self, prev_feat, curr_feat, most_common_angle, weight, curr_frame=None):
        """
        This method is used to estimate and distinguish between foreground and background objects.

        :param prev_feat: This parameter must hold a valid feature vector.
        :param curr_feat: This parameter must hold a valid feature vector.
        :param most_common_angle: This parameter must hold a valid integer representing the most common angle.
        :param weight: This parameter must hold a weight factor [0,1].
        :param curr_frame: This parameter must hold a valid numpy image.
        :return: This method returns the calculated most_common_angle, background_feat and new weight factor.
        """
        print("Convert relative pan / tilt units and estimate background pixel movement.")

        mag, ang = self.compute_magnitude_angle(prev_feat, curr_feat)
        count = np.histogram(ang, bins=np.arange(361))
        background_feat = curr_feat

        if most_common_angle is not None:
            print("\n".join([
                "Previous most common angle of entire optical flow is " + str(most_common_angle) + " degrees.",
                "Removing flow vectors whose angle differs more than " + str(self.angle_diff_limit) +
                " from it. Those are considered to be part of moving elements."
            ]))

            background = np.abs(ang - most_common_angle) < self.angle_diff_limit
            background_feat = curr_feat[background]
            moving_feat = curr_feat[~background]

            prev_background_feat = prev_feat[background]
            prev_moving_feat = prev_feat[~background]

            if self.mode > Runmodi.NORMAL_MODE and curr_frame is not None:
                img = curr_frame.copy()

                for f, f2 in zip(background_feat, prev_background_feat):
                    img = cv2.arrowedLine(img, (f2[0, 0], f2[0, 1]), (f[0, 0], f[0, 1]),
                                          [255, 0, 0], 2, 0)
                for m, m2 in zip(moving_feat, prev_moving_feat):
                    img = cv2.arrowedLine(img, (m2[0, 0], m2[0, 1]), (m[0, 0], m[0, 1]), [0, 255, 0], 2, 0)

                h, w, d = curr_frame.shape
                self.add_text(img, self.str_movement_info(), h)

                p1 = np.array([w / 2, h / 2]).reshape([2, 1])

                avg_mag = np.average(mag[background])
                if np.isnan(avg_mag):
                    avg_mag = 1
                rad = np.radians(most_common_angle)
                fac = weight * np.minimum(w, h)  # * avg_mag / 200
                p2 = p1 + np.array([fac * np.cos(rad), fac * np.sin(rad)])
                img = cv2.arrowedLine(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), [0, 0, 255], 2, 0)

                if self.mode > Runmodi.DEBUG_MODE:
                    # saving to output file
                    self.out["mca"].write(img)
                if self.mode == Runmodi.DEBUG_MODE or self.mode == Runmodi.DEBUG_AND_SAVE_MODE:
                    # is debug mode too
                    title=str("Background / Moving extracted with Most common angle.")
                    f = self.frame_size[1] / self.frame_size[0]
                    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(title, 500, np.round(500 * f).astype(int))
                    cv2.imshow(title, img)
                    self.key_handler.wait_and_handle_key()

        else:
            print("No previous most common angle. No background estimation.")

        max_idx = np.argmax(count[0])
        amount = count[0][max_idx]
        weight = amount / curr_feat.__len__()

        most_common_angle = count[1][max_idx]

        print("\n".join(["New most common angle " + str(most_common_angle) + " degrees.",
                         "- appears " + str(amount) + " times.",
                         "- remaining " + str(background_feat.__len__()) + " background features.",
                         "- weight: " + str(weight)]))

        return most_common_angle, background_feat, weight

    @staticmethod
    def contributes_to(angle, classifier, i, counter, offset):
        if classifier.classify(angle):
            counter.inc(1)
            counter.fill_totals(i, 1, offset)
            return True
        counter.inc(0)
        counter.fill_totals(i, 0, offset)
        return False

    def check_for_movement(self, angle, counter, classifier):
        contributes_to_movement = self.contributes_to(angle, classifier, self.i, counter, self.sf)

        if contributes_to_movement:
            if counter.equals(1, self.sensitivity):
                print("\n".join(["Last " + str(self.sensitivity) + " frames where " + str(classifier) + ".",
                                 "This is a " + str(classifier) + ".",
                                 "Resetting negative counter."]))
                counter.add_movement(self.i + self.sf - self.sensitivity)

            counter.reset(0)
        elif not contributes_to_movement:
            if counter.equals(0, self.specificity):
                print("\n".join(["Last " + str(self.specificity) + " frames where no " + str(classifier) + ".",
                                 "Resetting positive counter."]))
                counter.reset(1)
                counter.end_movement(self.i + self.sf)

    def interrupt(self):
        print("Interrupting.")
        self.end = self.i + 1

        self.most_common_angles = np.resize(self.most_common_angles, [self.end, 1])
        self.weights = np.resize(self.weights, [self.end, 1])
        self.pan_counter.resize(self.end, self.sf + self.i)
        self.tilt_counter.resize(self.end, self.sf + self.i)

    def init_savemode(self):
        if self.mode > Runmodi.DEBUG_MODE:
            print("Already in save mode. No action.")
            return
        print("Enabling save mode.")
        self.init_outputs(self.fpath, self.i + self.sf, self.ef, self.frame_size)
        self.mode = Runmodi.SAVE_MODE

    def init_debugmode(self):
        if self.mode > Runmodi.DEBUG_MODE:
            print("Decreasing mode. Clearing outputs.")
            self.clear_outputs()
        print("Enabling debug mode.")
        self.mode = Runmodi.DEBUG_MODE

    def init_savedebugmode(self):
        if self.mode <= Runmodi.DEBUG_MODE:
            print("Initializing outputs.")
            self.init_outputs(self.fpath, self.i + self.sf, self.ef, self.frame_size)
        print("Enabling save debug mode.")
        self.mode = Runmodi.DEBUG_AND_SAVE_MODE

    def init_normalmode(self):
        if self.mode > Runmodi.DEBUG_MODE:
            print("Decreasing mode. Clearing outputs.")
            self.clear_outputs()
        print("Enabling normal mode.")
        self.mode = Runmodi.NORMAL_MODE

    def plot_movements(self, show=True):
        fig = plt.figure()

        x = np.arange(self.sf, self.sf + self.end, 1)
        y = np.resize(self.most_common_angles, x.shape)

        xy = np.array([x, y])
        plt.scatter(x, y, color='lightgrey')
        # pans = xy[:, self.pan_counter.totals[:, 1].astype(bool)]
        # plt.scatter(pans[0], pans[1], color='red')
        # tilts = xy[:, self.tilt_counter.totals[:, 1].astype(bool)]
        # plt.scatter(tilts[0], tilts[1], color='blue')

        self.pan_counter.draw_movement(xy, self.sf + 1, 'red', 'darkred', 'pan')
        self.tilt_counter.draw_movement(xy, self.sf + 1, 'blue', 'darkblue', 'tilt')

        plt.xlabel('Frame Index')
        plt.ylabel('Angle Degrees')
        plt.ylim([0, 180])
        plt.title('Classified Frames')

        plt.legend()
        if show:
            plt.show()

    def to_avi(self, opath):
        print("saving annotated video to ", opath, " ...")
        self.cap = cv2.VideoCapture(self.fpath)
        self.cap.set(1, self.sf)
        ret, curr_frame = self.cap.read()

        if not ret:
            raise Exception("Failing to open video file.")

        h, w, _ = curr_frame.shape
        self.frame_size = (w, h)
        out = cv2.VideoWriter(opath, cv2.VideoWriter_fourcc(*"MJPG"), 20.0, self.frame_size)

        while self.i < self.end and self.cap.isOpened():

            img = curr_frame.copy()
            if self.pan_counter.is_movement(self.i + self.sf):
                self.add_text(img, "PAN", self.frame_size[1])
            if self.tilt_counter.is_movement(self.i + self.sf):
                self.add_text(img, "TILT", self.frame_size[1])

            p1 = np.array([w / 2, h / 2]).reshape([2, 1])
            rad = np.radians(self.most_common_angles[self.i - 1])
            fac = 0.5 * np.minimum(w, h)# * avg_mag / 200
            p2 = p1 + np.array([fac * np.cos(rad), fac * np.sin(rad)])
            img = cv2.arrowedLine(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), [0, 0, 255], 2, 0)
            out.write(img)
            if self.mode == Runmodi.DEBUG_MODE or self.mode == Runmodi.DEBUG_AND_SAVE_MODE:
                cv2.imshow("saving ...", img)
                self.key_handler.wait_and_handle_key()

            self.i = self.i + 1
            ret, curr_frame = self.cap.read()

        out.release()
        self.i = 1
        self.cap.release()
        if self.mode == Runmodi.DEBUG_MODE or self.mode == Runmodi.DEBUG_AND_SAVE_MODE:
            cv2.destroyAllWindows()
        print("saving annotated video to ", opath, " DONE.")

    def to_png(self, opath):
        print("saving movements plot to ", opath, " ...")
        self.plot_movements(False)
        plt.savefig(opath)
        print("saving movements plot to ", opath, " DONE.")

    def to_csv(self, opath):
        print("saving extracted movements to ", opath, " ...")
        import pandas
        # add all pans
        movements = []
        for (sf, ef) in self.pan_counter.movements:
            movements.append([self.fpath, 'PAN', sf, ef])
        for (sf, ef) in self.tilt_counter.movements:
            movements.append([self.fpath, 'TILT', sf, ef])
        movements = np.array(movements)
        if movements.__len__() == 0:
            print("no movements detected to save to csv.")
            return
        df = pandas.DataFrame({'name': movements[:, 0], 'movement': movements[:, 1], 'start': movements[:, 2], 'end': movements[:, 3]})
        df.to_csv(opath, mode='a', header=False, index=False, sep=";") # header must be there at beginning!!
        print("saving extracted movements to ", opath, " DONE.")