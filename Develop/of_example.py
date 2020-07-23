import numpy as np
import cv2 as cv
import argparse
from matplotlib import pyplot as plt


class SimpleOF(object):
    def __init__(self):
        print("create instance of simple OF")

        #self.video_name = "C:\\Users\\dhelm\\Documents\\1.m4v"
        self.video_name = "C:\\Users\\dhelm\\Documents\\training_data_patrick_link_reworked\\training_data\\tilt\\tilt_130_74088.mp4"
        #self.video_name = "C:\\Users\\dhelm\\Documents\\training_data_patrick_link_reworked\\training_data\\pan\\11_885.mp4"




    def run(self):
        import numpy as np
        import cv2 as cv

        mag_l = []
        ang_l = []
        cnt = 0

        cap = cv.VideoCapture(self.video_name)
        ret, frame1 = cap.read()
        prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        while (1):
            cnt = cnt + 1
            ret, frame2 = cap.read()
            if cnt == 1000 or ret == False:
                break
            next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

            print(cnt)

            mag_l.append(np.median(mag))
            ang_l.append(np.median(ang))



            '''
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
            cv.imshow('frame2', bgr)
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break
            elif k == ord('s'):
                cv.imwrite('opticalfb.png', frame2)
                cv.imwrite('opticalhsv.png', bgr)
            '''
            prvs = next

        plt.figure()
        plt.plot(np.arange(len(mag_l)), mag_l, color='k')
        plt.figure()
        plt.plot(np.arange(len(ang_l)), ang_l, color='y')
        plt.show()


    def crop(self, img: np.ndarray, dim: tuple):
        """
        This method is used to crop a specified region of interest from a given image.

        :param img: This parameter must hold a valid numpy image.
        :param dim: This parameter must hold a valid tuple including the crop dimensions.
        :return: This method returns the cropped image.
        """
        crop_w, crop_h = dim

        crop_h_1 = 0
        crop_h_2 = 0
        crop_w_1 = 0
        crop_w_2 = 0

        img_h = img.shape[0]
        img_w = img.shape[1]

        crop_w_1 = int(img_w / 2) - int(crop_w / 2)
        if (crop_w_1 < 0):
            crop_w_1 = 0

        crop_w_2 = int(img_w / 2) + int(crop_w / 2)
        if (crop_w_2 >= img_w):
            crop_w_2 = img_w

        crop_h_1 = int(img_h / 2) - int(crop_h / 2)
        if (crop_h_1 < 0):
            crop_h_1 = 0

        crop_h_2 = int(img_h / 2) + int(crop_h / 2)
        if (crop_h_2 >= img_h):
            crop_h_2 = img_h

        img_crop = img[crop_h_1:crop_h_2, crop_w_1:crop_w_2]
        return img_crop

    def compute_magnitude_angle(self, prev_feat, curr_feat):
        if prev_feat.__len__() <= 0:
            print("no previous features... returning")
            assert (prev_feat.__len__() > 0)
        if prev_feat.__len__() != curr_feat.__len__():
            print("length is not correct")
            assert (prev_feat.__len__() == curr_feat.__len__())
        d = curr_feat - prev_feat
        print(d.shape)
        mag = np.hypot(d[:, 0, 0], d[:, 0, 1])
        ang = np.round(np.degrees(np.arctan2(d[:, 0, 1], d[:, 0, 0])))

        return mag, ang


class SiftFeatures(object):
    def __init__(self, video_name):
        print("create instance of sift features")

        # self.video_name = "C:\\Users\\dhelm\\Documents\\slow_traffic_small.mp4"

        #self.video_name = "C:\\Users\\dhelm\\Documents\\training_data_patrick_link\\training_data\\pan\\72_65942.mp4"
        self.video_name = video_name

    def run(self):
        '''
        parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
                                                      The example file can be downloaded from: \
                                                      https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
        parser.add_argument('image', type=str, help='path to image file')
        args = parser.parse_args()
        cap = cv.VideoCapture(args.image)
        '''

        cap = cv.VideoCapture(self.video_name)

        # Take first frame and find corners in it
        ret, old_frame = cap.read()
        print(ret)

        # dim = (512, 512)
        # old_frame = self.crop(old_frame, dim)
        # print(old_frame.shape)

        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
        print(old_gray.shape)

        plt.imshow(old_frame)
        plt.show()

        print("A")

        plt.figure()
        cnt = 0

        frame_list = []
        while (1):
            ret, old_frame = cap.read()
            if (ret == False):
                break
            old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
            frame_list.append(old_gray)
        frames_np = np.array(frame_list)
        print(frames_np.shape)

        for i in range(1, 20):
            kp_prev_list, kp_curr_list = self.getORBMatches(frames_np[i - 1], frames_np[i])
            print(kp_curr_list[0])
            print(len(kp_prev_list))

            '''
            result1 = frames_np[i-1].copy()
            #result = cv2.circle(result, (int(kp_curr_list[0][0]), int(kp_curr_list[0][1])), radius=0, color=(0, 0, 255), thickness=-1)
            result1 = cv2.circle(result1, (int(kp_prev_list[0][0]), int(kp_prev_list[0][1])), radius=5, color=(255, 0, 0),
                                thickness=5)
            print(result1.shape)

            result2 = frames_np[i].copy()
            # result = cv2.circle(result, (int(kp_curr_list[0][0]), int(kp_curr_list[0][1])), radius=0, color=(0, 0, 255), thickness=-1)
            result2 = cv2.circle(result2, (int(kp_curr_list[0][0]), int(kp_curr_list[0][1])), radius=5,
                                 color=(255, 0, 0),
                                 thickness=5)
            print(result2.shape)

            result = np.concatenate((result1, result2), axis=1)

            # Display the best matching points
            plt.title('Best Matching Points')
            plt.imshow(result)
            plt.draw()
            plt.pause(0.5)
            '''

    def getORBMatches(self, frame1, frame2):
        kp_curr_list = []
        kp_prev_list = []

        orb = cv.ORB_create()

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
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

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
        return kp_prev_list, kp_curr_list

    def crop(self, img: np.ndarray, dim: tuple):
        """
        This method is used to crop a specified region of interest from a given image.

        :param img: This parameter must hold a valid numpy image.
        :param dim: This parameter must hold a valid tuple including the crop dimensions.
        :return: This method returns the cropped image.
        """
        crop_w, crop_h = dim

        crop_h_1 = 0
        crop_h_2 = 0
        crop_w_1 = 0
        crop_w_2 = 0

        img_h = img.shape[0]
        img_w = img.shape[1]

        crop_w_1 = int(img_w / 2) - int(crop_w / 2)
        if (crop_w_1 < 0):
            crop_w_1 = 0

        crop_w_2 = int(img_w / 2) + int(crop_w / 2)
        if (crop_w_2 >= img_w):
            crop_w_2 = img_w

        crop_h_1 = int(img_h / 2) - int(crop_h / 2)
        if (crop_h_1 < 0):
            crop_h_1 = 0

        crop_h_2 = int(img_h / 2) + int(crop_h / 2)
        if (crop_h_2 >= img_h):
            crop_h_2 = img_h

        img_crop = img[crop_h_1:crop_h_2, crop_w_1:crop_w_2]
        return img_crop

    def compute_magnitude_angle(self, prev_feat, curr_feat):
        if prev_feat.__len__() <= 0:
            print("no previous features... returning")
            assert (prev_feat.__len__() > 0)
        if prev_feat.__len__() != curr_feat.__len__():
            print("length is not correct")
            assert (prev_feat.__len__() == curr_feat.__len__())
        d = curr_feat - prev_feat
        print(d.shape)
        mag = np.hypot(d[:, 0, 0], d[:, 0, 1])
        ang = np.round(np.degrees(np.arctan2(d[:, 0, 1], d[:, 0, 0])))

        return mag, ang

of = SimpleOF()
of.run()