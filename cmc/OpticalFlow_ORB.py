import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt


class OpticalFlow_ORB(object):
    def __init__(self, video_frames=None):
        #self.video_name = "C:\\Users\\dhelm\\Documents\\slow_traffic_small.mp4"
        #self.video_name = "C:\\Users\\dhelm\\Documents\\training_data_patrick_link\\training_data\\tilt\\b88f0e71-a0f2-4efe-ae0d-5b83a0770b73_32.mp4"  # tilt unten nach oben
        #self.video_name = "C:\\Users\\dhelm\\Documents\\training_data_patrick_link\\training_data\\tilt\\35_25178.mp4"  # tilt oben nach unten
        #self.video_name = "C:\\Users\\dhelm\\Documents\\training_data_patrick_link\\training_data\\tilt\\94_24357.mp4" # tilt unten nach oben
        #1371a561-6b19-4d69-8210-1347ca75eb90_96
        # self.video_name = "C:\\Users\\dhelm\\Documents\\training_data_patrick_link\\training_data\\pan\\1371a561-6b19-4d69-8210-1347ca75eb90_104.mp4"
        # self.video_name = "C:\\Users\\dhelm\\Documents\\training_data_patrick_link\\training_data\\tilt\\tilt_130_74088.mp4"
        #self.video_name = "C:\\Users\\dhelm\\Documents\\training_data_patrick_link\\training_data\\pan\\130_74173.mp4"
        #self.video_name = "C:\\Users\\dhelm\\Documents\\training_data_patrick_link\\training_data\\pan\\1371a561-6b19-4d69-8210-1347ca75eb90_96.mp4"
        #self.video_name = "C:\\Users\\dhelm\\Documents\\training_data_patrick_link\\training_data\\pan\\066f929d-6434-4ea9-844b-e066f57b6c28_99.mp4" # rechts nach links
        #self.video_name = "C:\\Users\\dhelm\\Documents\\999.mp4"
        #self.video_name = "C:\\Users\\dhelm\\Documents\\444.mp4"
        #self.video_name = "C:\\Users\\dhelm\\Documents\\training_data_patrick_link\\training_data\\pan\\72_65942.mp4"  # pan links nach rechts

        self.video_frames = video_frames

    def getMatches(self, frame1, frame2, distance_threshold=0.75):
        kp_curr_list = []
        kp_prev_list = []

        orb = cv2.ORB_create(nfeatures=2000)
        kp_prev, descriptor_prev = orb.detectAndCompute(frame1, None)
        kp_curr, descriptor_curr = orb.detectAndCompute(frame2, None)

        #print(type(descriptor_prev))
        #print(type(descriptor_curr))

        out_frame1 = frame1.copy()
        #img = cv2.drawKeypoints(old_frame, kp, out_img)
        #out_img_prev = cv2.drawKeypoints(frames_np[i-1],
        #                                 kp_prev,
        #                                 out_img_prev,
        #                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        out_frame2 = frame2.copy()
        # img = cv2.drawKeypoints(old_frame, kp, out_img)
        #out_img_curr = cv2.drawKeypoints(frames_np[i - 1],
        #                                 kp_curr,
        #                                 out_img_curr,
        #                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #print(out_img_prev.shape)
        #print(out_img_curr.shape)

        ''''''
        bf = cv2.BFMatcher()
        # Perform the matching between the ORB descriptors of the training image and the test image
        try:
            matches = bf.knnMatch(descriptor_prev, descriptor_curr, k=2)
            #print(matches)
        except:
            return kp_prev_list, kp_curr_list

        # Apply ratio test
        good_matches = []
        if (len(matches) > 0):
            if (len(matches[0]) == 2):
                for m, n in matches:
                    if m.distance < distance_threshold * n.distance:
                        good_matches.append([m])
            elif (len(matches[0]) == 1):
                for m in matches:
                    if m[0].distance < distance_threshold:
                        good_matches.append([m])
        ''''''
        # good_matches = np.squeeze(np.array(good_matches)).tolist()
        # print((good_matches))

        # print(matches)
        # print(type(good_matches))
        if (len(good_matches) == 0):
            return kp_prev_list, kp_curr_list

        for match in good_matches:
            kp_curr_list.append(kp_curr[match[0].trainIdx].pt)
            kp_prev_list.append(kp_prev[match[0].queryIdx].pt)

        '''
        result = cv2.drawMatches(out_img_curr, kp_curr, out_img_prev, kp_prev, matches[:1], out_img_prev, flags=2)
        # Display the best matching points
        plt.title('Best Matching Points')
        plt.imshow(result)
        plt.draw()
        plt.pause(0.05)
        '''
        return kp_prev_list, kp_curr_list
