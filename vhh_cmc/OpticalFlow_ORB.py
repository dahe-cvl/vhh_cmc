import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt


class OpticalFlow_ORB(object):
    def __init__(self, video_frames=None):
        self.video_frames = video_frames

        print(self.video_frames[0].shape)
        frame_size = (self.video_frames[0].shape[1]*2, self.video_frames[0].shape[0])
        self.video_writer = cv2.VideoWriter("/data/share/output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 12, frame_size)

    def getMatches(self, frame1, frame2, distance_threshold=0.75):
        kp_curr_list = []
        kp_prev_list = []

        orb = cv2.ORB_create(nfeatures=2000,
                             #scaleFactor=0.5,
                             #nlevels=3,
                             edgeThreshold=5,
                             #firstLevel=None,
                             WTA_K=3,
                             #scoreType=None,
                             #patchSize=None,
                             #fastThreshold=None
                             )
        kp_prev, descriptor_prev = orb.detectAndCompute(frame1, None)
        kp_curr, descriptor_curr = orb.detectAndCompute(frame2, None)

        #print(type(descriptor_prev))
        #print(type(descriptor_curr))
        '''
        out_img_prev = frame1.copy()
        #img = cv2.drawKeypoints(old_frame, kp, out_img)
        out_img_prev = cv2.drawKeypoints(frame1,
                                         kp_prev,
                                         out_img_prev,
                                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        out_img_curr = frame2.copy()
        # img = cv2.drawKeypoints(old_frame, kp, out_img)
        out_img_curr = cv2.drawKeypoints(frame2,
                                         kp_curr,
                                         out_img_curr,
                                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #print(out_img_prev.shape)
        #print(out_img_curr.shape)
        '''

        ''''''
        #bf = cv2.BFMatcher()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Perform the matching between the ORB descriptors of the training image and the test image
        try:
            #matches = bf.knnMatch(descriptor_prev, descriptor_curr, k=2)
            matches = bf.match(descriptor_prev, descriptor_curr)
            #print(matches)
        except:
            return kp_prev_list, kp_curr_list

        # Sort them in the order of their distance.
        good_matches = sorted(matches, key=lambda x: x.distance)
        #print(good_matches)

        if (len(good_matches) <= 1):
            return kp_prev_list, kp_curr_list

        dist_l = [x.distance for x in good_matches]
        #dist_l.sort()
        #print(dist_l)

        x = np.array(dist_l)
        nom = (x - np.min(x)) / ((np.max(x) - np.min(x)) + 0.000000000001)
        dist_l_norm = np.nan_to_num(nom)
        #print(np.max(dist_l_norm))
        #print(np.min(dist_l_norm))

        idx = np.where(dist_l_norm < 0.15)[0]

        '''
        plt.plot(np.arange(len(dist_l_norm)), dist_l_norm)
        plt.draw()
        plt.pause(0.01)
        plt.cla()
        '''


        '''
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
        '''

        # good_matches = np.squeeze(np.array(good_matches)).tolist()
        # print((good_matches))

        # print(matches)
        # print(type(good_matches))


        for match in good_matches[:np.max(idx)]:
            #kp_curr_list.append(kp_curr[match[0].trainIdx].pt)
            #kp_prev_list.append(kp_prev[match[0].queryIdx].pt)
            kp_curr_list.append(kp_curr[match.trainIdx].pt)
            kp_prev_list.append(kp_prev[match.queryIdx].pt)

        #print(frame1.shape)
        #print(frame2.shape)

        print(len(good_matches))
        result = cv2.drawMatches(frame1, kp_prev, frame2, kp_curr, good_matches[:np.max(idx)], outImg=None, flags=2)
        #result = cv2.drawMatches(out_img_prev, kp_prev, out_img_curr, kp_prev, good_matches, None, flags=2)
        # Display the best matching points
        #plt.title('Best Matching Points')
        '''
        plt.imshow(result)
        plt.draw()
        plt.pause(0.01)
        plt.cla()
        #self.video_writer.write(result)
        '''

        # Draw first 10 matches.
        #img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], flags=2)

        return kp_prev_list, kp_curr_list
