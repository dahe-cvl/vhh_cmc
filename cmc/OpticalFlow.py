import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt
from cmc.OpticalFlow_ORB import OpticalFlow_ORB
from cmc.OpticalFlow_SIFT import OpticalFlow_SIFT
from cmc.OpticalFlow_SURF import OpticalFlow_SURF
from cmc.OpticalFlow_BRIEF import OpticalFlow_BRIEF
from cmc.OpticalFlow_FAST import OpticalFlow_FAST
from thirdparty.pyflow import pyflow

class OpticalFlow(object):
    def __init__(self, video_frames=None, algorithm="sift", config_instance=None):
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
        self.config_instance = config_instance

        if (algorithm == "sift"):
            self.feature_detector = OpticalFlow_SIFT(video_frames=video_frames)
        elif (algorithm == "orb"):
            self.feature_detector = OpticalFlow_ORB(video_frames=video_frames)
        else:
            print("ERROR: select valid feature extractor [e.g. sift, orb, surf, fast, brief, pesc]")
            exit()

    def run(self):
        frames_np = self.video_frames

        '''

        mag_l_n = []
        angles_l_n = []
        for i in range(0, len(frames_np)):
            # Flow Options:
            alpha = 0.012
            ratio = 0.75
            minWidth = 20
            nOuterFPIterations = 7
            nInnerFPIterations = 1
            nSORIterations = 30
            colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

            im1 = frames_np[0]
            im2 = frames_np[1]
            im1 = im1.astype(float) / 255.
            im2 = im2.astype(float) / 255.

            #s = time.time()
            u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
                nSORIterations, colType)
            #e = time.time()
            #print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
            #    e - s, im1.shape[0], im1.shape[1], im1.shape[2]))
            flow = np.concatenate((u[..., None], v[..., None]), axis=2)
            #print(flow)
            #print(flow.shape)
            # convert from cartesian to polar
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
            mag_l_n.append(mag)
            angles_l_n.append(ang)
            
            #print(mag)
            #print(ang)
            #print(mag.shape)
            #print(ang.shape)
            #exit()

        
        #mag_l_n = np.array(mag_l)
        #angles_l_n = np.array(angles_l)
        '''

        frames_np = self.video_frames

        filtered_mag_l_n = []
        filtered_angles_l_n = []
        number_of_features_l = []

        vector_x_sum_l = []
        vector_y_sum_l = []
        angles_l_n = []
        mag_l_n = []
        for i in range(1, len(frames_np)):
            #print("##########")

            prev_frame = frames_np[i - 1]
            curr_frame = frames_np[i]

            distance_threshold = self.config_instance.distance_threshold
            kp_prev_list, kp_curr_list = self.feature_detector.getMatches(prev_frame, curr_frame, distance_threshold)

            #print("---")
            #print("number of features")
            #print(len(kp_curr_list))
            #print(len(kp_prev_list))

            if (len(kp_prev_list) == 0 or len(kp_curr_list) == 0):
                #mag_l_n.append([0, 0])
                #angles_l_n.append([0, 0])
                mag_l_n.append(0)
                angles_l_n.append(0)
                continue

            curr_points = np.array(kp_curr_list).astype('float').reshape(-1, 1, 2)
            prev_points = np.array(kp_prev_list).astype('float').reshape(-1, 1, 2)
            print(len(curr_points))
            print(len(prev_points))

            number_of_features = len(curr_points)
            number_of_features_l.append(number_of_features)

            

            mag_n, angle_n = self.compute_magnitude_angle(prev_points,
                                                          curr_points)
            
            #print(mag_n)
            #print(angle_n)
            # angle_raw.append(angle_n.tolist())
            # mag_raw.append(mag_n.tolist())
            ''''''
            #mag_n = np.abs(mag_n)  # [:50])
            mag_n, outlier_idx = self.filter1D(mag_n, alpha=2.5)
            angles_cleanup = []
            angles_orig_np = angle_n
            for s in range(0, len(angles_orig_np)):
                if(outlier_idx == s):
                    angle_mean = (angles_orig_np[s-1] + angles_orig_np[s+1]) / 2.0
                    angles_cleanup.append(angle_mean)
                else:
                    angles_cleanup.append(angles_orig_np[s])
            angle_n = np.array(angles_cleanup)  
            
            #print(mag_n)
            #print(angle_n)

            #mag_n = np.delete(mag_n, outliers_idx)

            vector_y = np.multiply(mag_n, np.sin(np.deg2rad(angle_n)))
            vector_x = np.multiply(mag_n, np.cos(np.deg2rad(angle_n)))

            vector_y_sum = vector_y.sum() / len(vector_y)
            vector_x_sum = vector_x.sum() / len(vector_x)
            #print("vector_y_sum: " + str(vector_y_sum))
            #print("vector_x_sum: " + str(vector_x_sum))

            vector_x_sum_l.append([0, vector_x_sum])
            vector_y_sum_l.append([0, vector_y_sum])
            #exit()

            mag_mean_n = np.mean(mag_n)
            mag_l_n.append(mag_mean_n)

            angle_n = np.abs(angle_n)  # [:50])
            angle_mean_n = np.mean(angle_n)
            angles_l_n.append(angle_mean_n)

            filtered_angle_n = angles_l_n
            filtered_angles_l_n.append(filtered_angle_n)

            '''
            data_std = np.std(mag_n)
            data_mean = np.mean(mag_n)
            anomaly_cut_off = data_std * 3
            lower_limit = data_mean - anomaly_cut_off
            upper_limit = data_mean + anomaly_cut_off
            print(lower_limit)
            # Generate outliers
            outliers_idx = []
            for o, outlier in enumerate(mag_n):
                if outlier > upper_limit or outlier < lower_limit:
                    outliers_idx.append(o)
            filtered_mag_n = np.delete(mag_n, outliers_idx)
            filtered_angle_n = np.delete(angle_n, outliers_idx)
            filtered_curr_points = np.delete(curr_points, outliers_idx, axis=0)
            filtered_prev_points = np.delete(prev_points, outliers_idx, axis=0)
            filtered_mag_n = np.abs(filtered_mag_n)  # [:50])
            filtered_mag_mean_n = np.median(filtered_mag_n)
            filtered_mag_l_n.append([0, filtered_mag_mean_n])
            filtered_angle_n = np.abs(filtered_angle_n)  # [:50])
            filtered_angle_mean_n = np.median(filtered_angle_n)
            '''
            

            '''
            # plot angles over time
            #plt.figure(1)
            fig, axs = plt.subplots(2)
            fig.suptitle('mag and angles per feature point in one frame')
            axs[0].plot(np.arange(len(mag_n)), mag_n)
            axs[0].plot(outliers_idx, mag_n[outliers_idx])
            axs[0].plot(np.arange(len(filtered_mag_n)), filtered_mag_n)
            axs[1].plot(np.arange(len(angle_n)), angle_n)
            #plt.ylim(ymax=190, ymin=-190)
            plt.grid(True)
            plt.show()
            #plt.draw()
            #plt.pause(0.02)
            '''
            '''
            # draw the tracks
            mask = np.zeros_like(frames_np[i])
            for j, (new, old) in enumerate(zip(curr_points, prev_points)):
                if(j > 10):
                    break;
                a, b = new.ravel().astype('int')
                c, d = old.ravel().astype('int')
                mask = cv2.line(mask, (a, b), (c, d), (255, 0, 0), 1)
                frame_curr = cv2.circle(frames_np[i], (a, b), 5, (255, 0, 0), -1)
            img = cv2.add(frame_curr, mask)
            cv2.imshow('frame', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            '''
            

            '''
            # draw orig with n feature points
            n_feature_points = len(curr_points)
            for j, (new, old) in enumerate(zip(curr_points, prev_points)):
                if (j > n_feature_points):
                    break
                a, b = new.astype('int').ravel()
                c, d = old.astype('int').ravel()
                frame_curr = cv2.circle(frames_np[i], (a, b), 2, (255, 0, 0), -1)
                frame_curr = cv2.line(frame_curr, (a, b), (a + 5, b + 5), (255, 0, 0), 1)
            # img = cv2.add(frame_curr, mask)
            cv2.imshow('frame', frame_curr)
            k = cv2.waitKey(100) & 0xff
            if k == 27:
                break
            '''

            '''
            #print(curr_points)
            #print(curr_points.shape)
            #print(filtered_curr_points)
            #print(filtered_curr_points.shape)
            # draw orig with n feature points
            n_feature_points = len(filtered_curr_points)
            for j, (new, old) in enumerate(zip(filtered_curr_points, filtered_prev_points)):
                if (j > n_feature_points):
                    break
                a, b = new.astype('int').ravel()
                c, d = old.astype('int').ravel()
                frame_curr = cv2.circle(frames_np[i], (a, b), 3, (255, 0, 0), -1)
                #frame_curr = cv2.line(frame_curr, (a, b), (a + int(vector_x[j]) * 1, b),
                #                      (0, 0, 255), 2)
                #frame_curr = cv2.line(frame_curr, (a, b), (a, b + int(vector_y[j]) * 1),
                #                      (0, 255, 0), 2)
            # img = cv2.add(frame_curr, mask)
            cv2.imshow('frame', frame_curr)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            
            '''

            '''
            # plot angles over time
            #plt.figure(1)
            fig, axs = plt.subplots(3)
            fig.suptitle('mag and angles per feature point in one frame')
            axs[0].plot(np.arange(len(mag_n)), mag_n)
            axs[0].plot(outliers_idx, mag_n[outliers_idx])
            axs[0].plot(np.arange(len(filtered_mag_n)), filtered_mag_n)
            axs[1].plot(np.arange(len(angle_n)), angle_n)
            axs[2].imshow(frame_curr)
            #plt.ylim(ymax=190, ymin=-190)
            plt.grid(True)
            plt.show()
            k = cv2.waitKey(100) & 0xff
            if k == 27:
                break
            #plt.draw()
            #plt.pause(0.02)
            '''

            '''
            # plot angles over time
            #plt.figure(1)
            fig, axs = plt.subplots(2)
            fig.suptitle('mag and angles per feature point in one frame')
            axs[0].plot(np.arange(len(mag_n)), mag_n)
            axs[0].plot(outliers_idx, mag_n[outliers_idx])
            axs[0].plot(np.arange(len(filtered_mag_n)), filtered_mag_n)
            axs[1].plot(np.arange(len(angle_n)), angle_n)
            #plt.ylim(ymax=190, ymin=-190)
            plt.grid(True)
            plt.show()
            #plt.draw()
            #plt.pause(0.02)
            '''

        mag_filtered1, outlier_idx = self.filter1D(np.array(mag_l_n), alpha=1.5)
        angles_cleanup = []
        angles_orig_np = angles_l_n
        for s in range(0, len(angles_orig_np)):
            if(outlier_idx == s):
                angle_mean = (angles_orig_np[s-1] + angles_orig_np[s+1]) / 2.0
                angles_cleanup.append(angle_mean)
            else:
                angles_cleanup.append(angles_orig_np[s])
        angle_filtered1 = np.array(angles_cleanup)  

        mag_filtered2, outlier_idx = self.filter1D(mag_filtered1, alpha=0.3)
        angles_cleanup = []
        angles_orig_np = angle_filtered1
        for s in range(0, len(angles_orig_np)):
            if(outlier_idx == s):
                angle_mean = (angles_orig_np[s-1] + angles_orig_np[s+1]) / 2.0
                angles_cleanup.append(angle_mean)
            else:
                angles_cleanup.append(angles_orig_np[s])
        angle_filtered2 = np.array(angles_cleanup)  

        mag_filtered3, outlier_idx = self.filter1D(mag_filtered2, alpha=0.2)
        angles_cleanup = []
        angles_orig_np = angle_filtered1
        for s in range(0, len(angles_orig_np)):
            if(outlier_idx == s):
                angle_mean = (angles_orig_np[s-1] + angles_orig_np[s+1]) / 2.0
                angles_cleanup.append(angle_mean)
            else:
                angles_cleanup.append(angles_orig_np[s])
        angle_filtered3 = np.array(angles_cleanup)  




        angle_filtered4, outlier_idx = self.filter1D(angle_filtered3, alpha=0.5)
        mag_cleanup = []
        mag_orig_np = mag_filtered3
        for s in range(0, len(mag_orig_np)):
            if(outlier_idx == s):
                mag_mean = (mag_orig_np[s-1] + mag_orig_np[s+1]) / 2.0
                mag_cleanup.append(mag_mean)
            else:
                mag_cleanup.append(mag_orig_np[s])
        mag_filtered4 = np.array(mag_cleanup)  
        
        
        # plot number of features
        #plt.figure(1)
        fig, axs = plt.subplots(3)
        #fig.suptitle('number')
        axs[0].plot(np.arange(len(number_of_features_l)), number_of_features_l)
        #axs[1].plot(np.arange(len(mag_l_n)), mag_l_n)
        #axs[1].plot(np.arange(len(mag_filtered1)), mag_filtered1)
        axs[1].plot(np.arange(len(mag_filtered2)), mag_filtered2)
        axs[1].plot(np.arange(len(mag_filtered4)), mag_filtered4)
        axs[2].plot(np.arange(len(angles_l_n)), angles_l_n)
        axs[2].plot(np.arange(len(angle_filtered4)), angle_filtered4)
        #plt.ylim(ymax=190, ymin=-190)
        plt.grid(True)
        plt.show()
        #plt.draw()
        #plt.pause(0.02)
        ''''''

        # cv2.destroyAllWindows()
        #print(angles_l_n)
        exit()
        return mag_l_n, angles_l_n, vector_x_sum_l, vector_y_sum_l

    def predict_final_result(self, mag_l_n, angles_l_n, class_names):
        # print(type(mag_l_n))
        # print(len(mag_l_n))
        # exit()

        # calcualate final result
        angles_np = np.array(angles_l_n)
        mag_np = np.array(mag_l_n)

        print(np.mean(angles_np))
        print(np.std(angles_np))

        print(np.mean(mag_np))
        print(np.std(mag_np))

        exit()

        # add filter
        print(mag_np[:, 1:])
        filtered_mag_n, outlier_idx = self.filter1D(mag_np[:, 1:], alpha=3)
        filtered_angles_np = np.delete(angles_np[:, 1:], outlier_idx)

        filtered_angle_n, outlier_idx = self.filter1D(filtered_angles_np, alpha=2)
        filtered_mag_n = np.delete(filtered_mag_n, outlier_idx)

        # calculate x - y components - NOT USED YET
        vector_y = np.multiply(filtered_mag_n, np.sin(np.deg2rad(filtered_angle_n)))
        vector_x = np.multiply(filtered_mag_n, np.cos(np.deg2rad(filtered_angle_n)))

        b, bins, patches = plt.hist(filtered_angle_n, bins=8, range=[0, 360],
                                    cumulative=False)  # bins=None, range=None

        '''
        # plot angles over time (frames)
        fig, axs = plt.subplots(3)
        fig.suptitle('mag and angles per feature point in one frame')

        axs[0].plot(np.arange(len(filtered_mag_n)), filtered_mag_n)
        axs[1].plot(np.arange(len(filtered_angle_n)), filtered_angle_n)
        b, bins, patches = axs[2].hist(filtered_angle_n, bins=8, range=[0,360], cumulative=False)  #bins=None, range=None
        #plt.ylim(ymax=190, ymin=-190)
        plt.grid(True)
        plt.show()
        '''
        ''''''
        th = self.config_instance.min_magnitude_threshold  # 2.0  # manual set threshold for magnitude
        percentage = 0.5  # ratio threshold between no-movement and movement
        class_names_n = ['PAN', 'TILT', 'TILT', 'PAN', 'PAN', 'TILT', 'TILT', 'PAN']

        print("predicted median magnitude: " + str(np.median(filtered_mag_n)))
        print("threshold magnitude: " + str(th))

        class_name = class_names_n[np.argmax(b)]
        if ((class_name == "PAN" and np.median(filtered_mag_n) > th)):
            class_name = "PAN"
        elif ((class_name == "TILT" and np.median(filtered_mag_n) > th)):
            class_name = "TILT"
        else:
            class_name = "NA"

        '''
        print(np.mean(angles_np))
        angle = np.mean(angles_np)
        print("predicted angle: " + str(angle))

        print(np.mean(mag_np))
        mag = np.mean(mag_np)
        print("predicted mag: " + str(mag))

        if ((angle >= 140 and angle < 220) or (angle >= -40 and angle < 40) and (mag > 20)):
            class_name = class_names[0]
        elif ((angle >= 50 and angle < 130) or (angle >= 230 and angle < 310) and (mag > 20)):
            class_name = class_names[1]
        else:
            class_name = class_names[2]
        print(class_name)
        '''
        return class_name

    def filter1D(self, data_np, alpha=2):
        #print(type(data_np))
        #print(data_np.shape)
        data_std = np.std(data_np)
        data_mean = np.mean(data_np)
        anomaly_cut_off = data_std * alpha
        lower_limit = data_mean - anomaly_cut_off
        upper_limit = data_mean + anomaly_cut_off

        # Generate outliers
        outliers_idx = []
        filtered_data = []
        for j in range(0, len(data_np)):
            if(j < len(data_np) - 1):
                nxt = data_np[j+1]
                curr = data_np[j]
                prv = data_np[j-1]
            else:
                nxt = data_np[j]
                curr = data_np[j]
                prv = data_np[j-1]

            if curr > upper_limit or curr < lower_limit:
                data_mean = (prv + nxt) / 2.0
                filtered_data.append(data_mean)
                outliers_idx.append(j)
            else:
                filtered_data.append(curr)
        
        filtered_data_np = np.array(filtered_data)
        return filtered_data_np, outliers_idx

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

        # ang = np.round(np.arctan2(d[:, 0, 1], d[:, 0, 0])*180 / np.pi)

        return mag, ang
